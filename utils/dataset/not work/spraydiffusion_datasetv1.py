"""
SprayDiffusion Dataset Implementation.
This dataset is designed to handle loading data for SprayDiffusion models,
adapting the MaskPlanner dataset format for use with 3D Diffusion Policy.
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils import data
import zarr
import numcodecs

from utils.disk import get_dataset_path, get_dataset_meshes_path, read_mesh_as_pointcloud, read_traj_file
from utils.common import center_pair, get_max_distance
from utils.dataset.paintnet_ODv1 import orient_in, get_dim_traj_points

class SprayDiffusionDataset(data.Dataset):
    """
    Dataset for SprayDiffusion model training.
    Can load data from either a replay buffer zarr file or directly from the raw dataset.
    """
    def __init__(
        self,
        config,
        dataset_paths,
        split='train',
        replay_buffer_path=None,
        **kwargs
    ):
        """
        Initialize the SprayDiffusion dataset.
        
        Args:
            config: Configuration object with dataset parameters
            dataset_paths: Paths to the dataset directories
            split: Dataset split ('train' or 'test')
            replay_buffer_path: Path to a zarr replay buffer (optional)
        """
        self.config = config
        self.dataset_paths = dataset_paths
        self.split = split
        self.replay_buffer_path = replay_buffer_path
        
        # Load configuration parameters
        self.pc_points = config.pc_points
        self.traj_points = config.traj_points
        self.lambda_points = config.lambda_points if hasattr(config, 'lambda_points') else 1
        self.overlapping = config.overlapping if hasattr(config, 'overlapping') else 0
        self.normalization = config.normalization if hasattr(config, 'normalization') else 'per-dataset'
        self.weight_orient = config.weight_orient if hasattr(config, 'weight_orient') else 0.25
        
        # Set extra data parameters
        self.extra_data = tuple(config.extra_data) if hasattr(config, 'extra_data') else ('orientnorm',)
        self.outdim = get_dim_traj_points(self.extra_data)
        
        self.zarr_store = None
        self._cache = {}  # Memory cache for frequently accessed samples
        
        # Initialize dataset samples
        if self.replay_buffer_path and os.path.exists(self.replay_buffer_path):
            # Load from zarr replay buffer
            print(f"Loading data from replay buffer: {self.replay_buffer_path}")
            self.zarr_store = zarr.open(self.replay_buffer_path, mode='r')
            self.n_samples = len(self.zarr_store['index'])
            self.use_buffer = True
        else:
            # Load directly from dataset files
            print(f"Loading data directly from dataset paths: {dataset_paths}")
            self.samples = self._load_dataset_samples()
            self.n_samples = len(self.samples)
            self.use_buffer = False
            
            # Calculate dataset normalization factor if needed
            if self.normalization == 'per-dataset':
                from utils.disk import get_dataset_downscale_factor
                dataset_name = config.dataset
                if isinstance(dataset_name, list):
                    dataset_name = '-'.join(dataset_name)
                self.dataset_mean_max_distance = get_dataset_downscale_factor(dataset_name)
                print(f"Using dataset scale factor: {self.dataset_mean_max_distance}")
    
    def _load_dataset_samples(self):
        """Load samples from the dataset directories"""
        samples = []
        for root in self.dataset_paths:
            # Load split file
            split_file = os.path.join(root, f'{self.split}_split.json')
            if not os.path.exists(split_file):
                print(f"Warning: Split file {split_file} not found. Skipping this path.")
                continue
                
            with open(split_file, 'r') as f:
                dir_samples = [str(d) for d in json.load(f)]
            
            for curr_dir in dir_samples:
                mesh_file = os.path.join(root, curr_dir, f'{curr_dir}.obj')
                traj_file = os.path.join(root, curr_dir, 'trajectory.txt')
                
                if os.path.exists(mesh_file) and os.path.exists(traj_file):
                    samples.append((mesh_file, traj_file, curr_dir))
                else:
                    print(f"Warning: Missing files for {curr_dir}")
        
        return samples
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.n_samples
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Check if the sample is already in the cache
        if idx in self._cache:
            return self._cache[idx]
        
        if self.use_buffer:
            # Load from zarr buffer
            return self._get_from_buffer(idx)
        else:
            # Load from raw dataset
            return self._get_from_files(idx)
    
    def _get_from_buffer(self, idx):
        """Load a sample from the zarr replay buffer"""
        # Load and process data from zarr buffer
        sample = {}
        
        # Create a dict with all zarr arrays from the store
        for key in self.zarr_store.keys():
            if key == 'index' or key == 'metadata':
                continue
                
            value = self.zarr_store[key][idx]
            sample[key] = value
        
        # Cache the sample
        self._cache[idx] = sample
        return sample
    
    def _get_from_files(self, idx):
        """Load and process a sample directly from dataset files"""
        mesh_file, traj_file, dirname = self.samples[idx]
        
        # Load the mesh as a point cloud
        point_cloud = read_mesh_as_pointcloud(mesh_file)
        
        # Load the trajectory file
        traj, stroke_ids = read_traj_file(traj_file, extra_data=self.extra_data, weight_orient=self.weight_orient)
        
        # Center the point cloud and trajectory
        point_cloud, traj = center_pair(point_cloud, traj, mesh_file)
        
        # Apply normalization
        if self.normalization == 'per-dataset':
            point_cloud /= self.dataset_mean_max_distance
            traj[:, :3] /= self.dataset_mean_max_distance
        elif self.normalization == 'per-mesh':
            max_distance = get_max_distance(mesh_file)
            point_cloud /= max_distance
            traj[:, :3] /= max_distance
        
        # Subsample the point cloud
        if point_cloud.shape[0] > self.pc_points:
            choice = np.random.choice(point_cloud.shape[0], self.pc_points, replace=False)
            point_cloud = point_cloud[choice, :]
        
        # Prepare the sample
        sample = {
            'point_cloud': point_cloud.astype(np.float32),
            'traj': traj.astype(np.float32),
            'traj_as_pc': traj.astype(np.float32),  # Simplified for diffusion model
            'stroke_ids': stroke_ids.astype(np.float32),
            'stroke_ids_as_pc': stroke_ids.astype(np.float32),
            'dirname': dirname,
            'n_strokes': len(np.unique(stroke_ids))
        }
        
        # Create dummy stroke masks for compatibility with MaskPlanner
        unique_stroke_ids = np.unique(stroke_ids)
        n_strokes = len(unique_stroke_ids)
        stroke_masks = np.zeros((n_strokes, len(stroke_ids)), dtype=np.float32)
        for i, stroke_id in enumerate(unique_stroke_ids):
            stroke_masks[i, stroke_ids == stroke_id] = 1.0
        sample['stroke_masks'] = stroke_masks
        
        # Cache the sample
        self._cache[idx] = sample
        return sample
    
    def get_normalizer(self):
        """
        Get normalizer for compatibility with DP3.
        
        Returns:
            dict: Dictionary containing normalization parameters
        """
        return {
            'obs': {
                'point_cloud': {
                    'mean': np.zeros(3),
                    'std': np.ones(3) 
                },
            },
            'action': {
                'mean': np.zeros(self.outdim),
                'std': np.ones(self.outdim)
            }
        }
    
    def get_validation_dataset(self):
        """
        Get validation dataset for DP3 compatibility.
        
        Returns:
            SprayDiffusionDataset: Validation dataset
        """
        if self.split == 'train':
            return SprayDiffusionDataset(
                config=self.config,
                dataset_paths=self.dataset_paths,
                split='test'
            )
        return self

# 示例用法
"""
# 示例：使用带缓存的SprayDiffusionDataset
from utils.dataset.spraydiffusion_dataset import SprayDiffusionDataset
import os

# 缓存文件路径
cache_dir = 'cache/spraydiffusion'
os.makedirs(cache_dir, exist_ok=True)

replay_buffer_path = os.path.join(cache_dir, 'replay_buffer_train.zarr')

# 创建带缓存的数据集
dataset = SprayDiffusionDataset(
    dataset_paths=['/path/to/data'],
    config=config,
    split='train',
    replay_buffer_path=replay_buffer_path,  # 指定缓存文件路径
    force_rebuild_buffer=False  # 设置为True可以强制重建缓存
)

# 获取验证集（会自动使用replay_buffer_path + "_val"作为缓存路径）
val_dataset = dataset.get_validation_dataset()

# 创建数据加载器
from torch.utils.data import DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 使用数据集
for batch in train_loader:
    # batch['action']是时间序列轨迹数据，形状为[B, T, D]
    # batch['obs']['point_cloud']是点云数据，形状为[B, N, 3]
    # ...
    pass
"""

