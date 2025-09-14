from typing import Dict
import torch
import numpy as np
import copy
import os
import time
from spray_diffusion.common.pytorch_util import dict_apply
from spray_diffusion.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from spray_diffusion.dataset.base_dataset import BaseDataset
from utils.dataset.paintnet_ODv1 import PaintNetODv1Dataloader
from spray_diffusion.common.replay_buffer import ReplayBuffer
from spray_diffusion.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
import zarr
import numcodecs

class SprayDiffusionDataset(BaseDataset):
    def __init__(self,
            zarr_path=None, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            config={},
            dataset_paths=[],
            split='train',
            replay_buffer_path=None,  # 新增参数：ReplayBuffer缓存文件路径
            force_rebuild_buffer=False,  # 新增参数：是否强制重建ReplayBuffer
            ):
        super().__init__()
        self.task_name = task_name
        self.split = split
        self.horizon = horizon
        self.seed = seed
        self.config = config
        self.dataset_paths = dataset_paths
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.replay_buffer_path = replay_buffer_path

        # 创建相应的PaintNetODv1Dataloader实例
        self.dataset_paintnet = PaintNetODv1Dataloader(
            roots=dataset_paths,
            dataset=config.dataset,
            pc_points=config.pc_points,
            traj_points=config.traj_points,
            lambda_points=config.lambda_points,
            overlapping=config.overlapping if not config.asymm_overlapping else config.lambda_points-1,
            normalization=config.normalization,
            data_scale_factor=config.data_scale_factor,
            extra_data=tuple(config.extra_data) if hasattr(config, 'extra_data') else None,
            weight_orient=config.weight_orient,
            split=split,
            config=config,
            overfitting=(None if config.overfitting is False else config.seed),
            augmentations=config.augmentations if split == 'train' else [],
            train_portion=config.train_portion if split == 'train' else None
        )
        
        # 创建或加载ReplayBuffer
        if replay_buffer_path is not None and os.path.exists(replay_buffer_path) and not force_rebuild_buffer:
            # 如果提供了ReplayBuffer路径，并且文件存在，从文件加载
            print(f"从文件加载ReplayBuffer: {replay_buffer_path}")
            start_time = time.time()
            self.replay_buffer = self.load_replay_buffer(replay_buffer_path)
            print(f"加载ReplayBuffer完成，耗时: {time.time() - start_time:.2f}秒")
        else:
            # 否则创建新的ReplayBuffer
            print("创建新的ReplayBuffer...")
            start_time = time.time()
            self.replay_buffer = self._create_replay_buffer()
            print(f"创建ReplayBuffer完成，耗时: {time.time() - start_time:.2f}秒")
            
            # 如果提供了ReplayBuffer路径，保存到文件
            if replay_buffer_path is not None:
                print(f"保存ReplayBuffer到文件: {replay_buffer_path}")
                self.save_replay_buffer(replay_buffer_path)
        
        # 创建采样器
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        
        print(f"创建 SprayDiffusionDataset for split: {split} 包含 {len(self.dataset_paintnet)} 个样本")
        print(f"采样器大小: {len(self.sampler)}")
        if hasattr(config, 'extra_data'):
            print(f"使用 extra_data: {config.extra_data}")
        if hasattr(config, 'data_scale_factor'):
            print(f"使用 data_scale_factor: {config.data_scale_factor}")

    def _create_replay_buffer(self):
        """Create ReplayBuffer structure needed for SequenceSampler"""
        # Create an empty ReplayBuffer
        replay_buffer = ReplayBuffer.create_empty_numpy()
        
        # Collect all sample data
        episode_ends = []
        total_steps = 0
        
        # Store all trajectories and point cloud data
        all_trajs = []
        all_point_clouds = []
        all_stroke_ids = []
        all_sample_info = []  # Store other info like dirname, etc.
        
        # Step 1: Preprocessing to determine feature dimensions
        feature_dim = None
        pc_points = None
        
        # Check the first valid sample to determine dimensions
        for idx in range(len(self.dataset_paintnet)):
            try:
                # Get original mesh and trajectory files to locate preprocessed samples
                mesh_file, traj_file, dirname = self.dataset_paintnet.datapath[idx]
                
                # Check if preprocessed sample exists
                preprocessed_dir = os.path.join(os.path.abspath(os.path.join(mesh_file, os.pardir)), 'paintnet_preprocessed_sample')
                filename = self.dataset_paintnet._get_preprocessed_sample_name()
                
                # Load directly from preprocessed sample if available
                if os.path.isfile(os.path.join(preprocessed_dir, filename)):
                    sample_data = np.load(os.path.join(preprocessed_dir, filename))
                    traj, point_cloud = sample_data['traj'], sample_data['point_cloud']
                else:
                    # Fallback to original dataloader if preprocessed sample not available
                    sample = self.dataset_paintnet[idx]
                    traj = sample['traj']
                    point_cloud = sample['point_cloud']
                
                # Convert to numpy arrays if necessary
                if isinstance(traj, torch.Tensor):
                    traj = traj.detach().cpu().numpy()
                elif not isinstance(traj, np.ndarray):
                    traj = np.array(traj)
                    
                if isinstance(point_cloud, torch.Tensor):
                    point_cloud = point_cloud.detach().cpu().numpy()
                elif not isinstance(point_cloud, np.ndarray):
                    point_cloud = np.array(point_cloud)
                
                # Determine feature dimension
                if traj.size > 0:
                    if len(traj.shape) >= 2:
                        feature_dim = traj.shape[-1]
                    else:
                        # Single dimension feature, manually set to 1
                        feature_dim = 1
                
                # Determine point cloud dimension
                if point_cloud.size > 0:
                    if len(point_cloud.shape) >= 2:
                        pc_points = point_cloud.shape[0]
                    
                # If valid dimension info is found, break the loop
                if feature_dim is not None and pc_points is not None:
                    break
            except Exception as e:
                print(f"Error preprocessing sample {idx}: {e}")
                continue
        
        # If dimensions cannot be determined, use default values from config
        if feature_dim is None:
            feature_dim = 3  # Default value
            print(f"Warning: Could not determine feature dimension from data, using default value {feature_dim}")
            
        if pc_points is None:
            pc_points = self.config.pc_points
            print(f"Warning: Could not determine point cloud dimension from data, using config value {pc_points}")
        
        # Collect all data, now prioritizing preprocessed samples
        for idx in range(len(self.dataset_paintnet)):
            try:
                # Get original mesh and trajectory files
                mesh_file, traj_file, dirname = self.dataset_paintnet.datapath[idx]
                
                # Check if preprocessed sample exists
                preprocessed_dir = os.path.join(os.path.abspath(os.path.join(mesh_file, os.pardir)), 'paintnet_preprocessed_sample')
                filename = self.dataset_paintnet._get_preprocessed_sample_name()
                
                if os.path.isfile(os.path.join(preprocessed_dir, filename)):
                    # Load data from preprocessed sample
                    sample_data = np.load(os.path.join(preprocessed_dir, filename))
                    traj = sample_data['traj']
                    point_cloud = sample_data['point_cloud']
                    stroke_ids = sample_data['stroke_ids']
                    
                    # Get additional information from the original dataset
                    # This ensures we have necessary metadata not in the preprocessed file
                    original_sample = self.dataset_paintnet[idx]
                    n_strokes = original_sample.get('n_strokes', len(np.unique(stroke_ids)))
                else:
                    # Fallback to using original dataloader
                    sample = self.dataset_paintnet[idx]
                    traj = sample['traj']
                    point_cloud = sample['point_cloud']
                    stroke_ids = sample.get('stroke_ids', None)
                    n_strokes = sample.get('n_strokes', 1)
                
                # Ensure data is numpy arrays
                if isinstance(traj, torch.Tensor):
                    traj = traj.detach().cpu().numpy()
                elif not isinstance(traj, np.ndarray):
                    traj = np.array(traj)
                    
                if isinstance(point_cloud, torch.Tensor):
                    point_cloud = point_cloud.detach().cpu().numpy()
                elif not isinstance(point_cloud, np.ndarray):
                    point_cloud = np.array(point_cloud)
                    
                if stroke_ids is not None:
                    if isinstance(stroke_ids, torch.Tensor):
                        stroke_ids = stroke_ids.detach().cpu().numpy()
                    elif not isinstance(stroke_ids, np.ndarray):
                        stroke_ids = np.array(stroke_ids)
                
                # Ensure traj is at least 2D
                if len(traj.shape) == 1:
                    traj = traj.reshape(-1, 1)
                
                # Ensure point cloud dimensions are correct
                if point_cloud.shape[0] != pc_points and len(point_cloud.shape) >= 2:
                    # If point count is too many, trim; if too few, pad
                    if point_cloud.shape[0] > pc_points:
                        point_cloud = point_cloud[:pc_points]
                    else:
                        padding = np.zeros((pc_points - point_cloud.shape[0], point_cloud.shape[1]))
                        point_cloud = np.concatenate([point_cloud, padding], axis=0)
                
                # Record step count (trajectory length)
                num_steps = traj.shape[0]
                if num_steps == 0:
                    print(f"Skipping sample {idx}, trajectory length is 0")
                    continue
                
                total_steps += num_steps
                episode_ends.append(total_steps)
                
                # Process stroke_ids, create placeholder if not existing
                if stroke_ids is None or stroke_ids.size == 0:
                    stroke_ids = np.zeros(num_steps)
                elif stroke_ids.shape[0] != num_steps:
                    # Ensure stroke_ids length matches trajectory
                    if stroke_ids.shape[0] > num_steps:
                        stroke_ids = stroke_ids[:num_steps]
                    else:
                        padding = np.zeros(num_steps - stroke_ids.shape[0])
                        stroke_ids = np.concatenate([stroke_ids, padding])
                
                # Store data
                all_trajs.append(traj)
                
                # Duplicate point cloud to match trajectory length (all time steps share same point cloud)
                try:
                    point_cloud_expanded = np.tile(point_cloud[np.newaxis, :, :], (num_steps, 1, 1))
                    all_point_clouds.append(point_cloud_expanded)
                except Exception as e:
                    print(f"Error processing point cloud data: {e}, point cloud shape: {point_cloud.shape}, trajectory length: {num_steps}")
                    # Create a zero-filled point cloud
                    point_cloud_expanded = np.zeros((num_steps, pc_points, 3))
                    all_point_clouds.append(point_cloud_expanded)
                
                all_stroke_ids.append(stroke_ids)
                all_sample_info.append({
                    'dirname': dirname,
                    'n_strokes': n_strokes,
                    'sample_idx': idx
                })
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        # Ensure data is available
        if not all_trajs:
            print("Warning: No valid trajectory data!")
            # Create empty data structures
            trajs = np.zeros((0, feature_dim))
            point_clouds = np.zeros((0, pc_points, 3))
            stroke_ids = np.zeros(0)
            episode_ends = [0]  # Need at least one element
        else:
            try:
                # Merge all data
                trajs = np.concatenate(all_trajs, axis=0)
                point_clouds = np.concatenate(all_point_clouds, axis=0)
                stroke_ids = np.concatenate(all_stroke_ids, axis=0)
            except Exception as e:
                print(f"Error merging data: {e}")
                # Fallback to safe data structures
                trajs = np.zeros((1, feature_dim))
                point_clouds = np.zeros((1, pc_points, 3))
                stroke_ids = np.zeros(1)
                episode_ends = [1]
        
        # Build data dictionary
        data = {
            'action': trajs,
            'point_cloud': point_clouds,
            'stroke_ids': stroke_ids,
            'sample_info': np.zeros(total_steps, dtype=np.int32)  # Placeholder
        }
        
        # Update ReplayBuffer
        replay_buffer.root['meta']['episode_ends'] = np.array(episode_ends, dtype=np.int64)
        replay_buffer.root['data'] = data
        
        # Store original sample info for later retrieval
        self.sample_info = all_sample_info
        
        print(f"ReplayBuffer creation complete: {len(episode_ends)} trajectories, {total_steps} total time steps")
        print(f"Action shape: {trajs.shape}, Point Cloud shape: {point_clouds.shape}")
        
        return replay_buffer

    def get_validation_dataset(self):
        # 创建测试集版本
        # 为验证集构建缓存路径（如果原始路径存在）
        val_replay_buffer_path = None
        if self.replay_buffer_path is not None:
            # 在原始路径基础上添加_val后缀
            val_replay_buffer_path = self.replay_buffer_path + "_val"
        
        val_set = SprayDiffusionDataset(
            config=self.config,
            dataset_paths=self.dataset_paths,
            split='test',
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=self.seed,
            replay_buffer_path=val_replay_buffer_path  # 使用验证集专用的缓存路径
        )
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # 创建一个合适的归一化器
        data = {
            'action': self.replay_buffer['action'],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将 SequenceSampler 采样的数据转换为模型需要的格式
        
        参数:
            sample: SequenceSampler 返回的字典数据
            
        返回:
            data: 处理后的数据字典，符合 diffusion policy 的格式要求
        """
        # 安全地获取数据，避免键不存在的情况
        traj = sample.get('action', np.zeros((self.horizon, 3))).astype(np.float32)  # (T, D_action)
        
        # 点云数据处理
        if 'point_cloud' in sample and sample['point_cloud'].size > 0:
            # 默认取第一个时间步的点云，因为整个序列使用相同的点云
            point_cloud = sample['point_cloud'][0].astype(np.float32)  # (N, 3)
        else:
            # 如果点云数据不存在或为空，创建一个零填充的点云
            point_cloud = np.zeros((self.config.pc_points, 3), dtype=np.float32)
        
        # 笔画ID数据处理
        if 'stroke_ids' in sample and sample['stroke_ids'].size > 0:
            stroke_ids = sample['stroke_ids'].astype(np.float32)  # (T,)
        else:
            # 如果笔画ID不存在，创建全零数组
            stroke_ids = np.zeros(traj.shape[0], dtype=np.float32)
        
        # 确保维度一致性
        if len(traj.shape) == 1:
            # 如果轨迹是一维的，转换为二维
            traj = traj.reshape(-1, 1)
        
        if len(point_cloud.shape) == 1:
            # 如果点云是一维的，转换为二维
            point_cloud = point_cloud.reshape(1, -1)
        
        # 添加时间/批次维度
        # 点云形状：[1, N, 3]，表示 1 个时间步，N 个点，每个点 3 个坐标
        # 轨迹形状：[1, T, D]，表示 1 个时间步，T 个轨迹点，每个点 D 个特征
        pc_with_time = point_cloud[np.newaxis, :, :]
        traj_with_time = traj[np.newaxis, :, :]
        
        # 构建数据字典
        data = {
            'obs': {
                'point_cloud': pc_with_time,  # [1, N, 3]
            },
            'action': traj_with_time,  # [1, T, D_action]
            'stroke_ids': stroke_ids  # [T]
        }
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 使用SequenceSampler获取数据
        sequence = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sequence)
        
        # 转换为PyTorch张量
        torch_data = dict_apply(data, torch.from_numpy)
        
        # 获取原始PaintNetODv1Dataloader数据
        # 通过查找当前序列所属的样本索引
        episode_idx = np.searchsorted(self.replay_buffer.episode_ends, idx, side='right')
        if episode_idx >= len(self.sample_info):
            episode_idx = len(self.sample_info) - 1
        
        sample_idx = self.sample_info[episode_idx]['sample_idx']
        original_sample = self.dataset_paintnet[sample_idx]
        
        # 合并数据字典
        final_data = {
            'point_cloud': torch_data['obs']['point_cloud'][0],  # [N, 3]
            'traj': torch_data['action'][0],  # [T, dim]
            'traj_as_pc': original_sample['traj_as_pc'] if 'traj_as_pc' in original_sample else None,
            'stroke_ids': torch_data['stroke_ids'],  # [T]
            'stroke_ids_as_pc': original_sample['stroke_ids_as_pc'] if 'stroke_ids_as_pc' in original_sample else None,
            'stroke_masks': original_sample['stroke_masks'] if 'stroke_masks' in original_sample else None,
            'dirname': self.sample_info[episode_idx]['dirname'],
            'n_strokes': self.sample_info[episode_idx]['n_strokes'],
            'obs': torch_data['obs'],
            'action': torch_data['action']
        }
        
        # 添加其他字段（如果存在）
        for key in ['stroke_prototypes', 'segments_per_stroke', 'max_num_segments', 'points_per_stroke']:
            if key in original_sample:
                final_data[key] = original_sample[key]
        
        return final_data

    def save_replay_buffer(self, file_path):
        """
        将ReplayBuffer保存到文件
        
        参数:
            file_path: 保存文件的路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 创建压缩器（可选，提高存储效率）
        compressor = numcodecs.Blosc(cname='zstd', clevel=3)
        
        # 配置分块大小（可选，提高读写性能）
        chunks = {}
        for key, value in self.replay_buffer.data.items():
            if key == 'point_cloud':
                # 对于点云数据，使用更大的分块大小
                chunks[key] = (1000, min(value.shape[1], 1000), min(value.shape[2], 3))
            else:
                # 对于其他数据，使用默认分块大小
                chunks[key] = (1000,) + value.shape[1:]
        
        # 保存ReplayBuffer
        try:
            start_time = time.time()
            self.replay_buffer.save_to_path(
                file_path,
                chunks=chunks,
                compressors=compressor,
                if_exists='replace'
            )
            # 保存示例信息（pickle不能处理的部分单独保存）
            np.save(file_path + '_sample_info.npy', np.array(self.sample_info, dtype=object))
            
            print(f"ReplayBuffer保存成功，耗时: {time.time() - start_time:.2f}秒")
            print(f"文件路径: {file_path}")
            # 获取文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"文件大小: {file_size_mb:.2f} MB")
        except Exception as e:
            print(f"保存ReplayBuffer时出错: {e}")
            raise

    def load_replay_buffer(self, file_path):
        """
        从文件加载ReplayBuffer
        
        参数:
            file_path: ReplayBuffer文件路径
            
        返回:
            加载的ReplayBuffer对象
        """
        try:
            # 加载ReplayBuffer
            replay_buffer = ReplayBuffer.copy_from_path(file_path)
            
            # 加载示例信息
            sample_info_path = file_path + '_sample_info.npy'
            if os.path.exists(sample_info_path):
                self.sample_info = np.load(sample_info_path, allow_pickle=True).tolist()
            else:
                print(f"警告：未找到示例信息文件 {sample_info_path}，将使用空示例信息")
                self.sample_info = []
            
            print(f"ReplayBuffer加载成功")
            print(f"轨迹数: {replay_buffer.n_episodes}, 总时间步: {replay_buffer.n_steps}")
            print(f"数据键: {list(replay_buffer.keys())}")
            
            return replay_buffer
        except Exception as e:
            print(f"加载ReplayBuffer时出错: {e}")
            print("将创建新的ReplayBuffer")
            return self._create_replay_buffer()

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
    horizon=8,  # 时间步长度
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

