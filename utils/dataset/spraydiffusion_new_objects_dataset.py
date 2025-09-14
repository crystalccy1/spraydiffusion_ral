import torch
import numpy as np
import os
import logging
from torch.utils.data import Dataset
from pathlib import Path

class SprayDiffusionNewObjectsDataset(Dataset):
    """
    Dataset for SprayDiffusion model using new preprocessed objects from spray_diffusion/data_objects.
    """
    def __init__(self,
                 data_root="/usr/stud/dira/ccy/MaskPlanner/spray_diffusion/data_objects",
                 categories=None,
                 config={},
                 split='test',
                 seed=42):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.config = config
        self.seed = seed
        self.logger = logging.getLogger("SprayDiffusionNewObjectsDataset")
        
        # 如果没有指定categories，则使用所有可用的预处理类别
        if categories is None:
            categories = [
                'Chair', 'Table', 'Desk', 'Bed', 'Cabinet', 
                'ChestOfDrawers', 'Dresser', 'Bookcase', 'Bench', 'Stool'
            ]
        self.categories = categories
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载所有预处理的数据文件"""
        self.data_files = []
        self.point_clouds = []
        self.mesh_vertices = []
        self.mesh_faces = []
        self.object_names = []
        self.object_categories = []
        
        for category in self.categories:
            category_dir = self.data_root / f"{category}_preprocessed"
            if not category_dir.exists():
                self.logger.warning(f"Category directory {category_dir} does not exist, skipping.")
                continue
                
            # 查找所有.npz文件
            npz_files = list(category_dir.glob("*.npz"))
            self.logger.info(f"Found {len(npz_files)} objects in {category}")
            
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    
                    # 检查是否有visual_前缀或collision_前缀的数据
                    if 'visual_point_cloud' in data:
                        point_cloud = data['visual_point_cloud']
                        mesh_vertices = data['visual_mesh_vertices']
                        mesh_faces = data['visual_mesh_faces']
                    elif 'collision_point_cloud' in data:
                        point_cloud = data['collision_point_cloud']
                        mesh_vertices = data['collision_mesh_vertices']
                        mesh_faces = data['collision_mesh_faces']
                    else:
                        self.logger.warning(f"No point cloud data found in {npz_file}")
                        continue
                    
                    object_name = str(data['object_name'])
                    object_category = str(data['category'])
                    
                    # 数据预处理 - 如果需要采样到指定点数
                    pc_points = getattr(self.config, 'pc_points', 1024)
                    if len(point_cloud) > pc_points:
                        # 随机采样
                        np.random.seed(self.seed)
                        indices = np.random.choice(len(point_cloud), pc_points, replace=False)
                        point_cloud = point_cloud[indices]
                    elif len(point_cloud) < pc_points:
                        # 重复采样
                        np.random.seed(self.seed)
                        indices = np.random.choice(len(point_cloud), pc_points, replace=True)
                        point_cloud = point_cloud[indices]
                    
                    self.data_files.append(npz_file)
                    self.point_clouds.append(point_cloud.astype(np.float32))
                    self.mesh_vertices.append(mesh_vertices.astype(np.float32))
                    self.mesh_faces.append(mesh_faces.astype(np.int32))
                    self.object_names.append(object_name)
                    self.object_categories.append(object_category)
                    
                except Exception as e:
                    self.logger.error(f"Error loading {npz_file}: {e}")
                    continue
        
        self.logger.info(f"Successfully loaded {len(self.point_clouds)} objects from {len(self.categories)} categories")
        
        if len(self.point_clouds) > 0:
            pc_counts = [pc.shape[0] for pc in self.point_clouds]
            self.logger.info(f"Point cloud size statistics - Min: {min(pc_counts)}, Max: {max(pc_counts)}")
            
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.point_clouds):
            raise IndexError(f"Index {idx} out of range (0-{len(self.point_clouds)-1})")
            
        point_cloud = self.point_clouds[idx]
        mesh_vertices = self.mesh_vertices[idx]
        mesh_faces = self.mesh_faces[idx]
        object_name = self.object_names[idx]
        object_category = self.object_categories[idx]
        
        # 扩展点云维度以匹配预期格式 (1, N, 3)
        pc_expanded = np.expand_dims(point_cloud, axis=0)
        
        result = {
            'obs': {
                'point_cloud': pc_expanded
            },
            'episode_idx': idx,
            'mesh_vertices': mesh_vertices,
            'mesh_faces': mesh_faces,
            'object_name': object_name,
            'object_category': object_category,
            'data_file': str(self.data_files[idx])
        }
        
        return result 