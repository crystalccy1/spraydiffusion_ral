import torch
import numpy as np
import os
import time
import logging
import open3d as o3d
from torch.utils.data import Dataset
from utils.dataset.paintnet_ODv1 import PaintNetODv1Dataloader
import trimesh

class SprayDiffusionRolloutDataset(Dataset):
    """
    Dataset for SprayDiffusion model evaluation.
    Returns complete trajectories and point cloud observations directly from PaintNet dataset.
    """
    def __init__(self,
                dataset_paths=[],
                config={},
                split='test',
                seed=42):
        super().__init__()
        self.split = split
        self.config = config
        self.dataset_paths = dataset_paths
        self.logger = logging.getLogger("SprayDiffusionRolloutDataset")
        self.warning_counter = {
            'mesh_empty': 0,
            'mesh_load_error': 0,
            'mesh_not_found': 0,
            'datapath_missing_for_item': 0
        }
        self.max_warnings = 5
        
        # 创建PaintNetODv1Dataloader实例
        print(f"初始化PaintNet数据集，路径: {dataset_paths}")
        self.dataset = PaintNetODv1Dataloader(
            roots=dataset_paths,
            dataset=config.dataset,
            pc_points=config.pc_points,
            traj_points=getattr(config, 'traj_points', 1024),
            lambda_points=getattr(config, 'lambda_points', 16),
            overlapping=getattr(config, 'overlapping', 8) if not getattr(config, 'asymm_overlapping', False) else getattr(config, 'lambda_points', 16)-1,
            normalization=getattr(config, 'normalization', 'unit'),
            data_scale_factor=getattr(config, 'data_scale_factor', 1.0),
            extra_data=tuple(config.extra_data) if hasattr(config, 'extra_data') else None,
            weight_orient=getattr(config, 'weight_orient', 0.1),
            # split=split,
            split='test',
            config=config,
            overfitting=(None if getattr(config, 'overfitting', False) is False else config.seed),
            augmentations=getattr(config, 'augmentations', []) if split == 'train' else [],
            train_portion=getattr(config, 'train_portion', None) if split == 'train' else None
        )
        
        print(f"PaintNet数据集加载完成，共 {len(self.dataset)} 个样本")
        
        # 预处理数据
        self._prepare_data()
        
    def _prepare_data(self):
        """预处理数据，准备轨迹、点云和网格文件路径"""
        self.trajectories = []
        self.point_clouds = []
        self.traj_as_pcs = []
        self.stroke_ids_list = []
        self.mesh_file_paths = []
        self.mesh_vertices_np = []
        self.mesh_faces_np = []

        if not hasattr(self.dataset, 'datapath') or not self.dataset.datapath:
            self.logger.warning("PaintNetODv1Dataloader instance (self.dataset) does not have a populated 'datapath' attribute. Mesh loading will fail.")
            for _ in range(len(self.dataset)):
                 self.mesh_file_paths.append(None)

        print("准备数据 (including mesh paths)...")
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                
                traj = sample['traj']
                if traj is None or len(traj) == 0:
                    self.logger.warning(f"Skipping sample {idx} due to empty trajectory.")
                    if hasattr(self.dataset, 'datapath') and self.dataset.datapath :
                         if idx < len(self.mesh_file_paths): self.mesh_file_paths[idx] = None
                         else: self.mesh_file_paths.append(None)
                    continue
                    
                point_cloud = sample['point_cloud']
                traj_as_pc = sample.get('traj_as_pc', None)
                stroke_ids = sample.get('stroke_ids', torch.zeros(len(traj), dtype=torch.long))
                mesh_vertices_np = sample.get('mesh_vertices_np', None)
                mesh_faces_np = sample.get('mesh_faces_np', None)
                
                mesh_path_for_item = None
                if hasattr(self.dataset, 'datapath') and self.dataset.datapath and idx < len(self.dataset.datapath):
                    datapath_entry = self.dataset.datapath[idx]
                    if isinstance(datapath_entry, tuple) and len(datapath_entry) > 0:
                        mesh_path_for_item = datapath_entry[0]
                    elif isinstance(datapath_entry, str):
                        mesh_path_for_item = datapath_entry

                if mesh_path_for_item is None and self.warning_counter['datapath_missing_for_item'] < self.max_warnings:
                    self.logger.warning(f"No mesh path found in self.dataset.datapath for sample index {idx}.")
                    self.warning_counter['datapath_missing_for_item']+=1
                
                self.trajectories.append(traj)
                self.point_clouds.append(point_cloud)
                self.traj_as_pcs.append(traj_as_pc)
                self.stroke_ids_list.append(stroke_ids)
                self.mesh_file_paths.append(mesh_path_for_item)
                self.mesh_vertices_np.append(mesh_vertices_np)
                self.mesh_faces_np.append(mesh_faces_np)
                
                if idx % 100 == 0 and idx > 0:
                    print(f"已处理 {idx}/{len(self.dataset)} 个样本...")
                    
            except Exception as e:
                self.logger.error(f"处理样本 {idx} 时出错: {e}", exc_info=True)
                if len(self.trajectories) < (idx +1) : self.trajectories.append(None)
                if len(self.point_clouds) < (idx +1) : self.point_clouds.append(None)
                if len(self.traj_as_pcs) < (idx +1) : self.traj_as_pcs.append(None)
                if len(self.stroke_ids_list) < (idx +1) : self.stroke_ids_list.append(None)
                if len(self.mesh_file_paths) < (idx +1) : self.mesh_file_paths.append(None)
                continue
        
        valid_indices = [i for i, t in enumerate(self.trajectories) if t is not None]
        self.trajectories = [self.trajectories[i] for i in valid_indices]
        self.point_clouds = [self.point_clouds[i] for i in valid_indices]
        self.traj_as_pcs = [self.traj_as_pcs[i] for i in valid_indices]
        self.stroke_ids_list = [self.stroke_ids_list[i] for i in valid_indices]
        self.mesh_file_paths = [self.mesh_file_paths[i] for i in valid_indices]

        valid_count = len(self.trajectories)
        print(f"成功加载 {valid_count}/{len(self.dataset)} 个有效样本 (after filtering)")
        
        if valid_count > 0:
            traj_lengths = [t.shape[0] for t in self.trajectories]
            print(f"轨迹长度统计 - 最小: {min(traj_lengths)}, 最大: {max(traj_lengths)}, "
                  f"平均: {sum(traj_lengths)/len(traj_lengths):.1f}")
                  
            pc_counts = [pc.shape[0] for pc in self.point_clouds]
            print(f"点云大小统计 - 最小: {min(pc_counts)}, 最大: {max(pc_counts)}")
            
            valid_traj_pc = [t for t in self.traj_as_pcs if t is not None]
            if valid_traj_pc:
                traj_pc_lengths = [t.shape[0] for t in valid_traj_pc]
                print(f"轨迹点云长度统计 - 最小: {min(traj_pc_lengths)}, "
                      f"最大: {max(traj_pc_lengths)}, "
                      f"平均: {sum(traj_pc_lengths)/len(traj_pc_lengths):.1f}")

    def __len__(self):
        if self.split == 'test':
            return len(self.trajectories)
        else:
            return 1
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.trajectories):
            raise IndexError(f"索引 {idx} 超出范围 (0-{len(self.trajectories)-1})")
            
        trajectory = self.trajectories[idx]
        valid_mask = (trajectory != -100).all(axis=1)
        trajectory = trajectory[valid_mask]

        point_cloud = self.point_clouds[idx]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # o3d.io.write_point_cloud(f"point_cloud_{idx}.ply", pcd)

        traj_as_pc = self.traj_as_pcs[idx]
        mesh_vertices_np = self.mesh_vertices_np[idx]
        mesh_faces_np = self.mesh_faces_np[idx]
        # mesh_file_path = self.mesh_file_paths[idx]
        # mesh = trimesh.load(mesh_file_path)
        # mesh_vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
        # mesh_faces_np = np.asarray(mesh.faces, dtype=np.int32)

        pc_expanded = np.expand_dims(point_cloud, axis=0)
        
        # if len(trajectory) > 0:
        #     first_valid_traj = trajectory[0] if (trajectory[0] != -100).all() else np.zeros_like(trajectory[0])
        #     prev_true_trajectory = first_valid_traj.flatten()
        # else:
        #     prev_true_trajectory = np.zeros(24, dtype=np.float32)
        
        # action = trajectory
        
        result = {
            'obs': {
                'point_cloud': pc_expanded
            },
            # 'action': action,
            # 'prev_true_trajectory': prev_true_trajectory,
            'full_trajectory': trajectory,
            'episode_idx': idx,
            'mesh_vertices': mesh_vertices_np,
            'mesh_faces': mesh_faces_np
        }
        
        if traj_as_pc is not None:
            result['gt_traj_as_pc'] = traj_as_pc.astype(np.float32)
            result['traj_as_pc'] = traj_as_pc.astype(np.float32)
            
        if idx < len(self.stroke_ids_list) and self.stroke_ids_list[idx] is not None:
            result['stroke_ids'] = self.stroke_ids_list[idx]
        else:
            result['stroke_ids'] = torch.zeros(trajectory.shape[0], dtype=torch.long) if isinstance(trajectory, torch.Tensor) else torch.zeros(len(trajectory), dtype=torch.long)

        return result