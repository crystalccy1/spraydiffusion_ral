from typing import Dict
import torch
import numpy as np
import copy
import os
import time
import open3d as o3d
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
            replay_buffer_save_path=None,
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
        self.replay_buffer_save_path = replay_buffer_save_path
        
        # 添加警告计数器，用于限制显示的警告数量
        self.warning_counter = {
            'sample_info_empty': 0,
            'episode_idx_out_of_range': 0,
            'sample_idx_out_of_range': 0,
            'general_error': 0,
            'skipped_strokes': 0,
            'sample_info_missing_ep': 0,
            'paintnet_datapath_missing': 0
        }
        self.max_warnings = 5  # 每种警告最多显示次数

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
        
        # 确保 dataset_paintnet.datapath 可访问且包含有效路径信息
        if not hasattr(self.dataset_paintnet, 'datapath') or not self.dataset_paintnet.datapath:
            if self.warning_counter['paintnet_datapath_missing'] < self.max_warnings:
                print("警告: self.dataset_paintnet.datapath 不存在或为空。网格数据可能无法加载。")
                self.warning_counter['paintnet_datapath_missing'] +=1
        
        # 创建或加载ReplayBuffer
        load_success = False
        if replay_buffer_path is not None and os.path.exists(replay_buffer_path) and not force_rebuild_buffer:
            # 如果提供了ReplayBuffer路径，并且文件存在，从文件加载
            print(f"从文件加载ReplayBuffer: {replay_buffer_path}")
            start_time = time.time()
            try:
                self.replay_buffer = self.load_replay_buffer(replay_buffer_path)
                # 验证sample_info是否正确加载
                if hasattr(self, 'sample_info') and self.sample_info and len(self.sample_info) > 0:
                    print(f"sample_info加载成功，共 {len(self.sample_info)} 条记录")
                    load_success = True
                else:
                    print(f"警告：sample_info为空或无效，将重新创建ReplayBuffer")
                    load_success = False
            except Exception as e:
                print(f"加载ReplayBuffer时出错: {e}，将重新创建")
                load_success = False
                
            if load_success:
                print(f"加载ReplayBuffer完成，耗时: {time.time() - start_time:.2f}秒")
        
        if not load_success:
            # 创建新的ReplayBuffer
            print("创建新的ReplayBuffer...")
            start_time = time.time()
            self.replay_buffer = self._create_replay_buffer()
            print(f"创建ReplayBuffer完成，耗时: {time.time() - start_time:.2f}秒")

            # replay_buffer_path =  'data/windows-v2_train.zarr' if split == 'train' else 'data/windows-v2_test.zarr'
            replay_buffer_path =  replay_buffer_save_path
            
            # 如果提供了ReplayBuffer路径，保存到文件
            if replay_buffer_path is not None:
                print(f"保存ReplayBuffer到文件: {replay_buffer_path}")
                try:
                    self.save_replay_buffer(replay_buffer_path)
                    print(f"ReplayBuffer保存成功")
                except Exception as e:
                    print(f"保存ReplayBuffer时发生错误: {e}，将继续但不保存缓存")
        
        # 加载episode_ends数据，用于确定每个轨迹的起始和结束点
        if 'meta' in self.replay_buffer.root and 'episode_ends' in self.replay_buffer.root['meta']:
            self.episode_ends = self.replay_buffer.root['meta']['episode_ends'][:]  # 使用[:]加载到内存中
            print(f"加载episode_ends数据，形状: {self.episode_ends.shape}, 类型: {type(self.episode_ends)}")
            if len(self.episode_ends) <= 10:  # 只在数量少时打印完整列表
                print(f"Episode结束点: {self.episode_ends}")
            else:
                print(f"Episode结束点 (前5个): {self.episode_ends[:5]} ... (后5个): {self.episode_ends[-5:]}")
            
            # 计算每个episode的起始点
            self.episode_starts = np.zeros_like(self.episode_ends)
            if len(self.episode_ends) > 0:
                self.episode_starts[1:] = self.episode_ends[:-1]
                if len(self.episode_starts) <= 10:  # 只在数量少时打印完整列表
                    print(f"Episode起始点: {self.episode_starts}")
                else:
                    print(f"Episode起始点 (前5个): {self.episode_starts[:5]} ... (后5个): {self.episode_starts[-5:]}")
            
            # 打印每个episode的长度信息
            print("\n===== Episode长度详情 =====")
            episode_lengths = []
            for i in range(len(self.episode_ends)):
                start = int(self.episode_starts[i])
                end = int(self.episode_ends[i])
                length = end - start
                episode_lengths.append(length)
                # 只打印前10个和最后一个，避免输出过多
                if i < 10 or i == len(self.episode_ends) - 1:
                    print(f"Episode {i}: 起点={start}, 终点={end}, 长度={length}")
            
            # 打印长度统计信息
            if episode_lengths:
                min_len = min(episode_lengths)
                max_len = max(episode_lengths)
                avg_len = sum(episode_lengths) / len(episode_lengths)
                print(f"\nEpisode长度统计: 最小={min_len}, 最大={max_len}, 平均={avg_len:.1f}")
                print(f"总数据点: {self.episode_ends[-1] if len(self.episode_ends) > 0 else 0}")
                print("===========================\n")
            
            # 提取完整的ground truth轨迹
            self.all_gt_trajectories = []
            
            # 获取动作数据
            action_data = self.replay_buffer['action']
            print(f"动作数据形状: {action_data.shape}")
            
            # 对每个episode，提取完整的轨迹
            for i in range(len(self.episode_ends)):
                start = int(self.episode_starts[i])
                end = int(self.episode_ends[i])
                
                # 安全检查，确保索引在有效范围内
                if start < 0 or end > len(action_data) or start >= end:
                    print(f"警告: Episode {i} 索引无效，起点: {start}, 终点: {end}, 数据长度: {len(action_data)}")
                    self.all_gt_trajectories.append(None)
                    continue
                
                # 提取这个episode的完整轨迹
                try:
                    episode_traj = action_data[start:end]
                    
                    # 检查轨迹的有效性
                    if len(episode_traj) == 0:
                        print(f"警告: Episode {i} 轨迹为空，起点: {start}, 终点: {end}")
                        self.all_gt_trajectories.append(None)
                        continue
                        
                    # 存储为tensor
                    episode_traj_tensor = torch.from_numpy(episode_traj.astype(np.float32))
                    
                    # 检查是否需要预处理轨迹数据
                    if episode_traj_tensor.ndim == 1:
                        # 如果是一维的，扩展为二维 [T, 1]
                        episode_traj_tensor = episode_traj_tensor.unsqueeze(1)
                        print(f"将Episode {i} 的一维轨迹扩展为二维: {episode_traj_tensor.shape}")
                    
                    self.all_gt_trajectories.append(episode_traj_tensor)
                    
                    # 只打印前3个和最后一个episode的详细信息，避免日志过多
                    if i < 3 or i == len(self.episode_ends)-1:
                        print(f"Episode {i} 轨迹形状: {episode_traj_tensor.shape}, 起点: {start}, 终点: {end}")
                except Exception as e:
                    print(f"处理Episode {i} 轨迹时出错: {e}, 起点: {start}, 终点: {end}")
                    self.all_gt_trajectories.append(None)
            
            print(f"成功加载 {len(self.all_gt_trajectories)} 个ground truth轨迹")
            
            # 数据验证
            valid_trajs = sum(1 for traj in self.all_gt_trajectories if traj is not None)
            print(f"有效轨迹数量: {valid_trajs}/{len(self.all_gt_trajectories)}")
            if valid_trajs > 0:
                # 打印轨迹长度统计
                traj_lengths = [traj.shape[0] for traj in self.all_gt_trajectories if traj is not None]
                min_len = min(traj_lengths) if traj_lengths else 0
                max_len = max(traj_lengths) if traj_lengths else 0
                avg_len = sum(traj_lengths) / len(traj_lengths) if traj_lengths else 0
                print(f"轨迹长度统计 - 最小: {min_len}, 最大: {max_len}, 平均: {avg_len:.1f}")
                
                # 打印轨迹维度信息
                dim_info = {}
                for traj in self.all_gt_trajectories:
                    if traj is not None:
                        shape_key = f"{len(traj.shape)}D: {traj.shape}"
                        dim_info[shape_key] = dim_info.get(shape_key, 0) + 1
                
                print("轨迹维度统计:")
                for shape_key, count in dim_info.items():
                    print(f"  - {shape_key}: {count}个轨迹 ({(count/valid_trajs*100):.1f}%)")
                
                # 检查是否有异常长度的轨迹
                outlier_threshold = 3 * avg_len  # 超过平均长度3倍的视为异常
                outliers = [i for i, traj in enumerate(self.all_gt_trajectories) 
                           if traj is not None and traj.shape[0] > outlier_threshold]
                
                if outliers:
                    print(f"发现 {len(outliers)} 个异常长度的轨迹 (长度 > {outlier_threshold:.1f}):")
                    for i in outliers[:3]:  # 只打印前3个
                        print(f"  - Episode {i}: 长度 = {self.all_gt_trajectories[i].shape[0]}")
                    if len(outliers) > 3:
                        print(f"  - ... 以及 {len(outliers)-3} 个其他异常轨迹")
        else:
            print("找不到episode_ends数据，无法加载ground truth轨迹")
            self.episode_ends = None
            self.episode_starts = None
            self.all_gt_trajectories = []
        
        # 最后检查确认sample_info存在且有效
        if not hasattr(self, 'sample_info') or self.sample_info is None or len(self.sample_info) == 0:
            print("警告：初始化后sample_info仍然为空，创建默认sample_info")
            try:
                # 创建默认的sample_info
                self.sample_info = []
                for i in range(self.replay_buffer.n_episodes):
                    safe_idx = i % len(self.dataset_paintnet) if len(self.dataset_paintnet) > 0 else 0
                    self.sample_info.append({
                        'dirname': f"default_sample_{i}",
                        'n_strokes': 1,
                        'sample_idx': safe_idx
                    })
                print(f"已创建默认sample_info，共 {len(self.sample_info)} 条记录")
            except Exception as e:
                print(f"创建默认sample_info时出错: {e}，可能会导致后续使用问题")
        
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

        # SequenceSampler 被用于创建一个训练数据采样器，
        # 该采样器可以从 SprayDiffusionDataset 中的所有轨迹数据里高效地采样出固定长度（horizon）的连续片段，
        # 并且只从训练集数据（通过 train_mask 指定）中采样。
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
        """创建符合SequenceSampler需要的ReplayBuffer结构"""
        # 创建一个空的ReplayBuffer
        replay_buffer = ReplayBuffer.create_empty_numpy()
        
        # 收集所有样本数据
        episode_ends = []
        total_steps = 0
        
        # 存储所有轨迹和点云数据
        all_trajs = []
        all_point_clouds = []
        all_stroke_ids = []
        all_sample_info = []  # 存储其他信息，如dirname等
        
        # 第一步：预处理，确定特征维度
        feature_dim = None
        pc_points = None
        
        # 检查第一个有效样本确定维度
        for idx in range(len(self.dataset_paintnet)):
            try:
                sample = self.dataset_paintnet[idx]
                traj = sample['traj']
                point_cloud = sample['point_cloud']
                
                if isinstance(traj, torch.Tensor):
                    traj = traj.detach().cpu().numpy()
                elif not isinstance(traj, np.ndarray):
                    traj = np.array(traj)
                    
                if isinstance(point_cloud, torch.Tensor):
                    point_cloud = point_cloud.detach().cpu().numpy()
                elif not isinstance(point_cloud, np.ndarray):
                    point_cloud = np.array(point_cloud)
                
                # 确定特征维度
                if traj.size > 0:
                    if len(traj.shape) >= 2:
                        feature_dim = traj.shape[-1]
                    else:
                        # 单维度特征，手动设置为1
                        feature_dim = 1
                
                # 确定点云维度
                if point_cloud.size > 0:
                    if len(point_cloud.shape) >= 2:
                        pc_points = point_cloud.shape[0]
                    
                # 如果找到了有效的维度信息，就跳出循环
                if feature_dim is not None and pc_points is not None:
                    break
            except Exception as e:
                print(f"预处理样本 {idx} 时出错: {e}")
                continue
        
        # 如果无法确定维度，使用配置中的默认值
        if feature_dim is None:
            feature_dim = 3  # 默认值
            print(f"警告：无法从数据确定特征维度，使用默认值 {feature_dim}")
            
        if pc_points is None:
            pc_points = self.config.pc_points
            print(f"警告：无法从数据确定点云维度，使用配置值 {pc_points}")
        
        # 收集所有数据
        for idx in range(len(self.dataset_paintnet)):
            try:
                sample = self.dataset_paintnet[idx]
                
                # 获取轨迹和点云数据
                traj = sample['traj']
                valid_mask = ~(traj == -100).all(axis=1)
                traj = traj[valid_mask]
                assert traj.min() != -100, "traj 存在全为-100的行"
                point_cloud = sample['point_cloud']
                stroke_ids = sample.get('stroke_ids', None)
                
                # 确保数据是numpy数组
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
                
                # 确保traj至少是二维的
                if len(traj.shape) == 1:
                    traj = traj.reshape(-1, 1)
                
                # 确保点云维度正确
                if point_cloud.shape[0] != pc_points and len(point_cloud.shape) >= 2:
                    # 可能需要重新采样点云，这里简单处理：如果点数过多就裁剪，过少就填充
                    if point_cloud.shape[0] > pc_points:
                        point_cloud = point_cloud[:pc_points]
                    else:
                        padding = np.zeros((pc_points - point_cloud.shape[0], point_cloud.shape[1]))
                        point_cloud = np.concatenate([point_cloud, padding], axis=0)
                
                # 记录步数(轨迹长度)
                num_steps = traj.shape[0]
                if num_steps == 0:
                    print(f"跳过样本 {idx}，轨迹长度为0")
                    continue
                
                total_steps += num_steps
                episode_ends.append(total_steps)
                
                # 处理stroke_ids，如果不存在则创建占位符
                if stroke_ids is None or stroke_ids.size == 0:
                    stroke_ids = np.zeros(num_steps)
                elif stroke_ids.shape[0] != num_steps:
                    # 确保stroke_ids的长度与轨迹一致
                    if stroke_ids.shape[0] > num_steps:
                        stroke_ids = stroke_ids[:num_steps]
                    else:
                        padding = np.zeros(num_steps - stroke_ids.shape[0])
                        stroke_ids = np.concatenate([stroke_ids, padding])
                
                # 存储数据
                all_trajs.append(traj)
                
                # 复制点云以匹配轨迹长度（所有时间步共享同一个点云）
                try:
                    point_cloud_expanded = np.tile(point_cloud[np.newaxis, :, :], (num_steps, 1, 1))
                    all_point_clouds.append(point_cloud_expanded)
                except Exception as e:
                    print(f"处理点云数据时出错: {e}, 点云形状: {point_cloud.shape}, 轨迹长度: {num_steps}")
                    # 创建一个零填充的点云
                    point_cloud_expanded = np.zeros((num_steps, pc_points, 3))
                    all_point_clouds.append(point_cloud_expanded)
                
                all_stroke_ids.append(stroke_ids)
                all_sample_info.append({
                    'dirname': sample.get('dirname', f"sample_{idx}"),
                    'n_strokes': sample.get('n_strokes', 1),
                    'sample_idx': idx
                })
            except Exception as e:
                print(f"处理样本 {idx} 时出错: {e}")
                continue
        
        # 确保有数据可用
        if not all_trajs:
            print("警告：没有有效的轨迹数据！")
            # 创建空的数据结构
            trajs = np.zeros((0, feature_dim))
            point_clouds = np.zeros((0, pc_points, 3))
            stroke_ids = np.zeros(0)
            episode_ends = [0]  # 至少需要一个元素
        else:
            try:
                # 合并所有数据
                trajs = np.concatenate(all_trajs, axis=0)
                point_clouds = np.concatenate(all_point_clouds, axis=0)
                stroke_ids = np.concatenate(all_stroke_ids, axis=0)
            except Exception as e:
                print(f"合并数据时出错: {e}")
                # 回退到安全的数据结构
                trajs = np.zeros((1, feature_dim))
                point_clouds = np.zeros((1, pc_points, 3))
                stroke_ids = np.zeros(1)
                episode_ends = [1]
        
        # 构建数据字典
        data = {
            'action': trajs,
            'point_cloud': point_clouds,
            'stroke_ids': stroke_ids,
            'sample_info': np.zeros(total_steps, dtype=np.int32)  # 占位符
        }
        
        # 更新ReplayBuffer
        replay_buffer.root['meta']['episode_ends'] = np.array(episode_ends, dtype=np.int64)
        replay_buffer.root['data'] = data
        
        # 存储原始样本信息，以便后续检索
        self.sample_info = all_sample_info
        
        print(f"ReplayBuffer创建完成: {len(episode_ends)} 个轨迹, 总共 {total_steps} 个时间步")
        print(f"Action 形状: {trajs.shape}, Point Cloud 形状: {point_clouds.shape}")
        
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
        
        # 获取action数据
        action_data = self.replay_buffer['action'].copy()
        
        # 创建掩码，找出所有-100值的位置
        padding_mask = (action_data == -100)
        if padding_mask.any():
            print(f"[get_normalizer] 发现{padding_mask.sum()}个-100填充值，将在计算归一化参数时排除")
            
            # 第一步：先用均值替换-100值，以便初步计算归一化参数
            # 创建有效数据的掩码（非-100的值）
            valid_mask = ~padding_mask
            
            # 为了避免完全没有有效值的情况，检查每列是否有有效值
            col_has_valid = valid_mask.any(axis=0)
            
            # 对于每个特征维度，计算均值（只考虑非-100值）
            feature_means = np.zeros(action_data.shape[1])
            for i in range(action_data.shape[1]):
                if col_has_valid[i]:
                    valid_values = action_data[valid_mask[:, i], i]
                    if len(valid_values) > 0:
                        feature_means[i] = valid_values.mean()
                    else:
                        feature_means[i] = 0.0  # 如果没有有效值，使用0
                else:
                    feature_means[i] = 0.0  # 整列都是-100时使用0
            
            # 临时用均值替换-100值
            action_data_temp = action_data.copy()
            for i in range(action_data.shape[1]):
                action_data_temp[padding_mask[:, i], i] = feature_means[i]
            
            # 使用临时处理的数据构建初始归一化器
            temp_data = {
                'action': action_data_temp,
                'point_cloud': self.replay_buffer['point_cloud']
            }
            
            temp_normalizer = LinearNormalizer()
            temp_normalizer.fit(data=temp_data, last_n_dims=1, mode=mode, **kwargs)
            
            # 第二步：用得到的offset替换-100值
            if 'action' in temp_normalizer.params_dict:
                offsets = temp_normalizer.params_dict['action']['offset'].numpy()
                
                # 用offset替换-100值
                for i in range(action_data.shape[1]):
                    if i < len(offsets):
                        action_data[padding_mask[:, i], i] = offsets[i]
                    else:
                        # 如果offset维度不足，则使用均值
                        action_data[padding_mask[:, i], i] = feature_means[i]
                
                print(f"[get_normalizer] 已使用offset替换-100填充值")
            else:
                # 如果没有计算出offset，则使用前面计算的均值
                for i in range(action_data.shape[1]):
                    action_data[padding_mask[:, i], i] = feature_means[i]
                print(f"[get_normalizer] 未能获取offset，使用均值替换-100填充值")
        
        # 构建最终归一化数据字典
        data = {
            'action': action_data,
            'point_cloud': self.replay_buffer['point_cloud']
            # 不再单独为prev_true_trajectory创建归一化器，会使用action的归一化器
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # # 手动设置prev_true_trajectory的归一化参数，使其与action相同
        # if 'action' in normalizer.params_dict:
        #     normalizer.params_dict['prev_true_trajectory'] = normalizer.params_dict['action']
        #     print("[get_normalizer] 已将prev_true_trajectory的归一化参数设置为与action相同")
        
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
        traj = sample.get('action', np.zeros((self.horizon, 3))).astype(np.float32)  # [T, D]
        point_cloud = sample.get('point_cloud', np.zeros((self.config.pc_points, 3))).astype(np.float32)  # [N, 3]
        stroke_ids = sample.get('stroke_ids', np.zeros(traj.shape[0])).astype(np.float32)  # [T]

        # 确保 shape 合法
        if traj.ndim == 1:
            traj = traj.reshape(-1, 1)
        if point_cloud.ndim == 1:
            point_cloud = point_cloud.reshape(1, -1)
        if stroke_ids.shape[0] != traj.shape[0]:
            stroke_ids = np.resize(stroke_ids, traj.shape[0])

        pc_with_time = point_cloud[np.newaxis, :, :]     # [1, N, 3]
        traj_with_time = traj[np.newaxis, :, :]          # [1, T, D]

        obs = {'point_cloud': pc_with_time}
        
        # 提取prev_true_trajectory但不放入obs中
        prev_traj = None
        if 'prev_true_trajectory' in sample:
            prev_traj = sample['prev_true_trajectory'].astype(np.float32)
            if prev_traj.ndim > 1:
                prev_traj = prev_traj[0]  # fallback: [1, D] -> [D]

        return {
            'obs': obs,
            'action': traj_with_time,       # [1, T, D]
            'stroke_ids': stroke_ids,       # [T]
            'prev_true_trajectory': prev_traj  # 放在顶层
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据集中的一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            包含观察、动作和轨迹数据的字典
        """
        sequence = self.sampler.sample_sequence(idx)
        data_from_sampler = self._sample_to_data(sequence)
        
        torch_data = dict_apply(data_from_sampler, torch.from_numpy)
        
        T = torch_data['action'][0].shape[0]
        point_cloud = torch_data['obs']['point_cloud'][0]
        point_cloud_expanded = point_cloud
        
        episode_idx_from_sampler = sequence.get('episode_idx', None)
        full_trajectory_from_loaded = None

        if episode_idx_from_sampler is not None and episode_idx_from_sampler < len(self.all_gt_trajectories):
            full_trajectory_from_loaded = self.all_gt_trajectories[episode_idx_from_sampler]

        final_full_trajectory = full_trajectory_from_loaded if full_trajectory_from_loaded is not None else torch_data['action'][0]
        prev_true_trajectory_final = torch_data.get('prev_true_trajectory', final_full_trajectory[0] if final_full_trajectory.numel() > 0 else torch.zeros_like(final_full_trajectory[0] if final_full_trajectory.ndim > 1 else torch.zeros(final_full_trajectory.shape[-1] if final_full_trajectory.numel() >0 else 1, device=final_full_trajectory.device))) # 更鲁棒的默认值
        
        # 返回的字典，确保包含网格数据
        return_dict = {
            'obs': {
                'point_cloud': point_cloud_expanded, \
            },
            'action': torch_data['action'][0],      \
            'traj': torch_data['action'][0],         \
            'stroke_ids': torch_data.get('stroke_ids', None), \
            'full_trajectory': final_full_trajectory,     \
            'gt_trajectory': final_full_trajectory,       \
            'episode_idx': episode_idx_from_sampler if episode_idx_from_sampler is not None else -1, # 提供默认值
            'prev_true_trajectory': prev_true_trajectory_final, \
            'traj_as_pc': sequence.get('traj_as_pc', torch_data['action'][0].cpu().numpy()) # 确保存在
        }
        # 添加 traj_as_pc (如果 sampler 没有提供，则使用 action 作为回退)
        if 'traj_as_pc' not in return_dict or return_dict['traj_as_pc'] is None:
            return_dict['traj_as_pc'] = return_dict['action'].cpu().numpy() # 转换为 NumPy
        
        # 确保 traj_as_pc 是 torch tensor for metrics_handler if it expects so, or handle in runner
        # For now, metrics handler might expect a tensor, let's ensure it's passed as expected for pcd
        # However, if it's already part of `batch` in runner, it's fine.
        # The current runner call `batch['traj_as_pc']` implies it's already in the batch correctly.

        return return_dict

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
            print(f"检查sample_info文件: {sample_info_path}")
            
            sample_info_loaded = False
            if os.path.exists(sample_info_path):
                
                self.sample_info = np.load(sample_info_path, allow_pickle=True).tolist()
                if self.sample_info and len(self.sample_info) > 0:
                    print(f"成功加载sample_info，包含 {len(self.sample_info)} 个样本信息")
                    sample_info_loaded = True
            
            # 如果sample_info没有成功加载或为空，则创建默认的sample_info
            if not sample_info_loaded:
                print(f"为 {replay_buffer.n_episodes} 个轨迹创建默认sample_info")
                self.sample_info = []
                
                # 根据ReplayBuffer的结构创建默认的sample_info
                for i in range(replay_buffer.n_episodes):
                    # 使用dataset_paintnet中的索引，确保不超出范围
                    safe_idx = i % len(self.dataset_paintnet) if len(self.dataset_paintnet) > 0 else 0
                    self.sample_info.append({
                        'dirname': f"default_sample_{i}",
                        'n_strokes': 1,
                        'sample_idx': safe_idx
                    })
                print(f"已创建 {len(self.sample_info)} 个默认sample_info记录")
                
                # 保存创建的sample_info以便下次使用
                try:
                    np.save(sample_info_path, np.array(self.sample_info, dtype=object))
                    print(f"已将新创建的sample_info保存到: {sample_info_path}")
                except Exception as e:
                    print(f"保存新创建的sample_info时出错: {e}")
            
            # 确认sample_info与replay_buffer.n_episodes匹配
            if len(self.sample_info) != replay_buffer.n_episodes:
                print(f"警告: sample_info长度 ({len(self.sample_info)}) 与 replay_buffer轨迹数 ({replay_buffer.n_episodes}) 不匹配")
                # 如果数量不匹配，调整sample_info长度以匹配replay_buffer
                if len(self.sample_info) > replay_buffer.n_episodes:
                    # 如果sample_info较长，截断
                    self.sample_info = self.sample_info[:replay_buffer.n_episodes]
                    print(f"已截断sample_info至 {len(self.sample_info)} 条记录")
                else:
                    # 如果sample_info较短，扩展
                    current_len = len(self.sample_info)
                    for i in range(current_len, replay_buffer.n_episodes):
                        safe_idx = i % len(self.dataset_paintnet) if len(self.dataset_paintnet) > 0 else 0
                        self.sample_info.append({
                            'dirname': f"default_sample_{i}",
                            'n_strokes': 1,
                            'sample_idx': safe_idx
                        })
                    print(f"已扩展sample_info至 {len(self.sample_info)} 条记录")
            
            print(f"ReplayBuffer加载成功")
            print(f"轨迹数: {replay_buffer.n_episodes}, 总时间步: {replay_buffer.n_steps}")
            print(f"数据键: {list(replay_buffer.keys())}")
            print(f"sample_info 记录数: {len(self.sample_info)}")
            
            return replay_buffer
        except Exception as e:
            print(f"加载ReplayBuffer时出错: {e}")
            print("将创建新的ReplayBuffer")
            return self._create_replay_buffer()

class SprayDiffusionCollateBatch(object):
    """
    Handles batching of data from SprayDiffusionDataset.
    This class converts a list of samples into a mini-batch format suitable for model input.
    """

    def __init__(self, config):
        """
        Initialize the CollateFunction object
        
        Args:
            config: Configuration dictionary or object containing data processing parameters
        """
        self.config = config
        self.max_n_strokes = config.max_n_strokes if hasattr(config, 'max_n_strokes') else 10  # Maximum number of strokes in samples

    def __call__(self, data_list): # data 已被使用，改名为 data_list
        point_cloud_list = []
        prev_true_trajectory_list = []
        action_list = []
        stroke_ids_list = []
        full_trajectory_list = []
        episode_idx_list = []
        traj_as_pc_list = [] # 新增：收集 traj_as_pc

        # print(f"[Collate DEBUG] Received data_list of length: {len(data_list)}")

        for i, d_item in enumerate(data_list): # d 已被使用，改名为 d_item
            # print(f"[Collate DEBUG] Item {i} keys: {list(d_item.keys())}")
            point_cloud_list.append(torch.as_tensor(d_item['obs']['point_cloud'], dtype=torch.float))
            
            # 处理 prev_true_trajectory
            prev_traj_tensor = torch.as_tensor(d_item['prev_true_trajectory'], dtype=torch.float)
            if len(prev_traj_tensor.shape) > 1:
                prev_traj_tensor = prev_traj_tensor.reshape(-1)
            prev_true_trajectory_list.append(prev_traj_tensor)

            action_list.append(torch.as_tensor(d_item['action'], dtype=torch.float))
            stroke_ids_list.append(torch.as_tensor(d_item['stroke_ids'], dtype=torch.float) if d_item['stroke_ids'] is not None else torch.empty(0, dtype=torch.float) )
            full_trajectory_list.append(torch.as_tensor(d_item['full_trajectory'], dtype=torch.float))
            
            episode_idx_list.append(torch.tensor([d_item['episode_idx']], dtype=torch.long) if d_item.get('episode_idx') is not None else torch.tensor([-1], dtype=torch.long))
            
            # 处理 traj_as_pc
            traj_as_pc_np = d_item.get('traj_as_pc')
            if traj_as_pc_np is None: # 如果 __getitem__ 没有提供 traj_as_pc, 从 action 创建
                traj_as_pc_np = d_item['action']
                if isinstance(traj_as_pc_np, torch.Tensor):
                    traj_as_pc_np = traj_as_pc_np.cpu().numpy()
            traj_as_pc_list.append(torch.as_tensor(traj_as_pc_np, dtype=torch.float))

        # 批处理张量数据
        point_cloud = torch.stack(point_cloud_list)
        prev_true_trajectory = torch.stack(prev_true_trajectory_list)
        actions = torch.stack(action_list)
        stroke_ids = torch.stack(stroke_ids_list)
        episode_idx_tensor = torch.stack(episode_idx_list).squeeze() # [B]

        # 对 full_trajectory 和 traj_as_pc 进行填充以匹配最大长度
        max_len_full_traj = max(t.shape[0] for t in full_trajectory_list) if full_trajectory_list else 0
        padded_full_trajectories = []
        for t in full_trajectory_list:
            padding_needed = max_len_full_traj - t.shape[0]
            if padding_needed > 0:
                padding = torch.full((padding_needed, t.shape[1]), -100.0, dtype=t.dtype, device=t.device)
                padded_full_trajectories.append(torch.cat((t, padding), dim=0))
            else:
                padded_full_trajectories.append(t)
        full_trajectory_batch = torch.stack(padded_full_trajectories)

        max_len_traj_as_pc = max(t.shape[0] for t in traj_as_pc_list) if traj_as_pc_list else 0
        padded_traj_as_pc_list = []
        for t_pc in traj_as_pc_list:
            padding_needed_pc = max_len_traj_as_pc - t_pc.shape[0]
            if padding_needed_pc > 0:
                # 确保 t_pc 至少是二维的，以便知道特征维度
                current_D_feat_pc = t_pc.shape[1] if t_pc.ndim > 1 and t_pc.shape[0] > 0 else (actions.shape[-1] if actions.numel() > 0 else 1)
                padding_pc = torch.full((padding_needed_pc, current_D_feat_pc), -100.0, dtype=t_pc.dtype, device=t_pc.device)
                padded_traj_as_pc_list.append(torch.cat((t_pc, padding_pc), dim=0))
            else:
                padded_traj_as_pc_list.append(t_pc)
        traj_as_pc_batch = torch.stack(padded_traj_as_pc_list)

        batch = {
            'obs': {
                'point_cloud': point_cloud,
            },
            'action': actions,
            'stroke_ids': stroke_ids,
            'full_trajectory': full_trajectory_batch,
            'prev_true_trajectory': prev_true_trajectory,
            'episode_idx': episode_idx_tensor, # [B]
            'traj_as_pc': traj_as_pc_batch # 填充后的 traj_as_pc 张量 [B, max_len, D]
        }
        return batch
    
    def add_fake_vectors(self, list_of_vectors, total_needed):
        """
        Add fake vectors to make each element in the list have total_needed vectors
        
        Args:
            list_of_vectors: List of vector arrays [(N1, D), (N2, D), (N3, D)]
            total_needed: Total number of vectors needed for each element
            
        Returns:
            list of vectors: [(total_needed, D), (total_needed, D), ...]
        """
        fake_value = -100
        
        vectors_dims = np.array([vec.shape[-1] for vec in list_of_vectors])
        assert np.all(vectors_dims == vectors_dims[0]), 'Some vectors have different dimensionality than others'
        
        vectors_dim = vectors_dims[0]  # Dimensionality of each vector
        
        out_list_of_vectors = []
        for vec in list_of_vectors:
            # vec: (N1, D)
            assert vec.ndim == 2
            
            num_of_real_vectors = vec.shape[0]
            num_of_fake_vectors = total_needed - num_of_real_vectors
            
            if num_of_fake_vectors > 0:
                if isinstance(vec, torch.Tensor):
                    fake_vectors = torch.full((num_of_fake_vectors, vectors_dim), fake_value, dtype=vec.dtype, device=vec.device)
                    out_list_of_vectors.append(torch.cat((vec, fake_vectors), dim=0))
                else:  # numpy array
                    fake_vectors = fake_value * np.ones((num_of_fake_vectors, vectors_dim))
                    out_list_of_vectors.append(np.concatenate((vec, fake_vectors), axis=0))
            else:
                out_list_of_vectors.append(vec)
                
        return out_list_of_vectors
    
    def add_fake_vectors_v2(self, matrix, total_needed):
        """
        Add fake vectors to a single sequence of vectors
        
        Args:
            matrix: [N, D]
            total_needed: Total number of vectors needed
            
        Returns:
            matrix: [total_needed, D]
        """
        assert matrix.ndim == 2
        fake_value = -100
        N, D = matrix.shape
        
        n_fakes = total_needed - N
        
        if n_fakes > 0:
            if isinstance(matrix, torch.Tensor):
                fake_vectors = torch.full((n_fakes, D), fake_value, dtype=matrix.dtype, device=matrix.device)
                return torch.cat((matrix, fake_vectors), dim=0)
            else:  # numpy array
                fake_vectors = fake_value * np.ones((n_fakes, D))
                return np.concatenate((matrix, fake_vectors), axis=0)
        else:
            return matrix
