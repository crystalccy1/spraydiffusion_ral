#!/usr/bin/env python3
"""
Script to create and save replay buffer zarr files from preprocessed samples.
This significantly speeds up dataset loading for SprayDiffusion training.

Example usage:
    python scripts/create_replay_buffer.py --dataset cuboids_v2 --split train --output cache/spraydiffusion/replay_buffer_train.zarr

To create for all datasets and splits:
    python scripts/create_replay_buffer.py --dataset all --split all
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
import numcodecs
from omegaconf import OmegaConf
import zarr
from pathlib import Path

# 添加工程根目录到搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入需要的模块
from utils.dataset.paintnet_ODv1 import PaintNetODv1Dataloader
from utils.common import load_config


class ReplayBuffer:
    """
    A simple replay buffer implementation that stores data in a zarr file.
    """
    def __init__(self, store_path, store_mode='a', horizon=None, data_specs=None, verbose=True):
        """
        Initialize replay buffer.
        
        Args:
            store_path: Path to zarr store
            store_mode: Mode to open zarr store ('a' for append, 'w' for write)
            horizon: Maximum number of samples to store (None for unlimited)
            data_specs: List of data specifications
            verbose: Print verbose output
        """
        self.store_path = store_path
        self.store_mode = store_mode
        self.horizon = horizon
        self.data_specs = data_specs or []
        self.verbose = verbose
        self._size = 0
        
        # Create directory if it doesn't exist
        if store_path:
            os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
            self.store = zarr.open(store_path, mode=store_mode)
            
            # Initialize metadata
            if 'metadata' not in self.store:
                self.store.create_group('metadata')
            
            # Initialize index
            if 'index' not in self.store:
                self.store.create_dataset('index', shape=(0,), dtype=np.int64, 
                                         chunks=(10000,), compressor=numcodecs.Blosc(cname='zstd', clevel=3))
            
            self._size = len(self.store['index'])
            
            if self.verbose:
                print(f"Opened replay buffer at {store_path} with {self._size} samples")
    
    def add_sample(self, sample):
        """
        Add a single sample to the replay buffer.
        
        Args:
            sample: Dictionary of data to add
        """
        # Create arrays if they don't exist
        for key, value in sample.items():
            if key not in self.store:
                if isinstance(value, np.ndarray):
                    shape = list(value.shape)
                    dtype = value.dtype
                    self.store.create_dataset(
                        key, 
                        shape=(0,) + tuple(shape[1:]), 
                        dtype=dtype,
                        chunks=(1,) + tuple(shape[1:]),
                        compressor=numcodecs.Blosc(cname='zstd', clevel=3)
                    )
                elif isinstance(value, (dict, list)):
                    if key not in self.store:
                        self.store.create_dataset(
                            key,
                            shape=(0,),
                            dtype=object,
                            object_codec=numcodecs.JSON()
                        )
        
        # Add data to arrays
        for key, value in sample.items():
            if key in self.store:
                if isinstance(value, np.ndarray):
                    # Resize array if needed
                    if self.store[key].shape[0] <= self._size:
                        new_shape = list(self.store[key].shape)
                        new_shape[0] = self._size + 1
                        self.store[key].resize(new_shape)
                    
                    # Add data
                    self.store[key][self._size] = value
                elif isinstance(value, (dict, list)):
                    # Resize array if needed
                    if self.store[key].shape[0] <= self._size:
                        self.store[key].resize(self._size + 1)
                    
                    # Add data
                    self.store[key][self._size] = value
        
        # Update index
        if self.store['index'].shape[0] <= self._size:
            self.store['index'].resize(self._size + 1)
        self.store['index'][self._size] = self._size
        
        # Increment size
        self._size += 1
    
    def __len__(self):
        """Get the number of samples in the replay buffer"""
        return self._size


def get_dataset_downscale_factor(dataset_name):
    """
    Get the downscale factor for a dataset based on its name.
    
    Args:
        dataset_name: Name of dataset
    
    Returns:
        Downscale factor (float)
    """
    # 处理格式化和规范化数据集名称
    if isinstance(dataset_name, str):
        dataset_name = dataset_name.lower()
        if '/' in dataset_name:
            dataset_name = dataset_name.split('/')[-1]
    
        # 移除版本号
        if '-v' in dataset_name:
            dataset_name = dataset_name.split('-v')[0]
        elif '_v' in dataset_name:
            dataset_name = dataset_name.split('_v')[0]
    
    # 返回特定数据集的缩放因子
    if dataset_name in ['cuboid', 'cuboids']:
        return 0.02
    elif dataset_name in ['window', 'windows']:
        return 0.01
    elif dataset_name in ['shelf', 'shelves']:
        return 0.015
    elif dataset_name in ['container', 'containers']:
        return 0.015
    else:
        print(f"Warning: Unknown dataset {dataset_name}, using default scale factor 0.02")
        return 0.02


def parse_args():
    parser = argparse.ArgumentParser(description='Create ReplayBuffer zarr files from preprocessed samples')
    parser.add_argument('--dataset', type=str, default='windows-v2', 
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--dataset_dir', type=str, default='data',
                        help='Root directory containing dataset folders')
    parser.add_argument('--split', type=str, default='train', 
                        choices=['train', 'test', 'all'],
                        help='Dataset split')
    parser.add_argument('--config', type=str, default='configs/spraydiffusion/defaults.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for zarr file. If not provided, a default path will be used.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--force', action='store_true', 
                        help='Force rebuild even if zarr file exists')
    parser.add_argument('--pc_points', type=int, default=None,
                        help='Override number of point cloud points')
    parser.add_argument('--horizon', type=int, default=1, 
                        help='Horizon length for replay buffer')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    return parser.parse_args()


def get_local_dataset_paths(dataset_name, dataset_dir='data'):
    """
    获取本地数据集路径，无需环境变量
    
    Args:
        dataset_name: 数据集名称 (e.g., windows-v2)
        dataset_dir: 数据集目录 (默认: data)
    
    Returns:
        数据集路径列表
    """
    # 先尝试使用相对路径
    dataset_path = os.path.join(dataset_dir, dataset_name)
    
    # 检查路径是否存在
    if os.path.isdir(dataset_path):
        return [dataset_path]
    
    # 如果不存在，尝试其他变体
    # 例如可能数据集名为windows-v2但目录是windows_v2
    alt_dataset_name = dataset_name.replace('-', '_')
    alt_dataset_path = os.path.join(dataset_dir, alt_dataset_name)
    
    if os.path.isdir(alt_dataset_path):
        return [alt_dataset_path]
    
    # 再尝试不带版本号的方式
    base_name = dataset_name.split('-')[0]
    base_path = os.path.join(dataset_dir, base_name)
    
    if os.path.isdir(base_path):
        return [base_path]
    
    # 最后尝试从函数导入，如果设置了环境变量
    try:
        from utils.disk import get_dataset_paths
        return get_dataset_paths(dataset_name)
    except (ImportError, AssertionError):
        # 如果上面方法都失败，返回空列表
        return []


def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return OmegaConf.create(config)


def create_replay_buffer(
    dataset, 
    output_path,
    pc_points=5120,
    traj_points=500,
    lambda_points=1, 
    overlapping=0, 
    split='train', 
    horizon=100,
    normalization='per-dataset',
    weight_orient=0.25,
    asymm_overlapping=False,
    extra_data=["orientnorm"],
    augmentations=[],
    verbose=True,
    config=None
):
    """
    创建并填充回放缓冲区
    
    Args:
        dataset: 数据集名称或路径
        output_path: 回放缓冲区输出路径
        pc_points: 点云中点的数量
        traj_points: 轨迹中点的数量
        lambda_points: Lambda点数量
        overlapping: 重叠程度
        split: 数据分割 ('train', 'val', 'test')
        horizon: 回放缓冲区大小
        normalization: 归一化方法
        weight_orient: 方向权重
        asymm_overlapping: 是否使用非对称重叠
        extra_data: 额外数据列表
        augmentations: 增强列表
        verbose: 是否显示详细输出
        config: 配置对象
    
    Returns:
        创建的回放缓冲区
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if verbose:
        print(f"创建回放缓冲区: {output_path}")
        print(f"数据集: {dataset}, 分割: {split}")
        print(f"点云点数: {pc_points}, 轨迹点数: {traj_points}")
        print(f"归一化: {normalization}, Lambda点数: {lambda_points}")
    
    # 获取数据集缩放因子
    data_scale_factor = get_dataset_downscale_factor(dataset)
    
    # 创建数据加载器
    try:
        dataloader = PaintNetODv1Dataloader(
            roots=dataset, 
            dataset=dataset, 
            pc_points=pc_points,
            traj_points=traj_points,
            lambda_points=lambda_points,
            overlapping=overlapping,
            split=split,
            stroke_pred=False,
            stroke_points=0,
            extra_data=extra_data,
            weight_orient=weight_orient,
            cache_size=1000,
            overfitting=0,
            augmentations=augmentations,
            normalization=normalization,
            data_scale_factor=data_scale_factor,
            config=config
        )
    except FileNotFoundError as e:
        # 尝试使用替代格式的数据集名称
        alt_dataset = dataset
        if '-' in dataset:
            alt_dataset = dataset.replace('-', '_')
        elif '_' in dataset:
            alt_dataset = dataset.replace('_', '-')
        
        if verbose:
            print(f"原始数据集路径 '{dataset}' 不存在，尝试使用替代路径: '{alt_dataset}'")
        
        dataloader = PaintNetODv1Dataloader(
            roots=alt_dataset, 
            dataset=alt_dataset, 
            pc_points=pc_points,
            traj_points=traj_points,
            lambda_points=lambda_points,
            overlapping=overlapping,
            split=split,
            stroke_pred=False,
            stroke_points=0,
            extra_data=extra_data,
            weight_orient=weight_orient,
            cache_size=1000,
            overfitting=0,
            augmentations=augmentations,
            normalization=normalization,
            data_scale_factor=data_scale_factor,
            config=config
        )
    
    if verbose:
        print(f"数据集加载成功，样本数量: {len(dataloader)}")
    
    # 创建回放缓冲区
    buffer = ReplayBuffer(output_path, store_mode='w', horizon=horizon, verbose=verbose)
    
    # 填充回放缓冲区
    for i in tqdm(range(len(dataloader)), desc="填充回放缓冲区"):
        sample = dataloader[i]
        
        # 处理非numpy数组数据
        processed_sample = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                processed_sample[k] = v.cpu().numpy()
            else:
                processed_sample[k] = v
        
        buffer.add_sample(processed_sample)
        
        # 如果达到horizon，停止
        if horizon is not None and i >= horizon - 1:
            break
    
    if verbose:
        print(f"回放缓冲区创建完成，样本数量: {len(buffer)}")
    
    return buffer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="创建回放缓冲区")
    parser.add_argument('--dataset', type=str, default='windows-v2', help='数据集名称')
    parser.add_argument('--split', type=str, default='train', help='数据分割(train/val/test)')
    parser.add_argument('--pc_points', type=int, default=5120, help='点云点数')
    parser.add_argument('--traj_points', type=int, default=500, help='轨迹点数')
    parser.add_argument('--lambda_points', type=int, default=1, help='Lambda点数')
    parser.add_argument('--overlapping', type=float, default=0.0, help='重叠程度')
    parser.add_argument('--normalization', type=str, default='per-dataset', help='归一化方法')
    parser.add_argument('--weight_orient', type=float, default=0.25, help='方向权重')
    parser.add_argument('--horizon', type=int, default=None, help='回放缓冲区大小')
    parser.add_argument('--output_dir', type=str, default='replay_buffers', help='输出目录')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    
    args = parser.parse_args()
    
    # 处理数据集路径
    dataset_name = args.dataset
    dataset_path = dataset_name
    
    # 检查是否为相对路径
    if not os.path.isabs(dataset_name) and not dataset_name.startswith('data/'):
        # 尝试找到正确的数据集路径
        base_paths = ['data']
        found = False
        
        for base in base_paths:
            # 尝试两种格式 (windows-v2 和 windows_v2)
            for fmt in [dataset_name, dataset_name.replace('-', '_'), dataset_name.replace('_', '-')]:
                path = os.path.join(base, fmt)
                if os.path.exists(path):
                    dataset_path = path
                    found = True
                    break
            
            if found:
                break
        
        if not found and args.verbose:
            print(f"警告: 未找到数据集路径 '{dataset_name}'，将使用原始路径")
    
    if args.verbose:
        print(f"使用数据集路径: {dataset_path}")
    
    # 加载配置（如果提供）
    config = None
    if args.config:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            if args.verbose:
                print(f"配置文件 '{args.config}' 不存在，使用命令行参数")
    
    # 构建输出路径
    norm_str = args.normalization.replace('-', '_')
    output_path = os.path.join(
        args.output_dir, 
        f"{dataset_name.replace('/', '_')}_{args.split}_{args.pc_points}_{args.traj_points}_{norm_str}.zarr"
    )
    
    # 创建回放缓冲区
    create_replay_buffer(
        dataset=dataset_path,
        output_path=output_path,
        pc_points=args.pc_points,
        traj_points=args.traj_points,
        lambda_points=args.lambda_points,
        overlapping=args.overlapping,
        split=args.split,
        horizon=args.horizon,
        normalization=args.normalization,
        weight_orient=args.weight_orient,
        verbose=args.verbose,
        config=config
    )


if __name__ == "__main__":
    main() 