import wandb
import numpy as np
import torch
import tqdm
import os
from torch.utils.data import DataLoader
import time
import logging
import matplotlib.pyplot as plt
import open3d as o3d
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import omegaconf
import random
import imageio # For GIF creation
import shutil # For cleaning up temp GIF frames
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as colors

from spray_diffusion.policy.base_policy import BasePolicy
from spray_diffusion.common.pytorch_util import dict_apply
from spray_diffusion.env_runner.spraydiffusion_runner import SprayDiffusionRunner
import spray_diffusion.common.logger_util as logger_util
from termcolor import cprint

# Import the LossHandler class and related functions
from loss_spraydiffusion_handler import SprayDiffusionLossHandler
from metrics_handler_spraydiffusion import MetricsHandler
# Import the newly created utility functions
from spray_diffusion.env_runner.utils_spraydiffusion import visualize_traj, _save_trajectory_frame

class SprayDiffusionPartialObservationRunner(SprayDiffusionRunner):
    """
    Runner for SprayDiffusion model evaluation with partial observation support.
    Extends SprayDiffusionRunner to apply partial observation masking during evaluation.
    """
    def __init__(self,
                 output_dir,
                 eval_episodes=10,
                 tqdm_interval_sec=1.0,
                 chamfer_distance_threshold=0.05,
                 batch_size=1,  # Always use batch size of 1
                 num_workers=4,
                 seed=42,
                 # Partial observation parameters
                 partial_observation_config=None):
        super().__init__(output_dir, eval_episodes, tqdm_interval_sec, 
                        chamfer_distance_threshold, batch_size, num_workers, seed)
        
        # Initialize partial observation configuration
        if partial_observation_config is None:
            # Default partial observation configuration
            self.partial_observation_config = {
                'enabled': True,
                'face_selection_method': 'x_plane',
                'face_ratio': 0.3,
                'fixed_face_direction': [0, 0, 1],
                'camera_position': [1, 1, 1],
                'noise_level': 0.0,  # No noise during evaluation for consistency
                'augment_rotation': False  # No rotation augmentation during evaluation
            }
        else:
            self.partial_observation_config = partial_observation_config
        
        self.logger = logging.getLogger("SprayDiffusionPartialObservationRunner")
        
        # Log partial observation configuration
        if self.partial_observation_config['enabled']:
            cprint(f"Partial Observation Runner initialized with:", "green")
            cprint(f"  Method: {self.partial_observation_config['face_selection_method']}", "green")
            cprint(f"  Face ratio: {self.partial_observation_config['face_ratio']}", "green")
            cprint(f"  Noise level: {self.partial_observation_config['noise_level']}", "green")
            cprint(f"  Augment rotation: {self.partial_observation_config['augment_rotation']}", "green")
        else:
            cprint("Partial Observation Runner initialized but partial observation is DISABLED", "yellow")

    def apply_partial_observation_mask(self, point_cloud, method='x_plane', face_ratio=0.3, 
                                     fixed_direction=[0, 0, 1], camera_position=[1, 1, 1], 
                                     noise_level=0.02, augment_rotation=True):
        """
        对点云应用部分观察掩码，只保留一个面的点云数据
        
        Args:
            point_cloud: 输入点云 [N, 3] 或 [B, N, 3] 或 [B, T, N, 3]
            method: 面选择方法 ('random', 'fixed', 'camera_view', 'x_plane')
            face_ratio: 保留的点云比例（对x_plane方法无效）
            fixed_direction: 固定面的法向量方向
            camera_position: 相机位置
            noise_level: 噪声水平（对x_plane方法无效）
            augment_rotation: 是否进行旋转增强（对x_plane方法无效）
            
        Returns:
            masked_point_cloud: 掩码后的点云
        """
        if isinstance(point_cloud, torch.Tensor):
            device = point_cloud.device
            is_tensor = True
            pc_np = point_cloud.detach().cpu().numpy()
        else:
            device = None
            is_tensor = False
            pc_np = point_cloud.copy()
        
        # 处理不同维度的点云数据
        if pc_np.ndim == 4:  # [B, T, N, 3] - 批次中的时间序列点云
            batch_size, time_steps, num_points, _ = pc_np.shape
            
            # 首先处理所有时间步，收集所有masked点云
            all_masked_pcs = []
            for b in range(batch_size):
                for t in range(time_steps):
                    single_pc = pc_np[b, t]  # [N, 3]
                    
                    masked_pc = self._apply_single_mask(
                        single_pc, method, face_ratio, fixed_direction, 
                        camera_position, noise_level, augment_rotation
                    )
                    all_masked_pcs.append(masked_pc)
            
            # 计算所有masked点云的统计信息
            point_counts = [pc.shape[0] for pc in all_masked_pcs]
            max_points = max(point_counts)
            min_points = min(point_counts)
            avg_points = int(np.mean(point_counts))
            
            # 使用平均点数作为目标，避免过度padding
            target_points = min(max_points, avg_points + int(avg_points * 0.1))  # 最多增加10%
            
            # 重新组织为批次和时间序列结构
            masked_pcs = []
            pc_idx = 0
            for b in range(batch_size):
                batch_masked_pcs = []
                for t in range(time_steps):
                    pc = all_masked_pcs[pc_idx]
                    pc_idx += 1
                    
                    # 只有当点数少于目标时才进行padding
                    if pc.shape[0] < target_points:
                        n_pad = target_points - pc.shape[0]
                        if pc.shape[0] > 0:
                            # 使用最后几个点进行padding，避免重复单个点
                            pad_indices = np.random.choice(pc.shape[0], n_pad, replace=True)
                            pad_points = pc[pad_indices]
                            pc = np.concatenate([pc, pad_points], axis=0)
                        else:
                            # 如果没有点，用零点填充
                            pc = np.zeros((target_points, 3))
                    elif pc.shape[0] > target_points:
                        # 如果点数超过目标，随机采样到目标数量
                        indices = np.random.choice(pc.shape[0], target_points, replace=False)
                        pc = pc[indices]
                    
                    batch_masked_pcs.append(pc)
                
                batch_result = np.stack(batch_masked_pcs, axis=0)  # [T, target_points, 3]
                masked_pcs.append(batch_result)
            
            result = np.stack(masked_pcs, axis=0)  # [B, T, target_points, 3]
            
        elif pc_np.ndim == 3:  # [B, N, 3] - 批次点云
            batch_size = pc_np.shape[0]
            masked_pcs = []
            
            for b in range(batch_size):
                single_pc = pc_np[b]  # [N, 3]
                
                masked_pc = self._apply_single_mask(
                    single_pc, method, face_ratio, fixed_direction, 
                    camera_position, noise_level, augment_rotation
                )
                masked_pcs.append(masked_pc)
            
            # 找到最大的点数，用于padding
            max_points = max(pc.shape[0] for pc in masked_pcs)
            
            # 将所有masked点云padding到相同大小
            padded_pcs = []
            for i, pc in enumerate(masked_pcs):
                if pc.shape[0] < max_points:
                    # 用最后一个点来padding
                    n_pad = max_points - pc.shape[0]
                    if pc.shape[0] > 0:
                        last_point = pc[-1:].repeat(n_pad, axis=0)
                        padded_pc = np.concatenate([pc, last_point], axis=0)
                    else:
                        # 如果没有点，用零点
                        padded_pc = np.zeros((max_points, 3))
                else:
                    padded_pc = pc
                
                padded_pcs.append(padded_pc)
            
            result = np.stack(padded_pcs, axis=0)
            
        elif pc_np.ndim == 2:  # [N, 3] - 单个点云
            result = self._apply_single_mask(
                pc_np, method, face_ratio, fixed_direction, 
                camera_position, noise_level, augment_rotation
            )
        else:
            raise ValueError(f"Unsupported point cloud dimension: {pc_np.ndim}, shape: {pc_np.shape}")
        
        # 验证结果形状
        expected_last_dim = 3
        if result.shape[-1] != expected_last_dim:
            raise ValueError(f"Invalid result shape: {result.shape}")
        
        # 转换回原始格式
        if is_tensor:
            return torch.from_numpy(result).to(device)
        else:
            return result
    
    def _apply_single_mask(self, point_cloud, method, face_ratio, fixed_direction, 
                          camera_position, noise_level, augment_rotation):
        """
        对单个点云应用掩码
        """
        if point_cloud.shape[0] == 0:
            return point_cloud
        
        # 计算点云中心
        centroid = np.mean(point_cloud, axis=0)
        centered_pc = point_cloud - centroid
        
        if method == 'random':
            # 随机选择一个方向作为"面"的法向量
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
        elif method == 'fixed':
            # 使用固定方向
            direction = np.array(fixed_direction)
            direction = direction / np.linalg.norm(direction)
        elif method == 'camera_view':
            # 基于相机视角选择可见面
            cam_pos = np.array(camera_position)
            direction = cam_pos - centroid
            direction = direction / np.linalg.norm(direction)
        elif method == 'x_plane':
            # 选择x=0面的点（x坐标接近0的点）
            # 直接基于x坐标进行选择，不需要投影
            x_coords = point_cloud[:, 0]
            x_min, x_max = x_coords.min(), x_coords.max()
            x_range = x_max - x_min
            
            # 选择x坐标在最小值附近的点（x=0面）
            # 使用一个相对阈值，选择x坐标较小的点
            threshold = x_min + x_range * 0.1  # 选择x坐标在最小10%范围内的点
            mask = x_coords <= threshold
            
            masked_pc = point_cloud[mask]
            
            # 对于x_plane方法，不添加噪声和旋转增强，保持原始形状
            return masked_pc
        else:
            raise ValueError(f"Unknown face selection method: {method}")
        
        # 计算每个点到"面"的投影距离
        projections = np.dot(centered_pc, direction)
        
        # 选择投影距离最大的点（即朝向选定方向的面）
        threshold = np.percentile(projections, (1 - face_ratio) * 100)
        mask = projections >= threshold
        
        # 如果掩码后的点太少，至少保留一些点
        min_points = max(10, int(point_cloud.shape[0] * 0.05))  # 改为5%而不是10%
        if np.sum(mask) < min_points:
            # 选择投影距离最大的前min_points个点
            top_indices = np.argsort(projections)[-min_points:]
            mask = np.zeros(len(projections), dtype=bool)
            mask[top_indices] = True
        
        masked_pc = point_cloud[mask]
        
        # 添加噪声以增强鲁棒性
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, masked_pc.shape)
            masked_pc = masked_pc + noise
        
        # 旋转增强
        if augment_rotation:
            # 随机小角度旋转
            angle = np.random.uniform(-np.pi/12, np.pi/12)  # ±15度
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            # 罗德里格斯旋转公式
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            cross_matrix = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            
            rotation_matrix = (cos_angle * np.eye(3) + 
                             sin_angle * cross_matrix + 
                             (1 - cos_angle) * np.outer(axis, axis))
            
            masked_pc = masked_pc @ rotation_matrix.T
        
        return masked_pc

    def run(self, policy: BasePolicy, dataloader=None, dataset=None, split='test', run_name='default_run', dataset_name=None, paper_vis_mode=False, visualize_denoising_gifs=False):
        """
        Override the run method to apply partial observation masking to the input data
        """
        ## -------- Initialization ---------------##
        # 设置随机种子以确保结果可复现
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 设置CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        device = policy.device
        
        # Load config from file
        config = omegaconf.OmegaConf.load('configs/spraydiffusion/dp3.yaml')
        
        # Initialize metrics and loss handlers
        metrics_handler = MetricsHandler(config=config, metrics=['pcd', 'smoothness', 'coverage', 'inference_time', 'latency'])
        
        print(f"Using provided dataloader with {len(dataloader)} episodes")
        episode_iterator = dataloader
        # Determine the number of episodes to run, capped at 20 and also by args.eval_episodes
        num_episodes_to_run = min(len(dataloader), self.eval_episodes, 20)
        cprint(f"Will run evaluation for {num_episodes_to_run} episodes (capped at 20, original eval_episodes: {self.eval_episodes}).", "magenta")

        # Log partial observation status
        if self.partial_observation_config['enabled']:
            cprint(f"PARTIAL OBSERVATION ENABLED - Method: {self.partial_observation_config['face_selection_method']}, Ratio: {self.partial_observation_config['face_ratio']}", "red")
        else:
            cprint("PARTIAL OBSERVATION DISABLED - Using full point clouds", "yellow")

        all_metrics_pred_cond = []
        all_metrics_gt_cond = []
        policy.eval()
        
        # Create the base demo directory if it doesn't exist
        base_demo_dir = 'demo'
        os.makedirs(base_demo_dir, exist_ok=True)
        # Create the specific run's demo directory
        run_specific_demo_dir = os.path.join(base_demo_dir, run_name)
        os.makedirs(run_specific_demo_dir, exist_ok=True)

        # Setup output directories based on mode
        if paper_vis_mode:
            if dataset_name is None:
                cprint("Warning: paper_vis_mode is True, but dataset_name was not provided. Falling back to 'unknown_dataset'.", "yellow")
                dataset_name = "unknown_dataset"
            # Sanitize dataset_name for path
            safe_dataset_name = dataset_name.replace(" ", "_").replace("/", "-")
            run_specific_base_dir = os.path.join('paper_vis', safe_dataset_name, run_name)
            cprint(f"Paper visualization mode enabled. Saving outputs to: {run_specific_base_dir}", "green")
        else:
            # Use the output_dir passed during initialization
            if not self.output_dir:
                cprint("Warning: Runner output_dir is not set. Defaulting to current directory for base output.","yellow")
                base_output_dir = "."
            else:
                base_output_dir = self.output_dir
            run_specific_base_dir = os.path.join(base_output_dir, run_name)
            cprint(f"Standard visualization mode. Saving outputs to: {run_specific_base_dir}", "magenta")
        
        # Ensure the base directory for the run exists
        os.makedirs(run_specific_base_dir, exist_ok=True)
        
        with torch.no_grad():
            pbar = tqdm.tqdm(enumerate(episode_iterator), total=num_episodes_to_run, desc=f"Eval on {split}", leave=False, mininterval=self.tqdm_interval_sec)
            ## -------- Traverse through all the episodes ---------------##
            for episode_idx, data in pbar:
                if episode_idx >= num_episodes_to_run:
                    cprint(f"Reached episode limit of {num_episodes_to_run}. Stopping evaluation.", "magenta")
                    break 
                print(f"[PartialObsRunner] Processing episode {episode_idx} / {num_episodes_to_run}")

                # Apply partial observation masking to the input data
                if self.partial_observation_config['enabled']:
                    if 'obs' in data and 'point_cloud' in data['obs']:
                        original_pc = data['obs']['point_cloud']
                        
                        # Log original point cloud info
                        if episode_idx == 0:  # Only log for first episode to avoid spam
                            if isinstance(original_pc, torch.Tensor):
                                orig_shape = original_pc.shape
                            else:
                                orig_shape = np.array(original_pc).shape
                            cprint(f"Original point cloud shape: {orig_shape}", "cyan")
                        
                        # Apply partial observation mask
                        masked_pc = self.apply_partial_observation_mask(
                            original_pc,
                            method=self.partial_observation_config['face_selection_method'],
                            face_ratio=self.partial_observation_config['face_ratio'],
                            fixed_direction=self.partial_observation_config['fixed_face_direction'],
                            camera_position=self.partial_observation_config['camera_position'],
                            noise_level=self.partial_observation_config['noise_level'],
                            augment_rotation=self.partial_observation_config['augment_rotation']
                        )
                        
                        # Update the data with masked point cloud
                        data['obs']['point_cloud'] = masked_pc
                        
                        # Log masked point cloud info
                        if episode_idx == 0:  # Only log for first episode to avoid spam
                            if isinstance(masked_pc, torch.Tensor):
                                masked_shape = masked_pc.shape
                            else:
                                masked_shape = np.array(masked_pc).shape
                            cprint(f"Masked point cloud shape: {masked_shape}", "cyan")
                            
                            # Calculate retention rate
                            if len(orig_shape) >= 2 and len(masked_shape) >= 2:
                                orig_points = orig_shape[-2]  # Second to last dimension should be number of points
                                masked_points = masked_shape[-2]
                                retention_rate = masked_points / max(orig_points, 1) * 100
                                cprint(f"Point retention rate: {retention_rate:.1f}% ({masked_points}/{orig_points})", "cyan")

                # Continue with the rest of the evaluation using the parent class logic
                # but with the modified data that now contains partial observations
                
                mesh_vertices_np = None
                mesh_faces_np = None
                o3d_mesh = None
                if paper_vis_mode:
                    # Assuming data is B=1, access first element
                    mesh_vertices_np = data.get('mesh_vertices', [None])[0]
                    mesh_faces_np = data.get('mesh_faces', [None])[0]
                    if isinstance(mesh_vertices_np, torch.Tensor):
                        mesh_vertices_np = mesh_vertices_np.numpy()
                    if isinstance(mesh_faces_np, torch.Tensor):
                        mesh_faces_np = mesh_faces_np.numpy()
                    
                    # Create Open3D mesh if data is valid
                    if mesh_vertices_np is not None and mesh_faces_np is not None and mesh_vertices_np.shape[0] > 0 and mesh_faces_np.shape[0] > 0:
                        try:
                            o3d_mesh = o3d.geometry.TriangleMesh()
                            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_np)
                            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_np)
                            o3d_mesh.compute_vertex_normals()
                        except Exception as e_mesh:
                            cprint(f"Warning: Failed to create Open3D mesh for episode {episode_idx}: {e_mesh}", "yellow")
                            o3d_mesh = None
                    else:
                        cprint(f"Warning: Could not extract valid mesh data for episode {episode_idx} in paper_vis_mode.", "yellow")
                        o3d_mesh = None

                for condition_mode_str in ["GT_Cond", "Pred_Cond"]:
                    # Create a specific directory for this episode and condition mode
                    episode_condition_output_dir = os.path.join(run_specific_base_dir, f"episode_{episode_idx:03d}_{condition_mode_str}")
                    os.makedirs(episode_condition_output_dir, exist_ok=True)
                    cprint(f"  Saving visualizations for Ep {episode_idx} ({condition_mode_str}) to: {episode_condition_output_dir}", "blue")

                    current_debug_log_path = os.path.join(episode_condition_output_dir, f'debug_log_{split}_{condition_mode_str}.txt')
                    debug_file_already_existed = os.path.exists(current_debug_log_path)
                    with open(current_debug_log_path, 'a') as f:
                        is_empty = os.fstat(f.fileno()).st_size == 0
                        if not debug_file_already_existed or is_empty:
                            f.write(f"Debug log for {split} evaluation ({condition_mode_str}) - PARTIAL OBSERVATION\nRun Name: {run_name}\nSeed: {self.seed}\n")
                            f.write(f"Partial Observation Config: {self.partial_observation_config}\n" + "="*50 + "\n")
                        elif debug_file_already_existed and not is_empty:
                            f.write(f"\n\n======== New Log Session for Run: {run_name}, Seed: {self.seed} ({condition_mode_str}) - PARTIAL OBSERVATION ========\n")

                    # Create new metrics log file for this condition mode
                    metrics_log_file_path_current = os.path.join(episode_condition_output_dir, f'metrics_log_test_{condition_mode_str}.txt')
                    metrics_file_already_existed = os.path.exists(metrics_log_file_path_current)
                    with open(metrics_log_file_path_current, 'a') as m_f:
                        is_empty = os.fstat(m_f.fileno()).st_size == 0
                        if not metrics_file_already_existed or is_empty:
                            m_f.write(f"Metrics log for test evaluation ({condition_mode_str}) - PARTIAL OBSERVATION\nRun Name: {run_name}\nSeed: {self.seed}\n")
                            m_f.write(f"Partial Observation Config: {self.partial_observation_config}\n" + "="*50 + "\n")
                        elif metrics_file_already_existed and not is_empty:
                            m_f.write(f"\n\n======== New Log Session for Run: {run_name}, Seed: {self.seed} ({condition_mode_str}) - PARTIAL OBSERVATION ========\n")

                    # GIF generation setup (temp dir specific to this episode/condition)
                    temp_frames_dir = os.path.join(episode_condition_output_dir, "temp_gif_frames")
                    os.makedirs(temp_frames_dir, exist_ok=True)
                    episode_gif_frame_paths = []
                    frame_counter_gif = 0
                    # Lists to store segments for cumulative GIF visualization
                    accumulated_pred_segments_for_gif = []

                    # --- Process batch data and define key trajectory variables ---
                    time_start_batch = time.time()
                    # Start timing for end-to-end latency
                    time_start_episode_processing = time.time()
                    
                    batch = dict_apply(data, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                    gt_trajectory = batch['full_trajectory'] # Shape [1, T, D]
                    traj_length = gt_trajectory.shape[1]
                    D = gt_trajectory.shape[-1]
                    full_pred = torch.zeros((1, traj_length, D), device=device)

                    # Initialize timing collections for this episode
                    episode_inference_times = []
                    episode_preprocessing_times = []
                    episode_postprocessing_times = []

                    # Get full GT for static display in GIF and calculate bounds once
                    full_gt_episode_tensor_for_gif = data['full_trajectory']
                    overall_gt_bounds_for_gif = None
                    try:
                        gt_points_for_bounds_calc = SprayDiffusionRunner._extract_xyz_anim_static(full_gt_episode_tensor_for_gif.cpu().numpy()[0])
                        if gt_points_for_bounds_calc.shape[0] > 0:
                            # Filter -100 before calculating bounds
                            valid_gt_mask_for_bounds = ~np.all(gt_points_for_bounds_calc == -100.0, axis=1)
                            valid_gt_points_for_bounds = gt_points_for_bounds_calc[valid_gt_mask_for_bounds]
                            if valid_gt_points_for_bounds.shape[0] > 0:
                                # Adjust buffer based on paper_vis_mode
                                buffer = 0.01 if paper_vis_mode else 0.1
                                xlim = (valid_gt_points_for_bounds[:,0].min() - buffer, valid_gt_points_for_bounds[:,0].max() + buffer)
                                ylim = (valid_gt_points_for_bounds[:,1].min() - buffer, valid_gt_points_for_bounds[:,1].max() + buffer)
                                zlim = (valid_gt_points_for_bounds[:,2].min() - buffer, valid_gt_points_for_bounds[:,2].max() + buffer)
                                overall_gt_bounds_for_gif = (xlim, ylim, zlim)
                    except Exception as e_bounds:
                        cprint(f"Warning: Could not calculate overall_gt_bounds for GIF: {e_bounds}", "yellow")
                    if overall_gt_bounds_for_gif is None:
                        overall_gt_bounds_for_gif = ((-1,1), (-1,1), (-1,1))

                    # Create a temporary directory for local spray GIF frames if in paper_vis_mode
                    local_spray_temp_frames_dir = None
                    if paper_vis_mode:
                        local_spray_temp_frames_dir = os.path.join(episode_condition_output_dir, 'temp_local_spray_gif_frames')
                        os.makedirs(local_spray_temp_frames_dir, exist_ok=True)
                    local_spray_gif_frame_paths = []

                    # --- Initialize policy, loop variables, and initial observation --- 
                    obs_pc = batch.get('obs', {}).get('point_cloud', None)
                    policy.reset()
                    done = 0
                    res = {}

                    # Get the first ground truth point as the initial input for the policy
                    start_pred_traj = batch['full_trajectory'][:,0,:]
                    assert (start_pred_traj == -100).any().item() == False, "Initial trajectory has -100 padding"

                    obs_dict = {
                        'prev_true_trajectory': start_pred_traj,
                        'obs': {'point_cloud': obs_pc}  # This now contains the masked point cloud
                    }

                    time_start_pred_loop = time.time()
                    with open(current_debug_log_path, 'a') as f:
                        f.write(f"\nEpisode {episode_idx} ({condition_mode_str}) - PARTIAL OBSERVATION:\n")
                        f.write(f"  Starting prediction loop with masked point cloud...\n")
                        if self.partial_observation_config['enabled']:
                            f.write(f"  Partial observation method: {self.partial_observation_config['face_selection_method']}\n")
                            f.write(f"  Face ratio: {self.partial_observation_config['face_ratio']}\n")

                    #-------- start episode
                    iteration_count = 0
                    iteration_total_times = []
                    while done < traj_length:
                        iteration_start_time = time.time()
                        iteration_count += 1
                        
                        # Calculate and display progress information
                        progress_percentage = (done / traj_length) * 100
                        
                        # Estimate remaining time based on average iteration time
                        if iteration_total_times:
                            avg_iteration_time = np.mean(iteration_total_times)
                            remaining_steps = traj_length - done
                            estimated_horizon_size = policy.n_action_steps if hasattr(policy, 'n_action_steps') else 16
                            estimated_remaining_iterations = max(1, remaining_steps // estimated_horizon_size)
                            estimated_remaining_time = avg_iteration_time * estimated_remaining_iterations
                            time_info = f"Avg: {avg_iteration_time:.2f}s/iter, Est. remaining: {estimated_remaining_time:.1f}s"
                        else:
                            time_info = "Calculating..."
                        
                        # Print progress to console
                        print(f"    [PartialObs] Iteration {iteration_count}, Step {done}/{traj_length} ({progress_percentage:.1f}%) - {time_info}")
                        
                        # Log current state if needed for debugging
                        with open(current_debug_log_path, 'a') as f:
                            f.write(f"  Iteration {iteration_count}, Prediction step {done}/{traj_length} ({progress_percentage:.1f}%)\n")
                            if iteration_total_times:
                                f.write(f"    Progress timing - {time_info}\n")
                        
                        # Start timing for preprocessing
                        time_start_preprocessing = time.time()
                        
                        # Any preprocessing steps here (currently minimal)
                        # obs_dict is already prepared before the loop and contains masked point cloud
                        
                        time_end_preprocessing = time.time()
                        preprocessing_time = (time_end_preprocessing - time_start_preprocessing) * 1000
                        episode_preprocessing_times.append(preprocessing_time)
                        
                        # Start timing for pure model inference
                        time_start_inference = time.time()
                        
                        # policy.predict_action now returns denoising steps as well
                        prediction_result = policy.predict_action(obs_dict)
                        
                        time_end_inference = time.time()
                        inference_time = (time_end_inference - time_start_inference) * 1000
                        episode_inference_times.append(inference_time)
                        
                        # Start timing for postprocessing
                        time_start_postprocessing = time.time()
                        
                        res = prediction_result
                        denoising_steps_for_full_horizon = prediction_result.get('denoising_trajectory_steps')
                        
                        # Assuming policy returns prediction in 'action', adjust if it's 'action_pred'
                        seg = res['action'] 
                        pred_horizon_len = seg.shape[1]
                        num_steps_to_fill = min(pred_horizon_len, traj_length - done)            
                        
                        if num_steps_to_fill <= 0: 
                            break

                        # Segments for evaluation and visualization
                        current_pred_segment = seg[:, :num_steps_to_fill]
                        current_gt_segment = gt_trajectory[:, done : done + num_steps_to_fill]
                        
                        time_end_postprocessing = time.time()
                        postprocessing_time = (time_end_postprocessing - time_start_postprocessing) * 1000
                        episode_postprocessing_times.append(postprocessing_time)
                        
                        # Log timing information
                        with open(current_debug_log_path, 'a') as f:
                            f.write(f"    Timing (ms) - Preprocessing: {preprocessing_time:.2f}, Inference: {inference_time:.2f}, Postprocessing: {postprocessing_time:.2f}\n")

                        # Calculate and log per-horizon MSE
                        if current_pred_segment.shape[1] > 0:
                            mse_horizon = F.mse_loss(current_pred_segment, current_gt_segment).item()
                            with open(current_debug_log_path, 'a') as f:
                                f.write(f"    Horizon Prediction (sim steps {done} to {done + num_steps_to_fill -1}):\n")
                                f.write(f"      MSE: {mse_horizon:.4f}\n")

                            # Accumulate segments for GIF
                            accumulated_pred_segments_for_gif.append(current_pred_segment[0])
                            frame_counter_gif += 1

                        full_pred[:, done : done + num_steps_to_fill] = current_pred_segment

                        # Update done counter and obs_dict for next iteration
                        done += num_steps_to_fill
                        
                        # Update obs_dict with the latest predicted trajectory segment for next iteration
                        if done < traj_length:
                            # Use the last predicted point as the new starting point
                            obs_dict['prev_true_trajectory'] = current_pred_segment[:, -1:, :]  # [1, 1, D]
                            # Keep the same masked point cloud observation
                            # obs_dict['obs']['point_cloud'] remains the same (masked)
                        
                        # Record total iteration time
                        iteration_end_time = time.time()
                        iteration_total_time = iteration_end_time - iteration_start_time
                        iteration_total_times.append(iteration_total_time)

                    # End of episode prediction loop
                    time_end_pred_loop = time.time()
                    total_pred_loop_time = time_end_pred_loop - time_start_pred_loop

                    with open(current_debug_log_path, 'a') as f:
                        f.write(f"  Prediction loop completed in {total_pred_loop_time:.2f}s\n")
                        f.write(f"  Total iterations: {iteration_count}\n")
                        if iteration_total_times:
                            f.write(f"  Average iteration time: {np.mean(iteration_total_times):.2f}s\n")

                    # Calculate episode metrics using the metrics handler
                    episode_metrics = {}
                    try:
                        # Convert tensors to numpy for metrics calculation
                        pred_traj_np = full_pred.cpu().numpy()[0]  # [T, D]
                        gt_traj_np = gt_trajectory.cpu().numpy()[0]  # [T, D]
                        
                        # Calculate metrics
                        episode_metrics = metrics_handler.calculate_metrics(
                            pred_traj_np, gt_traj_np, 
                            inference_times=episode_inference_times,
                            preprocessing_times=episode_preprocessing_times,
                            postprocessing_times=episode_postprocessing_times
                        )
                        
                        # Log metrics
                        with open(metrics_log_file_path_current, 'a') as m_f:
                            m_f.write(f"\nEpisode {episode_idx} Metrics:\n")
                            for metric_name, metric_value in episode_metrics.items():
                                m_f.write(f"  {metric_name}: {metric_value}\n")
                        
                        # Add to collection based on condition mode
                        if condition_mode_str == "Pred_Cond":
                            all_metrics_pred_cond.append(episode_metrics)
                        else:
                            all_metrics_gt_cond.append(episode_metrics)
                            
                    except Exception as e_metrics:
                        cprint(f"Warning: Failed to calculate metrics for episode {episode_idx}: {e_metrics}", "yellow")
                        with open(current_debug_log_path, 'a') as f:
                            f.write(f"  Error calculating metrics: {e_metrics}\n")

                    # Visualization (using parent class methods)
                    try:
                        # Save trajectory visualization
                        traj_vis_path = os.path.join(episode_condition_output_dir, f'trajectory_comparison_ep_{episode_idx}.png')
                        self.visualize_trajectory_runner_points(
                            full_pred.cpu().numpy(), gt_trajectory.cpu().numpy(),
                            traj_vis_path, step=episode_idx, metrics=episode_metrics,
                            mesh_vertices=mesh_vertices_np, mesh_faces=mesh_faces_np,
                            paper_vis_mode=paper_vis_mode
                        )
                        
                        # Save input point cloud visualization (this will show the masked point cloud)
                        if obs_pc is not None:
                            pc_vis_path = os.path.join(episode_condition_output_dir, f'input_point_cloud_ep_{episode_idx}.png')
                            obs_pc_np = obs_pc.cpu().numpy()[0] if isinstance(obs_pc, torch.Tensor) else obs_pc[0]
                            self.visualize_input_point_cloud(
                                obs_pc_np, pc_vis_path,
                                mesh_vertices=mesh_vertices_np, mesh_faces=mesh_faces_np
                            )
                            
                    except Exception as e_vis:
                        cprint(f"Warning: Failed to create visualizations for episode {episode_idx}: {e_vis}", "yellow")
                        with open(current_debug_log_path, 'a') as f:
                            f.write(f"  Error creating visualizations: {e_vis}\n")

        # Aggregate results
        final_results = {}
        if all_metrics_pred_cond:
            pred_cond_results = self.aggregate_results(all_metrics_pred_cond, metrics_handler, prefix="pred_cond_")
            final_results.update(pred_cond_results)
            
        if all_metrics_gt_cond:
            gt_cond_results = self.aggregate_results(all_metrics_gt_cond, metrics_handler, prefix="gt_cond_")
            final_results.update(gt_cond_results)

        # Log final aggregated results
        cprint(f"\n=== Final Aggregated Results (Partial Observation) ===", "green")
        for key, value in final_results.items():
            cprint(f"{key}: {value}", "green")

        return final_results 