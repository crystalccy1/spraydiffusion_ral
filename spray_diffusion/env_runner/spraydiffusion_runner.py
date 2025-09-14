import wandb
import os
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
from utils.config import load_config, load_config_json

from spray_diffusion.policy.base_policy import BasePolicy
from spray_diffusion.common.pytorch_util import dict_apply
from spray_diffusion.env_runner.base_runner import BaseRunner
import spray_diffusion.common.logger_util as logger_util
from utils.visualize import visualize_mesh_traj_animated
from termcolor import cprint

# Import the LossHandler class and related functions
from loss_spraydiffusion_handler import SprayDiffusionLossHandler
from metrics_handler_spraydiffusion import MetricsHandler
# Import the newly created utility functions
from spray_diffusion.env_runner.utils_spraydiffusion import visualize_traj, _save_trajectory_frame

class SprayDiffusionRunner(BaseRunner):
    """
    Runner for SprayDiffusion model evaluation on dataset.
    Processes one episode at a time with batch size of 1.
    """
    def __init__(self,
                 model=None,
                 device=None,
                 output_dir=None,
                 run_name='default_run',
                 eval_episodes=10,
                 tqdm_interval_sec=1.0,
                 chamfer_distance_threshold=0.05,
                 batch_size=1,  # Always use batch size of 1
                 num_workers=4,
                 seed=42,
                 paper_vis_mode=False,
                 visualize_denoising_gifs=False,
                 ablation_prev_traj='normal',
                 random_traj_seed=123):
        super().__init__(
            model=model,
            device=device,
            output_dir=output_dir,
            run_name=run_name,
            eval_episodes=eval_episodes,
            tqdm_interval_sec=tqdm_interval_sec,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed
        )
        
        self.chamfer_distance_threshold = chamfer_distance_threshold
        self.paper_vis_mode = paper_vis_mode
        self.visualize_denoising_gifs = visualize_denoising_gifs
        self.ablation_prev_traj = ablation_prev_traj
        self.random_traj_seed = random_traj_seed
        
        # Initialize metrics handler
        self.metrics_handler = MetricsHandler()
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run(self, policy, dataloader, split='test', run_name=None, dataset_name=None, paper_vis_mode=False):
        """
        Run evaluation on the provided dataloader.
        
        Args:
            policy: The trained policy model
            dataloader: DataLoader containing episodes
            split: Dataset split name
            run_name: Name for this run
            dataset_name: Name of the dataset
            paper_vis_mode: Whether to use paper visualization mode
            
        Returns:
            dict: Aggregated results
        """
        if run_name is not None:
            self.run_name = run_name
        if paper_vis_mode:
            self.paper_vis_mode = paper_vis_mode
            
        # Create output directories
        self._create_output_dirs()
        
        # Initialize results storage
        all_results = {'pred_cond': [], 'gt_cond': []}
        
        # Process episodes
        episode_count = 0
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Processing episodes")):
            if episode_count >= self.eval_episodes:
                break
                
            try:
                # Process single episode
                episode_results = self._process_episode(
                    batch, episode_count, policy, dataset_name
                )
                
                # Store results
                if episode_results:
                    for key in ['pred_cond', 'gt_cond']:
                        if key in episode_results:
                            all_results[key].append(episode_results[key])
                            
                episode_count += 1
                
            except Exception as e:
                cprint(f"Error processing episode {episode_count}: {e}", "red")
                continue
        
        # Aggregate results
        aggregated_results = {}
        for key in ['pred_cond', 'gt_cond']:
            if all_results[key]:
                aggregated_results[key] = self.aggregate_results(
                    all_results[key], self.metrics_handler, prefix=f"{key}_"
                )
        
        return aggregated_results

    def _create_output_dirs(self):
        """Create necessary output directories."""
        self.demo_dir = os.path.join(self.output_dir, 'demo', self.run_name)
        os.makedirs(self.demo_dir, exist_ok=True)
        
        if self.paper_vis_mode:
            self.paper_vis_dir = os.path.join(self.output_dir, 'paper_vis', 
                                            getattr(self, 'dataset_name', 'unknown'), 
                                            self.run_name)
            os.makedirs(self.paper_vis_dir, exist_ok=True)

    def _process_episode(self, batch, episode_idx, policy, dataset_name):
        """
        Process a single episode and generate visualizations.
        
        Args:
            batch: Episode data batch
            episode_idx: Episode index
            policy: The policy model
            dataset_name: Name of the dataset
            
        Returns:
            dict: Episode results
        """
        # Extract data from batch
        obs = batch['obs']
        gt_trajectory = batch['full_trajectory']
        mesh_vertices = batch.get('mesh_vertices', None)
        mesh_faces = batch.get('mesh_faces', None)
        
        # Convert to numpy for visualization
        mesh_vertices_np = mesh_vertices[0].cpu().numpy() if mesh_vertices is not None else None
        mesh_faces_np = mesh_faces[0].cpu().numpy() if mesh_faces is not None else None
        
        # Run policy inference
        with torch.no_grad():
            policy_output = policy.predict_action(obs)
            full_pred = policy_output['action']
        
        # Create episode output directory
        episode_condition_output_dir = os.path.join(self.demo_dir, f"episode_{episode_idx:03d}_Pred_Cond")
        os.makedirs(episode_condition_output_dir, exist_ok=True)
        
        # Generate visualizations using the specified functions
        self._generate_visualizations(
            full_pred, gt_trajectory, episode_condition_output_dir, 
            episode_idx, mesh_vertices_np, mesh_faces_np, dataset_name
        )
        
        # Compute metrics
        episode_metrics = self.metrics_handler.compute(
            pred_trajectory=full_pred,
            gt_trajectory=gt_trajectory,
            mesh_vertices=mesh_vertices_np,
            mesh_faces=mesh_faces_np
        )
        
        return {'pred_cond': episode_metrics}

    def _generate_visualizations(self, full_pred, gt_trajectory, output_dir, 
                               episode_idx, mesh_vertices_np, mesh_faces_np, dataset_name):
        """
        Generate visualizations using the specified functions.
        
        Args:
            full_pred: Predicted trajectory
            gt_trajectory: Ground truth trajectory
            output_dir: Output directory for visualizations
            episode_idx: Episode index
            mesh_vertices_np: Mesh vertices as numpy array
            mesh_faces_np: Mesh faces as numpy array
            dataset_name: Name of the dataset
        """
        condition_mode_str = "Pred_Cond"
        run_name = getattr(self, 'run_name', 'unknown_checkpoint')
        
        # Load config for visualization
        config = load_config_json('configs/spraydiffusion/default.yaml')
        
        from utils.visualize import create_multiview_mesh_trajectory_visualization
        
        # Generate predicted trajectory multiview visualization (L700-712)
        vis_path_static_pred = os.path.join(output_dir, f'trajectory_ep{episode_idx}_{condition_mode_str}_pred.png')
        create_multiview_mesh_trajectory_visualization(
            mesh_vertices_np=mesh_vertices_np,
            mesh_faces_np=mesh_faces_np,
            traj=full_pred[0],
            config=config,
            save_path=vis_path_static_pred,
            point_size=14.0,
            trajc='#7596F3',
            offset=0.2,
            dataset_name=dataset_name,
            checkpoint_name=run_name,
            title=f"Predicted Trajectory - Episode {episode_idx}"
        )
        
        # Generate ground truth trajectory multiview visualization (L715-728)
        vis_path_static_gt = os.path.join(output_dir, f'trajectory_ep{episode_idx}_{condition_mode_str}_gt.png')
        create_multiview_mesh_trajectory_visualization(
            mesh_vertices_np=mesh_vertices_np,
            mesh_faces_np=mesh_faces_np,
            traj=gt_trajectory[0],
            config=config,
            save_path=vis_path_static_gt,
            point_size=14.0,
            trajc='#F37592',
            offset=0.2,
            dataset_name=dataset_name,
            checkpoint_name=run_name,
            title=f"Ground Truth Trajectory - Episode {episode_idx}"
        )
        
        # Generate paper visualization PNGs if in paper mode (L1439-1548)
        if self.paper_vis_mode:
            base_save_path_png = os.path.join(self.paper_vis_dir, f'episode_{episode_idx:03d}')
            self.save_paper_pngs(
                pred_traj=full_pred,
                gt_traj=gt_trajectory,
                base_save_path=base_save_path_png,
                mesh_vertices=mesh_vertices_np,
                mesh_faces=mesh_faces_np
            )

    @staticmethod
    def _extract_xyz_anim_static(traj_segment): # [current_T, D] -> [current_T*4, 3]
        """Helper to extract XYZ points from a trajectory segment for animation frames."""
        if traj_segment.shape[0] == 0: return np.empty((0,3))
        xyz_points_loc = []
        dims_per_point = 6; num_points_to_vis = 4
        for i in range(num_points_to_vis):
            if (i * dims_per_point + 2) < traj_segment.shape[1]:
                x = traj_segment[:, i * dims_per_point + 0]
                y = traj_segment[:, i * dims_per_point + 1]
                z = traj_segment[:, i * dims_per_point + 2]
                pts = np.stack([x, y, z], axis=1)
                xyz_points_loc.append(pts)
            else: break
        if not xyz_points_loc: return np.empty((0,3))
        return np.concatenate(xyz_points_loc, axis=0)

    def _plot_trajectory_lines_and_points(self, ax, traj_np, color_points, color_lines, marker, point_size=10, alpha=0.8, line_alpha=0.5, line_width=0.5, paper_vis_mode=False):
        """
        Plots points only for each of the 4 sub-trajectories (no connecting lines).
        Args:
            ax: Matplotlib 3D axis.
            traj_np: Trajectory numpy array [T, 24].
            color_points: Color for the scatter points.
            color_lines: Color for the connecting lines or arrows (not used anymore).
            marker: Marker style for scatter points.
            point_size: Size of the scatter points.
            alpha: Alpha for scatter points.
            line_alpha: Alpha for lines/arrows (not used anymore).
            line_width: Width of the lines (not used anymore).
            paper_vis_mode: If True, connecting arrows are plotted instead of lines (not used anymore).
        """
        T, D = traj_np.shape
        if T == 0: return # Nothing to plot

        for point_idx in range(4):
            start_dim = point_idx * 6
            if start_dim + 2 >= D: continue # Check if dimension exists

            # Extract XYZ for this specific sub-trajectory
            sub_traj_xyz = traj_np[:, start_dim : start_dim + 3]

            # Filter padding (-100)
            valid_mask = ~np.all(sub_traj_xyz == -100.0, axis=1)
            valid_points = sub_traj_xyz[valid_mask]

            if valid_points.shape[0] > 0:
                # Plot points only (no lines or arrows)
                ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
                           c=color_points, marker=marker, s=point_size, alpha=alpha, depthshade=False)

    def save_paper_pngs(self, pred_traj, gt_traj, base_save_path, mesh_vertices=None, mesh_faces=None):
        """
        Generates three specific static PNG images required for paper visualization mode:
        1. Prediction + Mesh
        2. Ground Truth + Mesh
        3. Prediction + Ground Truth + Mesh
        Args:
            pred_traj: Predicted trajectory tensor [B, T, 24]
            gt_traj: Ground truth trajectory tensor [B, T, 24]
            base_save_path: Base path for saving files (e.g., 'trajectory_ep0_Pred_Cond')
            mesh_vertices: Numpy array of mesh vertices [N, 3]
            mesh_faces: Numpy array of mesh faces [M, 3]
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # Ensure Axes3D is imported

        pred_traj_np = pred_traj.detach().cpu().numpy()[0]  # [T, 24]
        gt_traj_np = gt_traj.detach().cpu().numpy()[0]     # [T, 24]

        # --- Plot Function --- 
        def create_plot(save_suffix, plot_pred=False, plot_gt=False):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot mesh (consistent style)
            if mesh_vertices is not None and mesh_faces is not None and mesh_vertices.shape[0] > 0 and mesh_faces.shape[0] > 0:
                ax.plot_trisurf(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2],
                                triangles=mesh_faces, color='lightgrey', alpha=0.3)

            # Plot Trajectories using the helper function, passing paper_vis_mode
            if plot_pred:
                self._plot_trajectory_lines_and_points(ax, pred_traj_np, color_points='green', color_lines='palegreen', marker='o', paper_vis_mode=True)
            if plot_gt:
                self._plot_trajectory_lines_and_points(ax, gt_traj_np, color_points='red', color_lines='lightcoral', marker='o', paper_vis_mode=True)

            # Formatting
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            # No title, no legend, no metrics text

            # Set consistent axis limits based on combined GT and Pred (if available)
            all_points_for_limits = []
            if plot_pred:
                pred_xyz_all = np.concatenate([pred_traj_np[:, i*6:i*6+3] for i in range(4) if i*6+2 < pred_traj_np.shape[1]], axis=0)
                pred_valid_mask = ~np.all(pred_xyz_all == -100.0, axis=1)
                if np.any(pred_valid_mask): all_points_for_limits.append(pred_xyz_all[pred_valid_mask])
            if plot_gt:
                gt_xyz_all = np.concatenate([gt_traj_np[:, i*6:i*6+3] for i in range(4) if i*6+2 < gt_traj_np.shape[1]], axis=0)
                gt_valid_mask = ~np.all(gt_xyz_all == -100.0, axis=1)
                if np.any(gt_valid_mask): all_points_for_limits.append(gt_xyz_all[gt_valid_mask])

            if all_points_for_limits:
                combined_pts = np.concatenate(all_points_for_limits, axis=0)
                # Use a very small buffer for paper mode
                buffer = 0.01
                ax.set_xlim(combined_pts[:, 0].min() - buffer, combined_pts[:, 0].max() + buffer)
                ax.set_ylim(combined_pts[:, 1].min() - buffer, combined_pts[:, 1].max() + buffer)
                ax.set_zlim(combined_pts[:, 2].min() - buffer, combined_pts[:, 2].max() + buffer)
            else:
                ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-0.5, 0.5]); ax.set_zlim([0, 1.0]) # Fallback

            # Turn off axes for paper mode PNGs
            ax.axis('off')

            # Save
            full_save_path = f"{base_save_path}_{save_suffix}.png"
            # Use bbox_inches='tight' to remove extra whitespace
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved paper static image: {full_save_path}")
            plt.close(fig)

        # --- Generate the 3 PNGs ---
        create_plot(save_suffix="pred_mesh", plot_pred=True, plot_gt=False)
        create_plot(save_suffix="gt_mesh", plot_pred=False, plot_gt=True)
        create_plot(save_suffix="pred_gt_mesh", plot_pred=True, plot_gt=True)

    def aggregate_results(self, all_results, metrics_handler, prefix=""):
        """Aggregate results from multiple episodes."""
        if not all_results:
            print(f"Warning: No results to aggregate for prefix {prefix}.")
            num_metrics = metrics_handler.tot_num_of_metrics()
            
            # Create default named metrics based on all configured metrics
            default_aggregated_data = {'mean_metrics': np.full(num_metrics, np.nan)}
            value_idx = 0
            # metrics_handler.metrics should contain internal names like ['pcd', 'smoothness', 'coverage']
            for metric_internal_name in metrics_handler.metrics:
                # metrics_handler.output_metrics_names is a list indexed by metrics_handler.metric_index[metric_internal_name]
                # Each entry is a tuple, e.g., ('point-wise chamfer distance',)
                metric_display_names_tuple = metrics_handler.output_metrics_names[metrics_handler.metric_index[metric_internal_name]]
                for display_name in metric_display_names_tuple:
                    # Clean display name for key names (e.g., for wandb)
                    safe_display_name = display_name.replace(" ", "_").replace("%", "perc")
                    default_aggregated_data[f'{prefix}mean_{safe_display_name}'] = float('nan')
                    value_idx += 1
            return default_aggregated_data

        try:
            results_array = np.array(all_results, dtype=float)
            # mean_metrics should now contain the average of all metrics, in the same order as metrics_handler.compute returns
            mean_metrics = np.nanmean(results_array, axis=0) 
        except Exception as e:
            print(f"Error aggregating {prefix}: {e}. Returning NaN.")
            num_metrics = metrics_handler.tot_num_of_metrics()
            error_aggregated_data = {'mean_metrics': np.full(num_metrics, np.nan)}
            value_idx = 0
            for metric_internal_name in metrics_handler.metrics:
                metric_display_names_tuple = metrics_handler.output_metrics_names[metrics_handler.metric_index[metric_internal_name]]
                for display_name in metric_display_names_tuple:
                    safe_display_name = display_name.replace(" ", "_").replace("%", "perc")
                    error_aggregated_data[f'{prefix}mean_{safe_display_name}'] = float('nan')
                    value_idx += 1
            return error_aggregated_data

        # Create aggregated data dictionary
        aggregated_data = {'mean_metrics': mean_metrics}
        value_idx = 0
        
        for metric_internal_name in metrics_handler.metrics:
            metric_display_names_tuple = metrics_handler.output_metrics_names[metrics_handler.metric_index[metric_internal_name]]
            for display_name in metric_display_names_tuple:
                # Clean display name for key names (e.g., for wandb)
                safe_display_name = display_name.replace(" ", "_").replace("%", "perc")
                aggregated_data[f'{prefix}mean_{safe_display_name}'] = float(mean_metrics[value_idx])
                value_idx += 1

        return aggregated_data
