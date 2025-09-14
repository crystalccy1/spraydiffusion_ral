import os
import argparse
import torch
import numpy as np
import random
import dill # For loading pickled parts of checkpoint
import pathlib
from termcolor import cprint
from omegaconf import OmegaConf

# Project utilities
from utils import get_random_string, create_dirs
from utils.args import pformat_dict # Assuming load_args is not needed if cfg comes from checkpoint
from utils.disk import get_dataset_paths, get_output_dir # May need for dataset paths or output structure
from utils.dataset.spraydiffusion_rollout_dataset import SprayDiffusionRolloutDataset
from utils.dataset.spraydiffusion_new_objects_dataset import SprayDiffusionNewObjectsDataset
from spray_diffusion.env_runner.spraydiffusion_runner import SprayDiffusionRunner
from utils.visualize import visualize_mesh_traj_animated
from utils.visualize import visualize_mesh_coverage_pyvista
from models import get_model_diffusion # To instantiate the model

# Import point cloud converter utilities
# from utils.point_cloud_converter import save_as_npy, save_as_ply

# def save_test_point_clouds(rollout_dataset, output_dir, max_samples=10):
#     """
#     Save point clouds from the rollout dataset as NPY and PLY files.
    
#     Args:
#         rollout_dataset: The dataset to extract point clouds from
#         output_dir (str): Base output directory
#         max_samples (int): Maximum number of samples to save
#     """
#     pc_save_dir = os.path.join(output_dir, 'test_point_clouds')
#     os.makedirs(pc_save_dir, exist_ok=True)
    
#     cprint(f"Saving point clouds to: {pc_save_dir}", "cyan")
    
#     saved_count = 0
#     for idx, data in enumerate(rollout_dataset):
#         if saved_count >= max_samples:
#             break
            
#         try:
#             # Extract point cloud from observation
#             if isinstance(data, dict) and 'obs' in data:
#                 obs = data['obs']
#                 if isinstance(obs, dict) and 'point_cloud' in obs:
#                     point_cloud = obs['point_cloud']
#                 elif hasattr(obs, 'point_cloud'):
#                     point_cloud = obs.point_cloud
#                 else:
#                     continue
#             else:
#                 continue
            
#             # Convert to numpy if needed
#             if isinstance(point_cloud, torch.Tensor):
#                 point_cloud_np = point_cloud.detach().cpu().numpy()
#             else:
#                 point_cloud_np = np.array(point_cloud)
            
#             # Handle different dimensions
#             if point_cloud_np.ndim == 3:  # (B, N, 3)
#                 point_cloud_np = point_cloud_np[0]  # Take first batch
#             elif point_cloud_np.ndim == 1:  # Flattened
#                 if len(point_cloud_np) % 3 == 0:
#                     point_cloud_np = point_cloud_np.reshape(-1, 3)
#                 else:
#                     continue
            
#             # Ensure we have valid point cloud data
#             if point_cloud_np.ndim != 2 or point_cloud_np.shape[1] < 3:
#                 continue
            
#             # Take only XYZ coordinates
#             point_cloud_np = point_cloud_np[:, :3].astype(np.float32)
            
#             # Save as NPY and PLY
#             base_name = f"test_point_cloud_{idx:03d}"
            
#             npy_path = os.path.join(pc_save_dir, f"{base_name}.npy")
#             save_as_npy(point_cloud_np, npy_path)
            
#             ply_path = os.path.join(pc_save_dir, f"{base_name}.ply")
#             save_as_ply(point_cloud_np, ply_path)
            
#             saved_count += 1
            
#             if saved_count % 5 == 0:
#                 cprint(f"Saved {saved_count}/{max_samples} point clouds", "green")
                
#         except Exception as e:
#             cprint(f"Error saving point cloud {idx}: {e}", "yellow")
#             continue
    
#     cprint(f"Successfully saved {saved_count} point clouds", "green")
#     return saved_count

def main():
    parser = argparse.ArgumentParser(description='Test SprayDiffusion model with rollout evaluation.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the trained model checkpoint (.ckpt file).')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='A name for this test run. If None, generated from checkpoint name.')
    parser.add_argument('--output_dir_base', type=str, default='outputs',
                        help='Base directory for saving test run outputs (e.g., demo files).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of Dataloader workers.')
    parser.add_argument('--dataset_name', type=str, default=None, 
                        help='Name of the dataset to use for testing (e.g., window-v2). Overrides dataset in checkpoint cfg if provided.')
    parser.add_argument('--dataset_split', type=str, default='test',
                        help='Dataset split to use for evaluation (e.g., test, overfit).')
    parser.add_argument('--paper_vis_mode', action='store_true', default=False,
                        help='Enable paper visualization mode (saves mesh+traj to paper_vis/dataset/run_name).')
    parser.add_argument('--visualize_denoising_gifs', action='store_true', default=False,
                        help='Enable saving of GIFs for the denoising process visualization.')
    parser.add_argument('--use_new_objects', action='store_true', default=False,
                        help='Use new preprocessed objects from spray_diffusion/data_objects instead of original dataset.')
    parser.add_argument('--new_objects_categories', nargs='+', default=None,
                        help='Specific categories to use from new objects (e.g., Chair Table Desk). If not specified, uses all available.')
    parser.add_argument('--new_objects_root', type=str, default="/usr/stud/dira/ccy/MaskPlanner/spray_diffusion/data_objects",
                        help='Root directory for new preprocessed objects.')
    parser.add_argument('--ablation_prev_traj', type=str, default='normal', 
                        choices=['normal', 'remove', 'random', 'zeros', 'gaussian_noise', 'temporal_only'],
                        help='Ablation study for prev_true_trajectory: normal (use real), remove (point cloud only), random (use random trajectory), zeros (use fixed zero vectors), gaussian_noise (pure Gaussian noise), temporal_only (preserve temporal structure but remove geometry)')
    parser.add_argument('--random_traj_seed', type=int, default=123,
                        help='Random seed for generating random trajectories in ablation study')
    parser.add_argument('--save_point_clouds', action='store_true', default=False,
                        help='Save input point clouds as NPY and PLY files for visualization')
    parser.add_argument('--max_point_cloud_samples', type=int, default=10,
                        help='Maximum number of point cloud samples to save')
    parser.add_argument('--save_coverage_vis', action='store_true', default=False,
                        help='Save per-episode GT coverage visualization using PyVista (yellow=covered, original=base color).')

    args = parser.parse_args()

    # Setup run name and output directory
    if args.run_name is None:
        checkpoint_basename = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        args.run_name = f"test_{checkpoint_basename}_{get_random_string(4)}"
    
    # The env_runner will create demo/run_name, so we just need the base output dir for the runner
    # if needed for other things. For now, runner handles its own output within demo/
    test_output_dir_for_runner = args.output_dir_base # This path isn't directly used much, runner makes demo/run_name

    cprint(f"Starting test run: {args.run_name}", "green")
    cprint(f"Loading checkpoint from: {args.checkpoint_path}", "yellow")
    # if args.enable_opaque_mesh_vis:
    #     cprint("High-quality opaque mesh visualization: ENABLED", "cyan")
    # else:
    #     cprint("High-quality opaque mesh visualization: DISABLED", "cyan")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load checkpoint payload
    checkpoint_path = pathlib.Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        cprint(f"Error: Checkpoint file not found at {args.checkpoint_path}", "red")
        return

    try:
        payload = torch.load(checkpoint_path.open('rb'), pickle_module=dill, map_location='cpu')
    except Exception as e:
        cprint(f"Error loading checkpoint: {e}", "red")
        return

    # Load config from checkpoint
    if 'cfg' not in payload:
        cprint("Error: 'cfg' not found in checkpoint payload.", "red")
        return
    cfg = payload['cfg']

    # --- Potentially override dataset config from CLI ---
    if args.dataset_name:
        cprint(f"Overriding dataset from checkpoint with CLI arg: {args.dataset_name}", "cyan")
        # Assuming cfg.dataset is a list or a string. Adjust if it's structured differently.
        if isinstance(cfg.dataset, list):
            cfg.dataset[0] = args.dataset_name 
        else:
            cfg.dataset = args.dataset_name
    # Update other relevant parts if dataset name changes, e.g., task_name or specific paths if hardcoded based on dataset.
    # For now, assuming get_dataset_paths will handle the new name.

    # Instantiate model using the config from the checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model_diffusion(
        config=cfg,
        which=cfg.model.backbone, # Ensure this path is correct in your cfg
        io_type=cfg.task_name,    # Ensure this path is correct
        device=device
    )

    # Load model state dict
    if 'model' not in payload['state_dicts']:
        cprint("Error: Model state_dict not found in checkpoint.", "red")
        return
    model.load_state_dict(payload['state_dicts']['model'])
    cprint("Model state_dict loaded successfully.", "green")

    # Load normalizer by creating an instance and setting its state, then applying to model
    if 'normalizer' in payload['state_dicts']:
        from spray_diffusion.model.common.normalizer import LinearNormalizer # Ensure LinearNormalizer is imported
        normalizer_for_test = LinearNormalizer()
        try:
            normalizer_for_test.load_state_dict(payload['state_dicts']['normalizer'])
            if hasattr(model, 'set_normalizer'):
                model.set_normalizer(normalizer_for_test)
                cprint("Normalizer set on model successfully using set_normalizer.", "green")
            elif hasattr(model, 'normalizer') and hasattr(model.normalizer, 'load_state_dict'): # Fallback if set_normalizer is missing for some reason
                model.normalizer.load_state_dict(payload['state_dicts']['normalizer'])
                cprint("Normalizer loaded directly into model.normalizer successfully.", "green")
            else:
                cprint("Warning: Model does not have a set_normalizer method or a normalizer attribute to load state into.", "yellow")
        except Exception as e:
            cprint(f"Error loading normalizer state_dict into new instance: {e}", "red")
    else:
        cprint("Warning: Normalizer state_dict not found in checkpoint. Model will use its default initialized normalizer.", "yellow")
    
    model.to(device)
    model.eval()

    dataset_paths_for_rollout = get_dataset_paths(cfg.dataset) 

    rollout_dataset = SprayDiffusionRolloutDataset(
        dataset_paths=dataset_paths_for_rollout, 
        config=cfg, # Pass the loaded config
        split=args.dataset_split,
        seed=args.seed
    )
    if len(rollout_dataset) == 0:
        cprint(f"Error: Rollout dataset for split '{args.dataset_split}' (using dataset config from checkpoint: {cfg.dataset}) is empty. Check dataset paths and configuration.", "red")
        # cprint(f"Debug: cfg.dataset was: {cfg.dataset}", "magenta")
        # cprint(f"Debug: dataset_paths_for_rollout resolved to: {dataset_paths_for_rollout}", "magenta")
        return

    # # Prepare dataset for rollout
    # if args.use_new_objects:
    #     cprint("Using new preprocessed objects dataset from spray_diffusion/data_objects", "cyan")
        
    #     rollout_dataset = SprayDiffusionNewObjectsDataset(
    #         data_root=args.new_objects_root,
    #         categories=args.new_objects_categories,
    #         config=cfg,
    #         split=args.dataset_split,
    #         seed=args.seed
    #     )
        
    #     if len(rollout_dataset) == 0:
    #         cprint(f"Error: New objects dataset is empty. Check data path and categories.", "red")
    #         return
            
    #     dataset_display_name = f"NewObjects({','.join(args.new_objects_categories) if args.new_objects_categories else 'all'})"
    #     cprint(f"Loaded new objects dataset: {dataset_display_name}, with {len(rollout_dataset)} objects.", "green")
        
    # else:
    #     cprint("Using original rollout dataset", "cyan")
    #     # The SprayDiffusionRolloutDataset expects paths to the raw dataset, 
    #     # similar to how it's handled in train_spraydiffusion.py.
    #     # cfg.dataset from the checkpoint should contain the necessary dataset configuration.
        
    #     # Ensure cfg.dataset is in a format that get_dataset_paths expects
    #     # (e.g., a list of dataset names or a single dataset name string that get_dataset_paths can process)
    #     # The loaded cfg.dataset should already be in the correct format from training.
    #     dataset_paths_for_rollout = get_dataset_paths(cfg.dataset) 

    dataset_display_name = cfg.dataset[0] if isinstance(cfg.dataset, list) else cfg.dataset
    cprint(f"Loaded rollout dataset: {dataset_display_name} (split: {args.dataset_split}), with {len(rollout_dataset)} episodes.", "green")

        # rollout_dataset = SprayDiffusionRolloutDataset(
        #     dataset_paths=dataset_paths_for_rollout, 
        #     config=cfg, # Pass the loaded config
        #     split=args.dataset_split,
        #     seed=args.seed
        # )
        # if len(rollout_dataset) == 0:
        #     cprint(f"Error: Rollout dataset for split '{args.dataset_split}' (using dataset config from checkpoint: {cfg.dataset}) is empty. Check dataset paths and configuration.", "red")
        #     # cprint(f"Debug: cfg.dataset was: {cfg.dataset}", "magenta")
        #     # cprint(f"Debug: dataset_paths_for_rollout resolved to: {dataset_paths_for_rollout}", "magenta")
        #     return

        # dataset_display_name = cfg.dataset[0] if isinstance(cfg.dataset, list) else cfg.dataset
        # cprint(f"Loaded rollout dataset: {dataset_display_name} (split: {args.dataset_split}), with {len(rollout_dataset)} episodes.", "green")

    rollout_loader = torch.utils.data.DataLoader(
        rollout_dataset,
        batch_size=1, # Runner expects batch_size 1
        shuffle=False,
        num_workers=args.workers
    )

    # if dataloader is not None:
    cprint(f"Using provided dataloader with {len(rollout_loader)} data points (should be episodes).", "green")
    episode_iterator = rollout_loader # Dataloader is the iterator

    # Determine the number of episodes to run, capped at 20 and also by args.eval_episodes
    # The dataloader length is the number of available episodes.
    num_episodes_available = len(rollout_loader)
    num_episodes_to_run_cli = args.eval_episodes
    
    effective_eval_episodes = min(num_episodes_available, num_episodes_to_run_cli, 20)
    # effective_eval_episodes = 80
    cprint(f"Will run evaluation for {effective_eval_episodes} episodes (available: {num_episodes_available}, requested by --eval_episodes: {num_episodes_to_run_cli}, hard cap: 20).", "magenta")

    # Initialize the runner
    # runner = SprayDiffusionRunner(
    #     model=model,
    #     device=device,
    #     output_dir=test_output_dir_for_runner,
    #     run_name=args.run_name,
    #     paper_vis_mode=args.paper_vis_mode,
    #     visualize_denoising_gifs=args.visualize_denoising_gifs,
    #     ablation_prev_traj=args.ablation_prev_traj,
    #     random_traj_seed=args.random_traj_seed
    # )

    env_runner = SprayDiffusionRunner(
        output_dir=test_output_dir_for_runner, 
        eval_episodes=effective_eval_episodes, # Runner knows how many it will process
        batch_size=1,
        num_workers=args.workers,
        seed=args.seed
    )

    # Run evaluation by calling the runner. 
    # The runner's internal loop will respect its eval_episodes, which we've set.
    # Or, if we want to iterate here and pass one batch at a time (less ideal with current runner.run)
    # For now, the runner's run method should handle iterating up to its eval_episodes count.
    cprint(f"Starting rollout evaluation via SprayDiffusionRunner for {effective_eval_episodes} episodes...", "cyan")
    # if args.enable_opaque_mesh_vis:
    #     cprint("High-quality opaque mesh visualizations will be generated for each episode.", "green")
    # else:
    #     cprint("High-quality opaque mesh visualizations are disabled.", "yellow")
    
    results = env_runner.run(
        policy=model,
        dataloader=rollout_loader, # Dataloader provides the data stream
        split=args.dataset_split,
        run_name=args.run_name,
        dataset_name=dataset_display_name, # Pass dataset name
        paper_vis_mode=args.paper_vis_mode  # Pass the flag
    )

    cprint("Rollout evaluation complete.", "green")
    
    # Optional: Save GT coverage visualization per-episode using PyVista
    # if args.save_coverage_vis:
    #     cprint("Saving GT coverage visualizations (PyVista)...", "cyan")
    #     base_demo_dir = 'demo'
    #     run_specific_demo_dir = os.path.join(base_demo_dir, args.run_name)
    #     os.makedirs(run_specific_demo_dir, exist_ok=True)

    #     def compute_covered_faces_mask_from_traj(traj_item_6d_np: np.ndarray,
    #                                              mesh_vertices_np: np.ndarray,
    #                                              mesh_faces_np: np.ndarray,
    #                                              spray_radius: float = 0.05) -> np.ndarray:
    #         T_item, D_feat_item = traj_item_6d_np.shape
    #         if D_feat_item <= 0 or (D_feat_item % 6) != 0:
    #             return np.zeros(mesh_faces_np.shape[0], dtype=bool)
    #         num_keypoints = D_feat_item // 6
    #         all_positions = []
    #         for kp_idx in range(num_keypoints):
    #             kp_pos = traj_item_6d_np[:, kp_idx*6:kp_idx*6+3]
    #             valid = ~np.all(kp_pos == -100.0, axis=1)
    #             if np.any(valid):
    #                 all_positions.append(kp_pos[valid])
    #         if not all_positions:
    #             return np.zeros(mesh_faces_np.shape[0], dtype=bool)
    #         spray_positions = np.concatenate(all_positions, axis=0)
    #         face_centroids = mesh_vertices_np[mesh_faces_np].mean(axis=1) # [F,3]
    #         if spray_positions.shape[0] == 0 or face_centroids.shape[0] == 0:
    #             return np.zeros(mesh_faces_np.shape[0], dtype=bool)
    #         diffs = spray_positions[:, None, :] - face_centroids[None, :, :]
    #         min_dist_sq = np.min(np.sum(diffs**2, axis=2), axis=0)
    #         return min_dist_sq <= (spray_radius ** 2)

    #     spray_radius = getattr(cfg, 'coverage_spray_radius', 0.05) if hasattr(cfg, 'coverage_spray_radius') else 0.05
    #     for episode_idx, batch in enumerate(rollout_loader):
    #         if episode_idx >= effective_eval_episodes:
    #             break
    #         try:
    #             mesh_faces = batch.get('mesh_faces', None)[0].cpu().numpy()
    #             mesh_vertices = batch.get('mesh_vertices', None)[0].cpu().numpy()
    #             gt_traj = batch['full_trajectory'][0].cpu().numpy()  # [T, D]

    #             episode_condition_output_dir = os.path.join(run_specific_demo_dir, f"episode_{episode_idx:03d}_GT_Cond")
    #             os.makedirs(episode_condition_output_dir, exist_ok=True)

    #             gt_mask = compute_covered_faces_mask_from_traj(gt_traj, mesh_vertices, mesh_faces, spray_radius=spray_radius)
    #             cov_gt_perc = float(np.mean(gt_mask) * 100.0) if gt_mask.size > 0 else 0.0
    #             cov_gt_path = os.path.join(episode_condition_output_dir, f'coverage_ep{episode_idx}_GT_Cond_gt.png')

    #             visualize_mesh_coverage_pyvista(
    #                 mesh_vertices_np=mesh_vertices,
    #                 mesh_faces_np=mesh_faces,
    #                 covered_faces_mask=gt_mask,
    #                 traj=gt_traj,
    #                 offset=0.2,
    #                 save_path=cov_gt_path,
    #                 title=f'GT Coverage: {cov_gt_perc:.1f}%'
    #             )
    #             cprint(f"Saved GT coverage visualization: {cov_gt_path}", "green")
    #         except Exception as e:
    #             cprint(f"Warning: Failed to save GT coverage vis for episode {episode_idx}: {e}", "yellow")

    # Print aggregated results (example, adapt based on what runner.run returns)
    if results:
        cprint("Aggregated Test Results:", "blue")
        if 'pred_cond' in results and results['pred_cond']:
            cprint("--- Prediction Conditioned ---", "yellow")
            for key, value in results['pred_cond'].items():
                if isinstance(value, np.ndarray):
                    # Assuming MetricsHandler.pprint might be useful if available
                    # For now, just print simple means if they are single numbers
                    if value.size == 1:
                        cprint(f"  {key}: {value.item():.4f}", "yellow")
                    else:
                        # If it's an array of metrics, maybe print its mean
                        cprint(f"  {key} (mean over metrics): {np.nanmean(value):.4f}", "yellow") 
                elif isinstance(value, (float, int)):
                    cprint(f"  {key}: {value:.4f}", "yellow")
                # else: print(f"  {key}: {value}") # For other types
        if 'gt_cond' in results and results['gt_cond']:
            cprint("--- Ground Truth Conditioned ---", "yellow")
            for key, value in results['gt_cond'].items():
                if isinstance(value, np.ndarray) and value.size == 1:
                    cprint(f"  {key}: {value.item():.4f}", "yellow")
                elif isinstance(value, (float, int)):
                    cprint(f"  {key}: {value:.4f}", "yellow")

    # Save point clouds
    if args.save_point_clouds:
        cprint("Saving point clouds enabled", "cyan")
        # save_test_point_clouds(rollout_dataset, test_output_dir_for_runner, args.max_point_cloud_samples)
    else:
        cprint("Point cloud saving disabled (use --save_point_clouds to enable)", "yellow")

if __name__ == '__main__':
    main() 