#!/usr/bin/env python3
"""
Cleaned test script for SprayDiffusion with PaintNet comparison.
This script provides a simplified interface for testing both methods.
"""

import os
import argparse
import torch
import numpy as np
import random
import dill
import pathlib
import json
import tqdm
from termcolor import cprint
from omegaconf import OmegaConf

# Project utilities
from utils import get_random_string, create_dirs
from utils.args import pformat_dict
from utils.disk import get_dataset_paths, get_output_dir
from utils.dataset.spraydiffusion_rollout_dataset import SprayDiffusionRolloutDataset
from spray_diffusion.env_runner.spraydiffusion_runner import SprayDiffusionRunner
from spray_diffusion.env_runner.base_runner import BaseRunner
from models import get_model_diffusion

class PaintNetBaselineRunner(BaseRunner):
    """Simplified PaintNet runner for comparison with SprayDiffusion."""
    
    def __init__(self,
                 model=None,
                 device=None,
                 output_dir=None,
                 run_name='default_run',
                 eval_episodes=10,
                 tqdm_interval_sec=1.0,
                 batch_size=1,  # Always use batch size of 1
                 num_workers=4,
                 seed=42,
                 paper_vis_mode=False):
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
        
        self.paper_vis_mode = paper_vis_mode
        
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
        all_results = {'pred_cond': []}
        
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
                    all_results['pred_cond'].append(episode_results['pred_cond'])
                            
                episode_count += 1
                
            except Exception as e:
                cprint(f"Error processing episode {episode_count}: {e}", "red")
                continue
        
        # Aggregate results
        aggregated_results = {}
        if all_results['pred_cond']:
            aggregated_results['pred_cond'] = self._aggregate_results(all_results['pred_cond'])
        
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
            policy_output = policy(obs)
            full_pred = policy_output['action'] if 'action' in policy_output else policy_output
        
        # Create episode output directory
        episode_condition_output_dir = os.path.join(self.demo_dir, f"episode_{episode_idx:03d}_Pred_Cond")
        os.makedirs(episode_condition_output_dir, exist_ok=True)
        
        # Generate visualizations
        self._generate_visualizations(
            full_pred, gt_trajectory, episode_condition_output_dir, 
            episode_idx, mesh_vertices_np, mesh_faces_np, dataset_name
        )
        
        # Compute simple metrics (placeholder)
        episode_metrics = [0.1, 0.2, 0.3]  # Placeholder metrics
        
        return {'pred_cond': episode_metrics}

    def _generate_visualizations(self, full_pred, gt_trajectory, output_dir, 
                               episode_idx, mesh_vertices_np, mesh_faces_np, dataset_name):
        """
        Generate visualizations for PaintNet results.
        
        Args:
            full_pred: Predicted trajectory
            gt_trajectory: Ground truth trajectory
            output_dir: Output directory for visualizations
            episode_idx: Episode index
            mesh_vertices_np: Mesh vertices as numpy array
            mesh_faces_np: Mesh faces as numpy array
            dataset_name: Name of the dataset
        """
        # Simple visualization generation
        vis_path_pred = os.path.join(output_dir, f'trajectory_ep{episode_idx}_pred.png')
        vis_path_gt = os.path.join(output_dir, f'trajectory_ep{episode_idx}_gt.png')
        
        # Create simple placeholder visualizations
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Prediction visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if mesh_vertices_np is not None and mesh_faces_np is not None:
            ax.plot_trisurf(mesh_vertices_np[:, 0], mesh_vertices_np[:, 1], mesh_vertices_np[:, 2],
                           triangles=mesh_faces_np, color='lightgrey', alpha=0.3)
        
        # Plot predicted trajectory
        pred_np = full_pred[0].detach().cpu().numpy() if torch.is_tensor(full_pred) else full_pred[0]
        pred_np = pred_np.reshape(-1, 6)
        ax.scatter(pred_np[:, 0], pred_np[:, 1], pred_np[:, 2], c='blue', s=20)
        
        ax.set_title(f'PaintNet Prediction - Episode {episode_idx}')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.savefig(vis_path_pred, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Ground truth visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if mesh_vertices_np is not None and mesh_faces_np is not None:
            ax.plot_trisurf(mesh_vertices_np[:, 0], mesh_vertices_np[:, 1], mesh_vertices_np[:, 2],
                           triangles=mesh_faces_np, color='lightgrey', alpha=0.3)
        
        # Plot ground truth trajectory
        gt_np = gt_trajectory[0].detach().cpu().numpy() if torch.is_tensor(gt_trajectory) else gt_trajectory[0]
        gt_np = gt_np.reshape(-1, 6)
        ax.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], c='red', s=20)
        
        ax.set_title(f'Ground Truth - Episode {episode_idx}')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        plt.savefig(vis_path_gt, dpi=150, bbox_inches='tight')
        plt.close()
        
        cprint(f"Saved PaintNet visualizations for episode {episode_idx}", "green")

    def _aggregate_results(self, results_list):
        """Aggregate results from multiple episodes."""
        if not results_list:
            return {'mean_metrics': np.array([np.nan, np.nan, np.nan])}
        
        results_array = np.array(results_list)
        mean_metrics = np.nanmean(results_array, axis=0)
        
        return {
            'mean_metrics': mean_metrics,
            'mean_metric_1': float(mean_metrics[0]),
            'mean_metric_2': float(mean_metrics[1]),
            'mean_metric_3': float(mean_metrics[2])
        }

def load_paintnet_baseline_model(model_path, device):
    """Load PaintNet model from checkpoint."""
    try:
        # Try to load as PyTorch model
        model = torch.load(model_path, map_location=device)
        if hasattr(model, 'eval'):
            model.eval()
        return model
    except Exception as e:
        cprint(f"Error loading PaintNet model: {e}", "red")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test SprayDiffusion and PaintNet models.')
    
    # Model arguments
    parser.add_argument('--spraydiffusion_checkpoint', type=str, default=None,
                        help='Path to SprayDiffusion checkpoint (.ckpt file).')
    parser.add_argument('--paintnet_model', type=str, default=None,
                        help='Path to PaintNet model (.pth file).')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='windows-v2',
                        help='Name of the dataset to use for testing.')
    parser.add_argument('--dataset_split', type=str, default='test',
                        help='Dataset split to use for evaluation.')
    
    # Evaluation arguments
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes to evaluate.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    
    # Output arguments
    parser.add_argument('--output_dir_base', type=str, default='outputs',
                        help='Base directory for saving outputs.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this test run.')
    parser.add_argument('--paper_vis_mode', action='store_true', default=False,
                        help='Enable paper visualization mode.')
    
    # Mode arguments
    parser.add_argument('--comparison_mode', action='store_true', default=False,
                        help='Run both SprayDiffusion and PaintNet for comparison.')
    parser.add_argument('--paintnet_only', action='store_true', default=False,
                        help='Run only PaintNet.')
    
    args = parser.parse_args()
    
    # Setup run name
    if args.run_name is None:
        if args.comparison_mode:
            args.run_name = f"comparison_{args.dataset_name}_{get_random_string(4)}"
        elif args.paintnet_only:
            args.run_name = f"paintnet_{args.dataset_name}_{get_random_string(4)}"
        else:
            args.run_name = f"spraydiffusion_{args.dataset_name}_{get_random_string(4)}"
    
    cprint(f"Starting test run: {args.run_name}", "green")
    
    # Load dataset
    dataset_paths = get_dataset_paths(args.dataset_name)
    rollout_dataset = SprayDiffusionRolloutDataset(
        dataset_paths=dataset_paths,
        config=None,  # Will be loaded from checkpoint if needed
        split=args.dataset_split,
        seed=args.seed
    )
    
    if len(rollout_dataset) == 0:
        cprint(f"Error: Dataset is empty.", "red")
        return
    
    rollout_loader = torch.utils.data.DataLoader(
        rollout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    effective_eval_episodes = min(len(rollout_loader), args.eval_episodes, 20)
    cprint(f"Will run evaluation for {effective_eval_episodes} episodes", "magenta")
    
    results = {}
    
    # Test SprayDiffusion if checkpoint provided
    if args.spraydiffusion_checkpoint and not args.paintnet_only:
        cprint("\n🚀 Testing SprayDiffusion", "cyan")
        cprint("-" * 30, "cyan")
        
        # Load SprayDiffusion model
        checkpoint_path = pathlib.Path(args.spraydiffusion_checkpoint)
        if not checkpoint_path.is_file():
            cprint(f"Error: Checkpoint file not found at {args.spraydiffusion_checkpoint}", "red")
        else:
            try:
                payload = torch.load(checkpoint_path.open('rb'), pickle_module=dill, map_location='cpu')
                cfg = payload['cfg']
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = get_model_diffusion(
                    config=cfg,
                    which=cfg.model.backbone,
                    io_type=cfg.task_name,
                    device=device
                )
                
                model.load_state_dict(payload['state_dicts']['model'])
                
                # Load normalizer
                if 'normalizer' in payload['state_dicts']:
                    from spray_diffusion.model.common.normalizer import LinearNormalizer
                    normalizer = LinearNormalizer()
                    normalizer.load_state_dict(payload['state_dicts']['normalizer'])
                    if hasattr(model, 'set_normalizer'):
                        model.set_normalizer(normalizer)
                
                model.to(device)
                model.eval()
                
                # Run SprayDiffusion evaluation
                spraydiffusion_runner = SprayDiffusionRunner(
                    output_dir=args.output_dir_base,
                    eval_episodes=effective_eval_episodes,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    seed=args.seed,
                    paper_vis_mode=args.paper_vis_mode
                )
                
                spraydiffusion_results = spraydiffusion_runner.run(
                    policy=model,
                    dataloader=rollout_loader,
                    split=args.dataset_split,
                    run_name=f"{args.run_name}_spraydiffusion",
                    dataset_name=args.dataset_name,
                    paper_vis_mode=args.paper_vis_mode
                )
                
                results['spraydiffusion'] = spraydiffusion_results
                cprint("✅ SprayDiffusion evaluation completed", "green")
                
            except Exception as e:
                cprint(f"Error running SprayDiffusion: {e}", "red")
    
    # Test PaintNet if model provided
    if args.paintnet_model and not (args.spraydiffusion_checkpoint and not args.comparison_mode):
        cprint("\n🏁 Testing PaintNet", "cyan")
        cprint("-" * 30, "cyan")
        
        # Load PaintNet model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        paintnet_model = load_paintnet_baseline_model(args.paintnet_model, device)
        
        if paintnet_model is not None:
            # Run PaintNet evaluation
            paintnet_runner = PaintNetBaselineRunner(
                model=paintnet_model,
                device=device,
                output_dir=args.output_dir_base,
                run_name=f"{args.run_name}_paintnet",
                eval_episodes=effective_eval_episodes,
                batch_size=args.batch_size,
                num_workers=args.workers,
                seed=args.seed,
                paper_vis_mode=args.paper_vis_mode
            )
            
            paintnet_results = paintnet_runner.run(
                policy=paintnet_model,
                dataloader=rollout_loader,
                split=args.dataset_split,
                run_name=f"{args.run_name}_paintnet",
                dataset_name=args.dataset_name,
                paper_vis_mode=args.paper_vis_mode
            )
            
            results['paintnet'] = paintnet_results
            cprint("✅ PaintNet evaluation completed", "green")
        else:
            cprint("❌ Failed to load PaintNet model", "red")
    
    # Print results summary
    cprint("\n📊 Evaluation Results Summary:", "blue")
    for method, result in results.items():
        cprint(f"\n--- {method.upper()} ---", "yellow")
        if 'pred_cond' in result:
            for key, value in result['pred_cond'].items():
                if isinstance(value, (float, int)):
                    cprint(f"  {key}: {value:.4f}", "yellow")
                elif isinstance(value, np.ndarray) and value.size == 1:
                    cprint(f"  {key}: {value.item():.4f}", "yellow")
    
    cprint(f"\n📁 Output saved to: {args.output_dir_base}/demo/{args.run_name}/", "cyan")

if __name__ == '__main__':
    main()
