import os
import sys
import argparse
import copy
import random
import time
import socket
import pathlib
import shutil # Added for cleaning up temp GIF frames
import imageio # Added for GIF creation

import numpy as np
from pure_eval import Evaluator
import torch
import dill
import wandb
from tqdm import tqdm
from termcolor import cprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vis_utils import visualize_trajectories, visualize_trajectories_only_prev

# Project utilities
from utils import get_random_string, create_dirs
from utils.args import load_args, pformat_dict, to_dict
from utils.config import save_config
from utils.disk import get_dataset_paths, get_output_dir
from utils.dataset.spraydiffusion_dataset import SprayDiffusionDataset, SprayDiffusionCollateBatch
from utils.dataset.spraydiffusion_rollout_dataset import SprayDiffusionRolloutDataset
from spray_diffusion.env_runner.spraydiffusion_runner import SprayDiffusionRunner
# Import the new utility function for GIF frames
from spray_diffusion.env_runner.utils_spraydiffusion import save_training_gif_frame 
from spray_diffusion.common.pytorch_util import dict_apply, optimizer_to
from spray_diffusion.model.diffusion.ema_model import EMAModel
from spray_diffusion.model.common.lr_scheduler import get_scheduler
from models import get_model_diffusion

# Re-implement or ensure TopKCheckpointManager is available
# For now, let's add a placeholder or a simplified version if direct import fails
# Simplified TopKCheckpointManager (if direct import is not working)
class BestModelCheckpointManager:
    def __init__(self, save_dir, k=1, monitor_key='test_mean_score', mode='max', verbose=False):
        self.save_dir = pathlib.Path(save_dir)
        self.k = k
        self.monitor_key = monitor_key
        self.mode = mode
        self.verbose = verbose
        self.checkpoints = []  # Stores (score, path)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_score(self, metric_dict):
        score = metric_dict.get(self.monitor_key, None)
        if score is None:
            if self.verbose:
                print(f"[TopKCheckpointManager] Monitor key '{self.monitor_key}' not found in metrics.")
            return None if self.mode == 'max' else float('inf') if self.mode == 'min' else None
        return float(score)

    def get_ckpt_path(self, metric_dict):
        score = self._get_score(metric_dict)
        if score is None:
            return None

        epoch = metric_dict.get('epoch', 0)
        global_step = metric_dict.get('global_step', 0)
        
        # Sanitize score for filename
        score_str = f"{score:.4f}".replace('.', 'p') # replace . with p to avoid issues with extension

        # Attempt to keep k checkpoints
        if len(self.checkpoints) < self.k:
            ckpt_name = f"epoch={epoch}-step={global_step}-{self.monitor_key}={score_str}.ckpt"
            new_path = self.save_dir.joinpath(ckpt_name)
            self.checkpoints.append((score, new_path))
            self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
            if self.verbose:
                print(f"[TopKCheckpointManager] Adding checkpoint {new_path} (score: {score}).")
            return new_path
        else:
            current_best_or_worst_score = self.checkpoints[-1][0]
            should_replace = (self.mode == 'max' and score > current_best_or_worst_score) or \
                             (self.mode == 'min' and score < current_best_or_worst_score)
            if should_replace:
                old_ckpt_path = self.checkpoints.pop(-1)[1]
                if old_ckpt_path.exists():
                    old_ckpt_path.unlink() # Delete the old checkpoint
                    if self.verbose:
                        print(f"[TopKCheckpointManager] Removing checkpoint {old_ckpt_path}.")
                
                ckpt_name = f"epoch={epoch}-step={global_step}-{self.monitor_key}={score_str}.ckpt"
                new_path = self.save_dir.joinpath(ckpt_name)
                self.checkpoints.append((score, new_path))
                self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
                if self.verbose:
                    print(f"[TopKCheckpointManager] Adding checkpoint {new_path} (score: {score}). Replacing old one.")
                return new_path
        return None

# Parse CLI args
parser = argparse.ArgumentParser(description='Additional training arguments')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--overfit', action='store_true', help='Use overfit dataset', default=False)
parser.add_argument('--train_all_datasets', action='store_true', help='Train on all available datasets (windows-v2, cuboids-v2, shelves-v2, containers-v2) combined', default=False)
parser.add_argument('--ablation_prev_traj', type=str, default='normal', 
                    choices=['normal', 'remove', 'random', 'zeros', 'gaussian_noise', 'temporal_only'],
                    help='Ablation study for prev_true_trajectory: normal (use real), remove (point cloud only), random (use random trajectory), zeros (use fixed zero vectors), gaussian_noise (pure Gaussian noise), temporal_only (preserve temporal structure but remove geometry)')
parser.add_argument('--random_traj_seed', type=int, default=123,
                    help='Random seed for generating random trajectories in ablation study')
# Partial observation arguments
parser.add_argument('--partial_observation', action='store_true', help='Enable partial observation', default=False)
parser.add_argument('--partial_observation_method', type=str, default='fixed_camera',
                    choices=['fixed_camera', 'x_plane', 'random_plane'],
                    help='Method for partial observation: fixed_camera (use predefined camera positions), x_plane (select x=0 plane), random_plane (random plane selection)')
parser.add_argument('--partial_observation_ratio', type=float, default=0.3,
                    help='Ratio of point cloud to retain (default: 0.3)')
cli_args, _ = parser.parse_known_args()

# Load config
config = load_args(root='configs/spraydiffusion')
config.task_name = 'SprayDiffusion'
config.overfit = cli_args.overfit
config.overfit_dataset = cli_args.overfit
config.train_all_datasets = cli_args.train_all_datasets

# Add ablation study parameters to config
config.ablation_prev_traj = cli_args.ablation_prev_traj
config.random_traj_seed = cli_args.random_traj_seed

# Add partial observation configuration
# Check if partial observation parameters exist in config file first, otherwise use CLI args
if hasattr(config, 'partial_observation_enabled'):
    # Use config file values if they exist
    print("Using partial observation settings from config file")
else:
    # Use CLI arguments if not in config file
    config.partial_observation_enabled = cli_args.partial_observation
    config.partial_observation_method = cli_args.partial_observation_method
    config.partial_observation_ratio = cli_args.partial_observation_ratio
    print("Using partial observation settings from CLI arguments")

# Override CLI args if provided (CLI args take precedence)
if cli_args.partial_observation:
    config.partial_observation_enabled = True
    config.partial_observation_method = cli_args.partial_observation_method
    config.partial_observation_ratio = cli_args.partial_observation_ratio
    print("CLI partial observation arguments override config file settings")

if config.partial_observation_enabled:
    print(f"Partial observation enabled with method: {config.partial_observation_method}")
    print(f"Point cloud retention ratio: {config.partial_observation_ratio}")

# Override model backbone for SprayDiffusion - ensure we use dp3 instead of pointnet2
if not hasattr(config, 'model'):
    from omegaconf import OmegaConf
    config.model = OmegaConf.create({'backbone': 'dp3'})
else:
    config.model.backbone = 'dp3'

# Set default dataset if None or override with all datasets if requested
if config.train_all_datasets:
    config.dataset = ['windows-v2', 'cuboids-v2', 'shelves-v2', 'containers-v2']
    print(f"🔥 Training on ALL datasets: {config.dataset}")
elif config.dataset is None:
    config.dataset = 'windows-v2'  # Use windows-v2 as default dataset

# Ensure required SprayDiffusion parameters are set
if not hasattr(config, 'action_dim'):
    config.action_dim = 24
if not hasattr(config, 'horizon'):
    config.horizon = 16
if not hasattr(config, 'n_action_steps'):
    config.n_action_steps = 100
if not hasattr(config, 'n_obs_steps'):
    config.n_obs_steps = 1
if not hasattr(config, 'encoder_output_dim'):
    config.encoder_output_dim = 256

# Add diffusion configuration if missing
if not hasattr(config, 'diffusion'):
    from omegaconf import OmegaConf
    config.diffusion = OmegaConf.create({
        'num_inference_steps': 10,
        'obs_as_global_cond': True,
        'diffusion_step_embed_dim': 128,
        'down_dims': [512, 1024, 2048],
        'kernel_size': 5,
        'n_groups': 8,
        'condition_type': "film",
        'use_down_condition': True,
        'use_mid_condition': True,
        'use_up_condition': True
    })

# Add noise scheduler configuration if missing
if not hasattr(config, 'noise_scheduler'):
    from omegaconf import OmegaConf
    config.noise_scheduler = OmegaConf.create({
        'beta_schedule': 'squaredcos_cap_v2',
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'num_train_timesteps': 100,
        'clip_sample': True,
        'set_alpha_to_one': True,
        'steps_offset': 0,
        'prediction_type': 'sample'
    })

# Add shape_meta if missing
if not hasattr(config, 'shape_meta'):
    from omegaconf import OmegaConf
    config.shape_meta = OmegaConf.create({
        'obs': {
            'point_cloud': {'shape': [5120, 3]},
            'low_dim': {'shape': [24]}
        },
        'action': {'shape': [24]}
    })

class SprayDiffusionTrainingWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.run_name = os.path.basename(output_dir) if output_dir else "unknown_run"
        self.verbose = getattr(cfg, 'debug', False)

        # Default checkpoint config if not present
        if not hasattr(cfg, 'checkpoint'):
            from omegaconf import OmegaConf
            cfg.checkpoint = OmegaConf.create({
                'save_ckpt': True,
                'checkpoint_every': 5,
                'save_last_ckpt': True,
                'save_last_snapshot': False,
                'topk': {
                    'k': 1,
                    'monitor_key': 'pred_cond_Pred_Cond_mean_point-wise_chamfer_distance',
                    # 'monitor_key': 'train_action_mse_epoch',
                    'mode': 'min'
                }
            })

        if not hasattr(cfg, 'training'):
            from omegaconf import OmegaConf
            cfg.training = OmegaConf.create({
                'use_ema': True,
                'ema_decay': 0.995,
                'gradient_accumulate_every': 1,
                'lr_scheduler': 'constant',
                'lr_warmup_steps': 0,
                'max_train_steps': None,
                'sample_every': 10,
                'wandb_sample_vis_log_ratio': 100
            })
        
        # Reproducibility
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Model
        self.model = get_model_diffusion(
            config=cfg,
            which=cfg.model.backbone,
            io_type=cfg.task_name,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # EMA, TODO []
        self.ema_model = None
        if getattr(cfg, 'training', {}).get('use_ema', True):
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:
                self.ema_model = get_model_diffusion(
                    config=cfg,
                    which=cfg.model.backbone,
                    io_type=cfg.task_name,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=getattr(cfg, 'lr', 1e-4)
        )

        # State
        self.global_step = 0
        self.epoch = 0
        self.latest_rollout_monitor_metric = None # Initialize to store the latest monitor metric from rollouts

        # Env runner for rollout
        # self.env_runner = SprayDiffusionRunner(
        #     output_dir=self.output_dir,
        #     eval_episodes=10,
        #     chamfer_distance_threshold=0.05,
        #     batch_size=1,
        #     num_workers=getattr(cfg, 'workers', 4),
        #     seed=cfg.seed,
        #     train_all_datasets=getattr(cfg, 'train_all_datasets', False)
        # )

        self.env_runner = SprayDiffusionRunner(
            output_dir=self.output_dir,
            eval_episodes=10,
            chamfer_distance_threshold=0.05,
            batch_size=1,
            num_workers=getattr(cfg, 'workers', 4),
            seed=cfg.seed,
            # train_all_datasets=getattr(cfg, 'train_all_datasets', False)
        )

    @property
    def output_dir(self):
        return self._output_dir or ''
    
    def get_checkpoint_path(self, tag='latest'):
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
        if tag == 'latest':
            return checkpoint_dir.joinpath(f'{tag}.ckpt')
        elif tag == 'best': 
            files = list(checkpoint_dir.glob('epoch=*-*.ckpt'))
            if not files:
                return None
            print("Warning: Loading 'best' checkpoint directly via get_checkpoint_path is simplified.")
            return checkpoint_dir.joinpath('latest.ckpt') 
        else:
            return checkpoint_dir.joinpath(f'{tag}.ckpt')

    def save_checkpoint(self, path=None, tag='latest'):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': {},
            'pickles': {}
        }

        # Save model state
        payload['state_dicts']['model'] = self.model.state_dict()
        if self.ema_model is not None:
            payload['state_dicts']['ema_model'] = self.ema_model.state_dict()
        
        # Save optimizer state
        payload['state_dicts']['optimizer'] = self.optimizer.state_dict()

        # Save other attributes like global_step, epoch using dill or torch.save for simple types
        for key in self.include_keys:
            if hasattr(self, key):
                try:
                    payload['pickles'][key] = dill.dumps(getattr(self, key))
                except Exception as e:
                    print(f"Warning: Could not serialize '{key}' with dill: {e}. Skipping this key for checkpoint.")
        
        try:
            torch.save(payload, path.open('wb'), pickle_module=dill if 'dill' in sys.modules else None)
            print(f"Saved checkpoint to: {path}")
        except Exception as e:
            print(f"Error saving checkpoint to {path}: {e}")
        return str(path.absolute())

    def load_checkpoint(self, path=None, tag='latest'):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)

        if not path.is_file():
            print(f"Checkpoint file not found: {path}")
            return None

        try:
            payload = torch.load(path.open('rb'), pickle_module=dill if 'dill' in sys.modules else None, map_location='cpu')
        except Exception as e:
            print(f"Error loading checkpoint from {path}: {e}")
            return None

        # Load model state
        if 'model' in payload['state_dicts']:
            self.model.load_state_dict(payload['state_dicts']['model'])
        if self.ema_model is not None and 'ema_model' in payload['state_dicts']:
            self.ema_model.load_state_dict(payload['state_dicts']['ema_model'])
        
        # Load optimizer state
        if 'optimizer' in payload['state_dicts']:
            self.optimizer.load_state_dict(payload['state_dicts']['optimizer'])

        # Load other attributes
        for key in self.include_keys:
            if key in payload['pickles']:
                try:
                    setattr(self, key, dill.loads(payload['pickles'][key]))
                except Exception as e:
                     print(f"Warning: Could not deserialize '{key}' with dill: {e}. Skipping this key.")
        
        # Restore config if needed, though self.cfg is usually set at init
        if 'cfg' in payload:
            loaded_cfg = payload['cfg'] 

        print(f"Loaded checkpoint from: {path}. Resuming at epoch {self.epoch}, global_step {self.global_step}")
        return payload

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        verbose = getattr(cfg, 'debug', False)
        # Ensure run_name is set for the workspace instance if output_dir is available
        if self.output_dir:
            self.run_name = os.path.basename(self.output_dir)
        else: # Fallback if output_dir was not set during init (should not happen with current main())
            self.run_name = "fallback_run_name_" + get_random_string(4)
            cprint(f"Warning: output_dir not set, using fallback run_name: {self.run_name}", "yellow")

        # Resume if needed
        if getattr(cfg, 'resume', False):
            self.load_checkpoint()

        # Dataset initialization
        dataset_paths = get_dataset_paths(cfg.dataset)

        dataset_name = cfg.dataset[0] if isinstance(cfg.dataset, list) else cfg.dataset
        
        # # Handle multiple datasets case
        # if isinstance(cfg.dataset, list):
        #     dataset_name = cfg.dataset  # Keep as list for multi-dataset training
        #     dataset_display_name = '_'.join(cfg.dataset)  # For file naming and display
        #     print(f"🎯 Training on multiple datasets: {cfg.dataset}")
        #     print(f"📁 Dataset paths: {dataset_paths}")
        # else:
        #     dataset_name = cfg.dataset[0] if isinstance(cfg.dataset, list) else cfg.dataset
        #     dataset_display_name = dataset_name
        
        # For zarr file naming, we need individual dataset names
        name_map = {'window-v2':'windows-v2', 'cuboid-v2':'cuboids-v2',
                    'shelve-v2':'shelves-v2', 'container-v2':'containers-v2'}

        base_name = name_map.get(dataset_name, dataset_name)[0]

        # if isinstance(cfg.dataset, list):
        #     # For multiple datasets, create combined zarr files
        #     base_names = []
        #     for ds in cfg.dataset:
        #         base_names.append(name_map.get(ds, ds))
        #     combined_name = '_'.join(base_names)
        #     zarr_train = f"data/{combined_name}_{'overfit' if cfg.overfit else 'train'}.zarr"
        #     zarr_test = f"data/{combined_name}_{'overfit' if cfg.overfit else 'test'}.zarr"
        # else:
        #     base_name = name_map.get(dataset_name, dataset_name)
        #     base_name = base_name[0]
        #     zarr_train = f"data/{base_name}_{'overfit' if cfg.overfit else 'train'}.zarr"
        #     zarr_test = f"data/{base_name}_{'overfit' if cfg.overfit else 'test'}.zarr"

        # Training dataset
        zarr_train = f"data/{base_name}_{'overfit' if cfg.overfit else 'train'}.zarr"

        zarr_test = f"data/{base_name}_{'overfit' if cfg.overfit else 'test'}.zarr"

        tr_dataset = SprayDiffusionDataset(
            config=cfg,
            dataset_paths=dataset_paths,
            split='train',
            horizon=getattr(cfg, 'horizon', 16),
            pad_before=0,
            pad_after=0,
            replay_buffer_path=zarr_train if os.path.exists(zarr_train) else None,
            replay_buffer_save_path=zarr_train,
            force_rebuild_buffer=False,
            seed=cfg.seed
        )

        # DataLoaders
        batch_size = min(cfg.batch_size, len(tr_dataset))
        collate_fn = SprayDiffusionCollateBatch(cfg)
        tr_loader = torch.utils.data.DataLoader(
            tr_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            # drop_last=True,
            drop_last=False,
            collate_fn=collate_fn
        )

        print(f"\n加载评估数据集 (split='test', {'overfit' if cfg.overfit else '正常'}模式)")
        rollout_dataset = SprayDiffusionRolloutDataset(
            dataset_paths=dataset_paths,
            config=cfg,
            # split='test',
            split = 'overfit' if cfg.overfit else 'test',
            seed=cfg.seed
        )
        print(f"Rollout dataset episodes: {len(rollout_dataset)}")

        rollout_loader = torch.utils.data.DataLoader(
            rollout_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.workers
        )

        normalizer = tr_dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model: self.ema_model.set_normalizer(normalizer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if self.ema_model: self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        steps = (len(tr_loader) * cfg.epochs) // getattr(cfg.training, 'gradient_accumulate_every', 1)
        lr_scheduler = get_scheduler(
            name = getattr(cfg.training, 'lr_scheduler', 'constant'),
            optimizer=self.optimizer,
            num_warmup_steps=getattr(cfg.training, 'lr_warmup_steps', 500),
            num_training_steps=steps,
            last_epoch=self.global_step-1
        )

        ema_active = self.ema_model and hasattr(cfg.training, 'use_ema') and cfg.training.use_ema
        ema = EMAModel(model=self.ema_model) if ema_active else None
        
        # Initialize TopKCheckpointManager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            k=cfg.checkpoint.topk.k,
            monitor_key=cfg.checkpoint.topk.monitor_key, 
            mode=cfg.checkpoint.topk.mode,
            verbose=True
        )

        wandb.init(
            config=to_dict(cfg), project="SprayDiffusion",
            name=os.path.basename(self.output_dir), group=cfg.task_name,
            save_code=True, notes=cfg.notes, mode=cfg.wandb,
            dir=str(self.output_dir)
        )
        wandb.config.update({"path": self.output_dir, "hostname": socket.gethostname()}, allow_val_change=True)

        train_sampling_batch = None
        sample_vis_event_counter = 0 # Counter for sampling visualization events

        for epoch_idx in range(self.epoch, cfg.epochs):
            self.epoch = epoch_idx
            train_losses = []
            self.model.train()
            step_log = dict()

            with tqdm(tr_loader, desc=f"Epoch {self.epoch}", leave=False) as tepoch:
                for idx, batch_data in enumerate(tepoch):
                    device = next(self.model.parameters()).device 
                    batch = dict_apply(batch_data, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                    if train_sampling_batch is None:
                        train_sampling_batch = dict_apply(batch, lambda x: x.clone() if isinstance(x, torch.Tensor) else copy.deepcopy(x))
                    
                    save_dir = os.path.join('demo', self.run_name, 'training_horizon_vis')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    raw_loss, loss_dict = self.model.compute_loss(batch, save_dir=save_dir)
                    loss = raw_loss / getattr(cfg.training, 'gradient_accumulate_every', 1)
                    loss.backward()

                    if (self.global_step + 1) % getattr(cfg.training, 'gradient_accumulate_every', 1) == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if 'lr_scheduler' in locals(): 
                            lr_scheduler.step()
                    
                    if ema: ema.step(self.model)

                    train_losses.append(raw_loss.item())
                    current_log_data = {'train_loss_step': raw_loss.item(), 'global_step': self.global_step,
                                        'epoch': self.epoch, 'lr': lr_scheduler.get_last_lr()[0] if 'lr_scheduler' in locals() else 0}
                    current_log_data.update(loss_dict)
                    
                    if wandb.run is not None:
                        wandb.log(current_log_data, step=self.global_step)
                    self.global_step += 1

                    if getattr(cfg.training, 'max_train_steps', None) and self.global_step >= cfg.training.max_train_steps:
                        break
            
            step_log['train_loss_epoch'] = np.mean(train_losses) if train_losses else float('nan')
            step_log['epoch'] = self.epoch
            step_log['global_step'] = self.global_step

            #********* Diffusion Sampling for every epoch *********#
            if train_sampling_batch and (self.epoch % 1 == 0):
                sample_vis_event_counter += 1
                with torch.no_grad():
                    device = next(self.model.parameters()).device
                    vis_batch = dict_apply(train_sampling_batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                    
                    # --- Enhanced Point Cloud Visualization ---
                    # Extract point cloud from the observation
                    if 'obs' in vis_batch and 'point_cloud' in vis_batch['obs']:
                        point_cloud_tensor = vis_batch['obs']['point_cloud'][0]  # Take first batch item
                        
                        # Convert to numpy and ensure proper shape
                        if isinstance(point_cloud_tensor, torch.Tensor):
                            point_cloud_np = point_cloud_tensor.detach().cpu().numpy()
                        else:
                            point_cloud_np = np.array(point_cloud_tensor)
                        
                        # Ensure shape is (N, 3) - handle different input formats
                        original_shape = point_cloud_np.shape
                        if point_cloud_np.ndim == 2 and point_cloud_np.shape[1] >= 3:
                            point_cloud_np = point_cloud_np[:, :3]  # Take only XYZ coordinates
                        elif point_cloud_np.ndim == 1 and len(point_cloud_np) % 3 == 0:
                            # Handle flattened point cloud
                            point_cloud_np = point_cloud_np.reshape(-1, 3)
                        else:
                            if self.verbose:
                                cprint(f"Warning: Unexpected point cloud shape {original_shape}, skipping visualization", "yellow")
                            point_cloud_np = None
                            
                        if point_cloud_np is not None and point_cloud_np.shape[0] > 0:
                            # Create point cloud save directory
                            pc_save_dir = os.path.join('demo', self.run_name, 'point_clouds')
                            os.makedirs(pc_save_dir, exist_ok=True)
                            
                            try:
                                # Ensure data type is float32 for compatibility
                                point_cloud_np = point_cloud_np.astype(np.float32)
                                
                                # Save as NPY for easy loading in Python
                                npy_path = os.path.join(pc_save_dir, f'input_point_cloud_epoch_{self.epoch:04d}.npy')
                                np.save(npy_path, point_cloud_np)
                                
                                # Save as PLY for Open3D visualization
                                ply_path = os.path.join(pc_save_dir, f'input_point_cloud_epoch_{self.epoch:04d}.ply')
                                # Simple PLY format writer
                                with open(ply_path, 'w') as f:
                                    f.write("ply\n")
                                    f.write("format ascii 1.0\n")
                                    f.write(f"element vertex {point_cloud_np.shape[0]}\n")
                                    f.write("property float x\n")
                                    f.write("property float y\n")
                                    f.write("property float z\n")
                                    f.write("end_header\n")
                                    for point in point_cloud_np:
                                        f.write(f"{point[0]} {point[1]} {point[2]}\n")
                                
                                # Print detailed info every 10 epochs or for first few epochs
                                if self.epoch % 10 == 0 or self.epoch < 5:
                                    # Calculate point cloud statistics
                                    pc_min = point_cloud_np.min(axis=0)
                                    pc_max = point_cloud_np.max(axis=0)
                                    pc_mean = point_cloud_np.mean(axis=0)
                                    pc_std = point_cloud_np.std(axis=0)
                                    
                                    cprint(f"Point Cloud Stats (Epoch {self.epoch}):", "cyan")
                                    cprint(f"  Shape: {point_cloud_np.shape} (from original {original_shape})", "cyan")
                                    cprint(f"  Range: X[{pc_min[0]:.3f}, {pc_max[0]:.3f}] Y[{pc_min[1]:.3f}, {pc_max[1]:.3f}] Z[{pc_min[2]:.3f}, {pc_max[2]:.3f}]", "cyan")
                                    cprint(f"  Mean: [{pc_mean[0]:.3f}, {pc_mean[1]:.3f}, {pc_mean[2]:.3f}]", "cyan")
                                    cprint(f"  Std:  [{pc_std[0]:.3f}, {pc_std[1]:.3f}, {pc_std[2]:.3f}]", "cyan")
                                    cprint(f"  Saved: {os.path.basename(npy_path)} & {os.path.basename(ply_path)}", "green")
                                elif self.epoch % 50 == 0:
                                    cprint(f"Saved point cloud epoch {self.epoch}: {point_cloud_np.shape[0]} points", "green")
                                    
                            except Exception as e:
                                cprint(f"Error saving point cloud for epoch {self.epoch}: {e}", "red")
                                if self.verbose:
                                    import traceback
                                    traceback.print_exc()
                    
                    obs_for_pred = vis_batch['obs']
                    prev_true_trajectory_for_pred = vis_batch['prev_true_trajectory']
                    obs_dict_for_pred = {'obs': obs_for_pred, 'prev_true_trajectory': prev_true_trajectory_for_pred}
        
                    eval_model = self.ema_model if ema_active else self.model
                    eval_model.eval()
                    
                    # Get the full prediction for MSE calculation and static visualization
                    full_action_pred_result = eval_model.predict_action(obs_dict_for_pred)
                    action_pred_tensor = full_action_pred_result['action_pred'] # This is the predicted trajectory [B, T_pred, D_action]
                    action_gt_for_vis = vis_batch['action'] # Ground truth for the prediction window [B, T_pred, D_action]
                    
                    assert action_gt_for_vis.min() != -100, "action_gt_for_vis is not valid"
                    mse = torch.nn.functional.mse_loss(action_pred_tensor, action_gt_for_vis)
                    step_log['train_action_mse_epoch'] = mse.item()

                    # --- Local Static Visualization Saving ---
                    full_gt_traj_for_vis = vis_batch.get('full_trajectory', None) # Full ground truth episode [B, T_full, D_action]
                    
                    # Path for local static visualization
                    diff_sample_vis_dir = os.path.join('demo', self.run_name, 'diff_sample_vis')
                    os.makedirs(diff_sample_vis_dir, exist_ok=True)
                    
                    if full_gt_traj_for_vis is not None:
                        vis_mask_static = (action_gt_for_vis[0] != -100)
                        
                        pred_for_static_vis_tensor = action_pred_tensor[0] # Keep as tensor for vis_utils
                        gt_for_static_vis_tensor = action_gt_for_vis[0]   # Keep as tensor

                        prev_actions_for_static_vis_tensor = prev_true_trajectory_for_pred[0] # Keep as tensor

                        gt_episode_for_static_vis_tensor = full_gt_traj_for_vis[0] # Keep as tensor

                        valid_mask_gt_episode = (gt_episode_for_static_vis_tensor != -100)
                        gt_episode_for_static_vis_tensor = gt_episode_for_static_vis_tensor[valid_mask_gt_episode]
                        
                        # Call visualize_trajectories from vis_utils.py
                        # It will save to its hardcoded path and log to wandb from there.
                        visualize_trajectories(
                            pred_actions=pred_for_static_vis_tensor,    # Expected [T,D] or [D]
                            true_actions=gt_for_static_vis_tensor,    # Expected [T,D] or [D]
                            step=self.global_step,
                            save_dir=diff_sample_vis_dir,
                            loss=mse.item(),
                            full_gt_traj=gt_episode_for_static_vis_tensor, # Expected [T,D] or [D]
                            prev_true_actions=prev_actions_for_static_vis_tensor # Expected [T,D] or [D]
                            # Removed save_path_override
                        )
                        
                    # Delete variables from the with torch.no_grad() block
                    del obs_for_pred
                    del prev_true_trajectory_for_pred
                    del obs_dict_for_pred
                    del eval_model
                    del full_action_pred_result
                    del action_pred_tensor
                    del action_gt_for_vis
                    del mse
                    if 'full_gt_traj_for_vis' in locals(): del full_gt_traj_for_vis
                    del vis_batch


            #----------- Rollout Eval every five epoch -----------------#
            if self.epoch % 5 == 0:
                print("\n=== Rollout Eval via env_runner ===")
                eval_model = self.ema_model if ema_active else self.model
                eval_model.eval()
                current_run_name = os.path.basename(self.output_dir) if self.output_dir else "default_run_from_train"
                runner_log = self.env_runner.run(eval_model, dataloader=rollout_loader, split='test', run_name=current_run_name)
                eval_model.train()
                
                if runner_log and isinstance(runner_log, dict):
                    for mode_key, mode_data in runner_log.items():
                        if isinstance(mode_data, dict):
                            for metric_name, metric_value in mode_data.items():
                                current_metric_key = f'{mode_key}_{metric_name}'
                                if isinstance(metric_value, (float, int, np.number)) or (torch.is_tensor(metric_value) and metric_value.numel() == 1):
                                    value_to_log = metric_value.item() if torch.is_tensor(metric_value) else metric_value
                                    step_log[current_metric_key] = value_to_log
                                    # Update the latest monitor metric if this is the one we are tracking
                                    if current_metric_key == cfg.checkpoint.topk.monitor_key:
                                        self.latest_rollout_monitor_metric = value_to_log
                                        if self.verbose: print(f"[Info] Updated latest_rollout_monitor_metric ({current_metric_key}) to: {value_to_log}")
                                elif isinstance(metric_value, np.ndarray) and metric_value.size == 1:
                                     value_to_log = metric_value.item()
                                     step_log[current_metric_key] = value_to_log
                                     if current_metric_key == cfg.checkpoint.topk.monitor_key:
                                        self.latest_rollout_monitor_metric = value_to_log
                                        if self.verbose: print(f"[Info] Updated latest_rollout_monitor_metric ({current_metric_key}) to: {value_to_log}")
            
            #-------------- Save checkpoint every five epoch -------------------#
            if cfg.checkpoint.save_ckpt and (self.epoch % 5 == 0):
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(tag='latest')
                
                metric_dict_for_topk = {
                    'epoch': self.epoch,
                    'global_step': self.global_step
                }
                # Populate with current step_log (training losses, etc.)
                for k, v in step_log.items():
                     if isinstance(v, (float, int, np.number)) or (torch.is_tensor(v) and v.numel() == 1):
                        metric_dict_for_topk[k] = v.item() if torch.is_tensor(v) else v
                
                # Ensure the monitor key for TopK is present using the latest rollout value if available
                if self.latest_rollout_monitor_metric is not None:
                    # if cfg.checkpoint.topk.monitor_key not in metric_dict_for_topk:
                    # Always update with the latest known rollout monitor metric, 
                    # as step_log might not have it if rollout didn't run this epoch.
                    metric_dict_for_topk[cfg.checkpoint.topk.monitor_key] = self.latest_rollout_monitor_metric
                    if self.verbose: print(f"[Info] Ensured monitor_key '{cfg.checkpoint.topk.monitor_key}' in metric_dict_for_topk with value: {self.latest_rollout_monitor_metric}")
                
                if cfg.checkpoint.topk.monitor_key not in metric_dict_for_topk:
                    print(f"Warning: Monitor key '{cfg.checkpoint.topk.monitor_key}' still not found in metric_dict_for_topk for TopKCheckpointManager. Available keys: {list(metric_dict_for_topk.keys())}")
                
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict_for_topk)
                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            
            wandb.log(step_log, step=self.global_step)

            if getattr(cfg.training, 'max_train_steps', None) and self.global_step >= cfg.training.max_train_steps:
                print("Max training steps reached.")
                break

        print("\nTraining complete")
        wandb.finish()


def main():
    random_str = get_random_string(5)
    run_name = random_str + '-S' + str(config.seed)
    
    # # Create descriptive run name based on dataset configuration
    # if config.train_all_datasets:
    #     run_name = random_str + '-S' + str(config.seed) + '-AllDatasets'
    #     print(f"🔥 Training on ALL datasets: {config.dataset}")
    # else:
    #     run_name = random_str + '-S' + str(config.seed)
    
    base_output_dir = get_output_dir(config) 
    save_dir = os.path.join(base_output_dir, run_name)
    create_dirs(save_dir)
    save_config(config, save_dir)
    
    print(f"\n===== RUN: {run_name} @ {save_dir} =====\n")
    print(pformat_dict(config))
    ws = TrainSprayDiffusionWorkspace(config, output_dir=save_dir)
    ws.run()

if __name__ == '__main__':
     main()