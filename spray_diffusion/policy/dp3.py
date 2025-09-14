from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from termcolor import cprint
import copy
import time
# import pytorch3d.ops as torch3d_ops
from vis_utils import visualize_trajectories,visualize_trajectories_only_prev    

from spray_diffusion.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from spray_diffusion.policy.base_policy import BasePolicy
from spray_diffusion.model.diffusion.conditional_unet1d import ConditionalUnet1D
from spray_diffusion.model.diffusion.mask_generator import LowdimMaskGenerator
from spray_diffusion.common.pytorch_util import dict_apply
from spray_diffusion.common.model_util import print_params
from spray_diffusion.model.vision.pointnet_extractor import DP3Encoder

class DP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,      
            encoder_output_dim=128, ##TODO
            # encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            encoder_ablation="spraydiffusion",
            # encoder_ablation="pointnet++",
            traj_mlp_hidden_dim= 128, ## TODO 1
            # traj_mlp_hidden_dim=256,
            # traj_mlp_output_dim=64,
            traj_mlp_output_dim=128,  ## TODO 2.
            # Partial observation parameters
            partial_observation_enabled=False,  ## partial_observation_enabled=False,
            partial_observation_method='fixed_camera',
            partial_observation_ratio=0.3,
            fixed_camera_positions=None,
            fixed_camera_directions=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # Partial observation configuration
        self.partial_observation_enabled = partial_observation_enabled
        self.partial_observation_method = partial_observation_method
        self.partial_observation_ratio = partial_observation_ratio
        
        # Default fixed camera positions and directions if not provided
        if fixed_camera_positions is None:
            # Define multiple fixed camera viewpoints
            self.fixed_camera_positions = [
                [1.0, 0.0, 0.5],   # Right side view
                [0.0, 1.0, 0.5],   # Front view  
                [-1.0, 0.0, 0.5],  # Left side view
                [0.0, -1.0, 0.5],  # Back view
                [0.0, 0.0, 1.0],   # Top view
            ]
        else:
            self.fixed_camera_positions = fixed_camera_positions
            
        if fixed_camera_directions is None:
            # Corresponding viewing directions (pointing towards origin)
            self.fixed_camera_directions = [
                [-1.0, 0.0, -0.5],  # Right side looking left-down
                [0.0, -1.0, -0.5],  # Front looking back-down
                [1.0, 0.0, -0.5],   # Left side looking right-down
                [0.0, 1.0, -0.5],   # Back looking front-down
                [0.0, 0.0, -1.0],   # Top looking down
            ]
        else:
            self.fixed_camera_directions = fixed_camera_directions
            
        # Normalize directions
        import numpy as np
        self.fixed_camera_directions = [
            np.array(d) / np.linalg.norm(np.array(d)) for d in self.fixed_camera_directions
        ]
        
        if self.partial_observation_enabled:
            cprint(f"[DP3] Partial observation enabled with method: {self.partial_observation_method}", "cyan")
            cprint(f"[DP3] Using {len(self.fixed_camera_positions)} fixed camera positions", "cyan")
            cprint(f"[DP3] Point cloud retention ratio: {self.partial_observation_ratio}", "cyan")

        # Ablation study parameters
        self.ablation_prev_traj = kwargs.get('ablation_prev_traj', 'normal')
        self.random_traj_seed = kwargs.get('random_traj_seed', 123)
        
        # Initialize random trajectory generator for ablation study
        if self.ablation_prev_traj == 'random':
            import numpy as np
            self.random_traj_generator = np.random.RandomState(self.random_traj_seed)
            cprint(f"[DP3] Ablation study: Using random trajectories with seed {self.random_traj_seed}", "cyan")
        elif self.ablation_prev_traj == 'remove':
            cprint("[DP3] Ablation study: Removing prev_true_trajectory, using point cloud only", "cyan")
        elif self.ablation_prev_traj == 'zeros':
            cprint("[DP3] Ablation study: Using fixed zero vectors", "cyan")
        elif self.ablation_prev_traj == 'gaussian_noise':
            import numpy as np
            self.gaussian_noise_generator = np.random.RandomState(self.random_traj_seed)
            cprint(f"[DP3] Ablation study: Using pure Gaussian noise trajectories with seed {self.random_traj_seed}", "cyan")
        elif self.ablation_prev_traj == 'temporal_only':
            import numpy as np
            self.temporal_only_generator = np.random.RandomState(self.random_traj_seed)
            cprint(f"[DP3] Ablation study: Using temporal-only trajectories (preserve time structure, remove geometry) with seed {self.random_traj_seed}", "cyan")
        else:
            cprint("[DP3] Ablation study: Using normal mode with real prev_true_trajectory", "cyan")

        # parse shape_meta
        action_shape = shape_meta['action']['shape']  # [x] TODO change 26 to 24
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        # obs_shape_meta = shape_meta['obs']
        # # obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # obs_dict = dict_apply(
        #     obs_shape_meta,
        #     lambda x: x['shape'] if isinstance(x, dict) and 'shape' in x else x
        # )   
        ## TODO, make it in yaml file
        obs_dict = {'point_cloud': [5120, 3]}
        crop_shape = [80,80]
        pointcloud_encoder_cfg = {'in_channels': 3, 'out_channels': 128, 'use_layernorm': True, 'final_norm': 'layernorm', 'normal_channel': False}   ## TODO
        # pointcloud_encoder_cfg = {'in_channels': 3, 'out_channels': 128, 'use_layernorm': True, 'final_norm': 'layernorm', 'normal_channel': False}
        pointcloud_encoder_cfg = OmegaConf.create(pointcloud_encoder_cfg)
        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                img_crop_shape=None,
                                                # out_channel=256,
                                                out_channel=128, ## TODO
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                encoder_ablation=encoder_ablation,
                                                )

        # 添加轨迹处理MLP
        self.traj_mlp = nn.Sequential(
            nn.Linear(24, traj_mlp_hidden_dim),
            # nn.LayerNorm(traj_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(traj_mlp_hidden_dim, traj_mlp_output_dim),
            # nn.LayerNorm(traj_mlp_output_dim)
        )
        
        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape() # we don't have agent_pos, so here 64 is right.
        # input_dim = action_dim + obs_feature_dim        input_dim = action_dim + obs_feature_dim + traj_mlp_output_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                # 使用traj_mlp_output_dim而不是直接使用24
                global_cond_dim = obs_feature_dim + traj_mlp_output_dim
            else:
                # 使用traj_mlp_output_dim而不是直接使用24
                global_cond_dim = obs_feature_dim * n_obs_steps + traj_mlp_output_dim
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.traj_mlp_output_dim = traj_mlp_output_dim  # 保存traj_mlp输出维度
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] traj_mlp_output_dim: {self.traj_mlp_output_dim}", "yellow")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,  # TODO figure out the action dimension
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        # List to store intermediate denoising steps for visualization
        denoising_trajectory_steps = []
        # Determine which timesteps to log for visualization (e.g., 10 evenly spaced steps)
        num_total_steps = len(scheduler.timesteps)
        log_interval = max(1, num_total_steps // 10) # Log ~10 steps

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # Log the initial noisy trajectory first
        denoising_trajectory_steps.append(trajectory.clone())

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for i, t in enumerate(scheduler.timesteps):
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict noise residual using the model
            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
            # Log intermediate step for visualization
            if i % log_interval == 0 or i == num_total_steps - 1:
                denoising_trajectory_steps.append(trajectory.clone())
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   

        return trajectory, denoising_trajectory_steps


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
            
        # Handle -100 values in prev_true_trajectory before normalization
        prev_true_trajectory = None
        trajectory_mask = None

        #--- process prev_true_trajectory ---#
        assert 'prev_true_trajectory' in obs_dict, "prev_true_trajectory is not in obs_dict"
        prev_true_trajectory = obs_dict['prev_true_trajectory']
        assert prev_true_trajectory.min() != -100, "prev_true_trajectory is not valid"
        
        # Apply trajectory ablation study
        prev_true_trajectory = self._apply_trajectory_ablation(prev_true_trajectory)
        
        # prev_true_trajectory_mask = (prev_true_trajectory != -100)
        # offset = self.normalizer['action'].params_dict['offset']  
        # offset_expanded = offset.view(1, -1).expand_as(prev_true_trajectory)
        # prev_true_trajectory_for_norm = torch.where(prev_true_trajectory_mask, prev_true_trajectory, offset_expanded)
        # assert prev_true_trajectory_for_norm.min() != -100, "prev_true_trajectory is not valid"
        # nprev_true_trajectory = self.normalizer['action'].normalize(prev_true_trajectory_for_norm)
        # nprev_true_trajectory[~prev_true_trajectory_mask] = 0
        nprev_true_trajectory = self.normalizer['action'].normalize(prev_true_trajectory)

        zero_rows = (nprev_true_trajectory == 0).all(dim=1) 
        assert not zero_rows.any(), 'nprev_true_trajectory 存在全为0的行'

        # # Skip zero check for ablation modes that intentionally use zeros
        # if self.ablation_prev_traj not in ['remove', 'zeros']:
        #     zero_rows = (nprev_true_trajectory == 0).all(dim=1) 
        #     assert not zero_rows.any(), 'nprev_true_trajectory 存在全为0的行'

        #--- process obs ---#
        nobs = self.normalizer.normalize(obs_dict['obs'])
        
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            
        # Apply partial observation if enabled
        if self.partial_observation_enabled:
            original_pc = nobs['point_cloud'].clone()
            nobs['point_cloud'] = self._apply_partial_observation(nobs['point_cloud'])
            
            # Visualize partial observation results
            self._visualize_partial_observation(original_pc, nobs['point_cloud'], save_dir)
            
        this_n_point_cloud = nobs['point_cloud']
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            
            processed_traj = self.traj_mlp(nprev_true_trajectory)

            # 连接处理后的轨迹特征
            global_cond = torch.cat([global_cond, processed_traj], dim=-1)

            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        assert cond_mask.sum() == 0, "cond_mask should be all False"
        # # For predict_action, condition_mask is effectively all False as we predict the whole trajectory
        # # The conditional_sample function expects a mask for conditioning known parts, 
        # # but here we are generating everything from noise based on global_cond.
        # # So, an all-False mask means no part of the initial random trajectory is kept fixed.
        # pass # cond_mask is correctly all False here for unconditional generation part

        # run sampling
        nsample, denoising_steps = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get visualzation
        result = {
            'action': action,
            'action_pred': action_pred,
            # 'denoising_trajectory_steps': denoising_steps_list # Add the collected steps
        }
        
        return result


    def compute_loss(self, batch, save_dir):

        # Handle -100 values in prev_true_trajectory before normalization
        trajectory_mask = None
        prev_true_trajectory = None
        
        #--- process obs ---#
        nobs = self.normalizer.normalize(batch['obs'])

        #--- process action ---#
        # Normalize the observation data without prev_true_trajectory
        action = batch['action']
        padding_mask= (batch['action'] != -100)
        # batch['action'] -> [B, T, D]   padding_mask 同 shape
        offset = self.normalizer['action'].params_dict['offset']              # [D]
        # reshape + 广播到 [1,1,D] -> [B,T,D]
        offset_expanded = offset.view(1, 1, -1).expand_as(action)            
        # element-wise 替换
        action_for_norm = torch.where(padding_mask, action, offset_expanded)
        assert action_for_norm.min() != -100, "action is not valid"

        nactions = self.normalizer['action'].normalize(action_for_norm)
        nactions[~padding_mask] = 0
        
        #--- process prev_true_trajectory ---#
        assert batch.get('prev_true_trajectory', None) is not None, "prev_true_trajectory is None"
        prev_true_trajectory = batch['prev_true_trajectory']
        assert prev_true_trajectory.min() != -100, "prev_true_trajectory is not valid"
        
        # Apply trajectory ablation study
        prev_true_trajectory = self._apply_trajectory_ablation(prev_true_trajectory)
        
        # trajectory_mask = (prev_true_trajectory != -100)
        
        # # Check for -100 values and replace them with offset
        # offset = self.normalizer['action'].params_dict['offset']              # [D]
        
        # # reshape + 广播到适合的维度
        # offset_expanded = offset.view(1, -1).expand_as(prev_true_trajectory)

        # # element-wise 替换
        # prev_true_trajectory_for_norm = torch.where(trajectory_mask, prev_true_trajectory, offset_expanded)
        # assert prev_true_trajectory_for_norm.min() != -100, "prev_true_trajectory is not valid"
        
        # nprev_true_trajectory = self.normalizer['action'].normalize(prev_true_trajectory_for_norm)
        nprev_true_trajectory = self.normalizer['action'].normalize(prev_true_trajectory)
        # nprev_true_trajectory[~trajectory_mask] = 0

        # Skip zero check for ablation modes that intentionally use zeros
        zero_rows = (nprev_true_trajectory == 0).all(dim=1) 
        assert not zero_rows.any(), 'nprev_true_trajectory 存在全为0的行'
        # # Skip zero check for ablation modes that intentionally use zeros
        # if self.ablation_prev_traj not in ['remove', 'zeros']:
        #     zero_rows = (nprev_true_trajectory == 0).all(dim=1) 
        #     assert not zero_rows.any(), 'nprev_true_trajectory 存在全为0的行'

        #---- get global cond ----#        
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
            
        # Apply partial observation if enabled
        if self.partial_observation_enabled:
            original_pc = nobs['point_cloud'].clone()
            nobs['point_cloud'] = self._apply_partial_observation(nobs['point_cloud'])
            
            # Visualize partial observation results
            self._visualize_partial_observation(original_pc, nobs['point_cloud'], save_dir)
        
        batch_size = nobs['point_cloud'].shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,            ## self.n_obs_steps = 1 TODO
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            

            processed_traj = self.traj_mlp(nprev_true_trajectory)

            # 连接处理后的轨迹特征
            global_cond = torch.cat([global_cond, processed_traj], dim=-1)

            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        assert condition_mask.sum() == 0, "condition_mask should be all False"

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask - combine conditioning mask with padding mask
        loss_mask = ~condition_mask
        
        # Combine masks: only compute loss for non-padding, non-conditioned values
        loss_mask = loss_mask & padding_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                        local_cond=local_cond, 
                        global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # Compute loss and apply mask       
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        
        # Calculate mean only over non-masked values
        valid_elements = loss_mask.sum()
        if valid_elements > 0:
            loss = loss.sum() / valid_elements
        else:
            loss = loss.sum() * 0.0  # Return 0 if no valid elements

        
        full_gt_traj = batch['full_trajectory']
        
        vis_mask_target = (batch['action'] != -100)
        
        assert (self.normalizer['action'].unnormalize(pred) * vis_mask_target).min() != -100, "pred is not valid"
        assert (self.normalizer['action'].unnormalize(target) * vis_mask_target).min() != -100, "target is not valid"
        assert self.normalizer['action'].unnormalize(nprev_true_trajectory).min() != -100, "nprev_true_trajectory is not valid"
        assert (self.normalizer['action'].unnormalize(nprev_true_trajectory)).sum() != 0, "nprev_true_trajectory is all 0"
        # visualize_trajectories_only_prev(full_gt_traj[0].masked_select(full_gt_traj[0] != -100).reshape(-1, full_gt_traj[0].shape[-1])[0], timesteps[0], loss=None)
        visualize_trajectories(
            (self.normalizer['action'].unnormalize(pred) * vis_mask_target)[0],
            (self.normalizer['action'].unnormalize(target) * vis_mask_target)[0],
            timesteps[0],
            save_dir=save_dir,
            loss=loss,
            full_gt_traj=full_gt_traj[0].masked_select(full_gt_traj[0] != -100).reshape(-1, full_gt_traj[0].shape[-1]),
            prev_true_actions= (self.normalizer['action'].unnormalize(nprev_true_trajectory))[0]
        )

        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    
    def _apply_trajectory_ablation(self, original_traj):
        """
        Apply trajectory ablation based on the ablation mode.
        
        Args:
            original_traj: Original prev_true_trajectory tensor [B, D]
            
        Returns:
            Modified trajectory tensor based on ablation mode
        """
        if self.ablation_prev_traj == 'remove':
            # Return zeros to effectively remove trajectory information
            return torch.zeros_like(original_traj, device=original_traj.device)
        
        elif self.ablation_prev_traj == 'random':
            # Generate random trajectory following the same distribution as training data
            batch_size, action_dim = original_traj.shape
            
            # Generate random trajectory points within reasonable bounds
            # Assuming trajectory values are typically in range [-1, 1] after normalization
            random_traj = torch.from_numpy(
                self.random_traj_generator.uniform(-1.0, 1.0, size=(batch_size, action_dim))
            ).float().to(original_traj.device)
            
            return random_traj
        
        elif self.ablation_prev_traj == 'zeros':
            # Return fixed zero vectors with the same shape as original trajectory
            # This provides a fixed baseline that maintains the trajectory structure but with no actual trajectory information
            return torch.zeros_like(original_traj, device=original_traj.device)
        
        elif self.ablation_prev_traj == 'gaussian_noise':
            # Generate pure Gaussian noise trajectory
            # This removes all trajectory structure and provides random noise
            batch_size, action_dim = original_traj.shape
            
            # Generate Gaussian noise with mean=0, std=1
            gaussian_noise_traj = torch.from_numpy(
                self.gaussian_noise_generator.normal(0.0, 1.0, size=(batch_size, action_dim))
            ).float().to(original_traj.device)
            
            return gaussian_noise_traj
        
        elif self.ablation_prev_traj == 'temporal_only':
            # Preserve temporal structure but remove geometric information
            # This maintains the time-based ordering but scrambles spatial coordinates
            batch_size, action_dim = original_traj.shape
            
            # Assuming action_dim=24 represents 4 spray points with 6 DOF each (position + orientation)
            # We'll preserve the temporal structure by keeping the relative magnitudes but randomizing directions
            original_norms = torch.norm(original_traj.view(batch_size, -1, 6), dim=2, keepdim=True)  # [B, 4, 1]
            
            # Generate random unit directions for each spray point
            random_directions = torch.from_numpy(
                self.temporal_only_generator.normal(0.0, 1.0, size=(batch_size, action_dim))
            ).float().to(original_traj.device)
            
            # Normalize to unit vectors and scale by original magnitudes
            random_directions_reshaped = random_directions.view(batch_size, -1, 6)  # [B, 4, 6]
            random_unit_directions = random_directions_reshaped / (torch.norm(random_directions_reshaped, dim=2, keepdim=True) + 1e-8)
            
            # Scale by original magnitudes to preserve temporal intensity patterns
            temporal_only_traj = (random_unit_directions * original_norms).view(batch_size, action_dim)
            
            return temporal_only_traj
        
        else:  # 'normal' mode
            return original_traj
        
    def _apply_partial_observation(self, point_cloud):
        """
        Apply partial observation to point cloud based on fixed camera viewpoints.
        
        Args:
            point_cloud: Input point cloud tensor [B, N, 3] or [B, T, N, 3]
            
        Returns:
            Partially observed point cloud with reduced points
        """
        if not self.partial_observation_enabled:
            return point_cloud
            
        device = point_cloud.device
        dtype = point_cloud.dtype
        
        # Handle different input shapes
        if point_cloud.ndim == 4:  # [B, T, N, 3]
            batch_size, time_steps, num_points, _ = point_cloud.shape
            
            # Process each sample and collect results
            all_results = []
            for b in range(batch_size):
                batch_results = []
                for t in range(time_steps):
                    single_pc = point_cloud[b, t]  # [N, 3]
                    masked_pc = self._apply_single_camera_mask_torch(single_pc)
                    batch_results.append(masked_pc)
                all_results.append(batch_results)
            
            # Find the maximum number of points across all samples and timesteps
            max_points = 0
            for batch_results in all_results:
                for pc in batch_results:
                    max_points = max(max_points, pc.shape[0])
            
            # Pad all point clouds to the same size
            padded_results = []
            for batch_results in all_results:
                padded_batch = []
                for pc in batch_results:
                    if pc.shape[0] < max_points:
                        # Pad with zeros
                        padding = torch.zeros(max_points - pc.shape[0], 3, device=device, dtype=dtype)
                        pc_padded = torch.cat([pc, padding], dim=0)
                    else:
                        pc_padded = pc
                    padded_batch.append(pc_padded)
                padded_results.append(torch.stack(padded_batch, dim=0))
            
            result = torch.stack(padded_results, dim=0)
            
        elif point_cloud.ndim == 3:  # [B, N, 3]
            batch_size, num_points, _ = point_cloud.shape
            
            # Process each sample
            results = []
            for b in range(batch_size):
                single_pc = point_cloud[b]  # [N, 3]
                masked_pc = self._apply_single_camera_mask_torch(single_pc)
                results.append(masked_pc)
            
            # Find the maximum number of points across all samples
            max_points = max(pc.shape[0] for pc in results)
            
            # Pad all point clouds to the same size
            padded_results = []
            for pc in results:
                if pc.shape[0] < max_points:
                    # Pad with zeros
                    padding = torch.zeros(max_points - pc.shape[0], 3, device=device, dtype=dtype)
                    pc_padded = torch.cat([pc, padding], dim=0)
                else:
                    pc_padded = pc
                padded_results.append(pc_padded)
            
            result = torch.stack(padded_results, dim=0)
            
        elif point_cloud.ndim == 2:  # [N, 3]
            result = self._apply_single_camera_mask_torch(point_cloud)
        else:
            raise ValueError(f"Unsupported point cloud shape: {point_cloud.shape}")
            
        return result
    
    def _apply_single_camera_mask_torch(self, point_cloud):
        """
        Apply partial observation mask to a single point cloud using PyTorch operations.
        
        Args:
            point_cloud: Single point cloud tensor [N, 3]
            
        Returns:
            Masked point cloud tensor [M, 3] where M <= N
        """
        if point_cloud.shape[0] == 0:
            return point_cloud
            
        device = point_cloud.device
        
        if self.partial_observation_method == 'fixed_camera':
            # Randomly select one of the fixed camera positions
            import random
            camera_idx = random.randint(0, len(self.fixed_camera_positions) - 1)
            camera_pos = torch.tensor(self.fixed_camera_positions[camera_idx], 
                                    device=device, dtype=point_cloud.dtype)
            camera_dir = torch.tensor(self.fixed_camera_directions[camera_idx], 
                                    device=device, dtype=point_cloud.dtype)
            
            # Calculate point cloud centroid
            centroid = torch.mean(point_cloud, dim=0)
            
            # Translate camera position relative to point cloud centroid
            adjusted_camera_pos = centroid + camera_pos
            
            # Calculate viewing direction from camera to each point
            point_vectors = point_cloud - adjusted_camera_pos
            
            # Project points onto the camera viewing direction
            projections = torch.sum(point_vectors * camera_dir, dim=1)
            
            # Select points that are "visible" from this camera viewpoint
            visible_mask = projections > 0  # Points in front of camera
            
            if torch.sum(visible_mask) > 0:
                visible_points = point_cloud[visible_mask]
                visible_projections = projections[visible_mask]
                
                # Select the closest points (most visible)
                n_keep = max(1, int(len(visible_points) * self.partial_observation_ratio))
                if len(visible_points) > n_keep:
                    # Keep the closest points to the camera
                    _, closest_indices = torch.topk(visible_projections, n_keep, largest=False)
                    selected_points = visible_points[closest_indices]
                else:
                    selected_points = visible_points
            else:
                # Fallback: if no points are "visible", select closest points to camera
                distances = torch.norm(point_cloud - adjusted_camera_pos, dim=1)
                n_keep = max(1, int(len(point_cloud) * self.partial_observation_ratio))
                _, closest_indices = torch.topk(distances, n_keep, largest=False)
                selected_points = point_cloud[closest_indices]
                
        elif self.partial_observation_method == 'x_plane':
            # Select points from x=0 plane
            x_coords = point_cloud[:, 0]
            x_min, x_max = torch.min(x_coords), torch.max(x_coords)
            x_range = x_max - x_min
            
            # Select points with x coordinates in the smallest 10% range
            threshold = x_min + x_range * 0.1
            mask = x_coords <= threshold
            selected_points = point_cloud[mask]
            
            if len(selected_points) == 0:
                # Fallback: select at least one point
                min_idx = torch.argmin(x_coords)
                selected_points = point_cloud[min_idx:min_idx+1]
                
        elif self.partial_observation_method == 'random_plane':
            # Random plane selection
            direction = torch.randn(3, device=device, dtype=point_cloud.dtype)
            direction = direction / torch.norm(direction)
            
            centroid = torch.mean(point_cloud, dim=0)
            centered_pc = point_cloud - centroid
            
            projections = torch.sum(centered_pc * direction, dim=1)
            threshold = torch.quantile(projections, 1 - self.partial_observation_ratio)
            mask = projections >= threshold
            
            selected_points = point_cloud[mask]
            if len(selected_points) == 0:
                selected_points = point_cloud[0:1]  # Fallback
                
        else:
            raise ValueError(f"Unknown partial observation method: {self.partial_observation_method}")
            
        return selected_points

    def _visualize_partial_observation(self, original_pc, processed_pc, save_dir):
        """
        Visualize partial observation results by saving original and processed point clouds as PLY files.
        
        Args:
            original_pc: Original point cloud tensor [B, N, 3] or [B, T, N, 3]
            processed_pc: Processed point cloud tensor after partial observation
            save_dir: Directory to save PLY files
        """
        try:
            import open3d as o3d
            import os
            
            # Create save directory if it doesn't exist
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            # Handle different tensor shapes
            if original_pc.ndim == 4:  # [B, T, N, 3]
                batch_size, time_steps, _, _ = original_pc.shape
                
                for b in range(min(batch_size, 2)):  # Only visualize first 2 batches to avoid too many files
                    for t in range(min(time_steps, 1)):  # Only visualize first timestep
                        # Original point cloud
                        orig_points = original_pc[b, t].cpu().numpy()
                        orig_pcd = o3d.geometry.PointCloud()
                        orig_pcd.points = o3d.utility.Vector3dVector(orig_points)
                        orig_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for original
                        
                        # Processed point cloud
                        proc_points = processed_pc[b, t].cpu().numpy()
                        proc_pcd = o3d.geometry.PointCloud()
                        proc_pcd.points = o3d.utility.Vector3dVector(proc_points)
                        proc_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for processed
                        
                        # Save PLY files
                        if save_dir:
                            orig_path = os.path.join(save_dir, f"original_pc_batch{b}_time{t}.ply")
                            proc_path = os.path.join(save_dir, f"partial_pc_batch{b}_time{t}.ply")
                            combined_path = os.path.join(save_dir, f"combined_pc_batch{b}_time{t}.ply")
                            
                            o3d.io.write_point_cloud(orig_path, orig_pcd)
                            o3d.io.write_point_cloud(proc_path, proc_pcd)
                            
                            # Create combined visualization
                            combined_pcd = orig_pcd + proc_pcd
                            o3d.io.write_point_cloud(combined_path, combined_pcd)
                            
                            print(f"[DP3] Saved partial observation visualization: {orig_path}, {proc_path}, {combined_path}")
                        
            elif original_pc.ndim == 3:  # [B, N, 3]
                batch_size, _, _ = original_pc.shape
                
                for b in range(min(batch_size, 2)):  # Only visualize first 2 batches
                    # Original point cloud
                    orig_points = original_pc[b].cpu().numpy()
                    orig_pcd = o3d.geometry.PointCloud()
                    orig_pcd.points = o3d.utility.Vector3dVector(orig_points)
                    orig_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for original
                    
                    # Processed point cloud
                    proc_points = processed_pc[b].cpu().numpy()
                    proc_pcd = o3d.geometry.PointCloud()
                    proc_pcd.points = o3d.utility.Vector3dVector(proc_points)
                    proc_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for processed
                    
                    # Save PLY files
                    if save_dir:
                        orig_path = os.path.join(save_dir, f"original_pc_batch{b}.ply")
                        proc_path = os.path.join(save_dir, f"partial_pc_batch{b}.ply")
                        combined_path = os.path.join(save_dir, f"combined_pc_batch{b}.ply")
                        
                        o3d.io.write_point_cloud(orig_path, orig_pcd)
                        o3d.io.write_point_cloud(proc_path, proc_pcd)
                        
                        # Create combined visualization
                        combined_pcd = orig_pcd + proc_pcd
                        o3d.io.write_point_cloud(combined_path, combined_pcd)
                        
                        print(f"[DP3] Saved partial observation visualization: {orig_path}, {proc_path}, {combined_path}")
                        
            elif original_pc.ndim == 2:  # [N, 3]
                # Original point cloud
                orig_points = original_pc.cpu().numpy()
                orig_pcd = o3d.geometry.PointCloud()
                orig_pcd.points = o3d.utility.Vector3dVector(orig_points)
                orig_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for original
                
                # Processed point cloud
                proc_points = processed_pc.cpu().numpy()
                proc_pcd = o3d.geometry.PointCloud()
                proc_pcd.points = o3d.utility.Vector3dVector(proc_points)
                proc_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for processed
                
                # Save PLY files
                if save_dir:
                    orig_path = os.path.join(save_dir, "original_pc.ply")
                    proc_path = os.path.join(save_dir, "partial_pc.ply")
                    combined_path = os.path.join(save_dir, "combined_pc.ply")
                    
                    o3d.io.write_point_cloud(orig_path, orig_pcd)
                    o3d.io.write_point_cloud(proc_path, proc_pcd)
                    
                    # Create combined visualization
                    combined_pcd = orig_pcd + proc_pcd
                    o3d.io.write_point_cloud(combined_path, combined_pcd)
                    
                    print(f"[DP3] Saved partial observation visualization: {orig_path}, {proc_path}, {combined_path}")
                    
                # Print statistics
                if original_pc.ndim >= 2:
                    orig_count = original_pc.shape[-2] if original_pc.ndim >= 3 else original_pc.shape[0]
                    proc_count = processed_pc.shape[-2] if processed_pc.ndim >= 3 else processed_pc.shape[0]
                    retention_ratio = proc_count / orig_count if orig_count > 0 else 0
                    
                    print(f"[DP3] Partial observation statistics:")
                    print(f"  Original points: {orig_count}")
                    print(f"  Processed points: {proc_count}")
                    print(f"  Retention ratio: {retention_ratio:.3f}")
                    print(f"  Method: {self.partial_observation_method}")
                    
        except Exception as e:
            print(f"[DP3] Warning: Failed to visualize partial observation: {e}")
            import traceback
            traceback.print_exc()

    # def save_checkpoint(self, checkpoint_path):
    #     torch.save({
    #         'model': self.model.state_dict(),
    #         'normalizer': self.normalizer.state_dict(),
    #         'noise_scheduler': self.noise_scheduler.state_dict(),
    #         'noise_scheduler_pc': self.noise_scheduler_pc.state_dict(),
    #         'mask_generator': self.mask_generator.state_dict(),
    #         'horizon': self.horizon,
    #         'obs_feature_dim': self.obs_feature_dim,
    #         'action_dim': self.action_dim,
    #         'n_action_steps': self.n_action_steps,
    #         'n_obs_steps': self.n_obs_steps,
    #         'obs_as_global_cond': self.obs_as_global_cond,
    #         'kwargs': self.kwargs,
    #         'num_inference_steps': self.num_inference_steps,
    #         'use_pc_color': self.use_pc_color,
    #         'pointnet_type': self.pointnet_type,
    #         'condition_type': self.condition_type,
    #         'down_dims': self.model.down_dims,
    #         'kernel_size': self.model.kernel_size,
    #         'n_groups': self.model.n_groups,
    #         'diffusion_step_embed_dim': self.model.diffusion_step_embed_dim,
    #         'use_down_condition': self.model.use_down_condition,
    #         'use_mid_condition': self.model.use_mid_condition,
    #         'use_up_condition': self.model.use_up_condition,
    #         'encoder_output_dim': self.model.encoder_output_dim,
    #         'crop_shape': self.model.crop_shape,
    #         'pointcloud_encoder_cfg': self.obs_encoder.pointcloud_encoder_cfg,
    #     }, checkpoint_path)

    #     print("Saving normalizer state:", self.normalizer.state_dict())

    # def load_checkpoint(self, checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     self.model.load_state_dict(checkpoint['model'])
    #     self.normalizer.load_state_dict(checkpoint['normalizer'])
    #     self.noise_scheduler.load_state_dict(checkpoint['noise_scheduler'])
    #     self.noise_scheduler_pc.load_state_dict(checkpoint['noise_scheduler_pc'])
    #     self.mask_generator.load_state_dict(checkpoint['mask_generator'])
    #     self.horizon = checkpoint['horizon']
    #     self.obs_feature_dim = checkpoint['obs_feature_dim']
    #     self.action_dim = checkpoint['action_dim']
    #     self.n_action_steps = checkpoint['n_action_steps']
    #     self.n_obs_steps = checkpoint['n_obs_steps']
    #     self.obs_as_global_cond = checkpoint['obs_as_global_cond']
    #     self.kwargs = checkpoint['kwargs']
    #     self.num_inference_steps = checkpoint['num_inference_steps']
    #     self.use_pc_color = checkpoint['use_pc_color']
    #     self.pointnet_type = checkpoint['pointnet_type']
    #     self.condition_type = checkpoint['condition_type']
    #     self.model.down_dims = checkpoint['down_dims']
    #     self.model.kernel_size = checkpoint['kernel_size']
    #     self.model.n_groups = checkpoint['n_groups']
    #     self.model.diffusion_step_embed_dim = checkpoint['diffusion_step_embed_dim']
    #     self.model.use_down_condition = checkpoint['use_down_condition']
    #     self.model.use_mid_condition = checkpoint['use_mid_condition']
    #     self.model.use_up_condition = checkpoint['use_up_condition']
    #     self.model.encoder_output_dim = checkpoint['encoder_output_dim']
    #     self.model.crop_shape = checkpoint['crop_shape']
    #     self.obs_encoder.pointcloud_encoder_cfg = checkpoint['pointcloud_encoder_cfg']

    #     print("Loading normalizer state:", self.normalizer.state_dict())
