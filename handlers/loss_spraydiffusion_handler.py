"""
Loss handler for SprayDiffusion model.
This module provides a specialized wrapper around the LossHandler class
for calculating asymmetric v6 chamfer distance metrics.
"""

import torch
import numpy as np
import logging
from loss_handler import LossHandler
from pytorch3d_chamfer import chamfer_distance

class SprayDiffusionLossHandler:
    """
    A specialized loss handler for the SprayDiffusion model.
    This class wraps the LossHandler class and provides methods for
    calculating asymmetric v6 chamfer distance metrics.
    """
    
    def __init__(self, config=None):
        """
        Initialize the SprayDiffusionLossHandler.
        
        Args:
            config: Dictionary containing configuration parameters for the loss handler.
                   If None, default configuration will be used.
        """
        self.logger = logging.getLogger("SprayDiffusionLossHandler")
        
        # Default configuration for asymmetric v6 chamfer distance
        default_config = {
            'weight_asymm_segment_chamfer': 1.0,
            'weight_reverse_asymm_point_chamfer': 1.0,
            'weight_reverse_asymm_segment_chamfer': 1.0,
            'explicit_weight_segments_confidence': 1.0,
            'explicit_no_stroke_weight': 0.1,
            'per_segment_confidence': False,
            'smooth_target_stroke_masks': False,
            'lambda_points': 1,
            'extra_data': 'pos',
            'knn_repulsion': 10,
            'min_centroids': 0,
            'stroke_pred': False,
            'explicit_weight_stroke_masks': 1.0,
            'explicit_weight_stroke_masks_confidence': 1.0,
        }
        
        # Update default config with provided config
        if config is not None:
            default_config.update(config)
        
        # Store per_segment_confidence as a class attribute
        self.per_segment_confidence = default_config['per_segment_confidence']
        
        # Initialize the LossHandler with the configuration
        self.loss_handler = LossHandler(loss={}, config=default_config)
        
        # Initialize loss attribute
        self.loss = self.loss_handler.loss
        self.loss_index = self.loss_handler.loss_index
        self.loss_methods = self.loss_handler.loss_methods
        self.config = self.loss_handler.config

    def compute(self, return_list=True, **loss_args):
        """Return loss function
    
        Args:
            return_list: bool
                        if True, additionally return separate loss terms as list
            **loss_args: Arguments to pass to the loss functions
        """
        loss_val = 0
        loss_val_list = []

        for l in self.loss:  # Compute each loss term
            l_ind = self.loss_index[l]
            l_value = self.loss_methods[l_ind](**loss_args)  # (y_pred, y, **loss_args) as input parameters

            loss_val += self.config['weight_'+str(l)]*l_value  # Weight * loss_term
            loss_val_list.append(l_value.detach().cpu().numpy())

        if return_list:
            return loss_val, np.array(loss_val_list)
        else:
            return loss_val

    def pprint(self, loss_values, prefix=''):
        """
        Pretty print the loss values.
        
        Args:
            loss_values: Array of loss values to print
            prefix: Prefix string to add before each line
        """
        if len(self.loss) == 0:
            print(f"{prefix} No losses defined")
            return
            
        # Print each loss term with its value
        for i, loss_name in enumerate(self.loss):
            print(f"{prefix} {loss_name}: {loss_values[i]:.4f}")
            
        # Print total loss
        print(f"{prefix} Total: {np.sum(loss_values):.4f}")

    def calculate_asymmetric_v6_chamfer_distance(self, full_pred, gt_trajectory, device=None):
        """
        Calculate the asymmetric v6 chamfer distance between predicted and ground truth trajectories.
        
        Args:
            full_pred: Predicted trajectory tensor of shape [B, T, D]
            gt_trajectory: Ground truth trajectory tensor of shape [B, T, D]
            device: Device to use for calculations (default: same as full_pred)
            
        Returns:
            Dictionary containing the asymmetric v6 chamfer distance and its components
        """
        if device is None:
            device = full_pred.device
            
        # Ensure tensors are on the same device
        if not gt_trajectory.is_cuda and device.type == 'cuda':
            gt_trajectory = gt_trajectory.to(device, dtype=torch.float)
            
        try:
            # Create dummy stroke masks and scores if not provided
            B, traj_length, D = full_pred.shape
            
            # Create dummy stroke masks with shape [B, n_strokes, traj_length]
            n_strokes = 1  # Assume at least one stroke
            dummy_stroke_masks = torch.zeros((B, n_strokes, traj_length), device=device)
            dummy_stroke_masks[:, 0, :] = 1.0  # All points belong to the first stroke
            
            # Create dummy mask scores with shape [B, n_strokes]
            dummy_mask_scores = torch.ones((B, n_strokes), device=device)
            
            # Create dummy stroke IDs with shape [B, traj_length]
            dummy_stroke_ids = torch.zeros((B, traj_length), device=device)
            
            # Create dummy seg_logits with shape [B, traj_length]
            dummy_seg_logits = torch.zeros((B, traj_length), device=device)
            
            # Calculate components of asymmetric v6 chamfer distance
            # 1. asymm_segment_chamfer
            preds_to_gt_segments_chamfer_noReduction, _, pred_to_gt_match, _ = chamfer_distance(
                full_pred,
                gt_trajectory,
                padded=True,
                asymmetric=True,
                return_matching=True,
                point_reduction=None,
                batch_reduction=None
            )
            preds_to_gt_segments_chamfer = 100 * (preds_to_gt_segments_chamfer_noReduction.mean())
            
            # 1.1 per-segment confidence loss (if enabled)
            per_segment_confidence_loss = 0
            if self.per_segment_confidence:
                per_segment_confidence_loss = self.loss_handler._get_per_segment_confidence_loss(
                    nn_distance=preds_to_gt_segments_chamfer_noReduction,
                    logits=dummy_seg_logits
                )
            
            # 2. reverse_asymm_point_chamfer
            traj_as_pc = gt_trajectory.reshape(B, -1, D)  # Reshape to [B, N, D]
            if not traj_as_pc.is_cuda and device.type == 'cuda':
                traj_as_pc = traj_as_pc.to(device, dtype=torch.float)
            
            point_wise_y_pred = full_pred.reshape(B, -1, D)  # From pred segments to point-cloud
            
            gt_to_preds_points_chamfer = 100 * chamfer_distance(
                point_wise_y_pred,
                traj_as_pc,
                padded=True,
                reverse_asymmetric=True
            )[0]  # reverse asymmetric instead of reverting the first two arguments
            
            # 3. reverse_asymm_segment_chamfer
            gt_to_preds_segment_chamfer = 100 * chamfer_distance(
                full_pred,
                gt_trajectory,
                padded=True,
                reverse_asymmetric=True
            )[0]  # reverse asymmetric instead of reverting the first two arguments
            
            # Calculate the total asymmetric v6 chamfer distance without stroke masks loss
            asymm_v6_cd = (
                self.config['weight_asymm_segment_chamfer'] * preds_to_gt_segments_chamfer +
                per_segment_confidence_loss +
                self.config['weight_reverse_asymm_point_chamfer'] * gt_to_preds_points_chamfer +
                self.config['weight_reverse_asymm_segment_chamfer'] * gt_to_preds_segment_chamfer
            ).item()
            
            # Return all components for detailed analysis
            return {
                'asymm_v6_cd': asymm_v6_cd,
                'preds_to_gt_segments_chamfer': preds_to_gt_segments_chamfer.item(),
                'per_segment_confidence_loss': per_segment_confidence_loss,
                'gt_to_preds_points_chamfer': gt_to_preds_points_chamfer.item(),
                'gt_to_preds_segment_chamfer': gt_to_preds_segment_chamfer.item(),
                'per_segment_confidence': self.per_segment_confidence
            }
            
        except Exception as e:
            self.logger.warning(f"Asymmetric v6 chamfer distance calculation error: {e}")
            return {
                'asymm_v6_cd': float('inf'),
                'preds_to_gt_segments_chamfer': float('inf'),
                'per_segment_confidence_loss': 0,
                'gt_to_preds_points_chamfer': float('inf'),
                'gt_to_preds_segment_chamfer': float('inf'),
                'per_segment_confidence': False
            }
    
    def calculate_asymmetric_v6_chamfer_with_stroke_masks(self, full_pred, gt_trajectory, pred_stroke_masks, 
                                                         mask_scores, seg_logits, stroke_ids, device=None):
        """
        Calculate the asymmetric v6 chamfer distance with stroke masks between predicted and ground truth trajectories.
        
        Args:
            full_pred: Predicted trajectory tensor of shape [B, T, D]
            gt_trajectory: Ground truth trajectory tensor of shape [B, T, D]
            pred_stroke_masks: Predicted stroke masks tensor
            mask_scores: Mask scores tensor
            seg_logits: Segment logits tensor
            stroke_ids: Stroke IDs tensor
            device: Device to use for calculations (default: same as full_pred)
            
        Returns:
            Dictionary containing the asymmetric v6 chamfer distance with stroke masks and its components
        """
        if device is None:
            device = full_pred.device
            
        # Ensure tensors are on the same device
        if not gt_trajectory.is_cuda and device.type == 'cuda':
            gt_trajectory = gt_trajectory.to(device, dtype=torch.float)
            
        try:
            B, traj_length, D = full_pred.shape
            
            # Reshape trajectory as point cloud
            traj_as_pc = gt_trajectory.reshape(B, -1, D)  # Reshape to [B, N, D]
            if not traj_as_pc.is_cuda and device.type == 'cuda':
                traj_as_pc = traj_as_pc.to(device, dtype=torch.float)
            
            # Calculate asymmetric v6 chamfer distance with stroke masks
            loss = self.loss_handler.get_asymm_v6_chamfer_with_stroke_masks(
                y_pred=full_pred,
                y=gt_trajectory,
                pred_stroke_masks=pred_stroke_masks,
                mask_scores=mask_scores,
                seg_logits=seg_logits,
                stroke_ids=stroke_ids,
                traj_as_pc=traj_as_pc
            )
            
            return {
                'asymm_v6_cd_with_stroke_masks': loss.item()
            }
            
        except Exception as e:
            self.logger.warning(f"Asymmetric v6 chamfer distance with stroke masks calculation error: {e}")
            return {
                'asymm_v6_cd_with_stroke_masks': float('inf')
            } 