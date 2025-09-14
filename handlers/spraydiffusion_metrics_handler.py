"""SprayDiffusion-specific Metrics Handler

A simplified metrics handler for SprayDiffusion models that works with the metrics
returned by SprayDiffusionRunner.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb

def convert_tensor_to_numpy(tensor):
    """Convert tensor to numpy array"""
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor

class SprayDiffusionMetricsHandler:
    """Handle computation of evaluation metrics for SprayDiffusion.
    
    This metrics handler is specifically designed to work with the metrics
    returned by SprayDiffusionRunner. It focuses on two main metrics:
    1. Chamfer distance (mean squared error)
    2. Success rate (percentage of samples below threshold)
    """

    def __init__(self, config, metrics=None):
        """Initialize the SprayDiffusion metrics handler
        
        Args:
            config: Configuration object
            metrics: List of metrics to compute (defaults to ['chamfer_distance'])
        """
        self.config = config
        
        # Default to chamfer_distance if no metrics provided
        if metrics is None:
            metrics = ['chamfer_distance']
        self.metrics = metrics
        
        # SprayDiffusion metrics are simpler than the original metrics handler
        self.metrics_names = [
            'chamfer_distance',
            'success_rate'
        ]
        
        # Map metric names to their descriptive output names
        self.output_metrics_names = {
            'chamfer_distance': 'Chamfer Distance',
            'success_rate': 'Success Rate'
        }

    def compute(self, **kwargs):
        """Placeholder for metrics computation.
        
        In SprayDiffusion, metrics are computed by SprayDiffusionRunner.
        """
        return np.zeros(len(self.metrics))
    
    def pprint(self, metric_values, prefix=''):
        """Pretty print metric values
        
        This version handles the case where metric_values may contain
        values that don't match the expected metrics count.
        
        Args:
            metric_values: Array-like of metric values
            prefix: Optional prefix for the printed message
        """
        print(f"{prefix}")
        
        # Convert to numpy array for easier handling
        if isinstance(metric_values, list):
            metric_values = np.array(metric_values)
        elif isinstance(metric_values, (int, float)):
            metric_values = np.array([metric_values])
        
        # Handle scalar or empty values
        if not hasattr(metric_values, 'shape') or metric_values.shape == ():
            print(f"\tchamfer_distance: {float(metric_values):.5f}")
            return
            
        # Handle single or multiple values
        if len(metric_values) == 1:
            print(f"\tchamfer_distance: {float(metric_values[0]):.5f}")
        elif len(metric_values) == 2:
            print(f"\tchamfer_distance: {float(metric_values[0]):.5f}")
            print(f"\tsuccess_rate: {float(metric_values[1]):.5f}")
        else:
            # Handle more metrics than expected
            for i, value in enumerate(metric_values):
                if i < len(self.metrics_names):
                    metric_name = self.metrics_names[i]
                    print(f"\t{self.output_metrics_names.get(metric_name, metric_name)}: {float(value):.5f}")
                else:
                    print(f"\tmetric_{i}: {float(value):.5f}")
    
    def log_on_wandb(self, metric_values, wandb_instance, epoch=None, suffix=''):
        """Log metrics to wandb
        
        Args:
            metric_values: Array of metric values
            wandb_instance: WandB instance
            epoch: Current epoch (optional)
            suffix: Suffix to add to metric names (optional)
        """
        # Convert to numpy array for easier handling
        if isinstance(metric_values, list):
            metric_values = np.array(metric_values)
        elif isinstance(metric_values, (int, float)):
            metric_values = np.array([metric_values])
            
        # Create logging dictionary
        log_dict = {}
        
        # Handle scalar or empty values
        if not hasattr(metric_values, 'shape') or metric_values.shape == ():
            log_dict[f"chamfer_distance{suffix}"] = float(metric_values)
        elif len(metric_values) == 1:
            log_dict[f"chamfer_distance{suffix}"] = float(metric_values[0])
        elif len(metric_values) == 2:
            log_dict[f"chamfer_distance{suffix}"] = float(metric_values[0])
            log_dict[f"success_rate{suffix}"] = float(metric_values[1])
        else:
            # Handle more metrics than expected
            for i, value in enumerate(metric_values):
                if i < len(self.metrics_names):
                    metric_name = self.metrics_names[i]
                    log_dict[f"{metric_name}{suffix}"] = float(value)
                else:
                    log_dict[f"metric_{i}{suffix}"] = float(value)
        
        # Add epoch if provided
        if epoch is not None:
            log_dict["epoch"] = epoch + 1
        
        # Log to wandb
        wandb_instance.log(log_dict) 