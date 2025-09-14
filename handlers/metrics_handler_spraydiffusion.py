"""Class for implementing and computing evaluation metrics"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pdb
import time  # Added for timing measurements

import numpy as np
import torch
from scipy.spatial.distance import cdist # Added for coverage calculation
from shapely.geometry import LineString, Point  # Added for potential geometric operations
try:
    from pytorch3d_chamfer import chamfer_distance 
except ImportError:
    print(f'Warning! Unable to import pytorch3d package.'\
          f'Chamfer distance with velocities won\'t be available.'\
          f'(Check troubleshooting.txt for info on how to install pytorch3d)')
    pass
from sklearn.metrics.cluster import v_measure_score

from utils.pointcloud import get_dim_traj_points
from utils.postprocessing import remove_padding_from_tensors, postprocess_sop_predictions, process_pred_stroke_masks_to_stroke_ids


def convert_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor


class MetricsHandler():
    """Handle computation of evaluation metrics.

    E.g. compute pose-wise chamfer distance between
    predicted mini-sequences and ground-truth
    on the test set.
    """

    def __init__(self,
                 config,
                 metrics=[],
                 renormalize_output_config : Dict = {}
                ):
        """
        Parameters:
            metrics : list of str
                      metrics to be computed
        """
        super(MetricsHandler, self).__init__()
        self.metrics = metrics
        self.metrics_names = [
                    'pcd',  # Point-wise chamfer distance
                    'smoothness', # Smoothness of trajectories
                    'coverage',    # Paint coverage on mesh
                    # 'duration',    # Duration of the spray path
                    'inference_time',  # Model inference time per sample
                    'latency'      # End-to-end latency
                  ]

        # more than a single value may be output by a single function, hence a tuple is expected
        self.output_metrics_names = [
                    ('point-wise chamfer distance',),
                    ('smoothness loss',),
                    ('paint coverage %',),
                    # ('spray duration (s)',),
                    ('inference time (ms)',),
                    ('latency (ms)',)
                  ]

        self.metric_functions = [
                    self.get_pcd,
                    self.get_smoothness,
                    self.get_coverage,
                    # self.get_duration,
                    self.get_inference_time,
                    self.get_latency
                  ] 

        self.metric_index = {metric: i for i, metric in enumerate(self.metrics_names)}
        self.config = config

        # Handle renormalization of output trajectories to a different data_scale_factor for metric computation
        self.renormalize_output_config = renormalize_output_config
        self.renormalize_output = False
        if 'active' in self.renormalize_output_config and self.renormalize_output_config['active']:
            assert self.config['normalization'] == 'per-dataset'
            self.renormalize_output = True
        

    def get_eval_metric(self, metric, **kwargs):
        """Compute single metric"""
        assert metric in self.metrics_names, f"metric {metric} is not valid"
        metric = self.metric_functions[self.metric_index[metric]](**kwargs)
        return metric


    def compute(self, **kwargs):
        """Compute all metri·cs in self.metrics
        and returns them in a list"""
        if len(self.metrics) == 0:
            return 0
        else:
            metrics = []

            for metric in self.metrics:
                metrics += self._as_list(self.get_eval_metric(metric=metric, **kwargs))
            return np.array(metrics)


    # def summary_on_wandb(self, metric_values, wandb, suffix=''):
    #     """Log metrics on wandb as a summary"""
    #     assert len(metric_values) == len(self.metrics)

    #     for name, value in zip(self.metrics, metric_values):
    #             # wandb.log({str(name)+str(suffix): value})
    #             wandb.run.summary[f"{name}{suffix}"] = value


    def log_on_wandb(self, metric_values, wandb, epoch=None, suffix=''):
        """Log metrics on wandb"""
        if len(self.metrics) == 0:
            return

        else:
            assert self.tot_num_of_metrics() == len(metric_values)

            value_index = 0
            for name in self.metrics:
                index = self.metric_index[name]

                for k in range(self.num_of_metrics(name)): 
                    output_name = self.output_metrics_names[index][k]
                    value = metric_values[value_index]
                    if epoch is None:
                        wandb.log({str(output_name)+str(suffix): value})
                    else:
                        wandb.log({str(output_name)+str(suffix): value, "epoch": (epoch+1)})
                    
                    value_index += 1


    def pprint(self, metric_values, prefix=''):
        """Pretty print metric values"""
        if len(self.metrics) == 0:
            return

        else:
            assert self.tot_num_of_metrics() == len(metric_values)

            print(prefix)
            value_index = 0
            for name in self.metrics:
                index = self.metric_index[name]

                for k in range(self.num_of_metrics(name)): 
                    print(f"\t{self.output_metrics_names[index][k]}: {round(metric_values[value_index], 5)}")
                    value_index += 1


    def _as_list(self, item):
        """See item as a list"""
        return [to_numpy(item)] if not isinstance(item, list) else to_numpy(item)


    def tot_num_of_metrics(self):
        count = 0
        for name in self.metrics:
            count += len(self.output_metrics_names[self.metric_index[name]])
        return count


    def num_of_metrics(self, name):
        return len(self.output_metrics_names[self.metric_index[name]])


    def renormalize_traj(self, traj):
        """
            Renormalize trajectory according to a different
            data_scale_factor, as defined in self.renormalize_output_config

            traj : Tensor of size [N,D]
        """
        if not self.renormalize_output:
            return traj
        else:
            assert traj.shape[-1] == 6, 'point-wise format and orientnorm is assumed.'

            fake_mask = torch.all((traj[...] == -100), axis=-1)  # do not touch fake vectors
            traj[..., :3] = torch.where(~fake_mask.unsqueeze(-1), traj[..., :3] * self.renormalize_output_config['from'], traj[..., :3])
            traj[..., :3] = torch.where(~fake_mask.unsqueeze(-1), traj[..., :3] / self.renormalize_output_config['to'], traj[..., :3])

            return traj


    """
    
        EVALUATION METRICS

    """
    def get_pcd(self, y_pred, y, traj_as_pc=None, **kwargs):
        """Pose-wise Chamfer Distance between predictions and ground-truth poses"""
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if self.config['lambda_points'] > 1:
            y_pred = y_pred.reshape(B, -1, outdim)

            # Going from GT traj as segments to points is not ideal, because there is the overlapping parameter
            # and also end-of-stroke points may be thrown out. use the traj_as_pc instead.
            if traj_as_pc is None:
                raise ValueError('DEPRECATED: Going from GT traj as segments to points is not ideal. Use traj_as_pc instead.')
                # y = y.reshape(B, -1, outdim)
                # traj_as_pc = y.clone().detach()
        
        # Pred
        traj_pred_pc = y_pred.clone().detach()
        
        # GT
        if not traj_as_pc.is_cuda:
            traj_as_pc = traj_as_pc.to('cuda', dtype=torch.float)
        if not traj_pred_pc.is_cuda:
            traj_pred_pc = traj_pred_pc.to('cuda', dtype=torch.float)
        
        # if not traj_pred_pc.is_cuda:
        #     traj_pred_pc = traj_pred_pc.to('cuda')

        with torch.no_grad():
            if self.renormalize_output:
                traj_pred_pc, traj_as_pc = self.renormalize_traj(traj_pred_pc), self.renormalize_traj(traj_as_pc)
                
            chamfer = (10**4)*chamfer_distance(traj_pred_pc, traj_as_pc, padded=True)[0]

        traj_pred_pc = traj_pred_pc.cpu()
        traj_as_pc = traj_as_pc.cpu()

        return chamfer


    def get_chamfer_original(self, y_pred, y, traj_pc, **kwargs):
        """Chamfer between predictions and full, untrimmed ground truth traj_pc.

        trimming may happen because of splitting into lambda-sequences,
        but nevertheless it generally just skips a few poses."""
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if self.config['lambda_points'] > 1:
            y_pred = y_pred.reshape(B, -1, outdim)

        traj_pred_pc = torch.tensor(y_pred)

        print('effective points pred:', traj_pred_pc.shape[1])
        print('effective points GT original:', traj_pc.shape[1])

        chamfer = (10**4)*chamfer_distance(traj_pred_pc, traj_pc)[0]
        return chamfer


    def stroke_masks_metrics(self,
                             n_strokes = None,
                             pred_stroke_masks = None,
                             mask_scores = None,
                             confidence_threshold=0.5,
                             **kwargs):
        """
            Compute metrics on the predicted strokes (clusters of segments) by MaskPlanner

            - Percentage of samples with prediction of correct number of strokes
        """
        processed_stroke_ids_pred = process_pred_stroke_masks_to_stroke_ids(pred_stroke_masks.detach().cpu(), confidence_scores=mask_scores.detach().cpu(), confidence_threshold=confidence_threshold)

        n_strokes_pred = np.array([len(set(np.unique(stroke_ids_pred_b))) for stroke_ids_pred_b in processed_stroke_ids_pred]).astype(int)
        n_strokes = np.array(n_strokes).astype(int)

        perc_correct_n_strokes = np.mean((n_strokes == n_strokes_pred).astype(int))

        avg_num_of_pred_strokes = np.mean(n_strokes_pred)
        avg_num_of_gt_strokes = np.mean(n_strokes)

        mean_absolute_error_NoP = np.mean(np.abs(n_strokes_pred - n_strokes))

        return [perc_correct_n_strokes, avg_num_of_pred_strokes, avg_num_of_gt_strokes, mean_absolute_error_NoP]


    def strokewise_num_of_strokes_metrics(self,
                                          n_strokes,
                                          traj_pred,
                                          **kwargs):
        """
            Compute num-of-strokes metrics for strokeWise models.

            n_strokes: list of int, size B
            traj_pred: list of size B of Tensors [retained_n_strokes, max_n_stroke_points*outdim]
        """
        n_strokes_pred = np.array([traj_pred_b.shape[0] for traj_pred_b in traj_pred]).astype(int)
        n_strokes = np.array(n_strokes).astype(int)

        perc_correct_n_strokes = np.mean((n_strokes == n_strokes_pred).astype(int))

        avg_num_of_pred_strokes = np.mean(n_strokes_pred)
        avg_num_of_gt_strokes = np.mean(n_strokes)

        mean_absolute_error_NoP = np.mean(np.abs(n_strokes_pred - n_strokes))

        return [perc_correct_n_strokes, avg_num_of_pred_strokes, avg_num_of_gt_strokes, mean_absolute_error_NoP]


    def get_sop_metrics(self,
                        sop_pred,
                        processed_sop_pred,
                        sop_gt,
                        pred_sop_conf_scores,
                        sop_conf_threshold,
                        **kwargs):
        """Computes start-of-path (SoP) prediction metrics"""
        unpadded_sop_gt = [remove_padding_from_tensors(sop_gt_b) for sop_gt_b in sop_gt]

        num_of_pred_sops = [len(b_item) for b_item in processed_sop_pred]
        num_of_gt_sops = [len(b_item) for b_item in unpadded_sop_gt]

        avg_num_of_pred_sops = np.mean(num_of_pred_sops)
        avg_num_of_gt_sops = np.mean(num_of_gt_sops)

        avg_ratio_pred_over_gt_sops = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])
        
        # With higher threshold
        higher_threshold = (sop_conf_threshold + 1) / 2
        processed_sop_pred_higher_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=higher_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = [len(b_item) for b_item in processed_sop_pred_higher_t]
        avg_num_of_pred_sops_if_higher = np.mean(num_of_pred_sops)
        avg_ratio_pred_over_gt_sops_if_higher = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])

        # With lower threshold
        lower_threshold = (sop_conf_threshold) / 2
        processed_sop_pred_lower_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=lower_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = [len(b_item) for b_item in processed_sop_pred_lower_t]
        avg_num_of_pred_sops_if_lower = np.mean(num_of_pred_sops)
        avg_ratio_pred_over_gt_sops_if_lower = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])

        sop_metrics = [avg_num_of_pred_sops,
                       avg_num_of_gt_sops,
                       avg_ratio_pred_over_gt_sops,
                       avg_num_of_pred_sops_if_higher,
                       avg_num_of_pred_sops_if_lower,
                       avg_ratio_pred_over_gt_sops_if_higher,
                       avg_ratio_pred_over_gt_sops_if_lower]
        
        return sop_metrics


    def get_sop_metrics_v2(self,
                           sop_pred,
                           processed_sop_pred,
                           sop_gt,
                           pred_sop_conf_scores,
                           sop_conf_threshold,
                           **kwargs):
        """Computes start-of-path (SoP) prediction metrics

            v2:
                - Avg ratio is deprecated (avg num of pred already carries enough information),
                  as it may be misleading (goods and bads can cancel out)
                - Accuracy of num of strokes
                - Mean absolute error
        """
        unpadded_sop_gt = [remove_padding_from_tensors(sop_gt_b) for sop_gt_b in sop_gt]

        num_of_pred_sops = np.array([len(b_item) for b_item in processed_sop_pred]).astype(int)
        num_of_gt_sops = np.array([len(b_item) for b_item in unpadded_sop_gt]).astype(int)

        avg_num_of_pred_sops = np.mean(num_of_pred_sops)
        avg_num_of_gt_sops = np.mean(num_of_gt_sops)

        perc_correct_n_strokes = np.mean((num_of_gt_sops == num_of_pred_sops).astype(int))
        
        mean_absolute_error_NoP = np.mean(np.abs(num_of_pred_sops - num_of_gt_sops))

        # deprecated
        # avg_ratio_pred_over_gt_sops = np.mean([n_pred_sop/n_gt_sop for (n_pred_sop, n_gt_sop) in zip(num_of_pred_sops, num_of_gt_sops)])
        
        # With higher threshold
        higher_threshold = (sop_conf_threshold + 1) / 2
        processed_sop_pred_higher_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=higher_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = np.array([len(b_item) for b_item in processed_sop_pred_higher_t]).astype(int)
        avg_num_of_pred_sops_if_higher = np.mean(num_of_pred_sops)
        mean_absolute_error_NoP_if_higher = np.mean(np.abs(num_of_pred_sops - num_of_gt_sops))

        # With lower threshold
        lower_threshold = (sop_conf_threshold) / 2
        processed_sop_pred_lower_t = postprocess_sop_predictions(sop_pred=sop_pred, pred_sop_conf_scores=pred_sop_conf_scores, sop_conf_threshold=lower_threshold)  # list of size B of Tensors [retained_n_sop, config.stroke_prototype_dim]
        num_of_pred_sops = np.array([len(b_item) for b_item in processed_sop_pred_lower_t]).astype(int)
        avg_num_of_pred_sops_if_lower = np.mean(num_of_pred_sops)
        mean_absolute_error_NoP_if_lower = np.mean(np.abs(num_of_pred_sops - num_of_gt_sops))


        sop_metrics_v2 = [
                          perc_correct_n_strokes,
                          avg_num_of_pred_sops,
                          avg_num_of_gt_sops,
                          mean_absolute_error_NoP,
                          avg_num_of_pred_sops_if_higher,
                          avg_num_of_pred_sops_if_lower,
                          mean_absolute_error_NoP_if_higher,
                          mean_absolute_error_NoP_if_lower
                         ]

        return sop_metrics_v2


    def get_clustering_metrics(self, stroke_ids_pred, stroke_ids, clusterer, **kwargs):
        """Computes clustering and its evaluation metrics"""
        B, N = stroke_ids.shape

        clustering_metrics = clusterer.eval(labels_true=stroke_ids, labels_pred=stroke_ids_pred)

        return clustering_metrics


    def get_stroke_chamfer(self, y_pred, y, traj_pc, stroke_ids, **kwargs):
        """Debug: chamfer between predicted vectors and original strokes,
        with inner distance metric as an additional chamfer distance."""
        asymmetric = True
        print(f'---\nCAREFUL! Stroke-wise chamfer is with ASYMMETRIC={asymmetric}\n---')

        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        traj_pred = torch.tensor(y_pred)
        
        ##### 1° version
        chamfers = [0 for b in range(B)]
        for b in range(B):
            chamfer = 0

            n_pred_strokes = y_pred.shape[1]
            n_gt_strokes = stroke_ids[b, -1]+1
            unique, counts = np.unique(stroke_ids[b], return_counts=True)
            assert len(unique) == n_gt_strokes
            for i in range(n_pred_strokes):
                min_chamfer = 10000000
                
                pred_pc = traj_pred[b, i].view(-1, outdim)[None, :, :]
                for i_gt in range(n_gt_strokes):
                    curr_gt_pc = traj_pc[b, stroke_ids[b, :] == i_gt, :][None, :, :]
                    curr_chamfer = (10**4)*chamfer_distance(pred_pc, curr_gt_pc, asymmetric=asymmetric)[0]
                    # dist1, dist2, _, _ = NND.nnd(pred_pc, curr_gt_pc)  # Chamfer loss
                    # chamfer = (10**4)*(torch.mean(dist1))

                    min_chamfer = min(min_chamfer, curr_chamfer.item())

                chamfer += min_chamfer

            chamfers[b] = chamfer/n_pred_strokes

        chamfers = np.array(chamfers).mean()
        ##############################

        ##### 2° version (would require stroke-padding, so it currently does not work)
        # batch_stroke_chamfer = torch.empty((B, 0))

        # n_pred_strokes = y_pred.shape[1]
        # min_chamfer = torch.ones((B,))*10000000
        # for i in range(n_pred_strokes):

        #     pred_pc = traj_pred[:, i, :].view(B, -1, outdim)
        #     for i_gt in range(n_gt_strokes):
        #         curr_gt_pc = traj_pc[b, stroke_ids[b, :] == i_gt, :][None, :, :]
        #         chamfer = (10**4)*chamfer_distance(pred_pc, curr_gt_pc, asymmetric=True)[0]
        ##############################
        return chamfers


    def paint_coverage_metrics(self, pred_covered_faces=None, gt_covered_faces=None, **kwargs):
        """
        Compute paint coverage metrics based on the number of covered faces.
        
        Args:
            pred_covered_faces: Number of faces covered by prediction
            gt_covered_faces: Number of faces covered by ground truth
            
        Returns:
            Paint coverage percentage
        """
        if pred_covered_faces is None or gt_covered_faces is None:
            return np.array([0.0])  # Return 0 if data is not available
            
        # Calculate paint coverage percentage
        paint_coverage = (pred_covered_faces / gt_covered_faces * 100).item()
        return np.array([paint_coverage])


    def get_smoothness(self, y_pred, **kwargs):
        """
        Calculates the smoothness of the predicted trajectory.
        Smoothness is defined as the mean squared acceleration magnitude across all valid trajectory points.
        Lower values indicate smoother trajectories.

        Args:
            y_pred: Tensor of shape [B, T, D_feat], where D_feat is num_keypoints * 6.
                    Padding is assumed to be -100.
        
        Returns:
            A scalar float representing the mean smoothness metric.
        """
        # Ensure input is tensor and convert to float
        if not torch.is_tensor(y_pred):
            y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        else:
            y_pred_tensor = y_pred.float()

        B, T, D_feat = y_pred_tensor.shape
        
        # Check if D_feat is valid (must be multiple of 6 for 6D keypoints)
        if D_feat <= 0 or D_feat % 6 != 0:
            print(f"Warning: D_feat ({D_feat}) in get_smoothness must be a positive multiple of 6. Returning 0.0.")
            return 0.0

        # Need at least 3 time steps to compute acceleration
        if T < 3:
            return 0.0

        # Get time step from config, default to 1.0 if not specified
        dt = self.config.get('time_step', 1.0)
        dt_squared = dt * dt

        num_keypoints = D_feat // 6
        # Reshape to [B, T, num_keypoints, 6]
        keypoint_tensor = y_pred_tensor.reshape(B, T, num_keypoints, 6)

        total_smoothness_sum = 0.0
        total_valid_segments = 0

        # Process each batch item separately for better error handling
        for b in range(B):
            batch_smoothness_sum = 0.0
            batch_valid_segments = 0
            
            for kp_idx in range(num_keypoints):
                # Extract trajectory for current keypoint: [T, 6]
                kp_traj = keypoint_tensor[b, :, kp_idx, :]
                
                # Extract position components (first 3 dimensions: XYZ)
                positions = kp_traj[:, :3]  # [T, 3]
                
                # Check for valid positions (not padding)
                valid_mask = ~torch.all(positions == -100.0, dim=1)  # [T]
                
                if torch.sum(valid_mask) < 3:
                    continue  # Need at least 3 valid points for acceleration
                
                # Find consecutive valid segments of at least 3 points
                valid_indices = torch.where(valid_mask)[0]
                
                # Process consecutive segments
                i = 0
                while i <= len(valid_indices) - 3:
                    # Check if we have 3 consecutive valid points
                    if (valid_indices[i+1] == valid_indices[i] + 1 and 
                        valid_indices[i+2] == valid_indices[i] + 2):
                        
                        # Extract 3 consecutive points
                        p0 = positions[valid_indices[i]]      # [3]
                        p1 = positions[valid_indices[i+1]]    # [3]
                        p2 = positions[valid_indices[i+2]]    # [3]
                        
                        # Compute acceleration with proper time step normalization: a = (p2 - 2*p1 + p0) / dt²
                        acceleration = (p2 - 2 * p1 + p0) / dt_squared  # [3]
                        
                        # Compute squared magnitude of acceleration
                        acc_magnitude_sq = torch.sum(acceleration ** 2)
                        
                        batch_smoothness_sum += acc_magnitude_sq.item()
                        batch_valid_segments += 1
                        
                        i += 1  # Move to next potential segment
                    else:
                        i += 1
            
            total_smoothness_sum += batch_smoothness_sum
            total_valid_segments += batch_valid_segments

        # Compute average smoothness
        if total_valid_segments > 0:
            mean_smoothness = total_smoothness_sum / total_valid_segments
        else:
            mean_smoothness = 0.0

        return mean_smoothness
    
    @staticmethod
    def _compute_coverage_single_item(pred_traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np, spray_radius=0.05):
        """
        Computes the number of covered faces and total faces for a single item.

        Args:
            pred_traj_item_6d_tensor: Predicted trajectory for single item, shape [T, D_feat_item],
                                      where D_feat_item is num_keypoints * 6 (torch.Tensor).
            mesh_vertices_np: Mesh vertices, NumPy array, shape [V, 3].
            mesh_faces_np: Mesh faces (triangle indices), NumPy array, shape [F, 3].
            spray_radius: Spray radius (float).

        Returns:
            Tuple (number of covered faces, total faces).
        """
        # Validate mesh data
        if mesh_vertices_np is None or mesh_faces_np is None or mesh_faces_np.shape[0] == 0:
            return 0, 0

        T_item, D_feat_item = pred_traj_item_6d_tensor.shape

        # Validate feature dimensions
        if D_feat_item <= 0 or D_feat_item % 6 != 0:
            print(f"_compute_coverage_single_item: Invalid feature dimension D_feat ({D_feat_item}). Should be multiple of 6.")
            return 0, mesh_faces_np.shape[0]

        num_keypoints_item = D_feat_item // 6
        pred_traj_reshaped_item = pred_traj_item_6d_tensor.reshape(T_item, num_keypoints_item, 6)
        
        # Extract all spray positions from all keypoints (not just the first one)
        # This is more robust as different keypoints might represent different spray nozzles
        all_spray_positions = []
        
        for kp_idx in range(num_keypoints_item):
            # Extract XYZ positions for this keypoint
            kp_positions = pred_traj_reshaped_item[:, kp_idx, :3]  # [T_item, 3]
            
            # Filter out padding values
            valid_mask = ~torch.all(kp_positions == -100.0, dim=1)
            valid_positions = kp_positions[valid_mask]
            
            if valid_positions.shape[0] > 0:
                all_spray_positions.append(valid_positions.cpu().numpy())
        
        # Combine all valid spray positions
        if not all_spray_positions:
            return 0, mesh_faces_np.shape[0]
        
        spray_positions_np = np.concatenate(all_spray_positions, axis=0)  # [N_total_valid, 3]
        
        if spray_positions_np.shape[0] == 0:
            return 0, mesh_faces_np.shape[0]

        total_faces_item = mesh_faces_np.shape[0]

        # Validate mesh face indices
        max_vertex_idx = np.max(mesh_faces_np)
        if max_vertex_idx >= mesh_vertices_np.shape[0]:
            print(f"Invalid mesh: face indices exceed vertex count. Max face idx: {max_vertex_idx}, vertex count: {mesh_vertices_np.shape[0]}")
            return 0, total_faces_item

        # Compute face centroids more safely
        try:
            face_centroids_item = mesh_vertices_np[mesh_faces_np].mean(axis=1)  # [F, 3]
        except (IndexError, ValueError) as e:
            print(f"Error computing face centroids: {e}")
            return 0, total_faces_item
        
        # Compute distances from spray positions to face centroids
        # Use broadcasting for efficiency: [N_spray, 1, 3] - [1, F, 3] = [N_spray, F, 3]
        spray_positions_expanded = spray_positions_np[:, np.newaxis, :]  # [N_spray, 1, 3]
        face_centroids_expanded = face_centroids_item[np.newaxis, :, :]  # [1, F, 3]
        
        # Compute squared distances for efficiency
        distances_sq = np.sum((spray_positions_expanded - face_centroids_expanded) ** 2, axis=2)  # [N_spray, F]
        
        # Find minimum distance to each face from any spray position
        min_distances_sq_to_faces = np.min(distances_sq, axis=0)  # [F]
        
        # Check coverage (compare squared distances to avoid sqrt)
        spray_radius_sq = spray_radius ** 2
        covered_faces_mask = min_distances_sq_to_faces <= spray_radius_sq
        num_covered_faces_item = np.sum(covered_faces_mask)
        
        return num_covered_faces_item, total_faces_item

    @staticmethod
    def _compute_coverage_single_item_line_based(pred_traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np, spray_radius=0.05, use_primary_keypoint_only=False, use_triangle_distance=False):
        """
        Line-based coverage: consider the entire trajectory as connected line segments
        and mark a mesh face as covered if its centroid is within spray_radius of **any segment**.
        
        Args:
            use_primary_keypoint_only: If True, only use the first keypoint for coverage calculation
            use_triangle_distance: If True, compute distance to triangle edges instead of just centroid
        """
        # Basic validations (reuse logic from point-based version)
        if mesh_vertices_np is None or mesh_faces_np is None or mesh_faces_np.shape[0] == 0:
            return 0, 0

        T_item, D_feat_item = pred_traj_item_6d_tensor.shape
        if D_feat_item % 6 != 0:
            return 0, mesh_faces_np.shape[0]

        # Validate mesh data more thoroughly
        if mesh_vertices_np.shape[1] != 3:
            print(f"Invalid mesh vertices shape: {mesh_vertices_np.shape}, expected [N, 3]")
            return 0, mesh_faces_np.shape[0]
        
        if mesh_faces_np.shape[1] != 3:
            print(f"Invalid mesh faces shape: {mesh_faces_np.shape}, expected [M, 3]")
            return 0, mesh_faces_np.shape[0]
        
        # Check if face indices are valid
        max_face_idx = np.max(mesh_faces_np) if mesh_faces_np.size > 0 else -1
        if max_face_idx >= mesh_vertices_np.shape[0]:
            print(f"Invalid mesh: face index {max_face_idx} >= vertex count {mesh_vertices_np.shape[0]}")
            return 0, mesh_faces_np.shape[0]

        num_keypoints = D_feat_item // 6
        pred_traj_reshaped = pred_traj_item_6d_tensor.reshape(T_item, num_keypoints, 6)

        # Handle keypoints appropriately - build separate paths for each keypoint or use primary keypoint
        covered_mask = np.zeros(mesh_faces_np.shape[0], dtype=bool)
        spray_radius_sq = spray_radius ** 2

        # Face centroids
        try:
            face_centroids = mesh_vertices_np[mesh_faces_np].mean(axis=1)  # [F,3]
        except IndexError as e:
            print(f"Error computing face centroids: {e}")
            return 0, mesh_faces_np.shape[0]
            
        if face_centroids.shape[0] == 0:
            return 0, 0

        # Determine which keypoints to process
        keypoints_to_process = [0] if use_primary_keypoint_only else range(num_keypoints)

        # Create sentinel tensor for comparison
        sentinel_tensor = torch.tensor(-100.0, device=pred_traj_item_6d_tensor.device, dtype=pred_traj_item_6d_tensor.dtype)

        # Process each keypoint separately to avoid incorrect connections between keypoints
        for k in keypoints_to_process:
            # Extract trajectory for this keypoint
            kp_positions = pred_traj_reshaped[:, k, :3]  # [T_item, 3]
            
            # Filter out padding values to get valid trajectory points - use torch.isclose for robust comparison
            valid_mask = ~torch.isclose(kp_positions, sentinel_tensor).all(dim=1)
            valid_positions = kp_positions[valid_mask]
            
            if valid_positions.shape[0] < 2:
                continue  # Need at least 2 points to form segments
                
            spray_path_k = valid_positions.cpu().numpy()  # [N_valid, 3]
            
            # Create consecutive segments for this keypoint's path
            spray_segments_k = np.stack([spray_path_k[:-1], spray_path_k[1:]], axis=1)  # [N-1, 2, 3]

            # Check coverage for each segment of this keypoint
            for seg_start, seg_end in spray_segments_k:
                seg_vec = seg_end - seg_start
                seg_len_sq = np.dot(seg_vec, seg_vec)
                
                if seg_len_sq == 0:
                    # Degenerate segment; treat as point
                    dist_sq = np.sum((face_centroids - seg_start) ** 2, axis=1)
                    covered_mask |= dist_sq <= spray_radius_sq
                    continue
                    
                if use_triangle_distance:
                    # More accurate: compute distance from segment to triangle edges
                    # This is computationally more expensive but more accurate
                    for face_idx in range(len(face_centroids)):
                        if covered_mask[face_idx]:
                            continue  # Already covered
                        
                        triangle_vertices = mesh_vertices_np[mesh_faces_np[face_idx]]  # [3, 3]
                        min_dist_to_triangle = float('inf') 
                        
                        # Check distance to triangle edges and vertices
                        for i in range(3):
                            v1 = triangle_vertices[i]
                            v2 = triangle_vertices[(i + 1) % 3]
                            
                            # Distance from spray segment to triangle edge
                            dist = MetricsHandler._segment_to_segment_distance(seg_start, seg_end, v1, v2)
                            min_dist_to_triangle = min(min_dist_to_triangle, dist)
                        
                        # Also check distance to triangle vertices
                        for vertex in triangle_vertices:
                            dist = MetricsHandler._point_to_segment_distance(vertex, seg_start, seg_end)
                            min_dist_to_triangle = min(min_dist_to_triangle, dist)
                        
                        if min_dist_to_triangle <= spray_radius:
                            covered_mask[face_idx] = True
                else:
                    # Original method: Point-to-line-segment distance calculation using centroids
                    p = face_centroids - seg_start  # [F,3]
                    t_param = np.clip((p @ seg_vec) / seg_len_sq, 0.0, 1.0)  # [F]
                    proj = seg_start + t_param[:, None] * seg_vec  # [F,3]
                    dist_sq = np.sum((proj - face_centroids) ** 2, axis=1)
                    covered_mask |= dist_sq <= spray_radius_sq

        return int(np.sum(covered_mask)), len(face_centroids)

    def _compute_coverage_mask_line_based(self, pred_traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np, spray_radius=0.05, use_primary_keypoint_only=False, use_triangle_distance=False):
        """
        Returns boolean mask (F,) of covered faces using the same line-based method as _compute_coverage_single_item_line_based.
        """
        if mesh_vertices_np is None or mesh_faces_np is None or mesh_faces_np.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        T_item, D_feat_item = pred_traj_item_6d_tensor.shape
        if D_feat_item % 6 != 0:
            return np.zeros(mesh_faces_np.shape[0], dtype=bool)

        if mesh_vertices_np.shape[1] != 3 or mesh_faces_np.shape[1] != 3:
            return np.zeros(mesh_faces_np.shape[0], dtype=bool)

        max_face_idx = np.max(mesh_faces_np) if mesh_faces_np.size > 0 else -1
        if max_face_idx >= mesh_vertices_np.shape[0]:
            return np.zeros(mesh_faces_np.shape[0], dtype=bool)

        num_keypoints = D_feat_item // 6
        pred_traj_reshaped = pred_traj_item_6d_tensor.reshape(T_item, num_keypoints, 6)

        covered_mask = np.zeros(mesh_faces_np.shape[0], dtype=bool)
        spray_radius_sq = spray_radius ** 2

        try:
            face_centroids = mesh_vertices_np[mesh_faces_np].mean(axis=1)  # [F,3]
        except Exception:
            return np.zeros(mesh_faces_np.shape[0], dtype=bool)

        if face_centroids.shape[0] == 0:
            return covered_mask

        keypoints_to_process = [0] if use_primary_keypoint_only else range(num_keypoints)
        sentinel_tensor = torch.tensor(-100.0, device=pred_traj_item_6d_tensor.device, dtype=pred_traj_item_6d_tensor.dtype)

        for k in keypoints_to_process:
            kp_positions = pred_traj_reshaped[:, k, :3]
            valid_mask = ~torch.isclose(kp_positions, sentinel_tensor).all(dim=1)
            valid_positions = kp_positions[valid_mask]
            if valid_positions.shape[0] < 2:
                continue

            spray_path_k = valid_positions.cpu().numpy()
            spray_segments_k = np.stack([spray_path_k[:-1], spray_path_k[1:]], axis=1)

            for seg_start, seg_end in spray_segments_k:
                seg_vec = seg_end - seg_start
                seg_len_sq = np.dot(seg_vec, seg_vec)

                if seg_len_sq == 0:
                    dist_sq = np.sum((face_centroids - seg_start) ** 2, axis=1)
                    covered_mask |= dist_sq <= spray_radius_sq
                    continue

                if use_triangle_distance:
                    for face_idx in range(len(face_centroids)):
                        if covered_mask[face_idx]:
                            continue
                        triangle_vertices = mesh_vertices_np[mesh_faces_np[face_idx]]
                        min_dist_to_triangle = float('inf')
                        for i in range(3):
                            v1 = triangle_vertices[i]
                            v2 = triangle_vertices[(i + 1) % 3]
                            dist = MetricsHandler._segment_to_segment_distance(seg_start, seg_end, v1, v2)
                            min_dist_to_triangle = min(min_dist_to_triangle, dist)
                        for vertex in triangle_vertices:
                            dist = MetricsHandler._point_to_segment_distance(vertex, seg_start, seg_end)
                            min_dist_to_triangle = min(min_dist_to_triangle, dist)
                        if min_dist_to_triangle <= np.sqrt(spray_radius_sq):
                            covered_mask[face_idx] = True
                else:
                    p = face_centroids - seg_start
                    t_param = np.clip((p @ seg_vec) / seg_len_sq, 0.0, 1.0)
                    proj = seg_start + t_param[:, None] * seg_vec
                    dist_sq = np.sum((proj - face_centroids) ** 2, axis=1)
                    covered_mask |= dist_sq <= spray_radius_sq

        return covered_mask

    def _compute_coverage_mask_physics_model(self, pred_traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np, use_primary_keypoint_only=False):
        """
        Returns boolean mask (F,) of covered faces using the same physics-based model as _compute_coverage_physics_model.
        """
        spray_cone_half_angle = np.radians(self.config.get('spray_cone_half_angle_deg', 15.0))
        base_flux = self.config.get('spray_base_flux', 1.0)
        distance_decay_factor = self.config.get('spray_distance_decay', 2.0)
        angle_decay_factor = self.config.get('spray_angle_decay', 1.0)
        min_paint_threshold = self.config.get('spray_min_paint_threshold', 0.1)
        time_step = self.config.get('spray_time_step', 0.01)
        max_spray_distance = self.config.get('spray_max_distance', 0.5)

        if mesh_vertices_np is None or mesh_faces_np is None or mesh_faces_np.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        T_item, D_feat_item = pred_traj_item_6d_tensor.shape
        if D_feat_item % 6 != 0:
            return np.zeros(mesh_faces_np.shape[0], dtype=bool)

        try:
            face_centroids = mesh_vertices_np[mesh_faces_np].mean(axis=1)
            face_normals = self._compute_triangle_normals(mesh_vertices_np, mesh_faces_np)
        except Exception:
            return np.zeros(mesh_faces_np.shape[0], dtype=bool)

        total_faces = len(face_centroids)
        if total_faces == 0:
            return np.zeros(0, dtype=bool)

        accumulated_paint = np.zeros(total_faces)
        num_keypoints = D_feat_item // 6
        keypoints_to_process = [0] if use_primary_keypoint_only else range(num_keypoints)
        sentinel_tensor = torch.tensor(-100.0, device=pred_traj_item_6d_tensor.device, dtype=pred_traj_item_6d_tensor.dtype)

        for k in keypoints_to_process:
            kp_trajectory = pred_traj_item_6d_tensor.reshape(T_item, num_keypoints, 6)[:, k, :]
            valid_mask = ~torch.isclose(kp_trajectory, sentinel_tensor).all(dim=1)
            valid_traj = kp_trajectory[valid_mask]
            if valid_traj.shape[0] < 1:
                continue
            trajectory_np = valid_traj.cpu().numpy()
            for t_idx in range(len(trajectory_np)):
                spray_pos = trajectory_np[t_idx, :3]
                spray_orient = trajectory_np[t_idx, 3:6]
                spray_direction = self._orientation_to_direction_vector(spray_orient)
                for face_idx in range(total_faces):
                    face_center = face_centroids[face_idx]
                    face_normal = face_normals[face_idx]
                    ray_to_face = face_center - spray_pos
                    distance = np.linalg.norm(ray_to_face)
                    if distance > max_spray_distance or distance < 1e-6:
                        continue
                    ray_direction = ray_to_face / distance
                    cos_inclination = np.clip(np.dot(spray_direction, ray_direction), -1.0, 1.0)
                    inclination_angle = np.arccos(cos_inclination)
                    if inclination_angle > spray_cone_half_angle:
                        continue
                    cos_surface_angle = np.clip(np.dot(-spray_direction, face_normal), -1.0, 1.0)
                    surface_angle = np.arccos(abs(cos_surface_angle))
                    # Paint flux
                    distance_factor = 1.0 / (distance ** distance_decay_factor)
                    angle_factor = np.cos(inclination_angle / spray_cone_half_angle * np.pi/2) ** angle_decay_factor
                    surface_factor = abs(np.cos(surface_angle))
                    paint_flux = base_flux * distance_factor * angle_factor * surface_factor
                    paint_flux = max(0.0, paint_flux)
                    accumulated_paint[face_idx] += paint_flux * time_step

        covered_faces_mask = accumulated_paint >= min_paint_threshold
        return covered_faces_mask

    def compute_coverage_mask_for_item(self, traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np):
        """
        Public helper to compute per-face coverage mask using the same config and method as get_coverage.
        """
        spray_radius = self.config.get('coverage_spray_radius', 0.05)
        use_primary_keypoint_only = self.config.get('coverage_use_primary_keypoint_only', False)
        use_triangle_distance = self.config.get('coverage_use_triangle_distance', False)
        use_physics_model = self.config.get('coverage_use_physics_model', False)

        if use_physics_model:
            return self._compute_coverage_mask_physics_model(
                traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np, use_primary_keypoint_only
            )
        else:
            return self._compute_coverage_mask_line_based(
                traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np,
                spray_radius=spray_radius,
                use_primary_keypoint_only=use_primary_keypoint_only,
                use_triangle_distance=use_triangle_distance
            )

    def get_coverage(self, y_pred, batch_data, **kwargs):
        """
        Calculates the average paint coverage percentage over a batch.

        Args:
            y_pred: Predicted trajectory tensor, shape [B, T, D_feat].
            batch_data: Dictionary containing batch information. Expected to contain:
                        'mesh_vertices': List of NumPy arrays (mesh vertices for each batch item).
                        'mesh_faces': List of NumPy arrays (mesh faces for each batch item).
            **kwargs: Additional parameters.

        Returns:
            Scalar float representing the average paint coverage percentage.
        """
        # Check for required mesh data
        if 'mesh_vertices' not in batch_data or 'mesh_faces' not in batch_data:
            print("Warning (get_coverage): Missing 'mesh_vertices' or 'mesh_faces' in batch_data. Returning 0.0.")
            return 0.0

        mesh_vertices = batch_data['mesh_vertices']
        mesh_faces = batch_data['mesh_faces']
        
        # Get spray parameters from config
        spray_radius = self.config.get('coverage_spray_radius', 0.05)
        use_primary_keypoint_only = self.config.get('coverage_use_primary_keypoint_only', False)
        use_triangle_distance = self.config.get('coverage_use_triangle_distance', False)
        use_kdtree_optimization = self.config.get('coverage_use_kdtree', True)  # Enable KD-Tree by default
        
        # NEW: Physical spray cone model parameters
        use_physics_model = self.config.get('coverage_use_physics_model', False)
        
        B = y_pred.shape[0]
        
        # Handle different input formats for mesh data
        if isinstance(mesh_vertices, torch.Tensor):
            # If mesh data is batched tensors, convert to list of numpy arrays
            mesh_vertices = [mesh_vertices[i].cpu().numpy() for i in range(B)]
            mesh_faces = [mesh_faces[i].cpu().numpy() for i in range(B)]
        elif not isinstance(mesh_vertices, (list, tuple)):
            # If single mesh for all batch items - use deep copy to avoid shallow copy issues
            mesh_vertices_np = mesh_vertices.cpu().numpy() if torch.is_tensor(mesh_vertices) else mesh_vertices
            mesh_faces_np = mesh_faces.cpu().numpy() if torch.is_tensor(mesh_faces) else mesh_faces
            mesh_vertices = [mesh_vertices_np.copy() for _ in range(B)]  # Deep copy instead of shallow
            mesh_faces = [mesh_faces_np.copy() for _ in range(B)]

        # Validate batch size consistency
        if len(mesh_vertices) != B or len(mesh_faces) != B:
            print(f"Warning (get_coverage): Batch size mismatch. y_pred B={B}, mesh data length={len(mesh_vertices)}. Returning 0.0.")
            return 0.0

        # Compute coverage for each batch item
        batch_coverage_percentages = []
        for i in range(B):
            try:
                pred_traj_item_tensor = y_pred[i]  # Shape [T, D_feat]
                mesh_vertices_item_np = mesh_vertices[i]
                mesh_faces_item_np = mesh_faces[i]
                
                # Ensure numpy arrays
                if torch.is_tensor(mesh_vertices_item_np):
                    mesh_vertices_item_np = mesh_vertices_item_np.cpu().numpy()
                if torch.is_tensor(mesh_faces_item_np):
                    mesh_faces_item_np = mesh_faces_item_np.cpu().numpy()

                # Choose computation method based on config
                if use_physics_model:
                    # NEW: Use physics-based spray cone model
                    num_covered, total_faces = self._compute_coverage_physics_model(
                        pred_traj_item_tensor, 
                        mesh_vertices_item_np, 
                        mesh_faces_item_np,
                        use_primary_keypoint_only
                    )
                else:
                    # Original geometric method
                    num_covered, total_faces = self._compute_coverage_single_item_line_based(
                        pred_traj_item_tensor, 
                        mesh_vertices_item_np, 
                        mesh_faces_item_np, 
                        spray_radius,
                        use_primary_keypoint_only,
                        use_triangle_distance
                    )
            
                # Calculate coverage percentage
                if total_faces > 0:
                    coverage_perc = (num_covered / total_faces) * 100.0
                    batch_coverage_percentages.append(coverage_perc)
                else:
                    batch_coverage_percentages.append(0.0)
                    
            except Exception as e:
                print(f"Error computing coverage for batch item {i}: {e}")
                batch_coverage_percentages.append(0.0)

        # Compute mean coverage
        if batch_coverage_percentages:
            mean_coverage_percentage = np.mean(batch_coverage_percentages)
        else:
            mean_coverage_percentage = 0.0
            
        return mean_coverage_percentage

    def _compute_coverage_physics_model(self, pred_traj_item_6d_tensor, mesh_vertices_np, mesh_faces_np, use_primary_keypoint_only=False):
        """
        Physics-based spray cone coverage computation with paint deposition simulation.
        
        Args:
            pred_traj_item_6d_tensor: Predicted trajectory for single item, shape [T, D_feat_item]
            mesh_vertices_np: Mesh vertices, NumPy array, shape [V, 3]
            mesh_faces_np: Mesh faces (triangle indices), NumPy array, shape [F, 3]
            use_primary_keypoint_only: If True, only use the first keypoint
            
        Returns:
            Tuple (number of covered faces, total faces)
        """
        # Physics parameters from config (with sensible defaults)
        spray_cone_half_angle = np.radians(self.config.get('spray_cone_half_angle_deg', 15.0))  # 15 degrees
        base_flux = self.config.get('spray_base_flux', 1.0)  # Base paint flux rate
        distance_decay_factor = self.config.get('spray_distance_decay', 2.0)  # Distance decay exponent
        angle_decay_factor = self.config.get('spray_angle_decay', 1.0)  # Angle decay exponent
        min_paint_threshold = self.config.get('spray_min_paint_threshold', 0.1)  # Minimum paint for coverage
        time_step = self.config.get('spray_time_step', 0.01)  # Time step for integration (seconds)
        max_spray_distance = self.config.get('spray_max_distance', 0.5)  # Maximum effective spray distance
        
        # Validate inputs
        if mesh_vertices_np is None or mesh_faces_np is None or mesh_faces_np.shape[0] == 0:
            return 0, 0
            
        T_item, D_feat_item = pred_traj_item_6d_tensor.shape
        if D_feat_item % 6 != 0:
            return 0, mesh_faces_np.shape[0]
            
        num_keypoints = D_feat_item // 6
        pred_traj_reshaped = pred_traj_item_6d_tensor.reshape(T_item, num_keypoints, 6)
        
        # Compute face centroids and normals
        try:
            face_centroids = mesh_vertices_np[mesh_faces_np].mean(axis=1)  # [F, 3]
            face_normals = self._compute_triangle_normals(mesh_vertices_np, mesh_faces_np)  # [F, 3]
        except (IndexError, ValueError) as e:
            print(f"Error computing face properties: {e}")
            return 0, mesh_faces_np.shape[0]
            
        total_faces = len(face_centroids)
        if total_faces == 0:
            return 0, 0
            
        # Initialize paint accumulation array
        accumulated_paint = np.zeros(total_faces)
        
        # Determine which keypoints to process
        keypoints_to_process = [0] if use_primary_keypoint_only else range(num_keypoints)
        
        # Create sentinel tensor for comparison
        sentinel_tensor = torch.tensor(-100.0, device=pred_traj_item_6d_tensor.device, dtype=pred_traj_item_6d_tensor.dtype)
        
        # Process each keypoint's trajectory
        for k in keypoints_to_process:
            # Extract trajectory for this keypoint
            kp_trajectory = pred_traj_reshaped[:, k, :]  # [T_item, 6]
            
            # Filter out padding values
            valid_mask = ~torch.isclose(kp_trajectory, sentinel_tensor).all(dim=1)
            valid_trajectory = kp_trajectory[valid_mask]
            
            if valid_trajectory.shape[0] < 1:
                continue
                
            trajectory_np = valid_trajectory.cpu().numpy()  # [N_valid, 6]
            
            # Time-step integration over trajectory
            for t_idx in range(len(trajectory_np)):
                spray_pos = trajectory_np[t_idx, :3]  # [3] - XYZ position
                spray_orient = trajectory_np[t_idx, 3:6]  # [3] - orientation (rx, ry, rz)
                
                # Convert orientation to spray direction vector
                spray_direction = self._orientation_to_direction_vector(spray_orient)
                
                # Compute paint deposition for each face
                for face_idx in range(total_faces):
                    face_center = face_centroids[face_idx]
                    face_normal = face_normals[face_idx]
                    
                    # Vector from spray gun to face center
                    ray_to_face = face_center - spray_pos
                    distance = np.linalg.norm(ray_to_face)
                    
                    # Skip if too far away
                    if distance > max_spray_distance or distance < 1e-6:
                        continue
                        
                    # Normalize ray direction
                    ray_direction = ray_to_face / distance
                    
                    # Calculate inclination angle (angle between spray direction and ray to face)
                    cos_inclination = np.clip(np.dot(spray_direction, ray_direction), -1.0, 1.0)
                    inclination_angle = np.arccos(cos_inclination)
                    
                    # Check if within spray cone
                    if inclination_angle > spray_cone_half_angle:
                        continue
                        
                    # Calculate surface angle (angle between spray direction and face normal)
                    cos_surface_angle = np.clip(np.dot(-spray_direction, face_normal), -1.0, 1.0)
                    surface_angle = np.arccos(abs(cos_surface_angle))
                    
                    # Physics-based paint deposition calculation
                    paint_flux = self._calculate_paint_flux(
                        base_flux, distance, inclination_angle, surface_angle,
                        spray_cone_half_angle, distance_decay_factor, angle_decay_factor
                    )
                    
                    # Accumulate paint over time step
                    accumulated_paint[face_idx] += paint_flux * time_step
        
        # Determine coverage based on paint threshold
        covered_faces_mask = accumulated_paint >= min_paint_threshold
        num_covered_faces = np.sum(covered_faces_mask)
        
        return int(num_covered_faces), total_faces
    
    @staticmethod
    def _compute_triangle_normals(vertices, faces):
        """Compute normal vectors for triangular faces."""
        # Get triangle vertices
        v0 = vertices[faces[:, 0]]  # [F, 3]
        v1 = vertices[faces[:, 1]]  # [F, 3] 
        v2 = vertices[faces[:, 2]]  # [F, 3]
        
        # Compute edge vectors
        edge1 = v1 - v0  # [F, 3]
        edge2 = v2 - v0  # [F, 3]
        
        # Cross product to get normals
        normals = np.cross(edge1, edge2)  # [F, 3]
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normals = normals / norms
        
        return normals
    
    @staticmethod
    def _orientation_to_direction_vector(orientation):
        """
        Convert orientation (rx, ry, rz) to unit direction vector.
        Assumes orientation represents Euler angles in radians.
        """
        rx, ry, rz = orientation
        
        # Simple conversion - assumes spray direction is along negative Z axis in local frame
        # After rotation by rx, ry, rz
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)  
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        # Rotation matrix (ZYX convention)
        R = np.array([
            [cos_ry * cos_rz, -cos_ry * sin_rz, sin_ry],
            [sin_rx * sin_ry * cos_rz + cos_rx * sin_rz, -sin_rx * sin_ry * sin_rz + cos_rx * cos_rz, -sin_rx * cos_ry],
            [-cos_rx * sin_ry * cos_rz + sin_rx * sin_rz, cos_rx * sin_ry * sin_rz + sin_rx * cos_rz, cos_rx * cos_ry]
        ])
        
        # Local spray direction (negative Z)
        local_direction = np.array([0, 0, -1])
        
        # Transform to world coordinates
        world_direction = R @ local_direction
        
        return world_direction / np.linalg.norm(world_direction)
    
    @staticmethod
    def _calculate_paint_flux(base_flux, distance, inclination_angle, surface_angle, 
                            spray_cone_half_angle, distance_decay_factor, angle_decay_factor):
        """
        Calculate paint flux based on physical spray model.
        
        Args:
            base_flux: Base paint flux rate
            distance: Distance from spray gun to surface
            inclination_angle: Angle between spray direction and ray to surface
            surface_angle: Angle between spray direction and surface normal
            spray_cone_half_angle: Half angle of spray cone
            distance_decay_factor: Distance decay exponent
            angle_decay_factor: Angle decay exponent
            
        Returns:
            Paint flux value
        """
        # Distance attenuation (inverse square law with configurable exponent)
        distance_factor = 1.0 / (distance ** distance_decay_factor)
        
        # Angular attenuation within spray cone
        # Cosine distribution from center to edge of cone
        angle_factor = np.cos(inclination_angle / spray_cone_half_angle * np.pi/2) ** angle_decay_factor
        
        # Surface angle factor (perpendicular surfaces receive more paint)
        surface_factor = abs(np.cos(surface_angle))
        
        # Combine all factors
        paint_flux = base_flux * distance_factor * angle_factor * surface_factor
        
        return max(0.0, paint_flux)  # Ensure non-negative

    @staticmethod
    def _point_to_segment_distance(point, seg_start, seg_end):
        """Compute distance from point to line segment"""
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        
        if seg_len_sq == 0:
            return np.linalg.norm(point - seg_start)
        
        t = np.clip(np.dot(point - seg_start, seg_vec) / seg_len_sq, 0.0, 1.0)
        projection = seg_start + t * seg_vec
        
        return np.linalg.norm(point - projection)

    @staticmethod
    def _segment_to_segment_distance(seg1_start, seg1_end, seg2_start, seg2_end):
        """Compute minimum distance between two line segments (simplified version)"""
        # This is a simplified implementation - full 3D segment-to-segment distance is complex
        # For spray painting, checking endpoints and midpoints is often sufficient
        
        points_to_check = [
            (seg1_start, seg2_start, seg2_end),
            (seg1_end, seg2_start, seg2_end),
            (seg2_start, seg1_start, seg1_end),
            (seg2_end, seg1_start, seg1_end),
        ]
        
        min_dist = float('inf')
        for point, line_start, line_end in points_to_check:
            dist = MetricsHandler._point_to_segment_distance(point, line_start, line_end)
            min_dist = min(min_dist, dist)
        
        return min_dist

    def get_inference_time(self, **kwargs):
        """
        Calculate model inference time per sample.
        
        Args:
            **kwargs: Should contain 'inference_times' - list of inference times in seconds per sample
        
        Returns:
            Scalar float representing mean inference time in milliseconds
        """
        inference_times = kwargs.get('inference_times', None)
        
        if inference_times is None:
            print("Warning (get_inference_time): 'inference_times' not provided in kwargs. Returning 0.0.")
            return 0.0
            
        if len(inference_times) == 0:
            return 0.0
            
        # Convert to milliseconds and return mean
        inference_times_ms = [t * 1000.0 for t in inference_times]
        return np.mean(inference_times_ms)

    def get_latency(self, **kwargs):
        """
        Calculate end-to-end latency including preprocessing, inference, and postprocessing.
        
        Args:
            **kwargs: Should contain 'latencies' - list of total latencies in seconds per sample
        
        Returns:
            Scalar float representing mean latency in milliseconds
        """
        latencies = kwargs.get('latencies', None)
        
        if latencies is None:
            print("Warning (get_latency): 'latencies' not provided in kwargs. Returning 0.0.")
            return 0.0
            
        if len(latencies) == 0:
            return 0.0
            
        # Convert to milliseconds and return mean
        latencies_ms = [t * 1000.0 for t in latencies]
        return np.mean(latencies_ms)