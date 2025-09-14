import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import open3d as o3d

# Ensure Axes3D is imported for subplot projection='3d'
from mpl_toolkits.mplot3d import Axes3D 

def visualize_traj(traj_pred, traj_gt, save_path):
    """
    Visualize predicted and ground truth trajectories. Saves comparison PNG and PLY files.
    Args:
        traj_pred: Tensor or array of shape [T, D] - predicted trajectory
        traj_gt: Tensor or array of shape [T, D] - ground truth trajectory
        save_path: Path prefix/directory to save the visualization files. 
                   Files saved: 'trajectory_comparison.png', 'pred.ply', 'gt.ply'
    """
    # Convert to numpy if tensors
    if torch.is_tensor(traj_pred):
        traj_pred = traj_pred.detach().cpu().numpy()
    if torch.is_tensor(traj_gt):
        traj_gt = traj_gt.detach().cpu().numpy()

    # Reshape if necessary (assumes D divisible by 6 for 6D points)
    dims_per_point = 6 
    # Check if dimensions are sufficient before accessing shape[-1]
    if traj_pred.ndim < 2:
        print(f"Warning in visualize_traj: Predicted trajectory has insufficient dimensions {traj_pred.ndim}")
        return
    if traj_gt.ndim < 2:
        print(f"Warning in visualize_traj: Ground truth trajectory has insufficient dimensions {traj_gt.ndim}")
        return
        
    num_points_per_step = traj_pred.shape[-1] // dims_per_point
    if traj_pred.shape[-1] % dims_per_point != 0:
        print(f"Warning in visualize_traj: Trajectory dimension {traj_pred.shape[-1]} is not divisible by {dims_per_point}. Reshaping might be incorrect.")

    # Extract first 3D coordinates (assuming x,y,z are the first 3 dims of each 6D point)
    # Reshape to [T * num_points_per_step, 3]
    pos_pred = traj_pred.reshape(-1, dims_per_point)[:, :3]
    pos_gt_raw = traj_gt.reshape(-1, dims_per_point)[:, :3]

    # Handle padding in ground truth
    mask = ~np.all(pos_gt_raw == -100, axis=1)
    pos_gt = pos_gt_raw[mask]

    if pos_gt.shape[0] == 0 and pos_pred.shape[0] > 0:
        pos_gt = np.zeros((1,3)) # Placeholder if GT is all padding but pred exists
    elif pos_gt.shape[0] == 0 and pos_pred.shape[0] == 0:
         print("Warning in visualize_traj: Both predicted and GT trajectories are empty or all padding.")
         return # Nothing to visualize


    # Center based on combined valid points for consistent view
    all_valid_pos = []
    if pos_pred.shape[0] > 0: all_valid_pos.append(pos_pred)
    if pos_gt.shape[0] > 0: all_valid_pos.append(pos_gt)
    
    if not all_valid_pos:
        print("Warning in visualize_traj: No valid points found after processing.")
        return

    all_valid_pos_np = np.vstack(all_valid_pos)
    center = all_valid_pos_np.mean(axis=0)
    
    # Apply centering
    pos_pred_centered = pos_pred - center if pos_pred.shape[0] > 0 else np.empty((0,3))
    pos_gt_centered = pos_gt - center if pos_gt.shape[0] > 0 else np.empty((0,3))
            
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True) # save_path is now treated as directory
    
    # --- Save PNG ---
    fname_png = os.path.join(save_path, 'trajectory_comparison.png')
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    if pos_pred_centered.shape[0] > 0: ax.scatter(pos_pred_centered[:,0], pos_pred_centered[:,1], pos_pred_centered[:,2], s=1, label='Pred', c='red')
    if pos_gt_centered.shape[0] > 0: ax.scatter(pos_gt_centered[:,0], pos_gt_centered[:,1], pos_gt_centered[:,2], s=1, label='GT', c='blue')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); ax.set_title('Trajectory Comparison (Centered)')
    # Auto-scaling axes based on centered data
    plt.savefig(fname_png, dpi=300)
    print(f"    Saved comparison image: {fname_png}")
    plt.close(fig)

    # --- Save PLY Files ---
    # Save non-centered points
    pos_pred_orig = traj_pred.reshape(-1, dims_per_point)[:, :3]
    pos_gt_valid_orig = pos_gt_raw[mask] # Use original coordinates of valid points
    
    if pos_pred_orig.shape[0] > 0:
        pcd_p = o3d.geometry.PointCloud()
        pcd_p.points = o3d.utility.Vector3dVector(pos_pred_orig)
        pcd_p.paint_uniform_color([1, 0, 0]) # Red for predicted
        fname_ply_pred = os.path.join(save_path, 'pred.ply')
        o3d.io.write_point_cloud(fname_ply_pred, pcd_p)
        print(f"    Saved predicted PLY: {fname_ply_pred}")

    if pos_gt_valid_orig.shape[0] > 0:
        pcd_g = o3d.geometry.PointCloud()
        pcd_g.points = o3d.utility.Vector3dVector(pos_gt_valid_orig)
        pcd_g.paint_uniform_color([0, 1, 0]) # Green for ground truth
        fname_ply_gt = os.path.join(save_path, 'gt.ply')
        o3d.io.write_point_cloud(fname_ply_gt, pcd_g)
        print(f"    Saved ground truth PLY: {fname_ply_gt}")

    return save_path


def _save_trajectory_frame(pred_so_far, gt_so_far, frame_save_path, overall_gt_bounds=None, title_suffix=""):
    """
    Saves a single frame of the trajectory comparison for GIF creation (Runner version).
    Args:
        pred_so_far: Predicted trajectory tensor up to current step [1, current_T, D]
        gt_so_far: Ground truth trajectory tensor up to current step [1, current_T, D]
        frame_save_path: Full path to save the .png frame.
        overall_gt_bounds: Optional tuple (xlim, ylim, zlim) for consistent axis scaling.
        title_suffix: Suffix for the plot title.
    """
    # Helper to extract points (kept local)
    def extract_xyz_anim(traj_segment):
        if traj_segment.shape[0] == 0: return np.empty((0,3))
        xyz_points_loc = []
        dims_per_point = 6 ; num_points_to_vis = 4
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
        
    # Initial empty frame handling
    if pred_so_far.shape[1] == 0 and gt_so_far.shape[1] == 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'Trajectory Generation {title_suffix} (waiting for data)')
        if overall_gt_bounds: ax.set_xlim(overall_gt_bounds[0]); ax.set_ylim(overall_gt_bounds[1]); ax.set_zlim(overall_gt_bounds[2])
        else: ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        plt.savefig(frame_save_path); print(f"    Saved animation frame: {frame_save_path}"); plt.close(fig)
        return frame_save_path
        
    # Process non-empty frames
    pred_np = pred_so_far.detach().cpu().numpy()[0]; gt_np = gt_so_far.detach().cpu().numpy()[0]
    T_pred, T_gt = pred_np.shape[0], gt_np.shape[0]
    # Masking based on GT if lengths match
    if T_pred == T_gt: 
        gt_mask_flat_anim = ~np.all(gt_np == -100, axis=1).reshape(T_gt, 1)
        gt_mask_expanded_anim = np.repeat(gt_mask_flat_anim, pred_np.shape[1], axis=1)
        pred_np_masked_vis = np.where(gt_mask_expanded_anim, pred_np, -100.0)
    else: 
        pred_np_masked_vis = pred_np
        
    # Extract points
    pred_points_frame = extract_xyz_anim(pred_np_masked_vis); gt_points_frame = extract_xyz_anim(gt_np)
    # Filter points
    pred_valid_mask_fr = ~np.all(pred_points_frame == -100.0, axis=1); gt_valid_mask_fr = ~np.all(gt_points_frame == -100.0, axis=1)
    pred_points_frame = pred_points_frame[pred_valid_mask_fr]; gt_points_frame = gt_points_frame[gt_valid_mask_fr]
    # Plotting
    fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
    if pred_points_frame.shape[0] > 0: ax.scatter(pred_points_frame[:, 0], pred_points_frame[:, 1], pred_points_frame[:, 2], c='red', marker='o', label='Predicted', s=15, alpha=0.7)
    if gt_points_frame.shape[0] > 0: ax.scatter(gt_points_frame[:, 0], gt_points_frame[:, 1], gt_points_frame[:, 2], c='blue', marker='^', label='Ground Truth', s=15, alpha=0.7)
    # Set axis limits
    if overall_gt_bounds: ax.set_xlim(overall_gt_bounds[0]); ax.set_ylim(overall_gt_bounds[1]); ax.set_zlim(overall_gt_bounds[2])
    else: 
        all_pts_for_lims = []; 
        if pred_points_frame.shape[0] > 0: all_pts_for_lims.append(pred_points_frame)
        if gt_points_frame.shape[0] > 0: all_pts_for_lims.append(gt_points_frame)
        if all_pts_for_lims: 
            all_pts_for_lims_np = np.vstack(all_pts_for_lims); 
            ax.set_xlim(all_pts_for_lims_np[:,0].min()-0.1, all_pts_for_lims_np[:,0].max()+0.1); 
            ax.set_ylim(all_pts_for_lims_np[:,1].min()-0.1, all_pts_for_lims_np[:,1].max()+0.1); 
            ax.set_zlim(all_pts_for_lims_np[:,2].min()-0.1, all_pts_for_lims_np[:,2].max()+0.1)
        else: ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title(f'Trajectory Generation {title_suffix} (Step {pred_so_far.shape[1]})'); ax.legend()
    plt.savefig(frame_save_path); print(f"    Saved animation frame: {frame_save_path}"); plt.close(fig)
    # Save Frame PLY Files
    if pred_points_frame.shape[0] > 0:
        pcd_p = o3d.geometry.PointCloud(); pcd_p.points = o3d.utility.Vector3dVector(pred_points_frame); pcd_p.paint_uniform_color([1,0,0]); ply_pred_path = frame_save_path.replace('.png', '_pred.ply'); o3d.io.write_point_cloud(ply_pred_path, pcd_p); print(f"    Saved pred PLY for frame: {ply_pred_path}")
    if gt_points_frame.shape[0] > 0:
        pcd_g = o3d.geometry.PointCloud(); pcd_g.points = o3d.utility.Vector3dVector(gt_points_frame); pcd_g.paint_uniform_color([0,1,0]); ply_gt_path = frame_save_path.replace('.png', '_gt.ply'); o3d.io.write_point_cloud(ply_gt_path, pcd_g); print(f"    Saved GT PLY for frame: {ply_gt_path}")
    return frame_save_path


def save_training_gif_frame(pred_so_far, full_gt, frame_save_path, overall_gt_bounds=None, title_info=""):
    """
    Saves a single frame for a training visualization GIF.
    Plots the complete ground truth trajectory (static) and the predicted trajectory
    accumulated up to the current step.

    Args:
        pred_so_far: Predicted trajectory tensor up to current step [1, current_T_pred, D]
        full_gt: COMPLETE ground truth trajectory tensor [1, T_gt, D]
        frame_save_path: Full path to save the .png frame.
        overall_gt_bounds: Optional tuple (xlim, ylim, zlim) for consistent axis scaling based on full GT.
        title_info: Info string for the plot title.
    """
    # Helper to extract points (can be shared or redefined)
    def extract_xyz(traj_tensor):
        if traj_tensor is None or traj_tensor.numel() == 0:
            return np.empty((0, 3))
        traj_np = traj_tensor.detach().cpu().numpy()[0] # [T, D]
        if traj_np.shape[0] == 0: return np.empty((0,3))
        xyz_points_loc = []
        dims_per_point = 6 ; num_points_to_vis = 4
        for i in range(num_points_to_vis):
            if (i * dims_per_point + 2) < traj_np.shape[1]:
                x = traj_np[:, i * dims_per_point + 0]
                y = traj_np[:, i * dims_per_point + 1]
                z = traj_np[:, i * dims_per_point + 2]
                pts = np.stack([x, y, z], axis=1)
                xyz_points_loc.append(pts)
            else: break
        if not xyz_points_loc: return np.empty((0,3))
        return np.concatenate(xyz_points_loc, axis=0)

    pred_points = extract_xyz(pred_so_far)
    gt_points_full = extract_xyz(full_gt)

    # Filter padding from GT points for visualization
    gt_mask = ~np.all(gt_points_full == -100.0, axis=1) if gt_points_full.size > 0 else np.array([], dtype=bool)
    gt_points_valid = gt_points_full[gt_mask]

    # Filter padding from Predicted points (though ideally shouldn't be there)
    pred_mask = ~np.all(pred_points == -100.0, axis=1) if pred_points.size > 0 else np.array([], dtype=bool)
    pred_points_valid = pred_points[pred_mask]

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot full GT (static, blue, less prominent)
    if gt_points_valid.shape[0] > 0:
        ax.scatter(gt_points_valid[:, 0], gt_points_valid[:, 1], gt_points_valid[:, 2],
                   c='blue', marker='^', label='Full Ground Truth', s=10, alpha=0.4)

    # Plot cumulative prediction (growing, red, more prominent)
    if pred_points_valid.shape[0] > 0:
        ax.scatter(pred_points_valid[:, 0], pred_points_valid[:, 1], pred_points_valid[:, 2],
                   c='red', marker='o', label=f'Prediction (Step {pred_so_far.shape[1]})', s=15, alpha=0.8)

    # Set axis limits (use pre-calculated bounds for consistency)
    if overall_gt_bounds:
        ax.set_xlim(overall_gt_bounds[0]); ax.set_ylim(overall_gt_bounds[1]); ax.set_zlim(overall_gt_bounds[2])
    else: # Fallback if bounds failed
         ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'Training Sample Prediction Generation\n{title_info}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(frame_save_path)
    print(f"    Saved training GIF frame: {frame_save_path}")
    plt.close(fig)

    return frame_save_path 