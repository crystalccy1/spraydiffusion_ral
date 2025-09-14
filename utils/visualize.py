import pdb
import random
import os
from tqdm import tqdm
from threadpoolctl import ThreadpoolController

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pyvista as pv
import torch
from datetime import datetime

from . import orient_in, rot_from_representation
from .pointcloud import from_pc_to_seq, from_seq_to_pc, get_dim_traj_points, get_traj_feature_index, remove_padding, remove_padding_v2, get_mean_mesh, from_bbox_encoding_to_visual_format
from .disk import get_dataset_paths, get_dataset_name, get_dataset_downscale_factor
from . import get_root_of_dir

controller = ThreadpoolController()
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"

def convert_tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor

def create_animated_mesh_trajectory_visualization(meshfile,
                        traj,
                        plotter=None,
                        index=None,
                        text=None,
                        # trajc='lightblue',
                        trajc='blue',
                        trajvel=False,
                        lambda_points=1,
                        camera=None,
                        extra_data=[],
                        stroke_ids=None,
                        cmap=None,
                        arrow_color=None,
                        tour=None):
    """Visualize mesh-traj pair

        meshfile: str
                  mesh filename.objr
        traj : (N,k) array
        
        lambda_points: traj is set of sequences of lambda_points
    """
    curr_traj = traj.copy()
    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is not None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    mesh_obj = pv.read(meshfile)
    plotter.add_mesh(mesh_obj)
    
    
    if camera is not None:
        plotter.set_position(camera)

    if text is not None:
        plotter.add_text(text)

    if torch.is_tensor(curr_traj):
        curr_traj = curr_traj.cpu().detach().numpy()

    if lambda_points > 1:
        outdim = get_dim_traj_points(extra_data)
        assert curr_traj.shape[-1]%outdim == 0
        curr_traj = curr_traj.reshape(-1, outdim)
        curr_traj = remove_padding(curr_traj, extra_data)  # makes sense only if it's GT traj, but doesn't hurt
        if stroke_ids is not None:
            if stroke_ids.shape[0] != curr_traj.shape[0]:
                stroke_ids = stroke_ids[:curr_traj.shape[0]//lambda_points, None] # remove padding from stroke_ids
                stroke_ids = np.repeat(stroke_ids, lambda_points) # stroke_ids from sequence to point
            if tour is not None:
                tour = np.repeat(tour, lambda_points)
            assert (stroke_ids != -1).all()
            assert (stroke_ids.shape[0] == curr_traj.shape[0]), f"{stroke_ids.shape}, {curr_traj.shape}"

    traj_pc = pv.PolyData(curr_traj[:, :3])

    curr_traj_subdivided = []
    if stroke_ids is not None:
        stroke_ids_unique = np.unique(stroke_ids)
        for i in stroke_ids_unique:
            indexes = np.where(stroke_ids == i)[0]
            curr_traj_subdivided.append(indexes) # create a list with points grouped by stroke id

    cmaps =  ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    get_cmap = lambda i: cmaps[i]  if i < len(cmaps) else random.choice(cmaps)

    for i, curr_traj_stroke_indexes in tqdm(enumerate(curr_traj_subdivided), desc="Strokes"):
        plotter.subplot(*index)
        points_to_render = curr_traj[curr_traj_stroke_indexes, :3].copy()
        mesh = pv.PointSet(points_to_render.copy())
        mesh.points[:] = np.nan 
        scalars = tour[curr_traj_stroke_indexes] if tour is not None else np.arange(curr_traj_stroke_indexes.shape[0])
        scalars = scalars/scalars.max() # normalize to 1
        plotter.add_mesh(mesh, scalars=scalars, point_size=14.0, opacity=1.0, render_points_as_spheres=True, cmap=get_cmap(i))
        sorted_indexes = np.argsort(scalars)
        # for j in range(1, points_to_render.shape[0]//lambda_points+2):
        #     plotter.subplot(*index)
        #     render_index = min(j*lambda_points, points_to_render.shape[0])
        #     mesh.points[render_index-lambda_points:render_index] = points_to_render[render_index-lambda_points:render_index] 
        #     yield
        for j in range(1, points_to_render.shape[0]//lambda_points + 2):
            plotter.subplot(*index)
            render_index = min(j*lambda_points, points_to_render.shape[0])
            mesh.points[sorted_indexes[render_index-lambda_points:render_index]] = points_to_render[sorted_indexes[render_index-lambda_points:render_index]]
            yield

    if trajvel:
        assert 'vel' in extra_data, 'Cannot display traj velocity: trajectory does not contain velocities'
        plotter.add_arrows(curr_traj[:, :3], curr_traj[:, 3:6], mag=1, color='green', opacity=0.8)

    if orient_in(extra_data)[0]:
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)

        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, curr_traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor

        if arrow_color is None:
            arrow_color = 'red'
            
        plotter.add_arrows(curr_traj[:, :3]-e1_rots, e1_rots, mag=1, color=arrow_color, opacity=0.8)

    if show_plot:
        plotter.show_axes()
        plotter.show()
    
    ## save plotter
    # plotter.screenshot(os.path.join(save_dir, filename if filename != '' else 'mesh_traj.png'))
    plotter.screenshot('mesh_traj.png')
    return

def create_multiview_mesh_trajectory_visualization(
    mesh_vertices_np,
    mesh_faces_np,
    traj,
    config,
    save_path=None,
    point_size=14.0,
    trajc='lightgreen',
    offset=0.12,
    # Multi-view parameters
    views=None,
    n_views=6,
    n_cols=3,
    elevation_deg=20.0,
    radius_scale=10.0,
    # Other parameters
    background="white",
    title=None,
    save_individual=True,
    save_collage=True,
    auto_zoom=True,
    # New parameters: dataset name and checkpoint name
    dataset_name=None,
    checkpoint_name=None
):
    """
    Multi-view visualization of trajectory and mesh (PyVista):
    - Display mesh and trajectory points
    - Support automatic/custom camera; camera distance controlled by radius_scale
    - New auto_zoom parameter automatically adjusts view to ensure complete display
    - Headless off-screen rendering
    - Output: individual images for each view saved in subfolder; optionally export multi-view collage
    - New: support for including dataset name and checkpoint name in filenames
    Returns: (individual_dir, collage_path or None)
    """
    import os
    import numpy as np
    import pyvista as pv
    from datetime import datetime
    
    # --- Headless environment ---
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    if "DISPLAY" not in os.environ:
        try:
            pv.start_xvfb()
        except OSError:
            pass

    # --- Data validation/preparation ---
    verts = np.asarray(mesh_vertices_np, dtype=np.float32)
    faces = np.asarray(mesh_faces_np, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh_faces_np must be (F,3) triangular face indices")
    F = faces.shape[0]
    faces_flat = np.hstack([np.full((F, 1), 3, dtype=np.int64), faces]).ravel()

    # --- Trajectory processing ---
    import torch
    if torch.is_tensor(traj):
        traj_np = traj.detach().cpu().numpy()
    else:
        traj_np = np.asarray(traj)
    traj_np = traj_np.reshape(-1, 6)

    # Translate trajectory points based on orientation
    pos = traj_np[:, :3].copy()
    rot = traj_np[:, 3:]
    dir_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    mask = dir_norm.squeeze() > 1e-8
    dir_unit = np.zeros_like(rot)
    dir_unit[mask] = rot[mask] / dir_norm[mask]
    pos[mask] += offset * dir_unit[mask]

    # --- View generation ---
    center = verts.mean(axis=0)
    if views is None:
        # Calculate bounding box to better determine camera distance
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_diagonal = np.linalg.norm(bbox_size)
        
        # Use larger coefficient to ensure complete display
        dist = max(bbox_diagonal * radius_scale, 1e-3)
        
        azis = np.linspace(0.0, 360.0, int(n_views), endpoint=False)
        elev = np.deg2rad(elevation_deg)
        
        views = []
        for azi in azis:
            azi_rad = np.deg2rad(azi)
            # Convert spherical coordinates to Cartesian coordinates
            x = center[0] + dist * np.cos(elev) * np.cos(azi_rad)
            y = center[1] + dist * np.cos(elev) * np.sin(azi_rad)
            z = center[2] + dist * np.sin(elev)
            
            views.append({
                "position": (x, y, z),
                "focal_point": tuple(center),
                "viewup": (0.0, 0.0, 1.0)
            })

    # --- Output path processing ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name_parts = ["trajectory_multiview"]
    
    if dataset_name:
        # Clean dataset name, remove special characters
        clean_dataset_name = dataset_name.replace('-', '_').replace('/', '_').replace(' ', '_')
        dir_name_parts.append(clean_dataset_name)
    
    if checkpoint_name:
        # Clean checkpoint name, remove special characters
        clean_checkpoint_name = checkpoint_name.replace('-', '_').replace('/', '_').replace(' ', '_')
        dir_name_parts.append(clean_checkpoint_name)
    
    # Add timestamp
    dir_name_parts.append(timestamp)
    
    # Combine directory name
    dir_name = "_".join(dir_name_parts)
    
    if save_path is None:
        base_root = os.path.join("temp", "vis")
        os.makedirs(base_root, exist_ok=True)
        individual_dir = os.path.join(base_root, dir_name)
    else:
        # If directory or ends with separator → create subfolder directly inside
        if os.path.isdir(save_path) or str(save_path).endswith(os.sep):
            base_root = str(save_path).rstrip(os.sep)
            os.makedirs(base_root, exist_ok=True)
            individual_dir = os.path.join(base_root, dir_name)
        else:
            # File path passed → create subfolder at same level
            base_root = os.path.dirname(save_path) or os.path.join("temp", "vis")
            os.makedirs(base_root, exist_ok=True)
            individual_dir = os.path.join(base_root, dir_name)

    os.makedirs(individual_dir, exist_ok=True)
    collage_path = None

    # --- Pre-construct mesh data ---
    base_mesh = pv.PolyData(verts, faces_flat)

    # --- Render each view and save individual images ---
    if save_individual:
        for i, v in enumerate(views):
            pl = pv.Plotter(off_screen=True, window_size=(1280, 960))
            pl.set_background(background)
            pl.add_mesh(base_mesh, color='lightgrey', opacity=0.7)
            pl.add_points(pos, color=trajc, render_points_as_spheres=True, point_size=point_size)
            if title and i == 0:
                pl.add_text(title, position="upper_left", font_size=16)

            pos_cam = v.get("position")
            foc = v.get("focal_point", center)
            up = v.get("viewup", (0.0, 0.0, 1.0))
            if pos_cam is not None:
                pl.camera_position = (tuple(pos_cam), tuple(foc), tuple(up))

            # Automatically adjust zoom to ensure complete display
            if auto_zoom:
                pl.reset_camera()
                pl.camera.zoom(0.6)

            pl.show(auto_close=False)
            view_path = os.path.join(individual_dir, f"view_{i:02d}.png")
            pl.screenshot(view_path)
            pl.close()

    # --- Additional collage export ---
    if save_collage:
        n_views_eff = len(views)
        n_cols_eff = max(1, int(n_cols))
        n_rows_eff = int(np.ceil(n_views_eff / n_cols_eff))

        pl = pv.Plotter(off_screen=True, shape=(n_rows_eff, n_cols_eff), window_size=(1920, 1080))
        pl.set_background(background)

        for i, v in enumerate(views):
            r, c = divmod(i, n_cols_eff)
            pl.subplot(r, c)
            pl.add_mesh(base_mesh, color='lightgrey', opacity=0.7)
            pl.add_points(pos, color=trajc, render_points_as_spheres=True, point_size=point_size)
            if title and i == 0:
                pl.add_text(title, position="upper_left", font_size=16)

            pos_cam = v.get("position")
            foc = v.get("focal_point", center)
            up = v.get("viewup", (0.0, 0.0, 1.0))
            if pos_cam is not None:
                pl.camera_position = (tuple(pos_cam), tuple(foc), tuple(up))
                
            # Automatically adjust zoom to ensure complete display
            if auto_zoom:
                pl.reset_camera()
                pl.camera.zoom(0.6)

        collage_path = os.path.join(individual_dir, "collage.png")
        pl.screenshot(collage_path)
        pl.close()

    # If save_path is specified and not a directory, copy collage to specified location
    if save_path and not os.path.isdir(save_path) and collage_path:
        import shutil
        shutil.copy2(collage_path, save_path)
        print(f"Saved trajectory multiview collage → {save_path}")

    return individual_dir, collage_path

def create_mesh_coverage_visualization(
    mesh_vertices_np,
    mesh_faces_np,
    covered_faces_mask,
    face_colors_original=None,
    traj=None,
    traj_point_size=12.0,
    traj_color='deepskyblue',
    offset=0.12,
    camera=None,
    save_path=None,
    title=None,
    auto_zoom=True  # New: Automatically adjust zoom to ensure complete display
):
    """
    Visualize mesh coverage using PyVista:
    - Covered faces colored yellow (1.0, 1.0, 0.0)
    - Uncovered faces keep original color: from face_colors_original (per-face RGB), or light gray if not provided
    - Optional trajectory points overlay (offset along orientation)
    - New auto_zoom parameter automatically adjusts view to ensure complete display
    - Support for headless off-screen rendering and save_path normalization (same logic as visualize_mesh_traj_v4)
    """
    # Headless environment setup
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
    if "DISPLAY" not in os.environ:
        try:
            pv.start_xvfb()
        except OSError:
            pass

    # Data validation and preparation
    verts = np.asarray(mesh_vertices_np, dtype=np.float32)
    faces = np.asarray(mesh_faces_np, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh_faces_np must be (F,3) triangular face indices")
    F = faces.shape[0]
    faces_flat = np.hstack([np.full((F, 1), 3, dtype=np.int64), faces]).ravel()

    covered_faces_mask = np.asarray(covered_faces_mask, dtype=bool)
    if covered_faces_mask.shape[0] != F:
        raise ValueError(f"covered_faces_mask length should be {F}, not {covered_faces_mask.shape[0]}")

    if face_colors_original is not None:
        base_colors = np.asarray(face_colors_original, dtype=float)
        if base_colors.shape != (F, 3):
            raise ValueError("face_colors_original should be (F,3) RGB array, range 0-1 or 0-255")
        if base_colors.max() > 1.0:
            base_colors = base_colors / 255.0
    else:
        base_colors = np.full((F, 3), 0.85, dtype=float)  # Light gray background

    # Apply coverage coloring: covered -> yellow
    colors_to_apply = base_colors.copy()
    colors_to_apply[covered_faces_mask] = np.array([1.0, 1.0, 0.0])  # Yellow

    # Create Plotter and add mesh
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.subplot(0, 0)

    mesh = pv.PolyData(verts, faces_flat)
    mesh.cell_data['colors'] = colors_to_apply
    plotter.add_mesh(mesh, scalars='colors', rgb=True, show_scalar_bar=False)

    # Camera and title
    if camera is not None:
        plotter.set_position(camera)
    if title:
        plotter.add_text(title, position='upper_left', font_size=16)

    # Optional trajectory overlay
    if traj is not None:
        traj_np = traj.detach().cpu().numpy() if torch.is_tensor(traj) else np.asarray(traj)
        traj_np = traj_np.reshape(-1, 6)
        pos = traj_np[:, :3].copy()
        rot = traj_np[:, 3:]
        dir_norm = np.linalg.norm(rot, axis=1, keepdims=True)
        mask = dir_norm.squeeze() > 1e-8
        dir_unit = np.zeros_like(rot)
        dir_unit[mask] = rot[mask] / dir_norm[mask]
        pos[mask] += offset * dir_unit[mask]
        plotter.add_points(pos, color=traj_color, render_points_as_spheres=True, point_size=traj_point_size)

    # More aggressive automatic zoom adjustment to ensure complete display
    if auto_zoom and camera is None:
        plotter.reset_camera()
        # Further reduce to ensure complete display
        plotter.camera.zoom(0.6)  # Changed from 0.8 to 0.6, smaller zoom ratio

    # Normalize save path (consistent with v4)
    if save_path is None:
        base_dir = os.path.join("temp", "vis")
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(base_dir, f"coverage_{timestamp}.png")
    else:
        if os.path.isdir(save_path) or save_path.endswith(os.sep):
            base_dir = save_path.rstrip(os.sep)
            os.makedirs(base_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(base_dir, f"coverage_{timestamp}.png")
        else:
            base_dir = os.path.dirname(save_path) or os.path.join("temp", "vis")
            os.makedirs(base_dir, exist_ok=True)
            if not os.path.splitext(save_path)[1]:
                save_path = save_path + ".png"

    plotter.screenshot(save_path)
    plotter.close()
    return save_path
