"""
test.py

Example usage:
    python test_v2.py \
        --dataset cuboids-v1 \
        --pc_points 5120 \
        --traj_points 500 \
        --backbone pointnet2 \
        --ckpt /usr/stud/dira/ccy/paintnet/pretrained_models/pointnet2_cls_ssg.pth

Features:
    - Load PaintNet dataset (test split)
    - Load pretrained model
    - Run inference on test set and compute losses, metrics
    - Save inference results (ground truth & predicted trajectories) to .npy files (optional)
    - Visualize results (point clouds and trajectories)
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add or remove logging/wandb imports as needed
import wandb

# ========== Custom imports, modify according to actual code location ========== #
from paintnet_utils import get_random_string, set_seed, create_dirs
from paintnet_utils import get_train_config, get_dataset_path, save_config
from paintnet_loader import PaintNetDataloader
from model_utils import get_model, init_from_pretrained
from loss_handler import LossHandler
from metrics_handler import MetricsHandler
from vis_utils import visualize_point_cloud, visualize_traj

# Set default environment variable if not already set
if os.environ.get("PAINTNET_ROOT_V1") is None:
    os.environ["PAINTNET_ROOT_V1"] = "/usr/stud/dira/ccy/paintnet/datasets"
    print(f"Setting PAINTNET_ROOT_V1 to: {os.environ['PAINTNET_ROOT_V1']}")

def parse_args():
    parser = argparse.ArgumentParser("Testing script for a pretrained painting-trajectory model")
    # Dataset related
    parser.add_argument('--dataset',      default='cuboids-v1', type=str,
                        help='Dataset name [containers-v2, windows-v1, shelves-v1, cuboids-v1]')
    parser.add_argument('--pc_points',    default=5120, type=int,
                        help='Number of points to sub-sample for each point-cloud')
    parser.add_argument('--traj_points',  default=2000, type=int,
                        help='Number of points to sub-sample for each trajectory')
    parser.add_argument('--normalization', default='per-dataset', type=str,
                        help='Normalization type (per-dataset, per-mesh, none)')
    parser.add_argument('--extra_data',   default=['orientnorm'], type=str, nargs='+',
                        help="List of extra data modalities [vel, orientquat, orientrotvec, orientnorm]")
    parser.add_argument('--lambda_points',  default=4, type=int, help='Traj is considered as point-cloud made of vectors of <lambda> ordered points (Default=1, meaning that'\
                                                                     'chamfer distance would be computed normally on each traj point)')
    parser.add_argument('--overlapping',    default=0, type=int, help='Number of overlapping points between subsequent mini-sequences (only valid when lambda_points > 1)')
    # Model related
    parser.add_argument('--backbone',   default='pointnet2', type=str, help='Model backbone.')
    parser.add_argument('--ckpt',       default='/usr/stud/dira/ccy/paintnet/runs/BGDCG/best_model.pth', type=str,
                        help='Path to pretrained checkpoint (.pth)')
    parser.add_argument('--min_centroids',  default=False, action='store_true', help='Whether to compute chamfer distance on mini-sequences with centroids only')


    # Dataloader
    parser.add_argument('--batch_size', '-bs', default=2, type=int)
    parser.add_argument('--workers',    default=4, type=int)

    # Test losses/metrics
    parser.add_argument('--loss',       default=['chamfer'], type=str, nargs='+',
                        help='Which loss(es) to compute during test [chamfer, repulsion, mse, ...]')
    parser.add_argument('--eval_metrics', default=['pcd'], type=str, nargs='+',
                        help='Which metrics to compute [pcd, ...]')
    parser.add_argument('--weight_orient', default=0.25, type=float,
                        help='Weight for orientation loss vs positional loss')

    # Other options
    parser.add_argument('--output_dir', default='test_runs', type=str,
                        help='Directory for saving test predictions/results')
    parser.add_argument('--config',     default=None, type=str,
                        help='Name of .json file in configs/ to load additional settings')
    parser.add_argument('--no_save',    default=False, action='store_true',
                        help="Don't save .npy predictions")
    parser.add_argument('--seed',       default=0, type=int, help='Random seed (0 means not fixed)')
    parser.add_argument('--debug',      default=False, action='store_true',
                        help='If debug, wandb is disabled')
    parser.add_argument('--visualize',  default=True, action='store_true',
                        help='Visualize results')
    # Loss weights
    parser.add_argument('--weight_chamfer',         default=1., type=float, help='Weight for chamfer distance')
    parser.add_argument('--weight_attraction_chamfer', default=1., type=float, help='Weight for attraction chamfer loss')
    parser.add_argument('--weight_rich_attraction_chamfer', default=0.5, type=float, help='Weight for rich attraction chamfer loss')
    parser.add_argument('--soft_attraction',        default=False, action='store_true', help='Soft version of attraction loss')
    parser.add_argument('--weight_repulsion',       default=1., type=float, help='Weight for repulsion loss')
    parser.add_argument('--weight_mse',             default=1., type=float, help='Weight for mse loss')
    parser.add_argument('--weight_align',           default=1., type=float, help='Weight for align loss')
    parser.add_argument('--weight_velcosine',       default=1., type=float, help='Weight for velocity-cosine attraction loss')
    parser.add_argument('--weight_intra_align',     default=1., type=float, help='Weight for intra-align loss')
    parser.add_argument('--weight_discriminator',   default=1., type=float, help='Weight for learned discriminator loss')
    parser.add_argument('--weight_discr_training',  default=1., type=float, help='Weight for the discriminator training loss')
    parser.add_argument('--weight_wdiscriminator',  default=1., type=float, help='Weight for learned discriminator loss')
    parser.add_argument('--discr_train_iter',       default=1, type=int, help='Iterations of discr training on a single batch')
    parser.add_argument('--discr_lambdaGP',         default=10, type=int, help='Lambda for GP term.')
    return parser.parse_args()


def test(model, loader, loss_handler, metrics_handler=None, save=False, save_dir='.', split='test'):
    """Run forward pass on the loader, compute loss/metrics, optionally save predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tot_loss = 0
    tot_loss_list = np.zeros(len(loss_handler.loss))
    data_count = 0

    if metrics_handler is not None:
        metrics = np.zeros(len(metrics_handler.metrics))
    else:
        metrics = None

    with torch.no_grad():
        for i, data in enumerate(loader):
            point_cloud, traj, dirnames = data
            B, N, dim = point_cloud.size()
            data_count += B

            point_cloud = point_cloud.permute(0, 2, 1).to(device, dtype=torch.float)  # (B,3,N)
            traj = traj.to(device, dtype=torch.float)

            traj_pred = model(point_cloud)
            loss, loss_list = loss_handler.compute(traj_pred, traj, train=False)

            # Visualize the batch 0-th
            visualize_traj(traj_pred[0], traj[0], os.path.join(save_dir, "traj_vis"))

            tot_loss += loss.item() * B
            tot_loss_list += loss_list * B

            if metrics_handler is not None:
                metrics += B * metrics_handler.compute(traj_pred, traj)

            # Whether to save inference results
            if save:
                data_to_save = {
                    'dirnames': dirnames,
                    'traj': traj.cpu().numpy(),
                    'traj_pred': traj_pred.cpu().numpy(),
                    'batch_index': i,
                    'split': split,
                }
                # Can save only a subset of batches if needed
                outfile = os.path.join(save_dir, f"results_{split}_batch{i}.npy")
                np.save(outfile, data_to_save)

    avg_loss = tot_loss / data_count
    avg_loss_list = tot_loss_list / data_count

    if metrics_handler is not None:
        metrics /= data_count

    return avg_loss, avg_loss_list, metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    config = get_train_config(args.config)  # Load config from json if provided
    # Override: command line args take precedence
    config = {**args.__dict__, **config}

    if not args.debug:
        wandb.init(project='paintnet_test',
                   config=config,
                   mode='online')
    else:
        wandb.init(project='paintnet_test',
                   config=config,
                   mode='disabled')

    # Create output directory
    run_name = f"test_{get_random_string(5)}"
    output_dir = os.path.join(args.output_dir, run_name)
    create_dirs(output_dir)

    wandb.config.update({"output_dir": output_dir}, allow_val_change=True)
    save_config(config, output_dir)

    # ========== Load dataset (test) ==========
    dataset_path = get_dataset_path(config['dataset'])
    test_dataset = PaintNetDataloader(root=dataset_path,
                                      dataset=config['dataset'],
                                      pc_points=config['pc_points'],
                                      traj_points=config['traj_points'],
                                      lambda_points=config['lambda_points'],
                                      normalization=config['normalization'],
                                      extra_data=tuple(config['extra_data']),
                                      weight_orient=config['weight_orient'],
                                      split='test',
                                      augmentations=None)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              num_workers=config['workers'])

    # ========== Load model + checkpoint ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config['backbone'], config=config)

    # model = init_from_pretrained(model, config=config, device=device)  # Modify if additional initialization is needed
    model.to(device)

    # Manually load checkpoint if init_from_pretrained doesn't handle it
    if os.path.isfile(config['ckpt']):
        print(f"Loading checkpoint from {config['ckpt']}")
        state_dict = torch.load(config['ckpt'], map_location=device)
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'], strict=False)  # Set strict=True/False as needed
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: checkpoint not found at {config['ckpt']}")

    # ========== Configure loss & metrics handlers ==========
    loss_handler = LossHandler(config['loss'], config=config)
    metrics_handler = MetricsHandler(config=config, metrics=config['eval_metrics'])

    # ========== Run inference on test set ==========
    test_loss, test_loss_list, test_metrics = test(
        model=model,
        loader=test_loader,
        loss_handler=loss_handler,
        metrics_handler=metrics_handler,
        save=(not config['no_save']),
        save_dir=output_dir,
        split='test'
    )

    print(f"\n==== Test Results ====")
    print("Total test loss:", test_loss)
    print("Loss components:", test_loss_list)
    if test_metrics is not None:
        print("Metrics on test set:")
        metrics_handler.pprint(test_metrics)
        metrics_handler.log_on_wandb(test_metrics, wandb, suffix='_TEST_EVAL_METRIC')

    # ========== Visualize results if requested ==========
    # if config['visualize']:
    #     visualize_results(model, test_loader, output_dir, device)

    # ========== Wrap up ==========
    wandb.log({"test_loss": test_loss})
    wandb.finish()
    print(f"\nAll done. Results saved in {output_dir}")


if __name__ == "__main__":
    main()
