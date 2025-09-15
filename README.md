# SprayDiffusion: Diffusion-based Robotic Spray Painting Trajectory Generation

This repository contains the implementation of SprayDiffusion, a diffusion-based approach for robotic spray painting trajectory generation, submitted to IEEE Robotics and Automation Letters (RA-L).

## Abstract

Deep generative models—especially diffusion models—have recently shown strong promise for long-horizon robot skills, due to their ability to capture highly multi-modal trajectory distributions. However, in industrial spray painting, methods such as current method predict local segments and heuristically stitch them, causing local inflexibility and typically requiring category-specific training. We propose SprayDiffusion, an end-to-end diffusion approach that generates smooth, temporally coherent, long-horizon 6-DoF trajectories conditioned on object point clouds and task constraints. The iterative denoising process enforces temporal coherence and captures the multimodal distribution of expert behaviors, enabling a unified policy to generalize across diverse object categories. In experiments, our method improves trajectory continuity, maintains high coverage, and generalizes to unseen shapes. Ablation studies quantify the gains from conditional guidance. These results indicate that diffusion policies offer a scalable, robust route to learning executable spray-painting programs directly from demonstrations, paving the way for unified end-to-end trajectory learning across industrial surface-processing tasks without category-specific models.

## Table of Contents

- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
- [Dataset Support](#dataset-support)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Installation

### Prerequisites

- Linux system (tested on Ubuntu)
- CUDA 11.7+ compatible GPU
- Conda package manager
    
### Environment Setup

1. **Clone the repository:**
    ```bash
   git clone git@github.com:crystalccy1/spraydiffusion_ral.git
   cd spraydiffusion_ccy
   ```

2. **Create conda environment:**
   ```bash
   # Create environment from YAML file
   conda env create -f environment.yaml
   
   # Activate the environment
   conda activate spraydiffusion
   ```

3. **Verify installation:**
   ```bash
   # Check Python version
   python --version  # Should be 3.10.18
   
   # Check PyTorch installation
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   
   # Check PyTorch3D installation
   python -c "import pytorch3d; print(f'PyTorch3D version: {pytorch3d.__version__}')"
   
   # Check other key dependencies
   python -c "import open3d, pyvista, matplotlib, numpy, scipy; print('All dependencies installed successfully')"
   ```

### Key Dependencies

The environment includes the following key packages:

#### Core ML Libraries
- **PyTorch**: 1.13.1+cu117 (with CUDA 11.7 support)
- **PyTorch3D**: 0.7.3 (for 3D operations)
- **NumPy**: 1.26.4
- **SciPy**: 1.15.3

#### 3D Processing
- **Open3D**: 0.19.0 (point cloud processing)
- **PyVista**: 0.46.1 (3D visualization)
- **Trimesh**: 4.7.4 (mesh processing)
- **VTK**: 9.5.0 (visualization toolkit)

#### Visualization
- **Matplotlib**: 3.10.5
- **Plotly**: 6.3.0
- **Seaborn**: 0.13.2

#### Configuration & Utilities
- **OmegaConf**: 2.3.0 (configuration management)
- **Termcolor**: 3.1.0 (colored terminal output)
- **TQDM**: 4.67.1 (progress bars)
- **WandB**: 0.21.1 (experiment tracking)

### GPU Requirements

- **CUDA**: 11.7+ (compatible with PyTorch 1.13.1)
- **GPU Memory**: Minimum 8GB VRAM recommended
- **Driver**: NVIDIA driver compatible with CUDA 11.7

## Quick Start

### 1. Training SprayDiffusion

```bash
# Basic training command
python train_spraydiffusion.py --config windows-v2 --seed 42

# Example: Train on different datasets
python train_spraydiffusion.py --config cuboids-v2 --seed 42
python train_spraydiffusion.py --config shelves-v2 --seed 42
python train_spraydiffusion.py --config containers-v2 --seed 42
```

### 2. Testing SprayDiffusion

```bash
# Test with checkpoint
python test_spraydiffusion.py \
    --checkpoint_path [path_to_checkpoint] \
    --dataset_name windows-v2 \
    --dataset_split test \
    --eval_episodes 20 \
    --run_name test_run \
    --seed 42 \
    --workers 4
```

### 3. Testing PaintNet (Baseline)

```bash
# Test PaintNet only
python test_spraydiffusion_with_paintnet.py \
    --paintnet_only \
    --paintnet_model [path_to_paintnet_model] \
    --dataset_name windows-v2 \
    --dataset_split test \
    --eval_episodes 20 \
    --run_name paintnet_test \
    --seed 42 \
    --workers 4
```

### 4. Training PaintNet

```bash
# Navigate to PaintNet submodule
cd submodules/paintnet

# Train PaintNet
python train.py --config windows-v2 --seed 42
```

## Core Commands

### Training Commands

```bash
# SprayDiffusion training
python train_spraydiffusion.py --config [config_name] --seed 42

# PaintNet training
cd submodules/paintnet
python train.py --config [config_name] --seed 42
```

### Testing Commands

    ```bash
# SprayDiffusion testing
python test_spraydiffusion.py \
    --checkpoint_path [checkpoint_path] \
    --dataset_name [dataset_name] \
    --dataset_split test \
    --eval_episodes 20 \
    --run_name [run_name] \
    --seed 42 \
    --workers 4

# PaintNet testing
python test_spraydiffusion_with_paintnet.py \
    --paintnet_only \
    --paintnet_model [model_path] \
    --dataset_name [dataset_name] \
    --dataset_split test \
    --eval_episodes 20 \
    --run_name [run_name] \
    --seed 42 \
    --workers 4

# Comparison testing (both methods)
python test_spraydiffusion_with_paintnet.py \
    --spraydiffusion_checkpoint [checkpoint_path] \
    --paintnet_model [model_path] \
    --dataset_name [dataset_name] \
    --comparison_mode
```

## Dataset Support

The system supports the following datasets:

- **windows-v2**: Window objects for spray painting
- **cuboids-v2**: Cuboid objects
- **shelves-v2**: Shelf objects  
- **containers-v2**: Container objects

## Evaluation Metrics

The system evaluates the following metrics:

### Primary Metrics
- **PCD (Point-wise Chamfer Distance)**: Trajectory accuracy
- **Coverage**: Surface coverage percentage
- **Smoothness**: Trajectory smoothness measure

### Performance Metrics
- **Inference Time**: Model inference latency
- **Total Latency**: End-to-end processing time

## Experimental Results

### SprayDiffusion Results

#### Cuboids-v2 Dataset
| Test | PCD | Coverage | Smoothness |
|------|-----|----------|------------|
| Test 1 (seed=42) | 18.7087 | 92.8486% | 0.0471 |
| Test 2 (seed=1001) | 16.9353 | 92.9660% | 0.0466 |
| Test 3 (seed=2002) | 16.9718 | 92.9900% | 0.0468 |
| **Mean** | **17.5386** | **92.9349%** | **0.046833** |
| **Std** | **1.0135** | **0.0757** | **0.000252** |

#### Shelves-v2 Dataset
| Test | PCD | Coverage | Smoothness |
|------|-----|----------|------------|
| Test 1 (seed=42) | 15.1421 | 82.8697% | 0.0742 |
| Test 2 (seed=1001) | 15.1978 | 83.1833% | 0.0734 |
| Test 3 (seed=2002) | 17.7759 | 82.9080% | 0.0734 |
| **Mean** | **16.0386** | **82.9870%** | **0.073667** |
| **Std** | **1.5048** | **0.1711** | **0.000462** |

#### Windows-v2 Dataset
| Test | PCD | Coverage | Smoothness |
|------|-----|----------|------------|
| Test 1 (seed=42) | 16.6167 | 99.5786% | 0.0547 |
| Test 2 (seed=1001) | 27.5459 | 99.6046% | 0.0532 |
| Test 3 (seed=2002) | 20.8766 | 99.5103% | 0.0546 |
| **Mean** | **21.3464** | **99.5645%** | **0.0545** |
| **Std** | **4.7423** | **0.0379** | **0.0000767** |

### PaintNet Baseline Results

#### Lambda=4 Configuration
| Dataset | PCD | Coverage | Smoothness | Inference Time |
|---------|-----|----------|------------|----------------|
| Shelves-v2 | 4571.93 | 18.83% | 0.2471 | 14.63ms |
| Windows-v2 | 9271.90 | 27.92% | 0.2592 | 14.94ms |
| Cuboids-v2 | 2761.44 | 32.58% | 0.6929 | 93.22ms |
| Containers-v2 | 16170.92 | 7.13% | 0.0318 | 126.67ms |

#### Lambda=10 Configuration
| Dataset | PCD | Coverage | Smoothness | Inference Time |
|---------|-----|----------|------------|----------------|
| Shelves-v2 | 30665.19 | 63.88% | 2.9587 | 87.12ms |
| Windows-v2 | 3209.26 | 82.96% | 1.2526 | 82.89ms |
| Cuboids-v2 | 5796.82 | 62.81% | 2.2239 | 82.35ms |
| Containers-v2 | 22305.24 | 27.14% | 0.6482 | 114.41ms |

#### Lambda=1 Configuration
| Dataset | PCD | Coverage | Smoothness | Inference Time |
|---------|-----|----------|------------|----------------|
| Shelves-v2 | 33.32 | 96.22% | 2.0599 | 80.28ms |
| Windows-v2 | 28.00 | 98.98% | 2.6445 | 82.63ms |
| Cuboids-v2 | 28.11 | 99.86% | 3.5888 | 81.05ms |
| Containers-v2 | 313.98 | 82.86% | 1.7118 | 115.64ms |

## Visualization

The system generates several types of visualizations:

### 1. Trajectory Visualizations
- **Predicted Trajectory**: Blue trajectory points
- **Ground Truth Trajectory**: Red trajectory points
- **Mesh Overlay**: 3D object mesh with trajectory overlay

### 2. Coverage Visualizations
- **Coverage Heatmap**: Surface coverage visualization
- **Multi-view Rendering**: Multiple camera angle views

### 3. Paper Visualizations
- **Static PNGs**: High-quality images for paper figures
- **Denoising Process**: GIF animations showing diffusion process

## Configuration

### SprayDiffusion Configs
Located in `configs/spraydiffusion/`:
- Dataset-specific configurations
- Model architecture settings
- Training parameters

### PaintNet Configs
Located in `submodules/paintnet/configs/`:
- Lambda parameter configurations
- Training hyperparameters
- Dataset specifications

### Output Structure

```
outputs/
├── demo/
│   └── [run_name]/
│       └── episode_[XXX]_Pred_Cond/
│           ├── trajectory_ep[X]_pred.png
│           ├── trajectory_ep[X]_gt.png
│           └── coverage_ep[X].png
└── paper_vis/
    └── [dataset_name]/
        └── [run_name]/
            ├── episode_[XXX]_pred_mesh.png
            ├── episode_[XXX]_gt_mesh.png
            └── episode_[XXX]_pred_gt_mesh.png
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
    ```bash
   # Check CUDA version
   nvidia-smi
   
   # If CUDA version is different, reinstall PyTorch
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
   ```

2. **PyTorch3D Installation Issues**
   ```bash
   # Reinstall PyTorch3D
   pip uninstall pytorch3d
   pip install "git+https://github.com/facebookresearch/pytorch3d.git"
   ```

3. **Open3D Import Errors**
    ```bash
   # Reinstall Open3D
   pip uninstall open3d
   pip install open3d==0.19.0
   ```

4. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Process fewer episodes

5. **Dataset Loading Errors**
   - Check dataset paths
   - Verify data format
   - Ensure proper permissions

6. **Model Loading Issues**
   - Verify checkpoint paths
   - Check model compatibility
   - Ensure proper state dict loading

### Debug Mode

```bash
# Run with debugger
python -m pdb test_spraydiffusion.py [arguments]

# Enable verbose logging
python test_spraydiffusion.py [arguments] --verbose
```

### Environment Verification Script

Create and run this script to verify your installation:

```python
#!/usr/bin/env python3
"""Environment verification script"""

def check_imports():
    """Check if all required packages can be imported"""
    packages = [
        'torch', 'torchvision', 'torchaudio',
        'pytorch3d', 'open3d', 'pyvista', 'trimesh',
        'matplotlib', 'numpy', 'scipy', 'pandas',
        'omegaconf', 'termcolor', 'tqdm', 'wandb'
    ]
    
    failed_imports = []
    for package in packages:
        try:
            __import__(package)
            print(f"SUCCESS: {package}")
        except ImportError as e:
            print(f"FAILED: {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def check_cuda():
    """Check CUDA availability"""
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if __name__ == "__main__":
    print("Checking environment setup...")
    check_imports()
    check_cuda()
```

## Statistical Analysis

### Multiple Test Runs
Each experiment is run 3 times with different seeds (42, 1001, 2002) to ensure statistical significance.

### Statistical Measures
- **Mean**: Average performance across runs
- **Standard Deviation**: Measure of variance
- **Confidence Intervals**: 95% confidence intervals for mean estimates

## Performance Comparison

### SprayDiffusion vs PaintNet
- **Trajectory Quality**: SprayDiffusion shows significantly better PCD scores
- **Coverage**: PaintNet with lambda=1 achieves highest coverage but with poor trajectory quality
- **Smoothness**: SprayDiffusion maintains better trajectory smoothness
- **Inference Speed**: PaintNet generally faster but with trade-offs in quality

## Reproducibility

### Random Seeds
- All experiments use fixed seeds for reproducibility
- Training: seed=42
- Testing: seeds 42, 1001, 2002

### Environment
- Use the provided `environment.yaml` for consistent setup
- All package versions are pinned for reproducibility

## Citation

If you use this code in your research, please cite:

```bibtex
@article{spraydiffusion2024,
  title={SprayDiffusion: Diffusion-based Robotic Spray Painting Trajectory Generation},
  author={[Authors]},
  journal={IEEE Robotics and Automation Letters},
  year={2024}
}
```

## Repository Structure

```
spraydiffusion_ccy/
├── README.md
├── environment.yaml
├── requirements.txt
├── train_spraydiffusion.py
├── test_spraydiffusion.py
├── test_spraydiffusion_with_paintnet.py
├── configs/
│   └── spraydiffusion/
│       ├── ablation/
│       ├── containers_v2.yaml
│       ├── cuboids_v2.yaml
│       ├── default.yaml
│       ├── dp3.yaml
│       ├── shelves_v2.yaml
│       └── windows_v2.yaml
├── handlers/
│   ├── loss_handler.py
│   ├── loss_spraydiffusion_handler.py
│   ├── metrics_handler.py
│   ├── metrics_handler_spraydiffusion.py
│   └── spraydiffusion_metrics_handler.py
├── models/
│   ├── 3d_diffsuion_model.py
│   ├── dgcnn.py
│   ├── point_transformer.py
│   ├── pointnet.py
│   └── pointnet2_*.py
├── scripts/
│   └── create_replay_buffer.py
├── spray_diffusion/
│   ├── common/
│   ├── dataset/
│   ├── env/
│   ├── env_runner/
│   ├── gym_util/
│   ├── model/
│   │   ├── common/
│   │   ├── diffusion/
│   │   └── vision/
│   └── policy/
├── utils/
│   ├── dataset/
│   ├── metrics/
│   ├── visualize.py
│   └── config.py
└── submodules/
    └── paintnet/
        ├── configs/
        ├── models/
        ├── train.py
        └── torch-nndistance/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We acknowledge the contributions of the research community and the open-source projects that made this work possible.