import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import sys
import os

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from omegaconf import OmegaConf

# Add models directory to path for importing other encoders
models_path = os.path.join(os.path.dirname(__file__), '../../../models')
if models_path not in sys.path:
    sys.path.append(models_path)

# Import encoder modules with error handling
PointNetfeat = None
PointNetSetAbstraction = None
PointTransformer = None

try:
    from pointnet import PointNetfeat 
    cprint("PointNet import successful", "green")
except ImportError as e:
    cprint(f"Warning: Could not import PointNet: {e}", "yellow")

try:
    from pointnet2_utils import PointNetSetAbstraction
    cprint("PointNet2 utils import successful", "green")
except ImportError as e:
    cprint(f"Warning: Could not import PointNet2 utils: {e}", "yellow")

try:
    from point_transformer import PointTransformer
    cprint("Point Transformer import successful", "green")
except ImportError as e:
    cprint(f"Warning: Could not import Point Transformer: {e}", "yellow")

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class PointNetPlusEncoder(nn.Module):
    """Wrapper for PointNet++ encoder to match the interface"""
    
    def __init__(self, in_channels=3, out_channels=1024, **kwargs):
        super().__init__()
        
        if PointNetSetAbstraction is None:
            raise ImportError("PointNetSetAbstraction is not available. Please check the import.")
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # Final projection to match output dimensions
        self.final_projection = nn.Linear(1024, out_channels)
        
    def forward(self, x):
        # x shape: (B, N, C) -> need (B, C, N) for PointNet++
        if len(x.shape) == 3 and x.shape[-1] in [3, 6]:  # (B, N, C) -> (B, C, N)
            x = x.transpose(1, 2)
        
        B, _, _ = x.shape
        
        # Extract features using PointNet++ layers
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global feature
        global_feat = l3_points.view(B, 1024)
        
        # Project to output dimensions
        return self.final_projection(global_feat)

class PointNetClassicEncoder(nn.Module):
    """Wrapper for classic PointNet encoder to match the interface"""
    
    def __init__(self, in_channels=3, out_channels=1024, **kwargs):
        super().__init__()
        
        if PointNetfeat is None:
            raise ImportError("PointNetfeat is not available. Please check the import.")
        
        self.encoder = PointNetfeat(global_feat=True, feature_transform=False, affinetrans=False, in_channel=in_channels)
        self.projection = nn.Linear(1024, out_channels)
        
    def forward(self, x):
        # x shape: (B, N, C) -> need (B, C, N) for PointNet
        if len(x.shape) == 3 and x.shape[-1] in [3, 6]:  # (B, N, C) -> (B, C, N)
            x = x.transpose(1, 2)
        
        features, _, _ = self.encoder(x)
        return self.projection(features)

class PointTransformerEncoder(nn.Module):
    """Wrapper for Point Transformer encoder to match the interface"""
    
    def __init__(self, in_channels=3, out_channels=1024, **kwargs):
        super().__init__()
        
        if PointTransformer is None:
            raise ImportError("PointTransformer is not available. Please check the import.")
        
        self.encoder = PointTransformer(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            max_seq_len=2048,
            input_dim=in_channels,
            outdim=out_channels
        )
        
    def forward(self, x):
        # Point Transformer expects (B, N, C)
        if len(x.shape) == 3 and x.shape[1] == 3:  # (B, 3, N) -> (B, N, 3)
            x = x.transpose(1, 2)
        elif len(x.shape) == 3 and x.shape[1] == 6:  # (B, 6, N) -> (B, N, 6)
            x = x.transpose(1, 2)
        
        # Use encoder part of Point Transformer
        features = self.encoder.segments_embedding(x)
        features = self.encoder.encoder(features.permute(1, 0, 2))
        features = features.mean(dim=0)  # Global pooling
        return features

class DP3Encoder(nn.Module):
    """
    DP3 Encoder with support for different point cloud encoders for ablation studies.
    
    This encoder supports multiple point cloud encoder backends:
    - 'spraydiffusion': Original DP3 encoder (PointNetEncoderXYZ/XYZRGB)
    - 'pointnet': Classic PointNet encoder
    - 'pointnet++': PointNet++ encoder with set abstraction layers
    - 'point_transformer': Point Transformer encoder
    
    Args:
        observation_space (Dict): Dictionary defining the observation space shapes
        img_crop_shape: Image crop shape (unused currently)
        out_channel (int): Output channel dimension (default: 256)
        state_mlp_size (tuple): MLP layer sizes for state encoding (default: (64, 64))
        state_mlp_activation_fn: Activation function for state MLP (default: nn.ReLU)
        pointcloud_encoder_cfg: Configuration for point cloud encoder
        use_pc_color (bool): Whether to use RGB color information (default: False)
        pointnet_type (str): Type of pointnet for spraydiffusion encoder (default: 'pointnet')
        encoder_ablation (str): Which encoder to use for ablation study (default: 'spraydiffusion')
            - 'spraydiffusion': Original DP3 encoder
            - 'pointnet': Classic PointNet
            - 'pointnet++': PointNet++
            - 'point_transformer': Point Transformer
    
    Example usage:
        # Use original DP3 encoder
        encoder = DP3Encoder(obs_space, encoder_ablation='spraydiffusion')
        
        # Use PointNet for ablation study
        encoder = DP3Encoder(obs_space, encoder_ablation='pointnet')
        
        # Use PointNet++ for ablation study
        encoder = DP3Encoder(obs_space, encoder_ablation='pointnet++')
        
        # Use Point Transformer for ablation study
        encoder = DP3Encoder(obs_space, encoder_ablation='point_transformer')
    """
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 encoder_ablation='spraydiffusion',  # New parameter for ablation study
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        # self.state_key = 'action_last_step'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        
        # 检查是否存在state_key，不存在则设置为None
        self.has_state = self.state_key in observation_space.keys()
        if self.has_state:
            self.state_shape = observation_space[self.state_key]
        else:
            cprint(f"[DP3Encoder] Warning: {self.state_key} not found in observation space", "red")
            self.state_shape = None

        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        cprint(f"[DP3Encoder] encoder_ablation: {encoder_ablation}", "cyan")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.encoder_ablation = encoder_ablation
        
        # Determine input channels
        input_channels = 6 if use_pc_color else 3
        
        # Select encoder based on ablation study
        if encoder_ablation == 'spraydiffusion':
            # Original DP3 encoder logic
            if pointnet_type == "pointnet":
                if use_pc_color:
                    pointcloud_encoder_cfg.in_channels = 6
                    self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                else:
                    pointcloud_encoder_cfg.in_channels = 3
                    self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            else:
                raise NotImplementedError(f"pointnet_type: {pointnet_type}")
                
        elif encoder_ablation == 'pointnet':
            # Classic PointNet encoder
            self.extractor = PointNetClassicEncoder(
                in_channels=input_channels,
                out_channels=pointcloud_encoder_cfg.get('out_channels', 1024)
            )
            
        elif encoder_ablation == 'pointnet++':
            # PointNet++ encoder
            self.extractor = PointNetPlusEncoder(
                in_channels=input_channels,
                out_channels=pointcloud_encoder_cfg.get('out_channels', 1024)
            )
            
        elif encoder_ablation == 'point_transformer':
            # Point Transformer encoder
            self.extractor = PointTransformerEncoder(
                in_channels=input_channels,
                out_channels=pointcloud_encoder_cfg.get('out_channels', 1024)
            )
            
        else:
            raise NotImplementedError(f"encoder_ablation: {encoder_ablation} not supported. Choose from ['spraydiffusion', 'pointnet', 'pointnet++', 'point_transformer']")

        if self.has_state:
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels += output_dim
            self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        else:
            self.state_mlp = None

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        pn_feat = self.extractor(points)    # B * out_channel
        
        # 根据是否有状态信息处理
        if self.has_state and self.state_key in observations:
            state = observations[self.state_key]
            state_feat = self.state_mlp(state)  # B * 64
            final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        else:
            # 只返回点云特征
            final_feat = pn_feat
            
        return final_feat


    def output_shape(self):
        return self.n_output_channels