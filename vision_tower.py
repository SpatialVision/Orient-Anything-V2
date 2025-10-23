import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

from utils.paths import *
from typing import Dict, List, Optional, Set, Tuple, Union
import os

from contextlib import nullcontext
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose

class OriAny_CameraHead(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using iterative refinement.
    It applies a series of transformer blocks (the "trunk") to dedicated camera tokens.
    """
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "OriAny",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
    ):
        super().__init__()

        if pose_encoding_type == "OriAny":
            self.target_dim = 360+180+360+2
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trunk_depth = trunk_depth

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        # Normalizations for camera token and trunk output.
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # Learnable empty camera pose token.
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict camera parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted camera encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine camera pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized camera tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated camera encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            # Compute the delta update for the pose encoding.
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation, quaternion, and field-of-view.
            # activated_pose = activate_pose(
            #     pred_pose_enc,
            #     trans_act=self.trans_act,
            #     quat_act=self.quat_act,
            #     fl_act=self.fl_act,
            # )
            # pred_pose_enc_list.append(activated_pose)
            pred_pose_enc_list.append(pred_pose_enc)

        return pred_pose_enc_list

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift

def load_patch_embed_weights(model, checkpoint_path):
    # 1. 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 2. 获取 state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)

    # 3. 过滤只包含 aggregator.patch_embed 的参数
    patch_embed_state = {
        k.replace("aggregator.patch_embed.", ""): v
        for k, v in state_dict.items()
        if k.startswith("aggregator.patch_embed.")
    }

    # 4. 加载到目标模块
    missing_keys, unexpected_keys = model.aggregator.patch_embed.load_state_dict(
        patch_embed_state, strict=False
    )

    print("Loaded patch_embed weights.")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

class VGGT_OriAny_Ref(nn.Module):
    def __init__(self,
                 dtype,
                 out_dim,
                 nopretrain
                ) -> None:
        super().__init__()
        self.vggt = VGGT()

        self.dtype = dtype
        self.ref_sampler = MLP_dim(in_dim=2048, out_dim=out_dim)
        self.ref_sampler.apply(init_weights)
        self.tgt_sampler = MLP_dim(in_dim=2048, out_dim=out_dim)
        self.tgt_sampler.apply(init_weights)

    def forward(self, img_inputs):
        device = self.get_device()

        with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
            if img_inputs.shape == 4:
                img_inputs = img_inputs[None]
            aggregated_tokens_list, ps_idx = self.vggt.aggregator(img_inputs)
            
            # Predict Cameras
            # pose_enc = self.oriany_camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            # Use tokens from the last block for camera prediction.
            tokens = aggregated_tokens_list[-1]
            # Extract the camera tokens
            pose_tokens = tokens[:, :, 0]
            # tokens = aggregated_tokens_list[-1]

            B, S, C = pose_tokens.shape
            if S>1:
                # 分离每个 batch 的第一个 token 和其余 token
                ref_tokens = pose_tokens[:, 0, :]            # shape: (B, C)
                tgt_tokens = pose_tokens[:, 1:, :]           # shape: (B, S-1, C)

                # 下采样
                ref_feat = self.ref_sampler(ref_tokens)      # shape: (B, C')，假设输出 channel 为 C'
                tgt_feat = self.tgt_sampler(tgt_tokens.reshape(B * (S - 1), C))  # shape: (B*(S-1), C')

                # 合并结果
                pose_enc = torch.cat([
                    ref_feat.unsqueeze(1),                   # (B, 1, C')
                    tgt_feat.view(B, S - 1, -1)              # (B, S-1, C')
                ], dim=1)                                    # 最终 shape: (B*S, C')
            else:
                pose_enc = self.ref_sampler(pose_tokens.view(B*S,C))
        return pose_enc

    def get_device(self):
        return next(self.parameters()).device
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    else:
        return nn.ReLU(inplace=True)

class MLP_dim(nn.Module):
    def __init__(
        self, in_dim=512, out_dim=1024, bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(in_dim, int(out_dim), bias=bias),
            nn.BatchNorm1d(int(out_dim)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(out_dim), out_dim, bias=bias),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        return self.net2(self.net1(x))


