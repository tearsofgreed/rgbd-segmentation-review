# 文件位置: rgbd_fuse/model_fusion/depth_guided_cross_attention.py
import torch
import torch.nn as nn
from mmdet.registry import MODELS

@MODELS.register_module()
class DepthGuidedCrossAttention(nn.Module):
    """Depth-guided Cross Attention Fusion Module.

    将高频深度信息作为先验，通过 cross-attention 融合 RGB 特征。
    alpha 权重初始化为 0，可自适应调整。
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 embed_dim=256,
                 num_heads=8):
        super().__init__()

        self.num_stages = len(in_channels)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 将每个 stage RGB/Depth 特征投影到统一 embed_dim
        self.rgb_proj = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels
        ])
        self.depth_proj = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels
        ])

        # Cross-Attention 每个 stage
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim,
                                  num_heads=num_heads,
                                  batch_first=True)
            for _ in range(self.num_stages)
        ])

        # 初始 alpha=0，控制 Depth 先验对融合的贡献
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.num_stages)
        ])

        # 输出 projection 回原通道
        self.out_proj = nn.ModuleList([
            nn.Conv2d(embed_dim, c, kernel_size=1) for c in in_channels
        ])

    def forward(self, rgb_feats, depth_feats):
        fused_feats = []

        for i in range(self.num_stages):
            rgb = self.rgb_proj[i](rgb_feats[i])   # [B,C,H,W] -> embed_dim
            depth = self.depth_proj[i](depth_feats[i])

            B, C, H, W = rgb.shape
            rgb_flat = rgb.flatten(2).transpose(1, 2)   # [B, H*W, C]
            depth_flat = depth.flatten(2).transpose(1, 2)

            # cross-attention: query=rgb, key/value=depth
            attn_out, _ = self.attn[i](rgb_flat, depth_flat, depth_flat)

            # alpha 控制 Depth 先验贡献
            out_flat = rgb_flat + self.alpha[i] * attn_out
            out = out_flat.transpose(1, 2).reshape(B, C, H, W)

            # 输出回原通道
            out = self.out_proj[i](out)
            fused_feats.append(out)

        return fused_feats
