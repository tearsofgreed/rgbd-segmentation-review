import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class Stage2OcclusionDecoder(nn.Module):
    """第一版 query-level refinement stage2 decoder。

    逻辑：
    1. 用 coarse mask 在 low-frequency depth feature 上做 mask-aware pooling
    2. 得到每个 query 对应的 depth token
    3. 用 depth token 更新 query
    4. 用 refined query 再预测 refined cls / refined mask
    """

    def __init__(self,
                 feat_channels=256,
                 num_classes=9,
                 enable_low_freq_prior=True,
                 use_learnable_scale=True,
                 init_query_residual_scale=0.0,
                 pool_type='sigmoid_mask'):
        super().__init__()
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.enable_low_freq_prior = enable_low_freq_prior
        self.pool_type = pool_type

        self.depth_token_mlp = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels)
        )

        self.query_refine_mlp = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels)
        )

        if use_learnable_scale:
            self.query_residual_scale = nn.Parameter(
                torch.tensor(float(init_query_residual_scale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'query_residual_scale',
                torch.tensor(float(init_query_residual_scale), dtype=torch.float32)
            )

        self.cls_embed = nn.Linear(feat_channels, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels)
        )

    def get_query_residual_scale(self) -> float:
        return float(self.query_residual_scale.detach().cpu().item())

    def _masked_pool(self, low_freq_feat: torch.Tensor, coarse_mask_pred: torch.Tensor):
        """在 low-frequency feature 上按 coarse mask 做 pooling.

        low_freq_feat: [B, C, H, W]
        coarse_mask_pred: [B, Q, Hm, Wm]
        pooled: [B, Q, C]
        """
        B, C, H, W = low_freq_feat.shape

        mask = F.interpolate(
            coarse_mask_pred,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        if self.pool_type == 'sigmoid_mask':
            weight = mask.sigmoid()
        else:
            raise ValueError(f'Unsupported pool_type: {self.pool_type}')

        weight = weight.flatten(2)                # [B, Q, HW]
        feat = low_freq_feat.flatten(2)           # [B, C, HW]

        weight_sum = weight.sum(-1, keepdim=True).clamp(min=1e-6)
        pooled = torch.bmm(weight, feat.transpose(1, 2)) / weight_sum
        return pooled

    def forward(self,
                query_feat_last: torch.Tensor,
                coarse_mask_pred: torch.Tensor,
                mask_features: torch.Tensor,
                low_freq_feat: torch.Tensor = None):
        refined_query = query_feat_last
        depth_tokens = None

        if self.enable_low_freq_prior and low_freq_feat is not None:
            depth_tokens = self._masked_pool(low_freq_feat, coarse_mask_pred)
            depth_tokens = self.depth_token_mlp(depth_tokens)
            delta_query = self.query_refine_mlp(depth_tokens)
            refined_query = refined_query + self.query_residual_scale * delta_query

        refined_cls_pred = self.cls_embed(refined_query)
        refined_mask_embed = self.mask_embed(refined_query)
        refined_mask_pred = torch.einsum('bqc,bchw->bqhw', refined_mask_embed, mask_features)

        out = {
            'refined_query': refined_query,
            'refined_cls_pred': refined_cls_pred,
            'refined_mask_pred': refined_mask_pred,
            'query_residual_scale': self.get_query_residual_scale()
        }

        if depth_tokens is not None:
            out['depth_tokens'] = depth_tokens
            out['depth_token_mean'] = float(depth_tokens.detach().mean().cpu().item())
            out['depth_token_std'] = float(depth_tokens.detach().std().cpu().item())

        return out
