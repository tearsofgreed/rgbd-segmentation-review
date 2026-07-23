import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class StageFusionBlock(BaseModule):
    """单个 stage 的融合模块。

    支持：
    - identity_rgb
    - avg
    - sum
    - concat_conv
    """

    def __init__(self,
                 in_channels,
                 fuse_mode='avg',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert fuse_mode in ['identity_rgb', 'avg', 'sum', 'concat_conv'], \
            f'Unsupported fuse_mode: {fuse_mode}'

        self.in_channels = in_channels
        self.fuse_mode = fuse_mode

        if self.fuse_mode == 'concat_conv':
            layers = [
                nn.Conv2d(
                    in_channels * 2,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True)
            ]

            if norm_cfg is not None:
                if norm_cfg['type'] == 'BN':
                    layers.append(nn.BatchNorm2d(in_channels))
                else:
                    raise NotImplementedError(
                        f'Unsupported norm type in StageFusionBlock: {norm_cfg}')

            if act_cfg is not None:
                if act_cfg['type'] == 'ReLU':
                    layers.append(nn.ReLU(inplace=act_cfg.get('inplace', True)))
                else:
                    raise NotImplementedError(
                        f'Unsupported act type in StageFusionBlock: {act_cfg}')

            self.fuse_layer = nn.Sequential(*layers)
        else:
            self.fuse_layer = None

    def forward(self, rgb_feat: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.shape != aux_feat.shape:
            raise RuntimeError(
                f'[StageFusionBlock] shape mismatch: '
                f'rgb={tuple(rgb_feat.shape)} vs aux={tuple(aux_feat.shape)}'
            )

        if self.fuse_mode == 'identity_rgb':
            fused = rgb_feat
        elif self.fuse_mode == 'avg':
            fused = 0.5 * (rgb_feat + aux_feat)
        elif self.fuse_mode == 'sum':
            fused = rgb_feat + aux_feat
        elif self.fuse_mode == 'concat_conv':
            fused = self.fuse_layer(torch.cat([rgb_feat, aux_feat], dim=1))
        else:
            raise RuntimeError(f'Invalid fuse_mode: {self.fuse_mode}')

        return fused
