from typing import List, Sequence, Union

from mmengine.model import BaseModule, ModuleList
from mmdet.registry import MODELS


@MODELS.register_module()
class MultiStageFusion(BaseModule):
    """多 stage 融合模块。

    输入：
        rgb_feats: tuple/list of tensors
        aux_feats: tuple/list of tensors

    输出：
        fused_feats: tuple of tensors
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 fuse_mode: Union[str, Sequence[str]] = 'avg',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = list(in_channels)

        if isinstance(fuse_mode, str):
            fuse_modes = [fuse_mode] * len(self.in_channels)
        else:
            fuse_modes = list(fuse_mode)

        assert len(fuse_modes) == len(self.in_channels), \
            'Length of fuse_mode must match length of in_channels.'

        self.fuse_modes = fuse_modes
        self.fusion_blocks = ModuleList()

        for c, mode in zip(self.in_channels, self.fuse_modes):
            self.fusion_blocks.append(
                MODELS.build(
                    dict(
                        type='StageFusionBlock',
                        in_channels=c,
                        fuse_mode=mode,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )
            )

    def forward(self, rgb_feats, aux_feats):
        if len(rgb_feats) != len(aux_feats):
            raise RuntimeError(
                f'[MultiStageFusion] feature length mismatch: '
                f'{len(rgb_feats)} vs {len(aux_feats)}'
            )

        if len(rgb_feats) != len(self.fusion_blocks):
            raise RuntimeError(
                f'[MultiStageFusion] num feature levels mismatch with fusion blocks: '
                f'{len(rgb_feats)} vs {len(self.fusion_blocks)}'
            )

        fused_feats = []
        for i, (rgb_feat, aux_feat, block) in enumerate(
                zip(rgb_feats, aux_feats, self.fusion_blocks)):
            fused_feats.append(block(rgb_feat, aux_feat))

        return tuple(fused_feats)
