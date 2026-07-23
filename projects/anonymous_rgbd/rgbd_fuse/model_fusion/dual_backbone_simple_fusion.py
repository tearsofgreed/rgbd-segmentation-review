from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class DualBackboneSimpleFusion(BaseModule):
    """双 backbone + 最简单 stage-wise fusion.

    返回：
        {
            'rgb_feats': tuple(...),
            'aux_feats': tuple(...),
            'fused_feats': tuple(...)
        }
    """

    def __init__(self,
                 rgb_backbone_cfg,
                 aux_backbone_cfg,
                 fuse_mode='avg',
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert fuse_mode in ['avg', 'sum'], \
            f'Unsupported fuse_mode: {fuse_mode}'

        self.rgb_backbone = MODELS.build(rgb_backbone_cfg)
        self.aux_backbone = MODELS.build(aux_backbone_cfg)
        self.fuse_mode = fuse_mode

    def _fuse_one_stage(self, rgb_feat, aux_feat):
        if rgb_feat.shape != aux_feat.shape:
            raise RuntimeError(
                f'[DualBackboneSimpleFusion] shape mismatch: '
                f'rgb={tuple(rgb_feat.shape)} vs aux={tuple(aux_feat.shape)}'
            )

        if self.fuse_mode == 'sum':
            fused = rgb_feat + aux_feat
        elif self.fuse_mode == 'avg':
            fused = 0.5 * (rgb_feat + aux_feat)
        else:
            raise RuntimeError(f'Invalid fuse_mode: {self.fuse_mode}')

        return fused

    def forward(self, inputs, aux_inputs):
        rgb_feats = self.rgb_backbone(inputs)
        aux_feats = self.aux_backbone(aux_inputs)

        if len(rgb_feats) != len(aux_feats):
            raise RuntimeError(
                f'[DualBackboneSimpleFusion] feature length mismatch: '
                f'{len(rgb_feats)} vs {len(aux_feats)}'
            )

        fused_feats = []
        for rgb_feat, aux_feat in zip(rgb_feats, aux_feats):
            fused_feats.append(self._fuse_one_stage(rgb_feat, aux_feat))

        return {
            'rgb_feats': tuple(rgb_feats),
            'aux_feats': tuple(aux_feats),
            'fused_feats': tuple(fused_feats),
        }
