from mmdet.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class DualBackbone(BaseModule):
    """最薄双 backbone wrapper。

    输入：
        inputs: Tensor[B,C,H,W]
        aux_inputs: Tensor[B,C,H,W]

    输出：
        rgb_feats: tuple[Tensor, ...]
        aux_feats: tuple[Tensor, ...]
    """

    def __init__(self, rgb_backbone_cfg, aux_backbone_cfg, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.rgb_backbone = MODELS.build(rgb_backbone_cfg)
        self.aux_backbone = MODELS.build(aux_backbone_cfg)

    def forward(self, inputs, aux_inputs):
        rgb_feats = self.rgb_backbone(inputs)
        aux_feats = self.aux_backbone(aux_inputs)
        return rgb_feats, aux_feats
