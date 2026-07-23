from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class DualStageFeatureExtractor(BaseModule):
    """RGB 主干 + 可选 aux/depth 主干 的特征提取器。

    设计目标：
    1. RGB backbone 始终存在
    2. aux backbone 可选
    3. 支持 RGB-only / RGB-D 两种模式
    4. 不负责融合，只负责提特征

    返回：
        rgb_feats, aux_feats

    其中：
        - RGB-only 时 aux_feats = None
        - RGB-D 时 aux_feats 为 aux_backbone 输出
    """

    def __init__(self,
                 rgb_backbone_cfg,
                 aux_backbone_cfg=None,
                 enable_auxiliary_branch=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.rgb_backbone = MODELS.build(rgb_backbone_cfg)
        self.enable_auxiliary_branch = enable_auxiliary_branch

        if aux_backbone_cfg is not None:
            self.aux_backbone = MODELS.build(aux_backbone_cfg)
        else:
            self.aux_backbone = None

    def forward(self, inputs, aux_inputs=None):
        rgb_feats = self.rgb_backbone(inputs)

        if not self.enable_auxiliary_branch:
            return rgb_feats, None

        if aux_inputs is None:
            return rgb_feats, None

        if self.aux_backbone is None:
            return rgb_feats, None

        aux_feats = self.aux_backbone(aux_inputs)
        return rgb_feats, aux_feats
