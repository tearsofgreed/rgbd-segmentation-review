from mmdet.registry import MODELS
from .minimal_mask2former_head_probe import MinimalMask2FormerHeadProbe


@MODELS.register_module()
class Mask2FormerHeadProbe(MinimalMask2FormerHeadProbe):
    """训练零侵入版 Probe。

    说明：
    1. 不改写 forward / loss / predict
    2. 不在类内复制原始 forward 逻辑
    3. 训练时应与 MinimalMask2FormerHeadProbe 等价
    """
    pass
