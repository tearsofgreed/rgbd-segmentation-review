from mmdet.registry import MODELS
from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead


@MODELS.register_module()
class MinimalMask2FormerHeadProbe(Mask2FormerHead):
    """极简 Probe。

    目标：
    1. 完全不改写 forward / loss / predict
    2. 只用于验证“换成子类会不会本身掉点”
    """
    pass
