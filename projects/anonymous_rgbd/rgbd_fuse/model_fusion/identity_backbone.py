from mmengine.model import BaseModule
from mmdet.registry import MODELS


@MODELS.register_module()
class IdentityBackbone(BaseModule):
    """占位 backbone。

    实际不会被使用，只是为了兼容 Mask2Former 初始化流程。
    """

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

    def forward(self, x):
        return x
