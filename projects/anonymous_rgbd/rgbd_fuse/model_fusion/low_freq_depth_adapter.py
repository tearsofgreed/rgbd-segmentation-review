import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class LowFreqDepthAdapter(nn.Module):
    """低频深度对齐模块。

    功能：
    1. 输入单通道 low-frequency depth
    2. 尽早降采样到 1/4 或 1/8
    3. 输出 stage2 decoder 可消费的 feature

    推荐：
        downsample_factor=4
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=256,
                 mid_channels=64,
                 downsample_factor=4,
                 use_bn=True):
        super().__init__()

        assert downsample_factor in [1, 4, 8], \
            f'Unsupported downsample_factor: {downsample_factor}'

        def norm(c):
            return nn.BatchNorm2d(c) if use_bn else nn.Identity()

        layers = []

        if downsample_factor == 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
                norm(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
            ])

        elif downsample_factor == 4:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
            ])

        elif downsample_factor == 8:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                norm(out_channels),
                nn.ReLU(inplace=True),
            ])

        self.encoder = nn.Sequential(*layers)
        self.downsample_factor = downsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
