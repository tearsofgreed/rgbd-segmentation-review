"""
File: rgbd_fuse/__init__.py
================================================

【模块名称】
rgbd_fuse 模块初始化（注册触发器）

【在整体流程中的位置】
Config(custom_imports) -> import rgbd_fuse -> 触发 TRANSFORMS 注册
属于：训练启动阶段（Runner build dataloader 之前）

【存在意义 / 功能】
- 不做任何计算
- 仅用于通过 import 将自定义 Transform 注册进 mmengine 的 TRANSFORMS registry
- 让 config 里的 dict(type='LoadDepthFromFile') 等能被正确 build

【输入 / 输出】
输入：无（被 import 时执行）
输出：无（副作用：完成注册）

【依赖关系】
- mmengine.registry.TRANSFORMS
- rgbd_fuse.depth_loading.LoadDepthFromFile
- rgbd_fuse.depth_check.CheckRGBDPair
- rgbd_fuse.depth_preprocess.DepthPreprocess

【修改记录】
- 修改时间：2025-12-18
- 修改作用：触发自定义 pipeline transforms 的注册
- 修改前原代码：无
================================================
"""
# depth_pro/__init__.py

# 导入深度图处理相关模块
from .load_aux_image_from_file import LoadAuxImageFromFile
from .dual_random_flip import DualRandomFlip
from .dual_resize import DualResize
from .dual_random_crop import DualRandomCrop
from .save_dual_debug_images import SaveDualDebugImages
from .pack_dual_det_inputs import PackDualDetInputs
from .depth_preprocess import DepthPreprocess
from .depth_freq_split import split_depth_low_high
from .dual_random_resize import DualRandomResize
from .dual_random_flip_stage2 import DualRandomFlipStage2
from .dual_random_resize_stage2 import DualRandomResizeStage2
from .dual_random_crop_stage2 import DualRandomCropStage2


__all__ = [
    'LoadAuxImageFromFile',
    'DualRandomFlip',
    'DualResize',
    'DualRandomCrop',
    'SaveDualDebugImages',
    'PackDualDetInputs',
    'DepthPreprocess',
    'split_depth_low_high',
    'DualRandomResize',
    'DualRandomFlipStage2',
    'DualRandomResizeStage2',
    'DualRandomCropStage2',
]
