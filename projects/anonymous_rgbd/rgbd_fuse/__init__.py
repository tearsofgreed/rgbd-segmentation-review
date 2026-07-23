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

#from rgbd_fuse.prepro.depth_check import CheckRGBDPair

# 新增：TwoStream backbone 注册
# rgbd_fuse/__init__.py

# 导入 fuse 子包中的模块
from .fuse import CrossAttentionFusion, DepthResNet, resnet_with_attention_processing, RGBResNet

# 导入 depth_pro 子包中的模块

# 导入 prepro 子包中的模块
from .prepro import (
    depth_check,
    depth_loading,
    depth_preprocess,
    dump_before_backbone,
    pack_rgbd_inputs,
    rgbd_data_preprocessor,
    sync_geom_transforms
)



# 修改时间：2025-12-20
# 修改作用：import 触发 SyncRandomResize/SyncRandomCrop/SyncRandomFlip 注册
# 修改前原代码：无（新增 import）
