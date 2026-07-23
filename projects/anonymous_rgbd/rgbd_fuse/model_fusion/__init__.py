# rgbd_fuse/model_fusion/__init__.py

# --- 导入原有模块 ---
from .dual_backbone import DualBackbone
from .debug_dual_backbone_mask2former import DebugDualBackboneMask2Former
from .dual_backbone_simple_fusion import DualBackboneSimpleFusion
from .debug_simple_fusion_mask2former import DebugSimpleFusionMask2Former
from .simple_fusion_mask2former import SimpleFusionMask2Former

from .fusion_blocks import StageFusionBlock
from .multi_stage_fusion import MultiStageFusion
from .dual_stage_feature_extractor import DualStageFeatureExtractor
from .modular_fusion_mask2former import ModularFusionMask2Former
from .identity_backbone import IdentityBackbone
from .mask2former_head_probe import Mask2FormerHeadProbe
from .low_freq_depth_adapter import LowFreqDepthAdapter
from .stage2_occlusion_decoder import Stage2OcclusionDecoder
from .two_stage_wrapper_mask2former import TwoStageWrapperMask2Former
from .dual_decoder_mask2former_head import DualDecoderMask2FormerHead

from .minimal_mask2former_head_probe import MinimalMask2FormerHeadProbe


from .dual_branch_coco_metric import DualBranchCocoMetric

# --- 导入新的 depth-guided cross attention 模块 ---
# 确保在定义类时就注册到 MODELS
# 这里仅导入，注册已经在 depth_guided_cross_attention.py 中通过 @MODELS.register_module() 完成
from .depth_guided_window_cross_attention import DepthGuidedWindowCrossAttention


# --- 对外公开的模块列表 ---
__all__ = [
    'DualBackbone',
    'DebugDualBackboneMask2Former',
    'DualBackboneSimpleFusion',
    'DebugSimpleFusionMask2Former',
    'SimpleFusionMask2Former',
    'StageFusionBlock',
    'MultiStageFusion',
    'DualStageFeatureExtractor',
    'ModularFusionMask2Former',
    'IdentityBackbone',
    'DepthGuidedWindowCrossAttention',
    'Mask2FormerHeadProbe',
    'LowFreqDepthAdapter',
    'Stage2OcclusionDecoder',
    'TwoStageWrapperMask2Former',
    'DualDecoderMask2FormerHead',
    'MinimalMask2FormerHeadProbe',
    'DualBranchCocoMetric',
]
