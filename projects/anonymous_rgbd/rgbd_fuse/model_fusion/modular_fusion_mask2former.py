import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.mask2former import Mask2Former


@MODELS.register_module()
class ModularFusionMask2Former(Mask2Former):
    """可切换 RGB-only / RGB-D 的 Mask2Former.

    设计目标：
    1. 尽量贴近原生 Mask2Former 的使用方式
    2. RGB 是主路径
    3. Depth / aux 支路是可选路径
    4. fusion_module 可选
    5. RGB-only 时直接走 rgb_feats -> panoptic_head
    """

    def __init__(self,
                 fusion_module=None,
                 feature_extractor=None,
                 enable_auxiliary_branch=True,
                 debug_stop_after_fusion=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if feature_extractor is None:
            raise ValueError('feature_extractor must be provided.')

        self.feature_extractor = MODELS.build(feature_extractor)
        self.fusion_module = MODELS.build(fusion_module) if fusion_module is not None else None

        # 总开关：是否允许 aux / depth 支路参与
        self.enable_auxiliary_branch = enable_auxiliary_branch
        self.debug_stop_after_fusion = debug_stop_after_fusion

    def forward(self,
                inputs,
                data_samples=None,
                mode='tensor',
                aux_inputs=None,
                **kwargs):
        if mode == 'loss':
            return self.loss(
                inputs=inputs,
                data_samples=data_samples,
                aux_inputs=aux_inputs,
                **kwargs)
        elif mode == 'predict':
            return self.predict(
                inputs=inputs,
                data_samples=data_samples,
                aux_inputs=aux_inputs,
                **kwargs)
        elif mode == 'tensor':
            return self._forward(
                inputs=inputs,
                data_samples=data_samples,
                aux_inputs=aux_inputs,
                **kwargs)
        else:
            raise RuntimeError(f'Invalid mode: {mode}')

    def _describe_feats(self, prefix, feats):
        print(f'[{prefix}]')
        print(f'  type : {type(feats)}')
        if isinstance(feats, (list, tuple)):
            print(f'  num_feats: {len(feats)}')
            for i, feat in enumerate(feats):
                if isinstance(feat, torch.Tensor):
                    print(f'  feat[{i}] shape: {tuple(feat.shape)}')
                else:
                    print(f'  feat[{i}] type : {type(feat)}')

    def extract_dual_feat(self, batch_inputs, batch_aux_inputs=None):
        """提取 RGB / AUX 特征。

        返回：
            rgb_feats, aux_feats
        其中：
            - RGB-only 模式下 aux_feats 为 None
            - RGB-D 模式下 aux_feats 为 depth/aux backbone 输出
        """
        if not self.enable_auxiliary_branch:
            batch_aux_inputs = None

        rgb_feats, aux_feats = self.feature_extractor(batch_inputs, batch_aux_inputs)
        return rgb_feats, aux_feats

    def fuse_feat(self, rgb_feats, aux_feats):
        """融合特征。

        规则：
        - 如果没有 aux_feats，直接返回 rgb_feats
        - 如果 fusion_module 为 None，直接返回 rgb_feats
        - 否则执行融合
        """
        if aux_feats is None:
            return rgb_feats

        if self.fusion_module is None:
            return rgb_feats

        fused_feats = self.fusion_module(rgb_feats, aux_feats)
        return fused_feats

    def extract_feat(self, batch_inputs, batch_aux_inputs=None):
        """提特征，尽量贴近原生 detector 的 extract_feat 语义。"""
        rgb_feats, aux_feats = self.extract_dual_feat(batch_inputs, batch_aux_inputs)
        fused_feats = self.fuse_feat(rgb_feats, aux_feats)

        if self.debug_stop_after_fusion:
            self._describe_feats('rgb_feats', rgb_feats)
            self._describe_feats('aux_feats', aux_feats)
            self._describe_feats('fused_feats', fused_feats)
            raise RuntimeError('[ModularFusionMask2Former] Debug stop after fusion.')

        return fused_feats

    def loss(self, inputs, data_samples, aux_inputs=None, **kwargs):
        x = self.extract_feat(inputs, aux_inputs)
        losses = self.panoptic_head.loss(x, data_samples)
        return losses

    def _forward(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        """与原版 Mask2Former 一致：返回 head 的原始输出。"""
        x = self.extract_feat(inputs, aux_inputs)
        return self.panoptic_head.forward(x, data_samples)

    def predict(self,
                inputs,
                data_samples=None,
                aux_inputs=None,
                rescale=True,
                **kwargs):
        """推理流程。

        关键点：
        1. extract_feat
        2. panoptic_head.predict -> mask_cls_results, mask_pred_results
        3. panoptic_fusion_head.predict(..., rescale=rescale)
        4. add_pred_to_datasample
        """
        x = self.extract_feat(inputs, aux_inputs)

        mask_cls_results, mask_pred_results = self.panoptic_head.predict(
            x, data_samples
        )

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results,
            mask_pred_results,
            data_samples,
            rescale=rescale
        )

        return self.add_pred_to_datasample(data_samples, results_list)
