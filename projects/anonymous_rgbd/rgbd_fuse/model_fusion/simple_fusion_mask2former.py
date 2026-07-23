from mmdet.registry import MODELS
from mmdet.models.detectors.mask2former import Mask2Former


@MODELS.register_module()
class SimpleFusionMask2Former(Mask2Former):
    """可训练版 simple fusion Mask2Former。

    逻辑：
    - backbone(inputs, aux_inputs) -> dict(rgb_feats, aux_feats, fused_feats)
    - 仅将 fused_feats 送给 panoptic_head
    """

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

    def extract_feat(self, batch_inputs, batch_aux_inputs=None):
        if batch_aux_inputs is None:
            raise RuntimeError(
                '[SimpleFusionMask2Former] batch_aux_inputs is None.')

        fusion_outputs = self.backbone(batch_inputs, batch_aux_inputs)
        fused_feats = fusion_outputs['fused_feats']
        return fused_feats

    def loss(self, inputs, data_samples, aux_inputs=None, **kwargs):
        x = self.extract_feat(inputs, aux_inputs)
        losses = self.panoptic_head.loss(x, data_samples)
        return losses

    def _forward(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        x = self.extract_feat(inputs, aux_inputs)
        # 仅返回 head 的 forward 结果
        return self.panoptic_head.forward(x, data_samples)

    def predict(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        x = self.extract_feat(inputs, aux_inputs)
        results_list = self.panoptic_head.predict(x, data_samples)
        return self.add_pred_to_datasample(data_samples, results_list)
