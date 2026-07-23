import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.mask2former import Mask2Former


@MODELS.register_module()
class DebugSimpleFusionMask2Former(Mask2Former):
    """调试版 detector：
    - 接住双输入
    - 调 simple fusion backbone
    - 打印 rgb / aux / fused 三组特征
    - 然后主动停止
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

    def _describe_tensor(self, prefix, x):
        print(f'[{prefix}]')
        print(f'  type : {type(x)}')
        if isinstance(x, torch.Tensor):
            print(f'  shape: {tuple(x.shape)}')
            print(f'  dtype: {x.dtype}')
            print(f'  device: {x.device}')
        else:
            print('  shape: <not tensor>')

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
        else:
            print('  not a list/tuple of features')

    def loss(self, inputs, data_samples, aux_inputs=None, **kwargs):
        print('=' * 100)
        print('[DebugSimpleFusionMask2Former] detector entry reached.')

        self._describe_tensor('inputs', inputs)
        self._describe_tensor('aux_inputs', aux_inputs)

        if aux_inputs is None:
            raise RuntimeError(
                '[DebugSimpleFusionMask2Former] aux_inputs is None.')

        fusion_outputs = self.backbone(inputs, aux_inputs)

        rgb_feats = fusion_outputs['rgb_feats']
        aux_feats = fusion_outputs['aux_feats']
        fused_feats = fusion_outputs['fused_feats']

        self._describe_feats('rgb_feats', rgb_feats)
        self._describe_feats('aux_feats', aux_feats)
        self._describe_feats('fused_feats', fused_feats)

        if not (len(rgb_feats) == len(aux_feats) == len(fused_feats)):
            raise RuntimeError(
                '[DebugSimpleFusionMask2Former] feature length mismatch among '
                'rgb_feats / aux_feats / fused_feats.')

        for i, (rf, af, ff) in enumerate(zip(rgb_feats, aux_feats, fused_feats)):
            if rf.shape != af.shape:
                raise RuntimeError(
                    f'[DebugSimpleFusionMask2Former] stage {i} rgb/aux mismatch: '
                    f'{tuple(rf.shape)} vs {tuple(af.shape)}'
                )

            if ff.shape != rf.shape:
                raise RuntimeError(
                    f'[DebugSimpleFusionMask2Former] stage {i} fused shape mismatch: '
                    f'fused={tuple(ff.shape)} vs rgb={tuple(rf.shape)}'
                )

            print(f'[OK] stage {i} aligned and fused: {tuple(ff.shape)}')

        raise RuntimeError(
            '[DebugSimpleFusionMask2Former] Debug stop: '
            'simple stage-wise fusion has successfully produced fused_feats.'
        )

    def _forward(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        raise RuntimeError(
            '[DebugSimpleFusionMask2Former] `_forward` is not implemented in debug mode.')

    def predict(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        raise RuntimeError(
            '[DebugSimpleFusionMask2Former] `predict` is not implemented in debug mode.')
