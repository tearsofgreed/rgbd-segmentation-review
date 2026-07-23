import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.mask2former import Mask2Former


@MODELS.register_module()
class DebugDualBackboneMask2Former(Mask2Former):
    """调试版 detector：
    - 接住 inputs / aux_inputs
    - 调 dual backbone
    - 打印两路每个 stage 的特征 shape
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
        print('[DebugDualBackboneMask2Former] detector entry reached.')

        self._describe_tensor('inputs', inputs)
        self._describe_tensor('aux_inputs', aux_inputs)

        if aux_inputs is None:
            raise RuntimeError(
                '[DebugDualBackboneMask2Former] aux_inputs is None.')

        # 直接调双 backbone
        rgb_feats, aux_feats = self.backbone(inputs, aux_inputs)

        self._describe_feats('rgb_feats', rgb_feats)
        self._describe_feats('aux_feats', aux_feats)

        if len(rgb_feats) != len(aux_feats):
            raise RuntimeError(
                f'[DebugDualBackboneMask2Former] feature length mismatch: '
                f'{len(rgb_feats)} vs {len(aux_feats)}')

        for i, (rf, af) in enumerate(zip(rgb_feats, aux_feats)):
            if rf.shape != af.shape:
                print(f'[WARN] stage {i} shape mismatch: '
                      f'rgb={tuple(rf.shape)}, aux={tuple(af.shape)}')
            else:
                print(f'[OK] stage {i} aligned: {tuple(rf.shape)}')

        raise RuntimeError(
            '[DebugDualBackboneMask2Former] Debug stop: '
            'dual backbone has successfully received and processed both streams.'
        )

    def _forward(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        raise RuntimeError(
            '[DebugDualBackboneMask2Former] `_forward` is not implemented in debug mode.')

    def predict(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        raise RuntimeError(
            '[DebugDualBackboneMask2Former] `predict` is not implemented in debug mode.')
