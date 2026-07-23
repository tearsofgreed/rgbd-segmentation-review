import torch
from mmdet.registry import MODELS
from mmdet.models.detectors.mask2former import Mask2Former


@MODELS.register_module()
class DebugDualInputMask2Former(Mask2Former):
    """调试版双输入 Mask2Former 入口。

    目标：
    - 验证 detector.forward 能接住 inputs / aux_inputs / data_samples
    - 打印两路输入 shape
    - 在 loss 阶段主动停止，避免继续进入单流 backbone/head 报错
    """

    def forward(self,
                inputs,
                data_samples=None,
                mode='tensor',
                aux_inputs=None,
                **kwargs):
        """显式接住 aux_inputs。"""
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

    def _describe_tensor(self, name, x):
        print(f'[{name}]')
        print(f'  type : {type(x)}')
        if isinstance(x, torch.Tensor):
            print(f'  shape: {tuple(x.shape)}')
            print(f'  dtype: {x.dtype}')
            print(f'  device: {x.device}')
        else:
            print('  shape: <not a tensor>')

    def loss(self, inputs, data_samples, aux_inputs=None, **kwargs):
        print('=' * 100)
        print('[DebugDualInputMask2Former] detector entry reached successfully.')

        self._describe_tensor('inputs', inputs)
        self._describe_tensor('aux_inputs', aux_inputs)

        print('[data_samples]')
        print(f'  type : {type(data_samples)}')
        if data_samples is not None:
            print(f'  len  : {len(data_samples)}')
            if len(data_samples) > 0:
                ds = data_samples[0]
                print(f'  first sample type : {type(ds)}')
                print(f'  metainfo keys     : {list(ds.metainfo.keys())}')
                if hasattr(ds, 'gt_instances'):
                    print(f'  gt_instances keys : {list(ds.gt_instances.keys())}')

        if aux_inputs is None:
            raise RuntimeError(
                '[DebugDualInputMask2Former] aux_inputs is None. '
                'Detector did not receive the auxiliary stream correctly.')

        raise RuntimeError(
            '[DebugDualInputMask2Former] Debug stop: '
            'data_preprocessor and detector have successfully received both '
            'inputs and aux_inputs.'
        )

    def _forward(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        raise RuntimeError(
            '[DebugDualInputMask2Former] `_forward` is not implemented in debug mode.')

    def predict(self, inputs, data_samples=None, aux_inputs=None, **kwargs):
        raise RuntimeError(
            '[DebugDualInputMask2Former] `predict` is not implemented in debug mode.')
