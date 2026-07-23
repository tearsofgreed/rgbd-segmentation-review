import torch
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor


@MODELS.register_module()
class DualInputDetDataPreprocessorStage2(DetDataPreprocessor):
    """Stage2 专用 data preprocessor。

    在原始 DualInputDetDataPreprocessor 基础上，额外支持：
    - aux_low_inputs
    - aux_high_inputs
    """

    def __init__(self,
                 fixed_pad_size=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_pad_size = fixed_pad_size

    def _ensure_3d_tensor(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f'{name} must be torch.Tensor, but got {type(x)}')
        if x.dim() != 3:
            raise ValueError(f'{name} must have shape [C, H, W], but got {tuple(x.shape)}')
        return x.float()

    def _pad_to_shape(self, x: torch.Tensor, target_h: int, target_w: int, pad_value: float = 0.) -> torch.Tensor:
        _, h, w = x.shape
        if h > target_h or w > target_w:
            raise ValueError(
                f'input shape {tuple(x.shape)} is larger than target ({target_h}, {target_w})'
            )
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)

    def _get_target_hw(self, processed, use_fixed_pad=False):
        if use_fixed_pad and self.fixed_pad_size is not None:
            target_w, target_h = self.fixed_pad_size
            return target_h, target_w

        max_h = max(x.shape[1] for x in processed)
        max_w = max(x.shape[2] for x in processed)

        if self.pad_size_divisor > 1:
            max_h = int((max_h + self.pad_size_divisor - 1) // self.pad_size_divisor * self.pad_size_divisor)
            max_w = int((max_w + self.pad_size_divisor - 1) // self.pad_size_divisor * self.pad_size_divisor)

        return max_h, max_w

    def _stack_rgb_inputs(self, inputs, use_fixed_pad=False):
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            elif inputs.dim() != 4:
                raise ValueError(f'Unexpected RGB tensor dim: {inputs.dim()}')
            batch_inputs = inputs.float()

            if use_fixed_pad and self.fixed_pad_size is not None:
                target_w, target_h = self.fixed_pad_size
                padded = []
                for i in range(batch_inputs.size(0)):
                    padded.append(
                        self._pad_to_shape(batch_inputs[i], target_h, target_w, pad_value=self.pad_value)
                    )
                batch_inputs = torch.stack(padded, dim=0)
            elif self.pad_size_divisor > 1:
                padded = []
                for i in range(batch_inputs.size(0)):
                    _, h, w = batch_inputs[i].shape
                    target_h = int((h + self.pad_size_divisor - 1) // self.pad_size_divisor * self.pad_size_divisor)
                    target_w = int((w + self.pad_size_divisor - 1) // self.pad_size_divisor * self.pad_size_divisor)
                    padded.append(
                        self._pad_to_shape(batch_inputs[i], target_h, target_w, pad_value=self.pad_value)
                    )
                batch_inputs = torch.stack(padded, dim=0)

        elif isinstance(inputs, (list, tuple)):
            processed = []
            for i, x in enumerate(inputs):
                x = self._ensure_3d_tensor(x, name=f'inputs[{i}]')
                if getattr(self, '_channel_conversion', False) and x.size(0) == 3:
                    x = x[[2, 1, 0], ...]
                processed.append(x)

            target_h, target_w = self._get_target_hw(processed, use_fixed_pad=use_fixed_pad)
            processed = [self._pad_to_shape(x, target_h, target_w, pad_value=self.pad_value) for x in processed]
            batch_inputs = torch.stack(processed, dim=0)
        else:
            raise TypeError(f'Unsupported RGB input type: {type(inputs)}')

        if getattr(self, '_enable_normalize', False):
            mean = self.mean.to(batch_inputs.device)
            std = self.std.to(batch_inputs.device)
            batch_inputs = (batch_inputs - mean) / std

        return batch_inputs

    def _stack_aux_inputs(self, aux_inputs, target_shape=None):
        if isinstance(aux_inputs, torch.Tensor):
            if aux_inputs.dim() == 3:
                aux_inputs = aux_inputs.unsqueeze(0)
            elif aux_inputs.dim() != 4:
                raise ValueError(f'Unexpected aux tensor dim: {aux_inputs.dim()}')
            batch_aux_inputs = aux_inputs.float()

            if target_shape is not None:
                target_h, target_w = target_shape
                padded = []
                for i in range(batch_aux_inputs.size(0)):
                    padded.append(self._pad_to_shape(batch_aux_inputs[i], target_h, target_w))
                batch_aux_inputs = torch.stack(padded, dim=0)
            return batch_aux_inputs

        elif isinstance(aux_inputs, (list, tuple)):
            processed = []
            for i, x in enumerate(aux_inputs):
                x = self._ensure_3d_tensor(x, name=f'aux_inputs[{i}]')
                processed.append(x)

            if target_shape is not None:
                target_h, target_w = target_shape
                processed = [self._pad_to_shape(x, target_h, target_w) for x in processed]

            return torch.stack(processed, dim=0)
        else:
            raise TypeError(f'Unsupported aux input type: {type(aux_inputs)}')

    def _pad_data_samples(self, data_samples, target_h, target_w):
        if data_samples is None:
            return

        for ds in data_samples:
            if hasattr(ds, 'gt_instances') and hasattr(ds.gt_instances, 'masks'):
                masks = ds.gt_instances.masks
                if masks is not None:
                    ds.gt_instances.masks = masks.pad(out_shape=(target_h, target_w), pad_val=0)

            if hasattr(ds, 'ignored_instances') and hasattr(ds.ignored_instances, 'masks'):
                masks = ds.ignored_instances.masks
                if masks is not None:
                    ds.ignored_instances.masks = masks.pad(out_shape=(target_h, target_w), pad_val=0)

            if hasattr(ds, 'gt_sem_seg') and hasattr(ds.gt_sem_seg, 'sem_seg'):
                sem_seg = ds.gt_sem_seg.sem_seg
                if isinstance(sem_seg, torch.Tensor):
                    h, w = sem_seg.shape[-2:]
                    pad_h = target_h - h
                    pad_w = target_w - w
                    if pad_h < 0 or pad_w < 0:
                        raise ValueError(
                            f'gt_sem_seg shape {(h, w)} is larger than target ({target_h}, {target_w})'
                        )
                    if pad_h > 0 or pad_w > 0:
                        ds.gt_sem_seg.sem_seg = F.pad(sem_seg, (0, pad_w, 0, pad_h), value=0)

    def _update_meta(self, data_samples, batch_inputs, batch_aux_inputs=None,
                     batch_aux_low_inputs=None, batch_aux_high_inputs=None):
        if data_samples is None:
            return

        batch_input_shape = tuple(batch_inputs.shape[-2:])

        for ds in data_samples:
            meta = {
                'batch_input_shape': batch_input_shape,
                'pad_shape': batch_input_shape,
            }
            if batch_aux_inputs is not None:
                meta['batch_aux_input_shape'] = tuple(batch_aux_inputs.shape[-2:])
                meta['aux_pad_shape'] = tuple(batch_aux_inputs.shape[-2:])
            if batch_aux_low_inputs is not None:
                meta['batch_aux_low_input_shape'] = tuple(batch_aux_low_inputs.shape[-2:])
                meta['aux_low_pad_shape'] = tuple(batch_aux_low_inputs.shape[-2:])
            if batch_aux_high_inputs is not None:
                meta['batch_aux_high_input_shape'] = tuple(batch_aux_high_inputs.shape[-2:])
                meta['aux_high_pad_shape'] = tuple(batch_aux_high_inputs.shape[-2:])
            ds.set_metainfo(meta)

    def forward(self, data: dict, training: bool = False) -> dict:
        if 'inputs' not in data:
            raise KeyError('`inputs` not found in data batch.')

        data = self.cast_data(data)

        raw_rgb_inputs = data['inputs']
        raw_aux_inputs = data.get('aux_inputs', None)
        raw_aux_low_inputs = data.get('aux_low_inputs', None)
        raw_aux_high_inputs = data.get('aux_high_inputs', None)
        data_samples = data.get('data_samples', None)

        use_fixed_pad = bool(training and self.fixed_pad_size is not None)

        batch_inputs = self._stack_rgb_inputs(raw_rgb_inputs, use_fixed_pad=use_fixed_pad)
        target_h, target_w = batch_inputs.shape[-2:]

        self._pad_data_samples(data_samples, target_h, target_w)

        batch_aux_inputs = None
        if raw_aux_inputs is not None:
            batch_aux_inputs = self._stack_aux_inputs(raw_aux_inputs, target_shape=(target_h, target_w))

        batch_aux_low_inputs = None
        if raw_aux_low_inputs is not None:
            batch_aux_low_inputs = self._stack_aux_inputs(raw_aux_low_inputs, target_shape=(target_h, target_w))

        batch_aux_high_inputs = None
        if raw_aux_high_inputs is not None:
            batch_aux_high_inputs = self._stack_aux_inputs(raw_aux_high_inputs, target_shape=(target_h, target_w))

        self._update_meta(
            data_samples,
            batch_inputs,
            batch_aux_inputs=batch_aux_inputs,
            batch_aux_low_inputs=batch_aux_low_inputs,
            batch_aux_high_inputs=batch_aux_high_inputs
        )

        output = {
            'inputs': batch_inputs,
            'data_samples': data_samples
        }
        if batch_aux_inputs is not None:
            output['aux_inputs'] = batch_aux_inputs
        if batch_aux_low_inputs is not None:
            output['aux_low_inputs'] = batch_aux_low_inputs
        if batch_aux_high_inputs is not None:
            output['aux_high_inputs'] = batch_aux_high_inputs

        return output
