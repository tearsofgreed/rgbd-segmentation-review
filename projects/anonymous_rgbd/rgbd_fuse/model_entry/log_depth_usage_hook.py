from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LogDepthUsageHook(Hook):
    """在每次 val 前输出 depth residual scale 的使用情况。"""

    def __init__(self, eps=1e-6):
        self.eps = eps

    def before_val_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        fusion_module = getattr(model, 'fusion_module', None)
        if fusion_module is None:
            runner.logger.info('[DepthUsageHook] fusion_module is None.')
            return

        if not hasattr(fusion_module, 'get_residual_scales'):
            runner.logger.info('[DepthUsageHook] fusion_module has no get_residual_scales().')
            return

        scales = fusion_module.get_residual_scales()
        zero_flags = [abs(x) < self.eps for x in scales]

        msg = ', '.join([
            f'stage{i}: scale={scales[i]:.8f}, near_zero={zero_flags[i]}'
            for i in range(len(scales))
        ])
        runner.logger.info(f'[DepthUsageHook] {msg}')
