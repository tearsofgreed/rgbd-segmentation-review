from copy import deepcopy
import logging

import torch
from mmengine.logging import print_log

from mmdet.registry import MODELS
from mmdet.models.detectors.mask2former import Mask2Former


@MODELS.register_module()
class TwoStageWrapperMask2Former(Mask2Former):
    """Wrapper 负责：
    1. RGB / depth backbone
    2. regularized geometry-triggered fusion
    3. 把 aux_low_inputs 路由给 stage2 head
    4. 可选把 aux_high_inputs 路由给 fusion 触发分支
    5. 在 eval 时同时挂 stage1 / stage2 两套预测结果
    6. 在 eval 时打印 stage2 替换统计
    7. 决定最终 metric 使用 stage1 还是 stage2-corrected 结果
    """

    def __init__(self,
                 fusion_module=None,
                 feature_extractor=None,
                 enable_auxiliary_branch=True,
                 debug_stop_after_fusion=False,
                 return_both_stage_predictions=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if feature_extractor is None:
            raise ValueError('feature_extractor must be provided.')

        self.feature_extractor = MODELS.build(feature_extractor)
        self.fusion_module = MODELS.build(fusion_module) if fusion_module is not None else None

        self.enable_auxiliary_branch = enable_auxiliary_branch
        self.debug_stop_after_fusion = debug_stop_after_fusion
        self.return_both_stage_predictions = return_both_stage_predictions

    def forward(self,
                inputs,
                data_samples=None,
                mode='tensor',
                aux_inputs=None,
                aux_low_inputs=None,
                aux_high_inputs=None,
                **kwargs):
        if mode == 'loss':
            return self.loss(
                inputs=inputs,
                data_samples=data_samples,
                aux_inputs=aux_inputs,
                aux_low_inputs=aux_low_inputs,
                aux_high_inputs=aux_high_inputs,
                **kwargs)
        elif mode == 'predict':
            return self.predict(
                inputs=inputs,
                data_samples=data_samples,
                aux_inputs=aux_inputs,
                aux_low_inputs=aux_low_inputs,
                aux_high_inputs=aux_high_inputs,
                **kwargs)
        elif mode == 'tensor':
            return self._forward(
                inputs=inputs,
                data_samples=data_samples,
                aux_inputs=aux_inputs,
                aux_low_inputs=aux_low_inputs,
                aux_high_inputs=aux_high_inputs,
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
        if not self.enable_auxiliary_branch:
            batch_aux_inputs = None

        rgb_feats, aux_feats = self.feature_extractor(batch_inputs, batch_aux_inputs)
        return rgb_feats, aux_feats

    def fuse_feat(self,
                  rgb_feats,
                  aux_feats,
                  aux_inputs=None,
                  aux_high_inputs=None):
        if aux_feats is None:
            return rgb_feats

        if self.fusion_module is None:
            return rgb_feats

        return self.fusion_module(
            rgb_feats,
            aux_feats,
            high_depth_inputs=aux_high_inputs,
            base_depth_inputs=aux_inputs,
        )

    def extract_feat(self,
                     batch_inputs,
                     batch_aux_inputs=None,
                     batch_aux_high_inputs=None):
        rgb_feats, aux_feats = self.extract_dual_feat(batch_inputs, batch_aux_inputs)
        fused_feats = self.fuse_feat(
            rgb_feats,
            aux_feats,
            aux_inputs=batch_aux_inputs,
            aux_high_inputs=batch_aux_high_inputs,
        )

        if self.debug_stop_after_fusion:
            self._describe_feats('rgb_feats', rgb_feats)
            self._describe_feats('aux_feats', aux_feats)
            if batch_aux_high_inputs is not None:
                self._describe_feats('aux_high_inputs', [batch_aux_high_inputs])
            self._describe_feats('fused_feats', fused_feats)
            raise RuntimeError('[TwoStageWrapperMask2Former] Debug stop after fusion.')

        return fused_feats

    def _collect_fusion_losses(self):
        if self.fusion_module is None:
            return {}
        if not hasattr(self.fusion_module, 'get_regularization_losses'):
            return {}
        return self.fusion_module.get_regularization_losses(prefix='fusion')

    def _get_use_stage2_for_metrics(self):
        if hasattr(self.panoptic_head, '_should_use_stage2_for_metrics'):
            return bool(self.panoptic_head._should_use_stage2_for_metrics())

        use_stage2_output = getattr(self.panoptic_head, 'use_stage2_output', False)
        eval_use_stage2_for_metrics = getattr(self.panoptic_head, 'eval_use_stage2_for_metrics', False)

        if self.training:
            return False

        return bool(use_stage2_output or eval_use_stage2_for_metrics)

    def _maybe_log_stage2_replace_stats(self, both):
        if self.training:
            return

        log_stage2_replace_stats = getattr(self.panoptic_head, 'log_stage2_replace_stats', False)
        if not log_stage2_replace_stats:
            return

        stats = both.get('stage2_replace_stats', None)
        if stats is None:
            return

        use_stage2_for_metrics = self._get_use_stage2_for_metrics()

        print_log(
            '[Stage2 Replace Stats] '
            f'stage2_used_for_metrics={use_stage2_for_metrics}, '
            f'has_effect={stats["has_stage2_effect_per_img"]}, '
            f'num_imgs_with_effect={stats["num_imgs_with_stage2_effect"]}, '
            f'num_selected={stats.get("num_selected_queries_per_img", None)}, '
            f'num_replaced={stats["num_replaced_queries_per_img"]}, '
            f'replace_ratio={stats["replace_ratio_per_img"]}, '
            f'mean_abs_diff={stats["mean_abs_mask_diff_selected_per_img"]}, '
            f'global_mean_abs_diff={stats["mean_abs_mask_diff_selected_global"]:.6f}, '
            f'mean_quality_gain={stats.get("mean_quality_gain_selected_per_img", None)}, '
            f'global_mean_quality_gain={stats.get("mean_quality_gain_selected_global", 0.0):.6f}, '
            f'delta_mag={stats["delta_mag"]}',
            logger='current',
            level=logging.INFO
        )

    def _maybe_log_fusion_stats(self):
        if self.training:
            return
        if self.fusion_module is None:
            return
        if not hasattr(self.fusion_module, 'get_trigger_stats'):
            return
        stats = self.fusion_module.get_trigger_stats()
        msg = '[Fusion Trigger Stats] ' + ', '.join([f'{k}={v}' for k, v in stats.items()])
        print_log(msg, logger='current', level=logging.INFO)

    def loss(self,
             inputs,
             data_samples,
             aux_inputs=None,
             aux_low_inputs=None,
             aux_high_inputs=None,
             **kwargs):
        x = self.extract_feat(inputs, aux_inputs, batch_aux_high_inputs=aux_high_inputs)

        if hasattr(self.panoptic_head, 'enable_stage2_decoder'):
            if not self.panoptic_head.enable_stage2_decoder:
                losses = self.panoptic_head.loss(x, data_samples)
                losses.update(self._collect_fusion_losses())
                return losses

        if hasattr(self.panoptic_head, 'loss_with_aux'):
            losses = self.panoptic_head.loss_with_aux(
                x,
                data_samples,
                aux_low_inputs=aux_low_inputs
            )
            losses.update(self._collect_fusion_losses())
            return losses

        losses = self.panoptic_head.loss(x, data_samples)
        losses.update(self._collect_fusion_losses())
        return losses

    def _forward(self,
                 inputs,
                 data_samples=None,
                 aux_inputs=None,
                 aux_low_inputs=None,
                 aux_high_inputs=None,
                 **kwargs):
        x = self.extract_feat(inputs, aux_inputs, batch_aux_high_inputs=aux_high_inputs)

        if hasattr(self.panoptic_head, 'enable_stage2_decoder'):
            if not self.panoptic_head.enable_stage2_decoder:
                return self.panoptic_head.forward(x, data_samples)

        if hasattr(self.panoptic_head, 'forward_with_aux'):
            return self.panoptic_head.forward_with_aux(
                x,
                data_samples,
                aux_low_inputs=aux_low_inputs
            )

        return self.panoptic_head.forward(x, data_samples)

    def _attach_branch_predictions(self, base_data_samples, results_list, branch_name: str):
        tmp_samples = deepcopy(base_data_samples)
        tmp_samples = self.add_pred_to_datasample(tmp_samples, results_list)

        for dst, src in zip(base_data_samples, tmp_samples):
            if hasattr(src, 'pred_instances'):
                setattr(dst, f'pred_instances_{branch_name}', deepcopy(src.pred_instances))
            if hasattr(src, 'pred_panoptic_seg'):
                setattr(dst, f'pred_panoptic_seg_{branch_name}', deepcopy(src.pred_panoptic_seg))
            if hasattr(src, 'pred_sem_seg'):
                setattr(dst, f'pred_sem_seg_{branch_name}', deepcopy(src.pred_sem_seg))

        return base_data_samples

    def predict(self,
                inputs,
                data_samples=None,
                aux_inputs=None,
                aux_low_inputs=None,
                aux_high_inputs=None,
                rescale=True,
                **kwargs):
        x = self.extract_feat(inputs, aux_inputs, batch_aux_high_inputs=aux_high_inputs)
        self._maybe_log_fusion_stats()

        if not self.return_both_stage_predictions:
            if hasattr(self.panoptic_head, 'enable_stage2_decoder'):
                if not self.panoptic_head.enable_stage2_decoder:
                    mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                        x, data_samples
                    )
                elif hasattr(self.panoptic_head, 'predict_with_aux'):
                    mask_cls_results, mask_pred_results = self.panoptic_head.predict_with_aux(
                        x,
                        data_samples,
                        aux_low_inputs=aux_low_inputs
                    )
                else:
                    mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                        x, data_samples
                    )
            elif hasattr(self.panoptic_head, 'predict_with_aux'):
                mask_cls_results, mask_pred_results = self.panoptic_head.predict_with_aux(
                    x,
                    data_samples,
                    aux_low_inputs=aux_low_inputs
                )
            else:
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

        if hasattr(self.panoptic_head, 'predict_both_with_aux'):
            both = self.panoptic_head.predict_both_with_aux(
                x,
                data_samples,
                aux_low_inputs=aux_low_inputs
            )
        else:
            stage1_cls, stage1_mask = self.panoptic_head.predict(x, data_samples)
            both = dict(
                stage1_cls=stage1_cls,
                stage1_mask=stage1_mask,
                stage2_cls=None,
                stage2_mask=None,
                stage2_replace_stats=None
            )

        self._maybe_log_stage2_replace_stats(both)

        stage1_results_list = self.panoptic_fusion_head.predict(
            both['stage1_cls'],
            both['stage1_mask'],
            data_samples,
            rescale=rescale
        )

        stage2_results_list = None
        if both['stage2_cls'] is not None and both['stage2_mask'] is not None:
            stage2_results_list = self.panoptic_fusion_head.predict(
                both['stage2_cls'],
                both['stage2_mask'],
                data_samples,
                rescale=rescale
            )

        use_stage2_for_metrics = self._get_use_stage2_for_metrics()

        selected_results_list = (
            stage2_results_list
            if (use_stage2_for_metrics and stage2_results_list is not None)
            else stage1_results_list
        )

        data_samples = self.add_pred_to_datasample(data_samples, selected_results_list)

        data_samples = self._attach_branch_predictions(data_samples, stage1_results_list, 'stage1')
        if stage2_results_list is not None:
            data_samples = self._attach_branch_predictions(data_samples, stage2_results_list, 'stage2')

        return data_samples
