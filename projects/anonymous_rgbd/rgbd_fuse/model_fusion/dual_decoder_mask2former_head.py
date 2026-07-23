from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import reduce_mean
from mmdet.models.utils import get_uncertain_point_coords_with_randomness

from .mask2former_head_probe import Mask2FormerHeadProbe
from .mask2former_intermediate_helper import mask2former_forward_with_intermediate


@MODELS.register_module()
class DualDecoderMask2FormerHead(Mask2FormerHeadProbe):
    """双 decoder 版 head（depth-guided query routing + stage1-positive refine）。

    功能：
    1. stage1 先给出 baseline 预测
    2. stage2 在 stage1 基础上做 refine
    3. eval/test 时可打印 stage2 替换统计
    4. eval/test 时可把 stage2 修正后的结果返回给 metric
    """

    def __init__(self,
                 *args,
                 low_freq_depth_adapter: Optional[dict] = None,
                 enable_stage2_decoder: bool = True,
                 enable_stage2_loss: bool = True,
                 enable_stage2_loss_backward: bool = True,
                 use_stage2_output: bool = True,
                 enable_low_freq_prior: bool = False,
                 stage2_loss_weight: float = 0.1,
                 stage2_num_layers: int = 3,
                 stage1_mid_layer_index: int = -2,
                 detach_stage1_for_stage2: bool = False,
                 stage2_forward_no_grad_when_no_loss: bool = True,
                 preserve_stage1_path_when_stage2_inactive: bool = True,
                 run_stage2_side_forward: bool = False,
                 return_both_stage_predictions: bool = False,
                 init_inter_query_scale: float = 0.0,
                 init_depth_query_scale: float = 0.0,

                 enable_stage2_refine_only: bool = True,
                 stage2_refine_last_layer_only: bool = True,
                 stage2_refine_use_stage1_cls_for_output: bool = True,
                 stage2_mask_residual_scale: float = 1.0,

                 stage2_detach_shared_trunk: bool = False,

                 enable_depth_guided_query_routing: bool = True,
                 stage2_query_select_by_depth: bool = True,
                 stage2_query_select_by_uncertainty: bool = True,
                 stage2_query_select_by_score: bool = True,
                 stage2_query_depth_weight: float = 1.0,
                 stage2_query_uncertainty_weight: float = 0.5,
                 stage2_query_lowconf_weight: float = 0.5,
                 stage2_query_select_topk: int = 30,
                 stage2_query_score_thr: float = -1.0,
                 stage2_query_min_kept: int = 1,
                 stage2_use_selected_queries_for_output: bool = True,

                 # eval/test 时打印 stage2 替换统计
                 log_stage2_replace_stats: bool = True,

                 # eval/test 时是否强制把最终返回给 metric 的结果切成 stage2 修正后的结果
                 eval_use_stage2_for_metrics: bool = True,

                  # debug only: dump DG-SQR routing/refinement tensors for smoke visualization
                  enable_dgsqr_debug: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.enable_stage2_decoder = enable_stage2_decoder
        self.enable_stage2_loss = enable_stage2_loss
        self.enable_stage2_loss_backward = enable_stage2_loss_backward
        self.use_stage2_output = use_stage2_output
        self.enable_low_freq_prior = enable_low_freq_prior
        self.stage2_loss_weight = stage2_loss_weight
        self.stage1_mid_layer_index = stage1_mid_layer_index
        self.detach_stage1_for_stage2 = detach_stage1_for_stage2
        self.stage2_forward_no_grad_when_no_loss = stage2_forward_no_grad_when_no_loss
        self.preserve_stage1_path_when_stage2_inactive = preserve_stage1_path_when_stage2_inactive
        self.run_stage2_side_forward = run_stage2_side_forward
        self.return_both_stage_predictions = return_both_stage_predictions

        self.enable_stage2_refine_only = enable_stage2_refine_only
        self.stage2_refine_last_layer_only = stage2_refine_last_layer_only
        self.stage2_refine_use_stage1_cls_for_output = stage2_refine_use_stage1_cls_for_output
        self.stage2_mask_residual_scale = stage2_mask_residual_scale

        self.stage2_detach_shared_trunk = stage2_detach_shared_trunk

        self.enable_depth_guided_query_routing = enable_depth_guided_query_routing
        self.stage2_query_select_by_depth = stage2_query_select_by_depth
        self.stage2_query_select_by_uncertainty = stage2_query_select_by_uncertainty
        self.stage2_query_select_by_score = stage2_query_select_by_score
        self.stage2_query_depth_weight = stage2_query_depth_weight
        self.stage2_query_uncertainty_weight = stage2_query_uncertainty_weight
        self.stage2_query_lowconf_weight = stage2_query_lowconf_weight
        self.stage2_query_select_topk = stage2_query_select_topk
        self.stage2_query_score_thr = stage2_query_score_thr
        self.stage2_query_min_kept = stage2_query_min_kept
        self.stage2_use_selected_queries_for_output = stage2_use_selected_queries_for_output

        self.log_stage2_replace_stats = log_stage2_replace_stats
        self.eval_use_stage2_for_metrics = eval_use_stage2_for_metrics
        # DG-SQR debug cache. It is only populated when enable_dgsqr_debug=True.
        self.enable_dgsqr_debug = enable_dgsqr_debug
        self.latest_dgsqr_debug = None
        self._latest_dgsqr_route_components = None

        self.low_freq_depth_adapter = None
        if low_freq_depth_adapter is not None:
            self.low_freq_depth_adapter = MODELS.build(low_freq_depth_adapter)

        stage2_decoder_cfg = dict(
            return_intermediate=True,
            num_layers=stage2_num_layers,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=self.decoder_embed_dims,
                    num_heads=self.num_heads,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=self.decoder_embed_dims,
                    num_heads=self.num_heads,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=self.decoder_embed_dims,
                    feedforward_channels=self.decoder_embed_dims * 8,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None
        )
        self.stage2_transformer_decoder = type(self.transformer_decoder)(**stage2_decoder_cfg)

        out_channels = self.mask_embed[-1].out_features

        self.stage2_cls_embed = nn.Linear(self.decoder_embed_dims, self.num_classes + 1)

        self.stage2_mask_embed = nn.Sequential(
            nn.Linear(self.decoder_embed_dims, self.decoder_embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.decoder_embed_dims, self.decoder_embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.decoder_embed_dims, out_channels)
        )

        self.inter_query_proj = nn.Linear(self.decoder_embed_dims, self.decoder_embed_dims)
        self.inter_query_scale = nn.Parameter(
            torch.tensor(float(init_inter_query_scale), dtype=torch.float32)
        )

        self.depth_query_proj = nn.Linear(self.decoder_embed_dims, self.decoder_embed_dims)
        self.depth_query_scale = nn.Parameter(
            torch.tensor(float(init_depth_query_scale), dtype=torch.float32)
        )

    def init_weights(self) -> None:
        super().init_weights()

        for p in self.stage2_transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        nn.init.xavier_uniform_(self.inter_query_proj.weight)
        nn.init.constant_(self.inter_query_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.depth_query_proj.weight)
        nn.init.constant_(self.depth_query_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.stage2_cls_embed.weight)
        nn.init.constant_(self.stage2_cls_embed.bias, 0.0)

        # 关键修复：初期不让 stage2 随机扰动 stage1 mask
        nn.init.constant_(self.stage2_mask_embed[-1].weight, 0.0)
        nn.init.constant_(self.stage2_mask_embed[-1].bias, 0.0)

    def _should_use_stage2_for_metrics(self) -> bool:
        if self.training:
            return False
        return bool(self.use_stage2_output or self.eval_use_stage2_for_metrics)


    @staticmethod
    def _dgsqr_to_cpu(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: DualDecoderMask2FormerHead._dgsqr_to_cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [DualDecoderMask2FormerHead._dgsqr_to_cpu(v) for v in x]
        return x
    def _forward_head_stage2(self,
                             decoder_out: Tensor,
                             mask_feature: Tensor,
                             prev_mask_pred: Tensor,
                             attn_mask_target_size: Tuple[int, int]):
        decoder_out = self.stage2_transformer_decoder.post_norm(decoder_out)

        delta_mask_embed = self.stage2_mask_embed(decoder_out)
        delta_mask_pred = torch.einsum('bqc,bchw->bqhw', delta_mask_embed, mask_feature)

        refined_mask_pred = prev_mask_pred + self.stage2_mask_residual_scale * delta_mask_pred

        attn_mask = F.interpolate(
            refined_mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False
        )
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)
        ).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return delta_mask_pred, refined_mask_pred, attn_mask

    def _detach_stage1_outputs_for_stage2(self,
                                          stage1_outputs: Dict[str, Any],
                                          full_detach_shared_inputs: bool = False) -> Dict[str, Any]:
        out = dict(stage1_outputs)

        if self.detach_stage1_for_stage2 or full_detach_shared_inputs:
            out['query_feat_last'] = stage1_outputs['query_feat_last'].detach()
            out['coarse_mask_pred'] = stage1_outputs['coarse_mask_pred'].detach()
            out['query_state_list'] = [q.detach() for q in stage1_outputs['query_state_list']]
            out['cls_pred_list'] = [c.detach() for c in stage1_outputs['cls_pred_list']]

        if full_detach_shared_inputs:
            out['decoder_inputs'] = [t.detach() for t in stage1_outputs['decoder_inputs']]
            out['decoder_positional_encodings'] = [
                t.detach() for t in stage1_outputs['decoder_positional_encodings']
            ]
            out['query_embed'] = stage1_outputs['query_embed'].detach()
            out['mask_features'] = stage1_outputs['mask_features'].detach()
            out['multi_scale_memorys'] = [t.detach() for t in stage1_outputs['multi_scale_memorys']]

        return out

    def _masked_pool_vector(self, feat_map: Tensor, coarse_mask_pred: Tensor) -> Tensor:
        b, c, h, w = feat_map.shape
        mask = F.interpolate(
            coarse_mask_pred,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).sigmoid()
        weight = mask.flatten(2)
        feat = feat_map.flatten(2)
        weight_sum = weight.sum(-1, keepdim=True).clamp(min=1e-6)
        pooled = torch.bmm(weight, feat.transpose(1, 2)) / weight_sum
        return pooled

    def _compute_query_depth_score(self,
                                   coarse_mask_pred: Tensor,
                                   aux_low_inputs: Optional[Tensor]) -> Tensor:
        if aux_low_inputs is None:
            return coarse_mask_pred.new_zeros(
                coarse_mask_pred.shape[0], coarse_mask_pred.shape[1])

        if aux_low_inputs.dim() == 3:
            aux_low_inputs = aux_low_inputs.unsqueeze(1)

        pooled = self._masked_pool_vector(aux_low_inputs, coarse_mask_pred)
        depth_score = pooled.squeeze(-1)

        min_v = depth_score.min(dim=1, keepdim=True).values
        max_v = depth_score.max(dim=1, keepdim=True).values
        depth_score = (depth_score - min_v) / (max_v - min_v + 1e-6)
        return depth_score.detach()

    def _compute_query_uncertainty_score(self, coarse_mask_pred: Tensor) -> Tensor:
        prob = coarse_mask_pred.sigmoid()
        uncertainty = (prob * (1.0 - prob)).flatten(2).mean(-1)
        min_v = uncertainty.min(dim=1, keepdim=True).values
        max_v = uncertainty.max(dim=1, keepdim=True).values
        uncertainty = (uncertainty - min_v) / (max_v - min_v + 1e-6)
        return uncertainty.detach()

    def _compute_query_confidence_score(self, cls_pred: Tensor) -> Tensor:
        cls_prob = cls_pred.softmax(-1)[..., :-1]
        conf = cls_prob.max(-1).values
        return conf.detach()

    def _compute_query_quality(self,
                               cls_pred: Tensor,
                               mask_pred: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cls_prob = cls_pred.softmax(-1)[..., :-1]
        cls_score = cls_prob.max(-1).values

        mask_prob = mask_pred.sigmoid()
        mask_bin = (mask_prob > 0.5).float()

        mask_prob_flat = mask_prob.flatten(2)
        mask_bin_flat = mask_bin.flatten(2)

        mask_score = (mask_prob_flat * mask_bin_flat).sum(-1) / (
            mask_bin_flat.sum(-1) + 1e-6
        )

        quality = cls_score * mask_score
        return quality, cls_score, mask_score

    def _build_query_route_score(self,
                                 stage1_outputs: Dict[str, Any],
                                 aux_low_inputs: Optional[Tensor]) -> Tensor:
        stage1_cls_final = stage1_outputs['cls_pred_list'][-1]
        coarse_mask_pred = stage1_outputs['coarse_mask_pred']

        route_score = coarse_mask_pred.new_zeros(
            coarse_mask_pred.shape[0], coarse_mask_pred.shape[1])

        depth_score = coarse_mask_pred.new_zeros(
            coarse_mask_pred.shape[0], coarse_mask_pred.shape[1])
        uncertainty_score = coarse_mask_pred.new_zeros(
            coarse_mask_pred.shape[0], coarse_mask_pred.shape[1])
        low_conf_score = coarse_mask_pred.new_zeros(
            coarse_mask_pred.shape[0], coarse_mask_pred.shape[1])
        cls_conf_stage1 = coarse_mask_pred.new_zeros(
            coarse_mask_pred.shape[0], coarse_mask_pred.shape[1])

        if self.enable_depth_guided_query_routing:
            if self.stage2_query_select_by_depth:
                depth_score = self._compute_query_depth_score(coarse_mask_pred, aux_low_inputs)
                route_score = route_score + self.stage2_query_depth_weight * depth_score

            if self.stage2_query_select_by_uncertainty:
                uncertainty_score = self._compute_query_uncertainty_score(coarse_mask_pred)
                route_score = route_score + self.stage2_query_uncertainty_weight * uncertainty_score

            if self.stage2_query_select_by_score:
                cls_conf_stage1 = self._compute_query_confidence_score(stage1_cls_final)
                low_conf_score = torch.clamp(
                    1.0 - torch.abs(cls_conf_stage1 - 0.5) * 2.0,
                    min=0.0
                )
                route_score = route_score + self.stage2_query_lowconf_weight * low_conf_score

        route_score = route_score.detach()

        if getattr(self, 'enable_dgsqr_debug', False):
            self._latest_dgsqr_route_components = dict(
                route_score=route_score,
                depth_score=depth_score.detach(),
                uncertainty_score=uncertainty_score.detach(),
                low_conf_score=low_conf_score.detach(),
                cls_conf_stage1=cls_conf_stage1.detach(),
            )

        return route_score
    def _select_queries_from_pos_inds(self,
                                      route_score_img: Tensor,
                                      pos_inds: Tensor) -> Tuple[Tensor, Tensor]:
        if pos_inds.numel() == 0:
            empty = pos_inds.new_empty((0,), dtype=torch.long)
            return empty, empty

        pos_scores = route_score_img[pos_inds]

        if self.stage2_query_score_thr >= 0:
            keep_mask = pos_scores >= self.stage2_query_score_thr
            selected_local_inds = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        else:
            k = min(self.stage2_query_select_topk, pos_inds.numel())
            if k <= 0:
                k = pos_inds.numel()
            _, selected_local_inds = torch.topk(pos_scores, k=k, dim=0, largest=True)

        if selected_local_inds.numel() < self.stage2_query_min_kept:
            k = min(max(self.stage2_query_min_kept, 1), pos_inds.numel())
            _, selected_local_inds = torch.topk(pos_scores, k=k, dim=0, largest=True)

        selected_global_inds = pos_inds[selected_local_inds]
        return selected_global_inds, selected_local_inds

    def _prepare_stage2_query(self,
                              stage1_outputs: Dict[str, Any],
                              aux_low_inputs: Optional[Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        stage1_last_query = stage1_outputs['query_feat_last']
        stage1_mid_query = stage1_outputs['query_state_list'][self.stage1_mid_layer_index]
        coarse_mask_pred = stage1_outputs['coarse_mask_pred']

        stage2_query = stage1_last_query + self.inter_query_scale * self.inter_query_proj(stage1_mid_query)

        stats = dict(
            inter_query_scale=float(self.inter_query_scale.detach().cpu().item()),
            depth_query_scale=float(self.depth_query_scale.detach().cpu().item()),
        )

        if (self.enable_low_freq_prior and
                self.low_freq_depth_adapter is not None and
                aux_low_inputs is not None):
            if aux_low_inputs.dim() == 3:
                aux_low_inputs = aux_low_inputs.unsqueeze(1)

            low_freq_feat = self.low_freq_depth_adapter(aux_low_inputs)
            depth_token = self._masked_pool_vector(low_freq_feat, coarse_mask_pred)
            depth_token = self.depth_query_proj(depth_token)
            stage2_query = stage2_query + self.depth_query_scale * depth_token

            stats['low_freq_feat_mean'] = float(low_freq_feat.detach().mean().cpu().item())
            stats['low_freq_feat_std'] = float(low_freq_feat.detach().std().cpu().item())

        return stage2_query, stats

    def _run_stage2_decoder(self,
                            stage1_outputs: Dict[str, Any],
                            aux_low_inputs: Optional[Tensor]) -> Dict[str, Any]:
        stage2_query, stats = self._prepare_stage2_query(stage1_outputs, aux_low_inputs)

        decoder_inputs = stage1_outputs['decoder_inputs']
        decoder_positional_encodings = stage1_outputs['decoder_positional_encodings']
        query_embed = stage1_outputs['query_embed']
        mask_features = stage1_outputs['mask_features']
        multi_scale_memorys = stage1_outputs['multi_scale_memorys']

        prev_mask_pred = stage1_outputs['coarse_mask_pred']

        stage2_delta_mask_pred_list = []
        stage2_refined_mask_pred_list = []

        delta_mask_pred, refined_mask_pred, attn_mask = self._forward_head_stage2(
            stage2_query, mask_features, prev_mask_pred, multi_scale_memorys[0].shape[-2:]
        )
        stage2_delta_mask_pred_list.append(delta_mask_pred)
        stage2_refined_mask_pred_list.append(refined_mask_pred)
        prev_mask_pred = refined_mask_pred

        for i in range(len(self.stage2_transformer_decoder.layers)):
            level_idx = i % self.num_transformer_feat_level
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum

            layer = self.stage2_transformer_decoder.layers[i]
            stage2_query = layer(
                query=stage2_query,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None
            )

            next_level_idx = (i + 1) % self.num_transformer_feat_level
            delta_mask_pred, refined_mask_pred, attn_mask = self._forward_head_stage2(
                stage2_query, mask_features, prev_mask_pred, multi_scale_memorys[next_level_idx].shape[-2:]
            )
            stage2_delta_mask_pred_list.append(delta_mask_pred)
            stage2_refined_mask_pred_list.append(refined_mask_pred)
            prev_mask_pred = refined_mask_pred

        if self.stage2_refine_use_stage1_cls_for_output:
            stage2_cls_pred = stage1_outputs['cls_pred_list'][-1]
        else:
            stage2_cls_pred = self.stage2_cls_embed(
                self.stage2_transformer_decoder.post_norm(stage2_query)
            )

        route_score = self._build_query_route_score(stage1_outputs, aux_low_inputs)

        return dict(
            cls_pred=stage2_cls_pred,
            delta_mask_pred_list=stage2_delta_mask_pred_list,
            refined_mask_pred_list=stage2_refined_mask_pred_list,
            query_feat_last=stage2_query,
            route_score=route_score,
            stats=stats
        )

    def _prepare_batch_gt(self, batch_data_samples: SampleList):
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        batch_img_metas = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

            if hasattr(data_sample, 'gt_sem_seg'):
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg.sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        batch_gt_instances = self.preprocess_gt(
            batch_gt_instances, batch_gt_semantic_segs
        )
        return batch_gt_instances, batch_img_metas

    def _stage2_is_trainable(self) -> bool:
        return self.enable_stage2_decoder and self.enable_stage2_loss and self.enable_stage2_loss_backward

    def _build_stage2_outputs(self,
                              stage1_outputs: Dict[str, Any],
                              aux_low_inputs: Optional[Tensor]) -> Optional[Dict[str, Any]]:
        if not self.enable_stage2_decoder:
            return None

        stage2_trainable = self._stage2_is_trainable()
        force_full_detach = self.stage2_detach_shared_trunk

        stage1_outputs_for_stage2 = self._detach_stage1_outputs_for_stage2(
            stage1_outputs,
            full_detach_shared_inputs=(not stage2_trainable) or force_full_detach
        )

        if (not stage2_trainable) and self.stage2_forward_no_grad_when_no_loss:
            with torch.no_grad():
                return self._run_stage2_decoder(stage1_outputs_for_stage2, aux_low_inputs)

        return self._run_stage2_decoder(stage1_outputs_for_stage2, aux_low_inputs)

    def _log_only_name(self, loss_key: str) -> str:
        return f"s2log.{loss_key.replace('loss', 'metric')}"

    def _get_stage1_refine_targets_single(self,
                                          cls_score: Tensor,
                                          mask_pred: Tensor,
                                          gt_instances: InstanceData,
                                          img_meta: dict):
        (_, _, mask_targets, _, pos_inds, _, sampling_result) = self._get_targets_single(
            cls_score, mask_pred, gt_instances, img_meta
        )
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
        return pos_inds, pos_assigned_gt_inds, mask_targets

    def _get_stage1_refine_targets(self,
                                   stage1_cls: Tensor,
                                   stage1_mask: Tensor,
                                   batch_gt_instances: List[InstanceData],
                                   batch_img_metas: List[dict]):
        batch_pos_inds = []
        batch_pos_assigned_gt_inds = []
        batch_mask_targets = []

        num_imgs = stage1_cls.size(0)
        for i in range(num_imgs):
            pos_inds, pos_assigned_gt_inds, mask_targets = self._get_stage1_refine_targets_single(
                stage1_cls[i].detach(),
                stage1_mask[i].detach(),
                batch_gt_instances[i],
                batch_img_metas[i]
            )
            batch_pos_inds.append(pos_inds)
            batch_pos_assigned_gt_inds.append(pos_assigned_gt_inds)
            batch_mask_targets.append(mask_targets)

        return batch_pos_inds, batch_pos_assigned_gt_inds, batch_mask_targets

    def _loss_stage2_refine_from_stage1_assign(self,
                                               stage1_cls: Tensor,
                                               stage1_mask: Tensor,
                                               stage2_refined_mask: Tensor,
                                               route_score: Tensor,
                                               batch_gt_instances: List[InstanceData],
                                               batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        batch_pos_inds, batch_pos_assigned_gt_inds, batch_mask_targets = self._get_stage1_refine_targets(
            stage1_cls, stage1_mask, batch_gt_instances, batch_img_metas
        )

        selected_stage2_preds = []
        selected_mask_targets = []

        dgsqr_debug_items = []

        num_imgs = stage2_refined_mask.size(0)
        for i in range(num_imgs):
            pos_inds = batch_pos_inds[i]
            mask_targets = batch_mask_targets[i]

            if pos_inds.numel() == 0:
                continue

            selected_global_inds, selected_local_inds = self._select_queries_from_pos_inds(
                route_score[i], pos_inds
            )

            if getattr(self, 'enable_dgsqr_debug', False):
                num_queries = int(stage1_cls.size(1))
                selected_bool = torch.zeros(
                    (num_queries,), dtype=torch.bool, device=stage1_cls.device)
                if selected_global_inds.numel() > 0:
                    selected_bool[selected_global_inds] = True

                route_comps = getattr(self, '_latest_dgsqr_route_components', None) or {}
                img_meta = batch_img_metas[i] if i < len(batch_img_metas) else {}

                dgsqr_debug_items.append(dict(
                    img_id=img_meta.get('img_id', img_meta.get('ori_img_id', i)),
                    img_path=img_meta.get('img_path', img_meta.get('filename', None)),
                    ori_shape=img_meta.get('ori_shape', None),
                    img_shape=img_meta.get('img_shape', None),
                    pad_shape=img_meta.get('pad_shape', None),

                    query_ids=torch.arange(num_queries, device=stage1_cls.device),
                    positive_query_ids=pos_inds,
                    matched_gt_ids=batch_pos_assigned_gt_inds[i],
                    selected_mask=selected_bool,
                    selected_query_ids=selected_global_inds,
                    selected_local_inds=selected_local_inds,

                    route_score=route_score[i],
                    depth_score=route_comps.get('depth_score', None)[i]
                        if route_comps.get('depth_score', None) is not None else None,
                    uncertainty_score=route_comps.get('uncertainty_score', None)[i]
                        if route_comps.get('uncertainty_score', None) is not None else None,
                    low_conf_score=route_comps.get('low_conf_score', None)[i]
                        if route_comps.get('low_conf_score', None) is not None else None,
                    cls_conf_stage1=route_comps.get('cls_conf_stage1', None)[i]
                        if route_comps.get('cls_conf_stage1', None) is not None else None,

                    coarse_mask_logits=stage1_mask[i],
                    refined_mask_logits=stage2_refined_mask[i],
                    delta_mask_logits=stage2_refined_mask[i] - stage1_mask[i],

                    positive_mask_targets=mask_targets,
                    gt_labels=getattr(batch_gt_instances[i], 'labels', None),
                ))
            if selected_global_inds.numel() == 0:
                continue

            selected_stage2_preds.append(stage2_refined_mask[i, selected_global_inds])
            selected_mask_targets.append(mask_targets[selected_local_inds])

        if getattr(self, 'enable_dgsqr_debug', False):
            self.latest_dgsqr_debug = self._dgsqr_to_cpu(dict(
                source='DualDecoderMask2FormerHead._loss_stage2_refine_from_stage1_assign',
                note='debug tensors are detached and moved to CPU; no numerical behavior is changed',
                items=dgsqr_debug_items,
            ))
        if len(selected_stage2_preds) == 0:
            zero = stage2_refined_mask.sum() * 0
            return dict(loss_mask=zero, loss_dice=zero)

        mask_preds = torch.cat(selected_stage2_preds, dim=0)
        mask_targets = torch.cat(selected_mask_targets, dim=0)

        num_total_masks = reduce_mean(mask_preds.new_tensor([mask_targets.shape[0]]))
        num_total_masks = max(num_total_masks, 1)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1),
                None,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio
            )
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(),
                points_coords
            ).squeeze(1)

        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1),
            points_coords
        ).squeeze(1)

        loss_dice = self.loss_dice(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks
        )

        loss_mask = self.loss_mask(
            mask_point_preds.reshape(-1),
            mask_point_targets.reshape(-1),
            avg_factor=num_total_masks * self.num_points
        )

        return dict(loss_mask=loss_mask, loss_dice=loss_dice)

    def _apply_selected_queries_for_output(self,
                                           stage1_cls: Tensor,
                                           stage1_mask: Tensor,
                                           stage2_cls: Tensor,
                                           stage2_mask: Tensor,
                                           route_score: Tensor,
                                           delta_mask_pred_list: List[Tensor] = None
                                           ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """推理时只在 stage2 质量优于 stage1 时采用 stage2，并返回统计信息。"""
        bsz, num_queries = route_score.shape

        delta_mag = None
        if delta_mask_pred_list is not None and not self.training:
            delta_mag = torch.abs(delta_mask_pred_list[-1]).mean().item()

        if not self.stage2_use_selected_queries_for_output:
            stage1_quality, _, _ = self._compute_query_quality(stage1_cls, stage1_mask)
            stage2_quality, _, _ = self._compute_query_quality(stage2_cls, stage2_mask)
            quality_gain = stage2_quality - stage1_quality

            mean_abs_mask_diff_selected_per_img = torch.abs(
                stage2_mask - stage1_mask
            ).mean(dim=(1, 2, 3)).detach().cpu().tolist()

            mean_quality_gain_selected_per_img = quality_gain.mean(
                dim=1
            ).detach().cpu().tolist()

            has_stage2_effect_per_img = (
                (quality_gain > 0).any(dim=1).detach().cpu().tolist()
            )

            mean_abs_mask_diff_selected_global = float(
                sum(mean_abs_mask_diff_selected_per_img) /
                len(mean_abs_mask_diff_selected_per_img)
            ) if len(mean_abs_mask_diff_selected_per_img) > 0 else 0.0

            mean_quality_gain_selected_global = float(
                sum(mean_quality_gain_selected_per_img) /
                len(mean_quality_gain_selected_per_img)
            ) if len(mean_quality_gain_selected_per_img) > 0 else 0.0

            stats = dict(
                has_stage2_effect_per_img=has_stage2_effect_per_img,
                num_imgs_with_stage2_effect=int(sum(has_stage2_effect_per_img)),
                num_selected_queries_per_img=[num_queries] * bsz,
                num_replaced_queries_per_img=[num_queries] * bsz,
                replace_ratio_per_img=[1.0] * bsz,
                mean_abs_mask_diff_selected_per_img=mean_abs_mask_diff_selected_per_img,
                mean_abs_mask_diff_selected_global=mean_abs_mask_diff_selected_global,
                mean_quality_gain_selected_per_img=mean_quality_gain_selected_per_img,
                mean_quality_gain_selected_global=mean_quality_gain_selected_global,
                delta_mag=delta_mag
            )
            return stage2_cls, stage2_mask, stats

        final_cls = stage1_cls.clone()
        final_mask = stage1_mask.clone()
        conf = self._compute_query_confidence_score(stage1_cls)

        has_stage2_effect_per_img = []
        num_selected_queries_per_img = []
        num_replaced_queries_per_img = []
        replace_ratio_per_img = []
        mean_abs_mask_diff_selected_per_img = []
        mean_quality_gain_selected_per_img = []

        for i in range(bsz):
            fg_mask = conf[i] > 0.05
            pos_like_inds = torch.nonzero(fg_mask, as_tuple=False).squeeze(1)

            if pos_like_inds.numel() == 0:
                has_stage2_effect_per_img.append(False)
                num_selected_queries_per_img.append(0)
                num_replaced_queries_per_img.append(0)
                replace_ratio_per_img.append(0.0)
                mean_abs_mask_diff_selected_per_img.append(0.0)
                mean_quality_gain_selected_per_img.append(0.0)
                continue

            selected_global_inds, _ = self._select_queries_from_pos_inds(
                route_score[i], pos_like_inds
            )

            if selected_global_inds.numel() == 0:
                has_stage2_effect_per_img.append(False)
                num_selected_queries_per_img.append(0)
                num_replaced_queries_per_img.append(0)
                replace_ratio_per_img.append(0.0)
                mean_abs_mask_diff_selected_per_img.append(0.0)
                mean_quality_gain_selected_per_img.append(0.0)
                continue

            num_selected_queries_per_img.append(int(selected_global_inds.numel()))

            selected_stage1_cls = stage1_cls[i, selected_global_inds].unsqueeze(0)
            selected_stage1_mask = stage1_mask[i, selected_global_inds].unsqueeze(0)
            selected_stage2_cls = stage2_cls[i, selected_global_inds].unsqueeze(0)
            selected_stage2_mask = stage2_mask[i, selected_global_inds].unsqueeze(0)

            stage1_quality, _, _ = self._compute_query_quality(
                selected_stage1_cls, selected_stage1_mask
            )
            stage2_quality, _, _ = self._compute_query_quality(
                selected_stage2_cls, selected_stage2_mask
            )

            adopt_mask = (stage2_quality > stage1_quality).squeeze(0)
            adopted_global_inds = selected_global_inds[adopt_mask]

            num_adopted = int(adopted_global_inds.numel())
            num_replaced_queries_per_img.append(num_adopted)
            replace_ratio_per_img.append(float(num_adopted) / float(num_queries))

            if num_adopted == 0:
                has_stage2_effect_per_img.append(False)
                mean_abs_mask_diff_selected_per_img.append(0.0)
                mean_quality_gain_selected_per_img.append(0.0)
                continue

            selected_stage1_mask_adopted = stage1_mask[i, adopted_global_inds]
            selected_stage2_mask_adopted = stage2_mask[i, adopted_global_inds]

            mean_abs_diff = torch.abs(
                selected_stage2_mask_adopted - selected_stage1_mask_adopted
            ).mean().item()

            quality_gain = (stage2_quality - stage1_quality).squeeze(0)[adopt_mask]
            mean_quality_gain = quality_gain.mean().item()

            has_stage2_effect_per_img.append(True)
            mean_abs_mask_diff_selected_per_img.append(mean_abs_diff)
            mean_quality_gain_selected_per_img.append(mean_quality_gain)

            final_cls[i, adopted_global_inds] = stage2_cls[i, adopted_global_inds]
            final_mask[i, adopted_global_inds] = stage2_mask[i, adopted_global_inds]

        if len(mean_abs_mask_diff_selected_per_img) > 0:
            mean_abs_mask_diff_selected_global = float(
                sum(mean_abs_mask_diff_selected_per_img) / len(mean_abs_mask_diff_selected_per_img)
            )
        else:
            mean_abs_mask_diff_selected_global = 0.0

        if len(mean_quality_gain_selected_per_img) > 0:
            mean_quality_gain_selected_global = float(
                sum(mean_quality_gain_selected_per_img) / len(mean_quality_gain_selected_per_img)
            )
        else:
            mean_quality_gain_selected_global = 0.0

        stats = dict(
            has_stage2_effect_per_img=has_stage2_effect_per_img,
            num_imgs_with_stage2_effect=int(sum(has_stage2_effect_per_img)),
            num_selected_queries_per_img=num_selected_queries_per_img,
            num_replaced_queries_per_img=num_replaced_queries_per_img,
            replace_ratio_per_img=replace_ratio_per_img,
            mean_abs_mask_diff_selected_per_img=mean_abs_mask_diff_selected_per_img,
            mean_abs_mask_diff_selected_global=mean_abs_mask_diff_selected_global,
            mean_quality_gain_selected_per_img=mean_quality_gain_selected_per_img,
            mean_quality_gain_selected_global=mean_quality_gain_selected_global,
            delta_mag=delta_mag
        )

        return final_cls, final_mask, stats

    def forward_with_aux(self,
                         x: List[Tensor],
                         batch_data_samples: SampleList,
                         aux_low_inputs: Optional[Tensor] = None) -> Dict[str, Any]:
        if not self.enable_stage2_decoder:
            cls_pred_list, mask_pred_list = super().forward(x, batch_data_samples)
            return dict(
                stage1_cls_pred_list=cls_pred_list,
                stage1_mask_pred_list=mask_pred_list,
                stage2_outputs=None
            )

        stage1_outputs = mask2former_forward_with_intermediate(self, x)
        stage2_outputs = self._build_stage2_outputs(stage1_outputs, aux_low_inputs)

        return dict(
            stage1_cls_pred_list=stage1_outputs['cls_pred_list'],
            stage1_mask_pred_list=stage1_outputs['mask_pred_list'],
            stage2_outputs=stage2_outputs
        )

    def forward(self, x: List[Tensor], batch_data_samples: SampleList):
        if (self.enable_stage2_decoder and
                self.preserve_stage1_path_when_stage2_inactive and
                (not self.enable_stage2_loss) and
                (not self.use_stage2_output)):
            return super().forward(x, batch_data_samples)

        if not self.enable_stage2_decoder:
            return super().forward(x, batch_data_samples)

        outputs = self.forward_with_aux(x, batch_data_samples, aux_low_inputs=None)
        return outputs['stage1_cls_pred_list'], outputs['stage1_mask_pred_list']

    def loss_with_aux(self,
                      x: List[Tensor],
                      batch_data_samples: SampleList,
                      aux_low_inputs: Optional[Tensor] = None):
        if (self.preserve_stage1_path_when_stage2_inactive and
                self.enable_stage2_decoder and
                (not self.enable_stage2_loss) and
                (not self.use_stage2_output)):
            losses = super().loss(x, batch_data_samples)

            if self.run_stage2_side_forward:
                with torch.no_grad():
                    stage1_outputs = mask2former_forward_with_intermediate(self, x)
                    _ = self._build_stage2_outputs(stage1_outputs, aux_low_inputs)

            return losses

        if not self.enable_stage2_decoder and not self.enable_stage2_loss:
            return super().loss(x, batch_data_samples)

        outputs = self.forward_with_aux(x, batch_data_samples, aux_low_inputs=aux_low_inputs)

        batch_gt_instances, batch_img_metas = self._prepare_batch_gt(batch_data_samples)

        losses = self.loss_by_feat(
            outputs['stage1_cls_pred_list'],
            outputs['stage1_mask_pred_list'],
            batch_gt_instances,
            batch_img_metas
        )

        if self.enable_stage2_decoder and outputs['stage2_outputs'] is not None and self.enable_stage2_loss:
            stage1_cls_final = outputs['stage1_cls_pred_list'][-1]
            stage1_mask_final = outputs['stage1_mask_pred_list'][-1]
            stage2_mask_for_loss = outputs['stage2_outputs']['refined_mask_pred_list'][-1]
            route_score = outputs['stage2_outputs']['route_score']

            s2_losses = self._loss_stage2_refine_from_stage1_assign(
                stage1_cls_final,
                stage1_mask_final,
                stage2_mask_for_loss,
                route_score,
                batch_gt_instances,
                batch_img_metas
            )

            if self.enable_stage2_loss_backward:
                for k, v in s2_losses.items():
                    losses[f's2.{k}'] = v * self.stage2_loss_weight
            else:
                for k, v in s2_losses.items():
                    losses[self._log_only_name(k)] = (v * self.stage2_loss_weight).detach()

        return losses

    def predict_both_with_aux(self,
                              x: List[Tensor],
                              batch_data_samples: SampleList,
                              aux_low_inputs: Optional[Tensor] = None) -> Dict[str, Optional[Tensor]]:
        stage1_cls_base, stage1_mask_base = super().predict(x, batch_data_samples)

        if not self.enable_stage2_decoder:
            return dict(
                stage1_cls=stage1_cls_base,
                stage1_mask=stage1_mask_base,
                stage2_cls=None,
                stage2_mask=None,
                stage2_replace_stats=None
            )

        outputs = self.forward_with_aux(x, batch_data_samples, aux_low_inputs=aux_low_inputs)

        stage2_cls = None
        stage2_mask = None
        stage2_replace_stats = None

        if outputs['stage2_outputs'] is not None:
            raw_stage2_cls = outputs['stage2_outputs']['cls_pred']
            raw_stage2_mask = outputs['stage2_outputs']['refined_mask_pred_list'][-1]
            delta_mask_pred_list = outputs['stage2_outputs']['delta_mask_pred_list']
            route_score = outputs['stage2_outputs']['route_score']

            raw_stage2_mask = F.interpolate(
                raw_stage2_mask,
                size=stage1_mask_base.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            stage2_cls, stage2_mask, stage2_replace_stats = self._apply_selected_queries_for_output(
                stage1_cls_base,
                stage1_mask_base,
                raw_stage2_cls,
                raw_stage2_mask,
                route_score,
                delta_mask_pred_list
            )

        return dict(
            stage1_cls=stage1_cls_base,
            stage1_mask=stage1_mask_base,
            stage2_cls=stage2_cls,
            stage2_mask=stage2_mask,
            stage2_replace_stats=stage2_replace_stats
        )

    def predict_with_aux(self,
                         x: List[Tensor],
                         batch_data_samples: SampleList,
                         aux_low_inputs: Optional[Tensor] = None):
        use_stage2_for_metrics = self._should_use_stage2_for_metrics()

        if (self.preserve_stage1_path_when_stage2_inactive and
                self.enable_stage2_decoder and
                (not use_stage2_for_metrics) and
                (not self.return_both_stage_predictions) and
                (not self.log_stage2_replace_stats)):
            return super().predict(x, batch_data_samples)

        if not self.enable_stage2_decoder:
            return super().predict(x, batch_data_samples)

        both = self.predict_both_with_aux(
            x,
            batch_data_samples,
            aux_low_inputs=aux_low_inputs
        )

        if use_stage2_for_metrics and both['stage2_cls'] is not None and both['stage2_mask'] is not None:
            return both['stage2_cls'], both['stage2_mask']

        return both['stage1_cls'], both['stage1_mask']
