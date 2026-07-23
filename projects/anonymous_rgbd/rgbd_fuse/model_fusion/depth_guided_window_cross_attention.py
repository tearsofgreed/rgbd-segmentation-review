import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mmdet.registry import MODELS


@MODELS.register_module()
class DepthGuidedWindowCrossAttention(nn.Module):
    """Regularized Geometry-Triggered Window Cross-Attention.

    Design:
        - RGB provides appearance / semantic query.
        - Base depth features provide dense geometric content as Key/Value.
        - High-frequency depth does not enter as dense content; instead it is
          compressed into a window-level event trigger that modulates the depth
          residual injection.

    Notes:
        1. This module is backward compatible with the old call pattern
           ``forward(rgb_feats, depth_feats)``. In that case, it behaves like
           base-depth fusion without high-trigger modulation.
        2. For the new behavior, call
           ``forward(rgb_feats, base_depth_feats, high_depth_inputs=aux_high)``.
        3. If ``high_depth_inputs`` is None but ``base_depth_inputs`` is given,
           a fallback high-frequency map is computed by local blur-residual from
           the base depth input.

    Args:
        in_channels (list[int]): channels of multi-stage features.
        num_heads (int): number of attention heads.
        dropout (float): dropout for attention.
        attn_init_scale (float): init value for output projection weight.
        window_size (int): spatial window size.
        use_learnable_scale (bool): whether to use learnable residual scale.
        init_residual_scale (float): initial residual scale for depth branch.
        enabled_stage_indices (tuple[int]): stages that enable fusion.
        enable_high_trigger (bool): whether to use high-frequency trigger.
        trigger_hidden_dim (int): hidden dim for trigger MLP.
        trigger_use_base_consistency (bool): whether to use base/high joint stats.
        trigger_temperature (float): temperature for trigger sigmoid.
        trigger_threshold (float): threshold shift before sigmoid.
        trigger_power (float): post-sigmoid power to sharpen/soften trigger.
        init_event_scale_attn (float): initial event scale on attention tokens.
        init_event_scale_res (float): initial event scale on residual map.
        regularize_residual_scale (bool): regularize residual scale magnitude.
        residual_reg_weight (float): weight of residual scale regularization.
        regularize_trigger_sparse (bool): regularize sparse trigger usage.
        trigger_sparse_weight (float): weight of sparse trigger regularization.
        regularize_trigger_budget (bool): regularize stage-wise trigger budget.
        trigger_budget_weight (float): weight of trigger budget regularization.
        trigger_budget_targets (list[float] or None): desired mean trigger per stage.
        regularize_trigger_smooth (bool): regularize spatial smoothness of trigger map.
        trigger_smooth_weight (float): weight of trigger smoothness regularization.
        fallback_high_kernel_size (int): kernel size for blur-residual fallback high.
    """

    def __init__(self,
                 in_channels,
                 num_heads=8,
                 dropout=0.1,
                 attn_init_scale=0.0,
                 window_size=8,
                 use_learnable_scale=True,
                 init_residual_scale=0.0,
                 enabled_stage_indices=(0, 1, 2, 3),
                 enable_high_trigger=True,
                 trigger_hidden_dim=32,
                 trigger_use_base_consistency=True,
                 trigger_temperature=1.0,
                 trigger_threshold=0.0,
                 trigger_power=1.0,
                 init_event_scale_attn=0.0,
                 init_event_scale_res=1.0,
                 regularize_residual_scale=True,
                 residual_reg_weight=1e-4,
                 regularize_trigger_sparse=True,
                 trigger_sparse_weight=1e-4,
                 regularize_trigger_budget=False,
                 trigger_budget_weight=1e-4,
                 trigger_budget_targets=None,
                 regularize_trigger_smooth=False,
                 trigger_smooth_weight=1e-4,
                 fallback_high_kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = len(in_channels)
        self.num_heads = num_heads
        self.dropout = dropout
        self.attn_init_scale = attn_init_scale
        self.window_size = window_size
        self.use_learnable_scale = use_learnable_scale
        self.init_residual_scale = init_residual_scale
        self.enabled_stage_indices = tuple(enabled_stage_indices)

        self.enable_high_trigger = enable_high_trigger
        self.trigger_hidden_dim = trigger_hidden_dim
        self.trigger_use_base_consistency = trigger_use_base_consistency
        self.trigger_temperature = float(trigger_temperature)
        self.trigger_threshold = float(trigger_threshold)
        self.trigger_power = float(trigger_power)
        self.regularize_residual_scale = regularize_residual_scale
        self.residual_reg_weight = float(residual_reg_weight)
        self.regularize_trigger_sparse = regularize_trigger_sparse
        self.trigger_sparse_weight = float(trigger_sparse_weight)
        self.regularize_trigger_budget = regularize_trigger_budget
        self.trigger_budget_weight = float(trigger_budget_weight)
        self.regularize_trigger_smooth = regularize_trigger_smooth
        self.trigger_smooth_weight = float(trigger_smooth_weight)
        self.fallback_high_kernel_size = int(fallback_high_kernel_size)

        self.attn = nn.ModuleList()
        self.proj_out = nn.ModuleList()
        self.high_event_proj = nn.ModuleList()
        self.window_event_mlp = nn.ModuleList()
        self.event_scale_attn = nn.ParameterList()
        self.event_scale_res = nn.ParameterList()

        trigger_in_dim = 5 if self.trigger_use_base_consistency else 3

        for C in in_channels:
            self.attn.append(
                nn.MultiheadAttention(
                    embed_dim=C,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True)
            )

            proj = nn.Linear(C, C)
            if attn_init_scale == 0:
                nn.init.constant_(proj.weight, 0.0)
            else:
                nn.init.constant_(proj.weight, attn_init_scale)
            nn.init.constant_(proj.bias, 0.0)
            self.proj_out.append(proj)

            # High-frequency event branch uses raw/resized 1-channel high map.
            self.high_event_proj.append(nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
            ))
            self.window_event_mlp.append(nn.Sequential(
                nn.Linear(trigger_in_dim, trigger_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(trigger_hidden_dim, 1)
            ))
            self.event_scale_attn.append(
                nn.Parameter(torch.tensor(float(init_event_scale_attn), dtype=torch.float32))
            )
            self.event_scale_res.append(
                nn.Parameter(torch.tensor(float(init_event_scale_res), dtype=torch.float32))
            )

        if self.use_learnable_scale:
            self.residual_scale = nn.ParameterList([
                nn.Parameter(torch.tensor(float(init_residual_scale), dtype=torch.float32))
                for _ in range(self.num_stages)
            ])
        else:
            self.residual_scale = None

        if trigger_budget_targets is None:
            default_targets = [0.30, 0.20, 0.10, 0.05]
            if len(default_targets) >= self.num_stages:
                trigger_budget_targets = default_targets[:self.num_stages]
            else:
                trigger_budget_targets = default_targets + [default_targets[-1]] * (self.num_stages - len(default_targets))
        if len(trigger_budget_targets) != self.num_stages:
            raise ValueError('trigger_budget_targets must match num_stages.')
        self.register_buffer(
            'trigger_budget_targets',
            torch.tensor(trigger_budget_targets, dtype=torch.float32),
            persistent=False)

        self._cached_trigger_scores = []
        self._cached_trigger_maps = []
        self._cached_enabled_mask = []
        self._cached_trigger_debug = []

        # --- Smoke-only ablation (does NOT affect default forward math) ---
        self._ablation_mode = 'none'       # 'none'|'no_ratio'|'ratio_clamp'|'ratio_log'|'high_mean_proxy'
        self._ablation_ratio_clamp: Optional[float] = None
        self._ablation_proxy_gate: bool = False

    def set_trigger_debug_ablation(
        self,
        mode: str = 'none',
        ratio_clamp: Optional[float] = None,
        proxy_replace_gate: bool = False,
    ):
        """Smoke-only ablation control. Does NOT affect default forward math.

        Args:
            mode: 'none' (default), 'no_ratio', 'ratio_clamp',
                  'ratio_log', 'high_mean_proxy'.
            ratio_clamp: clamp value for 'ratio_clamp' mode.
            proxy_replace_gate: if True and mode='high_mean_proxy',
                replace learned gate with proxy gate.
        """
        valid = {'none', 'no_ratio', 'ratio_clamp', 'ratio_log', 'high_mean_proxy'}
        if mode not in valid:
            raise ValueError(f"Invalid ablation mode '{mode}'. Valid: {valid}")
        self._ablation_mode = mode
        self._ablation_ratio_clamp = ratio_clamp
        self._ablation_proxy_gate = bool(proxy_replace_gate)

    def _ensure_4d(self, x):
        if x is None:
            return None
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def _resize_like(self, x, ref_feat):
        x = self._ensure_4d(x)
        if x is None:
            return None
        target_hw = ref_feat.shape[-2:]
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x

    def _build_fallback_high(self, base_depth_inputs):
        base_depth_inputs = self._ensure_4d(base_depth_inputs)
        if base_depth_inputs is None:
            return None
        k = max(int(self.fallback_high_kernel_size), 3)
        if k % 2 == 0:
            k += 1
        blurred = F.avg_pool2d(base_depth_inputs, kernel_size=k, stride=1, padding=k // 2)
        high = torch.abs(base_depth_inputs - blurred)
        # keep range stable per sample
        high_min = high.amin(dim=(-2, -1), keepdim=True)
        high_max = high.amax(dim=(-2, -1), keepdim=True)
        high = (high - high_min) / (high_max - high_min + 1e-6)
        return high

    def window_partition(self, x):
        """Split x [B, C, H, W] into windows [num_windows*B, ws*ws, C]."""
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, ws * ws, C)
        return x, Hp, Wp

    def window_reverse(self, x_windows, Hp, Wp, B, C):
        ws = self.window_size
        x = x_windows.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, Hp, Wp)
        return x

    def _window_scores_to_map(self, scores, Hp, Wp, B):
        # scores: [B*num_windows, 1]
        ws = self.window_size
        score_windows = scores.unsqueeze(1).expand(-1, ws * ws, 1)
        score_map = self.window_reverse(score_windows, Hp, Wp, B, 1)
        return score_map

    def _compute_window_trigger(self, stage_idx, high_map, base_feat):
        high_proj = self.high_event_proj[stage_idx](high_map)
        high_windows, Hp, Wp = self.window_partition(high_proj)
        # [BnW, tokens, 1]
        high_abs = high_windows.abs()
        high_mean = high_abs.mean(dim=1)
        high_max = high_abs.max(dim=1).values
        high_std = high_abs.std(dim=1, unbiased=False)
        stats = [high_mean, high_max, high_std]

        if self.trigger_use_base_consistency:
            base_scalar = base_feat.mean(dim=1, keepdim=True)
            base_windows, _, _ = self.window_partition(base_scalar)
            base_var = base_windows.var(dim=1, unbiased=False)
            ratio = high_mean / (base_var.abs() + 1e-6)
            _ratio_before = ratio.detach().clone()
            stats.extend([base_var, ratio])

        # --- Smoke-only ablation: modify ratio inside stats (does NOT run unless ablation_mode != 'none') ---
        _ratio_before = ratio.detach().clone() if 'ratio' in dir() else None
        _logits_before = None
        if self._ablation_mode != 'none' and self.trigger_use_base_consistency:
            # ratio is the 5th element in stats (index 4)
            if self._ablation_mode == 'no_ratio':
                stats[4] = torch.zeros_like(stats[4])
            elif self._ablation_mode == 'ratio_clamp' and self._ablation_ratio_clamp is not None:
                c = self._ablation_ratio_clamp
                stats[4] = stats[4].clamp(-c, c)
            elif self._ablation_mode == 'ratio_log':
                r = stats[4]
                stats[4] = torch.sign(r) * torch.log1p(torch.abs(r))

        z = torch.cat(stats, dim=-1)
        logits = self.window_event_mlp[stage_idx](z)

        # --- high_mean_proxy: compute proxy gate from high_mean (smoke-only) ---
        _proxy_gate = None
        _proxy_logits = None
        if self._ablation_mode == 'high_mean_proxy':
            _proxy_logits = (high_mean - high_mean.mean()) / (high_mean.std() + 1e-6)
            _proxy_gate = torch.sigmoid(_proxy_logits)
        gate = torch.sigmoid((logits - self.trigger_threshold) / max(self.trigger_temperature, 1e-6))

        # --- Smoke-only: replace learned gate with proxy gate ---
        if self._ablation_mode == 'high_mean_proxy' and self._ablation_proxy_gate and _proxy_gate is not None:
            gate = _proxy_gate

        # --- Smoke-only: replace learned gate with proxy gate ---
        if self._ablation_mode == 'high_mean_proxy' and self._ablation_proxy_gate and _proxy_gate is not None:
            gate = _proxy_gate
        if abs(self.trigger_power - 1.0) > 1e-6:
            gate = gate.pow(self.trigger_power)
        gate_map = self._window_scores_to_map(gate, Hp, Wp, base_feat.shape[0])
        gate_map = gate_map[:, :, :base_feat.shape[-2], :base_feat.shape[-1]]

        # --- Read-only debug collection (does not affect gate/gate_map math) ---
        try:
            _dev = gate.device
            _detach = lambda t: t.detach().cpu()

            def _p(x, p_val):
                return float(_detach(torch.quantile(x.float(), p_val)))

            _logits_f = logits.float()
            _gate_f = gate.float()

            def _frac_lt(x, thr):
                return float(_detach((x.float() < thr).float().mean()))

            def _frac_gt(x, thr):
                return float(_detach((x.float() > thr).float().mean()))

            debug_entry = dict(
                stage_idx=int(stage_idx),
                trigger_score_shape=list(gate.squeeze(-1).shape) if gate.dim() > 1 else list(gate.shape),
                trigger_map_shape=list(gate_map.shape),
                logits_mean=float(_detach(_logits_f.mean())),
                logits_min=float(_detach(_logits_f.min())),
                logits_max=float(_detach(_logits_f.max())),
                logits_p05=_p(_logits_f, 0.05),
                logits_p50=_p(_logits_f, 0.50),
                logits_p95=_p(_logits_f, 0.95),
                gate_mean=float(_detach(_gate_f.mean())),
                gate_min=float(_detach(_gate_f.min())),
                gate_max=float(_detach(_gate_f.max())),
                gate_p05=_p(_gate_f, 0.05),
                gate_p50=_p(_gate_f, 0.50),
                gate_p95=_p(_gate_f, 0.95),
                frac_trigger_lt_0_05=_frac_lt(_gate_f, 0.05),
                frac_trigger_gt_0_95=_frac_gt(_gate_f, 0.95),
                trigger_score_std=float(_detach(_gate_f.std())),
                trigger_score_numel=int(_gate_f.numel()),
                trigger_score_unique_approx=len(set(torch.round(_gate_f, decimals=4).flatten().tolist())),
                high_mean_mean=float(_detach(high_mean.float().mean())),
                high_mean_min=float(_detach(high_mean.float().min())),
                high_mean_p05=_p(high_mean.float(), 0.05),
                high_mean_p50=_p(high_mean.float(), 0.50),
                high_mean_p95=_p(high_mean.float(), 0.95),
                high_mean_max=float(_detach(high_mean.float().max())),
                high_max_mean=float(_detach(high_max.float().mean())),
                high_std_mean=float(_detach(high_std.float().mean())),
                trigger_ablation_mode=self._ablation_mode,
                ratio_clamp_value=self._ablation_ratio_clamp,
                proxy_replace_gate=self._ablation_proxy_gate,
                logits_after_ablation_mean=None,  # filled below
                logits_after_ablation_p50=None,
                logits_after_ablation_p95=None,
            )
            if self.trigger_use_base_consistency:
                _bv = base_var.float()
                _rf = ratio.float()
                _bz = _detach((_bv.abs() < 1e-12).float().mean())
                debug_entry['base_var_mean'] = float(_detach(_bv.mean()))
                debug_entry['base_var_min'] = float(_detach(_bv.min()))
                debug_entry['base_var_p05'] = _p(_bv, 0.05)
                debug_entry['base_var_p50'] = _p(_bv, 0.50)
                debug_entry['base_var_p95'] = _p(_bv, 0.95)
                debug_entry['base_var_max'] = float(_detach(_bv.max()))
                debug_entry['base_var_zero_frac'] = float(_bz)
                debug_entry['ratio_mean'] = float(_detach(_rf.mean()))
                debug_entry['ratio_min'] = float(_detach(_rf.min()))
                debug_entry['ratio_p05'] = _p(_rf, 0.05)
                debug_entry['ratio_p50'] = _p(_rf, 0.50)
                debug_entry['ratio_p95'] = _p(_rf, 0.95)
                debug_entry['ratio_max'] = float(_detach(_rf.max()))
                debug_entry['ratio_abs_max'] = float(_detach(_rf.abs().max()))
                # Before/after ratio (from _ratio_before)
                if _ratio_before is not None:
                    _rbf = _ratio_before.float()
                    debug_entry['ratio_before_p50'] = _p(_rbf, 0.50)
                    debug_entry['ratio_before_p95'] = _p(_rbf, 0.95)
                    debug_entry['ratio_before_abs_max'] = float(_detach(_rbf.abs().max()))
                else:
                    debug_entry['ratio_before_p50'] = None
                    debug_entry['ratio_before_p95'] = None
                    debug_entry['ratio_before_abs_max'] = None
                # After ratio
                _ra = stats[4].float()
                debug_entry['ratio_after_p50'] = _p(_ra, 0.50)
                debug_entry['ratio_after_p95'] = _p(_ra, 0.95)
                debug_entry['ratio_after_abs_max'] = float(_detach(_ra.abs().max()))
                debug_entry['ratio_max'] = float(_detach(_rf.max()))
                debug_entry['ratio_abs_max'] = float(_detach(_rf.abs().max()))
            else:
                for k in ['base_var_mean','base_var_min','base_var_p05','base_var_p50','base_var_p95','base_var_max','base_var_zero_frac',
                          'ratio_mean','ratio_min','ratio_p05','ratio_p50','ratio_p95','ratio_max','ratio_abs_max']:
                    debug_entry[k] = None
                debug_entry['ratio_before_p50'] = None
                debug_entry['ratio_before_p95'] = None
                debug_entry['ratio_before_abs_max'] = None
                debug_entry['ratio_after_p50'] = None
                debug_entry['ratio_after_p95'] = None
                debug_entry['ratio_after_abs_max'] = None

            # Fill logits_after with actual logits (same as logits in default mode)
            debug_entry['logits_after_ablation_mean'] = debug_entry.get('logits_mean', None)
            debug_entry['logits_after_ablation_p50'] = debug_entry.get('logits_p50', None)
            debug_entry['logits_after_ablation_p95'] = debug_entry.get('logits_p95', None)
            if self._ablation_mode == 'high_mean_proxy' and _proxy_logits is not None:
                try:
                    _pl = _proxy_logits.float()
                    debug_entry['logits_after_ablation_mean'] = float(_detach(_pl.mean()))
                    debug_entry['logits_after_ablation_p50'] = _p(_pl, 0.50)
                    debug_entry['logits_after_ablation_p95'] = _p(_pl, 0.95)
                except Exception:
                    pass

            self._cached_trigger_debug.append(debug_entry)
        except Exception:
            self._cached_trigger_debug.append(dict(stage_idx=int(stage_idx), error='debug_collection_failed'))

        return gate, gate_map

    def get_residual_scales(self):
        if self.residual_scale is None:
            return [1.0 for _ in range(self.num_stages)]
        return [float(p.detach().cpu().item()) for p in self.residual_scale]

    def get_trigger_stats(self):
        out = dict(enabled_stage_indices=self.enabled_stage_indices)
        for i in range(self.num_stages):
            if i >= len(self._cached_trigger_scores) or self._cached_trigger_scores[i] is None:
                out[f'stage{i}_trigger_mean'] = 0.0
                out[f'stage{i}_trigger_max'] = 0.0
                continue
            score = self._cached_trigger_scores[i]
            out[f'stage{i}_trigger_mean'] = float(score.detach().mean().cpu().item())
            out[f'stage{i}_trigger_max'] = float(score.detach().max().cpu().item())
        return out

    def get_cached_trigger_maps(self):
        """Return detached CPU copies of per-stage trigger maps.
        
        Returns:
            list[Tensor|None]: length == num_stages. Each entry is a CPU tensor
            or None if no trigger was computed for that stage.
        """
        out = []
        for m in self._cached_trigger_maps:
            if m is None:
                out.append(None)
            else:
                out.append(m.detach().cpu().clone())
        return out

    def get_cached_trigger_scores(self):
        """Return detached CPU copies of per-stage trigger scores.

        Returns:
            list[Tensor|None]: length == num_stages. Each entry is a CPU tensor
            or None if no trigger was computed for that stage.
        """
        out = []
        for s in self._cached_trigger_scores:
            if s is None:
                out.append(None)
            else:
                out.append(s.detach().cpu().clone())
        return out

    def get_cached_trigger_debug(self):
        """Return a shallow copy of the cached debug info list.
        
        Returns:
            list[dict]: length == num_stages. Each dict contains Python scalars
            describing the trigger computation for that stage.
        """
        return [dict(d) for d in self._cached_trigger_debug]

    def get_regularization_losses(self, prefix='fusion'):
        losses = {}
        if self.use_learnable_scale and self.regularize_residual_scale and self.residual_reg_weight > 0:
            reg = 0.0
            cnt = 0
            for i in range(self.num_stages):
                if i not in self.enabled_stage_indices:
                    continue
                reg = reg + self.residual_scale[i].abs()
                cnt += 1
            if cnt > 0:
                losses[f'{prefix}.loss_residual_reg'] = reg / cnt * self.residual_reg_weight

        valid_scores = [s for s in self._cached_trigger_scores if s is not None]
        if self.enable_high_trigger and len(valid_scores) > 0:
            if self.regularize_trigger_sparse and self.trigger_sparse_weight > 0:
                sparse = sum([s.mean() for s in valid_scores]) / len(valid_scores)
                losses[f'{prefix}.loss_trigger_sparse'] = sparse * self.trigger_sparse_weight

            if self.regularize_trigger_budget and self.trigger_budget_weight > 0:
                budget = 0.0
                cnt = 0
                for i, s in enumerate(self._cached_trigger_scores):
                    if s is None:
                        continue
                    target = self.trigger_budget_targets[i].to(s.device)
                    budget = budget + (s.mean() - target).pow(2)
                    cnt += 1
                if cnt > 0:
                    losses[f'{prefix}.loss_trigger_budget'] = budget / cnt * self.trigger_budget_weight

            if self.regularize_trigger_smooth and self.trigger_smooth_weight > 0:
                smooth = 0.0
                cnt = 0
                for m in self._cached_trigger_maps:
                    if m is None:
                        continue
                    dh = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
                    dw = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
                    smooth = smooth + dh + dw
                    cnt += 1
                if cnt > 0:
                    losses[f'{prefix}.loss_trigger_smooth'] = smooth / cnt * self.trigger_smooth_weight
        return losses

    def forward(self,
                rgb_feats,
                depth_feats,
                high_depth_inputs=None,
                base_depth_inputs=None):
        """Fuse RGB feats with base depth feats, optionally modulated by high depth.

        Args:
            rgb_feats (list[Tensor]): RGB backbone multi-scale features.
            depth_feats (list[Tensor]): base depth backbone multi-scale features.
            high_depth_inputs (Tensor|None): raw high-frequency depth map [B,1,H,W] or [B,H,W].
            base_depth_inputs (Tensor|None): raw base depth map [B,1,H,W] or [B,H,W], used only for high fallback.
        """
        fused_feats = []
        self._cached_trigger_scores = []
        self._cached_trigger_maps = []
        self._cached_enabled_mask = []
        self._cached_trigger_debug = []

        # --- Smoke-only ablation (does NOT affect default forward math) ---
        self._ablation_mode = 'none'       # 'none'|'no_ratio'|'ratio_clamp'|'ratio_log'|'high_mean_proxy'
        self._ablation_ratio_clamp: Optional[float] = None
        self._ablation_proxy_gate: bool = False

        fallback_high = None
        if self.enable_high_trigger and high_depth_inputs is None and base_depth_inputs is not None:
            fallback_high = self._build_fallback_high(base_depth_inputs)

        for i in range(self.num_stages):
            rgb = rgb_feats[i]
            depth = depth_feats[i]
            B, C, H, W = rgb.shape

            if i not in self.enabled_stage_indices:
                fused_feats.append(rgb)
                self._cached_trigger_scores.append(None)
                self._cached_trigger_maps.append(None)
                self._cached_enabled_mask.append(False)
                continue

            rgb_windows, Hp, Wp = self.window_partition(rgb)
            depth_windows, _, _ = self.window_partition(depth)

            attn_out, _ = self.attn[i](
                query=rgb_windows,
                key=depth_windows,
                value=depth_windows,
                need_weights=False,
            )

            trigger_scores = None
            trigger_map = None
            if self.enable_high_trigger:
                src_high = high_depth_inputs if high_depth_inputs is not None else fallback_high
                if src_high is not None:
                    high_map = self._resize_like(src_high, rgb)
                    trigger_scores, trigger_map = self._compute_window_trigger(i, high_map, depth)
                    attn_out = attn_out * (1.0 + self.event_scale_attn[i] * trigger_scores.unsqueeze(1))

            attn_out = self.proj_out[i](attn_out)
            attn_out = self.window_reverse(attn_out, Hp, Wp, B, C)
            attn_out = attn_out[:, :, :H, :W]

            if self.use_learnable_scale:
                depth_residual = self.residual_scale[i] * attn_out
            else:
                depth_residual = attn_out

            if trigger_map is not None:
                depth_residual = depth_residual * (1.0 + self.event_scale_res[i] * trigger_map)

            fused = rgb + depth_residual
            fused_feats.append(fused)

            self._cached_trigger_scores.append(trigger_scores)
            self._cached_trigger_maps.append(trigger_map)
            self._cached_enabled_mask.append(True)

        # --- Post-hoc: enrich debug entries with forward-level info ---
        try:
            hd_src = 'aux_high_inputs' if high_depth_inputs is not None else ('fallback' if fallback_high is not None else 'none')
            for de in self._cached_trigger_debug:
                if isinstance(de, dict) and 'error' not in de:
                    de['used_high_depth_inputs'] = (high_depth_inputs is not None or fallback_high is not None)
                    de['high_depth_source'] = hd_src
                    de['aux_inputs_shape'] = list(base_depth_inputs.shape) if base_depth_inputs is not None and isinstance(base_depth_inputs, torch.Tensor) else None
                    de['aux_high_inputs_shape'] = list(high_depth_inputs.shape) if high_depth_inputs is not None and isinstance(high_depth_inputs, torch.Tensor) else None
        except Exception:
            pass

        return fused_feats
