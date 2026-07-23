_base_ = ['../mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py']

custom_imports = dict(
    imports=[
        'projects.anonymous_rgbd.rgbd_fuse.depth_pro',
        'projects.anonymous_rgbd.rgbd_fuse.model_entry',
        'projects.anonymous_rgbd.rgbd_fuse.model_fusion',
    ],
    allow_failed_imports=False
)

# =========================
# 0) 瀹為獙鎬诲紑鍏?# =========================

# ---------------------------------
# depth / fusion 鎬诲紑鍏?# ---------------------------------
enable_depth_branch = True
enable_fusion_module = True

# ---------------------------------
# stage2 涓撶敤鏁版嵁閾捐矾
# ---------------------------------
enable_stage2_data_chain = True

# ---------------------------------
# head 绫诲瀷閫夋嫨
# ---------------------------------
use_dual_decoder_head = True

# ---------------------------------
# stage2 涓诲姛鑳藉紑鍏?# ---------------------------------
enable_stage2_decoder = True
enable_stage2_loss = True
enable_stage2_loss_backward = True
use_stage2_output = True
enable_low_freq_prior = True
detach_stage1_for_stage2 = True

# ---------------------------------
# stage2 杩愯 / 璇勪及琛屼负
# ---------------------------------
run_stage2_side_forward = False
return_both_stage_predictions = True
log_stage2_replace_stats = True
eval_use_stage2_for_metrics = True

# ---------------------------------
# refinement-only 鐩稿叧
# ---------------------------------
enable_stage2_refine_only = True
stage2_refine_last_layer_only = True

# 褰撳墠闃舵锛歴tage2 缁х画浠?mask/localization refine 涓轰富
# 鏈€缁堝垎绫讳粛娌跨敤 stage1锛岄伩鍏?backbone fuse 涓?stage2 鍚屾椂鍦?cls 渚у紩鍏ヨ繃澶氬彉閲?stage2_refine_use_stage1_cls_for_output = True

# 璐ㄩ噺鎰熺煡鍙傛暟
stage2_quality_cls_power = 0.25
stage2_quality_mask_power = 0.75

# 褰撳墠涓嶅己璋?stage2 cls 瀛︿範锛岄伩鍏嶅拰鈥渓ow-freq mainly serves refinement鈥濆啿绐?stage2_cls_loss_weight = 0.0
stage2_depth_to_cls_branch = False

stage2_mask_residual_scale = 1.0

# ---------------------------------
# stage2 涓?shared trunk 鐨勮€﹀悎
# ---------------------------------
stage2_detach_shared_trunk = False

# ---------------------------------
# depth-guided query routing
# ---------------------------------
enable_depth_guided_query_routing = True
stage2_query_select_by_depth = True
stage2_query_select_by_uncertainty = True
stage2_query_select_by_score = True
stage2_query_depth_weight = 1.0
stage2_query_uncertainty_weight = 0.5
stage2_query_lowconf_weight = 0.5
stage2_query_select_topk = 30
stage2_query_score_thr = -1.0
stage2_query_min_kept = 1
stage2_use_selected_queries_for_output = True

# ---------------------------------
# depth 琛ㄧず鏂瑰紡
# ---------------------------------
depth_enable_freq_split = True
depth_output_mode = 'base'
depth_low_key = 'aux_img_low'
depth_high_key = 'aux_img_high'

# ---------------------------------
# stage2 loss 鏉冮噸
# ---------------------------------
stage2_loss_weight = 0.02

# ---------------------------------
# 姊害绱Н锛氬崟鍗?batch=2锛岀疮璁?8 娆?=> 鏈夋晥 batch 鈮?16
# ---------------------------------
accumulative_counts = 8


# =========================
# 1) 鏁版嵁闆嗕笌绫诲埆
# =========================
classes = (
    'shells', 'shellL', 'link', 'box', 'left',
    'right', 'gearL', 'gearS'
)

num_things_classes = len(classes)
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes
dataset_type = 'CocoDataset'
data_root = 'data/seed1/'


# =========================
# 2) 鍥惧儚灏哄
# =========================
image_size = (1024, 1024)


# =========================
# 3) Backbone 閰嶇疆
# =========================
rgb_backbone_cfg = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
)

aux_backbone_cfg = dict(
    type='ResNet',
    depth=50,
    in_channels=1,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
)


# =========================
# 4) Head 閫夋嫨
# =========================
if use_dual_decoder_head:
    panoptic_head_cfg = dict(
        type='DualDecoderMask2FormerHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,

        # ---- stage2 涓诲姛鑳?----
        enable_stage2_decoder=enable_stage2_decoder,
        enable_stage2_loss=enable_stage2_loss,
        enable_stage2_loss_backward=enable_stage2_loss_backward,
        use_stage2_output=use_stage2_output,
        enable_low_freq_prior=enable_low_freq_prior,
        stage2_loss_weight=stage2_loss_weight,
        stage2_num_layers=3,
        stage1_mid_layer_index=-2,
        detach_stage1_for_stage2=detach_stage1_for_stage2,
        stage2_forward_no_grad_when_no_loss=True,
        preserve_stage1_path_when_stage2_inactive=True,
        run_stage2_side_forward=run_stage2_side_forward,
        return_both_stage_predictions=return_both_stage_predictions,

        # ---- refinement-only ----
        enable_stage2_refine_only=enable_stage2_refine_only,
        stage2_refine_last_layer_only=stage2_refine_last_layer_only,
        stage2_refine_use_stage1_cls_for_output=stage2_refine_use_stage1_cls_for_output,
        stage2_mask_residual_scale=stage2_mask_residual_scale,

        # ---- quality-aware / localization-first ----
        stage2_quality_cls_power=stage2_quality_cls_power,
        stage2_quality_mask_power=stage2_quality_mask_power,
        stage2_cls_loss_weight=stage2_cls_loss_weight,
        stage2_depth_to_cls_branch=stage2_depth_to_cls_branch,

        # ---- trunk coupling ----
        stage2_detach_shared_trunk=stage2_detach_shared_trunk,

        # ---- query routing ----
        enable_depth_guided_query_routing=enable_depth_guided_query_routing,
        stage2_query_select_by_depth=stage2_query_select_by_depth,
        stage2_query_select_by_uncertainty=stage2_query_select_by_uncertainty,
        stage2_query_select_by_score=stage2_query_select_by_score,
        stage2_query_depth_weight=stage2_query_depth_weight,
        stage2_query_uncertainty_weight=stage2_query_uncertainty_weight,
        stage2_query_lowconf_weight=stage2_query_lowconf_weight,
        stage2_query_select_topk=stage2_query_select_topk,
        stage2_query_score_thr=stage2_query_score_thr,
        stage2_query_min_kept=stage2_query_min_kept,
        stage2_use_selected_queries_for_output=stage2_use_selected_queries_for_output,

        # ---- stage2 query 鍒濆鍖?----
        init_inter_query_scale=0.0,
        init_depth_query_scale=0.0,

        # ---- eval 琛屼负 ----
        log_stage2_replace_stats=log_stage2_replace_stats,
        eval_use_stage2_for_metrics=eval_use_stage2_for_metrics,

        # ---- low-freq adapter ----
        low_freq_depth_adapter=dict(
            type='LowFreqDepthAdapter',
            in_channels=1,
            out_channels=256,
            mid_channels=64,
            downsample_factor=4,
            use_bn=False,
        ),

        # 涓?decoder 杞婚噺 dropout
        transformer_decoder=dict(
            layer_cfg=dict(
                self_attn_cfg=dict(dropout=0.05),
                cross_attn_cfg=dict(dropout=0.05),
                ffn_cfg=dict(ffn_drop=0.05),
            )
        ),

        # pixel decoder encoder 杞婚噺 dropout
        pixel_decoder=dict(
            encoder=dict(
                layer_cfg=dict(
                    self_attn_cfg=dict(dropout=0.05),
                    ffn_cfg=dict(ffn_drop=0.05),
                )
            )
        ),

        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
    )
else:
    panoptic_head_cfg = dict(
        type='Mask2FormerHeadProbe',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
    )


# =========================
# 5) DataPreprocessor 閫夋嫨
# =========================
if enable_stage2_data_chain:
    data_preprocessor_cfg = dict(
        type='DualInputDetDataPreprocessorStage2',
        fixed_pad_size=image_size,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
    )
else:
    data_preprocessor_cfg = dict(
        type='DualInputDetDataPreprocessor',
        fixed_pad_size=image_size,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
    )


# =========================
# 6) fusion 妯″潡閫夋嫨锛圧GT-WCA锛?# =========================
if enable_fusion_module and enable_depth_branch:
    fusion_module_cfg = dict(
        type='DepthGuidedWindowCrossAttention',
        in_channels=[256, 512, 1024, 2048],
        num_heads=8,
        dropout=0.1,
        attn_init_scale=0.0,
        window_size=8,
        use_learnable_scale=True,
        init_residual_scale=0.01,

        # ---- RGT-WCA: 浠呭湪娴?涓眰鍚敤鏇村己鍑犱綍浠嬪叆 ----
        enabled_stage_indices=(0, 1, 2, 3),

        # ---- high-frequency 鍙仛 trigger锛屼笉鍋?K/V 鍐呭 ----
        enable_high_trigger=True,
        trigger_hidden_dim=32,
        trigger_use_base_consistency=True,
        trigger_temperature=1.5,
        trigger_threshold=0.0,
        trigger_power=1.0,
        init_event_scale_attn=0.0,
        init_event_scale_res=0.5,

        # ---- 姝ｅ垯锛氱害鏉?depth 浠嬪叆骞呭害涓?trigger 鍒嗗竷 ----
        regularize_residual_scale=True,
        residual_reg_weight=1e-4,

        regularize_trigger_sparse=True,
        trigger_sparse_weight=2e-4,

        regularize_trigger_budget=True,
        trigger_budget_weight=2e-4,
        trigger_budget_targets=[0.30, 0.20, 0.10, 0.05],

        regularize_trigger_smooth=False,
        trigger_smooth_weight=1e-4,

        # ---- 鑻ュ綋鍓嶆暟鎹摼璺病鏈夋樉寮忎紶 aux_high_inputs锛屽垯鐢?aux_inputs 鍦ㄧ嚎鍥為€€鏋勯€?high ----
        fallback_high_kernel_size=5,
    )
else:
    fusion_module_cfg = None


# =========================
# 7) 妯″瀷閰嶇疆
# =========================
model = dict(
    type='TwoStageWrapperMask2Former',
    debug_stop_after_fusion=False,
    return_both_stage_predictions=return_both_stage_predictions,
    enable_auxiliary_branch=enable_depth_branch,
    data_preprocessor=data_preprocessor_cfg,
    backbone=dict(_delete_=True, type='IdentityBackbone'),
    feature_extractor=dict(
        type='DualStageFeatureExtractor',
        rgb_backbone_cfg=rgb_backbone_cfg,
        aux_backbone_cfg=aux_backbone_cfg if enable_depth_branch else None,
        enable_auxiliary_branch=enable_depth_branch,
    ),
    fusion_module=fusion_module_cfg,
    panoptic_head=panoptic_head_cfg,
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes
    ),
    test_cfg=dict(panoptic_on=False)
)


# =========================
# 8) depth preprocess 鍏叡閰嶇疆
# =========================
depth_preprocess_cfg = dict(
    type='DepthPreprocess',
    key='aux_img',
    to_float32=True,
    squeeze_if_single_channel=True,
    to_gray=False,
    invalid_min=2.0,
    invalid_fill_value=0.0,
    median_ksize=3,
    percentile_clip=(2.0, 98.0),
    normalize_mode='minmax_on_valid',
    apply_log1p=False,
    enable_freq_split=depth_enable_freq_split,
    low_key=depth_low_key,
    high_key=depth_high_key,
    output_mode=depth_output_mode,
    low_mode='gaussian',
    gaussian_ksize=5,
    gaussian_sigma=0.0,
    high_mode='residual_abs',
    high_percentile_clip=(1.0, 99.0),
    normalize_low=True,
    normalize_high=True
)


# =========================
# 9) Pipeline 閫夋嫨
# =========================
if enable_stage2_data_chain:
    train_pipeline = [
        dict(
            type='LoadImageFromFile',
            to_float32=True,
            backend_args={{_base_.backend_args}}
        ),
        dict(
            type='LoadAuxImageFromFile',
            aux_folder='depth',
            aux_suffix='.png',
            to_float32=True,
            debug=True,
            flag='unchanged',
            imdecode_backend='cv2'
        ),
        depth_preprocess_cfg,
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='DualRandomFlipStage2', prob=0.5, direction='horizontal'),
        dict(
            type='DualRandomResizeStage2',
            scale=image_size,
            ratio_range=(0.1, 2.0),
            keep_ratio=True,
            interpolation='bilinear'
        ),
        dict(
            type='DualRandomCropStage2',
            crop_size=image_size,
            crop_type='absolute',
            recompute_bbox=True,
            allow_negative_crop=True
        ),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
        dict(
            type='PackDualDetInputsStage2',
            meta_keys=(
                'img_id', 'img_path', 'aux_img_path',
                'ori_shape', 'img_shape', 'aux_img_shape',
                'scale_factor', 'flip', 'flip_direction'
            )
        )
    ]

    val_test_pipeline = [
        dict(
            type='LoadImageFromFile',
            to_float32=True,
            backend_args={{_base_.backend_args}}
        ),
        dict(
            type='LoadAuxImageFromFile',
            aux_folder='depth',
            aux_suffix='.png',
            to_float32=True,
            debug=True,
            flag='unchanged',
            imdecode_backend='cv2'
        ),
        depth_preprocess_cfg,
        dict(
            type='DualResize',
            scale=(1333, 800),
            keep_ratio=True,
            interpolation='bilinear'
        ),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            type='PackDualDetInputsStage2',
            meta_keys=(
                'img_id', 'img_path', 'aux_img_path',
                'ori_shape', 'img_shape', 'aux_img_shape',
                'scale_factor'
            )
        )
    ]
else:
    train_pipeline = [
        dict(
            type='LoadImageFromFile',
            to_float32=True,
            backend_args={{_base_.backend_args}}
        ),
        dict(
            type='LoadAuxImageFromFile',
            aux_folder='depth',
            aux_suffix='.png',
            to_float32=True,
            debug=True,
            flag='unchanged',
            imdecode_backend='cv2'
        ),
        depth_preprocess_cfg,
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='DualRandomFlip', prob=0.5, direction='horizontal'),
        dict(
            type='DualRandomResize',
            scale=image_size,
            ratio_range=(0.8, 1.2),
            keep_ratio=True,
            interpolation='bilinear'
        ),
        dict(
            type='DualRandomCrop',
            crop_size=image_size,
            crop_type='absolute',
            recompute_bbox=True,
            allow_negative_crop=True
        ),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
        dict(
            type='PackDualDetInputs',
            meta_keys=(
                'img_id', 'img_path', 'aux_img_path',
                'ori_shape', 'img_shape', 'aux_img_shape',
                'scale_factor', 'flip', 'flip_direction'
            )
        )
    ]

    val_test_pipeline = [
        dict(
            type='LoadImageFromFile',
            to_float32=True,
            backend_args={{_base_.backend_args}}
        ),
        dict(
            type='LoadAuxImageFromFile',
            aux_folder='depth',
            aux_suffix='.png',
            to_float32=True,
            debug=True,
            flag='unchanged',
            imdecode_backend='cv2'
        ),
        depth_preprocess_cfg,
        dict(
            type='DualResize',
            scale=(1333, 800),
            keep_ratio=True,
            interpolation='bilinear'
        ),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(
            type='PackDualDetInputs',
            meta_keys=(
                'img_id', 'img_path', 'aux_img_path',
                'ori_shape', 'img_shape', 'aux_img_shape',
                'scale_factor'
            )
        )
    ]


# =========================
# 10) Dataloader
# =========================
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        metainfo=dict(classes=classes),
        pipeline=val_test_pipeline
    )
)

test_dataloader = val_dataloader


# =========================
# 11) Evaluator
# =========================
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=False,
    backend_args={{_base_.backend_args}}
)
test_evaluator = val_evaluator


# =========================
# 12) Training loop
# =========================
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=80000,
    val_interval=2000
)
val_cfg = dict(_delete_=True, type='ValLoop')
test_cfg = dict(_delete_=True, type='TestLoop')


# =========================
# 13) Hooks
# =========================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=20,
        log_metric_by_epoch=False
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=20,
        save_best='coco/segm_mAP',
        rule='greater',
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

log_processor = dict(
    type='LogProcessor',
    window_size=1,
    by_epoch=False
)

custom_hooks = []
if enable_depth_branch:
    custom_hooks.append(dict(type='LogDepthUsageHook', eps=1e-6))


# =========================
# 14) 鍙鍖?/ 鏃ュ織淇濆瓨
# =========================
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)


# =========================
# 15) 浼樺寲鍣?# =========================
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0),

    clip_grad=dict(max_norm=0.5, norm_type=2),
    accumulative_counts=accumulative_counts
)

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=80000,
    by_epoch=False,
    milestones=[65778, 70092],
    gamma=0.1
)

auto_scale_lr = dict(enable=False, base_batch_size=16)


# =========================
# 16) work_dir 鑷姩鍛藉悕
# =========================
exp_name = 'rgt_wca_basebody_hightrigger_stage2_lowprior_bs2_acc8'

load_from = None
resume = False
work_dir = f'./work_dirs/train_two_stage_wrapper_rgbd_{exp_name}'


# =========================
# 17) Analysis / Interpretability Dump
# =========================
analysis_output_mode = 'stage2_adopted'
# options:
#   'stage1'
#   'stage2_raw'
#   'stage2_adopted'

save_analysis_dump = False

analysis_dir = work_dir + '/analysis'
analysis_dump_path = analysis_dir + '/full_model_analysis.jsonl'
analysis_fusion_map_dir = analysis_dir + '/fusion_maps'

model['test_cfg'].update(
    dict(
        analysis_output_mode=analysis_output_mode,
        save_analysis_dump=save_analysis_dump,
        analysis_dump_path=analysis_dump_path,
        analysis_fusion_map_dir=analysis_fusion_map_dir,
        analysis_save_fusion_maps=True,
        analysis_save_masks_rle=False,
        analysis_max_predictions=100,
    )
)