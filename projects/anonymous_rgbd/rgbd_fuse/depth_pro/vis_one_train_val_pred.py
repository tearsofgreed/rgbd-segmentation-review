import os
import argparse
import importlib

import torch
import mmcv
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS, MODELS, VISUALIZERS
from mmengine.registry import init_default_scope


def import_custom_modules_from_cfg(cfg):
    """根据 config 中的 custom_imports 显式导入自定义模块。"""
    custom_imports = cfg.get('custom_imports', None)
    if custom_imports is None:
        return

    imports = custom_imports.get('imports', [])
    allow_failed_imports = custom_imports.get('allow_failed_imports', False)

    for module_name in imports:
        try:
            importlib.import_module(module_name)
            print(f'[OK] imported custom module: {module_name}')
        except Exception as e:
            if allow_failed_imports:
                print(f'[WARN] failed to import {module_name}: {e}')
            else:
                raise


def build_model_from_cfg(cfg, checkpoint=None, device='cuda'):
    # 初始化默认 scope，保证 mmdet registry 正常工作
    init_default_scope('mmdet')

    # 显式导入自定义模块，注册到 registry
    import_custom_modules_from_cfg(cfg)

    model = MODELS.build(cfg.model)
    model.cfg = cfg

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')

    model.to(device)
    model.eval()
    return model


def build_dataset(dataset_cfg):
    return DATASETS.build(dataset_cfg)


def run_one_sample(model, dataset, index, out_file, device='cuda', score_thr=0.1):
    sample = dataset[index]
    batch = pseudo_collate([sample])

    with torch.no_grad():
        outputs = model.test_step(batch)

    pred_data_sample = outputs[0]
    img_path = pred_data_sample.metainfo['img_path']
    img = mmcv.imread(img_path, channel_order='rgb')

    visualizer = VISUALIZERS.build(
        dict(type='DetLocalVisualizer', name='visualizer')
    )
    visualizer.dataset_meta = dataset.metainfo

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    visualizer.add_datasample(
        name=os.path.basename(out_file),
        image=img,
        data_sample=pred_data_sample,
        draw_gt=False,
        draw_pred=True,
        pred_score_thr=score_thr,
        out_file=out_file
    )

    print(f'[OK] saved visualization -> {out_file}')
    print(f'     img_path: {img_path}')

    if hasattr(pred_data_sample, 'pred_instances'):
        pred_instances = pred_data_sample.pred_instances
        if hasattr(pred_instances, 'scores'):
            print(f'     num_pred_instances: {len(pred_instances.scores)}')
            if len(pred_instances.scores) > 0:
                print(
                    f'     score range: '
                    f'{float(pred_instances.scores.min()):.4f} ~ '
                    f'{float(pred_instances.scores.max()):.4f}'
                )
    else:
        print('     no pred_instances found')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config path')
    parser.add_argument('checkpoint', help='checkpoint path')
    parser.add_argument('--train-index', type=int, default=0)
    parser.add_argument('--val-index', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='debug_vis')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--score-thr', type=float, default=0.1)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    model = build_model_from_cfg(cfg, args.checkpoint, device=args.device)

    train_dataset = build_dataset(cfg.train_dataloader['dataset'])
    val_dataset = build_dataset(cfg.val_dataloader['dataset'])

    run_one_sample(
        model=model,
        dataset=train_dataset,
        index=args.train_index,
        out_file=os.path.join(args.out_dir, 'train_sample_pred.png'),
        device=args.device,
        score_thr=args.score_thr
    )

    run_one_sample(
        model=model,
        dataset=val_dataset,
        index=args.val_index,
        out_file=os.path.join(args.out_dir, 'val_sample_pred.png'),
        device=args.device,
        score_thr=args.score_thr
    )


if __name__ == '__main__':
    main()
