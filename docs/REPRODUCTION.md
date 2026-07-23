# Reproduction

Complete the installation and overlay steps in the root `README.md`, download the separate dataset package, and select one of the released five-fold annotation pairs.

All commands below are run from the MMDetection v3.3.0 repository root.

## Main model

```bash
python tools/train.py configs/main/config_fusion_all_stage_strict.py
```

## RGB-only baseline

```bash
python tools/train.py configs/baselines/train_modular_fusion_rgb_only.py
```

## GT-Fusion baseline

```bash
python tools/train.py configs/baselines/train_modular_fusion_rgbd.py
```

## Evaluation

```bash
python tools/test.py CONFIG_FILE CHECKPOINT_FILE
```

## Ablations

The released ablation configurations are located in `configs/ablations/`.

## Robustness utilities

The supplementary robustness scripts are located in `tools/robustness_external/` after the overlay is copied into MMDetection.
