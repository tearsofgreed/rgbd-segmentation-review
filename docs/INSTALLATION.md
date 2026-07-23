# Installation

This release is an overlay for the source checkout of MMDetection v3.3.0. Do not install `mmdet` separately with `mim install mmdet`; clone the v3.3.0 source repository so that the original configuration files and tools are available.

```bash
git clone --branch v3.3.0 --depth 1 https://github.com/open-mmlab/mmdetection.git
cd mmdetection

conda create -n rgbd-seg python=3.10 -y
conda activate rgbd-seg

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
pip install -v -e .
```

Copy `projects/anonymous_rgbd`, the released configuration directories, and `tools/robustness_external` into the corresponding MMDetection directories. Install the additional packages from `environment/requirements.txt`.

Run training and evaluation commands from the MMDetection repository root.
