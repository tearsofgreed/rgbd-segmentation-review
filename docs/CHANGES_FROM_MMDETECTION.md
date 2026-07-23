# Changes from MMDetection v3.3.0

This release is an extension-only overlay. No MMDetection core source file is replaced or patched.

Custom components are located in `projects/anonymous_rgbd/rgbd_fuse/` and are registered through `custom_imports` in the released configuration files.

The release adds:

- aligned RGB-depth loading and synchronized transforms;
- depth preprocessing and frequency decomposition;
- dual-branch RGB-depth feature extraction;
- geometry-triggered RGB-D fusion;
- depth-guided query refinement;
- robustness evaluation utilities.

The original MMDetection training and testing entry points are used directly.
