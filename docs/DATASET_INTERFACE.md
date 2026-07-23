# Dataset interface

The dataset is distributed separately from this code release. The dataset package must provide aligned RGB images, 16-bit depth maps, the five-fold development annotations, and the fixed test-set annotation.

Place the extracted data under:

```text
mmdetection/data/seed1/
```

A typical layout is:

```text
data/seed1/
  images/        RGB images
  depth/         Pixel-aligned 16-bit depth PNG files
  annotations/   Five-fold split files and the fixed test annotation
```

The exact annotation filenames are defined by the separate dataset release. The provided configurations use `annotations/train.json` and `annotations/val.json` as local aliases. For each run, either copy the selected fold files to those aliases or edit the `ann_file` fields in the configuration.

## Categories

The paper benchmark uses eight categories:

```text
shells, shellL, link, box, left, right, gearL, gearS
```

The early development-only test label `little` is not part of the released benchmark or reported results.

## Depth format

- 16-bit grayscale PNG
- depth values encoded in millimetres
- pixel-aligned with the corresponding RGB image

## Annotation format

- COCO instance-segmentation JSON
- category names and category IDs must match the selected configuration
