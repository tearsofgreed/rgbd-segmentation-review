# RGB-D Dataset for Double-Blind Review

This repository provides documentation and anonymous reviewer access to the
RGB-D dataset associated with a manuscript currently undergoing double-blind
peer review.

Author names, affiliations, institutions, funding information, and other
identifying information have been omitted during review.

## Dataset overview

The dataset contains:

- 4,400 paired RGB-D samples
- 17,317 annotated object instances
- eight object categories
- COCO-format instance segmentation annotations
- five predefined cross-validation folds
- one fixed independent test set

Each RGB image has a corresponding depth image with the same sample
identifier.

## Zenodo record

The dataset is archived in a restricted-access Zenodo record:

[Zenodo DOI: 10.5281/zenodo.21510488](https://doi.org/10.5281/zenodo.21510488)

The DOI page provides the public dataset metadata. The dataset files remain
restricted.

## Anonymous reviewer access

Anonymous reviewers can access the restricted files through the following
confidential link:

[Access the dataset files for anonymous peer review](https://zenodo.org/records/21510488?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIwNzM1ZjZiLTc3NDgtNDE3My1iYjhiLWVmZTQ5NGEzYzhlNyIsImRhdGEiOnt9LCJyYW5kb20iOiI0MzlhNzQzMzE1ZDA4MWY5MGQ1ZWQ1ZmI0MWZiZTljMyJ9.6zkQM-bM8nodgPzF869Nnrw8ZfQJWL8R3wOd3KmmnRcZdhGLUa7eOdl_u3OTdiW8TzlYVxs_JkJVc210fiPHWw

This link allows reviewers to access the files without submitting their names
or email addresses to the authors. It is intended only for confidential peer
review and should not be redistributed.

Other researchers may request access through the public Zenodo DOI page.

## Dataset files

Download all five files:

```text
depth.zip
rgb_png.z01
rgb_png.z02
rgb_png.zip
annotations.zip
```

The three RGB files form one split ZIP archive:

```text
rgb_png.z01
rgb_png.z02
rgb_png.zip
```

They are not independent archives.

## Extraction

### RGB images

1. Download `rgb_png.z01`, `rgb_png.z02`, and `rgb_png.zip`.
2. Place all three files in the same directory.
3. Do not rename the files.
4. Open or extract `rgb_png.zip`.
5. Do not extract `.z01` or `.z02` separately.

Use software that supports split ZIP archives, such as 7-Zip:

```bash
7z x rgb_png.zip
```

### Depth images and annotations

Extract the following archives normally:

```text
depth.zip
annotations.zip
```

The expected structure after extraction is:

```text
dataset/
├── rgb_png/
├── depth/
└── annotations/
    ├── fold1_train.json
    ├── fold1_val.json
    ├── fold2_train.json
    ├── fold2_val.json
    ├── fold3_train.json
    ├── fold3_val.json
    ├── fold4_train.json
    ├── fold4_val.json
    ├── fold5_train.json
    ├── fold5_val.json
    └── test.json
```

The actual filenames contained in the archives should be treated as
authoritative.

## RGB-depth correspondence

RGB and depth images belonging to the same sample share the same sample
identifier.

For example:

```text
rgb_png/000001.png
depth/000001.png
```

Do not rename the images unless all corresponding annotation references are
updated consistently.

## Annotation format

The annotation files follow the COCO instance segmentation format and contain
the standard fields:

```text
images
annotations
categories
```

The JSON files determine which images belong to each training, validation,
or test split.

## Evaluation splits

The development set is organized into five predefined cross-validation
folds. Each fold contains one training JSON and one validation JSON.

A development sample may be used for training in one fold and validation in
another fold. This is expected in five-fold cross-validation.

The fixed independent test set is defined by:

```text
test.json
```

The test set should not be used for training, model selection,
hyperparameter tuning, early stopping, or threshold selection.

## Citation

During double-blind peer review, the dataset may be cited as:

```text
Anonymous Authors. (2026). RGB-D Dataset for Double-Blind Review
(Version 1) [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.21510488
```

The dataset DOI is:

```text
10.5281/zenodo.21510488
```

Creator information and the final dataset title may be updated after the
double-blind review process without changing this DOI.

## Access and redistribution

The dataset files are provided under restricted access. Researchers without
the confidential reviewer link may submit an access request through the
Zenodo record.

Access and reuse are governed by the rights and access conditions displayed
on the Zenodo record. Restricted files should not be redistributed without
the required permission.

## Acknowledgements

This work builds upon [MMDetection](https://github.com/open-mmlab/mmdetection),
an open-source object detection and instance segmentation toolbox developed
by the OpenMMLab project. We thank the MMDetection contributors for providing
the framework and implementations used in this work.

If you use the associated code, please also cite MMDetection:

```bibtex
@article{chen2019mmdetection,
  title   = {MMDetection: Open MMLab Detection Toolbox and Benchmark},
  author  = {Kai Chen and Jiaqi Wang and Jiangmiao Pang and Yuhang Cao and
             Yu Xiong and Xiaoxiao Li and Shuyang Sun and Wansen Feng and
             Ziwei Liu and Jiarui Xu and Zheng Zhang and Dazhi Cheng and
             Chenchen Zhu and Tianheng Cheng and Qijie Zhao and Buyu Li and
             Xin Lu and Rui Zhu and Yue Wu and Jifeng Dai and Jingdong Wang
             and Jianping Shi and Wanli Ouyang and Chen Change Loy and Dahua Lin},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
