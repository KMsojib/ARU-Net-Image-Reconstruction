# Dataset

This repository does **not** redistribute the rice-leaf image dataset. Follow
this guide to prepare your own copy locally.

## Source (as reported in the thesis)

- **Crop:** Rice leaves, collected under natural field conditions (not
  laboratory-controlled).
- **Raw image count:** 103 high-quality RGB images total -- 83 diseased, 20
  healthy (thesis Sec. 3.1.2).
- **Expanded (degraded) count:** 1,033 images, produced by applying 10
  noise/blur degradation types to each of the 103 originals (thesis Sec.
  3.1.10, Fig. 3.6).
- **Annotation:** Disease regions were annotated with **instance
  segmentation** polygons via Roboflow's Smart Polygon tool, exported in
  **COCO segmentation format**, and converted into binary disease masks
  (thesis Sec. 3.1.4-3.1.5).

## Important discrepancy: the annotation/degradation pipeline is not in the provided notebook

The thesis (Sec. 3.1.7.1) specifies a mask-guided degradation operator:

```
I_d = M * I + (1 - M) * D(I)
```

where `M` is the binary disease mask, `I` is the original image, and `D(.)`
is a degradation operator (noise or blur), so that disease regions are kept
pixel-perfect while only the background is degraded.

**No code cell in the supplied notebook (`4_AttentionResidualUNet.ipynb`)
implements this generation step.** The notebook only *consumes* already
-degraded image folders that must exist on disk beforehand. Rather than
inventing a degradation script that wasn't provided, this repository expects
you to supply pre-degraded folders (either from the thesis's original
Roboflow/COCO pipeline, or your own equivalent), using the layout below.

## Expected local layout

```
data/
├── clean/                          # original ("_original") images
│   ├── leaf01_original.jpg
│   ├── leaf02_original.jpg
│   └── ...
└── distorted/
    ├── Noise_Multiply_Strong/      # one folder per degradation type
    │   ├── leaf01_noise4_multiply_strong.jpg
    │   ├── leaf02_noise4_multiply_strong.jpg
    │   └── ...
    ├── Gaussian_Blur/
    ├── Median_Blur/
    ├── Bilateral_Blur/
    ├── Noise_Half_Split/
    ├── Noise_Half_Split_Strong/
    ├── Noise_Multiply/
    ├── Noise_Overlay/
    ├── Noise_Overlay_Strong/
    └── Normal_Noise/
```

Filenames are paired by string substitution: a distorted filename's
`dist_suffix` (e.g. `_noise4_multiply_strong`, configurable in
`configs/config.py`) is replaced with `clean_suffix` (default
`_original`) to find its ground-truth counterpart. Adjust
`configs/config.py`'s `DatasetConfig.dist_suffix` per degradation type when
switching folders (this mirrors the notebook, which hard-coded a single
suffix per run).

## Obtaining the dataset

Because the original 103 rice-leaf images and their COCO disease
annotations are not included in this repository (per Step 14 of the
project's data-handling requirements), you will need to either:

1. Use your own copy of the original dataset described in thesis Sec. 3.1.1
   (publicly available rice-leaf disease imagery), and reproduce the
   annotation/degradation steps described in thesis Sec. 3.1.4-3.1.9, or
2. Substitute any paired clean/degraded image dataset of your own, updating
   `configs/config.py` accordingly.

No proprietary, licensed, or personally identifiable image data should be
placed in this directory if you intend to publish the repository publicly.
