# Hankou Modern Historic Building Morphology

Reproducibility materials for:

**Multiscale Quantification of Visual Morphology in Modern Historic Buildings:
A Computer Vision Study of the Five Former Concessions in Hankou**

Manuscript ID: buildings-4519595

## Structure

- `code/01_OpenCV_Mask_and_Component_Descriptor_Extraction.py`
- `code/02_OpenCV_Record_Correction.py`
- `code/building_level_permutation/`
- `code/03_ResNet50_Feature_Extraction_Reproducible.py`
- `code/04_Unified_NoScaler_Clustering_and_Hierarchical_Sensitivity.ipynb`
- `data/`: small processed datasets, mapping, and corrected OpenCV workbook
- `results/`: final revised statistical outputs
- `docs/`: data dictionary, DINOv2 boundary, and release steps

## Large feature matrices

Download these from the associated Zenodo record and place them in `data/`:

- `Component_ResNet50_Features_2456x2048.csv`
- `Component_DINOv2_Features_2456x384.csv`

Zenodo: `ZENODO_DOI_TO_BE_ADDED`

## Primary deep-feature workflow

ResNet50 frozen features -> no per-dimension StandardScaler -> PCA 50
(full SVD) -> K-means with k-means++, n_init=20, max_iter=500,
tol=1e-4, random_state=42.

## OpenCV correction

Version 1.1 corrects six component area-ratio values in F030 and F072
because contour areas and denominator image areas were previously drawn from
different coordinate systems. The correction does not replace the coarse
OpenCV mask outputs with the separate high-precision deep-model crop dataset.

No facade-level or primary building-level OpenCV descriptor was significant
after BH-FDR correction.

## DINOv2

DINOv2 is a secondary archived-feature sensitivity comparison. The original
raw-image extraction script was not retained; see
`docs/DINOv2_REPRODUCIBILITY_BOUNDARY.md`.

## Original photographs

Original high-resolution field photographs are not included. Release them only
after ownership, privacy, heritage-site, and institutional review.
