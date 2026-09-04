# DINOv2 reproducibility boundary

The archived DINOv2 table contains 2456 image-linked, 384-dimensional
ViT-S/14 feature vectors. It supports reproduction of the downstream
no-standardization PCA, K-means, repeated-seed stability, and
ResNet50-DINOv2 agreement analyses.

The original raw-image DINOv2 extraction script was not retained.
Consequently, the exact checkpoint variant, raw-image resize/crop procedure,
normalization, and CLS-versus-patch pooling operation are not claimed as
known. DINOv2 is treated only as a secondary archived-feature sensitivity
comparison. The primary partitions are defined by the fully documented
ResNet50 workflow.
