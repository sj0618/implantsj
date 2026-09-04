"""Standalone evaluation add-on for band_vit_3.

Feeds a band-trained ViT checkpoint either band-only crops or the original
unmasked pixels (--input-mode) and reports attention/rollout band-focus
statistics. Reuses band_vit_3.segmentation unchanged; the original
band_vit_3 package is not modified.
"""

__all__ = ["evaluation", "gradient_rollout", "band_preprocessing", "data"]
