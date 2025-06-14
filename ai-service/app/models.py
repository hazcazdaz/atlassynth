"""
AtlasSynth core ML helpers.
Only very small baseline models are loaded; swap to larger ones later.
"""
from transformers import pipeline
from typing import Dict

# Basic land-cover classifier (zero-shot)
_CLASSIFIER = "openai/clip-vit-base-patch32"
classify = pipeline("zero-shot-image-classification", model=_CLASSIFIER)

# Depth estimator for satellite/street images
_DEPTH = "Intel/dpt-swinv2-tiny-256"
depth_est = pipeline("depth-estimation", model=_DEPTH)

def tag_tile(image) -> Dict:
    """Return top land-cover predictions for a PIL image."""
    labels = ["urban", "forest", "desert", "water", "farmland", "ice", "mountain"]
    preds = classify(image, candidate_labels=labels, top_k=3)
    return preds

def depth_map(image):
    """Return a normalized depth ndarray (H x W)"""
    return depth_est(image)["depth"]
