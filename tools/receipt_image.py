"""Caller preprocessing for the pinned TinyReceiptVQA grayscale model."""
from pathlib import Path

import numpy as np


def prepare_receipt_image(path: str | Path) -> np.ndarray:
    """Decode media, resize with Pillow bilinear, and return independent F32 pixels.

    This model consumes NCHW [1,1,320,672] in [-1,1]. Pixel processing remains
    application policy rather than an engine graph operation or media RPC.
    """
    from PIL import Image
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((672, 320), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.ascontiguousarray(((gray / np.float32(255) - np.float32(.5)) / np.float32(.5))[None, None])
