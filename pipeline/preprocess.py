"""Image preprocessing for the Hán OCR (step 3).

Old scanned pages OCR better after a light cleanup: grayscale, deskew (straighten
a slightly rotated scan), denoise, and binarise. Everything is optional and
controlled by `config.PREPROCESS`.

Public API:
    preprocess_array(bgr, cfg=None) -> np.ndarray           # numpy in, numpy out

OpenCV (`cv2`) is imported lazily so the rest of the pipeline works without it.
"""
from __future__ import annotations

from . import config


def _deskew(gray, cv2, np):
    """Estimate small skew from dark pixels and rotate to straighten."""
    inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inv > 0))
    if coords.shape[0] < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:   # ignore noise / implausible skew
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_array(bgr, cfg: dict | None = None):
    import cv2
    import numpy as np

    cfg = cfg or config.PREPROCESS
    img = bgr
    if cfg.get("grayscale", True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    up = cfg.get("upscale_min_height", 0)
    if up and img.shape[0] < up:
        scale = up / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if cfg.get("deskew", True) and img.ndim == 2:
        img = _deskew(img, cv2, np)

    if cfg.get("denoise", True):
        img = cv2.fastNlMeansDenoising(img, h=10) if img.ndim == 2 \
            else cv2.fastNlMeansDenoisingColored(img, h=10)

    mode = cfg.get("binarize")
    if mode and img.ndim == 2:
        if mode == "otsu":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif mode == "adaptive":
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 15)
    return img
