"""Heuristic severity estimator for mango leaves.

The CNN only classifies; it does not measure how much of the leaf is damaged.
This module estimates it from the image: it finds green (healthy) tissue and
yellow / brown / orange / red / dark-lesion tissue near it, then reports the
share of "leaf" pixels that look damaged.

The detection is tuned a little for the predicted condition:
  * Powdery Mildew: also counts pale / whitish patches on the leaf.
  * Sooty Mould:    also counts grey-black coating (a looser darkness limit).

It is a rough colour-based estimate, not a trained severity model. Plain,
dark or coloured backgrounds and good light give the best results. Young mango
leaves are naturally reddish-bronze and can be read as damage.
"""
import numpy as np
from PIL import Image

MAX_SIDE = 256

PALE_CONDITIONS = {"powdery mildew"}
SOOT_CONDITIONS = {"sooty mould", "sooty mold"}


def _box_mean(mask, k):
    """Mean of a boolean mask over a k x k window (integral image, numpy only)."""
    m = mask.astype(np.float32)
    p = np.pad(m, k // 2, mode="constant")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = m.shape
    s = c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]
    return s / float(k * k)


def level_from_percent(pct):
    if pct < 10:
        return "Mild"
    if pct < 30:
        return "Moderate"
    return "Severe"


def estimate_severity(pil_img, condition=None):
    """Return dict(percent, level, overlay) or None if no leaf could be found.

    condition: predicted condition name, e.g. "Powdery Mildew" (optional).
    """
    cond = (condition or "").strip().lower()

    img = pil_img.convert("RGB")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    rgb = np.asarray(img, dtype=np.uint8)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hue = hsv[..., 0] * 360.0 / 255.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    green = (hue >= 75) & (hue <= 170) & (sat >= 0.20) & (val >= 0.15)
    near_green_wide = _box_mean(green, 31) >= 0.05
    near_green_local = _box_mean(green, 9) >= 0.30

    # yellow / orange / brown / reddish tissue close to green tissue
    warm = ((hue < 75) | (hue >= 340)) & (sat >= 0.28) & (val >= 0.20) & near_green_wide

    # dark lesions (anthracnose, bacterial canker, gall spots) surrounded by green tissue.
    # Sooty mould is a grey-black film, so allow slightly brighter pixels for it.
    dark_limit = 0.38 if cond in SOOT_CONDITIONS else 0.25
    dark = (val < dark_limit) & near_green_local

    diseased = warm | dark

    # powdery mildew: whitish / grey-white powder lying on the leaf
    if cond in PALE_CONDITIONS:
        pale = (sat < 0.18) & (val > 0.60) & (_box_mean(green, 21) >= 0.40)
        diseased = diseased | pale

    diseased = diseased & ~green
    leaf = green | diseased

    if leaf.mean() < 0.05:
        return None

    pct = float(diseased.sum()) / float(leaf.sum()) * 100.0

    overlay = rgb.copy().astype(np.float32)
    overlay[diseased] = 0.4 * overlay[diseased] + 0.6 * np.array([255, 0, 80], dtype=np.float32)
    overlay_img = Image.fromarray(overlay.astype(np.uint8))

    return {"percent": pct, "level": level_from_percent(pct), "overlay": overlay_img}
