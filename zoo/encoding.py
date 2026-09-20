"""Canonical DINO-X HU <-> uint16 PNG encoding.

SINGLE SOURCE OF TRUTH for the project's 16-bit HU PNG format. Do not inline
this formula in other files — import it here. The encoding was silently
reimplemented across ~11 files during development; two of them drifted 10x
(encode `round(HU + 32768)` / decode `*1` instead of `*0.1`) and were only
caught by an external adversarial review (Gemini red-team, 2026-09-13).

Canonical (verified against the shipped corpus's on-disk PNG bytes)::

    encode:  u16 = round(clip(HU, -1000, 4000) * 10) + 32768   (clamped [0, 65535])
    decode:  HU  = (u16 - 32768) * 0.1

Encodable range without overflow: clip bounds [-1000, 4000] map to
[22768, 72768]; the +4000 edge exceeds uint16 max (65535) and would wrap, so
the pre-scale value is clamped to the safe max (4000 HU saturates at 65535 ->
decodes to 3276.7 HU).

Usage::

    from zoo.encoding import hu_to_u16, u16_to_hu

    u16 = hu_to_u16(hu_array)     # float HU -> uint16 PNG pixels
    hu  = u16_to_hu(u16_array)    # uint16 PNG pixels -> float HU
"""

from __future__ import annotations

import numpy as np

HU_CLIP_LO = -1000.0
HU_CLIP_HI = 4000.0
HU_OFFSET = 32768
HU_SCALE = 10.0  # one decimal of HU precision; canonical, matches the shipped corpus


def hu_to_u16(img_hu: np.ndarray) -> np.ndarray:
    """Convert a float HU array to uint16 for 16-bit PNG storage.

    Args:
        img_hu: Array of Hounsfield Units (float).

    Returns:
        uint16 array in the canonical `hu16_png` encoding. +4000 (and above)
        saturates at 65535 rather than wrapping past uint16 max.
    """
    x = np.clip(img_hu, HU_CLIP_LO, HU_CLIP_HI)
    x = np.rint(x * HU_SCALE + float(HU_OFFSET))
    x = np.clip(x, 0, 65535)
    return x.astype(np.uint16)


def u16_to_hu(arr: np.ndarray) -> np.ndarray:
    """Convert a uint16 `hu16_png` array (or any int array) back to float HU.

    Args:
        arr: uint16 (or int) array of encoded pixels.

    Returns:
        float32 array of Hounsfield Units.
    """
    return (arr.astype(np.float32) - float(HU_OFFSET)) * (1.0 / HU_SCALE)