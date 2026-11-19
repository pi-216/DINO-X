"""Canonical DINO-X intensity <-> uint16 PNG encodings.

SINGLE SOURCE OF TRUTH for the project's 16-bit PNG formats. Do not inline
these formulas in other files — import them here. The HU encoding was silently
reimplemented across ~11 files during development; two of them drifted 10x
(encode `round(HU + 32768)` / decode `*1` instead of `*0.1`) and were only
caught by an external adversarial review (Gemini red-team, 2026-09-13).

Canonical CT (`hu16_png`, verified against the shipped corpus's on-disk bytes)::

    encode:  u16 = round(clip(HU, -1000, 4000) * 10) + 32768   (clamped [0, 65535])
    decode:  HU  = (u16 - 32768) * 0.1

Encodable range without overflow: clip bounds [-1000, 4000] map to
[22768, 72768]; the +4000 edge exceeds uint16 max (65535) and would wrap, so
the pre-scale value is clamped to the safe max (4000 HU saturates at 65535 ->
decodes to 3276.7 HU).

Canonical MRI (`zscore16_png`): MRI intensities are coil-dependent and not
calibrated, so each acquired sequence is z-normalized per volume and stored
in sigma units::

    encode:  u16 = round(clip(z, -300, 300) * 100) + 32768
    decode:  z   = (u16 - 32768) / 100

Sigma units fit uint16 with headroom ([-327.68, 327.67] storable); the
[-300, 300] clip is a pre-scale guard so no value can reach the uint16 edge.

Canonical X-ray (`norm16_png`): radiograph pixel data is optical density,
which is display-scaled and not calibrated, so each image is min-max
normalized to [0, 1] and stored across the full uint16 range::

    encode:  u16 = round(clip(v, 0, 1) * 65535)
    decode:  v   = u16 / 65535

Usage::

    from zoo.encoding import hu_to_u16, u16_to_hu
    from zoo.encoding import zscore_to_u16, u16_to_zscore, zscore_encoding_token
    from zoo.encoding import norm01_to_u16, u16_to_norm01, norm01_encoding_token

    u16 = hu_to_u16(hu_array)     # float HU -> uint16 PNG pixels
    hu  = u16_to_hu(u16_array)    # uint16 PNG pixels -> float HU

    u16 = zscore_to_u16(z_array)  # float z-scores -> uint16 PNG pixels
    z   = u16_to_zscore(u16_arr)  # uint16 PNG pixels -> float z-scores

    u16 = norm01_to_u16(v_array)  # float [0,1] optical density -> uint16 PNG
    v   = u16_to_norm01(u16_arr)  # uint16 PNG pixels -> float [0,1]
"""

from __future__ import annotations

import numpy as np

HU_CLIP_LO = -1000.0
HU_CLIP_HI = 4000.0
HU_OFFSET = 32768
HU_SCALE = 10.0

Z_CLIP_LO = -300.0
Z_CLIP_HI = 300.0
Z_OFFSET = 32768
Z_SCALE = 100.0


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


def hu_encoding_token() -> str:
    """Encoding token written to index CSVs for `hu16_png` slices."""
    return (
        f"hu16_i16_offset{HU_OFFSET}_clip{int(HU_CLIP_LO)}_{int(HU_CLIP_HI)}"
    )


def zscore_to_u16(img_z: np.ndarray) -> np.ndarray:
    """Convert a float z-score array to uint16 for 16-bit PNG storage.

    Args:
        img_z: Array of per-volume z-scores (float).

    Returns:
        uint16 array in the canonical `zscore16_png` encoding. Values beyond
        the clip bounds saturate at the bound rather than wrapping.
    """
    x = np.clip(img_z, Z_CLIP_LO, Z_CLIP_HI)
    x = np.rint(x * Z_SCALE + float(Z_OFFSET))
    x = np.clip(x, 0, 65535)
    return x.astype(np.uint16)


def u16_to_zscore(arr: np.ndarray) -> np.ndarray:
    """Convert a uint16 `zscore16_png` array (or any int array) to float z-scores.

    Args:
        arr: uint16 (or int) array of encoded pixels.

    Returns:
        float32 array of z-scores (sigma units).
    """
    return (arr.astype(np.float32) - float(Z_OFFSET)) * (1.0 / Z_SCALE)


def zscore_encoding_token() -> str:
    """Encoding token written to index CSVs for `zscore16_png` slices."""
    return (
        f"zscore16_u16_offset{Z_OFFSET}_scale{int(Z_SCALE)}"
        f"_clip{int(Z_CLIP_LO)}_{int(Z_CLIP_HI)}"
    )


NORM_CLIP_LO = 0.0
NORM_CLIP_HI = 1.0
NORM_SCALE = 65535.0


def norm01_to_u16(img_norm: np.ndarray) -> np.ndarray:
    """Convert a float [0, 1] array to uint16 for 16-bit PNG storage.

    Args:
        img_norm: Array of normalized intensities (float, typically
            per-image min-max normalized optical density).

    Returns:
        uint16 array in the canonical `norm16_png` encoding. Values beyond
        the clip bounds saturate at the bound rather than wrapping.
    """
    x = np.clip(img_norm, NORM_CLIP_LO, NORM_CLIP_HI)
    x = np.rint(x * NORM_SCALE)
    x = np.clip(x, 0, 65535)
    return x.astype(np.uint16)


def u16_to_norm01(arr: np.ndarray) -> np.ndarray:
    """Convert a uint16 `norm16_png` array (or any int array) to float [0, 1].

    Args:
        arr: uint16 (or int) array of encoded pixels.

    Returns:
        float32 array of normalized intensities in [0, 1].
    """
    return arr.astype(np.float32) / NORM_SCALE


def norm01_encoding_token() -> str:
    """Encoding token written to index CSVs for `norm16_png` slices."""
    return (
        f"norm16_u16_scale{int(NORM_SCALE)}"
        f"_clip{int(NORM_CLIP_LO)}_{int(NORM_CLIP_HI)}"
    )