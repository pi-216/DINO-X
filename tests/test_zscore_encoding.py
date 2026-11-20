"""Tests for the canonical MRI z-score u16 encoding in zoo/encoding."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zoo.encoding import (
    Z_CLIP_HI,
    Z_CLIP_LO,
    Z_OFFSET,
    Z_SCALE,
    u16_to_zscore,
    zscore_encoding_token,
    zscore_to_u16,
)


def test_roundtrip_known_values():
    z = np.array([-3.5, -0.25, 0.0, 0.75, 4.2], dtype=np.float32)
    encoded = zscore_to_u16(z)
    assert encoded.dtype == np.uint16
    decoded = u16_to_zscore(encoded)
    np.testing.assert_allclose(decoded, z, atol=1.0 / Z_SCALE + 1e-3)


def test_zero_maps_to_offset():
    assert int(zscore_to_u16(np.array([0.0]))[0]) == Z_OFFSET


def test_saturation_clips_not_wraps():
    above = zscore_to_u16(np.array([Z_CLIP_HI + 40.0]))
    below = zscore_to_u16(np.array([Z_CLIP_LO - 40.0]))
    assert int(above[0]) == int(round(Z_CLIP_HI * Z_SCALE + Z_OFFSET))
    assert int(below[0]) == int(round(Z_CLIP_LO * Z_SCALE + Z_OFFSET))
    assert u16_to_zscore(above)[0] <= Z_CLIP_HI + 1e-3
    assert u16_to_zscore(below)[0] >= Z_CLIP_LO - 1e-3


def test_negative_never_collides_with_positive():
    grid = np.linspace(Z_CLIP_LO, Z_CLIP_HI, 51)
    encoded = zscore_to_u16(grid)
    assert len(set(encoded.tolist())) == len(grid)
    assert (np.diff(encoded) > 0).all()


def test_token_describes_the_encoding():
    token = zscore_encoding_token()
    assert token.startswith("zscore16_")
    assert f"offset{Z_OFFSET}" in token
    assert f"scale{int(Z_SCALE)}" in token
    assert f"clip{int(Z_CLIP_LO)}_{int(Z_CLIP_HI)}" in token
