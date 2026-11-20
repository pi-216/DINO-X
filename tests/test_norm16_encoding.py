"""Tests for the canonical X-ray norm16 u16 encoding in zoo/encoding."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from zoo.encoding import (
    NORM_CLIP_HI,
    NORM_CLIP_LO,
    NORM_SCALE,
    norm01_encoding_token,
    norm01_to_u16,
    u16_to_norm01,
)


def test_roundtrip_known_values():
    v = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    encoded = norm01_to_u16(v)
    assert encoded.dtype == np.uint16
    decoded = u16_to_norm01(encoded)
    np.testing.assert_allclose(decoded, v, atol=1.0 / NORM_SCALE + 1e-4)


def test_zero_maps_to_zero_and_one_to_max():
    assert int(norm01_to_u16(np.array([0.0]))[0]) == 0
    assert int(norm01_to_u16(np.array([1.0]))[0]) == 65535


def test_saturation_clips_not_wraps():
    above = norm01_to_u16(np.array([NORM_CLIP_HI + 0.5]))
    below = norm01_to_u16(np.array([NORM_CLIP_LO - 0.5]))
    assert int(above[0]) == 65535
    assert int(below[0]) == 0
    assert u16_to_norm01(above)[0] <= NORM_CLIP_HI + 1e-4
    assert u16_to_norm01(below)[0] >= NORM_CLIP_LO - 1e-4


def test_grid_is_strictly_monotonic():
    grid = np.linspace(NORM_CLIP_LO, NORM_CLIP_HI, 257)
    encoded = norm01_to_u16(grid)
    assert len(set(encoded.tolist())) == len(grid)
    assert (np.diff(encoded) > 0).all()


def test_token_describes_the_encoding():
    token = norm01_encoding_token()
    assert token.startswith("norm16_")
    assert f"scale{int(NORM_SCALE)}" in token
    assert f"clip{int(NORM_CLIP_LO)}_{int(NORM_CLIP_HI)}" in token
