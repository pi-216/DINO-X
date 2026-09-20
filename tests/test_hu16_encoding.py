"""Regression tests for the canonical DINO-X hu16 PNG encoding.

Pins the encode/decode round-trip so the latent 10x scale inconsistency found in
the external adversarial review (Gemini red-team, 2026-09-13) cannot silently
regress. The canonical encoding is::

    encode:  u16 = round(clip(HU, -1000, 4000) * 10) + 32768   (clamped [0, 65535])
    decode:  HU  = (u16 - 32768) * 0.1

The single source of truth is :mod:`zoo.encoding` (``hu_to_u16`` / ``u16_to_hu``).
Every consumer — ``zoo/data.py``, ``zoo/predict.py``, ``zoo/encode.py``,
``finetune_lora.py``, ``eval_external.py``, ``phase5_big_run.py``,
``evaluate_panorgan.py``, ``phase5_monitor.py``, and the preprocessing encoders —
imports those functions rather than inlining the formula.
"""

import numpy as np
import pytest

from zoo.encoding import HU_CLIP_HI, HU_CLIP_LO, HU_OFFSET, HU_SCALE, hu_to_u16, u16_to_hu


def _canonical_decode(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.float32) - 32768.0) * 0.1


def test_encode_decode_roundtrip():
    """encode -> decode recovers HU to within one decimal of precision."""
    hu = np.array([-1000.0, -500.0, -100.0, -30.0, 0.0, 40.0, 100.0,
                   500.0, 1000.0, 3000.0, 3276.7], dtype=np.float32)
    rt = u16_to_hu(hu_to_u16(hu))
    assert np.allclose(rt, hu, atol=0.1), f"round-trip mismatch {rt} vs {hu}"


def test_encode_no_uint16_overflow():
    """The +4000 clip edge must saturate, not wrap past uint16 max."""
    u = hu_to_u16(np.array([4000.0], dtype=np.float32))
    assert u.dtype == np.uint16
    assert int(u[0]) == 65535, f"+4000 HU should saturate at 65535, got {u[0]}"
    assert int(u[0]) > 60000


def test_encode_uses_x10_scale():
    """HU=0 -> 32768; HU=1 -> 32778 (x10), NOT 32769 (x1). Guards the regression."""
    assert int(hu_to_u16(np.array([0.0], dtype=np.float32))[0]) == 32768
    assert int(hu_to_u16(np.array([1.0], dtype=np.float32))[0]) == 32778


def test_decode_matches_canonical_formula():
    """u16_to_hu must equal (u16 - 32768) * 0.1 exactly."""
    u16 = np.array([0, 32768, 42768, 65535], dtype=np.uint16)
    assert np.array_equal(u16_to_hu(u16), _canonical_decode(u16))


def test_constants_exposed():
    """Canonical constants are importable so inliners can use them instead of literals."""
    assert HU_CLIP_LO == -1000.0
    assert HU_CLIP_HI == 4000.0
    assert HU_OFFSET == 32768
    assert HU_SCALE == 10.0