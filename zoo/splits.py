"""Canonical DINO-X split utilities: patient-level grouping and identity.

SINGLE SOURCE OF TRUTH for two things that were previously reimplemented ad hoc
and got wrong in shipped code:

1. **DICOM patient identity.** ``extract_crlm_vessel_labels.py`` wrote the
   referenced ``SeriesInstanceUID`` into a column named ``patient_id``, so every
   "patient count" derived from it was actually a *series* count. ``patient_id``
   must come from the DICOM PatientID tag (0010,0020), never the series UID.

2. **Split granularity.** ``phase4_make_split_manifest.py`` split at
   ``series_dir`` and its docstring claimed this *avoids* leakage. It does the
   opposite: a patient with multiple series (multi-phase CT, follow-up scans)
   can land in BOTH train and val. Splits must group by the coarsest identity
   unit (patient) and then assert no series leaks across the boundary.

Usage::

    from zoo.splits import patient_split, read_patient_id

    pids = [read_patient_id(dicom_bytes) for ...]
    train_idx, val_idx = patient_split(pids, test_size=0.2, seed=0)
"""

from __future__ import annotations

import struct
from pathlib import Path


# ---------------------------------------------------------------------------
# Identity: read the real DICOM PatientID from raw bytes (dependency-free).
# ---------------------------------------------------------------------------

def read_patient_id(dicom_bytes: bytes) -> str | None:
    """Read the DICOM PatientID (0010,0020) from raw file bytes.

    A minimal, dependency-free reader for the ASCII PatientID VR (LO/SH). It
    finds the tag pattern directly in the byte stream, so it is robust against
    SQ/OB nesting and odd-length padding that break naive sequential walkers.

    Prefer ``pydicom`` when it is installed (see :func:`read_patient_id_file`),
    which handles the full DICOM grammar; this is a fallback that needs no
    third-party package.

    Args:
        dicom_bytes: Raw bytes of a DICOM file (may include the 128-byte
            preamble + "DICM" magic; those are skipped).

    Returns:
        The PatientID string, or None if not found / not ASCII-decodable.
    """
    pattern = struct.pack("<HH", 0x0010, 0x0020)
    start = 132 if dicom_bytes[128:132] == b"DICM" else 0

    idx = start
    while True:
        idx = dicom_bytes.find(pattern, idx)
        if idx < 0:
            return None
        if idx + 12 > len(dicom_bytes):
            idx += 4
            continue
        vr = dicom_bytes[idx + 4:idx + 6].decode("ascii", "replace")
        if vr in ("OB", "OW", "OF", "SQ", "UT", "UN"):
            length = struct.unpack_from("<I", dicom_bytes, idx + 8)[0]
            offset = idx + 12
        else:
            length = struct.unpack_from("<H", dicom_bytes, idx + 6)[0]
            offset = idx + 8
        if offset + length > len(dicom_bytes):
            idx += 4
            continue
        raw = dicom_bytes[offset:offset + length]
        value = raw.split(b"\x00")[0].decode("ascii", "replace").strip()
        if value:
            return value
        idx += 4


def read_patient_id_file(path: str | Path) -> str | None:
    """Read PatientID from a DICOM file, preferring ``pydicom`` if installed.

    Args:
        path: Path to a DICOM file.

    Returns:
        PatientID string, or None if unavailable.
    """
    path = Path(path)
    data = path.read_bytes()
    try:
        import pydicom  # noqa: PLC0415

        ds = pydicom.dcmread(path, force=True, stop_before_pixels=True)
        pid = getattr(ds, "PatientID", None)
        if pid:
            return str(pid) if not isinstance(pid, str) else pid
    except Exception:
        pass
    return read_patient_id(data)


# ---------------------------------------------------------------------------
# Patient-level splits.
# ---------------------------------------------------------------------------

def patient_split(
    patient_ids: list[str | None],
    *,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Split indices into train/val grouped by patient ID.

    A patient's *all* indices stay together on one side, so identity cannot
    leak across the boundary even when a patient has many slices or series.

    Args:
        patient_ids: One patient ID per sample, aligned 1:1 with the dataset
            rows. ``None`` entries are treated as their own singleton group (so
            a missing identity never silently pools two patients together).
        test_size: Fraction of *patients* for the validation side.
        seed: RNG seed for the patient shuffle.

    Returns:
        ``(train_idx, val_idx)`` as integer indices into ``patient_ids``.

    Raises:
        ValueError: If ``test_size`` leaves zero patients on either side.
    """
    import random

    if not patient_ids:
        raise ValueError("patient_ids is empty")

    # Assign each distinct patient a stable group key; None -> unique sentinel.
    groups: list[object] = []
    seen: dict[str | None, object] = {}
    for pid in patient_ids:
        if pid is None:
            # unique singleton so missing identity never pools patients
            groups.append(object())
            continue
        if pid not in seen:
            seen[pid] = pid
        groups.append(seen[pid])

    distinct = list(seen.values())
    rng = random.Random(seed)
    rng.shuffle(distinct)

    n_val = max(1, round(test_size * len(distinct)))
    n_val = min(n_val, len(distinct) - 1)  # keep >=1 train patient
    val_groups = set(distinct[:n_val])

    train_idx, val_idx = [], []
    for i, g in enumerate(groups):
        if g in val_groups:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx


def assert_no_series_leak(
    train_series: set[str],
    val_series: set[str],
) -> None:
    """Assert disjointness of series across a split boundary.

    After mapping series -> patient and assigning patients to sides, no series
    belonging to a train patient may appear in val (and vice versa). Call this
    on every split before training.

    Args:
        train_series: Series IDs on the train side.
        val_series: Series IDs on the val side.

    Raises:
        AssertionError: If any series appears on both sides.
    """
    overlap = train_series & val_series
    if overlap:
        raise AssertionError(
            f"Series leakage across split: {len(overlap)} series in both "
            f"train and val (e.g. {sorted(overlap)[:5]})"
        )


def assert_no_patient_leak(
    train_patients: set[str],
    val_patients: set[str],
) -> None:
    """Assert no patient appears on both sides of a train/val split.

    Identity leakage at the patient level is the failure mode that inflates
    held-out metrics: a patient with several scans (series) can otherwise land
    in both train and val. Call this on the loaded label CSVs before training
    whenever they carry a ``patient_id`` column.

    Args:
        train_patients: Distinct patient IDs on the train side.
        val_patients: Distinct patient IDs on the val side.

    Raises:
        AssertionError: If any patient is shared across the boundary.
    """
    overlap = train_patients & val_patients
    if overlap:
        raise AssertionError(
            f"Patient leakage across split: {len(overlap)} patients in both "
            f"train and val (e.g. {sorted(overlap)[:5]})"
        )