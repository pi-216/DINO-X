"""Tests for scripts/release/run_lidc_holdout_lora_3seed.py (PRD-0001 T1)."""
from __future__ import annotations

import csv
import json
import struct
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("scripts", "scripts/release"):
    p = str(_REPO_ROOT / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from run_lidc_holdout_lora_3seed import (
    aggregate_auroc,
    build_finetune_cmd,
    classify_gate,
    verify_no_patient_leak,
)


def _fake_dcm_bytes(patient_id: str) -> bytes:
    tag = struct.pack("<HH", 0x0010, 0x0020)
    raw = patient_id.encode("ascii")
    return tag + b"LO" + struct.pack("<H", len(raw)) + raw


def _write_label_csv(path: Path, patients: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "spacing_x", "spacing_y", "spacing_z", "patient_id"])
        for i, pid in enumerate(patients):
            w.writerow([f"data/lidc-idri/nodule-crops/nodule_{i:05d}.png", i % 2, 0.65, 0.65, 3.0, pid])


@pytest.fixture
def fixture_repo(tmp_path: Path) -> dict:
    raw_root = tmp_path / "raw"
    series = {
        "data_raw_UID_TRAIN_1": "LIDC-IDRI-0001",
        "data_raw_UID_TRAIN_2": "LIDC-IDRI-0002",
        "data_raw_UID_VAL_1": "LIDC-IDRI-0003",
        "data_raw_UID_HOLD_1": "LIDC-IDRI-0100",
        "data_raw_UID_HOLD_2": "LIDC-IDRI-0101",
    }
    for sd, pid in series.items():
        d = raw_root / sd.removeprefix("data_raw_")
        d.mkdir(parents=True)
        (d / "0001.dcm").write_bytes(_fake_dcm_bytes(pid))

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_label_csv(labels_dir / "malignancy_val.csv", ["LIDC-IDRI-0100", "LIDC-IDRI-0100", "LIDC-IDRI-0101"])
    _write_label_csv(labels_dir / "malignancy_test.csv", ["LIDC-IDRI-0101"])

    manifest = {
        "kind": "split_manifest",
        "group_key": "patient_id",
        "counts": {"train_series": 2, "val_series": 1, "holdout_series": 2, "holdout_patients": 2},
        "train": {"series_dir": ["data_raw_UID_TRAIN_1", "data_raw_UID_TRAIN_2"]},
        "val": {"series_dir": ["data_raw_UID_VAL_1"]},
        "holdout": {
            "series_dir": ["data_raw_UID_HOLD_1", "data_raw_UID_HOLD_2"],
            "patients": ["LIDC-IDRI-0100", "LIDC-IDRI-0101"],
            "reason": "malignancy val+test patients — backbone must never see these",
        },
    }
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "tmp": tmp_path,
        "raw_root": raw_root,
        "labels_dir": labels_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "series": series,
    }


class TestLeakGuard:
    def test_clean_split_passes(self, fixture_repo):
        summary = verify_no_patient_leak(
            fixture_repo["manifest_path"], fixture_repo["labels_dir"], fixture_repo["raw_root"]
        )
        assert summary["pretrain_patients"] == 3
        assert summary["holdout_patients"] == 2
        assert summary["overlap"] == 0

    def test_detects_pretrain_patient_in_label_splits(self, fixture_repo):
        _write_label_csv(fixture_repo["labels_dir"] / "malignancy_val.csv", ["LIDC-IDRI-0100", "LIDC-IDRI-0101", "LIDC-IDRI-0002"])
        manifest = fixture_repo["manifest"]
        manifest["holdout"]["patients"] = ["LIDC-IDRI-0002", "LIDC-IDRI-0100", "LIDC-IDRI-0101"]
        fixture_repo["manifest_path"].write_text(json.dumps(manifest))
        with pytest.raises(AssertionError, match="Patient leakage|leak"):
            verify_no_patient_leak(
                fixture_repo["manifest_path"], fixture_repo["labels_dir"], fixture_repo["raw_root"]
            )

    def test_detects_manifest_label_disagreement(self, fixture_repo):
        _write_label_csv(fixture_repo["labels_dir"] / "malignancy_test.csv", ["LIDC-IDRI-9999"])
        with pytest.raises(ValueError, match="disagree"):
            verify_no_patient_leak(
                fixture_repo["manifest_path"], fixture_repo["labels_dir"], fixture_repo["raw_root"]
            )

    def test_detects_holdout_series_in_pretrain(self, fixture_repo):
        manifest = fixture_repo["manifest"]
        manifest["train"]["series_dir"].append("data_raw_UID_HOLD_1")
        fixture_repo["manifest_path"].write_text(json.dumps(manifest))
        with pytest.raises(AssertionError, match="series"):
            verify_no_patient_leak(
                fixture_repo["manifest_path"], fixture_repo["labels_dir"], fixture_repo["raw_root"]
            )

    def test_unattributed_series_fails_closed(self, fixture_repo):
        manifest = fixture_repo["manifest"]
        manifest["train"]["series_dir"].append("data_raw_UID_UNKNOWN")
        fixture_repo["manifest_path"].write_text(json.dumps(manifest))
        with pytest.raises(RuntimeError, match="cannot attribute"):
            verify_no_patient_leak(
                fixture_repo["manifest_path"], fixture_repo["labels_dir"], fixture_repo["raw_root"]
            )


class TestAggregation:
    def test_mean_std_se_ci(self):
        agg = aggregate_auroc([0.72, 0.74, 0.70])
        assert agg["n"] == 3
        assert agg["mean"] == pytest.approx(0.72)
        assert agg["std"] == pytest.approx(0.02)
        assert agg["se"] == pytest.approx(0.02 / (3 ** 0.5))
        assert agg["ci95_low"] == pytest.approx(0.72 - 1.96 * agg["se"])
        assert agg["ci95_high"] == pytest.approx(0.72 + 1.96 * agg["se"])

    def test_identical_seeds(self):
        agg = aggregate_auroc([0.73, 0.73, 0.73])
        assert agg["std"] == 0.0
        assert agg["se"] == 0.0
        assert agg["ci95_low"] == pytest.approx(0.73)

    def test_requires_values(self):
        with pytest.raises(ValueError):
            aggregate_auroc([])


class TestGateClassification:
    def test_go(self):
        assert classify_gate(mean=0.735, ci_low=0.712) == "GO"

    def test_pivot_when_ci_touches_baseline(self):
        assert classify_gate(mean=0.730, ci_low=0.705) == "PIVOT"

    def test_pivot_at_ci_boundary_not_go(self):
        assert classify_gate(mean=0.725, ci_low=0.710) == "PIVOT"

    def test_pivot_below_go_mean(self):
        assert classify_gate(mean=0.715, ci_low=0.711) == "PIVOT"

    def test_fail_below_baseline(self):
        assert classify_gate(mean=0.709, ci_low=0.700) == "FAIL"

    def test_baseline_mean_not_fail(self):
        assert classify_gate(mean=0.710, ci_low=0.708) == "PIVOT"

    def test_go_requires_both_conditions(self):
        assert classify_gate(mean=0.720, ci_low=0.7101) == "GO"
        assert classify_gate(mean=0.719, ci_low=0.715) == "PIVOT"


class TestFinetuneCommand:
    def test_standard_protocol_flags(self):
        cmd = build_finetune_cmd(
            python=".venv/bin/python3",
            finetune_script=Path("/repo/scripts/finetune_lora.py"),
            checkpoint=Path("/repo/runs/x/checkpoint_00020000.pth"),
            train_csv=Path("/labels/malignancy_train.csv"),
            val_csv=Path("/labels/malignancy_val.csv"),
            data_root=Path("/repo"),
            output=Path("/repo/adapters/lidc-holdout-lora-ck20000-seed42"),
            seed=42,
        )
        joined = " ".join(cmd)
        assert "--seed 42" in joined
        assert "--rank 8" in joined
        assert "--alpha 16" in joined
        assert "--lr 5e-4" in joined
        assert "--epochs 50" in joined
        assert "--batch-size 32" in joined
        assert "--task classification" in joined
        assert "--num-classes 2" in joined
        assert "--patience 10" in joined
        assert "--es-metric auroc" in joined
        assert "--window-level -30" in joined
        assert "--window-width 120" in joined
        assert "--device cuda" in joined

    def test_scale_aware_is_backbone_derived_not_a_cli_flag(self):
        cmd = build_finetune_cmd(
            python="python3",
            finetune_script=Path("/repo/scripts/finetune_lora.py"),
            checkpoint=Path("/repo/runs/x/checkpoint_00040000.pth"),
            train_csv=Path("/labels/malignancy_train.csv"),
            val_csv=Path("/labels/malignancy_val.csv"),
            data_root=Path("/repo"),
            output=Path("/repo/adapters/a"),
            seed=43,
        )
        assert "--scale-aware" not in cmd
