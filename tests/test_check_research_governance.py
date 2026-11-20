"""Tests for bin/check_research_governance."""

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

CHECKER_PATH = Path(__file__).resolve().parent.parent / "bin" / "check_research_governance"

_loader = SourceFileLoader("check_research_governance", str(CHECKER_PATH))
_spec = importlib.util.spec_from_loader("check_research_governance", _loader)
assert _spec is not None
governance = importlib.util.module_from_spec(_spec)
_loader.exec_module(governance)


def _manifest(**overrides):
    manifest = {
        "kind": "split_manifest",
        "version": 1,
        "group_key": "patient_id",
        "train": {"series_dir": ["s1", "s2", "s3"]},
        "val": {"series_dir": ["s4", "s5"]},
        "holdout": {"series_dir": ["s6", "s7"], "patients": ["p9", "p10"]},
    }
    manifest.update(overrides)
    return manifest


class TestValidateManifest:
    def test_valid_manifest_passes(self):
        assert governance.validate_manifest(_manifest()) == []

    def test_wrong_group_key_fails(self):
        errors = governance.validate_manifest(_manifest(group_key="series_id"))
        assert any("group_key" in e for e in errors)

    def test_train_val_series_overlap_fails(self):
        manifest = _manifest()
        manifest["val"]["series_dir"] = ["s4", "s2"]
        errors = governance.validate_manifest(manifest)
        assert any("train and val series_dir overlap" in e for e in errors)

    def test_pretrain_holdout_series_overlap_fails(self):
        manifest = _manifest()
        manifest["holdout"]["series_dir"] = ["s6", "s1"]
        errors = governance.validate_manifest(manifest)
        assert any("pretrain and holdout series_dir overlap" in e for e in errors)

    def test_empty_train_series_fails(self):
        manifest = _manifest()
        manifest["train"]["series_dir"] = []
        errors = governance.validate_manifest(manifest)
        assert any("train.series_dir is empty" in e for e in errors)

    def test_empty_val_series_fails(self):
        manifest = _manifest()
        manifest["val"]["series_dir"] = []
        errors = governance.validate_manifest(manifest)
        assert any("val.series_dir is empty" in e for e in errors)

    def test_empty_holdout_series_fails(self):
        manifest = _manifest()
        manifest["holdout"]["series_dir"] = []
        errors = governance.validate_manifest(manifest)
        assert any("holdout.series_dir is empty" in e for e in errors)

    def test_empty_holdout_patients_fails(self):
        manifest = _manifest()
        manifest["holdout"]["patients"] = []
        errors = governance.validate_manifest(manifest)
        assert any("holdout.patients is empty" in e for e in errors)

    def test_patient_overlap_fails_when_patient_keys_present(self):
        manifest = _manifest()
        manifest["train"]["patients"] = ["p1", "p2"]
        manifest["val"]["patients"] = ["p2", "p3"]
        errors = governance.validate_manifest(manifest)
        assert any("train and val patients overlap" in e for e in errors)

    def test_pretrain_holdout_patient_overlap_fails(self):
        manifest = _manifest()
        manifest["train"]["patients"] = ["p1", "p9"]
        errors = governance.validate_manifest(manifest)
        assert any("pretrain and holdout patients overlap" in e for e in errors)


def _build_tree(root: Path, manifest: dict | None):
    for rel in governance.REQUIRED_DOCS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n")
    if manifest is not None:
        manifest_path = root / governance.MANIFEST_REL
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))


class TestCheck:
    def test_valid_tree_passes(self, tmp_path):
        _build_tree(tmp_path, _manifest())
        assert governance.check(tmp_path) == []

    def test_missing_manifest_fails(self, tmp_path):
        _build_tree(tmp_path, None)
        errors = governance.check(tmp_path)
        assert any("missing patient-holdout split manifest" in e for e in errors)

    def test_missing_required_doc_fails(self, tmp_path):
        _build_tree(tmp_path, _manifest())
        (tmp_path / "AGENTS.md").unlink()
        errors = governance.check(tmp_path)
        assert any("missing required governance file: AGENTS.md" in e for e in errors)

    def test_sabotaged_series_overlap_fails(self, tmp_path):
        manifest = _manifest()
        manifest["holdout"]["series_dir"] = ["s6", "s1"]
        _build_tree(tmp_path, manifest)
        errors = governance.check(tmp_path)
        assert any("pretrain and holdout series_dir overlap" in e for e in errors)

    def test_invalid_json_fails(self, tmp_path):
        _build_tree(tmp_path, None)
        manifest_path = tmp_path / governance.MANIFEST_REL
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{not json")
        errors = governance.check(tmp_path)
        assert any("invalid" in e for e in errors)

    def test_repo_governance_passes(self):
        assert governance.check(governance.ROOT) == []
