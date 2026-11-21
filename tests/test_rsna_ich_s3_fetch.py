"""Tests for RSNA ICH AWS S3 range-zip fetching (scripts/preprocessing/fetch_rsna_ich_s3.py)."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "preprocessing")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fetch_rsna_ich_s3 import (
    Member,
    load_central_directory,
    main as fetch_main,
    open_source,
    select_members,
)

PREFIX = "comp/stage_2_train/"

TRAIN_A = b"\x01\x02\x03" * 100
TRAIN_B = b"stored-member-bytes"
TRAIN_C = b"zip64-member-" * 300


def _make_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("comp/stage_2_sample_submission.csv", b"Id,Label\n")
        zf.writestr("comp/stage_2_train.csv", b"Id,Label\nID_a.dcm_0,1\n")
        zf.writestr(PREFIX + "ID_a.dcm", TRAIN_A)
        zi = zipfile.ZipInfo(PREFIX + "ID_b.dcm")
        zi.compress_type = zipfile.ZIP_STORED
        zf.writestr(zi, TRAIN_B)
        zi64 = zipfile.ZipInfo(PREFIX + "ID_c.dcm")
        with zf.open(zi64, "w", force_zip64=True) as fh:
            fh.write(TRAIN_C)
        zf.writestr("comp/stage_2_test/ID_z.dcm", b"not-train")


@pytest.fixture()
def fixture_zip(tmp_path: Path) -> Path:
    zp = tmp_path / "fixture.zip"
    _make_zip(zp)
    return zp


def test_central_directory_lists_all_members(fixture_zip: Path) -> None:
    source = open_source(str(fixture_zip))
    members = load_central_directory(source)
    by_name = {m.name: m for m in members}
    assert len(members) == 6
    assert by_name[PREFIX + "ID_a.dcm"].usize == len(TRAIN_A)
    assert by_name[PREFIX + "ID_b.dcm"].usize == len(TRAIN_B)
    assert by_name[PREFIX + "ID_c.dcm"].usize == len(TRAIN_C)
    assert all(m.csize > 0 or m.method == 0 for m in members)
    assert all(isinstance(m, Member) for m in members)


def test_select_members_filters_prefix(fixture_zip: Path) -> None:
    source = open_source(str(fixture_zip))
    chosen = select_members(load_central_directory(source), PREFIX)
    assert {m.name for m in chosen} == {
        PREFIX + "ID_a.dcm", PREFIX + "ID_b.dcm", PREFIX + "ID_c.dcm",
    }


def test_main_extracts_only_prefix_members(tmp_path: Path, fixture_zip: Path) -> None:
    out = tmp_path / "raw"
    rc = fetch_main(["--url", str(fixture_zip), "--out-dir", str(out),
                     "--prefix", PREFIX, "--workers", "2"])
    assert rc == 0
    assert (out / "ID_a.dcm").read_bytes() == TRAIN_A
    assert (out / "ID_b.dcm").read_bytes() == TRAIN_B
    assert (out / "ID_c.dcm").read_bytes() == TRAIN_C
    assert not (out / "ID_z.dcm").exists()
    assert not (out / "stage_2_train.csv").exists()


def test_main_with_labels_out(tmp_path: Path, fixture_zip: Path) -> None:
    out = tmp_path / "raw"
    labels = tmp_path / "labels" / "stage_2_train.csv"
    rc = fetch_main(["--url", str(fixture_zip), "--out-dir", str(out),
                     "--prefix", PREFIX,
                     "--labels-out", str(labels),
                     "--labels-member", "comp/stage_2_train.csv"])
    assert rc == 0
    assert labels.read_bytes() == b"Id,Label\nID_a.dcm_0,1\n"


def test_main_resume_skips_existing(tmp_path: Path, fixture_zip: Path) -> None:
    out = tmp_path / "raw"
    assert fetch_main(["--url", str(fixture_zip), "--out-dir", str(out),
                       "--prefix", PREFIX]) == 0
    mtimes = {p: p.stat().st_mtime_ns for p in out.iterdir()}
    assert fetch_main(["--url", str(fixture_zip), "--out-dir", str(out),
                       "--prefix", PREFIX]) == 0
    assert {p: p.stat().st_mtime_ns for p in out.iterdir()} == mtimes


def test_corrupt_member_crc_exits_one(tmp_path: Path, fixture_zip: Path) -> None:
    source = open_source(str(fixture_zip))
    member = next(m for m in load_central_directory(source)
                  if m.name == PREFIX + "ID_a.dcm")
    raw = bytearray(fixture_zip.read_bytes())
    target = member.offset + 30 + len(member.name) + member.csize // 2
    raw[target] ^= 0xFF
    fixture_zip.write_bytes(bytes(raw))
    out = tmp_path / "raw"
    rc = fetch_main(["--url", str(fixture_zip), "--out-dir", str(out),
                     "--prefix", PREFIX])
    assert rc == 1
    assert not (out / "ID_a.dcm").exists()


def test_check_reports_counts_and_unreachable_exits_three(
    tmp_path: Path, fixture_zip: Path, capsys
) -> None:
    rc = fetch_main(["--url", str(fixture_zip), "--check", "--prefix", PREFIX])
    out = capsys.readouterr().out
    assert rc == 0
    assert "check=true" in out
    assert "stage_2_train" in out
    assert "3" in out
    assert fetch_main(["--url", str(tmp_path / "missing.zip"), "--check"]) == 3


_SERVE_DIR: Path | None = None


class _RangeHandler(BaseHTTPRequestHandler):
    def _fs_path(self) -> str:
        from urllib.parse import unquote, urlparse
        name = unquote(urlparse(self.path).path).lstrip("/")
        assert _SERVE_DIR is not None
        return str(_SERVE_DIR / name)

    def do_GET(self) -> None:
        path = Path(self._fs_path())
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, b = rng[6:].split("-")
            start = int(a)
            end = int(b) if b else len(data) - 1
            chunk = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, format, *args) -> None:
        return


def test_http_range_source_and_extraction(tmp_path: Path, fixture_zip: Path) -> None:
    import threading

    global _SERVE_DIR
    serve_dir = fixture_zip.parent
    _SERVE_DIR = serve_dir
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/fixture.zip"
        source = open_source(url)
        assert source.size == fixture_zip.stat().st_size
        chunk = source.fetch(10, 40)
        assert chunk == fixture_zip.read_bytes()[10:40]
        out = tmp_path / "raw-http"
        rc = fetch_main(["--url", url, "--out-dir", str(out),
                         "--prefix", PREFIX, "--workers", "2",
                         "--window-mb", "0.001"])
        assert rc == 0
        assert (out / "ID_a.dcm").read_bytes() == TRAIN_A
        assert (out / "ID_c.dcm").read_bytes() == TRAIN_C
    finally:
        server.shutdown()


def _run_shell(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", "scripts/preprocessing/download_rsna_ich.sh", *args],
        cwd=_REPO_ROOT, capture_output=True, text=True, env=full_env,
    )


def test_shell_check_missing_kaggle_cli_exits_four(tmp_path: Path) -> None:
    proc = _run_shell(["--check", "--source", "kaggle"],
                      env={"KAGGLE_BIN": str(tmp_path / "no-kaggle")})
    assert proc.returncode == 4
    assert "kaggle CLI not found" in proc.stderr
    assert "install" in proc.stderr.lower()


def test_shell_ingest_missing_kaggle_cli_exits_four_before_download(
    tmp_path: Path,
) -> None:
    proc = _run_shell(["--source", "kaggle"],
                      env={"KAGGLE_BIN": str(tmp_path / "no-kaggle"),
                           "RSNA_RAW_DIR": str(tmp_path / "raw")})
    assert proc.returncode == 4
    assert "kaggle CLI not found" in proc.stderr


def test_shell_check_aws_via_url_override(tmp_path: Path, fixture_zip: Path) -> None:
    proc = _run_shell(["--check"],
                      env={"RSNA_S3_ZIP_URL": str(fixture_zip)})
    assert proc.returncode == 0
    assert "check=true" in proc.stdout


def _make_python_stub(tmp_path: Path) -> tuple[Path, Path]:
    stub = tmp_path / "python_stub.sh"
    argv_log = tmp_path / "argv.log"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {argv_log}\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub, argv_log


def test_shell_aws_ingest_refetches_when_partial_corpus_present(
    tmp_path: Path,
) -> None:
    stub, argv_log = _make_python_stub(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ID_partial.dcm").write_bytes(b"partial")
    proc = _run_shell(
        ["--source", "aws"],
        env={
            "PYTHON_BIN": str(stub),
            "RSNA_RAW_DIR": str(raw),
            "RSNA_PROCESSED_DIR": str(tmp_path / "processed"),
            "RSNA_LABELS_OUT": str(tmp_path / "labels.csv"),
            "RSNA_MANIFEST_PATH": str(tmp_path / "manifest.parquet"),
        },
    )
    assert proc.returncode == 0, proc.stderr
    calls = argv_log.read_text().splitlines()
    fetch_calls = [c for c in calls if "fetch_rsna_ich_s3.py" in c]
    preprocess_calls = [c for c in calls if "preprocess_rsna_ich.py" in c]
    assert len(fetch_calls) == 1
    assert len(preprocess_calls) == 1
    assert calls.index(fetch_calls[0]) < calls.index(preprocess_calls[0])
