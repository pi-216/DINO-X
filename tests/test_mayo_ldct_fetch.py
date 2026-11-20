"""Tests for Mayo LDCT IDC S3 fetching (scripts/preprocessing/fetch_mayo_ldct.py)."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "preprocessing")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fetch_mayo_ldct import (
    CensusError,
    RemoteObject,
    load_series_manifest,
    main as fetch_main,
    md5_matches,
    parse_continuation,
    parse_list_objects,
)

MANIFEST_PATH = (
    _REPO_ROOT / "scripts" / "preprocessing" / "manifests" /
    "mayo_ldct_recon_series.json"
)

SERIES_A = "1.2.840.113713.4.100.1.3.9001"
SERIES_B = "1.2.840.113713.4.100.1.3.9010"
UUID_A = "aaaaaaaa-0000-0000-0000-000000000001"
UUID_B = "bbbbbbbb-0000-0000-0000-000000000002"


def _entries() -> list[dict]:
    return [
        {
            "series_instance_uid": SERIES_A,
            "crdc_series_uuid": UUID_A,
            "aws_bucket": "idc-open-data",
            "instance_count": 3,
        },
        {
            "series_instance_uid": SERIES_B,
            "crdc_series_uuid": UUID_B,
            "aws_bucket": "idc-open-data",
            "instance_count": 2,
        },
    ]


def _write_series_manifest(path: Path, *, series_total: int = 2,
                          instances_total: int = 5) -> None:
    doc = {
        "recon_series_total": series_total,
        "recon_instances_total": instances_total,
        "series": _entries(),
    }
    path.write_text(json.dumps(doc))


@pytest.fixture()
def fixture_manifest(tmp_path: Path) -> Path:
    mp = tmp_path / "series.json"
    _write_series_manifest(mp)
    return mp


@pytest.fixture()
def local_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    files_a = {f"o{i}.dcm": bytes([10 + i]) * 40 for i in range(3)}
    files_b = {f"o{i}.dcm": bytes([90 + i]) * 60 for i in range(2)}
    for uuid, files in ((UUID_A, files_a), (UUID_B, files_b)):
        d = src / uuid
        d.mkdir(parents=True)
        for name, body in files.items():
            (d / name).write_bytes(body)
    return src


def _list_xml(bucket: str, prefix: str, entries: list[tuple[str, int, str]],
              truncated: bool = False, token: str | None = None) -> bytes:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ListBucketResult><Name>' + bucket + '</Name>',
        f"<Prefix>{prefix}</Prefix>",
    ]
    for key, size, etag in entries:
        parts.append(
            f"<Contents><Key>{key}</Key><Size>{size}</Size>"
            f"<ETag>&quot;{etag}&quot;</ETag></Contents>"
        )
    parts.append(f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>")
    if token:
        parts.append(f"<NextContinuationToken>{token}</NextContinuationToken>")
    parts.append("</ListBucketResult>")
    return "".join(parts).encode()


def test_committed_manifest_census_is_pinned() -> None:
    doc = load_series_manifest(MANIFEST_PATH)
    assert doc["recon_series_total"] == 299
    assert doc["recon_instances_total"] == 72588
    series = doc["series"]
    assert len(series) == 299
    assert sum(s["instance_count"] for s in series) == 72588
    assert all(s["tcia_table"] in {"chest", "liver"} for s in series)
    assert all(s["aws_bucket"] == "idc-open-data" for s in series)
    assert len({s["series_instance_uid"] for s in series}) == 299


def test_census_mismatch_is_rejected(tmp_path: Path) -> None:
    mp = tmp_path / "bad.json"
    _write_series_manifest(mp, series_total=7)
    with pytest.raises(CensusError):
        load_series_manifest(mp)
    mp2 = tmp_path / "bad2.json"
    _write_series_manifest(mp2, instances_total=99)
    with pytest.raises(CensusError):
        load_series_manifest(mp2)


def test_main_missing_manifest_exits_two(tmp_path: Path) -> None:
    rc = fetch_main([
        "--series-manifest", str(tmp_path / "absent.json"),
        "--out-dir", str(tmp_path / "raw"),
    ])
    assert rc == 2


def test_main_census_failure_exits_two(tmp_path: Path) -> None:
    mp = tmp_path / "bad.json"
    _write_series_manifest(mp, series_total=7)
    rc = fetch_main([
        "--series-manifest", str(mp),
        "--out-dir", str(tmp_path / "raw"),
        "--source-dir", str(tmp_path / "src"),
    ])
    assert rc == 2


def test_main_fetches_local_source(tmp_path: Path, fixture_manifest: Path,
                                   local_source: Path) -> None:
    out = tmp_path / "raw"
    rc = fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--out-dir", str(out),
        "--source-dir", str(local_source),
    ])
    assert rc == 0
    assert sorted(p.name for p in (out / SERIES_A).glob("*.dcm")) == [
        "o0.dcm", "o1.dcm", "o2.dcm",
    ]
    assert (out / SERIES_B / "o1.dcm").read_bytes() == bytes([91]) * 60
    assert not any(p.name.startswith(UUID_A) for p in out.iterdir())


def test_main_resume_skips_completed_series(tmp_path: Path,
                                            fixture_manifest: Path,
                                            local_source: Path) -> None:
    out = tmp_path / "raw"
    assert fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--out-dir", str(out), "--source-dir", str(local_source),
    ]) == 0
    mtimes = {
        p: p.stat().st_mtime_ns
        for p in out.rglob("*.dcm")
    }
    assert fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--out-dir", str(out), "--source-dir", str(local_source),
    ]) == 0
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*.dcm")} == mtimes


def test_main_force_rewrites(tmp_path: Path, fixture_manifest: Path,
                             local_source: Path) -> None:
    out = tmp_path / "raw"
    assert fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--out-dir", str(out), "--source-dir", str(local_source),
    ]) == 0
    assert fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--out-dir", str(out), "--source-dir", str(local_source),
        "--force",
    ]) == 0
    assert len(list(out.rglob("*.dcm"))) == 5


def test_parse_list_objects_and_pagination() -> None:
    xml = _list_xml("idc-open-data", f"{UUID_A}/", [
        (f"{UUID_A}/a.dcm", 12, "ab12"),
        (f"{UUID_A}/b.dcm", 34, "cd34"),
    ], truncated=True, token="tok+1")
    objects = parse_list_objects(xml.decode())
    assert objects == [
        RemoteObject(f"{UUID_A}/a.dcm", 12, "\"ab12\""),
        RemoteObject(f"{UUID_A}/b.dcm", 34, "\"cd34\""),
    ]
    truncated, token = parse_continuation(xml.decode())
    assert truncated is True
    assert token == "tok+1"


def test_md5_matches_handles_multipart_etags() -> None:
    body = b"slice-bytes"
    digest = hashlib.md5(body).hexdigest()
    assert md5_matches(body, f'"{digest}"') is True
    assert md5_matches(body, digest) is True
    assert md5_matches(body, f'"{digest}-2"') is True
    assert md5_matches(body, f'"{hashlib.md5(b"other").hexdigest()}"') is False


class _S3Handler(BaseHTTPRequestHandler):
    objects: dict[str, bytes] = {}
    list_pages: dict[str, list[str]] = {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if "list-type" in query:
            prefix = query.get("prefix", [""])[0]
            token = query.get("continuation-token", [None])[0]
            pages = self.list_pages.get(prefix, [])
            idx = 0 if token is None else int(token)
            body = (pages[idx].encode() if idx < len(pages)
                    else _list_xml("idc-open-data", prefix, []))
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.end_headers()
            self.wfile.write(body)
            return
        key = parsed.path.lstrip("/")
        if key in self.objects:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.objects[key])
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args) -> None:
        return


@pytest.fixture()
def http_s3(tmp_path: Path):
    bodies = {
        f"{UUID_A}/a0.dcm": b"recon-bytes-a0",
        f"{UUID_A}/a1.dcm": b"recon-bytes-a1-longer",
        f"{UUID_A}/a2.dcm": b"recon-bytes-a2",
        f"{UUID_B}/b0.dcm": b"recon-bytes-b0",
        f"{UUID_B}/b1.dcm": b"recon-bytes-b1",
    }
    _S3Handler.objects = dict(bodies)
    md5 = {k: hashlib.md5(v).hexdigest() for k, v in bodies.items()}
    _S3Handler.list_pages = {
        f"{UUID_A}/": [
            _list_xml("idc-open-data", f"{UUID_A}/", [
                (f"{UUID_A}/a0.dcm", len(bodies[f"{UUID_A}/a0.dcm"]),
                 md5[f"{UUID_A}/a0.dcm"]),
                (f"{UUID_A}/a1.dcm", len(bodies[f"{UUID_A}/a1.dcm"]),
                 md5[f"{UUID_A}/a1.dcm"] + "-2"),
            ], truncated=True, token="1").decode(),
            _list_xml("idc-open-data", f"{UUID_A}/", [
                (f"{UUID_A}/a2.dcm", len(bodies[f"{UUID_A}/a2.dcm"]),
                 md5[f"{UUID_A}/a2.dcm"]),
            ]).decode(),
        ],
        f"{UUID_B}/": [
            _list_xml("idc-open-data", f"{UUID_B}/", [
                (f"{UUID_B}/b0.dcm", len(bodies[f"{UUID_B}/b0.dcm"]),
                 md5[f"{UUID_B}/b0.dcm"]),
                (f"{UUID_B}/b1.dcm", len(bodies[f"{UUID_B}/b1.dcm"]),
                 md5[f"{UUID_B}/b1.dcm"]),
            ]).decode(),
        ],
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _S3Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_main_paginates_and_verifies_via_http(tmp_path: Path,
                                              fixture_manifest: Path,
                                              http_s3: str) -> None:
    out = tmp_path / "raw"
    rc = fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--out-dir", str(out),
        "--endpoint", http_s3,
        "--workers", "2",
    ])
    assert rc == 0
    assert (out / SERIES_A / "a0.dcm").read_bytes() == b"recon-bytes-a0"
    assert (out / SERIES_B / "b0.dcm").read_bytes() == b"recon-bytes-b0"


def test_main_listing_census_mismatch_exits_one(tmp_path: Path,
                                                local_source: Path) -> None:
    mp = tmp_path / "shifted.json"
    entries = _entries()
    entries[0]["instance_count"] = 4
    mp.write_text(json.dumps({
        "recon_series_total": 2,
        "recon_instances_total": 6,
        "series": entries,
    }))
    rc = fetch_main([
        "--series-manifest", str(mp),
        "--out-dir", str(tmp_path / "raw"),
        "--source-dir", str(local_source),
    ])
    assert rc == 1
    assert not (tmp_path / "raw" / SERIES_A / "o0.dcm").exists()


def test_main_md5_mismatch_exits_one(tmp_path: Path, fixture_manifest: Path,
                                     http_s3: str) -> None:
    key = f"{UUID_A}/a0.dcm"
    _S3Handler.objects[key] = b"Xecon-bytes-a0"
    out = tmp_path / "raw"
    rc = fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--out-dir", str(out),
        "--endpoint", http_s3,
        "--workers", "2",
    ])
    assert rc == 1
    assert not (out / SERIES_A / "a0.dcm").exists()


def test_check_reports_ok_on_reachable_source(fixture_manifest: Path,
                                              http_s3: str) -> None:
    rc = fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--endpoint", http_s3,
        "--check",
    ])
    assert rc == 0


def test_check_unreachable_exits_three(tmp_path: Path,
                                       fixture_manifest: Path) -> None:
    rc = fetch_main([
        "--series-manifest", str(fixture_manifest),
        "--endpoint", "http://127.0.0.1:9",
        "--check",
    ])
    assert rc == 3
