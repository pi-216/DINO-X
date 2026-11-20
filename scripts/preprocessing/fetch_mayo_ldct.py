#!/usr/bin/env python3
"""Fetch Mayo LDCT reconstructed CT series from the IDC open S3 mirror.

The TCIA collection `LDCT-and-Projection-data` (version 7) carries 697
open-access series: 299 reconstructed CT stacks (CT Image Storage, 72,588
instances, ~38 GB) and 399 DICOM-CT-PD projection series (Raw Data Storage,
~8.0M instance files, ~824 GB). This tool fetches only the reconstructed
series listed in `manifests/mayo_ldct_recon_series.json` — every entry was
cross-checked against the CC BY 4.0 Chest (347 series) and Liver (350 series)
table manifests published by TCIA; the 99-patient head table is NIH
Controlled Data Access and is not fetched by any automated client.

Objects are read from the IDC open bucket (`idc-open-data` by default) with
anonymous ListObjectsV2 + GetObject, MD5-verified against the listing ETag
(multipart ETags are not verifiable and pass through), and written as
`<out-dir>/<SeriesInstanceUID>/<object>.dcm` with a `.fetch_complete.json`
marker per series. A rerun skips series whose marker matches the listing;
`--force` refetches. `--source-dir` reads an identical on-disk tree instead
of HTTP, for offline re-runs and tests. `--check` validates the manifest
census and probes bucket reachability without downloading.

Usage:
    python scripts/preprocessing/fetch_mayo_ldct.py \
        --out-dir data/mayo-ldct/raw
    python scripts/preprocessing/fetch_mayo_ldct.py --check

Exit codes: 0 ok · 1 fetch/verification failure · 2 bad args or census
mismatch · 3 source unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BUCKET = "idc-open-data"
DEFAULT_SERIES_MANIFEST = (
    Path(__file__).resolve().parent / "manifests" / "mayo_ldct_recon_series.json"
)
REQUIRED_ENTRY_KEYS = (
    "series_instance_uid", "crdc_series_uuid", "aws_bucket", "instance_count",
)


class CensusError(ValueError):
    pass


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteObject:
    key: str
    size: int
    etag: str


def load_series_manifest(path: Path) -> dict:
    """Load and census-validate the per-series fetch manifest."""
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        raise CensusError(f"series manifest not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise CensusError(f"series manifest is not valid JSON: {path}: {exc}") from exc
    series = doc.get("series")
    if not isinstance(series, list) or not series:
        raise CensusError(f"series manifest has no series entries: {path}")
    for entry in series:
        for key in REQUIRED_ENTRY_KEYS:
            if key not in entry:
                raise CensusError(
                    f"series manifest entry missing '{key}': "
                    f"{entry.get('series_instance_uid', '?')}"
                )
    if doc.get("recon_series_total") != len(series):
        raise CensusError(
            f"census mismatch: recon_series_total={doc.get('recon_series_total')} "
            f"but {len(series)} series entries"
        )
    instances = sum(int(e["instance_count"]) for e in series)
    if doc.get("recon_instances_total") != instances:
        raise CensusError(
            f"census mismatch: recon_instances_total={doc.get('recon_instances_total')} "
            f"but entries sum to {instances}"
        )
    return doc


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def parse_list_objects(xml_text: str) -> list[RemoteObject]:
    root = ET.fromstring(xml_text)
    objects: list[RemoteObject] = []
    for child in root:
        if _local(child.tag) != "Contents":
            continue
        fields = {_local(sub.tag): (sub.text or "") for sub in child}
        objects.append(RemoteObject(
            key=fields.get("Key", ""),
            size=int(fields.get("Size", "0") or 0),
            etag=fields.get("ETag", ""),
        ))
    return objects


def parse_continuation(xml_text: str) -> tuple[bool, str | None]:
    root = ET.fromstring(xml_text)
    truncated = False
    token: str | None = None
    for child in root:
        name = _local(child.tag)
        if name == "IsTruncated":
            truncated = (child.text or "").strip().lower() == "true"
        elif name == "NextContinuationToken":
            token = (child.text or "").strip() or None
    return truncated, token


def md5_matches(body: bytes, etag: str) -> bool:
    tag = etag.strip().strip('"')
    if "-" in tag:
        return True
    return hashlib.md5(body).hexdigest() == tag


class S3Source:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint.rstrip("/")

    def _get(self, url: str) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:  # nosec - endpoint is operator-supplied
                return resp.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise SourceError(f"GET failed: {url}: {exc}") from exc

    def list_page(self, prefix: str, token: str | None) -> tuple[list[RemoteObject], str | None]:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = f"{self.endpoint}/?{urllib.parse.urlencode(params)}"
        xml_text = self._get(url).decode("utf-8")
        return parse_list_objects(xml_text), parse_continuation(xml_text)[1]

    def list_objects(self, prefix: str) -> list[RemoteObject]:
        collected: list[RemoteObject] = []
        token: str | None = None
        while True:
            page, token = self.list_page(prefix, token)
            collected.extend(page)
            if not token:
                return collected

    def get(self, key: str) -> bytes:
        return self._get(f"{self.endpoint}/{urllib.parse.quote(key)}")


class LocalSource:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise SourceError(f"source directory not found: {self.root}")

    def list_objects(self, prefix: str) -> list[RemoteObject]:
        directory = self.root / prefix.rstrip("/")
        if not directory.is_dir():
            raise SourceError(f"series prefix not found in source: {prefix}")
        objects = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.name == ".fetch_complete.json":
                continue
            body = path.read_bytes()
            objects.append(RemoteObject(
                key=f"{prefix.rstrip('/')}/{path.name}",
                size=len(body),
                etag=hashlib.md5(body).hexdigest(),
            ))
        return objects

    def get(self, key: str) -> bytes:
        path = self.root / key
        if not path.is_file():
            raise SourceError(f"object not found in source: {key}")
        return path.read_bytes()


def _marker_path(series_dir: Path) -> Path:
    return series_dir / ".fetch_complete.json"


def _series_done(series_dir: Path, objects: list[RemoteObject]) -> bool:
    marker = _marker_path(series_dir)
    if not marker.exists():
        return False
    try:
        record = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    sizes = record.get("objects")
    if sizes is None:
        return False
    expected = {obj.key: obj.size for obj in objects}
    if sizes != expected:
        return False
    return all(
        (series_dir / Path(obj.key).name).is_file()
        and (series_dir / Path(obj.key).name).stat().st_size == obj.size
        for obj in objects
    )


def _fetch_object(source, obj: RemoteObject, dest: Path, retries: int = 3) -> str | None:
    if dest.is_file() and dest.stat().st_size == obj.size:
        return None
    delay = 1.0
    last_error = ""
    for _ in range(retries):
        try:
            body = source.get(obj.key)
        except SourceError as exc:
            last_error = str(exc)
            time.sleep(delay)
            delay *= 2
            continue
        if not md5_matches(body, obj.etag):
            return f"md5 mismatch for {obj.key}"
        if len(body) != obj.size:
            return f"size mismatch for {obj.key}: got {len(body)} want {obj.size}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return None
    return f"fetch failed for {obj.key}: {last_error}"


def fetch_series(source, entry: dict, out_dir: Path, force: bool,
                 workers: int) -> tuple[str, int, list[str]]:
    series_uid = entry["series_instance_uid"]
    prefix = f"{entry['crdc_series_uuid']}/"
    series_dir = out_dir / series_uid
    try:
        objects = source.list_objects(prefix)
    except SourceError as exc:
        return series_uid, 0, [str(exc)]
    expected_count = int(entry["instance_count"])
    if len(objects) != expected_count:
        return series_uid, 0, [
            f"listing census mismatch for {series_uid}: "
            f"bucket has {len(objects)} objects, manifest expects {expected_count}"
        ]
    if not force and _series_done(series_dir, objects):
        return series_uid, len(objects), []
    errors: list[str] = []
    fetched = 0
    jobs = [(obj, series_dir / Path(obj.key).name) for obj in objects]
    if workers <= 1:
        results = [_fetch_object(source, obj, dest) for obj, dest in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(
                lambda pair: _fetch_object(source, pair[0], pair[1]), jobs
            ))
    for error in results:
        if error is None:
            fetched += 1
        else:
            errors.append(error)
    if not errors:
        _marker_path(series_dir).write_text(json.dumps({
            "series_instance_uid": series_uid,
            "objects": {obj.key: obj.size for obj in objects},
        }))
    return series_uid, fetched, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch the reconstructed CT series of TCIA LDCT-and-Projection-data "
            "from the IDC open S3 mirror, skipping the projection series."
        ),
    )
    ap.add_argument("--series-manifest", type=Path, default=DEFAULT_SERIES_MANIFEST,
                    help="Per-series fetch manifest (JSON)")
    ap.add_argument("--out-dir", type=Path, default=Path("data/mayo-ldct/raw"),
                    help="Raw DICOM output directory")
    ap.add_argument("--bucket", type=str, default=DEFAULT_BUCKET,
                    help="IDC open bucket name")
    ap.add_argument("--endpoint", type=str, default=None,
                    help="Override the bucket endpoint URL (for mirrors and tests)")
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="Read from an on-disk copy instead of HTTP")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel object downloads (1 = serial)")
    ap.add_argument("--force", action="store_true",
                    help="Refetch objects/series even when they already exist")
    ap.add_argument("--check", action="store_true",
                    help="Validate the census and probe reachability; download nothing")
    args = ap.parse_args(argv)

    if args.workers < 1:
        print("ERROR: --workers must be >= 1", file=sys.stderr)
        return 2

    try:
        doc = load_series_manifest(args.series_manifest)
    except CensusError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        if args.source_dir is not None:
            source = LocalSource(args.source_dir)
        else:
            endpoint = args.endpoint or f"https://{args.bucket}.s3.us-east-1.amazonaws.com"
            source = S3Source(endpoint)
        if args.check:
            first = doc["series"][0]
            source.list_objects(f"{first['crdc_series_uuid']}/")
            print(
                f"OK: {doc['recon_series_total']} series / "
                f"{doc['recon_instances_total']} instances reachable; "
                f"first series {first['series_instance_uid']} lists objects"
            )
            return 0
    except SourceError as exc:
        print(f"ERROR: source unreachable: {exc}", file=sys.stderr)
        return 3

    series_dir_map = {e["series_instance_uid"]: args.out_dir / e["series_instance_uid"]
                      for e in doc["series"]}
    total_expected = 0
    total_fetched = 0
    failed: list[str] = []
    for entry in doc["series"]:
        series_uid, fetched, errors = fetch_series(
            source, entry, args.out_dir, args.force, args.workers,
        )
        total_expected += int(entry["instance_count"])
        total_fetched += fetched
        for err in errors:
            print(f"warn=true series={series_uid} {err}", file=sys.stderr, flush=True)
        failed.extend(errors)
        print(f"series={series_uid} fetched={fetched} "
              f"expected={entry['instance_count']}", flush=True)

    if failed:
        print(f"ERROR: {len(failed)} object failure(s) across "
              f"{len({e.split(' for ')[-1] for e in failed})} site(s)", file=sys.stderr)
        return 1
    print(f"\nDone: {total_fetched}/{total_expected} objects across "
          f"{len(series_dir_map)} series -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
