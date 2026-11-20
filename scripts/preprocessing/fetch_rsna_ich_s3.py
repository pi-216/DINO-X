#!/usr/bin/env python3
"""Fetch RSNA ICH stage-2 DICOM slices from the RSNA-managed AWS S3 mirror.

The canonical RSNA Brain CT Hemorrhage corpus is published as one Zip64
archive (`rsna-ihd-dataset.zip`) in the S3 bucket listed in the Registry of
Open Data on AWS (https://registry.opendata.aws/rsna-intracranial-hemorrhage-detection,
s3://intracranial-hemorrhage). The archive interleaves stage_2_train,
stage_2_test and CSV members; the stage-2 train members occupy a contiguous
data region, so this tool reads only the windows those members live in via
parallel HTTP range requests — the full archive is never downloaded.

Members are CRC-32 verified before they are written; extraction is resumable
(per-member size check, `--force` to rewrite). `--check` probes reachability
and reports the member census without downloading anything. The Kaggle
competition file API for this corpus returns empty/404 listings, so this S3
path is the ingest source of record; Kaggle remains a human browser mirror.

Usage:
    python scripts/preprocessing/fetch_rsna_ich_s3.py \
        --out-dir data/rsna-ich/raw \
        --labels-out data/rsna-ich/labels/stage_2_train.csv
    python scripts/preprocessing/fetch_rsna_ich_s3.py --check

Exit codes: 0 ok · 1 member extraction/CRC failure · 2 bad args · 3 source unreachable.
"""

from __future__ import annotations

import argparse
import http.client
import struct
import sys
import threading
import urllib.parse
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL = (
    "https://intracranial-hemorrhage.s3.us-west-2.amazonaws.com/"
    "rsna-ihd-dataset.zip"
)
DEFAULT_PREFIX = "rsna-intracranial-hemorrhage-detection/stage_2_train/"
DEFAULT_LABELS_MEMBER = "rsna-intracranial-hemorrhage-detection/stage_2_train.csv"

_EOCD_SIG = b"PK\x05\x06"
_ZIP64_LOC_SIG = b"PK\x06\x07"
_ZIP64_EOCD_SIG = b"PK\x06\x06"
_CD_SIG = b"PK\x01\x02"
_LH_SIG = b"PK\x03\x04"
_LH_FIXED_LEN = 30


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    crc: int
    csize: int
    usize: int
    offset: int


class SourceError(RuntimeError):
    pass


class FileSource:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise SourceError(f"file not found: {path}")
        self.size = self.path.stat().st_size

    def fetch(self, start: int, end: int) -> bytes:
        with self.path.open("rb") as fh:
            fh.seek(start)
            data = fh.read(end - start)
        if len(data) != end - start:
            raise SourceError(f"short read at {start}")
        return data


class HttpRangeSource:
    def __init__(self, url: str, attempts: int = 4) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise SourceError(f"unsupported scheme: {url}")
        self.url = url
        self.attempts = attempts
        self._conn_cls = (
            http.client.HTTPSConnection if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        self._host = parsed.netloc
        self._target = urllib.parse.quote(parsed.path or "/", safe="/@:+&=,~")
        if parsed.query:
            self._target += "?" + parsed.query
        self._conn = self._conn_cls(self._host, timeout=180)
        self.size = self._probe_size()

    def _reconnect(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = self._conn_cls(self._host, timeout=180)

    def _request(self, start: int, end: int) -> tuple[int, dict, bytes]:
        last_error: Exception | None = None
        for _ in range(self.attempts):
            try:
                self._conn.request(
                    "GET", self._target,
                    headers={"Range": f"bytes={start}-{end - 1}"},
                )
                resp = self._conn.getresponse()
                body = resp.read()
                headers = dict(resp.getheaders())
                if resp.status in (200, 206, 404):
                    return resp.status, headers, body
                last_error = SourceError(f"status {resp.status}")
            except (http.client.HTTPException, OSError) as exc:
                last_error = exc
            self._reconnect()
        raise SourceError(f"range read failed: {last_error}")

    def _probe_size(self) -> int:
        status, headers, _ = self._request(0, 2)
        content_range = headers.get("Content-Range", "")
        if status == 206 and "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        if status == 200:
            length = headers.get("Content-Length")
            if length:
                return int(length)
        raise SourceError(f"source does not support range reads (status {status})")

    def fetch(self, start: int, end: int) -> bytes:
        status, _, body = self._request(start, end)
        if status != 206:
            raise SourceError(f"range read rejected (status {status})")
        if len(body) != end - start:
            raise SourceError(f"short range body at {start}")
        return body


def open_source(url: str) -> FileSource | HttpRangeSource:
    if "://" in url:
        return HttpRangeSource(url)
    return FileSource(url)


def _zip64_extra(
    extra: bytes, csize: int, usize: int, offset: int
) -> tuple[int, int, int]:
    pos = 0
    while pos + 4 <= len(extra):
        fid, fsize = struct.unpack("<HH", extra[pos:pos + 4])
        field = extra[pos + 4:pos + 4 + fsize]
        if fid == 0x0001:
            fpos = 0
            if usize == 0xFFFFFFFF and fpos + 8 <= len(field):
                usize = struct.unpack("<Q", field[fpos:fpos + 8])[0]
                fpos += 8
            if csize == 0xFFFFFFFF and fpos + 8 <= len(field):
                csize = struct.unpack("<Q", field[fpos:fpos + 8])[0]
                fpos += 8
            if offset == 0xFFFFFFFF and fpos + 8 <= len(field):
                offset = struct.unpack("<Q", field[fpos:fpos + 8])[0]
        pos += 4 + fsize
    return csize, usize, offset


def load_central_directory(
    source: FileSource | HttpRangeSource,
) -> list[Member]:
    tail_len = min(source.size, 65536)
    tail = source.fetch(source.size - tail_len, source.size)
    eocd = tail.rfind(_EOCD_SIG)
    if eocd < 0:
        raise SourceError("end-of-central-directory not found")
    eocd_fields = struct.unpack("<IHHHHIIH", tail[eocd:eocd + 22])
    cd_size, cd_off = eocd_fields[5], eocd_fields[6]
    locator = tail.rfind(_ZIP64_LOC_SIG, 0, eocd)
    if locator >= 0:
        z64_off = struct.unpack("<IQI", tail[locator + 4:locator + 20])[1]
        record = source.fetch(z64_off, z64_off + 56)
        if record[:4] != _ZIP64_EOCD_SIG:
            raise SourceError("bad zip64 end-of-central-directory")
        fields = struct.unpack("<IQHHIIQQQQ", record)
        cd_size, cd_off = fields[8], fields[9]
    if cd_off + cd_size > source.size:
        raise SourceError("central directory out of bounds")
    cd = source.fetch(cd_off, cd_off + cd_size)
    members: list[Member] = []
    pos = 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == _CD_SIG:
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
        method, crc = fields[4], fields[7]
        csize, usize = fields[8], fields[9]
        nlen, elen, clen = fields[10], fields[11], fields[12]
        offset = fields[16]
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8", errors="replace")
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + elen]
        csize, usize, offset = _zip64_extra(extra, csize, usize, offset)
        members.append(Member(name=name, method=method, crc=crc,
                              csize=csize, usize=usize, offset=offset))
        pos += 46 + nlen + elen + clen
    return members


def select_members(members: list[Member], prefix: str) -> list[Member]:
    return sorted((m for m in members if m.name.startswith(prefix)),
                  key=lambda m: m.offset)


def decompress_member(member: Member, data: bytes) -> bytes:
    if member.method == 0:
        raw = data
    elif member.method == 8:
        raw = zlib.decompressobj(-15).decompress(data, member.usize + 1)
    else:
        raise SourceError(f"unsupported zip method {member.method} for {member.name}")
    if len(raw) != member.usize:
        raise SourceError(f"size mismatch for {member.name}")
    if zlib.crc32(raw) & 0xFFFFFFFF != member.crc:
        raise SourceError(f"CRC mismatch for {member.name}")
    return raw


def _member_data(
    source: FileSource | HttpRangeSource, member: Member
) -> bytes:
    head = source.fetch(member.offset, member.offset + _LH_FIXED_LEN)
    if head[:4] != _LH_SIG:
        raise SourceError(f"bad local header for {member.name}")
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = member.offset + _LH_FIXED_LEN + nlen + elen
    return decompress_member(member, source.fetch(start, start + member.csize))


def _member_in_window(
    member: Member, buf: bytes, window_start: int
) -> bytes:
    rel = member.offset - window_start
    if rel < 0 or rel + _LH_FIXED_LEN > len(buf):
        raise SourceError(f"local header outside window for {member.name}")
    head = buf[rel:rel + _LH_FIXED_LEN]
    if head[:4] != _LH_SIG:
        raise SourceError(f"bad local header for {member.name}")
    nlen, elen = struct.unpack("<HH", head[26:30])
    data_start = rel + _LH_FIXED_LEN + nlen + elen
    if data_start + member.csize > len(buf):
        raise SourceError(f"data outside window for {member.name}")
    return decompress_member(member, buf[data_start:data_start + member.csize])


def plan_windows(
    members: list[Member], window_bytes: int
) -> list[tuple[int, int | None, list[Member]]]:
    windows: list[tuple[int, int | None, list[Member]]] = []
    i = 0
    while i < len(members):
        start = members[i].offset
        j = i + 1
        while j < len(members) and members[j].offset - start < window_bytes:
            j += 1
        if j == i + 1:
            j += 1
        end = members[j].offset if j < len(members) else None
        windows.append((start, end, members[i:j]))
        i = j
    return windows


_TLS = threading.local()


def _thread_source(url: str) -> FileSource | HttpRangeSource:
    src = getattr(_TLS, "source", None)
    if src is None:
        src = open_source(url)
        _TLS.source = src
    return src


def _extract_window(args: tuple) -> list[tuple[str, str]]:
    url, window_start, window_end, window_members, prefix, out_dir, force = args
    pending = []
    for member in window_members:
        out_path = out_dir / member.name[len(prefix):]
        if not force and out_path.is_file() and out_path.stat().st_size == member.usize:
            pending.append((member.name, "skipped"))
        else:
            pending.append((member.name, "pending", member, out_path))
    if all(item[1] == "skipped" for item in pending):
        return [(item[0], "skipped") for item in pending]
    source = _thread_source(url)
    end = window_end
    if end is None:
        end = (window_members[-1].offset + _LH_FIXED_LEN
               + len(window_members[-1].name) + 65536
               + window_members[-1].csize + window_members[-1].usize)
        end = min(end, source.size)
    buf = source.fetch(window_start, end)
    results: list[tuple[str, str]] = []
    for item in pending:
        if item[1] == "skipped":
            results.append((item[0], "skipped"))
            continue
        member, out_path = item[2], item[3]
        try:
            try:
                raw = _member_in_window(member, buf, window_start)
            except SourceError:
                raw = _member_data(source, member)
        except (SourceError, zlib.error) as exc:
            results.append((member.name, f"failed: {exc}"))
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".part")
        tmp_path.write_bytes(raw)
        tmp_path.replace(out_path)
        results.append((member.name, "extracted"))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Range-extract RSNA ICH stage-2 train DICOM members from "
                    "the RSNA AWS S3 Zip64 archive.",
    )
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="Archive URL or local path (default: RSNA S3 mirror)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for extracted DICOM (members are written "
                         "relative to --prefix)")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help="Member path prefix to extract")
    ap.add_argument("--labels-out", type=Path, default=None,
                    help="Also extract the labels CSV member to this path")
    ap.add_argument("--labels-member", default=DEFAULT_LABELS_MEMBER,
                    help="Member name of the labels CSV")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel range readers")
    ap.add_argument("--window-mb", type=float, default=32.0,
                    help="Maximum archive window fetched per range request")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract members that already exist")
    ap.add_argument("--check", action="store_true",
                    help="Probe reachability and report the member census only")
    args = ap.parse_args(argv)

    if args.window_mb <= 0:
        print("ERROR: --window-mb must be positive", file=sys.stderr)
        return 2

    try:
        source = open_source(args.url)
        members = load_central_directory(source)
    except SourceError as exc:
        print(f"ERROR: source unavailable: {exc}", file=sys.stderr)
        return 3

    chosen = select_members(members, args.prefix)
    if args.check:
        print(f"check=true url={args.url} members_total={len(members)} "
              f"prefix={args.prefix} prefix_members={len(chosen)} "
              f"prefix_compressed_bytes={sum(m.csize for m in chosen)} "
              f"prefix_uncompressed_bytes={sum(m.usize for m in chosen)}")
        return 0

    if args.out_dir is None:
        print("ERROR: --out-dir is required unless --check", file=sys.stderr)
        return 2
    if not chosen:
        print(f"ERROR: no members match prefix: {args.prefix}", file=sys.stderr)
        return 2

    window_bytes = int(args.window_mb * 1024 * 1024)
    tasks = [
        (args.url, start, end, window, args.prefix, args.out_dir, args.force)
        for start, end, window in plan_windows(chosen, window_bytes)
    ]
    extracted = skipped = failed = 0
    total = len(chosen)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for results in pool.map(_extract_window, tasks):
            for name, status in results:
                if status == "extracted":
                    extracted += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"warn=true member={name} {status}", file=sys.stderr)
            done = extracted + skipped + failed
            if done % 2000 < len(results) or done == total:
                print(f"progress={done}/{total} extracted={extracted} "
                      f"skipped={skipped} failed={failed}")

    if args.labels_out is not None:
        labels = [m for m in members if m.name == args.labels_member]
        if not labels:
            print(f"ERROR: labels member not found: {args.labels_member}",
                  file=sys.stderr)
            return 1
        raw = _member_data(source, labels[0])
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        args.labels_out.write_bytes(raw)
        print(f"labels=true member={args.labels_member} bytes={len(raw)}")

    print(f"done=true extracted={extracted} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
