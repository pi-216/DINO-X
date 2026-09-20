#!/usr/bin/env python3
"""Minimal stdlib DICOM identity-tag scanner (targeted byte-search).

Reads (0010,0020) PatientID, (0020,000D) StudyInstanceUID, (0020,000E)
SeriesInstanceUID from explicit-VR little-endian DICOM files (ASCII values only).
Robust against SQ/OB/odd-padding by finding each tag pattern directly in place
rather than sequentially walking the element stream.
"""
import struct
import sys
from pathlib import Path

# (group, element) -> semantic name, all ASCII VRs (LO/SH/UI)
TAGS = {
    (0x0010, 0x0020): "PatientID",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
}

LONG_VR = {"OB", "OW", "OF", "SQ", "UT", "UN"}


def find_tag(data: bytes, group: int, elem: int):
    """Return decoded value for a tag, or None. Searches all occurrences."""
    pattern = struct.pack("<HH", group, elem)
    start = 132 if data[128:132] == b"DICM" else 0
    idx = start
    while True:
        idx = data.find(pattern, idx)
        if idx < 0:
            return None
        # parse in place
        if idx + 12 > len(data):
            idx += 4
            continue
        vr = data[idx + 4:idx + 6].decode("ascii", "replace")
        if vr in LONG_VR:
            length = struct.unpack_from("<I", data, idx + 8)[0]
            offset = idx + 12
        else:
            length = struct.unpack_from("<H", data, idx + 6)[0]
            offset = idx + 8
        if offset + length > len(data):
            idx += 4
            continue
        raw = data[offset:offset + length]
        val = raw.split(b"\x00")[0].decode("ascii", "replace").strip()
        if val:
            return val
        idx += 4


def main():
    ct_root = Path(sys.argv[1])
    series_dirs = sorted(d for d in ct_root.iterdir() if d.is_dir())
    print(f"series_dirs={len(series_dirs)}", file=sys.stderr)
    rows = []
    for d in series_dirs:
        dcms = sorted(d.glob("*.dcm"))
        if not dcms:
            continue
        data = (d / dcms[0]).read_bytes()
        tag = {}
        for (g, e), name in TAGS.items():
            tag[name] = find_tag(data, g, e)
        tag["series_dir_name"] = d.name
        rows.append(tag)
    import json
    print(json.dumps(rows))


if __name__ == "__main__":
    main()