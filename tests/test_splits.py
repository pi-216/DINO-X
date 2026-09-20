"""Tests for zoo.splits — patient-level splits and DICOM identity."""

import struct

from zoo.splits import assert_no_series_leak, patient_split, read_patient_id


class TestPatientSplit:
    def test_patient_grouped_no_leak(self):
        # Patient A has 3 slices, B has 2, C has 1.
        pids = ["A", "A", "A", "B", "B", "C"]
        train, val = patient_split(pids, test_size=0.33, seed=0)
        # all indices present exactly once
        assert sorted(train + val) == list(range(6))
        # a patient's indices never split across train AND val
        for pid in ("A", "B", "C"):
            idxs = {i for i, p in enumerate(pids) if p == pid}
            assert idxs <= set(train) or idxs <= set(val), f"patient {pid} leaked"
        # at least one patient on each side
        assert set(pids[i] for i in train) and set(pids[i] for i in val)

    def test_none_patients_are_singletons(self):
        # Two None entries must NOT pool together (missing identity).
        pids = [None, None, "A", "A"]
        train, val = patient_split(pids, test_size=0.5, seed=1)
        assert sorted(train + val) == [0, 1, 2, 3]
        # A's two indices stay together
        a_idx = {2, 3}
        assert a_idx <= set(train) or a_idx <= set(val)

    def test_empty_raises(self):
        import pytest as _pytest
        with _pytest.raises(ValueError):
            patient_split([], test_size=0.2, seed=0)


class TestSeriesLeakAssertion:
    def test_no_overlap_passes(self):
        assert_no_series_leak({"s1", "s2"}, {"s3", "s4"})

    def test_overlap_raises(self):
        import pytest as _pytest
        with _pytest.raises(AssertionError):
            assert_no_series_leak({"s1", "s2"}, {"s2", "s3"})


class TestPatientLeakAssertion:
    def test_no_overlap_passes(self):
        from zoo.splits import assert_no_patient_leak
        assert_no_patient_leak({"P1", "P2"}, {"P3"})

    def test_overlap_raises(self):
        import pytest as _pytest
        from zoo.splits import assert_no_patient_leak
        with _pytest.raises(AssertionError):
            assert_no_patient_leak({"P1", "P2"}, {"P2", "P3"})


class TestReadPatientID:
    def _dicom(self, patient_id: str) -> bytes:
        # minimal explicit-VR little-endian DICOM with preamble + PatientID
        pat = patient_id.encode("ascii")
        vr = b"LO"
        length = struct.pack("<H", len(pat))
        elem = struct.pack("<HH", 0x0010, 0x0020) + vr + length + pat
        # pad to even
        if len(pat) % 2:
            elem += b" "
        preamble = b"\x00" * 128 + b"DICM"
        return preamble + elem

    def test_reads_patient_id(self):
        assert read_patient_id(self._dicom("CRLM-CT-1193")) == "CRLM-CT-1193"

    def test_returns_none_if_absent(self):
        preamble = b"\x00" * 128 + b"DICM"
        assert read_patient_id(preamble) is None