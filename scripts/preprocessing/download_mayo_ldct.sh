#!/usr/bin/env bash
set -uo pipefail

DATASET_NAME="mayo-ldct"
RAW_DIR="${MAYO_RAW_DIR:-data/mayo-ldct/raw}"
PROCESSED_DIR="${MAYO_PROCESSED_DIR:-data/processed}"
MANIFEST_PATH="${MAYO_MANIFEST_PATH:-data/manifests/mayo_ldct.parquet}"
FETCH_SCRIPT="scripts/preprocessing/fetch_mayo_ldct.py"
PREPROCESS_SCRIPT="scripts/preprocessing/preprocess_mayo_ldct.py"
WORKERS="${MAYO_WORKERS:-8}"

MODE="ingest"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="check" ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x .venv/bin/python3 ]; then
    PYTHON_BIN=".venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ "$MODE" = "check" ]; then
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$FETCH_SCRIPT" --check
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "BLOCKED: the IDC open bucket is unreachable for this host" >&2
    echo "Probe: https://idc-open-data.s3.us-east-1.amazonaws.com" >&2
  fi
  exit $rc
fi

echo "=== Mayo LDCT Download & Preprocess Pipeline ==="

if [ -d "$RAW_DIR" ] && [ -n "$(find "$RAW_DIR" -name '*.dcm' -print -quit 2>/dev/null)" ]; then
  echo "Step 1: SKIP — DICOM files already exist in $RAW_DIR (marker-checked refetch still runs)"
fi

echo "Step 1: Fetching 299 reconstructed CT series from the IDC open bucket..."
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$FETCH_SCRIPT" \
    --out-dir "$RAW_DIR" \
    --workers "$WORKERS"
rc=$?
if [ $rc -ne 0 ]; then
  echo "ERROR: fetch failed with exit code $rc" >&2
  echo "Probe reachability with: $0 --check" >&2
  exit $rc
fi

N_DCM=$(find "$RAW_DIR" -name '*.dcm' | wc -l)
echo "  DICOM files: $N_DCM"
echo ""

echo "Step 2: Preprocessing DICOM -> hu16 PNG slices + Parquet manifest..."
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PREPROCESS_SCRIPT" \
    --dicom-dir "$RAW_DIR" \
    --out-root "$PROCESSED_DIR" \
    --dataset-name "$DATASET_NAME" \
    --workers "$WORKERS" \
    --manifest-out "$MANIFEST_PATH"
rc=$?
if [ $rc -ne 0 ]; then
  echo "ERROR: preprocessing failed with exit code $rc" >&2
  exit $rc
fi

echo ""
echo "=== mayo-ldct ready for index combination ==="
echo "  Processed:  $PROCESSED_DIR/$DATASET_NAME/"
echo "  Manifest:   $MANIFEST_PATH"
