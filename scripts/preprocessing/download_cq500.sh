#!/usr/bin/env bash
set -uo pipefail

DATASET="crawford/qureai-headct"
RAW_DIR="data/cq500/raw"
PROCESSED_DIR="data/processed"
DATASET_NAME="cq500"
MANIFEST_PATH="data/manifests/cq500.parquet"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"

if [ "${1:-}" = "--check" ]; then
  echo "Probing Kaggle access for dataset '$DATASET'..."
  FILE_LIST=$("$KAGGLE_BIN" datasets files "$DATASET" 2>&1)
  rc=$?
  if [ $rc -ne 0 ] || ! echo "$FILE_LIST" | grep -qi "qct"; then
    echo "BLOCKED: the Kaggle API cannot list this dataset's files" >&2
    echo "$FILE_LIST" >&2
    echo "" >&2
    echo "Create a Kaggle API token and install it:" >&2
    echo "  https://www.kaggle.com/settings -> API -> Create New Token" >&2
    echo "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json" >&2
    echo "Then re-run: scripts/preprocessing/download_cq500.sh --check" >&2
    exit 3
  fi
  echo "$FILE_LIST"
  echo "OK: dataset files are reachable for this Kaggle account."
  exit 0
fi

echo "=== CQ500 Download & Preprocess Pipeline ==="

if [ -d "$RAW_DIR" ] && [ -n "$(find "$RAW_DIR" -name '*.dcm' -print -quit 2>/dev/null)" ]; then
  echo "Step 1: SKIP — DICOM files already exist in $RAW_DIR"
  echo "  $(find "$RAW_DIR" -name '*.dcm' | wc -l) DICOM files found"
else
  echo "Step 1: Downloading CQ500 from Kaggle (~26.6 GB)..."
  if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: Kaggle API credentials not found." >&2
    echo "Run scripts/preprocessing/download_cq500.sh --check" >&2
    echo "for the exact credential setup step, or download manually:" >&2
    echo "  https://www.kaggle.com/datasets/$DATASET" >&2
    echo "  Extract to: $RAW_DIR" >&2
    exit 1
  fi
  mkdir -p "$RAW_DIR"
  if ! "$KAGGLE_BIN" datasets download "$DATASET" -p "$RAW_DIR" --unzip; then
    echo "ERROR: could not download '$DATASET' from Kaggle." >&2
    echo "Run scripts/preprocessing/download_cq500.sh --check" >&2
    echo "for the exact unblock step." >&2
    exit 1
  fi
fi

N_DCM=$(find "$RAW_DIR" -name '*.dcm' | wc -l)
echo "  Raw DICOM files: $N_DCM"

echo ""
echo "Step 2: Preprocessing DICOM -> canonical hu16 PNG slices + Parquet manifest..."
PYTHONUNBUFFERED=1 python3 scripts/preprocessing/preprocess_cq500.py \
    --dicom-dir "$RAW_DIR" \
    --out-root "$PROCESSED_DIR" \
    --dataset-name "$DATASET_NAME" \
    --manifest-out "$MANIFEST_PATH"
rc=$?
if [ $rc -ne 0 ]; then
  echo "ERROR: preprocessing failed with exit code $rc" >&2
  exit $rc
fi

INDEX_FILE="$PROCESSED_DIR/$DATASET_NAME/index.csv"
N_SLICES=$(($(wc -l < "$INDEX_FILE") - 1))
N_SERIES=$(tail -n +2 "$INDEX_FILE" | cut -d',' -f2 | sort -u | wc -l)
echo ""
echo "=== cq500 ready for index combination ==="
echo "  Series:     $N_SERIES"
echo "  Slices:     $N_SLICES"
echo "  Processed:  $PROCESSED_DIR/$DATASET_NAME/"
echo "  Manifest:   $MANIFEST_PATH"
