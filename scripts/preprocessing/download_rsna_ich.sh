#!/usr/bin/env bash
set -uo pipefail

COMPETITION="rsna-intracranial-hemorrhage-detection"
RAW_DIR="data/rsna-ich/raw"
PROCESSED_DIR="data/processed"
DATASET_NAME="rsna-ich"
MANIFEST_PATH="data/manifests/rsna_ich.parquet"
KAGGLE_BIN="${KAGGLE_BIN:-kaggle}"
STAGE1_ZIP="stage_1_train_images.zip"

MODE="ingest"
if [ "${1:-}" = "--check" ]; then
  MODE="check"
fi

if [ "$MODE" = "check" ]; then
  echo "Probing Kaggle access for competition '$COMPETITION'..."
  FILE_LIST=$("$KAGGLE_BIN" competitions files -c "$COMPETITION" 2>&1)
  rc=$?
  if [ $rc -ne 0 ] || ! echo "$FILE_LIST" | grep -q "stage"; then
    echo "BLOCKED: the Kaggle API cannot list this competition's files" >&2
    echo "$FILE_LIST" >&2
    echo "" >&2
    echo "This account has not entered the RSNA ICH 2019 competition." >&2
    echo "A human must accept the competition rules in a browser:" >&2
    echo "  https://www.kaggle.com/competitions/$COMPETITION/rules" >&2
    echo "Then re-run: scripts/preprocessing/download_rsna_ich.sh --check" >&2
    exit 3
  fi
  echo "$FILE_LIST"
  echo "OK: competition files are reachable for this Kaggle account."
  exit 0
fi

echo "=== RSNA ICH Download & Preprocess Pipeline ==="

if [ -d "$RAW_DIR" ] && [ -n "$(find "$RAW_DIR" -name '*.dcm' -print -quit 2>/dev/null)" ]; then
  echo "Step 1: SKIP — DICOM files already exist in $RAW_DIR"
else
  echo "Step 1: Downloading stage-1 train DICOM from Kaggle..."
  if ! "$KAGGLE_BIN" competitions download -c "$COMPETITION" -f "$STAGE1_ZIP" -p "$RAW_DIR"; then
    echo "ERROR: could not download '$STAGE1_ZIP' from competition '$COMPETITION'." >&2
    echo "If the account has not accepted the competition rules, run:" >&2
    echo "  scripts/preprocessing/download_rsna_ich.sh --check" >&2
    echo "for the exact unblock step; if Kaggle has renamed the file, list it with:" >&2
    echo "  $KAGGLE_BIN competitions files -c $COMPETITION" >&2
    exit 1
  fi
  mkdir -p "$RAW_DIR"
  echo "  Extracting $STAGE1_ZIP ..."
  unzip -q -o "$RAW_DIR/$STAGE1_ZIP" -d "$RAW_DIR"
  rm -f "$RAW_DIR/$STAGE1_ZIP"
fi

N_FILES=$(find "$RAW_DIR" -type f | wc -l)
echo "  Raw files: $N_FILES"

echo ""
echo "Step 2: Preprocessing DICOM -> hu16 PNG slices + Parquet manifest..."
PYTHONUNBUFFERED=1 python3 scripts/preprocessing/preprocess_rsna_ich.py \
    --dicom-dir "$RAW_DIR" \
    --out-root "$PROCESSED_DIR" \
    --dataset-name "$DATASET_NAME" \
    --manifest-out "$MANIFEST_PATH"
rc=$?
if [ $rc -ne 0 ]; then
  echo "ERROR: preprocessing failed with exit code $rc" >&2
  exit $rc
fi

echo ""
echo "=== rsna-ich ready for index combination ==="
echo "  Processed:  $PROCESSED_DIR/$DATASET_NAME/"
echo "  Manifest:   $MANIFEST_PATH"
