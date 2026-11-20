#!/usr/bin/env bash
set -uo pipefail

COMPETITION="rsna-intracranial-hemorrhage-detection"
S3_ZIP="${RSNA_S3_ZIP_URL:-https://intracranial-hemorrhage.s3.us-west-2.amazonaws.com/rsna-ihd-dataset.zip}"
RAW_DIR="${RSNA_RAW_DIR:-data/rsna-ich/raw}"
PROCESSED_DIR="${RSNA_PROCESSED_DIR:-data/processed}"
LABELS_OUT="${RSNA_LABELS_OUT:-data/rsna-ich/labels/stage_2_train.csv}"
DATASET_NAME="rsna-ich"
MANIFEST_PATH="${RSNA_MANIFEST_PATH:-data/manifests/rsna_ich.parquet}"
FETCH_SCRIPT="scripts/preprocessing/fetch_rsna_ich_s3.py"
PREPROCESS_SCRIPT="scripts/preprocessing/preprocess_rsna_ich.py"
WORKERS="${RSNA_WORKERS:-8}"
STAGE1_ZIP="stage_1_train_images.zip"

MODE="ingest"
SOURCE="aws"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE="check" ;;
    --source) shift; SOURCE="${1:-}" ;;
    *) echo "usage: $0 [--check] [--source aws|kaggle]" >&2; exit 2 ;;
  esac
  shift
done
case "$SOURCE" in
  aws|kaggle) ;;
  *) echo "ERROR: --source must be aws or kaggle" >&2; exit 2 ;;
esac

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x .venv/bin/python3 ]; then
    PYTHON_BIN=".venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ -z "${KAGGLE_BIN:-}" ]; then
  if command -v kaggle >/dev/null 2>&1; then
    KAGGLE_BIN="kaggle"
  elif [ -x .venv/bin/kaggle ]; then
    KAGGLE_BIN=".venv/bin/kaggle"
  else
    KAGGLE_BIN="kaggle"
  fi
fi

require_kaggle() {
  if command -v "$KAGGLE_BIN" >/dev/null 2>&1 || [ -x "$KAGGLE_BIN" ]; then
    return 0
  fi
  echo "ERROR: kaggle CLI not found (KAGGLE_BIN=$KAGGLE_BIN)." >&2
  echo "Install it into the project venv: .venv/bin/pip install kaggle" >&2
  echo "Or use the RSNA AWS S3 mirror instead: $0 --source aws" >&2
  exit 4
}

run_preprocess() {
  echo "Step 2: Preprocessing DICOM -> hu16 PNG slices + Parquet manifest..."
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$PREPROCESS_SCRIPT" \
      --dicom-dir "$RAW_DIR" \
      --out-root "$PROCESSED_DIR" \
      --dataset-name "$DATASET_NAME" \
      --workers "$WORKERS" \
      --manifest-out "$MANIFEST_PATH"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "ERROR: preprocessing failed with exit code $rc" >&2
    exit $rc
  fi
  echo ""
  echo "=== rsna-ich ready for index combination ==="
  echo "  Processed:  $PROCESSED_DIR/$DATASET_NAME/"
  echo "  Manifest:   $MANIFEST_PATH"
}

if [ "$MODE" = "check" ]; then
  if [ "$SOURCE" = "aws" ]; then
    "$PYTHON_BIN" "$FETCH_SCRIPT" --url "$S3_ZIP" --check
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "BLOCKED: the RSNA S3 mirror is unreachable for this host" >&2
      echo "Expected archive: $S3_ZIP" >&2
      echo "Registry entry: https://registry.opendata.aws/rsna-intracranial-hemorrhage-detection" >&2
    fi
    exit $rc
  fi
  echo "Probing Kaggle access for competition '$COMPETITION'..."
  require_kaggle
  FILE_LIST=$("$KAGGLE_BIN" competitions files -c "$COMPETITION" 2>&1)
  rc=$?
  if [ $rc -ne 0 ] || ! echo "$FILE_LIST" | grep -q "stage"; then
    echo "BLOCKED: the Kaggle API cannot list this competition's files" >&2
    echo "$FILE_LIST" >&2
    echo "" >&2
    echo "The Kaggle file API for this competition returns empty/404 listings;" >&2
    echo "the stage archives are only browsable on the Kaggle website. The" >&2
    echo "canonical corpus is served by the RSNA AWS S3 mirror — use:" >&2
    echo "  $0 --source aws" >&2
    exit 3
  fi
  echo "$FILE_LIST"
  echo "OK: competition files are reachable for this Kaggle account."
  exit 0
fi

echo "=== RSNA ICH Download & Preprocess Pipeline (source: $SOURCE) ==="

if [ -d "$RAW_DIR" ] && [ -n "$(find "$RAW_DIR" -name '*.dcm' -print -quit 2>/dev/null)" ]; then
  echo "Step 1: SKIP — DICOM files already exist in $RAW_DIR"
else
  if [ "$SOURCE" = "aws" ]; then
    echo "Step 1: Range-extracting stage_2_train DICOM from the RSNA S3 mirror..."
    PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$FETCH_SCRIPT" \
        --url "$S3_ZIP" \
        --out-dir "$RAW_DIR" \
        --labels-out "$LABELS_OUT" \
        --workers "$WORKERS"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "ERROR: S3 range extraction failed with exit code $rc" >&2
      echo "Probe reachability with: $0 --check" >&2
      exit $rc
    fi
  else
    echo "Step 1: Downloading stage-1 train DICOM from Kaggle..."
    require_kaggle
    if ! "$KAGGLE_BIN" competitions download -c "$COMPETITION" -f "$STAGE1_ZIP" -p "$RAW_DIR"; then
      echo "ERROR: could not download '$STAGE1_ZIP' from competition '$COMPETITION'." >&2
      echo "The Kaggle file API for this competition returns 404s; the canonical" >&2
      echo "corpus is on the RSNA AWS S3 mirror — run: $0 --source aws" >&2
      exit 1
    fi
    mkdir -p "$RAW_DIR"
    echo "  Extracting $STAGE1_ZIP ..."
    unzip -q -o "$RAW_DIR/$STAGE1_ZIP" -d "$RAW_DIR"
    rm -f "$RAW_DIR/$STAGE1_ZIP"
  fi
fi

N_DCM=$(find "$RAW_DIR" -name '*.dcm' | wc -l)
echo "  DICOM files: $N_DCM"
echo ""

run_preprocess
