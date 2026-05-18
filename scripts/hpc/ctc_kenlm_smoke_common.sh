#!/bin/bash
set -euo pipefail

if [[ -z "${SMOKE_DECODERS:-}" ]]; then
  echo "ERROR: SMOKE_DECODERS must be set by the smoke wrapper." >&2
  exit 2
fi

export DANISH_ASR_PROJECT_DIR="${DANISH_ASR_PROJECT_DIR:-"$HOME/danish_asr"}"
BASE_MANIFEST="${CTC_KENLM_MANIFEST:-$DANISH_ASR_PROJECT_DIR/configs/eval/ctc_kenlm_finetuned_hpc.yaml}"
SMOKE_MODEL_LABEL="${SMOKE_MODEL_LABEL:-300m_e6_50k}"
SMOKE_SPLIT_LABEL="${SMOKE_SPLIT_LABEL:-combined}"
SMOKE_DATASET_SPLIT="${SMOKE_DATASET_SPLIT:-test}"
SMOKE_TMP_ROOT="${SMOKE_TMP_ROOT:-${TMPDIR:-/tmp}}"
SMOKE_MANIFEST="${SMOKE_MANIFEST:-$SMOKE_TMP_ROOT/ctc_kenlm_smoke_${SMOKE_DECODERS}_${LSB_JOBID:-manual}.yaml}"

source "$DANISH_ASR_PROJECT_DIR/scripts/hpc/env.sh"
setup_omniasr
cd "$DANISH_ASR_PROJECT_DIR"

mkdir -p "$(dirname "$SMOKE_MANIFEST")"

SMOKE_MODEL_LABEL="$SMOKE_MODEL_LABEL" \
SMOKE_SPLIT_LABEL="$SMOKE_SPLIT_LABEL" \
SMOKE_DATASET_SPLIT="$SMOKE_DATASET_SPLIT" \
BASE_MANIFEST="$BASE_MANIFEST" \
SMOKE_MANIFEST="$SMOKE_MANIFEST" \
python - <<'PY'
import os
from pathlib import Path

import yaml

base_manifest = Path(os.environ["BASE_MANIFEST"])
out_manifest = Path(os.environ["SMOKE_MANIFEST"])
model_label = os.environ["SMOKE_MODEL_LABEL"]
split_label = os.environ["SMOKE_SPLIT_LABEL"]
dataset_split = os.environ["SMOKE_DATASET_SPLIT"]

manifest = yaml.safe_load(base_manifest.read_text(encoding="utf-8"))
models = [model for model in manifest["models"] if model["label"] == model_label]
if not models:
    labels = ", ".join(model["label"] for model in manifest["models"])
    raise SystemExit(f"model label {model_label!r} not found in {base_manifest}; available: {labels}")

manifest["models"] = models
manifest["my_method"]["dataset_splits"] = [{"label": split_label, "split": dataset_split}]
out_manifest.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
print(out_manifest)
PY

export CTC_KENLM_MANIFEST="$SMOKE_MANIFEST"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/work3/$USER/outputs/ctc_kenlm_smoke}"
export MAX_SAMPLES="${MAX_SAMPLES:-2}"
export DECODERS="$SMOKE_DECODERS"
export OVERWRITE="${OVERWRITE:-true}"

echo "=== CTC + KenLM smoke test ==="
echo "Base manifest:   $BASE_MANIFEST"
echo "Smoke manifest:  $SMOKE_MANIFEST"
echo "Model label:     $SMOKE_MODEL_LABEL"
echo "Split label:     $SMOKE_SPLIT_LABEL"
echo "Dataset split:   $SMOKE_DATASET_SPLIT"
echo "Decoder:         $DECODERS"
echo "Output root:     $OUTPUT_ROOT"
echo "Max samples:     $MAX_SAMPLES"
echo "Overwrite:       $OVERWRITE"
echo "=============================="
echo "=== Smoke manifest contents ==="
sed -n '1,220p' "$SMOKE_MANIFEST"
echo "==============================="

bash "$DANISH_ASR_PROJECT_DIR/scripts/hpc/benchmark_ctc_kenlm_my_method.sh"
