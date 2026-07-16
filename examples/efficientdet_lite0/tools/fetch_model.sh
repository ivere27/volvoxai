#!/usr/bin/env bash
# Download MediaPipe EfficientDet-Lite0 and export VolvoxAI example packages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-python3}"
EXPORTER="$ROOT/tools/export_safetensors.py"
LABELS="$ROOT/examples/efficientdet_lite0/assets/coco_labels.txt"
MODELS="$ROOT/models"
MEDIAPIPE_BASE="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0"

log() { printf '\033[1;36m[efficientdet-lite0]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[efficientdet-lite0] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

require_python() {
  command -v "$PY" >/dev/null 2>&1 || die "Python interpreter '$PY' not found."
  "$PY" - "$@" <<'PYEOF' || die "Missing Python deps. Install: $PY -m pip install -r tools/requirements-export.txt"
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print("missing modules:", ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PYEOF
}

download() {
  local url="$1" destination="$2"
  mkdir -p "$(dirname "$destination")"
  log "download $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$destination" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$destination" "$url"
  else
    die "Need curl or wget to download models."
  fi
}

export_one() {
  local precision="$1" directory_suffix="$2" tflite_name="$3" weight_dtype="$4" normalization="$5"
  local directory="$MODELS/efficientdet_lite0_${directory_suffix}"
  local tflite="$directory/$tflite_name"

  download "$MEDIAPIPE_BASE/$precision/latest/efficientdet_lite0.tflite" "$tflite"
  log "export ($precision) -> $directory/config.json + model.safetensors"
  "$PY" "$EXPORTER" \
    --model "$tflite" \
    --out "$directory/model.safetensors" \
    --weight-dtype "$weight_dtype" \
    --image-normalization "input0=$normalization" \
    --output-name scores \
    --output-name boxes
  cp "$LABELS" "$directory/labels.txt"
  log "labels -> $directory/labels.txt"
}

main() {
  [ -f "$EXPORTER" ] || die "Missing generic exporter: $EXPORTER"
  [ -f "$LABELS" ] || die "Missing labels asset: $LABELS"
  require_python numpy flatbuffers safetensors torch

  local only="${ONLY:-}"
  case "$only" in
    ""|int8|float16|float32) ;;
    *) die "Unknown ONLY value '$only' (use: int8 | float16 | float32)" ;;
  esac

  if [ -z "$only" ] || [ "$only" = "int8" ]; then
    export_one int8 int8 efficientdet_lite0.tflite auto raw-255
  fi
  if [ -z "$only" ] || [ "$only" = "float16" ]; then
    export_one float16 fp16 efficientdet_lite0_float16.tflite float16 zero-one
  fi
  if [ -z "$only" ] || [ "$only" = "float32" ]; then
    export_one float32 fp32 efficientdet_lite0_float32.tflite float32 zero-one
  fi
  log "ready under $MODELS/efficientdet_lite0_*"
}

main "$@"
