#!/usr/bin/env bash
# Regenerate the TinyStories-1M example package from its public checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-python3}"
MODEL_ID="${TINYSTORIES_ID:-roneneldan/TinyStories-1M}"
OUTPUT_DIR="${TINYSTORIES_OUTPUT_DIR:-$ROOT/models/tinystories_1m}"
EXPORT_MODEL="$ROOT/examples/tinystories/tools/export_gptneo_safetensors.py"
EXPORT_TOKENIZER="$ROOT/examples/tinystories/tools/export_tokenizer.py"

command -v "$PY" >/dev/null 2>&1 || {
  echo "Error: Python interpreter '$PY' not found." >&2
  exit 1
}

if ! "$PY" - <<'PYEOF'
import importlib.util
import sys

required = ("numpy", "safetensors", "torch", "transformers")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("missing modules: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PYEOF
then
  echo "Missing Python dependencies. Install with: make models_deps" >&2
  exit 1
fi

echo "[tinystories] export $MODEL_ID -> $OUTPUT_DIR"
"$PY" "$EXPORT_MODEL" --model "$MODEL_ID" --out "$OUTPUT_DIR/model.safetensors"
"$PY" "$EXPORT_TOKENIZER" "$MODEL_ID" "$OUTPUT_DIR"
echo "[tinystories] ready under $OUTPUT_DIR"
