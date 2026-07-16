#!/usr/bin/env bash
#
# Compatibility dispatcher for example-owned model fetchers.
#
# Usage:
#   tools/fetch_models.sh                 # everything (EfficientDet variants + TinyStories)
#   tools/fetch_models.sh efficientdet    # only the EfficientDet-Lite0 detector (fp32/fp16/int8)
#   tools/fetch_models.sh tinystories     # only the TinyStories-1M language model
#
# New automation should call the fetcher under each example directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
EFFICIENTDET_FETCH="$ROOT/examples/efficientdet_lite0/tools/fetch_model.sh"
TINYSTORIES_FETCH="$ROOT/examples/tinystories/tools/fetch_model.sh"

log()  { printf '\033[1;36m[fetch-models]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fetch-models] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

fetch_efficientdet() {
  [ -f "$EFFICIENTDET_FETCH" ] || die "Missing EfficientDet example fetcher: $EFFICIENTDET_FETCH"
  PY="$PY" bash "$EFFICIENTDET_FETCH"
}

fetch_tinystories() {
  [ -f "$TINYSTORIES_FETCH" ] || die "Missing TinyStories example fetcher: $TINYSTORIES_FETCH"
  PY="$PY" bash "$TINYSTORIES_FETCH"
}

main() {
  local what="${1:-all}"
  case "$what" in
    all)          fetch_efficientdet; fetch_tinystories ;;
    efficientdet) fetch_efficientdet ;;
    tinystories)  fetch_tinystories ;;
    *) die "Unknown target '$what' (use: all | efficientdet | tinystories)" ;;
  esac
  log "Done."
}

main "$@"
