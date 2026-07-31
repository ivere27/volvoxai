#!/bin/bash
# Compile the wasm32 inference artifacts from the current sources.
#
# kernels.c amalgamates qbatch_matmul_wasm_simd.c only under __wasm_simd128__,
# and qlinear_w8a8_wasm_relaxed.c is a separate freestanding artifact, so neither
# is reachable from any CMake target. A rename or a new header include in the
# shared kernels can therefore break the web build while every native test still
# passes -- which is exactly what happened once. This check closes that gap
# without needing the release Docker image.
#
# Skips (exit 0) when clang cannot target wasm32, so it never blocks a host that
# simply lacks the target.
set -eu

CLANG="${CLANG:-clang}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if ! printf 'int f(int x){return x+1;}\n' > "$work/probe.c" ||
   ! "$CLANG" --target=wasm32 -nostdlib -O0 -c "$work/probe.c" \
        -o "$work/probe.o" 2>/dev/null; then
  echo "verify_wasm_build: $CLANG cannot target wasm32; skipping."
  exit 0
fi

root="${1:?usage: verify_wasm_build.sh <repo-root>}"

# Baseline SIMD128 inference sidecar: the amalgamated kernel set.
"$CLANG" --target=wasm32 -O3 -msimd128 -nostdlib \
  -Wl,--no-entry -Wl,--export-all -Wl,--allow-undefined \
  -o "$work/volvoxai.wasm" "$root/native/src/kernels/kernels.c"

# Optional relaxed-SIMD child, built with the release warning contract.
"$CLANG" --target=wasm32 -std=c11 -O3 -Wall -Wextra -Werror \
  -msimd128 -mrelaxed-simd -nostdlib -ffreestanding \
  -Wl,--no-entry -Wl,--import-memory -Wl,--strip-all \
  -o "$work/relaxed.wasm" \
  "$root/native/src/kernels/qlinear_w8a8_wasm_relaxed.c"

for artifact in volvoxai.wasm relaxed.wasm; do
  if [ ! -s "$work/$artifact" ]; then
    echo "verify_wasm_build: $artifact was not produced."
    exit 1
  fi
done
echo "verify_wasm_build: wasm32 artifacts compiled."
