#!/bin/bash
# Static ISA codegen checks on target-attributed W8A8 kernels. These prove the
# per-function specializations still emit the expected VPDPBUSD / stay baseline,
# even where the local CPU can't run them. Ports verify_native_* from the old
# Makefile. Uses plain `clang` (matching the historical recipes) + objdump.
# NOTE: no `pipefail` on purpose. `grep -q` short-circuits and SIGPIPEs
# objdump; with pipefail the pipeline would report objdump's failure even on a
# match, exactly the false negative we must avoid. grep's exit alone decides.
set -eu

CLANG="${CLANG:-clang}"
mode="$1"

emit() {  # emit <reg> <src> <includedir>
  local reg="$1" src="$2" inc="$3"
  local obj; obj="$(mktemp --suffix=.o)"
  "$CLANG" -O3 -I"$inc" -c "$src" -o "$obj"
  if ! objdump -d "$obj" | grep -Eqi "vpdpbusd.*%${reg}"; then
    echo "Error: $(basename "$src") did not emit ${reg^^} VPDPBUSD."
    rm -f "$obj"; exit 1
  fi
  rm -f "$obj"
  echo "Verified ${reg^^} VPDPBUSD specialization in $(basename "$src")."
}

case "$mode" in
  emit-ymm) emit ymm "$2" "$3" ;;
  emit-zmm) emit zmm "$2" "$3" ;;
  baseline)  # baseline <kerneldir>: dispatchers must stay YMM/ZMM-free
    inc="$2"
    ql="$(mktemp --suffix=.o)"; qc="$(mktemp --suffix=.o)"
    "$CLANG" -O3 -I"$inc" -c "$inc/qlinear_w8a8_x86.c" -o "$ql"
    "$CLANG" -O3 -I"$inc" -c "$inc/qconv_w8a8_x86.c" -o "$qc"
    rc=0
    objdump -d --disassemble=vx_qlinear_i8u8_native "$ql" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    objdump -d --disassemble=vx_qconv2d_i8u8_native "$qc" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    rm -f "$ql" "$qc"
    if [ "$rc" -ne 0 ]; then
      echo "Error: a baseline W8A8 dispatcher contains YMM/ZMM instructions."; exit 1
    fi
    echo "Verified baseline W8A8 dispatchers contain no YMM/ZMM instructions."
    ;;
  *) echo "usage: verify_codegen.sh {emit-ymm|emit-zmm <src> <inc>|baseline <inc>}"; exit 2 ;;
esac
