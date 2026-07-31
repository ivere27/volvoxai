#!/bin/bash
# Static ISA codegen checks on target-attributed W8A8 kernels. These prove the
# per-function specializations still emit the expected integer dot instruction
# and public dispatchers stay baseline,
# even where the local CPU can't run them. Uses plain `clang` plus objdump so
# the checks do not require executing target code.
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

emit_maddubs() {  # emit_maddubs <src> <includedir>
  local src="$1" inc="$2"
  local obj; obj="$(mktemp --suffix=.o)"
  "$CLANG" -O3 -I"$inc" -c "$src" -o "$obj"
  if ! objdump -d "$obj" | grep -Eqi 'vpmaddubsw.*%ymm'; then
    echo "Error: $(basename "$src") did not emit YMM VPMADDUBSW."
    rm -f "$obj"; exit 1
  fi
  rm -f "$obj"
  echo "Verified YMM VPMADDUBSW specialization in $(basename "$src")."
}

emit_signed_maddubs() {  # emit_signed_maddubs <src> <includedir>
  local src="$1" inc="$2"
  local obj; obj="$(mktemp --suffix=.o)"
  "$CLANG" -O3 -I"$inc" -c "$src" -o "$obj"
  for instruction in vpabsb vpsignb vpmaddubsw; do
    if ! objdump -d "$obj" | grep -Eqi "${instruction}.*%ymm"; then
      echo "Error: $(basename "$src") did not emit YMM ${instruction^^}."
      rm -f "$obj"; exit 1
    fi
  done
  rm -f "$obj"
  echo "Verified exact signed-absolute YMM VPMADDUBSW specialization in $(basename "$src")."
}

case "$mode" in
  emit-ymm) emit ymm "$2" "$3" ;;
  emit-zmm) emit zmm "$2" "$3" ;;
  emit-maddubs) emit_maddubs "$2" "$3" ;;
  emit-signed-maddubs) emit_signed_maddubs "$2" "$3" ;;
  baseline)  # baseline <kerneldir>: dispatchers must stay YMM/ZMM-free
    inc="$2"
    ql="$(mktemp --suffix=.o)"; qc="$(mktemp --suffix=.o)"
    qb="$(mktemp --suffix=.o)"; qg="$(mktemp --suffix=.o)"
    qn="$(mktemp --suffix=.o)"; qgn="$(mktemp --suffix=.o)"
    "$CLANG" -O3 -I"$inc" -c "$inc/qlinear_w8a8_x86.c" -o "$ql"
    "$CLANG" -O3 -I"$inc" -c "$inc/qconv_w8a8_x86.c" -o "$qc"
    "$CLANG" -O3 -I"$inc" -c "$inc/qbatch_matmul_w8a8_native.c" -o "$qb"
    "$CLANG" -O3 -I"$inc" -c "$inc/packed_quant_gemm.c" -o "$qg"
    "$CLANG" -O3 -I"$inc" -c "$inc/qnorm_activation_w8a8_native.c" -o "$qn"
    "$CLANG" -O3 -I"$inc" -c "$inc/qgroupnorm_w8a8_native.c" -o "$qgn"
    rc=0
    objdump -d --disassemble=vx_qlinear_i8u8_native "$ql" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    objdump -d --disassemble=vx_qconv2d_i8u8_native "$qc" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    objdump -d --disassemble=vx_qbatch_matmul_i8u8_native "$qb" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    objdump -d --disassemble=vx_qlinear_i8u8_packed "$qg" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    objdump -d --disassemble=vx_qsilu_i8u8_native_validated "$qn" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    objdump -d --disassemble=vx_qlayernorm_i8u8_native_validated "$qn" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    objdump -d --disassemble=vx_qgroupnorm_i8u8_native_validated "$qgn" | grep -Eqi '%(ymm|zmm)' && rc=1 || true
    rm -f "$ql" "$qc" "$qb" "$qg" "$qn" "$qgn"
    if [ "$rc" -ne 0 ]; then
      echo "Error: a baseline W8A8 dispatcher contains YMM/ZMM instructions."; exit 1
    fi
    echo "Verified baseline W8A8 dispatchers contain no YMM/ZMM instructions."
    ;;
  *) echo "usage: verify_codegen.sh {emit-ymm|emit-zmm|emit-maddubs|emit-signed-maddubs <src> <inc>|baseline <inc>}"; exit 2 ;;
esac
