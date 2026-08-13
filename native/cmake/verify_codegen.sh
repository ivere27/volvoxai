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

emit_packed_vnni() {  # emit_packed_vnni <src> <includedir>
  local src="$1" inc="$2"
  local obj avx avx512; obj="$(mktemp --suffix=.o)"
  "$CLANG" -O3 -I"$inc" -c "$src" -o "$obj"
  avx="$(objdump -d \
    --disassemble=vx_qgemm_w8a8_avx2_pair_signed_n32_avxvnni "$obj")"
  avx512="$(objdump -d \
    --disassemble=vx_qgemm_w8a8_avx2_pair_signed_n32_avx512vnni "$obj")"
  if ! echo "$avx" | grep -Eqi \
      ':[[:space:]]+c4([[:space:]][[:xdigit:]]{2})+.*vpdpbusd.*%ymm'; then
    echo "Error: $(basename "$src") did not emit VEX YMM VPDPBUSD in its packed AVX-VNNI specialization."
    rm -f "$obj"; exit 1
  fi
  if ! echo "$avx512" | grep -Eqi \
      ':[[:space:]]+62([[:space:]][[:xdigit:]]{2})+.*vpdpbusd.*%ymm'; then
    echo "Error: $(basename "$src") did not emit EVEX YMM VPDPBUSD in its packed AVX-512-VNNI specialization."
    rm -f "$obj"; exit 1
  fi
  rm -f "$obj"
  echo "Verified packed VEX and EVEX YMM VPDPBUSD specializations in $(basename "$src")."
}

emit_qconv_im2col() {  # emit_qconv_im2col <src> <includedir>
  local src="$1" inc="$2"
  local obj symbols body; obj="$(mktemp --suffix=.o)"
  "$CLANG" -O3 -I"$inc" -c "$src" -o "$obj"
  symbols="vx_w8a8_qconv_im2col_worker_plain
vx_w8a8_qconv_im2col_worker_remap
vx_w8a8_qconv_im2col_3x3_rows_worker_plain
vx_w8a8_qconv_im2col_3x3_rows_worker_remap"
  for symbol in $symbols; do
    body="$(objdump -d --disassemble="$symbol" "$obj")"
    if ! echo "$body" | grep -Fq "<$symbol>:"; then
      echo "Error: $(basename "$src") did not emit specialized im2col worker $symbol."
      rm -f "$obj"; exit 1
    fi
  done
  if objdump -t "$obj" | grep -Eq \
      'vx_w8a8_qconv_im2col_(copy|fill|worker_body|3x3_rows_worker_body)'; then
    echo "Error: $(basename "$src") left an out-of-line im2col copy/fill/body helper."
    rm -f "$obj"; exit 1
  fi
  body="$(objdump -d \
    --disassemble=vx_w8a8_qconv_im2col_worker_remap "$obj";
    objdump -d \
    --disassemble=vx_w8a8_qconv_im2col_3x3_rows_worker_remap "$obj")"
  if ! echo "$body" | grep -Eqi \
      'v[^[:space:]]*xor[^[:space:]]*[[:space:]].*%ymm'; then
    echo "Error: $(basename "$src") did not inline a YMM xor into its remap workers."
    rm -f "$obj"; exit 1
  fi
  rm -f "$obj"
  echo "Verified activation-domain-specialized AVX2 im2col workers in $(basename "$src")."
}

case "$mode" in
  emit-ymm) emit ymm "$2" "$3" ;;
  emit-zmm) emit zmm "$2" "$3" ;;
  emit-maddubs) emit_maddubs "$2" "$3" ;;
  emit-signed-maddubs) emit_signed_maddubs "$2" "$3" ;;
  emit-packed-vnni) emit_packed_vnni "$2" "$3" ;;
  emit-qconv-im2col) emit_qconv_im2col "$2" "$3" ;;
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
  *) echo "usage: verify_codegen.sh {emit-ymm|emit-zmm|emit-maddubs|emit-signed-maddubs|emit-packed-vnni|emit-qconv-im2col <src> <inc>|baseline <inc>}"; exit 2 ;;
esac
