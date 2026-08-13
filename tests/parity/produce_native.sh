#!/usr/bin/env bash
# Native producer for the cross-tier parity harness.
#
#   tests/parity/produce_native.sh <tier>
#
# <tier> is a label recorded in the report and used to pick the backend flag:
#   native-cpu      (--cpu)              — the always-runnable gate tier
#   native-vulkan   (--vulkan)           — physical capability probe
#   native-opengl   (--opengl)           — physical capability probe
#
# Runs the fixed native binary `native/volvoxai run` on each model's fixed inputs.
# CPU publishes raw tensors; native GPU tiers publish policy-bound capability
# evidence when strict compilation reaches an exact expected rejection.
# `run.mjs native-sig` validates and imports either result.
set -euo pipefail
cd "$(dirname "$0")/../.."

TIER="${1:-native-cpu}"
NODE="${NODE:-node}"
BIN="native/volvoxai"
OUT="tests/parity/out/native/${TIER}"
FIX="tests/parity/out/fixtures"
FLAG_ARGS=()
CAPABILITY_EXPECTED=0

case "$TIER" in
  native-cpu)    EXPECTED_BACKEND="cpu"; FLAG_ARGS=(--cpu) ;;
  native-vulkan) EXPECTED_BACKEND="vulkan"; FLAG_ARGS=(--vulkan); CAPABILITY_EXPECTED=1 ;;
  native-opengl) EXPECTED_BACKEND="opengl"; FLAG_ARGS=(--opengl); CAPABILITY_EXPECTED=1 ;;
  *) echo "unknown tier: $TIER" >&2; exit 2 ;;
esac

# Remove only this producer's exact prior outputs before any preflight can fail.
# This keeps an ignored producer error from making native-sig consume an older run.
rm -f -- \
  "$OUT/tinystories_1m/logits.f32" "$OUT/tinystories_1m/ms.txt" "$OUT/tinystories_1m/backend.txt" \
  "$OUT/tinystories_1m/adapter.json" "$OUT/tinystories_1m/runtime-evidence.json" \
  "$OUT/tinystories_1m/runtime-failure-evidence.json" "$OUT/tinystories_1m/native.log" "$OUT/tinystories_1m/capability-skip.json" \
  "$OUT/efficientdet_lite0_int8/scores.f32" "$OUT/efficientdet_lite0_int8/boxes.f32" "$OUT/efficientdet_lite0_int8/ms.txt" "$OUT/efficientdet_lite0_int8/backend.txt" "$OUT/efficientdet_lite0_int8/adapter.json" "$OUT/efficientdet_lite0_int8/runtime-evidence.json" \
  "$OUT/efficientdet_lite0_int8/runtime-failure-evidence.json" "$OUT/efficientdet_lite0_int8/native.log" "$OUT/efficientdet_lite0_int8/capability-skip.json" \
  "$OUT/efficientdet_lite0_fp32/scores.f32" "$OUT/efficientdet_lite0_fp32/boxes.f32" "$OUT/efficientdet_lite0_fp32/ms.txt" "$OUT/efficientdet_lite0_fp32/backend.txt" "$OUT/efficientdet_lite0_fp32/adapter.json" "$OUT/efficientdet_lite0_fp32/runtime-evidence.json" \
  "$OUT/efficientdet_lite0_fp32/runtime-failure-evidence.json" "$OUT/efficientdet_lite0_fp32/native.log" "$OUT/efficientdet_lite0_fp32/capability-skip.json" \
  "tests/parity/out/manifests/whole-${TIER}.json" \
  "tests/parity/out/tinystories_1m.${TIER}.json" \
  "tests/parity/out/efficientdet_lite0_int8.${TIER}.json" \
  "tests/parity/out/efficientdet_lite0_fp32.${TIER}.json"

if [ ! -x "$BIN" ]; then
  echo "native binary $BIN not built (run: make build_native)" >&2
  exit 2
fi

for required in \
  models/tinystories_1m/graph.json models/tinystories_1m/model.safetensors \
  models/tinystories_1m/tokens.i32 models/tinystories_1m/positions.i32 \
  models/efficientdet_lite0_int8/graph.json models/efficientdet_lite0_int8/model.safetensors \
  models/efficientdet_lite0_fp32/graph.json models/efficientdet_lite0_fp32/model.safetensors \
  "${FIX}/efficientdet_lite0_int8/input0.u8" \
  "${FIX}/efficientdet_lite0_fp32/input0.f32"; do
  if [ ! -f "$required" ]; then
    echo "native ${TIER}: required input missing: $required" >&2
    exit 2
  fi
done

expected_elements() {
  case "$1:$2" in
    tinystories_1m:logits) echo 12865792 ;;
    efficientdet_lite0_int8:scores|efficientdet_lite0_fp32:scores) echo 1728540 ;;
    efficientdet_lite0_int8:boxes|efficientdet_lite0_fp32:boxes) echo 76824 ;;
    *) return 1 ;;
  esac
}

cleanup_stage() {
  local stage="$1"
  rm -f -- "$stage/native.log" "$stage/ms.txt" "$stage/adapter.json" "$stage/runtime-evidence.json" \
    "$stage/capability-skip.json" \
    "$stage/logits.f32" "$stage/scores.f32" "$stage/boxes.f32"
  rmdir "$stage" 2>/dev/null || true
}

run_model() {
  local model_dir="$1"; shift
  local out_sub="$1"; shift
  local dir="${OUT}/${out_sub}"
  mkdir -p "$dir"
  local stage
  stage=$(mktemp -d "${dir}/.stage.XXXXXX")
  local -a cli_args=() staged_outputs=() final_outputs=() output_names=()
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--output" ]; then
      if [ "$#" -lt 2 ] || [[ "$2" != *=* ]]; then
        echo "native ${TIER} ${out_sub}: malformed --output" >&2
        cleanup_stage "$stage"
        return 2
      fi
      local output_name="${2%%=*}"
      local final_output="${2#*=}"
      local staged_output="${stage}/${output_name}.f32"
      cli_args+=(--output "${output_name}=${staged_output}")
      output_names+=("$output_name")
      staged_outputs+=("$staged_output")
      final_outputs+=("$final_output")
      shift 2
    else
      cli_args+=("$1")
      shift
    fi
  done
  if [ "${#staged_outputs[@]}" -eq 0 ]; then
    echo "native ${TIER} ${out_sub}: no outputs requested" >&2
    cleanup_stage "$stage"
    return 2
  fi

  local start end cli_status
  start=$(date +%s%N)
  set +e
  "$BIN" run "$model_dir" "${FLAG_ARGS[@]}" "${cli_args[@]}" \
    --report-json "$stage/runtime-evidence.json" >"$stage/native.log" 2>&1
  cli_status=$?
  set -e
  if [ "$CAPABILITY_EXPECTED" -eq 1 ]; then
    if [ "$cli_status" -eq 0 ]; then
      echo "native ${TIER} ${out_sub}: unexpectedly compiled despite its declared capability skip" >&2
      cat "$stage/native.log" >&2
      cleanup_stage "$stage"
      return 1
    fi
    for staged_output in "${staged_outputs[@]}"; do
      if [ -e "$staged_output" ]; then
        echo "native ${TIER} ${out_sub}: rejected compile still published an output" >&2
        cleanup_stage "$stage"
        return 1
      fi
    done
    if ! "$NODE" tests/parity/seal_native_capability.mjs \
        "$TIER" "$out_sub" "$cli_status" "$stage/native.log" \
        "$stage/runtime-evidence.json" "$stage/capability-skip.json"; then
      cat "$stage/native.log" >&2
      cleanup_stage "$stage"
      return 1
    fi
    if [ ! -f "$stage/capability-skip.json" ]; then
      echo "native ${TIER} ${out_sub}: capability sealer did not publish its artifact" >&2
      cleanup_stage "$stage"
      return 1
    fi
    if ! mv -f -- "$stage/runtime-evidence.json" "${dir}/runtime-failure-evidence.json" ||
       ! mv -f -- "$stage/native.log" "${dir}/native.log" ||
       ! mv -f -- "$stage/capability-skip.json" "${dir}/capability-skip.json"; then
      echo "native ${TIER} ${out_sub}: cannot publish capability evidence" >&2
      cleanup_stage "$stage"
      return 1
    fi
    if ! rmdir "$stage"; then
      echo "native ${TIER} ${out_sub}: capability staging directory is not empty" >&2
      cleanup_stage "$stage"
      return 1
    fi
    return 0
  fi
  if [ "$cli_status" -ne 0 ]; then
    cat "$stage/native.log" >&2
    cleanup_stage "$stage"
    return 1
  fi
  end=$(date +%s%N)

  if ! "$NODE" --input-type=module -e '
    import fs from "node:fs";
    import { validateNativeRuntimeEvidence } from "./tests/parity/lib/artifact.mjs";
    import { requirePhysicalNativeGpuLog } from "./tests/parity/lib/backend.mjs";
    const text = fs.readFileSync(process.argv[1], "utf8");
    const routes = [...text.matchAll(/^Backend: ([^\r\n]+)\r?$/gm)].map((match) => match[1]);
    if (routes.length !== 1 || routes[0] !== process.argv[2]) {
      throw new Error(`native CLI reported ${JSON.stringify(routes)}, expected ${process.argv[2]}`);
    }
    if (process.argv[2] !== "cpu") {
      const identity = requirePhysicalNativeGpuLog(text, process.argv[2], `native producer ${process.argv[2]}`);
      fs.writeFileSync(process.argv[3], `${JSON.stringify(identity)}\n`, { flag: "wx" });
    }
    const evidence = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
    validateNativeRuntimeEvidence(evidence, process.argv[2]);
  ' "$stage/native.log" "$EXPECTED_BACKEND" "$stage/adapter.json" "$stage/runtime-evidence.json"; then
    cat "$stage/native.log" >&2
    cleanup_stage "$stage"
    return 1
  fi

  local index expected
  for index in "${!staged_outputs[@]}"; do
    if ! expected=$(expected_elements "$out_sub" "${output_names[$index]}"); then
      echo "native ${TIER} ${out_sub}: no logical size for ${output_names[$index]}" >&2
      cleanup_stage "$stage"
      return 2
    fi
    if ! "$NODE" -e '
      const fs = require("fs");
      const data = fs.readFileSync(process.argv[1]);
      const elements = Number(process.argv[2]);
      if (!Number.isSafeInteger(elements) || elements <= 0 || data.byteLength !== elements * 4) {
        throw new Error(`${process.argv[1]} has ${data.byteLength} bytes; expected ${elements * 4}`);
      }
      const values = new Float32Array(data.buffer, data.byteOffset, elements);
      for (let i = 0; i < values.length; i++) {
        if (!Number.isFinite(values[i])) throw new Error(`${process.argv[1]} is non-finite at index ${i}`);
      }
    ' "${staged_outputs[$index]}" "$expected"; then
      cleanup_stage "$stage"
      return 1
    fi
  done

  if command -v bc >/dev/null 2>&1; then
    echo "scale=2; ($end - $start)/1000000" | bc > "$stage/ms.txt"
  else
    echo $(( (end - start) / 1000000 )) > "$stage/ms.txt"
  fi
  for index in "${!staged_outputs[@]}"; do
    mv -f -- "${staged_outputs[$index]}" "${final_outputs[$index]}"
  done
  mv -f -- "$stage/ms.txt" "${dir}/ms.txt"
  mv -f -- "$stage/runtime-evidence.json" "${dir}/runtime-evidence.json"
  if [ "$EXPECTED_BACKEND" != "cpu" ]; then
    mv -f -- "$stage/adapter.json" "${dir}/adapter.json"
  fi
  rm -f -- "$stage/native.log"
  rmdir "$stage"
  echo "native ${TIER} ${out_sub}: ok"
}

failed=0

# TinyStories — explicit [1,256] maximum-capacity whole-forward oracle. This
# fixture is parity evidence, not the deployment generation path.
if ! run_model models/tinystories_1m tinystories_1m \
  --input 'tokens[1,256]=models/tinystories_1m/tokens.i32' \
  --input 'positions[1,256]=models/tinystories_1m/positions.i32' \
  --output "logits=${OUT}/tinystories_1m/logits.f32"; then
  failed=1
fi

# EfficientDet int8 — needs the seeded input fixture (generate it first, node-only).
if ! run_model models/efficientdet_lite0_int8 efficientdet_lite0_int8 \
  --input "input0=${FIX}/efficientdet_lite0_int8/input0.u8" \
  --output "scores=${OUT}/efficientdet_lite0_int8/scores.f32" \
  --output "boxes=${OUT}/efficientdet_lite0_int8/boxes.f32"; then
  failed=1
fi

# EfficientDet fp32 — seeded float fixture.
if ! run_model models/efficientdet_lite0_fp32 efficientdet_lite0_fp32 \
  --input "input0=${FIX}/efficientdet_lite0_fp32/input0.f32" \
  --output "scores=${OUT}/efficientdet_lite0_fp32/scores.f32" \
  --output "boxes=${OUT}/efficientdet_lite0_fp32/boxes.f32"; then
  failed=1
fi

[ "$failed" -eq 0 ] || exit 1
if [ "$CAPABILITY_EXPECTED" -eq 1 ]; then
  "$NODE" tests/parity/seal_native_capability.mjs verify-tier "$TIER" "$OUT"
  echo "native ${TIER}: physical initialization and exact policy-declared compile rejections verified"
else
  echo "native ${TIER}: strict '${EXPECTED_BACKEND}' tier and no-operator-fallback route verified"
fi
