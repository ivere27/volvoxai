#!/usr/bin/env bash
# EfficientDet GPU consensus gate: webgpu (Deno) vs native OpenGL vs native Vulkan.
# All are GPU compute of the same graph, so they must agree — bit-identical for int8
# (true-int8 requant is deterministic on GPU) and to ~1e-6 for fp32. This is the
# correct gate for the *true-int8 GPU* tier, which can't be gated against the
# int8-folded-to-fp32 CPU oracle (see webgpu_deno.js / TODO #2). Run on a GPU box.
#
#   [DENO=~/deno NODE=~/node/bin/node] bash tests/parity/gpu_consensus_check.sh
set -euo pipefail
cd "$(dirname "$0")/../.."            # repo root
DENO=${DENO:-deno}
NODE=${NODE:-node}
BIN=native/volvoxai
OUT=tests/parity/out/gpu_consensus
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "ERROR: no $BIN (build the requested native GPU tiers first)" >&2; exit 2; }

for model in efficientdet_lite0_fp32 efficientdet_lite0_int8; do
  mkdir -p "$OUT/$model"
  rm -f -- \
    "$OUT/$model/wg_scores.f32" "$OUT/$model/wg_boxes.f32" \
    "$OUT/$model/gl_scores.f32" "$OUT/$model/gl_boxes.f32" \
    "$OUT/$model/vk_scores.f32" "$OUT/$model/vk_boxes.f32" \
    "$OUT/$model/webgpu_adapter.json" "$OUT/$model/opengl_adapter.json" \
    "$OUT/$model/vulkan_adapter.json"
done
rm -f -- "$OUT/in.u8" "$OUT/in.f32" "$OUT/summary.json"

# Materialize the policy seeded inputs to files so all three backends read identical bytes.
"$NODE" -e '
import("./tests/parity/lib/tensorio.mjs").then((m)=>{
  const fs=require("fs");
  const u8=m.seededUint8(1*320*320*3,1234);
  const f32=m.seededFloat32(1*320*320*3,1234,0,1);
  for (const [file, value] of [["in.u8",u8],["in.f32",f32]]) {
    const final=`tests/parity/out/gpu_consensus/${file}`;
    const temporary=`${final}.${process.pid}.tmp`;
    try {
      fs.writeFileSync(temporary,Buffer.from(value.buffer,value.byteOffset,value.byteLength),{flag:"wx"});
      fs.renameSync(temporary,final);
    } catch (error) { fs.rmSync(temporary,{force:true}); throw error; }
  }
});'

fail=0
cleanup_stage () {
  local stage="$1"
  rm -f -- \
    "$stage/wg_scores.f32" "$stage/wg_boxes.f32" \
    "$stage/gl_scores.f32" "$stage/gl_boxes.f32" \
    "$stage/vk_scores.f32" "$stage/vk_boxes.f32" \
    "$stage/webgpu_adapter.json" "$stage/opengl_adapter.json" "$stage/vulkan_adapter.json" \
    "$stage/webgpu.log" \
    "$stage/opengl.log" "$stage/vulkan.log"
  rmdir "$stage" 2>/dev/null || true
}

write_native_adapter_evidence () {
  local log="$1" expected="$2" destination="$3"
  "$NODE" --input-type=module -e '
    import fs from "node:fs";
    import { requirePhysicalNativeGpuLog } from "./tests/parity/lib/backend.mjs";
    const text=fs.readFileSync(process.argv[1],"utf8");
    const identity=requirePhysicalNativeGpuLog(text,process.argv[2],`GPU consensus ${process.argv[2]}`);
    fs.writeFileSync(process.argv[3],`${JSON.stringify(identity)}\n`,{flag:"wx"});
  ' "$log" "$expected" "$destination"
}

run_model () {              # $1=model  $2=dtype(u8|f32)  $3=tolerance
  local model=$1 dt=$2 tol=$3
  local infile="$OUT/in.$dt"
  local final_dir="$OUT/$model"
  local stage
  stage=$(mktemp -d "$final_dir/.stage.XXXXXX")
  echo "== $model =="

  # Each producer writes only into this run's staging directory. The WebGPU
  # helper attests the selected WebGPU backend; native stdout proves the selected route.
  if ! "$DENO" run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi \
    tests/parity/webgpu_efficientdet_deno.js "$model" "$infile" "$dt" "$stage" >"$stage/webgpu.log" 2>&1; then
    cat "$stage/webgpu.log" >&2
    cleanup_stage "$stage"
    return 1
  fi
  # Native OpenGL + Vulkan
  if ! "$BIN" run "models/$model" --input "input0=$infile" \
    --output "scores=$stage/gl_scores.f32" --output "boxes=$stage/gl_boxes.f32" \
    --opengl >"$stage/opengl.log" 2>&1; then
    cat "$stage/opengl.log" >&2
    cleanup_stage "$stage"
    return 1
  fi
  if ! write_native_adapter_evidence "$stage/opengl.log" opengl "$stage/opengl_adapter.json"; then
    cat "$stage/opengl.log" >&2
    cleanup_stage "$stage"
    return 1
  fi
  if ! "$BIN" run "models/$model" --input "input0=$infile" \
    --output "scores=$stage/vk_scores.f32" --output "boxes=$stage/vk_boxes.f32" \
    --vulkan >"$stage/vulkan.log" 2>&1; then
    cat "$stage/vulkan.log" >&2
    cleanup_stage "$stage"
    return 1
  fi
  if ! write_native_adapter_evidence "$stage/vulkan.log" vulkan "$stage/vulkan_adapter.json"; then
    cat "$stage/vulkan.log" >&2
    cleanup_stage "$stage"
    return 1
  fi

  # Require the exact logical shapes and finite values before comparison. At
  # tolerance zero compare raw F32 bytes, including signed-zero bit patterns.
  if ! O="$stage" "$NODE" -e '
    const fs=require("fs"), O=process.env.O, tol=Number(process.argv[1]);
    if(!Number.isFinite(tol)||tol<0) throw new Error(`invalid tolerance ${process.argv[1]}`);
    const expected={scores:19206*90,boxes:19206*4};
    const load=(p,n)=>{
      const bytes=fs.readFileSync(p);
      if(bytes.byteLength!==n*4) throw new Error(`${p}: ${bytes.byteLength} bytes, expected ${n*4}`);
      const values=new Float32Array(bytes.buffer,bytes.byteOffset,n);
      for(let i=0;i<values.length;i++) if(!Number.isFinite(values[i])) throw new Error(`${p}: non-finite at ${i}`);
      return {bytes,values};
    };
    const md=(a,b)=>{let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]-b[i]));return m;};
    let bad=0;
    for (const t of ["scores","boxes"]) {
      const wg=load(`${O}/wg_${t}.f32`,expected[t]);
      const gl=load(`${O}/gl_${t}.f32`,expected[t]);
      const vk=load(`${O}/vk_${t}.f32`,expected[t]);
      const wgGl=md(wg.values,gl.values), wgVk=md(wg.values,vk.values), glVk=md(gl.values,vk.values);
      const ok = tol===0
        ? wg.bytes.equals(gl.bytes) && wg.bytes.equals(vk.bytes)
        : wgGl<=tol && wgVk<=tol && glVk<=tol;
      console.log(`  ${t}: webgpu-gl=${wgGl.toExponential(2)} webgpu-vk=${wgVk.toExponential(2)} gl-vk=${glVk.toExponential(2)}  ${ok?"PASS":"FAIL"}`);
      if(!ok) bad++;
    }
    if(bad) process.exit(1);
  ' "$tol"; then
    cleanup_stage "$stage"
    return 1
  fi

  local file
  for file in wg_scores.f32 wg_boxes.f32 gl_scores.f32 gl_boxes.f32 vk_scores.f32 vk_boxes.f32 \
    webgpu_adapter.json opengl_adapter.json vulkan_adapter.json; do
    if ! mv -f -- "$stage/$file" "$final_dir/$file"; then
      cleanup_stage "$stage"
      return 1
    fi
  done
  rm -f -- "$stage/webgpu.log" "$stage/opengl.log" "$stage/vulkan.log"
  if ! rmdir "$stage"; then return 1; fi
  echo "  route evidence: WebGPU/OpenGL/Vulkan selected; native routes may still use documented per-node CPU fallback"
}

if ! run_model efficientdet_lite0_fp32 f32 1e-5; then fail=1; fi
if ! run_model efficientdet_lite0_int8 u8 0; then fail=1; fi # int8: exact bit-consensus

echo ""
if [ "$fail" -ne 0 ]; then
  echo "GPU_CONSENSUS_FAIL"
  exit 1
fi

# Seal the exact successful output bytes and every physical adapter identity into
# one small uploadable summary. Full tensors stay local to the runner.
"$NODE" --input-type=module <<'NODE'
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import {
  GPU_CONSENSUS_MODELS,
  gpuConsensusFingerprint,
} from './tests/parity/lib/gpu_campaign.mjs';

const root = 'tests/parity/out/gpu_consensus';
const digest = (file) => {
  const bytes = fs.readFileSync(file);
  return { size: bytes.byteLength, sha256: createHash('sha256').update(bytes).digest('hex') };
};
const models = [];
for (const { model, tolerance } of GPU_CONSENSUS_MODELS) {
  const dir = path.join(root, model);
  const outputs = [];
  for (const [backend, prefix] of [['webgpu', 'wg'], ['native-opengl', 'gl'], ['native-vulkan', 'vk']]) {
    for (const output of ['scores', 'boxes']) {
      const relative = `${model}/${prefix}_${output}.f32`;
      outputs.push({ backend, output, path: relative, ...digest(path.join(root, relative)) });
    }
  }
  models.push({
    model,
    tolerance,
    adapters: {
      webgpu: JSON.parse(fs.readFileSync(path.join(dir, 'webgpu_adapter.json'), 'utf8')),
      'native-opengl': JSON.parse(fs.readFileSync(path.join(dir, 'opengl_adapter.json'), 'utf8')),
      'native-vulkan': JSON.parse(fs.readFileSync(path.join(dir, 'vulkan_adapter.json'), 'utf8')),
    },
    outputs,
  });
}
const summary = {
  schema: 'volvoxai.gpu-consensus-summary',
  version: 3,
  passed: true,
  generatedAt: new Date().toISOString(),
  fingerprint: gpuConsensusFingerprint(),
  models,
};
const destination = path.join(root, 'summary.json');
const temporary = `${destination}.${process.pid}.tmp`;
try {
  fs.writeFileSync(temporary, `${JSON.stringify(summary, null, 2)}\n`, { flag: 'wx' });
  fs.renameSync(temporary, destination);
} catch (error) {
  fs.rmSync(temporary, { force: true });
  throw error;
}
NODE

echo "GPU_CONSENSUS_OK"
