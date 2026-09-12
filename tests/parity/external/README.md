# Required ONNX Runtime tiny-model oracle

This directory contains a network-free correctness gate for the inference
runtime. Each case authors one tiny ONNX graph, runs that exact source graph in
the pinned ONNX Runtime CPU provider, lowers the same source through
`tools/export_safetensors.py`, and compares full outputs from strict inference
WASM and native CPU execution directly with ORT. The same cases can be promoted
to strict WebGPU and native GPU candidates without changing their oracle data.

The gate is fail-closed. Missing dependencies or binaries, a provider or route
fallback, a changed lowered-op inventory, missing output, dtype/shape/byte-count
mismatch, non-finite float, or numerical mismatch fails the run. Integer and
quantized outputs compare exact bytes. Floating outputs use the case-specific
tolerances in `cases.json`; compact signatures and sampled comparisons are not
used.

Install the exact development dependencies deliberately (the runner never
downloads anything), build the inference artifacts, and run:

```sh
python3 -m pip install -r tests/parity/external/requirements.txt
npm run build
make build_wasm build_native
python3 tests/parity/external/onnx_oracle.py
```

Use `--case ID` to select cases. `--work-dir NEW_DIRECTORY` preserves all ONNX,
package, raw-output, exporter-report, and native lifecycle artifacts for an
audit. By default a temporary directory is removed after the run. `--report`
writes the final JSON summary.

For a device job, repeat `--native-backend` and/or pass `--webgpu`. Every
candidate remains a required backend with operator fallback forbidden. A
physical WebGPU job must also pass `--require-physical-webgpu`, which rejects a
missing identity and known software adapters. Physical native jobs likewise
pass `--require-physical-native`; Vulkan, OpenGL, and Metal must publish one
non-software adapter, while CUDA combines strict successful execution with an
exact selected-device index/name cross-check against `nvidia-smi`. Set
`VOLVOXAI_PARITY_WEBGPU_ADAPTER` or `VOLVOXAI_PARITY_GPU_ADAPTER` to a required
case-insensitive identity substring on dedicated runners. For example:

```sh
python3 tests/parity/external/onnx_oracle.py \
  --native-backend vulkan --native-backend cuda \
  --require-physical-native --webgpu --require-physical-webgpu
```

## Initial seam coverage

The initial cases cover both declared min/max fixtures for bounded dynamic
broadcasting, exact Erf-GELU recognition, LayerNorm plus affine projection,
ONNX NCHW to Volvox NHWC
convolution lowering, I32-indexed Gather, explicit ONNX I64 ArgMax to Volvox
I32 Cast, exact U8 QuantizeLinear output, and an odd-sequence explicit
QK-softmax-V attention decomposition with a non-causal hard mask. Every case
asserts the exact lowered `opType` sequence before either Volvox runtime
executes it.

The dynamic-min WebGPU case is also the lifecycle gate for one compiled model
and one execution context. It executes min, then max, rejects one shape above
the declared maximum with typed `INVALID_ARGUMENT`, executes changed min data,
and finally proves that the first result remains readable and byte-stable after
the context closes. The same-context max and changed-min outputs are each
compared with ORT; distinct shape signatures and changed output bytes exclude a
stale specialization replay.

`tests/contracts/oracle_coverage.json` is the registry-derived strategy
inventory, not a claim that every listed operator has run against ORT. Its
checker joins this case manifest to the generated portable intersection and
reports executable case/operator counts separately. An operator may move from
`oracle-unsupported` to an ONNX route only when that route faithfully preserves
the VolvoxAI contract; merely spelling a dequantize/float/quantize graph is not
enough for project-specific byte rounding.

This is deliberately not a claim that ONNX Runtime defines every VolvoxAI
operator. Project-specific quantized fusions and rounding boundaries,
MoE/scans, bounded-capacity NMS, decode/KV state, and the complete symbolic
shape failure contract are not faithfully representable by these ONNX graphs.
Those require portable-C canonical vectors, invariant tests, and negative
shape/domain tests. ORT is the independent numerical oracle only for the
source-ONNX semantics declared by this case manifest.

`quantized_oracle.py` supplies that separate project-specific seam. Its fixed
QLinear vectors cover asymmetric input/weight/output zero points, odd K tails,
ties-to-even, and both saturation limits. WASM and native CPU are always
checked byte-for-byte; the same `--native-backend` and WebGPU flags add strict
device candidates. Two additional `[rows=2,K=33,N=36]` mixed I8/U8 fixtures
cross every tiled GPU admission threshold while retaining an odd K tail. Their
full output is generated with the portable-C accumulator and two-step F32
requantization contract and compared byte-for-byte. WebGPU jobs require strict
successful execution on a physical adapter with public route evidence showing a
quantized WebGPU selection and zero fallback; native Vulkan/OpenGL/Metal/CUDA
reports record the same strict execution plus the deterministic tiled/warp
predicate proof available through the current public runner. This is
intentionally identified as portable-C canonical evidence, not as an ONNX
Runtime result.
