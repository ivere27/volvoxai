# PackedLinear storage groups and execution tiles

A storage group determines which weights share one scale. An execution tile
determines how a backend divides work. Changing a tile must preserve the
stored weights and arithmetic contract. Changing a group size requires fitting
and exporting a new artifact from the floating source.

## Storage and authoring

The public `WeightFormat` and graph `PackedLinear.group_size` admit G=32, 64,
128 and 256. They admit I2/I3/I4/I8 storage and the portable sixteen-element
SIMD dot where that width has a SIMD decoder. G256's largest
dynamic I8 group dot is bounded by `256 * 127 * 128 = 4,161,536`, safely within
INT32 and exactly representable in F32. The finite set bounds authoring scratch
and the validation matrix without compiling a kernel for every group.

G=0 remains per-output-channel I8 for static byte regions and individual weight
helpers, with separate Conv-only I2/I3/I4 static formats. It is not a default
group or a whole-row PackedLinear group. Grouped static activations remain
unsupported. Low-bit Conv supports RTN/MSE/IMPORTANCE independently.
Grouped RTN, MSE_CLIP and IMPORTANCE share the same storage; only IMPORTANCE
requires input-channel observations. Defaults remain G128.

For logical weights `[N,K]`, each row has `ceil(K*bits/32)` packed words and
`ceil(K/G)` scales. Within a word, earlier K positions occupy lower bits.
Codes are offset binary: decode subtracts `2^(bits-1)`. W2/W4/W8 packed
zero words are `0xaaaaaaaa`, `0x88888888` and `0x80808080`. W3 uses a dense
bitstream and codes may cross word boundaries; unused final high bits are zero.
There is no padding between groups. Other widths' unused lanes in the final
word must contain their encoded zero;
each partial final group has its own scale computed only from valid weights.
Scale indexing is `scales[n * ceil(K/G) + k/G]`.

Load/shape validation checks the declared G, exact physical shapes, positive
finite scales and tail codes. Shared-weight export reuse includes group size
in its identity; rules choosing different G cannot share a packed conversion.
Shape validation catches many mislabeled artifacts, including G128 scales
relabelled as G32 when their counts differ. When counts happen to coincide,
metadata cannot prove which fitting policy originally produced the bytes:
the declared G is authoritative. No automatic reinterpretation or repacking
occurs at load or compile time.

## Arithmetic boundaries

All modes produce F32 outputs and add bias once after the dot. A nonfinite
input makes its complete output row NaN. Finite inputs which overflow during
F16 rounding retain the existing IEEE arithmetic behavior.

* **DYNAMIC_I8:** find one absmax activation scale per row, round to nearest
  even in [-127,127], finish the exact INT32 dot for each storage group, then
  accumulate `float(dot) * (activation_scale * weight_scale)` in ascending
  group order. A G128/BK32 kernel combines four integer subtiles before
  scaling. A tile crossing group boundaries needs separate integer sums and
  scales for each group; a single scale for the whole tile is invalid.
* **F16:** round the input to binary16, reconstruct `float(code) * scale` in
  F32, round that reconstructed weight to binary16, then multiply and accumulate
  in F32. Moving the scale outside the operand rounding changes the contract.
* **F32:** reconstruct and multiply F32 operands without reducing operand
  precision. Reassociation and FMA remain allowed by the public contract.
* **F32_FAST:** explicitly permits compensated TF32 high/low arithmetic and
  ordinary F32 fallback. It never authorizes F16 or dynamic I8 rounding.

CPU/WASM share one implementation. W3 currently uses the direct scalar decoder;
W2 shares the sixteen-code SIMD dynamic-I8 path. W2/W3 device execution is
explicitly refused. For W4/W8, WGSL is the portable device authority and
is translated to SPIR-V, GLSL/GLES and MSL through the existing build. G128 CUDA
specializations retain their group reductions; other G values use a runtime
group loop rather than a template cross product. Padded execution lanes are
masked before reads and contribute arithmetic zero, independently of packed
storage zero encoding.

Fused operators must forward the same parameters and defaults as their
unfused counterparts. RMSNorm uses `eps` with default `1e-6`; fusion must not
substitute another parameter or default. Required-row inference with fusion
enabled needs validation in addition to isolated projection tests.

## CUDA variants and launch planning

The existing `cuda_forward_function_registry_host.inc` remains the function
loading authority. Its packed variant records associate unique internal IDs
with a function handle, execution route, BM/BN/BK, block threads, shared memory,
minimum build architecture, arithmetic and group restrictions, preparation
requirements and workspace layout. `packed_launch_plan.h` projects those
records into checked plans. No public kernel names, tile controls or second
plugin registry are added.

A selected variant determines both its function and launch geometry. Prepared
execution selects a variant from the complete operation before workspace
chunking; each chunk obtains a validated plan for that exact variant. Adding
32 to a row-size heuristic cannot silently run an M16 function with an M32
grid. Missing workspace, aliasing or unsupported device limits fall back
before submission. Failure after preparation or computation begins ends the
operation rather than retrying with a different kernel.

Shared-memory admission includes both static storage reported for the compiled
function and the dynamic bytes requested at launch. Device tests must check
that the registry's static bound covers the loaded function.

The CUDA-private execution descriptor carries route and column tile. The
shared operator retains `rate=G`, `groups=weight_bits`, `normalize=arithmetic`
as documented in its private ABI; `heads` and `window` no longer encode packed
launch choices. G128-only fused decode and prepared row commands explicitly
require G128. New groups use ordinary dispatch so captured scalar commands
cannot reinterpret their scales.

The current policy preserves measured G128 row tiles 16/64 and column tiles
32/64/128, with the restrictions of their existing kernels. Ordinary A8 matrix
instructions advance K by 16; prepared A8 variants advance by 32; A16 matrix
instructions advance by 8. All complete the applicable G128 arithmetic before
advancing its scale. G32 support does
not add a BM=BN=BK=32 kernel. There is no evidence here to select such a new
specialization automatically and no new autotuner. The generic path remains
available as the correctness reference and coverage path.

To add a fork kernel:

1. Implement a body reusing common unpack, rounding, group reduction and store
   helpers. Use a template only for dimensions that change generated code.
2. Add its unique function and packed variant records to the existing registry.
   Declare exact G/arithmetic/architecture/preparation restrictions and memory
   requirements. The CUDA include/build inventory discovers the existing
   `cuda_forward_*.inc` pattern.
3. Prove explicit variant admission and refusal with launch-plan tests and
   force that entry in physical-device numerical tests. Cover tails, multiple
   scales inside BK, nonfinite rows and output guards.
4. Measure the complete launch sequence, including preparation and reduction,
   before adding a narrow automatic-selection rule. Keep the general fallback.

## Review coverage

The audit followed the proto contract, Python and C authoring, shared-weight
deduplication, helper round trips, graph shape contracts, runtime descriptor
construction, scalar/fused row admission, CPU/WASM kernels, portable GPU plan
and WGSL, CUDA functions and replay arguments, registry/PTX embedding, and
exporter/backend tests. The historical G128 TinyLlama benchmark remains a
separate measurement; it is not evidence for new groups or a new code revision.

`tools/llm_baseline/benchmark_packed_groups.py` provides a reproducible synthetic
projection comparison, with separate weight reconstruction error, output error,
packed/scale bytes, completed-call latency, artifact hashes and observed program
names. Run identical G128 fixtures with baseline and candidate libraries for a
regression comparison. Compare freshly fitted groups separately. These errors
do not establish model perplexity or downstream task quality.

Keep dated measurements and validation receipts as build or CI artifacts.
Retain the input, weight, output and library hashes alongside each comparison.
Completed-call timing includes transport and copies; it does not isolate kernel
latency. Packed-word and scale byte counts do not measure peak device memory.

Group support does not establish equal throughput or language-model quality.
Evaluate freshly fitted artifacts on held-out task data and the target backend
before changing the default. Shader translation and raw WGSL execution do not
qualify native Metal, Vulkan or OpenGL graph execution or other CUDA devices.
Native CUDA execution belongs to the full profile; native inference is CPU-only.

Cached WebGPU generation currently has a separate EOS stop-policy limitation:
it can return more IDs than the configured stop policy permits. Passing packed
projection tests does not qualify cached generation termination behavior.
