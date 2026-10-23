# Portable modern decoder shader ABI

The authoritative WGSL files live in `shaders/inference`. Normal Naga shader
publication derives SPIR-V, desktop GLSL, GLES and MSL; these outputs are never
edited by hand. Native adapters and the C WebGPU owner share the private plan
in `llm_gpu_plan.h`/`.c`. All integer hashes and shard addresses remain I32/U32.
Every module has the `main` compute entry and 64 threads per workgroup.

## Uniforms

All modules beginning `llm`, and `packedLinear`, use a 128-byte `Params` struct. Its first 24 words
are U32 in this order:

```
batch sequence keys heads kv_heads key_dim value_dim streams
groups rate window top_k iterations index_heads index_dim rotary_dim
rotary_offset normalize causal interleaved mask_mode flags work first_unit
```

The next four words are F32 `eps hc_eps scale pad1`; the final four are U32
`query_offset pad3 pad4 pad5`. Padding is zero. `query_offset` is the
absolute first query row; causal and sliding-window admission use this offset
while query-shaped masks and output buffers use local row indices. `first_unit`
selects the first output unit for a bounded KVCompress or NGramIds update.
The plan dispatches only its requested unit count, preserving all other rows. `work` means actual invocation count,
except `llmSparseSelect`, where it means actual workgroup count. Flags are:
initial state = 1, final state output = 2, compressed bank = 4, attention sinks
= 8, compressed indexer = 16, explicit positions = 32, compression cos/sin = 64, explicit compressed mask = 128.
SparseIndexer reuses flags 1/2 for candidate input/output, and window/groups for
candidate_block_size/candidate_top_k. These inputs and outputs are exclusive.
Mask modes match `VxLLMOpArgs`: 0 absent, 1 [K], 2 [B,K], 3 [Q,K], 4 [B,Q,K].

The three new `moe` modules use another 128-byte `Params`. First 20 U32 words:

```
rows d_in hidden d_out experts top_k normalize scoring
groups selected_groups group_top_k has_bias has_correction has_hash vocabulary input_scale
slot work pad0 pad1
```

Then four F32 words `temperature scale epsilon limit` and eight U32 padding
words `pad2` through `pad9`. `moeRouterExtended` dispatches one workgroup per
row; all other `moe` modules dispatch `ceil(work/64)` workgroups.

## Bindings and dispatch

Bindings are consecutive storage buffers starting at 0 in the order below.
`RW` marks outputs; other storage is read-only. Uniform bindings are explicit,
Unused optional inputs are
bound to a valid four-byte zero buffer and never read. Missing state output is
bound to a four-byte writable buffer and never written. No null binding is
submitted.

| Shader | Storage bindings in order | Uniform binding | Work |
| --- | --- | --- | --- |
| llmDeltaGates | q, k, a, b, a_log, dt_bias, gates RW | 7 | B*S*H |
| llmDeltaNet | q, k, v, gates, initial_state, out RW, state RW | 7 | B*H*Dv |
| llmSparseSelect | index_q, index_k, index_weights, positions, selection RW | 5 | B*Q workgroups |
| llmSparseAttention | q, k, v, compressed_kv, sinks, mask I32, selection I32, out RW | 8 | B*Q*H |
| llmKVCompress | projected_kv, projected_gate, position_bias, norm_weight, cos, sin, out RW | 7 | B*S |
| llmHyperConnection | input, weight, bias, scale, pre RW, post RW, comb RW | 7 | B*S |
| llmHyperHead | input, weight, bias, scale, out RW | 5 | B*S |
| llmHyperResidual | input, update, post, comb, out RW | 5 | B*S*M*D |
| llmRotaryEmbedding | input, cos, sin, out RW | 4 | B*S*H*D |
| llmNGramIds | token_ids I32, multipliers U32, vocab_sizes I32, offsets I32, out I32 RW | 5 | B*S*H |
| llmQSAScores | q, k, norm_weight, cos, sin, mask I32, scores RW | 7 | B*Q*ceil(K/rate) |
| llmSparseIndexerScores | q, k, index_weights, candidates I32, scores RW | 5 | B*S*S |
| llmSparseIndexerMask | scores, out I32 RW, candidates I32 RW | 3 | B*S*S |
| llmQSAMask | scores, mask I32, out I32 RW | 3 | B*Q |
| llmSignedSqrt | input, out RW | 2 | output elements |
| llmEmbeddingShard | input I32, weight, out RW | 3 | output elements |
| packedLinear | input, packed weight I32, scale, bias, out RW | 5 | output elements |
| moeRouterExtended | input, weight, bias, correction, token_ids I32, hash_indices I32, indices RW, routes RW | 8 | rows workgroups |
| moeGatedUp | input, gate_weight, up_weight, indices, routes, hidden RW | 6 | rows*hidden |
| moeGatedDown | hidden, down_weight, indices, routes, out RW | 5 | rows*d_out |

PackedLinear uses `key_dim=K`, `value_dim=N`, `rate=G` (32, 64, 128 or 256),
`groups=weight_bits` (4 or 8), and private `normalize` arithmetic code
(8 for DYNAMIC_I8, 16 for F16, 32 for F32). Compilation fixes this code;
the graph contains no activation arithmetic parameter. F32_FAST uses the F32
route on these portable shaders.
Flag 1 means bias is present. Each I32 weight word contains `32/weight_bits`
offset-binary values, low bits first; decoding subtracts `2^(weight_bits-1)`.
A row has `ceil(K/(32/weight_bits))` words and `ceil(K/G)` F32 scales. Storage groups are independent of workgroup size;
zero is reserved for the separate per-channel byte format, not PackedLinear.
A16 rounds both operands to IEEE binary16 with ties to even before F32
accumulation. A8 selects one activation scale per row, rounds activations to
signed INT8, and accumulates exact INT32 dots separately for each group before
applying scales. F32 reconstructs weights without reducing operand precision. All modes add bias last. The shader consumes packed weights
directly and reserves no dequantized weight buffer. Its bit-based binary16
rounding does not require optional shader-f16 support.
The portable shader uses integer significand division for A8 quantization:
ordinary WGSL division can move an activation across an integer midpoint.
FP32 arithmetic on portable GPUs may flush subnormal scale products to zero;
backend parity uses an absolute and relative error budget, not bit equality.
For A8 with `N` and `first` divisible by 64, each ordinary 64-invocation
workgroup shares its row maximum, scale and one quantized storage group across
64 output columns. Its static storage is bounded to 64 maximum words, one
scale and 256 I32 activation codes (1,296 bytes after catalogue alignment).
All lanes reach the group barriers, including masked final columns and
nonfinite rows. Each selected output retains its complete integer group dot
and ascending F32 group sums. Other output ranges and arithmetic modes keep
the direct path; buffer bindings, workgroups, partial writes and graph workspace
are unchanged. Intentional NaN/Inf bitcasts are runtime expressions because
WGSL constant evaluation rejects nonfinite results.
CUDA's generic path accepts all supported storage groups. Its measured G128
variants use a warp per output for short batches and sparse tile grids. For
A16 with at least 16 rows and one 16x8 tile per device SM, it uses half
Tensor Core operands, combining F32 partial sums every 128 K elements to
bound cancellation error. This path requires compute capability 7.5 and
PTX ISA 6.5; older PTX targets retain the scalar implementation.
For A8 whole-row ranges with at least 32 output columns, CUDA uses one
256-thread block per row and 32 columns. It shares the row maximum and one
128-element quantized activation group in 160 shared bytes, then computes
exact signed four-byte integer dots (`dp4a` on compute capability 6.1+).
The CUDA-private execution argument selects this path; shared `heads` and
`window` no longer carry tile or route choices. Partial-column ranges
use the warp implementation. Both paths preserve ascending group order and
need no graph workspace. The grid is bounded by the existing two-outputs-per-
block admission proof because the shared path requires at least 32 columns.

GDN launches gates before recurrence. Gates hold four F32 values per
[B,S,value_head], in order decay, beta, query inverse norm, key inverse norm.
GDN recurrence has one independent invocation per value column, private state
of at most 512 F32 values, and writes the exact canonical [B,H,Dk,Dv] final
state layout when requested. A retained state output uses a second device
workspace, followed by a bit-preserving copy to the retained tensor. This keeps
read-only initial-state and writable next-state bindings disjoint during
continuation. Each owner reserves the separate graph-wide maxima of both
workspace slots before execution; their maxima can occur in different nodes.

Sparse attention launches selection only when a compressed bank is present.
Selection is I32 [B,Q,top_k+1]. A nonnegative header is the number of directly
selected consecutive compressed rows. A negative header means `-header` rows
are explicitly listed after the header. This preserves the per-query causal
valid count, including explicit position IDs. A zero top_k uses just one
header per query. The selection stage has two 512-element workgroup arrays;
physical admission bounds `floor(K/rate) <= 512` when an indexer is present.
Non-indexed banks publish their count before accessing workgroup arrays and
have no 512-row cap. Explicit compressed masks use the same selection envelope
with top_k set to K and a B*Q*(K+1) I32 workspace. Their selection pass copies
only causally ready mask entries; it does not access the workgroup arrays.

SparseIndexer first writes B*S*S scores, then ranks each visible compressed
entry by score and lowest index. The optional source candidate mask uses block
maxima and pins the last reachable block. Candidate input and output are
exclusive; a source layer selects final entries from its unrestricted scores.
Both masks retain padded S capacity. Final selection clears future entries;
candidates expand complete selected blocks up to floor(S/rate), with causality
reapplied when a later layer consumes them. This operator is
whole-prefix only; required-row admission refuses it before input mutation.
Regular attention reads the ordinary keep mask and causal/window controls.
Attention streams a numerically stable online softmax over local then selected
compressed keys, starting with a sink denominator when present. It uses at
most 512 private F32 accumulators and no dense score matrix.

KVCompress reserves at most 512 private F32 values per invocation, including
normalization and optional partial/interleaved rotary. Its output keeps S
capacity. Full execution zeroes the incomplete tail exactly as the shared CPU
contract does. Incremental execution updates one newly completed block, including
the previous block overlap for groups=2; incomplete blocks emit no dispatch.

QSA scores hold F32 [B,Q,ceil(K/rate)]. They cache each complete visible key
block once, after weighted RMS normalization and partial rotary. The mask
stage publishes Boolean I32 [B,Q,K], retains incomplete visible tails, and
ranks complete blocks with lowest-index ties. No 512-block cap is needed;
only key_dim is bounded to 512. Capacity is computed with overflow-checked
ceil division in the plan.

For NGramIds `heads=(ngram_size-1)*heads_per_ngram`, `groups=heads_per_ngram`,
`rotary_offset=eos` (-1 disables token resets), and `normalize=pad_token_id`.
The padding ID is independent of the reset ID. Multipliers retain two U32 words each. Arithmetic
emulates wrapping signed I64 products and Euclidean signed remainders exactly;
it requires no optional shader I64 capability. EmbeddingShard uses
`rotary_offset=vocab_offset`, `value_dim=resident shard rows`,
`key_dim=embedding width`, preserving global addresses above 2^24.

MoE router uses at most 1024 experts and 128 groups, with 8704 bytes of
workgroup storage (two 1024-element F32 arrays and one 128-element array).
Inputs/hash metadata and the compiled parameter domain are preflighted by the
owning runtime. Routes and expert IDs preserve the existing F32 MoE tensor
contract. Gate/up/down weights remain independent buffers. Gated MLP launches
Up then Down for each selected `slot`, reusing F32 [rows,hidden] temporary
storage across slots. Slot 0 initializes output; subsequent slots accumulate
in canonical expert order. This needs no 32 KiB workgroup hidden bank and no
`rows*top_k*hidden` scratch. `input_scale` and clipping follow the portable CPU
contract before the nonlinear activation.

All products, device buffer sizes, binding ranges, dispatches, private bounds
and workgroup limits are checked before a plan is submitted. The backend may
refuse unsupported devices or oversized domains; it must not silently execute
CPU math when a GPU backend was required. Mutable temporaries belong to one
execution owner and appear in its reserved workspace inventory.
