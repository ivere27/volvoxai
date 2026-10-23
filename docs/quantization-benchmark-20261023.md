# TinyLlama 1.1B quantization benchmark — RTX 3090, 2026-10-23

This benchmark used a fresh CUDA build of the source on an NVIDIA GeForce RTX 3090 with 24 GiB of memory. It compared nine variants: a dense F32 baseline and four arithmetic modes for each of the W4 and W8 RTN artifacts. All arithmetic modes within a weight width used identical weights. **W4 / DYNAMIC_I8 was fastest overall. W8 / F16 combined high throughput with observed quality close to the baseline. Context creation added substantial latency to each request.**

- With a 512-token prompt and 128 generated tokens, W4 / DYNAMIC_I8 decoded at 521.90 tok/s, or 3.28 times the dense F32 rate. It had the lowest median generation time in all ten workloads.
- W8 / F16 decoded at 391.47 tok/s. Its perplexity on the WikiText subset was 8.4885, compared with 8.4834 for the baseline. This limited evaluation does not establish quality equivalence or deployment suitability.
- Compared with packed F32, F32_FAST improved 512-token prefill TTFT by a factor of 1.88 for W4 and 1.62 for W8. Decode throughput differed by less than 1% under the same conditions.
- Context creation took approximately 0.56–0.58 seconds for W4, 0.81–0.82 seconds for W8, and 2.31 seconds for dense F32. The first `Generate` call on an existing context and the first token of a new request have different latency costs.

## Throughput and latency

The following results use a 512-token prompt, 128 generated tokens, batch size 1, and capacity 2048. Each condition had one initial cycle and two warmup cycles, followed by seven measured cycles. TTFT is the duration of the first `Generate(max_steps=1)` call; it excludes context creation, model loading, compilation, and tokenization. TTFT including context creation is the median of the actual sum for each cycle.

| Storage / PackedArithmetic | TTFT p50 / p95 (ms) | Decode p50 (tok/s) | Generation total p50 (ms) | Context creation p50 (ms) | TTFT including context creation p50 (ms) | Sampled process GPU memory maximum (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Dense F32 | 237.94 / 239.04 | 158.98 | 1036.97 | 2308.18 | 2546.43 | 11962 |
| W4 / DYNAMIC_I8 | 74.78 / 74.83 | 521.90 | 318.06 | 556.05 | 630.89 | 8692 |
| W4 / F16 | 85.19 / 85.28 | 446.88 | 369.43 | 573.01 | 658.19 | 8692 |
| W4 / F32 | 393.69 / 394.18 | 440.10 | 682.27 | 558.80 | 952.50 | 8692 |
| W4 / F32_FAST | 209.30 / 209.77 | 443.32 | 495.72 | 579.78 | 789.08 | 8692 |
| W8 / DYNAMIC_I8 | 153.48 / 153.62 | 296.18 | 582.28 | 810.66 | 964.16 | 9230 |
| W8 / F16 | 86.08 / 86.14 | 391.47 | 410.57 | 808.54 | 894.59 | 9220 |
| W8 / F32 | 419.82 / 420.20 | 377.68 | 756.18 | 820.74 | 1240.56 | 9220 |
| W8 / F32_FAST | 258.99 / 259.07 | 380.74 | 592.49 | 808.66 | 1067.73 | 9220 |

Decode throughput covers the 127 token IDs after the first token. GPU memory is the maximum process usage sampled through NVML at 20 ms intervals during a separate replay after the timed cycles. It may miss brief allocation peaks. No other GPU compute processes were observed in those samples.

The next table shows TTFT by prompt length and latency between individual token responses. TTFT is the median for workloads generating 32 tokens. The last column summarizes 217 continuation response intervals with a 512-token prompt and one token requested per call.

| Storage / arithmetic | 31-token prompt TTFT (ms) | 128-token prompt TTFT (ms) | 512-token prompt TTFT (ms) | 1536-token prompt TTFT (ms) | Per-token response p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| Dense F32 | 52.77 | 129.49 | 237.27 | 673.81 | 6.60 / 6.81 |
| W4 / DYNAMIC_I8 | 48.76 | 68.01 | 74.73 | 225.04 | 2.29 / 2.43 |
| W4 / F16 | 48.72 | 63.85 | 85.13 | 261.06 | 2.58 / 2.72 |
| W4 / F32 | 68.11 | 131.21 | 393.32 | 1197.31 | 2.63 / 2.74 |
| W4 / F32_FAST | 68.22 | 92.31 | 209.44 | 594.71 | 2.60 / 2.68 |
| W8 / DYNAMIC_I8 | 55.36 | 74.44 | 153.45 | 460.92 | 3.68 / 3.84 |
| W8 / F16 | 49.21 | 68.75 | 86.04 | 266.61 | 2.83 / 2.99 |
| W8 / F32 | 69.70 | 138.05 | 418.86 | 1277.07 | 3.00 / 3.12 |
| W8 / F32_FAST | 69.82 | 107.77 | 258.98 | 726.70 | 2.98 / 3.11 |

For a 512-token prompt, requesting one token at a time reduced decode throughput by 4.7–16.3% compared with grouped calls generating the same 32 tokens. The additional cost was 0.29–0.40 ms per token. The explicitly measured Python time between `Generate` calls had a median of 0.019–0.026 ms. Further profiling is needed to attribute the remaining overhead to transport, synchronization, or execution paths.

## Task quality

The WikiText evaluation used the first 16,384 tokens of WikiText-2 raw test, split into sixteen disjoint 1024-token blocks, each starting with BOS. Perplexity was calculated by dividing the total scored-token NLL by the scored-token count and taking the exponential. These are fixed-subset results, not standard full-corpus WikiText perplexity scores.

The ARC evaluation used the first 100 ARC-Easy test questions and their 399 answer choices. It was a zero-shot likelihood evaluation with the prompt `Question: {question}\nAnswer:` and no chat template. Both the raw summed likelihood and the score divided by the answer choice's Unicode character count were specified in advance and are reported below.

| Storage / arithmetic | WikiText PPL | ΔNLL | KL (nats) | Top-1 agreement with baseline (%) | ARC raw (%) | ARC character-normalized (%) |
|---|---:|---:|---:|---:|---:|---:|
| Dense F32 | 8.4834 | 0.000000 | 0.000000 | 100.000 | 65.0 | 59.0 |
| W4 / DYNAMIC_I8 | 9.6193 | 0.125662 | 0.154545 | 81.799 | 62.0 | 54.0 |
| W4 / F16 | 9.4647 | 0.109455 | 0.137509 | 82.489 | 63.0 | 56.0 |
| W4 / F32 | 9.4648 | 0.109468 | 0.137518 | 82.483 | 63.0 | 56.0 |
| W4 / F32_FAST | 9.4648 | 0.109468 | 0.137518 | 82.483 | 63.0 | 56.0 |
| W8 / DYNAMIC_I8 | 8.6343 | 0.017630 | 0.015801 | 94.470 | 64.0 | 56.0 |
| W8 / F16 | 8.4885 | 0.000608 | 0.000441 | 98.792 | 64.0 | 59.0 |
| W8 / F32 | 8.4886 | 0.000609 | 0.000441 | 98.779 | 64.0 | 59.0 |
| W8 / F32_FAST | 8.4886 | 0.000609 | 0.000441 | 98.779 | 64.0 | 59.0 |

W8 / F16 changed the raw predictions for three questions relative to dense F32: one correct answer became incorrect, and no incorrect answers became correct. Its character-normalized prediction changed for one question, but both predictions were incorrect, leaving accuracy unchanged at 59%. Within each weight width, F16, F32, and F32_FAST produced identical predictions for all 100 questions under both scoring rules.

The KL direction is `P_dense_F32 || P_candidate`. The arithmetic setting applies to PackedLinear; it does not change every other operator to I8 or F16. Quality scores use full causal prefill logits. GPU checks covered dense F32 prefill/incremental consistency and, for each packed mode, consistency across sequence-length changes of 128 → 31 → 128 and fresh contexts. These checks do not replace evaluation of complete autoregressive generations.

With 100 questions, each question accounts for one percentage point. This is an exploratory comparison of fixed initial subsets, with no predefined equivalence margin or adoption threshold. Evaluation data was not used for calibration or fitting. MSE_CLIP/IMPORTANCE fitting, other models and GPUs, and models of 7B parameters or larger were outside the scope of this benchmark.

## Opportunities in the API and execution structure

Even W4 / DYNAMIC_I8 takes 630.89 ms to produce the first token when context creation is included, compared with 74.78 ms for `Generate` alone. Each autoregressive context currently creates a private engine, materializes/uploads GPU weights, and reserves memory for the maximum shape. Releasing the last context also releases CUDA module/device references. The source shows that these operations can recur, but their individual contributions to latency were not measured.

A priority for subsequent optimization is to retain immutable GPU weights and CUDA module/device resources for the compiled model's lifetime, while contexts own KV caches and mutable execution state. The results also support keeping storage format and execution arithmetic separate in the API: DYNAMIC_I8 performed best for W4, F16 performed best for W8, and F32_FAST mainly benefited long prefill workloads.

Source references: [context creation and preload](../native/src/runtime/public_api_execution_context.inc), [CUDA lifetime](../native/src/backends/cuda/host/cuda_backend_lifecycle_host.inc), and [F32_FAST shape routing](../native/src/backends/cuda/host/cuda_graph_llm_host.inc). This structural optimization was neither implemented nor measured in the benchmark.

## Environment and reproduction records

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. The dense baseline uses the original BF16 storage with F32 computation.
- GPU: RTX 3090 24 GiB, driver 535.309.01. Release native full library, clang 14 / nvcc 11.5, SM86. The repository default was `VOLVOXAI_CUDA_FAST_FP32=OFF`; explicit F32_FAST arithmetic is a separate setting.
- W4 and W8 use RTN with group size 128. The four arithmetic modes reused identical weight hashes within each width. Existing shared weights were read-only; task-owned graph copies removed only the obsolete `activation_bits` field.
- Graph and weight sizes: dense 2098.35 MiB, W4 649.46 MiB, W8 1142.71 MiB. These sizes exclude the tokenizer and native library.
- GPU clocks were automatic, and variants ran sequentially. The p95 values are descriptive statistics from seven cycles or the retained token responses, not estimates of production service tails or confidence intervals. This benchmark is not a comparison against a previous code revision or another inference engine.

The repository retains the [performance CSV](benchmarks/quantization-20261023/performance.csv), [quality CSV](benchmarks/quantization-20261023/quality.csv), and [provenance record](benchmarks/quantization-20261023/provenance.json). The [benchmarking guide](quantization-benchmarking.md) describes the reusable workflow. Raw logs, native binaries, full result files, and detailed audit receipts remain separate local or external artifacts and are not included in Git.

Validation covered 90 performance conditions, 630 measured cycles, and 18 comparisons across generation chunk boundaries. Quality aggregation and per-question predictions were independently reviewed for nine variants × 415 cases. Thirteen benchmark-tool tests passed. A result-buffer budget error found during preflight was fixed to account for the graph's maximum output size; a regression test was added before rerunning quality evaluation. The evaluation policy and numerical tolerances were unchanged. All task-owned remote temporary files were removed after retrieving the results.
