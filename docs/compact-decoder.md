# Run a decoder with compact KV reuse

A decoder can keep key/value vectors from earlier tokens and compute only the
next token row. Grouped query attention maps several query heads directly to
one KV head: the cache stores `Hkv * D` values per token instead of repeating
them for every query head. MHA, GQA and MQA share this contract.

The compact route uses F32 on native CPU or CUDA. Selected checkpoints include
SmolLM2-135M, Qwen3-0.6B, TinyLlama-1.1B-Chat and a public Llama 3.2 1B fine-tune,
with an additional original Qwen3-4B check on full CUDA. These checks use
batch one and an exported 128-token context bound. Qwen's
normalization reduces each head's D features independently; RoPE rotates each
head at that token's supplied absolute position. The engine owns cache pages,
decoder state and ID feedback. Your application owns tokenization and task stops.

CPU runs in both native profiles. CUDA requires the full native library and a
compatible NVIDIA device; the inference library remains CPU-only. Follow the
[CUDA build instructions](cuda.md#build-and-selection), adding the
`volvoxai_shared` target to build the Python-consumed full library. The tested
CUDA route uses strict F32, direct grouped attention with head widths up to 128,
head RoPE and head Q/K normalization. Training and low-precision arithmetic have
separate contracts.

Native Vulkan and OpenGL also implement caller-directed compact-cache Graph
rows. The [SmolLM2 Vulkan check](../build/llm/memory-performance-001/native-final-vulkan-first-cell-004.json)
passes the six-case HF comparison at capacity 128 with a 1024 MiB Vulkan arena.
The [SmolLM2 OpenGL check](../build/llm/memory-performance-001/native-final-opengl-first-cell-006.json)
also passes those six cases after closing the cache activation pass before
retained output copying. Both checks use prefixes of 1, 16 and 31 tokens with
four decode steps; reserved capacity 128 does not mean an active 128-token test.
Dense compact-cache autoregressive `Generate` remains unsupported on native
Vulkan, OpenGL and Metal. Modern hybrid/MoE and DeepSeek cache generation uses
the separate contracts in the [HF decoder guide](hf-decoder.md#generate-with-a-modern-decoder).
Metal execution on Apple hardware remains unqualified.

## Prepare a compact package

Prepare a local checkpoint and native runtime using the
[HF decoder guide](hf-decoder.md#prepare-the-checkpoint-and-runtime), then export
to a fresh directory:

```sh
python3 tools/export_safetensors.py --model build/llm/snapshots/smollm2-135m --out build/llm/compact/smollm2-135m/model.safetensors --target native-cpu --max-context 128 --compact-kv --cached-decoder --report build/llm/compact/smollm2-135m/export-report.json --report-format json
```

Use the Qwen snapshot and destination for Qwen3-0.6B. Source BF16 weights are
converted to F32 storage. `--compact-kv` chooses direct grouped attention.
`--cached-decoder` adds initialization, row transition and greedy generation
contracts. It requires compact KV and excludes `--capture-kv`: paged cache K/V
remain private to their context.

The exporter validates this graph with `--target native-cpu`. The same portable
graph can be compiled with a required CUDA policy on the CUDA-enabled artifact;
that compilation validates the actual device and complete shape domain. An
export check alone does not qualify a CUDA device.

Omitting `--cached-decoder` creates a stateless compact full-prefix graph. That
version can use `--capture-kv` for a full-forward comparison. The
[baseline workflow](hf-decoder.md#generate-four-tokens-with-a-caller-loop) remains
available for complete-prefix evaluation.

## Generate IDs while reusing the cache

Run this example from the repository root with `PYTHONPATH=python`. Set
`package` to your exported directory; the validation execution packages below
also work. The example uses the existing public Python service clients.

```python
from pathlib import Path
import json
import numpy as np
from transformers import AutoTokenizer
import volvoxai as vx

alias = "smollm2-135m"
backend = "cpu"  # Use "cuda" with the full CUDA-enabled native library.
package = Path("build/llm/compact") / alias
snapshot = Path("build/llm/snapshots") / alias
tokenizer = AutoTokenizer.from_pretrained(
    str(snapshot), local_files_only=True, trust_remote_code=False)
conditioning = tokenizer.encode("The future of on-device AI is", add_special_tokens=False)
capacity, steps = 128, 4
if not conditioning or len(conditioning) + steps > capacity:
    raise ValueError("Choose a nonempty prefix that fits the context bound.")
tokens = np.zeros((1, capacity), dtype=np.int32)
tokens[0, :len(conditioning)] = conditioning
keep = np.zeros_like(tokens)
keep[0, :len(conditioning)] = 1
values = {"token_ids": tokens, "attention_mask": keep,
          "position_ids": np.arange(capacity, dtype=np.int32),
          "active_lengths": np.array([len(conditioning)], dtype=np.int32)}
graph = json.loads((package / "graph.json").read_text())
cache_names = [node["inputs"][port] for node in graph["nodes"]
               if node["opType"] == "CrossSDPA" for port in ("k", "v")]
p = vx.pb
host = vx.open_library()
try:
    engine = vx.VxInferenceServiceClient(host)
    runtime = engine.create_runtime(p.CreateRuntimeRequest(cpu_threads=2))
    model = engine.load_model(p.LoadModelRequest(runtime_id=runtime.runtime_id,
        graph_path=str(package / "graph.json"),
        weight_paths=[str(package / "model.safetensors")]))
    compiled = engine.compile_model(p.CompileModelRequest(model_id=model.model_id,
        policy=p.BackendPolicy(mode=p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
            backends=[backend], operator_fallback=p.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
    context = engine.create_execution_context(p.CreateExecutionContextRequest(
        compiled_model_id=compiled.compiled_model_id,
        autoregressive=p.AutoregressiveExecution(
            sampling=p.SamplingMode.SAMPLING_MODE_GREEDY, max_steps=steps),
        limits=p.ExecutionLimits(cache=p.ContextCacheOptions(
            tensors=cache_names, page_tokens=2, clear_on_recycle=True))))
    initial = [p.Tensor(name=name, dtype=p.DataType.DATA_TYPE_I32,
        shape=list(array.shape), inline=array.tobytes()) for name, array in values.items()]
    generated = []
    for index, count in enumerate((1, 2, 1)):
        packet = engine.generate(p.GenerateRequest(context_id=context.context_id,
            initial_inputs=p.GenerationInputs(values=initial) if index == 0 else None,
            budget=p.GenerationBudget(max_steps=count),
            outputs=p.OutputOptions(storage=p.OutputStorage.OUTPUT_STORAGE_INLINE,
                selection=p.TensorOutputSelection(names=["generated_ids"]))))
        generated.extend(np.frombuffer(packet.outputs[0].inline, dtype="<i4").tolist())
    print(tokenizer.decode(generated))
finally:
    host.close()
```

The fixed-capacity tensors contain the prefix and masked padding.
`active_lengths` explicitly identifies the initialized prefix. The position
table covers every row, and feedback uses the next logical position. The first
`Generate` initializes state; later calls reuse it without resupplying inputs.
Each call returns complete selected IDs. Host close releases all its resources,
including partially created resources after a failure. Returned inline bytes
belong to the Python message.

## Choose sampling and completion

The same context can use seeded random generation instead of greedy selection:

```python
policy = p.AutoregressiveExecution(
    sampling=p.SamplingMode.SAMPLING_MODE_RANDOM,
    max_steps=128, seed=42, temperature=0.8, top_k=40, top_p=0.9,
    eos_ids=[eos_token_id], stop_sequences=[p.IdStopSequence(ids=stop_token_ids)])
```

Pass `policy` as `CreateExecutionContextRequest.autoregressive`.
`eos_token_id` and the nonempty `stop_token_ids` come from your checkpoint and
application policy. The model's
categorical contract still names its F32 logits, integer output and feedback.
CPU, WASM CPU and CUDA select from these logits inside the engine. CUDA retains
device feedback and reads one selected I32 value per new ID for control;
greedy without a stop policy retains its existing device-only control route.
WebGPU currently refuses the additional sampling and stop settings.

Random generation defaults to seed 0, temperature 1, top-k 0 (all categories)
and top-p 1. Temperature must be finite and positive; top-p must be in `(0,1]`.
A positive top-k cannot exceed the vocabulary size. Candidates are ordered by
descending score, with smaller IDs winning ties; top-p is applied after top-k
and includes the ID that crosses the probability threshold. Random selection
requires F32 logits and at most 1,048,576 categories. Greedy rejects random-only
settings; temperature zero is invalid.

Each context owns its RNG. The same input, model revision, backend, settings
and seed produce the same IDs across different Generate step/chunk boundaries.
This does not promise bit-identical random sequences across backends or
compatibility with an external caller's RNG.

`eos_ids`, `stop_ids` and nonempty `stop_sequences` match generated IDs, excluding
the prompt. Sequences may cross Generate calls. Matching IDs stay in the final
packet; EOS takes precedence and reports `EOS`, while other matches report
`ID_STOP`. The context becomes terminal and a new task needs a new context.
The combined stop policy admits at most 4,096 IDs and 128 sequences. Text
decoding and special-token removal remain caller choices; see the
[tokenizer workflow](tokenizers.md).

For CUDA, select the full library before starting Python:

```sh
VOLVOXAI_LIBRARY="$PWD/native/libvolvoxai.so" PYTHONPATH=python python3 path/to/script.py
```

This profile uses fixed-count greedy selection. A per-call `max_steps` divides
output into chunks without changing the token sequence. CUDA keeps selected IDs
and KV feedback on the device between rows; returning a selected ID chunk copies
those IDs for the caller. The example uses an external tokenizer and supplies
no stop policy; sampling and stopping use the options above.

For caller-directed rows, invoke `initialize` with a full scope, then
`transition` with `SlotRows`. Bind each input exactly once and reuse the
invariant `active_lengths`. Create a graph context with
`DECODE_ROW_MODE_REQUIRED` and `require_incremental=true`; successful transitions
report `DECODE_MODE_ROW`. The [execution guide](scheduling-and-dynamic-batching-design.md)
and [proto schema](../proto/volvoxai.proto) define the complete contracts.

CUDA row execution requires the base adapter, fixed batch/slot count, public
I32 active lengths and host-supplied or immutable mask/position controls. Row
buffers and their maximum capacities are reserved when the context is created.
Oversized domains, GPU-produced row controls and incompatible row fusions are
refused before mutation. Paged KV uses bounded gather/compute/scatter on the
device; it does not implement a fused attention kernel with direct page-table
addressing. CUDA prefix publication/reuse is refused until its storage has a
bounded resource proof.

## Reproduce the cached numerical check

If baseline F32 packages and frozen HF references exist, prepare separate C
execution/capture graphs while sharing the existing weight bytes:

```sh
python3 tools/llm_cached_compare.py --prepare smollm2-135m --out build/llm/c-stage/packages/smollm2-135m
python3 tools/llm_cached_compare.py --prepare qwen3-0.6b --out build/llm/c-stage/packages/qwen3-0.6b
python3 tools/llm_cached_compare.py --package build/llm/c-stage/packages/smollm2-135m/execution --capture-package build/llm/c-stage/packages/smollm2-135m/capture --reference build/llm/reference-v2/smollm2-135m --profile full --out build/llm/c-stage/smollm2-full.json
```

Run each model with both profiles and fresh report filenames. The comparison
requires actual row mode and compact pool accounting. It checks every prefill
and continuation logit against HF and a fresh full forward, with exact greedy
IDs. Full reads every layer's live private K/V through the existing debug API,
including paged prefill, and checks prior rows for exact F32 value equality.
This check does not distinguish signed-zero bit patterns. Inference records
that live debug KV observation is unavailable.

To test CUDA, use the full profile with `--backend cuda` and a fresh report:

```sh
python3 tools/llm_cached_compare.py --package build/llm/c-stage/packages/qwen3-0.6b/execution --capture-package build/llm/c-stage/packages/qwen3-0.6b/capture --reference build/llm/reference-v2/qwen3-0.6b --profile full --backend cuda --out build/llm/qwen3-cuda-cached-full.json
VOLVOXAI_TEST_NATIVE_GPU_BACKEND=cuda PYTHONPATH=python VOLVOXAI_LIBRARY="$PWD/native/libvolvoxai.so" python3 -m unittest discover -s python/tests -p test_cuda_compact_decode.py -v
```

The CUDA comparison requires attested CUDA execution for compilation, fresh
forwards, cached rows and each Generate chunk, with operator fallback forbidden.
It compares live private K/V, checks earlier rows for exact F32 value equality
and verifies that four steps in one chunk produce the same IDs as chunks of
one, two and one.

Fixed numerical tolerances are unchanged from the baseline. Numerical and
known-allocation cleanup evidence does not qualify latency, total process memory
nonregression, browser model loading, training backward or a robot policy.
Other device and training routes must independently qualify the attention
contract before accepting it.

## Original Qwen3-4B on CUDA

The official `Qwen/Qwen3-4B-Instruct-2507` checkpoint at revision
`cdbee75f17c01a7cc42f958dc650907174af0554` passes full-profile native CUDA
qualification on an RTX 3090. Its original BF16 weights are widened exactly to
F32: the execution payload is 16,089,872,384 bytes, without quantization.
This verifies F32 execution rather than BF16 arithmetic.

The [qualification summary](../build/llm/larger-checkpoint-001/attempt5/qualification-002/original-model-qualification-summary.json)
records six fresh full-forward and six cached cases against HF, covering English
and Korean prefixes of 1, 16 and 31 tokens followed by four row transitions.
All layer K/V values, logits, prior K/V value preservation and greedy IDs pass
the fixed budgets with CUDA required and fallback forbidden. Four separate
`Generate` runs cover both languages with one four-ID call and calls split
into one, two and one; their IDs match HF and each other. The
[independent audit](../build/llm/larger-checkpoint-001/attempt5/independent-final-audit-001.json)
checks the retained numerical records, reference IDs, provenance and owner
retirement.

The configured capacity is 128, while the longest exercised active sequence is
35 tokens. These results do not qualify an active 128-token prompt, concurrent
sessions or the largest model that any 3090 can run. Sampled task GPU residency
through NVML reaches 18.35 GB during instrumented correctness runs; it is neither
an exact whole-device memory peak nor a latency result. Whole-system swap
activity also prevents performance and whole-memory regression claims for
these runs. This 4B checkpoint has not been qualified on CPU, Vulkan, OpenGL,
Metal or WebGPU.

## Additional Llama checkpoints and scaled positions

TinyLlama-1.1B-Chat exercises untied output weights and 32 query / 4 KV heads.
The public `axel-datos/Llama-3.2-1B_gsm8k_full-finetuning` checkpoint exercises
32 query / 8 KV heads, tied embeddings and Llama3 scaled RoPE. Both passed the
same six-case HF comparison on native CPU full/lite and full CUDA. Their exact
revisions, independent conversion checks and complete preparation commands are
in the [reference guide](../tools/llm_baseline/README.md#additional-llama-checkpoints).

For `rope_type=llama3`, explicitly choose `--max-context` and `--compact-kv`.
The exporter computes the model's scaled frequencies and bounded sine/cosine
tables, then applies `RotaryEmbedding`. Query and key heads across every layer
share one table pair, occupying `8 × max_context × head_dim` bytes. Absolute position
IDs must be inside the exported range; a 128-row table accepts positions 0–127.
This bounded export does not qualify the checkpoint's original long context.
Other scaling types remain unsupported. Repeated-head noncompact export is
unsupported for Llama3 scaling.
For the selected fine-tune's 128-position, 64-channel rotary geometry, these
tables occupy 65,536 bytes. An 8B model with 128 rotary channels and an 8192-position
bound uses 8 MiB. The exporter creates no dense rotation matrices; the retained
checkpoint comparisons above predate this lowering change.

Input encoding remains external HF. TinyLlama's text decoding also remains
external. The fine-tune's actual ByteLevel vocabulary passed native decode-only
bytes and caller UTF-8 streaming checks; full HF Encode, its default text cleanup,
chat formatting and sampling remain caller workflows.

The comparison tool accepts `--policy` for these separately pinned checkpoints
and `--weights` with `--prepare` to share an already exported F32 shard. Reused
scaled constants are checked exactly before graphs are published.
`--serial-models` loads the fresh companion first, retains only logical prefix
arrays, releases its model and native snapshots, then loads the cached model.
It performs all numerical checks while keeping one model loaded at a time.
This reduces validation resource demand without qualifying whole-process memory.
