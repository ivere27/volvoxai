# LLM development tools

These tools prepare independent references, check numerical and ownership
contracts, and measure completed public calls. They are development inputs;
none is shipped with the engine. Use the tool matching the evidence you need:

| Workflow | Entry points | Purpose |
| --- | --- | --- |
| Trained checkpoints | `models.json`, `../llm_reference.py`, `verify_tokenizer_generate.py` | Pinned model assets, independent HF captures, tokenizer and Generate checks |
| Tiny model corpus | `generate_modern_incremental_stress.py` | Shared offline HF factories and repeatable prefix/boundary references |
| Cached generation | `verify_modern_incremental_native.py`, `verify_modern_incremental.mjs` | `--mode replay` for numerical rows and greedy IDs; `--mode policies` for capacity refusal, retry and seeded sampling |
| Physical WebGPU boundaries | `verify_modern_incremental.mjs --mode boundaries --backend webgpu` | Complete 11-family capacity and retirement checks, requiring a physical device |
| CUDA operators | `probe_cuda_{attention,gemv,reductions,sampling}.py` | Production PTX against independent numerical references; sampling also owns shared Driver helpers |
| Portable operators | `verify_portable_numeric.{py,mjs}`, `verify_portable_wgsl.py`, `verify_wasm_attention_accuracy.mjs` | Independent numerical checks for raw shaders and public operators |
| Backend and lifecycle checks | `verify_modern_{native_gpu,wasm,webgpu}.*`, `verify_webgpu_generation_lifecycle.mjs` | Whole-prefix generation, backend route and owner lifetime checks |
| Native measurement | `measure_native_baseline.py`, `summarize_native_baseline.py`, `native_measurement.json` | Select `--workload cached_execute` or `full_prefix_generate`; keep timing, memory and trace workers separate |
| Capacity experiment | `measure_native_capacity.py`, `run_native_capacity_worker.py` | Source-pinned admission arithmetic and bounded owner workloads; review current source admission before reuse |
| Portable attention measurement | `measure_portable_attention.py` | Prepare a comparison, run a worker, or use `--series` for paired fresh timing/memory processes |
| Multimodal checks | `prepare_*multimodal*.py`, `verify_*multimodal*.*`, `browser_*case.mjs`, `browser_media_processors.mjs` | Prepare raw media and compare complete workflows with retained references |
| Browser checkpoint checks | `browser_checkpoint_case.mjs`, `verify_browser_checkpoint.mjs`, `observe_browser_checkpoint.py` | Browser model execution and separate external device-memory observation |
| Multimodal measurement | `observe_multimodal_native.py`, `run_browser_multimodal_series.py` | Raw-input observation; browser `--selection all` runs the matrix, `--selection qualified` uses bounded environmental retries |
| WASM control measurement | `measure_wasm_baseline.mjs` | Small CPU fixture timing or memory, in fresh processes |

See [measurement workflows](MEASUREMENT.md) for scheduling and interpretation.
Multimodal fixtures live in `multimodal_sensor.json`; its Whisper mel filter
coefficients are reusable input data, rather than historical run results.
`native_sensor_paced.json` supplies the separate paced sensor workload.

## DeepSeek V4 reference policies

The generator defaults to `--reference-policy raw-hf`. This executes the
unmodified HF indexer and preserves its logits, including failures where
`torch.topk` chooses a different equal-scoring block from VolvoxAI's deterministic
lowest-index rule. PyTorch [does not guarantee stable tied indices](https://docs.pytorch.org/docs/stable/generated/torch.topk.html).
The [official V4 reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py)
also delegates selection to `topk`.

For the portable deterministic model definition, select the separate policy:

```sh
python3 tools/llm_baseline/generate_modern_incremental_stress.py \
  --families deepseek_v4 --capacity 128 --prompts 31,63 --steps 16 \
  --budgets 1,3,5,7 --with-boundary --deepseek-v4-seed 746 \
  --reference-policy canonical-lowest-index \
  --out build/llm/v4-canonical/fixtures.json --receipt build/llm/v4-canonical/oracle.json
```

This scoped reference changes only V4 indexer selection: it stably sorts the
actual HF scores, retaining their causal mask and sentinel behavior. All score,
compression, attention and subsequent model calculations remain HF code.
Other model families remain raw HF. Corpus cases and native/browser validation
receipts distinguish the policies. A canonical pass does not replace a raw-HF
failure or establish downloaded-checkpoint quality. Independent F64 operator
tests cover tied and non-tied selection; one-hot compressed values expose the
native selected indices. No score epsilon, seed search or relaxed tolerance is
part of either policy.

## Native decoder reference

To convert a checkpoint and generate text through the native engine, follow
the [HF decoder workflow](../../docs/hf-decoder.md). This page explains its
independent HF reference captures and fixed numerical policy.

The [tokenizer and Generate validation workflow](../../docs/tokenizer-generate-validation.md)
checks complete pinned HF tokenizer definitions with actual trained models,
seeded sampling, ID stops and lossless text across request boundaries.

The first decoder qualification uses two public, ungated checkpoints. SmolLM2
uses the HF `LlamaForCausalLM` architecture and exercises GQA with head width 64.
Qwen3 adds Q/K normalization and head width 128. This proves selected architecture
settings, not every Llama/Qwen checkpoint or the original models' full context
limits. [models.json](models.json) fixes revisions, actual metadata hashes,
published SafeTensors object hashes, input policy and numerical budgets.

| Model | Revision | Layers | Hq/Hkv/D | Source weights |
| --- | --- | --- | --- | --- |
| [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2) | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | 30 | 9/3/64 | BF16, 269,060,552 bytes |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de289a04d12100db370d81485cdf75e47ca` | 28 | 16/8/128 | BF16, 1,503,300,328 bytes |

Both use tied embeddings, SiLU, bias-free projections and unscaled half-split
RoPE. SmolLM2 uses RMS epsilon `1e-5` and theta `100000`; Qwen3 uses epsilon
`1e-6` and theta `1000000`. The qualification domain is batch one, one stream,
context at most 128, with two fixed prompts, prefix lengths 1/16/31, and four
greedy continuation steps per prefix. The full prompt tokenization uses no
automatically inserted special tokens; the saved token bytes are authoritative.

Run from the repository root. All large files stay in ignored `build/llm/`;
download uses the snapshot directory directly without another weight copy.
Keep machine-local validation receipts and absolute artifact paths there too.
The historical `validation.json` is not a build, runtime, or test input; retain
local copies under `build/llm/local-qualification/` instead of committing them.
Commit reusable policies, fixtures, and reproduction tools in this directory.

Capture provenance binds the complete policy file bytes. Consolidating a
catalog changes that identity even when all resolved numerical settings match.
Keep old receipts with the policy bytes from their source commit, or produce a
new capture with the current catalog. Do not rewrite hashes in old receipts.

```sh
python3 tools/llm_reference.py download --model smollm2-135m --snapshot build/llm/snapshots/smollm2-135m
python3 tools/llm_reference.py capture --model smollm2-135m --snapshot build/llm/snapshots/smollm2-135m --output build/llm/reference-v2/smollm2-135m
python3 tools/llm_reference.py download --model qwen3-0.6b --snapshot build/llm/snapshots/qwen3-0.6b
python3 tools/llm_reference.py capture --model qwen3-0.6b --snapshot build/llm/snapshots/qwen3-0.6b --output build/llm/reference-v2/qwen3-0.6b
```

`metadata` obtains only config/tokenizer assets. `verify` hashes every actual
snapshot file. `capture` requires those verified local files and loads offline;
it uses CPU F32 eager attention, seed 1701, two intra-op threads and deterministic
algorithms. Transformers 5.13.0 is pinned to release commit
`6af945f436d85f2b0c5dff9b14feccd27b1d470b`. Its decoder, cache and model-loading
source hashes must match that release. The manifest records exact versions of
PyTorch, NumPy, tokenizer, SafeTensors and Hub, the harness commit/status/hash,
command, config, input bytes and every captured tensor hash.

Each case contains prefill logits, all layer K/V snapshots after RoPE, cached
single-token logits/K/V and the last logit row from fresh full-prefix execution.
Prefill retains all logit rows. KV layout is
`[batch, kv_head, sequence, head_dim]`, retaining compact KV heads. Input files
also expose engine I32 token bytes; HF I64 inputs, positions and masks remain in
the SafeTensors capture. DynamicCache tensors are cloned before each transition
so later updates cannot alter earlier reference snapshots. Cached/full HF
logits, K/V and greedy IDs are checked independently before writing a complete
manifest. EOS never shortens the fixed four-step workload.

Candidate budgets were fixed before candidate execution: logits elementwise
`atol=2e-3, rtol=1e-3` and RMS error at most `5e-4`; layer K/V elementwise
`atol=1e-3, rtol=5e-4`; greedy IDs exact. These allow F32 accumulation differences
across 28–30 layers while requiring per-layer state and selected IDs to agree.
Independent operator fixtures use `1e-5`. The HF cached/full self-check uses
`2e-4` absolute/relative. A mismatch does not widen any budget.

These runs capture numerical evidence and whole-process RSS, including imports
and model load. They do not measure inference latency or GPU performance. Cold
load/compile, warm prefill/continuation and engine allocation/device peaks need
separate runs with capture disabled.

The initial capture and a second independent CPU process produced identical
prefill, continuation, K/V and selected full-prefix logit bytes in all twelve
cases. `build/llm/independent-reproduction.json` records the comparison and both
capture identities. The first capture's repeated full-prefix logit files were
removed after comparison to save disk; its manifests remain as history. The
second capture retains final-row observations and is the active baseline.

The capture dependencies are development-only. The initial reference environment
uses PyTorch `2.12.0+cu130`, Transformers `5.13.0`, tokenizers `0.22.2`,
SafeTensors `0.8.0`, Hub `1.21.0` and NumPy `2.2.6`. The portable runtime acquires
no PyTorch/Transformers dependency. Tiny offline reference checks run with:

```sh
python3 -m unittest tools.tests.test_llm_primitives -v
```


## Additional Llama checkpoints

Two additional public checkpoints passed native CPU full/lite and full CUDA
with the same F32 numerical budgets and six-case scope. All four checkpoint
identities now live in `models.json`. Shared reference settings are stored once;
the F16 fine-tune keeps its original rationale and scaled-RoPE source pin in
`reference_overrides`. `read_policy(model=...)` resolves that model explicitly.

| Alias and policy | Checkpoint / revision | Layers / Hq / Hkv / D | Distinct settings |
| --- | --- | --- | --- |
| `tinyllama-1.1b-chat`, [models.json](models.json) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 / `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | 22 / 32 / 4 / 64 | Untied output, default RoPE, BF16 source |
| `llama3.2-1b-gsm8k`, [models.json](models.json) | axel-datos/Llama-3.2-1B_gsm8k_full-finetuning / `91a56cd31b9a9400ac8a613e1887af43ba9986b4` | 16 / 32 / 8 / 64 | Tied output, Llama3 scaled RoPE, F16 source |

The second is a third-party Meta Llama 3.2 derivative. The publisher's training
state records actual training steps; the numerical test does not rate its task
quality or qualify official gated Meta weights. Independent checks verified
complete source-to-F32 tensor bytes. TinyLlama's F32 shard is 4,400,216,592 bytes;
the fine-tune's shard is 4,963,722,136 bytes, including 20,447,232 bytes of scaled
position constants. The engine executes F32, rather than native F16/BF16.

Prepare the fine-tune from the repository root using fresh output paths:

```sh
python3 tools/llm_reference.py download --policy tools/llm_baseline/models.json --model llama3.2-1b-gsm8k --snapshot build/llm/snapshots/llama3.2-1b-gsm8k
python3 tools/llm_reference.py capture --policy tools/llm_baseline/models.json --model llama3.2-1b-gsm8k --snapshot build/llm/snapshots/llama3.2-1b-gsm8k --output build/llm/reference/llama3.2-1b-gsm8k
python3 tools/export_safetensors.py --model build/llm/snapshots/llama3.2-1b-gsm8k --out build/llm/compact/llama3.2-1b-gsm8k/model.safetensors --target native-cpu --max-context 128 --compact-kv --capture-kv --report build/llm/compact/llama3.2-1b-gsm8k/export-report.json --report-format json
python3 tools/llm_cached_compare.py --policy tools/llm_baseline/models.json --prepare llama3.2-1b-gsm8k --weights build/llm/compact/llama3.2-1b-gsm8k/model.safetensors --out build/llm/cached/llama3.2-1b-gsm8k
```

`--prepare --weights` shares one F32 weight file between execution and capture
graphs, checking all scaled initializers exactly. Run both CPU profiles:

```sh
python3 tools/llm_cached_compare.py --policy tools/llm_baseline/models.json --package build/llm/cached/llama3.2-1b-gsm8k/execution --capture-package build/llm/cached/llama3.2-1b-gsm8k/capture --reference build/llm/reference/llama3.2-1b-gsm8k --profile full --backend cpu --serial-models --out build/llm/llama32-full-cpu.json
python3 tools/llm_cached_compare.py --policy tools/llm_baseline/models.json --package build/llm/cached/llama3.2-1b-gsm8k/execution --capture-package build/llm/cached/llama3.2-1b-gsm8k/capture --reference build/llm/reference/llama3.2-1b-gsm8k --profile inference --backend cpu --serial-models --out build/llm/llama32-lite-cpu.json
```

For CUDA, use the full CUDA-enabled library and the full comparison command
with `--backend cuda` and a new `--out` filename. Operator fallback is forbidden;
the checker requires CUDA route evidence for both compiles, 60 forward/ROW
executions and four Generate calls. For TinyLlama, substitute its alias, paths
in the same `models.json` catalog throughout the workflow.

`--serial-models` retains independently owned logical fresh arrays, releases that
model and verifies empty known allocation/native snapshot inventories, then
loads the cached model. Both phases still execute all 30 states and compare HF
logits/KV; full additionally reads live KV and checks prior-row preservation.
Generate `[4]` and `[1,2,1]` must produce identical exact HF IDs.

Scaled exports require explicit compact KV and bounded absolute positions.
Independent small fixtures also exercise position 127 and reject out-of-range
positions. Context 2,048/131,072 from the original checkpoints, browser loading,
latency and whole-process memory nonregression remain unqualified.
Input encoding uses external HF; TinyLlama also uses external text decode.
Actual fine-tune ByteLevel decode-only bytes/caller streaming passed native
full/lite checks, with `clean_up_tokenization_spaces=False` for the HF comparator.
Full native Encode, default HF text cleanup, chat templates and sampling are
separate workflows.

Completion evidence is in
[TinyLlama qualification](../../build/llm/llama-variants/qualification.json) and
[fine-tune qualification](../../build/llm/llama-finetune/qualification.json).
Large temporary source and F32 weights were removed after closing their proofs;
saved references and reports remain. Recreate weights before replaying a run.
