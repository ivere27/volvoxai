# Validate a tokenizer with actual text generation

An application first encodes its prompt with the checkpoint's tokenizer, then
passes those IDs to a Generate context. The context owns ID selection, its seed
and generated-ID stop history. The caller turns completed ID packets into text.
Use [the tokenizer guide](tokenizers.md) for application code and
[the generation guide](compact-decoder.md) for context and output budgets.

`tools/llm_baseline/verify_tokenizer_generate.py` provides a repeatable correctness
workflow for actual trained SmolLM2, Qwen3 and TinyLlama checkpoints. It captures
independent HF observations before loading the engine, then sends every engine
operation through the generated public native API. CPU full and inference
profiles are selected explicitly; CUDA requires the full profile and refuses
operator fallback. The workflow recomputes the full input prefix. It does not
qualify incremental-cache performance, maximum context or idle-host latency.

## Prepare a pinned checkpoint

The existing [baseline policy](../tools/llm_baseline/models.json) fixes SmolLM2
and Qwen3 revisions and hashes. Reuse existing local snapshots and F32 packages.
If they are absent, obtain and export them using the
[HF decoder workflow](hf-decoder.md#prepare-the-checkpoint-and-runtime).
Preparation below writes only a small graph and reuses the existing weight file.
It keeps the package's numerical operators, selects its logits output and
attaches the canonical full-prefix Generate contract and ArgMax ID output.

```sh
python3 tools/llm_baseline/verify_tokenizer_generate.py prepare --graph build/llm/packages/smollm2-135m/graph.json --config build/llm/snapshots/smollm2-135m/config.json --out build/llm/e-actual-model/smollm2-135m-generation.graph.json
python3 tools/llm_baseline/verify_tokenizer_generate.py capture --model smollm2-135m --snapshot build/llm/snapshots/smollm2-135m --out build/llm/e-actual-model/hf-smollm2-135m
```

Capture verifies the actual snapshot hashes and the pinned Transformers source,
then evaluates HF F32 eager attention independently. The unmodified pinned
`tokenizer.json` is interpreted by Rust `tokenizers` as the exact tokenizer
authority. Captures contain prompt IDs, last-row logits, selected IDs, decode
results and categorical CDF witnesses. PyTorch and HF are development
dependencies; the deployed engine uses its C tokenizer and numerical kernels.

RANDOM observations apply the public SplitMix64 inverse-CDF contract to HF
logits. They do not use PyTorch's different random generator as an ID oracle.
The cases cover default seed zero, seed seven with temperature/top-k/top-p and
the maximum uint64 seed with top-k one. Candidate ordering, protobuf F32 option
rounding and CDF margins are recorded. Universal sample-ID equality across
backends is outside the contract; equality is checked for these selected cases.

## Replay through the engine

Build the normal native libraries first. Select each release profile explicitly:

```sh
python3 tools/llm_baseline/verify_tokenizer_generate.py replay --model smollm2-135m --snapshot build/llm/snapshots/smollm2-135m --reference build/llm/e-actual-model/hf-smollm2-135m --graph build/llm/e-actual-model/smollm2-135m-generation.graph.json --weights build/llm/packages/smollm2-135m/model.safetensors --library native/libvolvoxai.so --profile full --backend cpu --out build/llm/e-actual-model/cpu-full-smollm2.json
python3 tools/llm_baseline/verify_tokenizer_generate.py replay --model smollm2-135m --snapshot build/llm/snapshots/smollm2-135m --reference build/llm/e-actual-model/hf-smollm2-135m --graph build/llm/e-actual-model/smollm2-135m-generation.graph.json --weights build/llm/packages/smollm2-135m/model.safetensors --library native/libvolvoxai-lite.so --profile inference --backend cpu --out build/llm/e-actual-model/cpu-lite-smollm2.json
```

For Qwen3, replace the model alias and paths with `qwen3-0.6b`. For physical
CUDA, select a full library compiled with CUDA and use `--profile full --backend
cuda`. `--loader` selects the matching Synurang native host library when it is
not beside the engine. Missing backends or incomplete routes fail the run.
Each output receipt must use a new filename; failed attempts are preserved.

Each model has two Korean/emoji prompts and four selection policies. Four newly
selected IDs are compared under one call, step budgets `[1,2,1]` and output
budgets `[8,4,4]` bytes. `max_output_bytes` bounds I32 output storage: each ID
costs four bytes. It does not bound the number of UTF-8 bytes represented by
that ID. Identical IDs, raw token bytes and whole text must survive these
request divisions.

EOS and ID-stop checks explicitly configure the first independently selected
ID as a stop. The sequence test spans the first two IDs across calls; matching
does not use prompt IDs. EOS wins when both configured stop kinds match.
Cancellation follows a completed two-ID chunk, refuses continued generation
and permits a fresh context to reproduce the reference. Retained INLINE packets
remain immutable after their context retires. Resource receipts check that
known allocation inventory and unconsumed result bytes return to baseline.
Whole-process RSS is recorded separately and does not assert a whole-system or
GPU peak-memory bound.

## Preserve bytes and choose an explicit text end policy

ByteLevel decode returns raw bytes for each generated packet. A caller feeds
those bytes to one incremental UTF-8 decoder and finalizes exactly once at
normal termination. These actual SmolLM2 prompts produce both Korean and emoji
split over output IDs. The Korean four-ID continuation ends in an incomplete
UTF-8 sequence; its normal finalization inserts one replacement character.
The cancellation chunk leaves two pending bytes, so finalize and discard have
different, checked results. Qwen3 exercises complete Korean tokens as well.

The receipt separately compares raw token bytes with an independent inverse
of HF's ByteLevel alphabet. Tokenizer-only cases cover English, digits,
whitespace, normalization, special-token keep/skip, invalid bytes and incomplete
UTF-8. A raw case requiring a missing standalone byte ID is recorded as
unavailable and is excluded from engine success counts.

## Validate converted SentencePiece BPE with TinyLlama

[The TinyLlama policy](../tools/llm_baseline/models.json) pins
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` at
`fe8a4ea1ffedaf415f4da2f062534de366a451e6`. This is a public trained Llama2-derived
checkpoint and a real converted SentencePiece BPE workflow. It does not qualify
the original gated Meta Llama 1/2 checkpoints. Source weights occupy about
2.2 GB and the F32 export about 4.4 GB; use a task-owned temporary directory
when repository storage is limited.

```sh
python3 tools/llm_reference.py download --policy tools/llm_baseline/models.json --model tinyllama-1.1b-chat --snapshot /tmp/volvox-e-tinyllama/snapshot
python3 tools/export_safetensors.py --model /tmp/volvox-e-tinyllama/snapshot --out /tmp/volvox-e-tinyllama/package/model.safetensors --target native-cpu --max-context 128
python3 tools/llm_baseline/verify_tokenizer_generate.py prepare --graph /tmp/volvox-e-tinyllama/package/graph.json --config /tmp/volvox-e-tinyllama/snapshot/config.json --out build/llm/e-actual-model/tinyllama-generation.graph.json
python3 tools/llm_baseline/verify_tokenizer_generate.py capture --model tinyllama-1.1b-chat --policy tools/llm_baseline/models.json --snapshot /tmp/volvox-e-tinyllama/snapshot --out build/llm/e-actual-model/hf-tinyllama
python3 tools/llm_baseline/verify_tokenizer_generate.py replay --model tinyllama-1.1b-chat --policy tools/llm_baseline/models.json --snapshot /tmp/volvox-e-tinyllama/snapshot --reference build/llm/e-actual-model/hf-tinyllama --graph build/llm/e-actual-model/tinyllama-generation.graph.json --weights /tmp/volvox-e-tinyllama/package/model.safetensors --library native/libvolvoxai.so --profile full --backend cpu --out build/llm/e-actual-model/cpu-full-tinyllama.json
```

Repeat replay with `libvolvoxai-lite.so --profile inference`. Accumulate IDs and
decode the complete sequence because this decoder owns leading-space and byte
fallback context. `DecodeTokenBytes` must refuse this profile; the harness
checks refusal and subsequent tokenizer reuse. Transformers 5.13's legacy Llama
wrapper changes leading-space processing for some inputs. Its different IDs
are recorded separately; the unchanged pinned JSON remains the engine's exact
independent authority. Raw `.model` loading and Unigram remain outside this
profile. After saving receipts and hashes, remove only the newly created large
temporary assets if they are no longer needed.

Small offline reference checks need no model download:

```sh
python3 -m unittest tools.tests.test_hf_tokenizer -v
```
