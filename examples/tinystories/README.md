# TinyStories GPT-Neo example

This directory owns the model-family policy for the TinyStories-1M example:
GPT-Neo checkpoint names and graph construction, the bounded 256-token sequence
domain, prefill inputs, and byte-level BPE vocabulary export. None of those helpers
are imported by VolvoxAI's JavaScript entries or linked into its fixed native
executables.

Regenerate the package from the repository root:

```bash
make models_tinystories
```

That produces the existing package filenames under `models/tinystories_1m/`:

```text
graph.json
model.safetensors
vocab.bin
vocab.json
merges.txt
tokens.i32
positions.i32
```

The graph exporter can also be invoked directly:

```bash
python3 examples/tinystories/tools/export_gptneo_safetensors.py \
  --model roneneldan/TinyStories-1M \
  --out models/tinystories_1m/model.safetensors
python3 examples/tinystories/tools/export_tokenizer.py \
  roneneldan/TinyStories-1M models/tinystories_1m
```

Open [`../tinystories.html`](../tinystories.html) from a repository-root HTTP
server after building the full browser runtime. The workbench keeps prompt,
sampling controls, generated text, token predictions and model provenance in
one viewport. It shares the graph/tensor debugger and performance views with
the receipt model. The opt-in native generation command lives in
[`../native_task_cli/`](../native_task_cli/).

```bash
make build_web
python3 -m http.server 8000
# http://localhost:8000/examples/tinystories.html?auto=0&n=16&seed=42
```

The URL accepts `auto=0` to disable initial generation, `backend=wasm|webgpu`,
`prompt`, `n` (new tokens), `seed`, `temp` and `topk`. An optional `model` URL
names a package directory containing the same graph/weights/tokenizer files.
The selected backend is required and attested; a missing WebGPU route is
reported as an error. Temperature zero uses greedy sampling. Other temperatures
sample the displayed top-K distribution with the chosen seeded PRNG.

Sequence capacity comes from the graph contract and the complete prompt plus
requested continuation. An overlong prompt is rejected; generation stops at
EOS, the admitted capacity, the requested length, or **Stop**. Stop takes effect
after the current token operation completes. The token navigator selects a
prediction operation: prediction zero is prefill, and later predictions are
incremental decode steps. Its candidate scores come from that operation's real
logits; probabilities are conditional on temperature and top-K.

Choose **Debug prediction**, then **Start debug session** in the shared Debug
tab. Prefill works before generating. A selected later prediction first replays
the exact generated token prefix in a separate context, then attaches the
`Execute` debugger using the transition entrypoint and selected slot row.
Debugging does not consume the ordinary
generation context. Original-node and optimized schedules use the same backend
policy. Capture settings control retained bytes; **Decode / KV** exposes the
real slot cursors and cache descriptors, with explicit refresh and download.
Changing token, prompt, sampling settings or backend releases the previous
debug session and context. Closing the page also releases adapter resources.

**Performance & memory → Record** records an ordinary generation using the
engine profiler, with explicit prefill/decode ranges and per-prediction execution
identities. Select a phase or token operation to inspect its operators and
follow one into the debugger. Memory and overall wall time describe the complete
recording; they are not per-token allocations. Displayed latency is measured
RPC, completion wait and logits read time. Output rate and generation wall time
also include sampling and UI work. Debugging and recording run only on request.
Export run JSON to retain prompt/token IDs, settings, candidate logits and
operation identity; the shared tabs separately export diagnostic captures.

Run the focused seeded-sampling and real WASM generation/debug parity checks:

```bash
node --test examples/tinystories/tests/session.test.mjs
```

The engine check requires the full runtime and model files; it verifies repeated
seeded output, prefill and two isolated decode captures, complete predicted-logit row equality,
profile execution identities and cooperative cancellation.
