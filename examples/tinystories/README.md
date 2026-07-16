# TinyStories GPT-Neo example

This directory owns the model-family policy for the TinyStories-1M example:
GPT-Neo checkpoint names and graph construction, the fixed 256-token package
shape, seed inputs, and byte-level BPE vocabulary export. None of those helpers
are imported by VolvoxAI's JavaScript entries or linked into its fixed native
executables.

Regenerate the package from the repository root:

```bash
make models_tinystories
```

That produces the existing package filenames under `models/tinystories_1m/`:

```text
config.json
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

Run its offline tests with:

```bash
python3 -m unittest discover \
  -s examples/tinystories/tests \
  -p 'test_*.py'
```

The browser presentation remains at [`../tinystories.html`](../tinystories.html),
and the opt-in native generation command lives in
[`../native_task_cli/`](../native_task_cli/).
