# Repository instructions for coding agents

Read `ARCHITECTURE.md` before changing source layout, build composition,
operators, backends, training, or shader generation.

Required invariants:

- Keep the inference entry free of every training dependency.
- Keep the native inference profile free of compiled training code and public
  training symbols.
- Do not hand-edit generated shader outputs or embedded byte arrays.
- Preserve the `VOLVOXAI_SHADER_DIR` development override; log once when an
  external override is actually used.
- Add or update correctness tests for operator and backend changes.
- Run the smallest relevant tests during development and both inference/full
  build checks before handoff.
- Do not combine unrelated formatting or cleanup with functional changes.

Release filenames are fixed unless the user explicitly changes them:

```text
dist/<package-version>/volvoxai.js
dist/<package-version>/volvoxai.min.js
dist/<package-version>/volvoxai.full.js
dist/<package-version>/volvoxai.full.min.js
dist/<package-version>/volvoxai.wasm.js
dist/<package-version>/volvoxai.wasm.min.js
dist/<package-version>/volvoxai.wasm
dist/<package-version>/volvoxai.full.wasm
native/volvoxai
native/volvoxai-full
```
