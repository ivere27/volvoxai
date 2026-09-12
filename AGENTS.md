# Repository instructions for coding agents

Read `ARCHITECTURE.md` before changing source layout, build composition,
operators, backends, training, or shader generation.

Required invariants:

- `proto/volvoxai.proto` is the single public API. Every application-facing
  operation is declared there and reached through the generated dispatch. Do
  not add a public API operation without a schema change, and do not declare
  engine lifecycle functions or hand-written enums in `native/include/`.
  `make api_conformance` enforces this and runs inside the native build.
- Keep the inference entry free of every training dependency.
- Keep the native inference profile free of compiled training code and public
  training symbols.
- Do not hand-edit generated shader outputs or embedded byte arrays.
- Preserve the `VOLVOXAI_SHADER_DIR` development override; log once when an
  external override is actually used.
- Add or update numerical correctness tests for operator and backend changes.
- Run the smallest relevant static/build checks during development and both
  inference/full build checks before handoff.
- Do not combine unrelated formatting or cleanup with functional changes.

Write documentation for people: explain the product, concepts, features, and
complete workflows before transport or generated API details. Keep examples
aligned with the current API. The proto schema and generated references serve
applications and AI agents as well as people; link to them for exhaustive
contracts instead of replacing user guides with service inventories.

Release filenames are fixed unless the user explicitly changes them:

```text
dist/<package-version>/volvoxai.js
dist/<package-version>/volvoxai.min.js
dist/<package-version>/volvoxai.full.js
dist/<package-version>/volvoxai.full.min.js
dist/<package-version>/volvoxai.wasm
dist/<package-version>/volvoxai.full.wasm
native/volvoxai
native/volvoxai-full
```
