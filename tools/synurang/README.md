# Synurang GitHub release integration

`tools/generate_proto.py` uses the official
[v0.8.0 release](https://github.com/ivere27/synurang/releases/tag/v0.8.0).
The release tag resolves to commit
`53180b484cf7ca07a1e7d6f24e58b8a19a2dcfa8`.

The generator is downloaded from the release assets. Its pinned SHA-256 digest
is checked against the downloaded archive before extracting or executing it.

| Platform | Release archive SHA-256 |
| --- | --- |
| Linux x64 | `759148a4a1568f73c39d8fa127c71204226ef56a1cdab53ec0778b69e6f9db0b` |
| Linux ARM64 | `7f5196ce1eb0818d7b6993084922bb038c07d9468b931ee3ce6d0b49dffe294d` |
| Windows x64 | `07226a91ba8cbaf7cf996824612231d5d818967d974bd0700e760dc5e50617d9` |

Runtime files come from
[the v0.8.0 source archive](https://codeload.github.com/ivere27/synurang/tar.gz/refs/tags/v0.8.0),
with SHA-256
`c0c0615a824565fa4ee581ae49993a361b65d342bd334039f1411cfe7bdbd4f6`.
No source archive is stored in the repository and no Cargo build is needed.

```sh
make proto_codegen_fetch
make proto_codegen
make proto_codegen_check SYNURANG_OFFLINE=1
```

The fetch step prepares the generator and runtime-source caches under
`build/cache/synurang-codegen/`. Offline mode requires verified cached archives;
it never downloads them. Normal native/npm builds use the committed generated
files and vendored C runtime. An explicit `PROTOC_GEN_SYNURANG_FFI` executable
can be supplied on a platform without a published release binary.

C `mode=module` emits both dispatch and codecs, once per full/inference profile.
TypeScript `mode=client` emits Promise clients and codecs. Both profiles share
one call runtime and WASM adapter implementation. Python `mode=client` emits
sync/async module clients. Its vendored runtime includes only their import
closure; the generated package entry excludes unused plugin and gRPC transports.

Copied runtime implementations retain upstream behavior. The two TypeScript
`details` parameters receive explicit `Uint8Array` annotations for TS7 because
their defaults otherwise infer a narrower backing-buffer type. The Python
package entry is projected for the retained module runtime. These deterministic
transformations and the release/source/schema pins are recorded by generation.
Do not edit generated or vendored files by hand.

For an upgrade, update the release version, revision and archive digests together,
regenerate all five profiles, and run offline codegen plus inference/full build
and release checks. There are no snapshot, Cargo or callback-template fallbacks.
