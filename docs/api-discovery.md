# Discovering and correcting API calls

The public contract remains `proto/volvoxai.proto`. All operations below use
its Synurang-generated `Vx*ServiceClient` Promise clients. `DescribeApi` returns methods available in the
current build; specifying a service and method keeps the response focused.
`includeTypes` adds the dependent message/enum definitions and their rules.

```js
const platform = new VxPlatformServiceClient(host);
const contract = await platform.describeApi(new pb.DescribeApiRequest({
  service: 'VxInferenceService',
  method: 'LoadModel',
  includeTypes: true,
}));
console.log(contract.methods, contract.messages);
```

Use `GetPlatformInfo` for build/transport and compiled backends, `ListBackends`
for initialized runtime providers, and `CompileModel` for model-specific
admission. A method's availability does not prove a particular graph/backend
combination is supported. Metadata `effect` describes an operation's purpose;
it is not a retry policy. `PENDING` means execution was accepted: inspect the
returned result with `GetResult` instead of repeating `Run` or `Execute`.

Query a model's logical I/O before compiling or constructing tensors:

```js
const info = await inference.getModelInfo(new pb.ModelRef(model));
console.log(info.inputs, info.outputs);
```

On a rejected input batch, inspect `error.report?.inputIssue` in a `catch` block. Its code
identifies missing/unknown/duplicate names, dtype/rank/bounds/symbol mismatches,
byte-size mismatches, unsupported memory/transport or invalid payloads. Expected
and actual tensor metadata are present when known; actual payloads are never
returned. `inputIndex` and `axis` are optional, so distinguish absent values
from index/axis zero. In the current Synurang TS projection, absent numeric
fields still read as zero; use `Object.hasOwn(issue.toJson(), "inputIndex")`
(or `"axis"`) to test presence. C uses `has_input_index` / `has_axis`.
`requiredExtent` identifies an already bound symbol or
decode extent. `expectedByteSize` identifies the exact size for a valid concrete
shape. The rejected error preserves the original decoded response in
`error.response`; TypeScript callers can narrow it with `instanceof pb.ResponseType`.
Transport/codec failures have no decoded report; their original cause is retained.

Runtime reports bound names to 255 bytes and symbols to 63 bytes. When
`namesTruncated` is true, retrieve complete names through `GetModelInfo` and use
the input index to identify the supplied tensor. An invalid native rank above
eight preserves `actualRank` and reports only the first eight shape entries.
Input conversion failures may lack an expected spec because they precede model
binding. Other failures continue to use status/code/stage and existing evidence;
absence of `inputIssue` says nothing about whether retry is safe.

Authoring/export output connects directly to inference:

```js
const model = await inference.loadModel(new pb.LoadModelRequest({
  runtimeId: runtime.runtimeId,
  package: new pb.ModelPackage({
    graphDocument: exported.source.graphDocument, // ExportGraphPlan
    weightShards: [weights.data],                 // WriteSafetensors
    // ExportTrainerWeights.shards can be passed directly as weightShards.
  }),
}));
console.log(model.modelId);
```

Use either `package` or `graphPath` with optional `weightPaths`. Mixed forms,
empty graph/shards and packages exceeding 64 MiB fail before publication.
A web host's smaller `maxPackageBytes` also applies to inline packages. Accepted
bytes are snapshotted by C, so caller buffers can be reused after the call;
compiled models and contexts retain the snapshot after the Model ID is released.
Native snapshots use private files, while WASM uses its internal VFS.

The generated [inference reference](generated/api-contract.inference.md) and
[full reference](generated/api-contract.full.md) include every projected type.
The corresponding [inference JSON](generated/api-contract.inference.json) and
[full JSON](generated/api-contract.full.json) can be indexed by tooling without
starting an engine. Both distinguish protobuf defaults from documented engine
defaults. Explicit semantic annotations supplement the ordinary proto comments;
a missing `required` flag or rule does not guarantee every omission is valid.

Regenerate with `make proto_codegen`. The local metadata generator validates
annotation keys, enum values and field paths, and the normal checks detect drift.
Synurang's Promise clients and module callbacks come from the pinned source
snapshot. Calls accept `{ signal, timeoutMs }`; native clients and providers
rebuild against the generated module headers. Domain reports and call status
remain separate typed contracts.
