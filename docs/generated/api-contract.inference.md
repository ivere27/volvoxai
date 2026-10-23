# Generated inference API contract

Source: `proto/volvoxai.proto`. Regenerate with `make proto_codegen`.

Schema SHA-256: `89a58f15424cced76ed89f36f3b669a9826358f098df1b939751df83d3c4fc2e`.

Protobuf defaults and engine defaults are distinct. Required flags and
structured rules are explicit annotations; remaining semantic constraints
are described in the source comments and enforced by C. An absent rule
does not imply that every value or state is accepted.

`DescribeApi` returns the same contracts, filtered to the current build.
Use `GetPlatformInfo` and `ListBackends` for build/transport and runtime
backend availability; `CompileModel` decides model-specific admission.

## VxPlatformService.GetPlatformInfo

`volvoxai.v1.Empty` → `volvoxai.v1.PlatformInfo`

Effect: `API_EFFECT_READ_ONLY`.



## VxPlatformService.DescribeApi

`volvoxai.v1.DescribeApiRequest` → `volvoxai.v1.ApiDescription`

Effect: `API_EFFECT_READ_ONLY`.

Discover the methods available in this build and their schema contracts.

## VxPlatformService.GetMonotonicTime

`volvoxai.v1.Empty` → `volvoxai.v1.MonotonicTime`

Effect: `API_EFFECT_READ_ONLY`.



## VxPlatformService.DescribeStatus

`volvoxai.v1.DescribeStatusRequest` → `volvoxai.v1.StatusDescription`

Effect: `API_EFFECT_READ_ONLY`.



## VxProfilingService.StartTrace

`volvoxai.v1.StartTraceRequest` → `volvoxai.v1.TraceInfo`

Effect: `API_EFFECT_CREATE`.

Starts collection for one runtime; a runtime has at most one active trace.

- `API_RULE_KIND_LIVE_HANDLE`: `runtime_id`

## VxProfilingService.StopTrace

`volvoxai.v1.TraceRef` → `volvoxai.v1.TraceInfo`

Effect: `API_EFFECT_MUTATE`.

Stops admission and replies when the trace is READY, after accepted host
operations and pending device timestamps drain. Cancelling or timing out
this call stops waiting only; the trace keeps draining.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`

## VxProfilingService.GetTrace

`volvoxai.v1.TraceRef` → `volvoxai.v1.TraceInfo`

Effect: `API_EFFECT_READ_ONLY`.

Non-blocking status. Never stops collection.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`

## VxProfilingService.ListTraceEvents

`volvoxai.v1.ListTraceEventsRequest` → `volvoxai.v1.ListTraceEventsResponse`

Effect: `API_EFFECT_READ_ONLY`.

Events of a READY trace in sequence order. Pages are repeatable.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`
- `API_RULE_KIND_HANDLE_STATE`: `trace_id` in `TRACE_STATE_READY`

## VxProfilingService.ListTraceResourceSnapshots

`volvoxai.v1.ListTraceResourceSnapshotsRequest` → `volvoxai.v1.ListTraceResourceSnapshotsResponse`

Effect: `API_EFFECT_READ_ONLY`.

Resource snapshots of a READY trace, oldest first.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`
- `API_RULE_KIND_HANDLE_STATE`: `trace_id` in `TRACE_STATE_READY`

## VxProfilingService.GetTracePlan

`volvoxai.v1.GetTracePlanRequest` → `volvoxai.v1.GetTracePlanResponse`

Effect: `API_EFFECT_READ_ONLY`.

One execution plan of a READY trace, by TraceEvent.plan_id.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`
- `API_RULE_KIND_HANDLE_STATE`: `trace_id` in `TRACE_STATE_READY`

## VxProfilingService.ExportChromeTrace

`volvoxai.v1.ExportChromeTraceRequest` → `volvoxai.v1.ExportChromeTraceResponse`

Effect: `API_EFFECT_READ_ONLY`.

Chrome Trace Event JSON for Perfetto or chrome://tracing, one page per call.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`
- `API_RULE_KIND_HANDLE_STATE`: `trace_id` in `TRACE_STATE_READY`

## VxProfilingService.AnnotateTrace

`volvoxai.v1.AnnotateTraceRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.

Records an application-defined range in a COLLECTING trace.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`
- `API_RULE_KIND_HANDLE_STATE`: `trace_id` in `TRACE_STATE_COLLECTING`

## VxProfilingService.GetTraceSummary

`volvoxai.v1.GetTraceSummaryRequest` → `volvoxai.v1.TraceSummary`

Effect: `API_EFFECT_READ_ONLY`.

Aggregates of a READY trace: slow nodes, operators, calls or activities,
with achieved rates from plan costs. The same records ListTraceEvents returns.

- `API_RULE_KIND_LIVE_HANDLE`: `trace_id`
- `API_RULE_KIND_HANDLE_STATE`: `trace_id` in `TRACE_STATE_READY`

## VxProfilingService.ReleaseTrace

`volvoxai.v1.TraceRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.

Stops collection and retires the handle. Idempotent.

## VxProfilingService.GetResourceSnapshot

`volvoxai.v1.GetResourceSnapshotRequest` → `volvoxai.v1.ResourceSnapshotResponse`

Effect: `API_EFFECT_READ_ONLY`.

On-demand observation. Does not require or start a trace.

## VxTextService.CreateTokenizer

`volvoxai.v1.CreateTokenizerRequest` → `volvoxai.v1.TokenizerHandle`

Effect: `API_EFFECT_CREATE`.



## VxTextService.EncodeText

`volvoxai.v1.EncodeTextRequest` → `volvoxai.v1.EncodedText`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `tokenizer_id`

## VxTextService.DecodeTokens

`volvoxai.v1.DecodeTokensRequest` → `volvoxai.v1.DecodedText`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `tokenizer_id`

## VxTextService.DecodeTokenBytes

`volvoxai.v1.DecodeTokenBytesRequest` → `volvoxai.v1.DecodedTokenBytes`

Effect: `API_EFFECT_READ_ONLY`.

Lossless token-domain bytes. BYTE_LEVEL_DECODE_ONLY or HF_BYTE_LEVEL_BPE
profile required; contextual HF_SENTENCEPIECE_BPE uses DecodeTokens.
Unknown IDs fail the entire request, including when special tokens are skipped.

- `API_RULE_KIND_LIVE_HANDLE`: `tokenizer_id`

## VxTextService.ReleaseTokenizer

`volvoxai.v1.TokenizerRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.CreateRuntime

`volvoxai.v1.CreateRuntimeRequest` → `volvoxai.v1.RuntimeHandle`

Effect: `API_EFFECT_CREATE`.



## VxInferenceService.ReleaseRuntime

`volvoxai.v1.RuntimeRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.ListBackends

`volvoxai.v1.RuntimeRef` → `volvoxai.v1.BackendList`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `runtime_id`

## VxInferenceService.LoadModel

`volvoxai.v1.LoadModelRequest` → `volvoxai.v1.ModelHandle`

Effect: `API_EFFECT_CREATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `runtime_id`

## VxInferenceService.GetModelInfo

`volvoxai.v1.ModelRef` → `volvoxai.v1.ModelInfo`

Effect: `API_EFFECT_READ_ONLY`.

Inspect the immutable logical input/output contract without compiling or
executing the model. Symbol names, bounds and divisibility are preserved.

- `API_RULE_KIND_LIVE_HANDLE`: `model_id`

## VxInferenceService.GetModelRevision

`volvoxai.v1.ModelRef` → `volvoxai.v1.RevisionInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `model_id`

## VxInferenceService.PublishAdapter

`volvoxai.v1.PublishAdapterRequest` → `volvoxai.v1.AdapterRevision`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `model_id`

## VxInferenceService.ReleaseModel

`volvoxai.v1.ModelRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.CompileModel

`volvoxai.v1.CompileModelRequest` → `volvoxai.v1.CompiledModelHandle`

Effect: `API_EFFECT_CREATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `model_id`

## VxInferenceService.ReleaseCompiledModel

`volvoxai.v1.CompiledModelRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.CreateExecutionContext

`volvoxai.v1.CreateExecutionContextRequest` → `volvoxai.v1.ExecutionContextHandle`

Effect: `API_EFFECT_CREATE`.

Context execution mode and resource limits are fixed at creation.

- `API_RULE_KIND_LIVE_HANDLE`: `compiled_model_id`

## VxInferenceService.GetExecutionContextInfo

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.ExecutionContextInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.CancelExecutionContext

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.

Cooperative cancellation of the logical context, including queued work.
Already published buffers retain their independent lifetime.

- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.ReleaseExecutionContext

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxInferenceService.GetInputAffineQuantization

`volvoxai.v1.GetInputAffineQuantizationRequest` → `volvoxai.v1.AffineQuantization`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.GetTensorInteropInfo

`volvoxai.v1.ExecutionContextRef` → `volvoxai.v1.TensorInteropInfo`

Effect: `API_EFFECT_READ_ONLY`.

Query backend representation and producer ordering. No address is returned.

- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.Execute

`volvoxai.v1.ExecuteRequest` → `volvoxai.v1.ExecuteResponse`

Effect: `API_EFFECT_EXECUTE`.

Completes graph execution and device work before returning independent
selected outputs. Only Graph contexts accept this operation.

- `API_RULE_KIND_LIVE_HANDLE`: `context_id`
- `API_RULE_KIND_COMPLETE_INPUT_BATCH`: `inputs`

## VxInferenceService.Generate

`volvoxai.v1.GenerateRequest` → `volvoxai.v1.GenerateResponse`

Effect: `API_EFFECT_EXECUTE`.

Runs a bounded chunk of categorical ID generation. Only autoregressive
contexts accept this operation; initial_inputs is required exactly once.

- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.PublishContextPrefix

`volvoxai.v1.PublishContextPrefixRequest` → `volvoxai.v1.ContextCacheState`

Effect: `API_EFFECT_MUTATE`.

Prefix and slot ownership is separate from graph/generation execution.
Cache configuration is fixed by CreateExecutionContext.limits.cache.

- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.ReuseContextPrefix

`volvoxai.v1.ReuseContextPrefixRequest` → `volvoxai.v1.ContextCacheState`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.ReleaseContextSlot

`volvoxai.v1.ContextSlotRef` → `volvoxai.v1.ContextCacheState`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxInferenceService.EvictContextPrefixes

`volvoxai.v1.EvictContextPrefixesRequest` → `volvoxai.v1.ContextCacheState`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `context_id`

## VxSchedulerService.Submit

`volvoxai.v1.SubmitRequest` → `volvoxai.v1.RequestHandle`

Effect: `API_EFFECT_CREATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `execute.context_id`, `generate.context_id`
- `API_RULE_KIND_SCHEDULED_RUNTIME`: `execute.context_id`, `generate.context_id`

## VxSchedulerService.PollRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.RequestInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `request_id`

## VxSchedulerService.WaitRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.RequestInfo`

Effect: `API_EFFECT_READ_ONLY`.

Completes asynchronously when the request becomes terminal. A call deadline
or cancellation ends only this wait; CancelRequest cancels engine work.

- `API_RULE_KIND_LIVE_HANDLE`: `request_id`

## VxSchedulerService.CancelRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `request_id`

## VxSchedulerService.TakeRequestResult

`volvoxai.v1.RequestRef` → `volvoxai.v1.ExecutionPacket`

Effect: `API_EFFECT_MUTATE`.

Transfers the completed packet exactly once. Repeated take is NOT_FOUND.
Caller-owned outputs outlive the request; release them with ReleaseBuffers.

- `API_RULE_KIND_LIVE_HANDLE`: `request_id`

## VxSchedulerService.ReleaseRequest

`volvoxai.v1.RequestRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxSchedulerService.CreateBatchQueue

`volvoxai.v1.CreateBatchQueueRequest` → `volvoxai.v1.BatchQueueHandle`

Effect: `API_EFFECT_CREATE`.



## VxSchedulerService.SubmitBatchWork

`volvoxai.v1.SubmitBatchWorkRequest` → `volvoxai.v1.BatchWorkHandle`

Effect: `API_EFFECT_CREATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.NextBatchDispatch

`volvoxai.v1.BatchQueueRef` → `volvoxai.v1.BatchDispatch`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.CompleteBatchDispatch

`volvoxai.v1.CompleteBatchDispatchRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.GetBatchWork

`volvoxai.v1.BatchWorkRef` → `volvoxai.v1.BatchWorkInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.TakeBatchWork

`volvoxai.v1.BatchWorkRef` → `volvoxai.v1.BatchWorkInfo`

Effect: `API_EFFECT_MUTATE`.

Copies a terminal value/result packet and removes it from queue retention.
Returns BUSY for active work and NOT_FOUND after an earlier take/eviction.

- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.GetBatchQueue

`volvoxai.v1.BatchQueueRef` → `volvoxai.v1.BatchQueueInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.CancelBatchWork

`volvoxai.v1.BatchWorkRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.CloseBatchQueue

`volvoxai.v1.CloseBatchQueueRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_MUTATE`.



- `API_RULE_KIND_LIVE_HANDLE`: `queue_id`

## VxSchedulerService.ReleaseBatchQueue

`volvoxai.v1.BatchQueueRef` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxBufferService.GetBufferInfo

`volvoxai.v1.BufferHandle` → `volvoxai.v1.BufferInfo`

Effect: `API_EFFECT_READ_ONLY`.



- `API_RULE_KIND_LIVE_HANDLE`: `buffer_id`

## VxBufferService.AllocateBuffers

`volvoxai.v1.AllocateBuffersRequest` → `volvoxai.v1.BufferHandles`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.RetainBuffers

`volvoxai.v1.BufferRefs` → `volvoxai.v1.BufferHandles`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.ReleaseBuffers

`volvoxai.v1.BufferRefs` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxBufferService.CopyTensors

`volvoxai.v1.CopyTensorsRequest` → `volvoxai.v1.TensorBatch`

Effect: `API_EFFECT_EXECUTE`.



## VxBufferService.BeginBufferAccess

`volvoxai.v1.BufferAccessRequest` → `volvoxai.v1.BufferAccess`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.EndBufferAccess

`volvoxai.v1.EndBufferAccessRequest` → `volvoxai.v1.OperationReport`

Effect: `API_EFFECT_RELEASE`.



## VxBufferService.ImportDLPack

`volvoxai.v1.ImportDLPackRequest` → `volvoxai.v1.TensorBatch`

Effect: `API_EFFECT_CREATE`.



## VxBufferService.ExportDLPack

`volvoxai.v1.ExportDLPackRequest` → `volvoxai.v1.DLPackExport`

Effect: `API_EFFECT_CREATE`.



## volvoxai.v1.VocabularyToken



### id (1)

`uint32`.

Protobuf default: `0`.

### value (2)

`bytes`.

Protobuf default: `""`.

## volvoxai.v1.TokenizerVocabulary



### tokens (1)

`volvoxai.v1.VocabularyToken` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ByteLevelVocabularyToken

Encoded HF ByteLevel vocabulary strings, not predecoded raw bytes. The engine
inverts the fixed byte alphabet for ordinary tokens. Added tokens contribute
their literal UTF-8 content; only entries marked special are skippable.

### id (1)

`uint32`.

Protobuf default: `0`.

### token (2)

`string`.

Protobuf default: `""`.

### added (3)

`bool`.

Protobuf default: `false`.

### special (4)

`bool`.

Protobuf default: `false`.

Requires added=true.

## volvoxai.v1.ByteLevelTokenizerVocabulary



### tokens (1)

`volvoxai.v1.ByteLevelVocabularyToken` repeated.

Protobuf default: `[]`.

## volvoxai.v1.CreateTokenizerRequest



### tokens (1)

`volvoxai.v1.TokenizerVocabulary`; oneof `vocabulary`.

Protobuf default: `null`.

### vocabulary_json (2)

`bytes`; oneof `vocabulary`.

Protobuf default: `""`.

### vocabulary_binary (3)

`bytes`; oneof `vocabulary`.

Protobuf default: `""`.

### byte_level (5)

`volvoxai.v1.ByteLevelTokenizerVocabulary`; oneof `vocabulary`.

Protobuf default: `null`.

Complete immutable byte decoding for the supplied IDs. Does not implement
HF normalization, pretokenization, BPE encoding or chat templates.

### hugging_face_tokenizer_json (6)

`bytes`; oneof `vocabulary`.

Protobuf default: `""`.

Complete Hugging Face tokenizer.json definition (UTF-8, at most 32 MiB).
The engine admits supported ByteLevel or SentencePiece-style BPE models,
including their normalizer, pretokenizer, merges, added tokens and optional
single-sequence postprocessor. Unsupported definitions fail atomically.
No network access or external tokenizer runtime is used after creation.

### merges (4)

`bytes`.

Protobuf default: `""`.

UTF-8 merge lines, ordered by priority. Blank/comment lines, unresolved
pairs and repeated pairs are ignored. Leading U+0120 denotes a space.
Nonempty merges are rejected for the ByteLevel decode-only vocabulary.

## volvoxai.v1.TokenizerRef



### tokenizer_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Tokenizer`.

## volvoxai.v1.TokenizerHandle



### tokenizer_id (1)

`int64`.

Protobuf default: `"0"`.

### vocabulary_size (2)

`uint32`.

Protobuf default: `0`.

### merge_count (3)

`uint32`.

Protobuf default: `0`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### profile (5)

`volvoxai.v1.TokenizerProfile`.

Protobuf default: `"TOKENIZER_PROFILE_LEGACY"`.

### supports_encode (6)

`bool`.

Protobuf default: `false`.

### supports_decode_token_bytes (7)

`bool`.

Protobuf default: `false`.

### token_byte_table_complete (8)

`bool`.

Protobuf default: `false`.

All registered IDs have immutable engine-produced raw bytes. This does not
assert a dense/model-complete ID domain or a complete standalone alphabet.

### byte_alphabet_complete (9)

`bool`.

Protobuf default: `false`.

All 256 ordinary single-byte tokens exist.

## volvoxai.v1.EncodeTextRequest



### tokenizer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Tokenizer`.

### text (2)

`string`.

Protobuf default: `""`.

### max_tokens (3)

`uint32`.

Protobuf default: `0`.

Engine default: 256

Default 256; explicit zero returns empty.

### mode (4)

`volvoxai.v1.TokenizationMode`.

Protobuf default: `"TOKENIZATION_MODE_AUTO"`.

HF profiles require AUTO; GREEDY/BPE_WORD are legacy-only modes.

### add_special_tokens (5)

`bool`.

Protobuf default: `false`.

Engine default: true for HF profiles

HF profiles use the tokenizer's admitted single-sequence postprocessor when
true. This does not apply a chat template. Legacy profiles have no HF
postprocessor; an omitted field preserves their existing encoding behavior.

## volvoxai.v1.EncodedText



### tokens (1)

`uint32` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.DecodeTokensRequest



### tokenizer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Tokenizer`.

### tokens (2)

`uint32` repeated.

Protobuf default: `[]`.

### skip_special_tokens (3)

`bool`.

Protobuf default: `false`.

Default keeps every token. HF special entries are omitted when true;
ordinary added tokens remain literal UTF-8. Legacy skip=true is unsupported.

## volvoxai.v1.DecodedText



### text (1)

`string`.

Protobuf default: `""`.

Join token bytes, then apply UTF-8 replacement once to the whole message.
ByteLevel preserves BOM and uses no cleanup/normalization. SentencePiece BPE
uses its admitted contextual decoder over the complete sequence and does
not expose independently concatenable token bytes. Legacy removes
a leading UTF-8 BOM from each vocabulary entry before joining, replaces
U+0120 with space afterwards and keeps unknown IDs as empty placeholders.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.DecodeTokenBytesRequest



### tokenizer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Tokenizer`.

### tokens (2)

`uint32` repeated.

Protobuf default: `[]`.

### skip_special_tokens (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.TokenByteMetadata



### token_id (1)

`uint32`.

Protobuf default: `0`.

### byte_offset (2)

`uint64`.

Protobuf default: `"0"`.

Offsets refer to DecodedTokenBytes.value, in request order. Skipped special
entries have zero byte_length; duplicate IDs remain distinct observations.

### byte_length (3)

`uint64`.

Protobuf default: `"0"`.

### added (4)

`bool`.

Protobuf default: `false`.

### special (5)

`bool`.

Protobuf default: `false`.

### skipped (6)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.DecodedTokenBytes



### value (1)

`bytes`.

Protobuf default: `""`.

Raw joined bytes, including invalid/incomplete UTF-8 and NUL.

### tokens (2)

`volvoxai.v1.TokenByteMetadata` repeated.

Protobuf default: `[]`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### complete (4)

`bool`.

Protobuf default: `false`.

True only on success after every requested ID resolved. Skipping a known
special entry counts as resolved. Failure publishes no bytes or metadata.

## volvoxai.v1.BatchCacheOptions



### slots (1)

`uint32`.

Protobuf default: `0`.

### page_tokens (2)

`uint32`.

Protobuf default: `0`.

### slot_token_capacity (3)

`uint32`.

Protobuf default: `0`.

### max_pages (4)

`uint32`.

Protobuf default: `0`.

default: fully private slot capacity

### policy (5)

`volvoxai.v1.DecodeCachePolicy`.

Protobuf default: `"DECODE_CACHE_POLICY_PAGED"`.

## volvoxai.v1.CreateBatchQueueRequest



### max_queue_depth (1)

`uint32`.

Protobuf default: `0`.

default: 4 * slot count

### token_budget (2)

`uint32`.

Protobuf default: `0`.

per dispatch, including padding; default 2048

### max_slots (3)

`uint32`.

Protobuf default: `0`.

default: cache slots, or 8 without a cache

### multiple_of (4)

`uint32`.

Protobuf default: `0`.

stateless batch divisibility; default 1

### policy (5)

`volvoxai.v1.BatchQueuePolicy`.

Protobuf default: `"BATCH_QUEUE_POLICY_WORK_CONSERVING"`.

### max_wait_ns (6)

`uint64`.

Protobuf default: `"0"`.

fill-first only; rounded up to the scheduler clock tick

### queue_depth_window (7)

`uint32`.

Protobuf default: `0`.

default 1024 recent rounds

### retained_results (8)

`uint32`.

Protobuf default: `0`.

default: 2 * queue depth + slot count

### cache (9)

`volvoxai.v1.BatchCacheOptions`.

Protobuf default: `null`.

required for decode work; metadata only

## volvoxai.v1.BatchQueueRef



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

## volvoxai.v1.BatchQueueHandle



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchGroup



### model (1)

`string`.

Protobuf default: `""`.

omitted/empty defaults to "default"

### adapter_revision (2)

`string`.

Protobuf default: `""`.

absence differs from an empty revision

### shape_signature (3)

`string`.

Protobuf default: `""`.

defaults to "stateless" or "llm"

## volvoxai.v1.StatelessBatchWork



### rows (1)

`uint32`.

Protobuf default: `0`.

### inputs (2)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

copied at submission, returned unchanged to worker

## volvoxai.v1.DecodeBatchWork



### prompt_tokens (1)

`uint32`.

Protobuf default: `0`.

### max_tokens (2)

`uint32`.

Protobuf default: `0`.

### prompt_key (3)

`string`.

Protobuf default: `""`.

Context-local identity of the complete prompt semantics, including model,
adapter and positional inputs. Only complete prompt pages are published.

## volvoxai.v1.SubmitBatchWorkRequest



### queue_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### group (2)

`volvoxai.v1.BatchGroup`.

Protobuf default: `null`.

### payload (3)

`bytes`.

Protobuf default: `""`.

uninterpreted application data, copied at submission

### stateless (4)

`volvoxai.v1.StatelessBatchWork`; oneof `work`.

Protobuf default: `null`.

### decode (5)

`volvoxai.v1.DecodeBatchWork`; oneof `work`.

Protobuf default: `null`.

## volvoxai.v1.BatchWorkRef



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### work_id (2)

`uint32`.

Protobuf default: `0`.

Handle kind: `BatchWork`.

queue-scoped, never reused

## volvoxai.v1.BatchWorkHandle



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### work_id (2)

`uint32`.

Protobuf default: `0`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchDispatchWork



### work_id (1)

`uint32`.

Protobuf default: `0`.

zero for a padding item

### rows (2)

`uint32`.

Protobuf default: `0`.

### slot (3)

`sint32`.

Protobuf default: `0`.

-1 for stateless work

### slot_generation (4)

`uint32`.

Protobuf default: `0`.

### position (5)

`uint32`.

Protobuf default: `0`.

### tokens (6)

`uint32`.

Protobuf default: `0`.

### kv_length (7)

`uint32`.

Protobuf default: `0`.

### pages (8)

`sint32` repeated.

Protobuf default: `[]`.

logical-to-physical pages; -1 is unmapped

### page_tokens (9)

`uint32`.

Protobuf default: `0`.

### generated (10)

`uint32`.

Protobuf default: `0`.

### padding (11)

`bool`.

Protobuf default: `false`.

duplicates the first item's inputs; outcome is discarded

### payload (12)

`bytes`.

Protobuf default: `""`.

### inputs (13)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

## volvoxai.v1.BatchDispatch



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### dispatch_id (2)

`uint64`.

Protobuf default: `"0"`.

Zero means no work is ready. The same nonzero dispatch is returned until
completed; polling never repeats admission or a reservation. Only one
dispatch is exposed at a time. A round visits each ready group once.

### kind (3)

`volvoxai.v1.BatchWorkKind`.

Protobuf default: `"BATCH_WORK_KIND_STATELESS"`.

### group (4)

`volvoxai.v1.BatchGroup`.

Protobuf default: `null`.

### rows_per_item (5)

`uint32`.

Protobuf default: `0`.

### useful_items (6)

`uint32`.

Protobuf default: `0`.

### useful_rows (7)

`uint32`.

Protobuf default: `0`.

### total_rows (8)

`uint32`.

Protobuf default: `0`.

### work (9)

`volvoxai.v1.BatchDispatchWork` repeated.

Protobuf default: `[]`.

### round (10)

`uint64`.

Protobuf default: `"0"`.

### report (11)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchWorkOutcome



### finished (1)

`bool`.

Protobuf default: `false`.

stop decode before max_tokens; ignored for prefill

### value (2)

`bytes`.

Protobuf default: `""`.

appended to this request's value history

### outputs (3)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

copied; replaces this request's last outputs

### error (4)

`string`.

Protobuf default: `""`.

any useful item's error fails the entire dispatch

## volvoxai.v1.CompleteBatchDispatchRequest



### queue_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### dispatch_id (2)

`uint64`.

Protobuf default: `"0"`.

### outcomes (3)

`volvoxai.v1.BatchWorkOutcome` repeated.

Protobuf default: `[]`.

Exactly one outcome per dispatched item, including padding, unless error
is supplied. Invalid IDs/counts/payloads leave the pending dispatch intact.

### error (4)

`string`.

Protobuf default: `""`.

worker-wide failure; rolls back reservations

## volvoxai.v1.BatchValue



### data (1)

`bytes`.

Protobuf default: `""`.

## volvoxai.v1.BatchWorkInfo



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### work_id (2)

`uint32`.

Protobuf default: `0`.

### state (3)

`volvoxai.v1.BatchWorkState`.

Protobuf default: `"BATCH_WORK_STATE_UNKNOWN"`.

### generated (4)

`uint32`.

Protobuf default: `0`.

### slot (5)

`sint32`.

Protobuf default: `0`.

### values (6)

`volvoxai.v1.BatchValue` repeated.

Protobuf default: `[]`.

### outputs (7)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### error (8)

`string`.

Protobuf default: `""`.

### report (9)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BatchQueueInfo



### queue_id (1)

`int64`.

Protobuf default: `"0"`.

### closed (2)

`bool`.

Protobuf default: `false`.

### draining (3)

`bool`.

Protobuf default: `false`.

### pending_dispatch_id (4)

`uint64`.

Protobuf default: `"0"`.

### rounds (5)

`uint64`.

Protobuf default: `"0"`.

### dispatches (6)

`uint64`.

Protobuf default: `"0"`.

### rows_dispatched (7)

`uint64`.

Protobuf default: `"0"`.

### rows_useful (8)

`uint64`.

Protobuf default: `"0"`.

### worker_busy_ns (9)

`uint64`.

Protobuf default: `"0"`.

Time from first exposure to completion, summed over nonoverlapping worker
dispatches. This measures worker turnaround, not GPU hardware utilization.

### wall_ns (10)

`uint64`.

Protobuf default: `"0"`.

### worker_utilization (11)

`double`.

Protobuf default: `0`.

### padding_waste (12)

`double`.

Protobuf default: `0`.

### queue_depth (13)

`uint32`.

Protobuf default: `0`.

### max_queue_depth_seen (14)

`uint32`.

Protobuf default: `0`.

### queue_depth_p50 (15)

`uint32`.

Protobuf default: `0`.

### queue_depth_p99 (16)

`uint32`.

Protobuf default: `0`.

### group_count (17)

`uint32`.

Protobuf default: `0`.

### active_slots (18)

`uint32`.

Protobuf default: `0`.

### free_slots (19)

`uint32`.

Protobuf default: `0`.

### admitted (20)

`uint64`.

Protobuf default: `"0"`.

### completed (21)

`uint64`.

Protobuf default: `"0"`.

### cancelled (22)

`uint64`.

Protobuf default: `"0"`.

### failed (23)

`uint64`.

Protobuf default: `"0"`.

### steps (24)

`uint64`.

Protobuf default: `"0"`.

### admission_stalls (25)

`uint64`.

Protobuf default: `"0"`.

### queue_delay_rounds (26)

`uint64`.

Protobuf default: `"0"`.

### prefix_reuses (27)

`uint64`.

Protobuf default: `"0"`.

### prefix_publications (28)

`uint64`.

Protobuf default: `"0"`.

### prefill_tokens (29)

`uint64`.

Protobuf default: `"0"`.

### shared_prompt_tokens (30)

`uint64`.

Protobuf default: `"0"`.

### active_records (31)

`uint32`.

Protobuf default: `0`.

### retained_results (32)

`uint32`.

Protobuf default: `0`.

### max_retained_results (33)

`uint32`.

Protobuf default: `0`.

### resident_pages (34)

`uint32`.

Protobuf default: `0`.

### reserved_pages (35)

`uint32`.

Protobuf default: `0`.

### free_pages (36)

`uint32`.

Protobuf default: `0`.

### shared_pages (37)

`uint32`.

Protobuf default: `0`.

### evictions (38)

`uint64`.

Protobuf default: `"0"`.

### report (39)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CloseBatchQueueRequest



### queue_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `BatchQueue`.

### drain (2)

`bool`.

Protobuf default: `false`.

Stops admission and cancels queued work. With drain, already admitted work
remains available through Next/Complete until terminal. Without drain all
pending work is cancelled and dispatch IDs are revoked. Release always
cancels and frees the policy owner; worker device resources remain its own.

## volvoxai.v1.Empty



## volvoxai.v1.RuntimeRef



### runtime_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Runtime`.

## volvoxai.v1.ModelRef



### model_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Model`.

## volvoxai.v1.CompiledModelRef



### compiled_model_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `CompiledModel`.

## volvoxai.v1.ExecutionContextRef



### context_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

## volvoxai.v1.RequestRef



### request_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `Request`.

## volvoxai.v1.BufferHandle

An owner-scoped capability, never a pointer. Copying an ID does not retain it.
RetainBuffers returns fresh IDs; ReleaseBuffers retires IDs idempotently.

### buffer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Buffer`.

## volvoxai.v1.BufferRefs



### buffer_ids (1)

`int64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.BufferHandles



### buffers (1)

`volvoxai.v1.BufferHandle` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BufferView

A range within the public logical extent, not allocation capacity.

### buffer_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Buffer`.

### offset_bytes (2)

`uint64`.

Protobuf default: `"0"`.

### length_bytes (3)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.NativeResource

Local transport only. handle is a host/CUDA pointer, VkBuffer, GLuint, or
id<MTLBuffer>. device_context identifies the owning VkDevice/EGLContext/
MTLDevice; CUDA validates its primary context. These are not wire identities.

### kind (1)

`volvoxai.v1.NativeResourceKind`.

Protobuf default: `"NATIVE_RESOURCE_KIND_UNSPECIFIED"`.

### handle (2)

`uint64`.

Protobuf default: `"0"`.

### size_bytes (3)

`uint64`.

Protobuf default: `"0"`.

### device_id (4)

`int32`.

Protobuf default: `0`.

### device_context (5)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.BorrowedBuffer

Caller-owned local memory. All accesses finish before native dispatch
returns. The caller keeps memory valid and unchanged during the call.
Browser dispatch rejects this descriptor before dereferencing. A remote
native adapter must reject local descriptors before forwarding the call.
Offsets apply to resource contents, never to an opaque object address.

### resource (1)

`volvoxai.v1.NativeResource`.

Protobuf default: `null`.

### offset_bytes (2)

`uint64`.

Protobuf default: `"0"`.

### length_bytes (3)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.Tensor

A dense tensor. name is a binding label, not storage identity. It is ignored
by storage operations, and required by model input and output bindings.
Storage ownership belongs exclusively to buffer.buffer_id.

- `API_RULE_KIND_EXACTLY_ONE`: `inline`, `buffer`, `borrowed`

### name (1)

`string`.

Protobuf default: `""`.

### shape (2)

`int64` repeated.

Protobuf default: `[]`.

### dtype (3)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### inline (5)

`bytes`; oneof `payload`.

Protobuf default: `""`.

### buffer (6)

`volvoxai.v1.BufferView`; oneof `payload`.

Protobuf default: `null`.

### borrowed (7)

`volvoxai.v1.BorrowedBuffer`; oneof `payload`.

Protobuf default: `null`.

## volvoxai.v1.TensorBatch

Outputs are independent snapshots unless explicitly documented as shared.

### outputs (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BufferInfo



### size_bytes (1)

`uint64`.

Protobuf default: `"0"`.

### kind (2)

`volvoxai.v1.NativeResourceKind`.

Protobuf default: `"NATIVE_RESOURCE_KIND_UNSPECIFIED"`.

### device_id (3)

`int32`.

Protobuf default: `0`.

### device_context (4)

`uint64`.

Protobuf default: `"0"`.

### cpu_accessible (5)

`bool`.

Protobuf default: `false`.

### read_only (6)

`bool`.

Protobuf default: `false`.

### report (7)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.AllocateBuffersRequest

Allocates zero-initialized CPU storage. Device snapshots are produced by
execution; allocation on arbitrary GPU devices is not yet supported.

### sizes_bytes (1)

`uint64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.CopyTensorsRequest

Produces independent CPU snapshots or explicit host readback. inline_result
returns portable bytes; into specifies one local host destination per source.
Both options cannot be set together. The default returns owned buffer IDs.
Batches validate every source and destination before any destination write.

### sources (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### into (2)

`volvoxai.v1.BorrowedBuffer` repeated.

Protobuf default: `[]`.

### inline_result (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.BufferAccessRequest

Access is an explicit lease. Conflicting writes return BUSY. host_mapping
requires direct CPU access and never introduces a hidden staging copy.

### view (1)

`volvoxai.v1.BufferView`.

Protobuf default: `null`.

### mode (2)

`volvoxai.v1.BufferAccessMode`.

Protobuf default: `"BUFFER_ACCESS_MODE_READ"`.

### host_mapping (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.BufferAccess



### access_id (1)

`int64`.

Protobuf default: `"0"`.

### memory (2)

`volvoxai.v1.BorrowedBuffer`.

Protobuf default: `null`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.CudaStreamCompletion

Local-native CUDA completion evidence. All accesses must already be queued
on these 1..64 streams, on the allocation's device and primary context.
Keep every stream alive through the call. Values are live CUstream handles,
or 1 for the legacy default stream. Zero and the per-thread stream (2) are
rejected: dispatch may run on another host thread. Capturing streams are
unsupported. Arbitrary integers are not safe substitutes for live handles.

### device_id (1)

`int32`.

Protobuf default: `0`.

### streams (2)

`uint64` repeated.

Protobuf default: `[]`.

## volvoxai.v1.EndBufferAccessRequest

Retires the access permission; all views bound to it must stop being used.
With cuda, C records stream completion without a host wait and defers the
dependency until this allocation is accessed/reused. Final destruction or
shutdown may wait for recorded work. The caller must declare every stream
that accessed the range, and must enqueue no further accesses. At most 64
distinct pending consumer streams are retained per allocation; exceeding
that budget returns BUSY and leaves the access permission active.
Without cuda, external work must already be complete; native CUDA also drains
the context conservatively. Retired IDs are harmless. Validation failure
leaves live IDs active; a driver failure can add ordering dependencies but
does not retire IDs. CUDA evidence is rejected on WASM/remote transports.

### access_ids (1)

`int64` repeated.

Protobuf default: `[]`.

### cuda (2)

`volvoxai.v1.CudaStreamCompletion`.

Protobuf default: `null`.

## volvoxai.v1.ImportDLPackRequest

A local pointer to a standard DLManagedTensor or DLManagedTensorVersioned.
A successful import consumes its deleter exactly once, on final release.
Failure leaves ownership with the caller. Producer synchronization must be
complete before import. Dense CPU and CUDA tensors are supported.

### managed_tensor (1)

`uint64`.

Protobuf default: `"0"`.

### versioned (2)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ExportDLPackRequest



### tensor (1)

`volvoxai.v1.Tensor`.

Protobuf default: `null`.

### versioned (2)

`bool`.

Protobuf default: `false`.

### access_id (3)

`int64`.

Protobuf default: `"0"`.

Optional existing writable BeginBufferAccess lease covering this tensor.
EndBufferAccess ends its permission even if DLPack aliases remain alive;
those aliases only retain allocation lifetime and must no longer be used.
Zero creates the ordinary lease whose permission ends at the deleter.

## volvoxai.v1.DLPackExport

The standard DLPack deleter releases the allocation reference and, for an
ordinary export, its writable permission. Scoped permission ends explicitly
via EndBufferAccess. Public IDs may retire earlier. No autograd edge is created.
Keep the native library loaded until every exported deleter has run.

### managed_tensor (1)

`uint64`.

Protobuf default: `"0"`.

### versioned (2)

`bool`.

Protobuf default: `false`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.DimensionConstraint



### kind (1)

`volvoxai.v1.DimensionKind`.

Protobuf default: `"DIMENSION_KIND_UNSPECIFIED"`.

### symbol (2)

`string`.

Protobuf default: `""`.

### min (3)

`int64`.

Protobuf default: `"0"`.

### max (4)

`int64`.

Protobuf default: `"0"`.

### multiple_of (5)

`int64`.

Protobuf default: `"0"`.

## volvoxai.v1.TensorSpec

Logical model contract. Concrete request shapes are carried only by Tensor.
No byte size is reported because a symbolic tensor has many concrete byte
sizes within its bounded domain.

### name (1)

`string`.

Protobuf default: `""`.

### dtype (2)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### dimensions (3)

`volvoxai.v1.DimensionConstraint` repeated.

Protobuf default: `[]`.

### location (4)

`volvoxai.v1.MemoryLocation`.

Protobuf default: `"MEMORY_LOCATION_HOST"`.

## volvoxai.v1.TensorInfo



### name (1)

`string`.

Protobuf default: `""`.

### shape (2)

`int64` repeated.

Protobuf default: `[]`.

### dtype (3)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### byte_size (4)

`uint64`.

Protobuf default: `"0"`.

### location (5)

`volvoxai.v1.MemoryLocation`.

Protobuf default: `"MEMORY_LOCATION_HOST"`.

## volvoxai.v1.TensorInteropInfo



### backend (1)

`string`.

Protobuf default: `""`.

### device (2)

`volvoxai.v1.NativeResource`.

Protobuf default: `null`.

Kind and device identity; handle and size_bytes are zero.

### consumer_stream (3)

`int64`.

Protobuf default: `"0"`.

CUDA legacy default stream (DLPack stream value 1); zero otherwise.
Borrowed producers order writes on this stream before Execute.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.AffineQuantization

Resolved per-tensor affine metadata. Numeric values are loaded from the
safetensors tensors referenced by graph.quantization.tensors; they are not
graph JSON parameters.

### defined (1)

`bool`.

Protobuf default: `false`.

### scale (2)

`float`.

Protobuf default: `0`.

### zero_point (3)

`int32`.

Protobuf default: `0`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.NodeRef



### index (1)

`int32`.

Protobuf default: `0`.

### op (2)

`string`.

Protobuf default: `""`.

### output (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.CompilationCandidate



### backend (1)

`string`.

Protobuf default: `""`.

### outcome (2)

`volvoxai.v1.CandidateOutcome`.

Protobuf default: `"CANDIDATE_OUTCOME_UNSPECIFIED"`.

### status (3)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

### reason (4)

`string`.

Protobuf default: `""`.

## volvoxai.v1.CompilationEvidence



### policy_mode (1)

`volvoxai.v1.BackendPolicyMode`.

Protobuf default: `"BACKEND_POLICY_MODE_PREFER"`.

### operator_fallback (2)

`volvoxai.v1.OperatorFallback`.

Protobuf default: `"OPERATOR_FALLBACK_ALLOW"`.

### candidates (3)

`volvoxai.v1.CompilationCandidate` repeated.

Protobuf default: `[]`.

### tier_fallback_used (4)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ShapePlanEvidence

Dynamic-shape binding evidence for one committed shape signature.

### signature_digest (1)

`fixed64`.

Protobuf default: `"0"`.

### signature (2)

`string`.

Protobuf default: `""`.

### origin (3)

`volvoxai.v1.ShapePlanOrigin`.

Protobuf default: `"SHAPE_PLAN_ORIGIN_UNSPECIFIED"`.

### bind_time_ns (4)

`uint64`.

Protobuf default: `"0"`.

### logical_bytes (5)

`uint64`.

Protobuf default: `"0"`.

### arena_required_bytes (6)

`uint64`.

Protobuf default: `"0"`.

### arena_capacity_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### arena_high_water_bytes (8)

`uint64`.

Protobuf default: `"0"`.

### arena_grow_count (9)

`uint64`.

Protobuf default: `"0"`.

### resource_generation (10)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.RouteEvidence

Which backend actually executed. `attested` is false when any active node
lacked a proved route; a consumer must not read the selected backend as
execution proof without it.

### provider (1)

`string`.

Protobuf default: `""`.

### builtin (2)

`bool`.

Protobuf default: `false`.

### attested (3)

`bool`.

Protobuf default: `false`.

### active_nodes (4)

`uint32`.

Protobuf default: `0`.

### selected_nodes (5)

`uint32`.

Protobuf default: `0`.

### fallback_nodes (6)

`uint32`.

Protobuf default: `0`.

### missing_nodes (7)

`uint32`.

Protobuf default: `0`.

### digest (8)

`fixed64`.

Protobuf default: `"0"`.

### shape_plan (9)

`volvoxai.v1.ShapePlanEvidence`.

Protobuf default: `null`.

## volvoxai.v1.FallbackEvidence



### operator_fallback_used (1)

`bool`.

Protobuf default: `false`.

### count (2)

`uint32`.

Protobuf default: `0`.

### first (3)

`volvoxai.v1.NodeRef`.

Protobuf default: `null`.

## volvoxai.v1.DecodeState



### enabled (1)

`bool`.

Protobuf default: `false`.

### prefilled (2)

`bool`.

Protobuf default: `false`.

### mode (3)

`volvoxai.v1.DecodeMode`.

Protobuf default: `"DECODE_MODE_UNSPECIFIED"`.

### last_position (4)

`int32`.

Protobuf default: `0`.

Absent before the first successful step.

## volvoxai.v1.Lineage

Immutable identity of the graph, weight, and adapter generations an
operation ran against.

### execution_id (1)

`uint64`.

Protobuf default: `"0"`.

### runtime_id (2)

`uint64`.

Protobuf default: `"0"`.

### model_id (3)

`uint64`.

Protobuf default: `"0"`.

### compiled_model_id (4)

`uint64`.

Protobuf default: `"0"`.

### context_id (5)

`uint64`.

Protobuf default: `"0"`.

### graph_id (6)

`uint64`.

Protobuf default: `"0"`.

### graph_revision (7)

`uint64`.

Protobuf default: `"0"`.

### weight_id (8)

`uint64`.

Protobuf default: `"0"`.

### weight_revision (9)

`uint64`.

Protobuf default: `"0"`.

### adapter_id (10)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (11)

`uint64`.

Protobuf default: `"0"`.

### graph_plan_id (12)

`uint64`.

Protobuf default: `"0"`.

Nonzero when a planning operation creates or names a positive GraphPlan
id. A stale positive reference is echoed for diagnosis; a constructor
failure before handle allocation and SafeTensors inspection leave zero.
A model-derived plan also carries the retained Model revision above; a
standalone authoring plan leaves those model fields zero.

## volvoxai.v1.OperationReport



### status (1)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

### stage (2)

`volvoxai.v1.OperationStage`.

Protobuf default: `"OPERATION_STAGE_NONE"`.

### code (3)

`volvoxai.v1.OperationCode`.

Protobuf default: `"OPERATION_CODE_NONE"`.

Machine-readable detail, independent of message wording.

### message (4)

`string`.

Protobuf default: `""`.

### offending_node (5)

`volvoxai.v1.NodeRef`.

Protobuf default: `null`.

### backend (6)

`string`.

Protobuf default: `""`.

### device (7)

`string`.

Protobuf default: `""`.

### lineage (8)

`volvoxai.v1.Lineage`.

Protobuf default: `null`.

### compilation (11)

`volvoxai.v1.CompilationEvidence`.

Protobuf default: `null`.

### route (12)

`volvoxai.v1.RouteEvidence`.

Protobuf default: `null`.

### fallback (13)

`volvoxai.v1.FallbackEvidence`.

Protobuf default: `null`.

### decode (14)

`volvoxai.v1.DecodeState`.

Protobuf default: `null`.

### input_issue (16)

`volvoxai.v1.InputValidationIssue`.

Protobuf default: `null`.

First input violation, reported before the input batch changes state.
Absent for failures unrelated to input validation. Never infer retry
safety for other failures from the absence of this field.

## volvoxai.v1.InputValidationIssue



### code (1)

`volvoxai.v1.InputValidationCode`.

Protobuf default: `"INPUT_VALIDATION_CODE_UNSPECIFIED"`.

### input_name (2)

`string`.

Protobuf default: `""`.

### input_index (3)

`uint64`.

Protobuf default: `"0"`.

Zero-based index into the supplied batch; absent for a missing input.

### axis (4)

`uint32`.

Protobuf default: `0`.

### expected (5)

`volvoxai.v1.TensorSpec`.

Protobuf default: `null`.

Present when a declared input can be identified.

### actual (6)

`volvoxai.v1.TensorInfo`.

Protobuf default: `null`.

Metadata only; no tensor payload is echoed. Absent for missing inputs.

### expected_byte_size (7)

`uint64`.

Protobuf default: `"0"`.

### required_extent (8)

`int64`.

Protobuf default: `"0"`.

Previously bound extent for symbol equality or decode shape continuity.

### names_truncated (9)

`bool`.

Protobuf default: `false`.

Runtime evidence bounds names to 255 bytes and symbols to 63 bytes.
Query GetModelInfo for the complete names when this flag is true.

### actual_rank (10)

`uint64`.

Protobuf default: `"0"`.

Actual rank is preserved even for an invalid native binding exceeding
max_tensor_rank. actual.shape then contains only the first eight axes.

## volvoxai.v1.PlatformInfo



### api_version (1)

`uint32`.

Protobuf default: `0`.

### library_version (2)

`string`.

Protobuf default: `""`.

### build_profile (3)

`volvoxai.v1.BuildProfile`.

Protobuf default: `"BUILD_PROFILE_UNSPECIFIED"`.

### transport (4)

`volvoxai.v1.TransportProfile`.

Protobuf default: `"TRANSPORT_PROFILE_UNSPECIFIED"`.

### compiled_backends (5)

`string` repeated.

Protobuf default: `[]`.

Backends compiled into this library, in canonical lower-case spelling.
Availability at runtime is reported by VxInferenceService.ListBackends.

### max_tensor_rank (6)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.DescribeApiRequest

Empty filters return the method catalogue for this build. include_types
adds only the transitive request/response types of the selected methods.
A method filter requires a service (e.g. VxInferenceService / Execute).
Unknown or unavailable names return NOT_FOUND, never an empty success.

- `API_RULE_KIND_REQUIRES`: `method`, `service`

### service (1)

`string`.

Protobuf default: `""`.

### method (2)

`string`.

Protobuf default: `""`.

### include_types (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ApiRule

Fields are proto field paths, relative to the annotated request/message.
REQUIRES means the first non-default/present field requires the remaining
fields. EXACTLY_ONE counts non-default/present fields (including messages).
These discoverable rules supplement the C validator; graph-dependent
validity and backend admission are still decided by the operation.

### kind (1)

`volvoxai.v1.ApiRuleKind`.

Protobuf default: `"API_RULE_KIND_UNSPECIFIED"`.

### fields (2)

`string` repeated.

Protobuf default: `[]`.

### description (3)

`string`.

Protobuf default: `""`.

### states (4)

`string` repeated.

Protobuf default: `[]`.

HANDLE_STATE only: accepted state enum value names.

## volvoxai.v1.ApiField



### name (1)

`string`.

Protobuf default: `""`.

### number (2)

`uint32`.

Protobuf default: `0`.

### type (3)

`string`.

Protobuf default: `""`.

Scalar protobuf spelling, or a fully-qualified message/enum name.

### repeated (4)

`bool`.

Protobuf default: `false`.

### has_presence (5)

`bool`.

Protobuf default: `false`.

### oneof (6)

`string`.

Protobuf default: `""`.

### description (7)

`string`.

Protobuf default: `""`.

### required (8)

`bool`.

Protobuf default: `false`.

True only for explicitly documented semantic requirements. False is
not proof that omission is valid for every operation using this type.

### proto_default (9)

`string`.

Protobuf default: `""`.

Protobuf default (JSON spelling), distinct from an engine default.

### engine_default (10)

`string`.

Protobuf default: `""`.

Documented engine behavior when omitted/zero, not a codec substitution.

### handle_kind (11)

`string`.

Protobuf default: `""`.

### rules (12)

`volvoxai.v1.ApiRule` repeated.

Protobuf default: `[]`.

### minimum (13)

`string`.

Protobuf default: `""`.

Inclusive numeric bounds (JSON Schema minimum/maximum), in decimal. Empty
when unbounded. Checked by the engine for the documented operations.

### maximum (14)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ApiMessage



### name (1)

`string`.

Protobuf default: `""`.

### description (2)

`string`.

Protobuf default: `""`.

### fields (3)

`volvoxai.v1.ApiField` repeated.

Protobuf default: `[]`.

### rules (4)

`volvoxai.v1.ApiRule` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ApiEnumValue



### name (1)

`string`.

Protobuf default: `""`.

### number (2)

`int32`.

Protobuf default: `0`.

### description (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ApiEnum



### name (1)

`string`.

Protobuf default: `""`.

### description (2)

`string`.

Protobuf default: `""`.

### values (3)

`volvoxai.v1.ApiEnumValue` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ApiMethod



### service (1)

`string`.

Protobuf default: `""`.

### name (2)

`string`.

Protobuf default: `""`.

### request_type (3)

`string`.

Protobuf default: `""`.

### response_type (4)

`string`.

Protobuf default: `""`.

### description (5)

`string`.

Protobuf default: `""`.

### effect (6)

`volvoxai.v1.ApiEffect`.

Protobuf default: `"API_EFFECT_UNSPECIFIED"`.

### rules (7)

`volvoxai.v1.ApiRule` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ApiDescription



### methods (1)

`volvoxai.v1.ApiMethod` repeated.

Protobuf default: `[]`.

### messages (2)

`volvoxai.v1.ApiMessage` repeated.

Protobuf default: `[]`.

### enums (3)

`volvoxai.v1.ApiEnum` repeated.

Protobuf default: `[]`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### schema_sha256 (5)

`string`.

Protobuf default: `""`.

## volvoxai.v1.BackendList



### backends (1)

`string` repeated.

Protobuf default: `[]`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.MonotonicTime

Current CLOCK_MONOTONIC reading in nanoseconds. Submit deadlines use this
clock domain. The representation does not imply nanosecond clock precision.

### nanoseconds (1)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.DescribeStatusRequest



### status (1)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

## volvoxai.v1.StatusDescription



### status (1)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

### name (2)

`string`.

Protobuf default: `""`.

### description (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.RuntimeBudget

Queued, in-flight, and caller-retained output batches share these Runtime-wide
bounds. Admission precharges the selected output sum and shrinks that charge
to the completed snapshot bytes. Inline storage returns its reservation after
serialization/discard; owned bytes remain charged through the final storage
lease and the batch slot through the final owned output release.

Every field is optional: an unset field takes the engine default. Defaults
are 64 scheduled requests, 64 MiB scheduled input bytes, zero batch delay,
64 unconsumed results, and 64 MiB unconsumed result bytes.

### max_scheduled_requests (1)

`uint64`.

Protobuf default: `"0"`.

Engine default: 64

### max_scheduled_input_bytes (2)

`uint64`.

Protobuf default: `"0"`.

Engine default: 67108864 bytes

### max_batch_delay_ns (3)

`uint64`.

Protobuf default: `"0"`.

Engine default: 0

Maximum coalescing delay for scheduled requests. Zero dispatches
immediately; deadlines may shorten a nonzero delay.
Rounded up to the scheduler's millisecond coalescing tick; at most
4294967295000000 ns. Representation and scheduling precision are distinct.

### max_unconsumed_results (4)

`uint64`.

Protobuf default: `"0"`.

Engine default: 64

### max_unconsumed_result_bytes (5)

`uint64`.

Protobuf default: `"0"`.

Engine default: 67108864 bytes

## volvoxai.v1.CreateRuntimeRequest



### debug (1)

`bool`.

Protobuf default: `false`.

### cpu_threads (2)

`int32`.

Protobuf default: `0`.

Engine default: 0 selects the engine default thread count.

### execution_mode (3)

`volvoxai.v1.ExecutionMode`.

Protobuf default: `"EXECUTION_MODE_DIRECT"`.

Engine default: EXECUTION_MODE_DIRECT

### budget (4)

`volvoxai.v1.RuntimeBudget`.

Protobuf default: `null`.

Engine default: All RuntimeBudget engine defaults.

Absence takes every engine default.

## volvoxai.v1.RuntimeHandle



### runtime_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BankResidency

Which slots of a declared weight bank to materialize.

Residency is part of the immutable loaded Model source. Every compiled
context owns a private engine and descriptor table, borrows the compiled
immutable weight blobs, and materializes the same selected rows in a
context-owned copy-on-write overlay. Slot ids are ascending, unique, and
index the bank's current declared extent, not its dimension's append
capacity. Route indices stay in that global slot space at execution, so a
model loaded with a subset still routes by the ids the exporter emitted.
Each present entry names a nonempty strict subset; omit a bank to make every
currently declared slot resident. Duplicate bank entries are invalid.

### bank (1)

`string`.

Protobuf default: `""`.

Weight tensor named by the graph document's "banks" table.

### slots (2)

`uint32` repeated.

Protobuf default: `[]`.

## volvoxai.v1.LoadModelRequest

graph_path names graph.json or a named *.graph.json document. The loader
requires its root discriminator to be exactly "volvox-graph/v1" and never
searches alternate filenames. weight_paths are retained in order. A bank
left out of bank_residency is fully resident.
Use exactly one of graph_path, graph_document or package. weight_paths may
accompany either graph source, but cannot accompany an inline ModelPackage.

- `API_RULE_KIND_EXACTLY_ONE`: `graph_path`, `graph_document`, `package`

### runtime_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Runtime`.

### graph_path (2)

`string`.

Protobuf default: `""`.

### weight_paths (3)

`string` repeated.

Protobuf default: `[]`.

### bank_residency (4)

`volvoxai.v1.BankResidency` repeated.

Protobuf default: `[]`.

Engine default: All declared slots are resident.

### package (5)

`volvoxai.v1.ModelPackage`.

Protobuf default: `null`.

### source_options (6)

`volvoxai.v1.ModelSourceOptions`.

Protobuf default: `null`.

Optional source manifest and byte budgets. Required to opt into the
browser source profile above 64 MiB; cannot accompany inline ModelPackage.
C validates the manifest before browser fetch or snapshot publication.

### graph_document (7)

`bytes`.

Protobuf default: `""`.

Inline volvox-graph/v1 document with optional original weight_paths.
Contains 1..67,108,864 bytes. The loader snapshots these exact bytes;
source_options.graph, when present, describes this document, and totals
include it alongside the ordered path shards. No graph path is required.

## volvoxai.v1.ModelSourceIntegrity

Integrity of the complete decoded source body, after HTTP decompression.
C checks SHA-256 against the accepted immutable snapshot. A mismatch
publishes no model handle and releases snapshots from the failed load.

### byte_length (1)

`uint64`; required.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

### sha256 (2)

`bytes`; required.

Protobuf default: `""`.

Exactly 32 raw digest bytes; hexadecimal text is not a digest.

## volvoxai.v1.ModelSourceOptions

One graph descriptor and one descriptor per ordered weight path are required.
With graph_document, the graph descriptor hashes those exact inline bytes.
max_shard_bytes also bounds the graph; totals include all source files.
Native path loads without this message keep their filesystem profile.
Browser path loads without it retain the ordinary 64 MiB source budget.
Browser hosts may impose a tighter maxPackageBytes. The WASM32 source
profile admits at most 1 GiB, subject to allocation and device limits.
Source budgets do not reserve prepared weights, activations, execution
scratch or device memory; those costs have separate admission checks.
Large browser sources require streaming bodies. The host forwards bounded
chunks and transfers uniquely staged VFS storage to immutable snapshots.
Accepted descendants retain those snapshots after ReleaseModel.

### max_total_bytes (1)

`uint64`; required.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

### max_shard_bytes (2)

`uint64`; required.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

### graph (3)

`volvoxai.v1.ModelSourceIntegrity`; required.

Protobuf default: `null`.

### weight_shards (4)

`volvoxai.v1.ModelSourceIntegrity` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ModelPackage

Portable source for authoring/export -> inference. The loader snapshots all
bytes before publishing the Model; callers may reuse their buffers after
the call. Descendants retain the accepted snapshot after ReleaseModel.
graph_document uses volvox-graph/v1; shards are ordered SafeTensors files.
Total graph + shard bytes must be <= 64 MiB on every transport. A web host
may impose a tighter maxPackageBytes. No path resolution or fetch occurs.

### graph_document (1)

`bytes`; required.

Protobuf default: `""`.

### weight_shards (2)

`bytes` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ModelInfo



### model_id (1)

`int64`.

Protobuf default: `"0"`.

### inputs (2)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### outputs (3)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### execution (5)

`volvoxai.v1.ModelExecutionContract`.

Protobuf default: `null`.

## volvoxai.v1.ModelHandle



### model_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.RevisionInfo



### graph_id (1)

`uint64`.

Protobuf default: `"0"`.

### graph_revision (2)

`uint64`.

Protobuf default: `"0"`.

### weight_id (3)

`uint64`.

Protobuf default: `"0"`.

### weight_revision (4)

`uint64`.

Protobuf default: `"0"`.

### adapter_id (5)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (6)

`uint64`.

Protobuf default: `"0"`.

### report (7)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.PublishAdapterRequest

Publishing another package under the same adapter_name creates the next
immutable revision of that adapter.

### model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Model`.

### adapter_name (2)

`string`.

Protobuf default: `""`.

### package_path (3)

`string`.

Protobuf default: `""`.

Empty publishes a metadata-only route revision, which is useful to
providers that own adapter storage.

### version_name (4)

`string`.

Protobuf default: `""`.

Empty uses the package/provider version.

## volvoxai.v1.AdapterRevision



### adapter_id (1)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (2)

`uint64`.

Protobuf default: `"0"`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.BackendPolicy



### mode (1)

`volvoxai.v1.BackendPolicyMode`.

Protobuf default: `"BACKEND_POLICY_MODE_PREFER"`.

### backends (2)

`string` repeated.

Protobuf default: `[]`.

Engine default: The host's default ordered backend policy.

PREFER tries this ordered list. REQUIRE accepts exactly one entry. An
empty list selects the host's default ordered backend policy. A populated
list has at most 16 unique names. Each name is 1..63 ASCII characters,
begins with a lower-case letter, and then contains only lower-case letters,
digits, dot, underscore, or dash.

### operator_fallback (3)

`volvoxai.v1.OperatorFallback`.

Protobuf default: `"OPERATOR_FALLBACK_ALLOW"`.

## volvoxai.v1.CompileModelRequest



### model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Model`.

### policy (2)

`volvoxai.v1.BackendPolicy`.

Protobuf default: `null`.

### preserve_node_boundaries (3)

`bool`.

Protobuf default: `false`.

Compile a separate executable that preserves source node boundaries for
debugging. Disables fusion and alias elimination, not arena reuse. The
selected backend must support this graph; there is no implicit CPU route.

### adapter_revision (4)

`volvoxai.v1.AdapterRevisionRef`.

Protobuf default: `null`.

Absent selects the immutable base revision. Present pins one exact
adapter revision published by this Model; contexts cannot rebind it.

### packed_arithmetic (5)

`volvoxai.v1.PackedArithmetic`.

Protobuf default: `"PACKED_ARITHMETIC_UNSPECIFIED"`.

## volvoxai.v1.CompiledModelHandle



### compiled_model_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### metrics (3)

`volvoxai.v1.CompilationMetrics`.

Protobuf default: `null`.

### memory_bounds (4)

`volvoxai.v1.MemoryDomainAttestation`.

Protobuf default: `null`.

### preserves_node_boundaries (5)

`bool`.

Protobuf default: `false`.

### packed_arithmetic (6)

`volvoxai.v1.PackedArithmetic`.

Protobuf default: `"PACKED_ARITHMETIC_UNSPECIFIED"`.

Resolved immutable arithmetic; UNSPECIFIED is never returned on success.

## volvoxai.v1.ExecutionEntrypoint



### name (1)

`string`.

Protobuf default: `""`.

### effect (2)

`volvoxai.v1.ExecutionEffect`.

Protobuf default: `"EXECUTION_EFFECT_UNSPECIFIED"`.

### allowed_ranges (3)

`volvoxai.v1.ExecutionRangeKind` repeated.

Protobuf default: `[]`.

### inputs (4)

`string` repeated.

Protobuf default: `[]`.

### outputs (5)

`string` repeated.

Protobuf default: `[]`.

## volvoxai.v1.IncrementalStateContract



### active_lengths_input (1)

`string`.

Protobuf default: `""`.

Required I32[slots] graph input carrying positive initialized lengths.
This is an explicit control binding, never inferred from padding/shape.

### position_input (2)

`string`.

Protobuf default: `""`.

Optional graph input carrying the model's position values.

### sequence_axis (3)

`uint32`.

Protobuf default: `0`.

### update_inputs (4)

`string` repeated.

Protobuf default: `[]`.

Input roots refreshed by row transitions, including retained values.

## volvoxai.v1.TensorFeedbackStateContract



### bindings (1)

`volvoxai.v1.TensorFeedback` repeated.

Protobuf default: `[]`.

### position_input (2)

`string`.

Protobuf default: `""`.

At least one explicit position/active-length control binding is required.

### active_lengths_input (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.FullPrefixGenerationContract

Re-evaluate the complete growing prefix for every selected ID. This profile
has no persistent attention/recurrent cache and requires a stateless graph
with EVALUATE/FULL initialization and transition entrypoints. Initial inputs
contain exactly the active prompt, with no inactive/padded suffix. All
unlisted inputs must be fixed-shape invariant_inputs retained from that
first call. Every growing dimension shares token_input's sequence symbol.
Initial bindings must be host-readable. CPU/WASM/CUDA execute the complete
graph; CUDA samples/selects on device and transfers the selected ID for
prefix construction. Chunk IDs, including owned tensors, use host storage.

### position_input (1)

`string`.

Protobuf default: `""`.

Required I32[S]; retain the prompt values and append the last value + 1.

### sequence_input (2)

`string`.

Protobuf default: `""`.

Optional I32[S], required to equal arange(S); extend it with the new index.

### chunk_mask_input (3)

`string`.

Protobuf default: `""`.

Optional I32[1,S,S] keep mask: one exactly when query/key are in the
same chunk. Causality remains the attention operator's declared policy.

### chunk_size (4)

`uint32`.

Protobuf default: `0`.

Positive iff chunk_mask_input is present.

### zero_append_inputs (5)

`string` repeated.

Protobuf default: `[]`.

Optional I32[S] or I32[1,S] controls; retain the prompt and append zero.

### multimodal_position_input (6)

`string`.

Protobuf default: `""`.

Optional I32[3,S]; append one plus the maximum existing value to all
three axes, preserving multimodal prompt positions and text continuation.

## volvoxai.v1.IncrementalGenerationContract

Additional feedback controls for the incremental row profile. Inputs have
the admitted fixed sequence capacity; initialization consumes only the
prefix selected by incremental.active_lengths_input. Each transition writes
the newly active row without changing or recomputing earlier token rows.
Unlisted conditioning inputs must be declared invariant_inputs.

### position_input (1)

`string`.

Protobuf default: `""`.

Required I32[S]; append the last active position + 1.

### sequence_input (2)

`string`.

Protobuf default: `""`.

Optional I32[S]; write the new absolute sequence index.

### chunk_mask_input (3)

`string`.

Protobuf default: `""`.

Optional I32[1,S,S] mask; update the new query/key chunk memberships.

### chunk_size (4)

`uint32`.

Protobuf default: `0`.

Positive iff chunk_mask_input is present.

### zero_append_inputs (5)

`string` repeated.

Protobuf default: `[]`.

Optional I32[S] or I32[1,S] routing controls; append zero.

### multimodal_position_input (6)

`string`.

Protobuf default: `""`.

Optional I32[3,S]; append one plus the maximum active position across
the three axes, preserving prepared multimodal prompt positions.

## volvoxai.v1.GenerationContract



### initialize_entrypoint (1)

`string`.

Protobuf default: `""`.

### transition_entrypoint (2)

`string`.

Protobuf default: `""`.

### token_input (3)

`string`.

Protobuf default: `""`.

### token_output (4)

`string`.

Protobuf default: `""`.

Graph ArgMax/QArgMax I32 output used for engine feedback.

### keep_input (5)

`string`.

Protobuf default: `""`.

### logits_output (6)

`string`.

Protobuf default: `""`.

### logits_axis (7)

`uint32`.

Protobuf default: `0`.

### id_domain_size (8)

`uint32`.

Protobuf default: `0`.

IDs are in [0, id_domain_size).

### output_name (9)

`string`.

Protobuf default: `""`.

Public chunk output; phase one is I32[1,T], sequence_axis=1.

### sequence_axis (10)

`uint32`.

Protobuf default: `0`.

### full_prefix (11)

`volvoxai.v1.FullPrefixGenerationContract`.

Protobuf default: `null`.

Presence selects the explicit whole-prefix profile. Absence retains the
incremental row profile and its state/cache contract.

### incremental_controls (12)

`volvoxai.v1.IncrementalGenerationContract`.

Protobuf default: `null`.

Optional controls for the incremental state/cache profile. Mutually
exclusive with full_prefix; absence preserves the original dense profile.

## volvoxai.v1.ModelExecutionContract



- `API_RULE_KIND_EXACTLY_ONE`: `stateless`, `incremental`, `tensor_feedback`

### entrypoints (1)

`volvoxai.v1.ExecutionEntrypoint` repeated.

Protobuf default: `[]`.

### invariant_inputs (2)

`string` repeated.

Protobuf default: `[]`.

### stateless (3)

`volvoxai.v1.Empty`; oneof `state`.

Protobuf default: `null`.

### incremental (4)

`volvoxai.v1.IncrementalStateContract`; oneof `state`.

Protobuf default: `null`.

### tensor_feedback (5)

`volvoxai.v1.TensorFeedbackStateContract`; oneof `state`.

Protobuf default: `null`.

### generation (6)

`volvoxai.v1.GenerationContract`.

Protobuf default: `null`.

## volvoxai.v1.GraphExecution



### row_mode (1)

`volvoxai.v1.DecodeRowMode`.

Protobuf default: `"DECODE_ROW_MODE_DISABLED"`.

### require_incremental (2)

`bool`.

Protobuf default: `false`.

### slots (3)

`uint32`.

Protobuf default: `0`.

Engine default: 1

Range: `[1, ∞]`.

## volvoxai.v1.IdStopSequence



### ids (1)

`int32` repeated.

Protobuf default: `[]`.

## volvoxai.v1.AutoregressiveExecution



### sampling (1)

`volvoxai.v1.SamplingMode`; required.

Protobuf default: `"SAMPLING_MODE_UNSPECIFIED"`.

Explicit GREEDY or RANDOM is required. RANDOM requires F32 categorical
logits with at most 1,048,576 categories and an admitted CPU/WASM/CUDA
sampling route. Admitted full-prefix GREEDY routes also support EOS,
stop IDs and ID sequences on Vulkan, OpenGL, Metal and WebGPU.

### max_steps (2)

`uint64`; required.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

Total newly selected IDs, excluding conditioning/prefill.

### seed (3)

`uint64`.

Protobuf default: `"0"`.

RANDOM defaults to seed zero. Reproducibility is scoped to the same model,
input, backend and configuration, independent of Generate chunk boundaries.
The RNG advances once per newly selected ID. GREEDY rejects a seed.

### stop_ids (4)

`int32` repeated.

Protobuf default: `[]`.

Terminal packets include the matching generated IDs. Prompt IDs do not
participate. Up to 4096 IDs total across stop_ids, eos_ids and stop_sequences;
at most 128 nonempty sequences. Matching can span Generate calls.

### stop_sequences (5)

`volvoxai.v1.IdStopSequence` repeated.

Protobuf default: `[]`.

### temperature (6)

`float`.

Protobuf default: `0`.

Engine default: 1

RANDOM-only options. Temperature must be finite and strictly positive.
Explicit temperature zero is invalid; use GREEDY.

### top_k (7)

`uint32`.

Protobuf default: `0`.

Engine default: 0

Zero selects the entire categorical domain; positive values retain the
highest scoring IDs, ordered by score then ascending ID for ties.

### top_p (8)

`float`.

Protobuf default: `0`.

Engine default: 1

Range: `[-∞, 1]`.

Retain the smallest descending-score prefix whose probability reaches this
threshold, including the crossing ID, after top-k filtering. Must be finite
and strictly positive, at most one.

### eos_ids (9)

`int32` repeated.

Protobuf default: `[]`.

EOS takes precedence over ID_STOP when both match the same generated ID.

## volvoxai.v1.ExecutionLimits



### max_memory_bytes (1)

`uint64`.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

Context state/cache/scratch, excluding weights and retained outputs.
Absence uses the admitted backend/runtime resource domain.

### cache (2)

`volvoxai.v1.ContextCacheOptions`.

Protobuf default: `null`.

## volvoxai.v1.CreateExecutionContextRequest



- `API_RULE_KIND_EXACTLY_ONE`: `graph`, `autoregressive`

### compiled_model_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `CompiledModel`.

### graph (2)

`volvoxai.v1.GraphExecution`; oneof `execution`.

Protobuf default: `null`.

### autoregressive (3)

`volvoxai.v1.AutoregressiveExecution`; oneof `execution`.

Protobuf default: `null`.

### limits (4)

`volvoxai.v1.ExecutionLimits`.

Protobuf default: `null`.

## volvoxai.v1.ExecutionContextHandle



### context_id (1)

`int64`.

Protobuf default: `"0"`.

### inputs (2)

`volvoxai.v1.TensorSpec` repeated.

Protobuf default: `[]`.

### report (3)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.ExecutionContextInfo



### context_id (1)

`int64`.

Protobuf default: `"0"`.

### state (2)

`volvoxai.v1.ExecutionContextState`.

Protobuf default: `"EXECUTION_CONTEXT_STATE_UNSPECIFIED"`.

### initialized (3)

`bool`.

Protobuf default: `false`.

### total_steps (4)

`uint64`.

Protobuf default: `"0"`.

### stop_reason (5)

`volvoxai.v1.ExecutionStopReason`.

Protobuf default: `"EXECUTION_STOP_REASON_UNSPECIFIED"`.

### slots (6)

`uint32`.

Protobuf default: `0`.

### active_lengths (7)

`uint32` repeated.

Protobuf default: `[]`.

### empty (8)

`bool` repeated.

Protobuf default: `[]`.

### cache_generation (9)

`uint64`.

Protobuf default: `"0"`.

### mode (10)

`volvoxai.v1.DecodeMode`.

Protobuf default: `"DECODE_MODE_UNSPECIFIED"`.

### report (11)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### cache (12)

`volvoxai.v1.ContextCacheState`.

Protobuf default: `null`.

### bindings_available (13)

`bool`.

Protobuf default: `false`.

False initially and after an executed stateless request is discarded.

## volvoxai.v1.GetInputAffineQuantizationRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### name (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.TensorOutputSelection



### names (1)

`string` repeated.

Protobuf default: `[]`.

Empty explicitly selects no outputs. Names must be unique public outputs
of the selected entrypoint, or of the model's generation contract.

## volvoxai.v1.TensorFeedback



### input_name (1)

`string`.

Protobuf default: `""`.

### output_name (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.InputBindings



### values (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### reuse (2)

`string` repeated.

Protobuf default: `[]`.

### feedback (3)

`volvoxai.v1.TensorFeedback` repeated.

Protobuf default: `[]`.

## volvoxai.v1.PrefixRows



### count (1)

`uint32`; required.

Protobuf default: `0`.

Range: `[1, ∞]`.

## volvoxai.v1.SlotRow



- `API_RULE_KIND_EXACTLY_ONE`: `position`, `recompute`, `empty`

### position (1)

`uint32`; oneof `action`.

Protobuf default: `0`.

Must equal this slot's next active position; zero is a valid position.

### recompute (2)

`volvoxai.v1.Empty`; oneof `action`.

Protobuf default: `null`.

### empty (3)

`volvoxai.v1.Empty`; oneof `action`.

Protobuf default: `null`.

## volvoxai.v1.SlotRows



### slots (1)

`volvoxai.v1.SlotRow` repeated.

Protobuf default: `[]`.

## volvoxai.v1.DependencyClosure



### roots (1)

`string` repeated.

Protobuf default: `[]`.

Includes every changed input root. Empty requires no changed binding.

## volvoxai.v1.ExecutionScope



- `API_RULE_KIND_EXACTLY_ONE`: `full`, `prefix_rows`, `slot_rows`, `dependencies`

### entrypoint (1)

`string`; required.

Protobuf default: `""`.

### full (2)

`volvoxai.v1.Empty`; oneof `range`.

Protobuf default: `null`.

### prefix_rows (3)

`volvoxai.v1.PrefixRows`; oneof `range`.

Protobuf default: `null`.

### slot_rows (4)

`volvoxai.v1.SlotRows`; oneof `range`.

Protobuf default: `null`.

### dependencies (5)

`volvoxai.v1.DependencyClosure`; oneof `range`.

Protobuf default: `null`.

## volvoxai.v1.ExecutionBudget



### max_output_bytes (1)

`uint64`.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

Selected output payload only. Absent uses the runtime's available output
budget; present values must be positive, also for an empty selection.

### deadline_monotonic_ns (2)

`uint64`.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

Absolute VxPlatformService monotonic nanoseconds. A present zero value
is invalid. Before commit: refuse unchanged; after commit: FAILED and no
output handoff. Unsupported clocks/deadlines are refused at admission.

## volvoxai.v1.GenerationBudget



### max_steps (1)

`uint32`; required.

Protobuf default: `0`.

Range: `[1, ∞]`.

New IDs in this chunk, excluding prefill; must be positive.

### max_output_bytes (2)

`uint64`.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

### deadline_monotonic_ns (3)

`uint64`.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

Same absolute clock and failure semantics as ExecutionBudget.

## volvoxai.v1.OutputOptions



### selection (1)

`volvoxai.v1.TensorOutputSelection`.

Protobuf default: `null`.

Absent selects all public outputs; present empty selects none.

### storage (2)

`volvoxai.v1.OutputStorage`.

Protobuf default: `"OUTPUT_STORAGE_UNSPECIFIED"`.

Engine default: OUTPUT_STORAGE_INLINE

UNSPECIFIED defaults to INLINE. OWNED_BUFFER requires support from this
transport/backend and keeps an independent snapshot on its backend.

## volvoxai.v1.ExecuteRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### inputs (2)

`volvoxai.v1.InputBindings`; required.

Protobuf default: `null`.

Values, reuse and feedback bind every entrypoint input exactly once.

### budget (3)

`volvoxai.v1.ExecutionBudget`.

Protobuf default: `null`.

### outputs (4)

`volvoxai.v1.OutputOptions`.

Protobuf default: `null`.

### scope (5)

`volvoxai.v1.ExecutionScope`; required.

Protobuf default: `null`.

## volvoxai.v1.GenerationInputs



### values (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

## volvoxai.v1.GenerateRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### initial_inputs (2)

`volvoxai.v1.GenerationInputs`.

Protobuf default: `null`.

Required before initialization, including an explicitly empty message;
forbidden afterward. List emptiness does not select an execution mode.

### budget (3)

`volvoxai.v1.GenerationBudget`; required.

Protobuf default: `null`.

### outputs (4)

`volvoxai.v1.OutputOptions`.

Protobuf default: `null`.

## volvoxai.v1.ExecutionProgress



### state (1)

`volvoxai.v1.ExecutionContextState`.

Protobuf default: `"EXECUTION_CONTEXT_STATE_UNSPECIFIED"`.

### steps (2)

`uint64`.

Protobuf default: `"0"`.

Newly committed categorical IDs in this call; zero for ordinary graph work.

### total_steps (3)

`uint64`.

Protobuf default: `"0"`.

### stop_reason (4)

`volvoxai.v1.ExecutionStopReason`.

Protobuf default: `"EXECUTION_STOP_REASON_UNSPECIFIED"`.

A context terminal reason; chunk budget exhaustion leaves it unspecified.

### can_continue (5)

`bool`.

Protobuf default: `false`.

### initialized (6)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ExecuteResponse

Every tensor is complete and independently owned. Release buffer outputs
with ReleaseBuffers. Failure or cancellation transfers no graph outputs.

### outputs (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### progress (2)

`volvoxai.v1.ExecutionProgress`.

Protobuf default: `null`.

### metrics (3)

`volvoxai.v1.ExecutionMetrics`.

Protobuf default: `null`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### execution_id (5)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.GenerateResponse

Failure transfers no outputs. Cancellation may return already committed
generation IDs with a successful report and terminal CANCELLED progress.

### outputs (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### progress (2)

`volvoxai.v1.ExecutionProgress`.

Protobuf default: `null`.

### metrics (3)

`volvoxai.v1.ExecutionMetrics`.

Protobuf default: `null`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### execution_id (5)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.ExecutionPacket

Scheduler response envelope; it adds no execution operation or lifecycle.

### outputs (1)

`volvoxai.v1.Tensor` repeated.

Protobuf default: `[]`.

### progress (2)

`volvoxai.v1.ExecutionProgress`.

Protobuf default: `null`.

### metrics (3)

`volvoxai.v1.ExecutionMetrics`.

Protobuf default: `null`.

### report (4)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### execution_id (5)

`uint64`.

Protobuf default: `"0"`.

### kind (6)

`volvoxai.v1.ExecutionKind`.

Protobuf default: `"EXECUTION_KIND_UNSPECIFIED"`.

## volvoxai.v1.ContextCacheOptions

Fixed at context creation. Materialized atomically by initialization;
allocation/admission failure preserves the uninitialized context.

### tensors (1)

`string` repeated.

Protobuf default: `[]`.

### page_tokens (2)

`uint32`.

Protobuf default: `0`.

### max_pages (3)

`uint32`.

Protobuf default: `0`.

Omitted uses all complete physical pages in the admitted tensor pool.

### policy (4)

`volvoxai.v1.DecodeCachePolicy`.

Protobuf default: `"DECODE_CACHE_POLICY_PAGED"`.

### clear_on_recycle (5)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ContextSlotRef



### context_id (1)

`int64`.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### slot (2)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.PublishContextPrefixRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### key (2)

`string`.

Protobuf default: `""`.

Context-local identity. Include every input value affecting the prefix.

### slot (3)

`uint32`.

Protobuf default: `0`.

### tokens (4)

`uint32`.

Protobuf default: `0`.

Must end at a complete page. Published pages remain immutable through COW.

## volvoxai.v1.ReuseContextPrefixRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### key (2)

`string`.

Protobuf default: `""`.

### slot (3)

`uint32`.

Protobuf default: `0`.

Must be empty, for example after ReleaseContextSlot. C restores the saved
decoder-input/output prefix as well as shared K/V pages. Other slots and
independently retained output buffers remain unchanged.

## volvoxai.v1.EvictContextPrefixesRequest



### context_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `ExecutionContext`.

### free_pages (2)

`uint32`.

Protobuf default: `0`.

Reclaim unused prefixes in LRU order until this many pages are free.

## volvoxai.v1.ContextCacheSlot



### active_length (1)

`uint32`.

Protobuf default: `0`.

### generation (2)

`uint32`.

Protobuf default: `0`.

### pages (3)

`int32` repeated.

Protobuf default: `[]`.

Logical page -> physical page; -1 means unmapped.

## volvoxai.v1.ContextCacheState



### configured (1)

`bool`.

Protobuf default: `false`.

### policy (2)

`volvoxai.v1.DecodeCachePolicy`.

Protobuf default: `"DECODE_CACHE_POLICY_PAGED"`.

### page_tokens (3)

`uint32`.

Protobuf default: `0`.

### slot_token_capacity (4)

`uint32`.

Protobuf default: `0`.

### max_pages (5)

`uint32`.

Protobuf default: `0`.

### slots (6)

`volvoxai.v1.ContextCacheSlot` repeated.

Protobuf default: `[]`.

### page_generations (7)

`uint32` repeated.

Protobuf default: `[]`.

### tensors (8)

`string` repeated.

Protobuf default: `[]`.

### logical_bytes (9)

`uint64`.

Protobuf default: `"0"`.

Logical/resident counts describe page use, not GPU allocations. Pool
storage stays allocated until the context closes; snapshots are separate.

### resident_bytes (10)

`uint64`.

Protobuf default: `"0"`.

### reserved_bytes (11)

`uint64`.

Protobuf default: `"0"`.

### pool_bytes (12)

`uint64`.

Protobuf default: `"0"`.

### prefix_snapshot_bytes (13)

`uint64`.

Protobuf default: `"0"`.

### resident_pages (14)

`uint32`.

Protobuf default: `0`.

### reserved_pages (15)

`uint32`.

Protobuf default: `0`.

### free_pages (16)

`uint32`.

Protobuf default: `0`.

### shared_pages (17)

`uint32`.

Protobuf default: `0`.

### fragmentation_bytes (18)

`uint64`.

Protobuf default: `"0"`.

### high_water_pages (19)

`uint32`.

Protobuf default: `0`.

### prefix_hits (20)

`uint64`.

Protobuf default: `"0"`.

### prefix_misses (21)

`uint64`.

Protobuf default: `"0"`.

### copy_on_writes (22)

`uint64`.

Protobuf default: `"0"`.

### evictions (23)

`uint64`.

Protobuf default: `"0"`.

### allocation_failures (24)

`uint64`.

Protobuf default: `"0"`.

### cache_generation (25)

`uint64`.

Protobuf default: `"0"`.

### report (26)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.AdapterRevisionRef



### adapter_id (1)

`uint64`.

Protobuf default: `"0"`.

### adapter_revision (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.SubmitOptions

Priority is clamped to [-1000, 1000]. Higher values run first; queued age
raises effective priority, and equal effective priorities use
earliest-deadline-first, then request id.

### priority (1)

`int32`.

Protobuf default: `0`.

### deadline_monotonic_ns (2)

`uint64`.

Protobuf default: `"0"`.

Absolute monotonic nanoseconds in the VxPlatformService clock domain.
Rounded up to the scheduler clock tick.
Zero disables the deadline. It is always an admission/EDF target; only
DROP_IF_LATE makes it a hard queued/completion cutoff.

### freshness (3)

`volvoxai.v1.RequestFreshness`.

Protobuf default: `"REQUEST_FRESHNESS_ALL"`.

### stream_key (4)

`uint64`.

Protobuf default: `"0"`.

Required and nonzero for LATEST. Numeric identity copied by value.

## volvoxai.v1.SubmitRequest

Acceptance snapshots inputs and reserves exactly one context and the output
budget atomically. ALL accepts both operations. LATEST/DROP_IF_LATE accept
only stateless evaluate/full with complete new values, never reuse/feedback.
Refusal precedes replacing any existing request. Same-context resubmission
is BUSY; LATEST matching uses compiled route and stream key across contexts.

- `API_RULE_KIND_EXACTLY_ONE`: `execute`, `generate`

### execute (1)

`volvoxai.v1.ExecuteRequest`; oneof `execution`.

Protobuf default: `null`.

### generate (2)

`volvoxai.v1.GenerateRequest`; oneof `execution`.

Protobuf default: `null`.

### options (3)

`volvoxai.v1.SubmitOptions`.

Protobuf default: `null`.

## volvoxai.v1.RequestHandle



### request_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.RequestInfo



### request_id (1)

`int64`.

Protobuf default: `"0"`.

### state (2)

`volvoxai.v1.RequestState`.

Protobuf default: `"REQUEST_STATE_UNSPECIFIED"`.

### status (3)

`volvoxai.v1.NativeStatus`.

Protobuf default: `"NATIVE_STATUS_OK"`.

BUSY while queued or running; otherwise the terminal execution status.

### owned_input_bytes (4)

`uint64`.

Protobuf default: `"0"`.

### deadline_missed (5)

`bool`.

Protobuf default: `false`.

Set when accepted work crossed its target. ALL and LATEST still publish a
successful result; DROP_IF_LATE instead reports DEADLINE_EXCEEDED.

### report (6)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

## volvoxai.v1.MemoryOwnerRef

Opaque owner identity. Public handles use decimal spelling; process scope uses "0".

### kind (1)

`volvoxai.v1.MemoryOwnerKind`.

Protobuf default: `"MEMORY_OWNER_KIND_UNSPECIFIED"`.

### owner_id (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.MemoryByteSize

Message presence, unlike a proto3 scalar default, distinguishes an omitted
measurement from an exact zero in every generated language.

### bytes (1)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.MemoryBoundTerm

One disjoint accounting unit in a compile-time peak case. term_id is unique
within its case. Shared/aliased physical storage appears once in a case
even when multiple objects consume it.

### term_id (1)

`string`.

Protobuf default: `""`.

### owner_kind (2)

`volvoxai.v1.MemoryOwnerKind`.

Protobuf default: `"MEMORY_OWNER_KIND_UNSPECIFIED"`.

### role (3)

`volvoxai.v1.MemoryResourceRole`.

Protobuf default: `"MEMORY_RESOURCE_ROLE_UNSPECIFIED"`.

### space (4)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### upper_bound_bytes (5)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

## volvoxai.v1.MemoryPeakCase

Every term in one case is simultaneously resident and physically disjoint.
A present case contains at least one term; exact-zero cases use a present
zero-byte term. total_bytes must equal the checked sum of upper_bound_bytes.
Cases are alternatives: callers take their maximum and never sum case
totals.

### case_id (1)

`string`.

Protobuf default: `""`.

### terms (2)

`volvoxai.v1.MemoryBoundTerm` repeated.

Protobuf default: `[]`.

### total_bytes (3)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

## volvoxai.v1.MemoryBoundProof

maximum_bytes must equal max(cases.total_bytes) when cases are present.
Empty cases allow a provider to publish its existing coarse attestation
before it implements detailed terms. limit_bytes is absent when the backend
or device exposes no meaningful limit.

### budget_domain_id (1)

`string`.

Protobuf default: `""`.

Stable provider-defined accounting domain, for example
"provider-resident", "activation-capacity", or "wasm-linear". It
disambiguates independent limits with the same canonical kind.

### kind (2)

`volvoxai.v1.MemoryBoundKind`.

Protobuf default: `"MEMORY_BOUND_KIND_UNSPECIFIED"`.

### cases (3)

`volvoxai.v1.MemoryPeakCase` repeated.

Protobuf default: `[]`.

### maximum_bytes (4)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

### limit_bytes (5)

`volvoxai.v1.MemoryByteSize`.

Protobuf default: `null`.

## volvoxai.v1.MemoryDomainAttestation

Compile-time evidence for the complete bounded shape domain. Bounds are
computed proof, not observed allocation or RSS/VRAM measurements. Consumers
accept only the canonical proof/resource protocols named below and require
at least one bound. They reject missing maxima, missing case/term byte
presence, duplicate (budget_domain_id, kind), duplicate case/term IDs,
checked-sum or max mismatches, and a maximum above its present limit. Empty
cases are allowed only for a coarse maximum supplied by an existing
provider attestation.

### proof_protocol (1)

`string`.

Protobuf default: `""`.

Exactly "canonical-symbolic-domain-proof/v1" in format v1.

### resource_protocol (2)

`string`.

Protobuf default: `""`.

Exactly "bounded-resource-maxima/v1" in format v1.

### graph_fingerprint (3)

`string`.

Protobuf default: `""`.

### shape_domain_proof_identity (4)

`string`.

Protobuf default: `""`.

### bounds (5)

`volvoxai.v1.MemoryBoundProof` repeated.

Protobuf default: `[]`.

## volvoxai.v1.ExecutionMetrics



### host_time_ns (1)

`uint64`.

Protobuf default: `"0"`.

Host execution and output submission; excludes asynchronous device completion.
Includes synchronous device waits performed by the backend. Not kernel time.

### output_bytes (2)

`uint64`.

Protobuf default: `"0"`.

Logical output payload; never process RSS or total allocator residency.

## volvoxai.v1.CompilationMetrics



### host_time_ns (1)

`uint64`.

Protobuf default: `"0"`.

Host compilation duration measured with the monotonic clock.

## volvoxai.v1.RecordCount

A bounded collection: records kept and records lost to capacity or errors.

### count (1)

`uint64`.

Protobuf default: `"0"`.

### dropped (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TraceOptions

What to collect. StartTraceRequest supplies it; TraceInfo echoes the
effective options with defaults resolved.

### detail (1)

`volvoxai.v1.TraceDetail`.

Protobuf default: `"TRACE_DETAIL_BASIC"`.

One detail level for host and device observations.

### device_timing (2)

`bool`.

Protobuf default: `false`.

GPU elapsed time; the engine chooses the timestamp mechanism. False creates
no GPU timing resources. Node timing may split passes or add barriers.

### memory (3)

`bool`.

Protobuf default: `false`.

Allocation history, allocator peaks and memory in resource snapshots.

### utilization (4)

`bool`.

Protobuf default: `false`.

Process CPU time and hardware-wide GPU utilization in resource snapshots.

### execution_plans (5)

`bool`.

Protobuf default: `false`.

Immutable tensor/schedule metadata for each concrete context layout.

### capacity_bytes (6)

`uint64`.

Protobuf default: `"0"`.

Engine default: 4194304

Range: `[4096, 67108864]`.

Collector storage, allocated once at start, including event metadata.

### sample_interval_ns (7)

`uint64`.

Protobuf default: `"0"`.

Engine default: 100000000

Range: `[10000000, 60000000000]`.

Minimum spacing of resource snapshots taken at operation boundaries. There
is no sampler thread; this is not a guaranteed sampling period.

### external (8)

`volvoxai.v1.TraceExternalOptions`.

Protobuf default: `null`.

Ranges and labels for vendor profilers and GPU debuggers.

## volvoxai.v1.TraceExternalOptions

External profiler and debugger integration. Only operations admitted while
the trace is COLLECTING are annotated; ordinary execution never is.
Numerical results and backend selection are unchanged.

### annotations (1)

`bool`.

Protobuf default: `false`.

NVTX ranges for operations, and with detail NODES for nodes and training
phases (native builds with a GPU backend; Nsight Systems/Compute), plus a
label around each node's device commands: VK_EXT_debug_utils (Vulkan),
KHR_debug groups (OpenGL) and debug groups in labeled passes (WebGPU) for
RenderDoc, vendor tools and WebGPU validation messages.

### capture (2)

`bool`.

Protobuf default: `false`.

Ask a RenderDoc instance already injected into the process to capture from
the first traced operation until StopTrace. Headless compute has no present
call to delimit a frame. A missing tool is reported, never an error.

## volvoxai.v1.TraceExternalCoverage

What one vendor-tool mechanism did for this trace. AVAILABLE means the engine
emitted through it; it does not prove that a tool recorded anything.

### mechanism (1)

`string`.

Protobuf default: `""`.

nvtx, vk_debug_utils, khr_debug, webgpu_debug_group or renderdoc.

### backend (2)

`string`.

Protobuf default: `""`.

### status (3)

`volvoxai.v1.ObservationStatus`.

Protobuf default: `"OBSERVATION_STATUS_UNSPECIFIED"`.

### ranges (4)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.StartTraceRequest



### runtime_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Runtime`.

### options (2)

`volvoxai.v1.TraceOptions`.

Protobuf default: `null`.

## volvoxai.v1.TraceRef



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

## volvoxai.v1.TraceDeviceIntervalCounts

Device intervals successfully recorded for one backend, by scope.

### passes (1)

`uint64`.

Protobuf default: `"0"`.

### nodes (2)

`uint64`.

Protobuf default: `"0"`.

### programs (3)

`uint64`.

Protobuf default: `"0"`.

Engine-owned dispatches, not a complete driver-level kernel inventory.

### copies (4)

`uint64`.

Protobuf default: `"0"`.

### calibrated (5)

`uint64`.

Protobuf default: `"0"`.

Intervals placed on the host clock by calibration or by causal bounds.

### bounded (6)

`uint64`.

Protobuf default: `"0"`.

### failed (7)

`uint64`.

Protobuf default: `"0"`.

### unsupported_passes (8)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TraceHostActivityCounts

Host activity observations for one backend.

### copies (1)

`uint64`.

Protobuf default: `"0"`.

### submits (2)

`uint64`.

Protobuf default: `"0"`.

### synchronizations (3)

`uint64`.

Protobuf default: `"0"`.

Blocking host calls.

### completions (4)

`uint64`.

Protobuf default: `"0"`.

Asynchronous completions; not CPU busy time.

## volvoxai.v1.TraceDeviceCoverage

One row per encountered built-in backend. StartTrace probes no device.

### backend (1)

`string`.

Protobuf default: `""`.

### support (2)

`volvoxai.v1.TraceTimingSupport`.

Protobuf default: `"TRACE_TIMING_SUPPORT_UNSPECIFIED"`.

### node_timing_available (3)

`bool`.

Protobuf default: `false`.

Observed on at least one encountered device path; not a promise that every
operation has node/program coverage.

### program_timing_available (4)

`bool`.

Protobuf default: `false`.

### splits_passes (5)

`bool`.

Protobuf default: `false`.

Instrumentation changes that may alter the measured execution schedule.

### adds_barriers (6)

`bool`.

Protobuf default: `false`.

### device_intervals (7)

`volvoxai.v1.TraceDeviceIntervalCounts`.

Protobuf default: `null`.

### host_activities (8)

`volvoxai.v1.TraceHostActivityCounts`.

Protobuf default: `null`.

## volvoxai.v1.TraceInfo



### trace_id (1)

`int64`.

Protobuf default: `"0"`.

### report (2)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### state (3)

`volvoxai.v1.TraceState`.

Protobuf default: `"TRACE_STATE_UNSPECIFIED"`.

### options (4)

`volvoxai.v1.TraceOptions`.

Protobuf default: `null`.

### events (5)

`volvoxai.v1.RecordCount`.

Protobuf default: `null`.

Counts include reserved pending slots until READY; then they are final.

### resource_snapshots (6)

`volvoxai.v1.RecordCount`.

Protobuf default: `null`.

### plans (7)

`volvoxai.v1.RecordCount`.

Protobuf default: `null`.

### active_operations (8)

`uint64`.

Protobuf default: `"0"`.

Host scopes and pending asynchronous host completions.

### pending_device_intervals (9)

`uint64`.

Protobuf default: `"0"`.

### host_clock_resolution_ns (10)

`uint64`.

Protobuf default: `"0"`.

Effective host clock resolution. Absent when unknown (for example, browser
clock quantization). Nanosecond storage does not imply that precision.

### collector_bytes (11)

`uint64`.

Protobuf default: `"0"`.

Requested collector storage, including snapshots and plans.

### devices (12)

`volvoxai.v1.TraceDeviceCoverage` repeated.

Protobuf default: `[]`.

### allocators (13)

`volvoxai.v1.TraceAllocatorMemory` repeated.

Protobuf default: `[]`.

### external (14)

`volvoxai.v1.TraceExternalCoverage` repeated.

Protobuf default: `[]`.

One row per external mechanism the trace attempted.

### capture_origin_ns (15)

`uint64`.

Protobuf default: `"0"`.

The VxPlatformService monotonic time of capture-relative zero, to align
events with external timelines such as browser DevTools.

## volvoxai.v1.TraceHostSpan



### start_ns (1)

`uint64`.

Protobuf default: `"0"`.

Monotonic host time relative to capture start. A COMPLETION activity measures
asynchronous completion latency, not time occupying the submitting thread.

### duration_ns (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TraceDeviceInterval



### host_observed_ns (1)

`uint64`.

Protobuf default: `"0"`.

Host marker observed before device work is encoded/submitted, not GPU start.

### elapsed_ns (2)

`uint64`.

Protobuf default: `"0"`.

### correlation (3)

`volvoxai.v1.TraceClockCorrelation`.

Protobuf default: `null`.

Absent when no valid relation to the capture-relative host clock exists.

## volvoxai.v1.TraceClockCorrelation



### method (1)

`volvoxai.v1.TraceClockMethod`.

Protobuf default: `"TRACE_CLOCK_METHOD_UNSPECIFIED"`.

### earliest_start_ns (2)

`uint64`.

Protobuf default: `"0"`.

Range for the device interval's start on the capture-relative host clock.
It includes calibration uncertainty; it is not a precise start time.

### latest_start_ns (3)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.TraceQueue



### device_id (1)

`uint64`.

Protobuf default: `"0"`.

Opaque identities, meaningful only within this capture. No native handles.
Devices from different backend APIs are not asserted to be the same GPU.

### queue_id (2)

`uint64`.

Protobuf default: `"0"`.

### submission_id (3)

`uint64`.

Protobuf default: `"0"`.

Absent when the observation is not tied to one known submission/batch.

## volvoxai.v1.TraceCopy



### source (1)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### destination (2)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### bytes (3)

`uint64`.

Protobuf default: `"0"`.

Bytes passed to this copy, including required padding.

## volvoxai.v1.TraceNode



### schedule_index (1)

`uint32`.

Protobuf default: `0`.

Index in the executable schedule. Fused duration belongs to this work;
it must never be divided among eliminated source nodes.

### output_name (2)

`string`.

Protobuf default: `""`.

### fused (3)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.TraceProgram

A measured engine program invocation. A CUDA program can launch several
internal kernels; this is not a CUPTI kernel activity record.

### name (1)

`string`.

Protobuf default: `""`.

### entry_point (2)

`string`.

Protobuf default: `""`.

## volvoxai.v1.TraceEvent



### sequence (1)

`uint64`.

Protobuf default: `"0"`.

### name (2)

`string`.

Protobuf default: `""`.

Host operations use the RPC method name, for example "Execute".

### track_id (3)

`uint64`.

Protobuf default: `"0"`.

Host thread; device intervals annotate this submitter.

### lineage (4)

`volvoxai.v1.Lineage`.

Protobuf default: `null`.

### backend (5)

`string`.

Protobuf default: `""`.

### node (6)

`volvoxai.v1.TraceNode`.

Protobuf default: `null`.

With program: the owning executable node, when known. Without program:
the node measured by this event. A node's duration includes its programs;
these nested observations must not be added together as independent work.

### metadata_truncated (7)

`bool`.

Protobuf default: `false`.

### host (8)

`volvoxai.v1.TraceHostSpan`; oneof `observation`.

Protobuf default: `null`.

### device (9)

`volvoxai.v1.TraceDeviceInterval`; oneof `observation`.

Protobuf default: `null`.

### memory (10)

`volvoxai.v1.TraceMemoryEvent`; oneof `observation`.

Protobuf default: `null`.

### phase (11)

`volvoxai.v1.TracePhase`.

Protobuf default: `"TRACE_PHASE_UNSPECIFIED"`.

### program (12)

`volvoxai.v1.TraceProgram`.

Protobuf default: `null`.

Presence distinguishes a program interval from its enclosing node/pass.

### tensor_name (13)

`string`.

Protobuf default: `""`.

A known work target, e.g. the parameter updated by an optimizer. Empty when
unknown or work spans multiple tensors.

### activity (14)

`volvoxai.v1.TraceActivity`.

Protobuf default: `"TRACE_ACTIVITY_UNSPECIFIED"`.

### queue (15)

`volvoxai.v1.TraceQueue`.

Protobuf default: `null`.

### copy (16)

`volvoxai.v1.TraceCopy`.

Protobuf default: `null`.

Present exactly on COPY events.

### plan_id (17)

`uint64`.

Protobuf default: `"0"`.

The execution plan of this work (GetTracePlan). Absent when not captured.

## volvoxai.v1.TraceMemoryEvent



### allocation_id (1)

`uint64`.

Protobuf default: `"0"`.

### allocator (2)

`string`.

Protobuf default: `""`.

### space (3)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### action (4)

`volvoxai.v1.TraceMemoryAction`.

Protobuf default: `"TRACE_MEMORY_ACTION_UNSPECIFIED"`.

### bytes (5)

`uint64`.

Protobuf default: `"0"`.

### live_bytes (6)

`uint64`.

Protobuf default: `"0"`.

This allocator's tracked requested capacity after the event.

### role (7)

`volvoxai.v1.MemoryResourceRole`.

Protobuf default: `"MEMORY_RESOURCE_ROLE_UNSPECIFIED"`.

### timestamp_ns (8)

`uint64`.

Protobuf default: `"0"`.

Capture-relative host time.

## volvoxai.v1.TraceAllocatorMemory

Requested capacities of observed owned allocations, not physical residency.
Inventories are PARTIAL: metadata, driver allocations and unvisited owners
can be absent. Counters are additive only within their stated allocator.
Existing resources are inventoried when an owner first participates. A peak
covers observed allocator transitions from that point until StopTrace.

### allocator (1)

`string`.

Protobuf default: `""`.

### backend (2)

`string`.

Protobuf default: `""`.

### space (3)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### inventory (4)

`volvoxai.v1.MemoryInventoryKind`.

Protobuf default: `"MEMORY_INVENTORY_KIND_UNSPECIFIED"`.

### observation_start_ns (5)

`uint64`.

Protobuf default: `"0"`.

### existing_bytes (6)

`uint64`.

Protobuf default: `"0"`.

### live_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### peak_bytes (8)

`uint64`.

Protobuf default: `"0"`.

### allocated_bytes (9)

`uint64`.

Protobuf default: `"0"`.

### freed_bytes (10)

`uint64`.

Protobuf default: `"0"`.

### dropped_events (11)

`uint64`.

Protobuf default: `"0"`.

### accounting_complete (12)

`bool`.

Protobuf default: `false`.

False if the bounded live-identity table overflowed. Byte counters then
describe only the tracked subset. Event loss alone does not lose counters.

### scope (13)

`volvoxai.v1.MemoryOwnerKind`.

Protobuf default: `"MEMORY_OWNER_KIND_UNSPECIFIED"`.

RUNTIME aggregates observed owners of the captured runtime. BACKEND_SHARED
includes every runtime using the same host bridge (currently WebGPU).
Never add BACKEND_SHARED counters from simultaneous runtime captures.

## volvoxai.v1.ExecutionPlan

Immutable metadata for one concrete context layout. Tensor lifetimes are
planner-derived schedule intervals, not measured resident device memory.
Plan and event IDs are capture-local; source IDs are model graph node IDs.
A plan remains readable after context/model release and never retains tensors.

### plan_id (1)

`uint64`.

Protobuf default: `"0"`.

### lineage (2)

`volvoxai.v1.Lineage`.

Protobuf default: `null`.

### graph_fingerprint (3)

`string`.

Protobuf default: `""`.

### shape_signature (4)

`string`.

Protobuf default: `""`.

### backend (5)

`string`.

Protobuf default: `""`.

### steps (6)

`volvoxai.v1.ExecutionPlanStep` repeated.

Protobuf default: `[]`.

### tensors (7)

`volvoxai.v1.ExecutionPlanTensor` repeated.

Protobuf default: `[]`.

### allocations (8)

`volvoxai.v1.MemoryAllocation` repeated.

Protobuf default: `[]`.

### placement_complete (9)

`bool`.

Protobuf default: `false`.

Every tensor with storage names its allocation.

### source_mapping_complete (10)

`bool`.

Protobuf default: `false`.

Every step names all source nodes it executes.

### metadata_truncated (11)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ExecutionPlanStep



### schedule_index (1)

`uint32`.

Protobuf default: `0`.

### operator_name (2)

`string`.

Protobuf default: `""`.

### source_node_ids (3)

`string` repeated.

Protobuf default: `[]`.

Source graph nodes executed by this step: the node itself and any node
fused into it. Debug node selectors match these IDs.

### inputs (4)

`uint32` repeated.

Protobuf default: `[]`.

ExecutionPlanTensor.tensor_id values.

### outputs (5)

`uint32` repeated.

Protobuf default: `[]`.

### skipped (6)

`bool`.

Protobuf default: `false`.

Eliminated by optimization; executes no kernel.

### fused (7)

`bool`.

Protobuf default: `false`.

### cost (8)

`volvoxai.v1.StepCost`.

Protobuf default: `null`.

Work one call implies at these shapes, including a fused peer's.

### phase (9)

`volvoxai.v1.TracePhase`.

Protobuf default: `"TRACE_PHASE_UNSPECIFIED"`.

Training debug plans: which part of the step this entry performs.
UNSPECIFIED for inference plans.

## volvoxai.v1.StepCost

Counts, not measurements. Multiply-accumulates cover matrix products,
convolutions and attention, float or integer (one FLOP-style operation pair
each). Other arithmetic is one operation per output element; exp, tanh and
erf-class functions are also counted as transcendental. Data movement has
no arithmetic. Bytes are the logical sizes of the step's inputs (including
weights) and outputs: a lower bound on memory traffic.

### status (1)

`volvoxai.v1.CostStatus`.

Protobuf default: `"COST_STATUS_UNSPECIFIED"`.

### multiply_accumulates (2)

`uint64`.

Protobuf default: `"0"`.

### elementwise_operations (3)

`uint64`.

Protobuf default: `"0"`.

### transcendental_operations (4)

`uint64`.

Protobuf default: `"0"`.

### input_bytes (5)

`uint64`.

Protobuf default: `"0"`.

### output_bytes (6)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.ExecutionPlanTensor



### tensor_id (1)

`uint32`.

Protobuf default: `0`.

### name (2)

`string`.

Protobuf default: `""`.

### dtype (3)

`volvoxai.v1.DataType`.

Protobuf default: `"DATA_TYPE_UNSPECIFIED"`.

### shape (4)

`int64` repeated.

Protobuf default: `[]`.

### logical_bytes (5)

`uint64`.

Protobuf default: `"0"`.

### allocation_id (6)

`uint64`.

Protobuf default: `"0"`.

Absent if the backend did not expose this tensor's physical placement.

### offset_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### first_step (8)

`uint32`.

Protobuf default: `0`.

Inclusive schedule interval. Inputs start before step zero; retained
inputs/outputs may remain live across calls. Absent lifetime is unknown.

### last_step (9)

`uint32`.

Protobuf default: `0`.

### retained (10)

`bool`.

Protobuf default: `false`.

### source_tensor_name (11)

`string`.

Protobuf default: `""`.

The float tensor this value represents: a PTQ package records it for each
renamed activation. Empty when the name already is the source name.

### gradient_of (12)

`uint32`.

Protobuf default: `0`.

Training debug plans: the tensor whose gradient this is. Gradient tensors
are named "<tensor>.grad".

## volvoxai.v1.ListTraceEventsRequest



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### page_size (2)

`uint32`.

Protobuf default: `0`.

Engine default: 256

Range: `[1, 4096]`.

### page_token (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ListTraceEventsResponse



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### events (2)

`volvoxai.v1.TraceEvent` repeated.

Protobuf default: `[]`.

### next_page_token (3)

`string`.

Protobuf default: `""`.

### total_size (4)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.ListTraceResourceSnapshotsRequest



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### page_size (2)

`uint32`.

Protobuf default: `0`.

Engine default: 256

Range: `[1, 4096]`.

### page_token (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ListTraceResourceSnapshotsResponse



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### resource_snapshots (2)

`volvoxai.v1.ResourceSnapshot` repeated.

Protobuf default: `[]`.

Capture-relative times, oldest first.

### next_page_token (3)

`string`.

Protobuf default: `""`.

### total_size (4)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.GetTracePlanRequest



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### plan_id (2)

`uint64`; required.

Protobuf default: `"0"`.

Range: `[1, ∞]`.

TraceEvent.plan_id. Plans are numbered from 1 to TraceInfo.plans.count.

## volvoxai.v1.GetTracePlanResponse



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### plan (2)

`volvoxai.v1.ExecutionPlan`.

Protobuf default: `null`.

## volvoxai.v1.GetTraceSummaryRequest



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### group_by (2)

`volvoxai.v1.TraceSummaryGrouping`; required.

Protobuf default: `"TRACE_SUMMARY_GROUPING_UNSPECIFIED"`.

### max_rows (3)

`uint32`.

Protobuf default: `0`.

Engine default: 256

Range: `[1, 4096]`.

## volvoxai.v1.TraceSummaryRow

Rows never mix host and device time, or nodes, programs and calls: those
intervals overlap and are not additive. A program interval is not added to
its node. Truncated metadata is excluded and counted.

### lineage (1)

`volvoxai.v1.Lineage`.

Protobuf default: `null`.

Model, compiled model and context identities.

### plan_id (2)

`uint64`.

Protobuf default: `"0"`.

### backend (3)

`string`.

Protobuf default: `""`.

### phase (4)

`volvoxai.v1.TracePhase`.

Protobuf default: `"TRACE_PHASE_UNSPECIFIED"`.

### activity (5)

`volvoxai.v1.TraceActivity`.

Protobuf default: `"TRACE_ACTIVITY_UNSPECIFIED"`.

### domain (6)

`volvoxai.v1.TraceTimeDomain`.

Protobuf default: `"TRACE_TIME_DOMAIN_UNSPECIFIED"`.

### name (7)

`string`.

Protobuf default: `""`.

Operator, RPC method or activity name.

### schedule_index (8)

`uint32`.

Protobuf default: `0`.

### output_name (9)

`string`.

Protobuf default: `""`.

### source_node_ids (10)

`string` repeated.

Protobuf default: `[]`.

### fused (11)

`bool`.

Protobuf default: `false`.

### count (12)

`uint64`.

Protobuf default: `"0"`.

### total_ns (13)

`uint64`.

Protobuf default: `"0"`.

### min_ns (14)

`uint64`.

Protobuf default: `"0"`.

### median_ns (15)

`uint64`.

Protobuf default: `"0"`.

### p95_ns (16)

`uint64`.

Protobuf default: `"0"`.

Nearest rank.

### max_ns (17)

`uint64`.

Protobuf default: `"0"`.

### share (18)

`double`.

Protobuf default: `0`.

Share of total_ns among rows with the same lineage, plan, backend, phase
and domain. Not application latency or utilization.

### cost_per_call (19)

`volvoxai.v1.StepCost`.

Protobuf default: `null`.

Node rows: the plan step's cost for one call.

### achieved_flops_per_second (20)

`double`.

Protobuf default: `0`.

Achieved rates over this row's time: 2 x multiply-accumulates per second,
and step bytes per second. Present when every call has a known cost and
the time measures that work: device intervals, or host nodes on CPU/WASM.
Absent for decode steps, whose rows run only part of the plan's shape.
Rates, not utilization: the engine does not know peak throughput.

### achieved_bytes_per_second (21)

`double`.

Protobuf default: `0`.

### copy_bytes (22)

`uint64`.

Protobuf default: `"0"`.

Activity rows: bytes copied.

## volvoxai.v1.TraceSummary



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### group_by (2)

`volvoxai.v1.TraceSummaryGrouping`.

Protobuf default: `"TRACE_SUMMARY_GROUPING_UNSPECIFIED"`.

### rows (3)

`volvoxai.v1.TraceSummaryRow` repeated.

Protobuf default: `[]`.

total_ns descending

### truncated (4)

`bool`.

Protobuf default: `false`.

More rows than max_rows.

### events (5)

`volvoxai.v1.RecordCount`.

Protobuf default: `null`.

Copied from TraceInfo: lost events make totals partial.

### excluded_events (6)

`uint64`.

Protobuf default: `"0"`.

Truncated metadata.

## volvoxai.v1.ExportChromeTraceRequest



### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### page_size (2)

`uint32`.

Protobuf default: `0`.

Engine default: 128

Range: `[1, 1024]`.

Source events serialized per page. Trace metadata is added to the first
and last page; JSON page bytes vary.

### page_token (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.ExportChromeTraceResponse



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### data (2)

`bytes`.

Protobuf default: `""`.

A UTF-8 fragment. Concatenating every page from an empty page_token until
next_page_token is empty produces one Chrome Trace Event JSON document.
Repeating a request returns the same bytes.

### next_page_token (3)

`string`.

Protobuf default: `""`.

## volvoxai.v1.AnnotateTraceRequest

An application-defined range, like torch.profiler.record_function or
jax.profiler.TraceAnnotation. Times use the VxPlatformService monotonic clock
(GetMonotonicTime), not capture-relative time.

### trace_id (1)

`int64`; required.

Protobuf default: `"0"`.

Handle kind: `Trace`.

### name (2)

`string`; required.

Protobuf default: `""`.

### start_ns (3)

`uint64`; required.

Protobuf default: `"0"`.

### end_ns (4)

`uint64`; required.

Protobuf default: `"0"`.

## volvoxai.v1.ObservedBytes



### status (1)

`volvoxai.v1.ObservationStatus`.

Protobuf default: `"OBSERVATION_STATUS_UNSPECIFIED"`.

### bytes (2)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.MemoryAllocation

One backing allocation. IDs are local to this snapshot; no pointer is exposed.
Aliases and tensor placements refer to allocation_id and add no storage.
Capacity is allocator/API storage, not physical RSS or resident VRAM.

### allocation_id (1)

`uint64`.

Protobuf default: `"0"`.

### owner (2)

`volvoxai.v1.MemoryOwnerRef`.

Protobuf default: `null`.

### role (3)

`volvoxai.v1.MemoryResourceRole`.

Protobuf default: `"MEMORY_RESOURCE_ROLE_UNSPECIFIED"`.

### space (4)

`volvoxai.v1.MemorySpace`.

Protobuf default: `"MEMORY_SPACE_UNSPECIFIED"`.

### allocator (5)

`string`.

Protobuf default: `""`.

### name (6)

`string`.

Protobuf default: `""`.

### capacity_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### trace_allocation_id (8)

`uint64`.

Protobuf default: `"0"`.

Present only when a captured plan can identify the matching allocator
event in this trace. Snapshot-local IDs alone never join different pages.

## volvoxai.v1.MemorySnapshot

Known owned allocations reachable from the selected scope. Shared backing
allocations occur once. A PARTIAL inventory omits unclassified owners,
driver-private storage and retired handles that the scope cannot reach.

### subject (1)

`volvoxai.v1.MemoryOwnerRef`.

Protobuf default: `null`.

### allocations (2)

`volvoxai.v1.MemoryAllocation` repeated.

Protobuf default: `[]`.

### inventory (3)

`volvoxai.v1.MemoryInventoryKind`.

Protobuf default: `"MEMORY_INVENTORY_KIND_UNSPECIFIED"`.

### truncated (4)

`bool`.

Protobuf default: `false`.

### unconsumed_result_bytes (5)

`uint64`.

Protobuf default: `"0"`.

These accounting values overlap allocations and are not additive to them.

### retained_result_capacity_bytes (6)

`uint64`.

Protobuf default: `"0"`.

### idle_result_capacity_bytes (7)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.WasmMemorySnapshot

Exact partition of this WASM module's linear memory at observation time.
linear = allocated + free + allocator_metadata + module_prefix + page_slack
+ untracked. Untracked covers raw heap reservations outside malloc.
All runtimes in the module share this heap. Free bytes may be fragmented;
they are not the unused tensor ranges inside an allocated arena.

### status (1)

`volvoxai.v1.ObservationStatus`.

Protobuf default: `"OBSERVATION_STATUS_UNSPECIFIED"`.

### linear_bytes (2)

`uint64`.

Protobuf default: `"0"`.

### allocated_bytes (3)

`uint64`.

Protobuf default: `"0"`.

### free_bytes (4)

`uint64`.

Protobuf default: `"0"`.

### allocator_metadata_bytes (5)

`uint64`.

Protobuf default: `"0"`.

### module_prefix_bytes (6)

`uint64`.

Protobuf default: `"0"`.

### page_slack_bytes (7)

`uint64`.

Protobuf default: `"0"`.

### largest_free_block_bytes (8)

`uint64`.

Protobuf default: `"0"`.

### allocated_blocks (9)

`uint64`.

Protobuf default: `"0"`.

### free_blocks (10)

`uint64`.

Protobuf default: `"0"`.

### untracked_bytes (11)

`uint64`.

Protobuf default: `"0"`.

## volvoxai.v1.ProcessMemorySnapshot

OS process observations overlap all engine accounting. They are never summed
with allocation capacities or WASM memory. Peak covers the process lifetime.

### source (1)

`string`.

Protobuf default: `""`.

### resident (2)

`volvoxai.v1.ObservedBytes`.

Protobuf default: `null`.

### peak_resident (3)

`volvoxai.v1.ObservedBytes`.

Protobuf default: `null`.

## volvoxai.v1.ProcessCpuSample

Cumulative CPU time consumed by all threads in this process. Two observations
from the same process and clock yield average occupied CPU cores:
delta(process_time_ns) / delta(observation time). 1.0 means one CPU core;
the value can exceed 1. This is neither engine-only nor system-wide load.
Browser APIs cannot supply process CPU time: UNSUPPORTED, never wall time.

### status (1)

`volvoxai.v1.ObservationStatus`.

Protobuf default: `"OBSERVATION_STATUS_UNSPECIFIED"`.

### source (2)

`string`.

Protobuf default: `""`.

### process_time_ns (3)

`uint64`.

Protobuf default: `"0"`.

### online_processors (4)

`uint32`.

Protobuf default: `0`.

## volvoxai.v1.GpuResourceSample

Hardware-wide driver telemetry (nvidia-smi style), including work from other
applications. Ratios describe the driver's preceding sample window, not this
request or model. GPU utilization is not SM occupancy or percentage of peak
FLOPS; memory utilization is memory-interface busy time, not used/capacity.

### device_id (1)

`string`.

Protobuf default: `""`.

Stable driver device UUID when available.

### name (2)

`string`.

Protobuf default: `""`.

### source (3)

`string`.

Protobuf default: `""`.

### utilization_status (4)

`volvoxai.v1.ObservationStatus`.

Protobuf default: `"OBSERVATION_STATUS_UNSPECIFIED"`.

### gpu_utilization (5)

`double`.

Protobuf default: `0`.

### memory_utilization (6)

`double`.

Protobuf default: `0`.

### sampling_window_ns (7)

`uint64`.

Protobuf default: `"0"`.

Absent when the driver omits it.

### memory_total (8)

`volvoxai.v1.ObservedBytes`.

Protobuf default: `null`.

### memory_used (9)

`volvoxai.v1.ObservedBytes`.

Protobuf default: `null`.

## volvoxai.v1.GpuResourceSamples



### status (1)

`volvoxai.v1.ObservationStatus`.

Protobuf default: `"OBSERVATION_STATUS_UNSPECIFIED"`.

### source (2)

`string`.

Protobuf default: `""`.

### devices (3)

`volvoxai.v1.GpuResourceSample` repeated.

Protobuf default: `[]`.

### truncated (4)

`bool`.

Protobuf default: `false`.

## volvoxai.v1.ResourceSnapshot

Independent observations, not an atomic system snapshot. Times are monotonic
host nanoseconds; trace snapshots use the capture-relative clock. CPU and GPU
utilization remain separate from host spans and timestamp-query intervals.

### observation_start_ns (1)

`uint64`.

Protobuf default: `"0"`.

### observation_end_ns (2)

`uint64`.

Protobuf default: `"0"`.

### memory (3)

`volvoxai.v1.MemorySnapshot`.

Protobuf default: `null`.

### wasm (4)

`volvoxai.v1.WasmMemorySnapshot`.

Protobuf default: `null`.

### process (5)

`volvoxai.v1.ProcessMemorySnapshot`.

Protobuf default: `null`.

### cpu (6)

`volvoxai.v1.ProcessCpuSample`.

Protobuf default: `null`.

### gpu (7)

`volvoxai.v1.GpuResourceSamples`.

Protobuf default: `null`.

## volvoxai.v1.GetResourceSnapshotRequest

Exactly one scope. module selects every known owner in this engine module,
not every engine loaded in the process. External observations always retain
their actual scope.

### module (1)

`volvoxai.v1.Empty`; oneof `scope`.

Protobuf default: `null`.

### runtime_id (2)

`int64`; oneof `scope`.

Protobuf default: `"0"`.

### context_id (3)

`int64`; oneof `scope`.

Protobuf default: `"0"`.

### include_device (4)

`bool`.

Protobuf default: `false`.

Opt-in driver sampling; never initializes a compute backend.

### max_allocations (5)

`uint32`.

Protobuf default: `0`.

Engine default: 4096

Range: `[1, 65536]`.

Truncation is explicit in MemorySnapshot.truncated.

## volvoxai.v1.ResourceSnapshotResponse



### report (1)

`volvoxai.v1.OperationReport`.

Protobuf default: `null`.

### snapshot (2)

`volvoxai.v1.ResourceSnapshot`.

Protobuf default: `null`.

## volvoxai.v1.TokenizerProfile



- `TOKENIZER_PROFILE_LEGACY = 0`
- `TOKENIZER_PROFILE_BYTE_LEVEL_DECODE_ONLY = 1`
- `TOKENIZER_PROFILE_HF_BYTE_LEVEL_BPE = 2`
- `TOKENIZER_PROFILE_HF_SENTENCEPIECE_BPE = 3`

## volvoxai.v1.TokenizationMode



- `TOKENIZATION_MODE_AUTO = 0`: HF profiles use their admitted tokenizer.json pipeline. Legacy profiles
use greedy byte matching without valid merges, otherwise Unicode
letter/number/punctuation/whitespace pretokenization and ranked BPE.
- `TOKENIZATION_MODE_GREEDY = 1`
- `TOKENIZATION_MODE_BPE_WORD = 2`: Apply BPE to the whole text as one chunk, without pretokenization.

## volvoxai.v1.BatchWorkKind

C owns compatibility grouping, admission, token budgets, padding, slot/page
reservations, prefix reuse and retirement. The application supplies work and
executes each returned dispatch using its chosen worker. There is no product
TypeScript scheduler. Inference workers can use VxInferenceService; page plans
here describe worker-owned storage and are not an ExecutionContext binding.

- `BATCH_WORK_KIND_STATELESS = 0`
- `BATCH_WORK_KIND_PREFILL = 1`
- `BATCH_WORK_KIND_DECODE = 2`

## volvoxai.v1.BatchWorkState



- `BATCH_WORK_STATE_UNKNOWN = 0`: invalid ID or evicted terminal record
- `BATCH_WORK_STATE_QUEUED = 1`
- `BATCH_WORK_STATE_PREFILL = 2`
- `BATCH_WORK_STATE_DECODING = 3`
- `BATCH_WORK_STATE_COMPLETED = 4`
- `BATCH_WORK_STATE_CANCELLED = 5`
- `BATCH_WORK_STATE_FAILED = 6`

## volvoxai.v1.BatchQueuePolicy



- `BATCH_QUEUE_POLICY_WORK_CONSERVING = 0`
- `BATCH_QUEUE_POLICY_FILL_FIRST = 1`

## volvoxai.v1.DataType

Canonical tensor element/storage vocabulary. Enum membership describes a
representation, not execution support: each operation validates the subset
it can execute. Keep existing numbers stable and append future dtypes.

- `DATA_TYPE_UNSPECIFIED = 0`
- `DATA_TYPE_BOOL = 1`
- `DATA_TYPE_F4 = 2`
- `DATA_TYPE_F6_E2M3 = 3`
- `DATA_TYPE_F6_E3M2 = 4`
- `DATA_TYPE_U8 = 5`
- `DATA_TYPE_I8 = 6`
- `DATA_TYPE_F8_E5M2 = 7`
- `DATA_TYPE_F8_E4M3 = 8`
- `DATA_TYPE_F8_E8M0 = 9`
- `DATA_TYPE_F8_E4M3FNUZ = 10`
- `DATA_TYPE_F8_E5M2FNUZ = 11`
- `DATA_TYPE_I16 = 12`
- `DATA_TYPE_U16 = 13`
- `DATA_TYPE_F16 = 14`
- `DATA_TYPE_BF16 = 15`
- `DATA_TYPE_I32 = 16`
- `DATA_TYPE_U32 = 17`
- `DATA_TYPE_F32 = 18`
- `DATA_TYPE_C64 = 19`
- `DATA_TYPE_F64 = 20`
- `DATA_TYPE_I64 = 21`
- `DATA_TYPE_U64 = 22`

## volvoxai.v1.MemoryLocation



- `MEMORY_LOCATION_HOST = 0`
- `MEMORY_LOCATION_DEVICE = 1`

## volvoxai.v1.BackendPolicyMode



- `BACKEND_POLICY_MODE_PREFER = 0`
- `BACKEND_POLICY_MODE_REQUIRE = 1`

## volvoxai.v1.OperatorFallback



- `OPERATOR_FALLBACK_ALLOW = 0`
- `OPERATOR_FALLBACK_FORBID = 1`

## volvoxai.v1.DecodeRowMode



- `DECODE_ROW_MODE_DISABLED = 0`
- `DECODE_ROW_MODE_AUTO = 1`
- `DECODE_ROW_MODE_REQUIRED = 2`

## volvoxai.v1.ExecutionMode

Runtime dispatch ownership. DIRECT bypasses Runtime scheduling entirely and
rejects every VxSchedulerService operation. SCHEDULED permits both direct
Execute/Generate calls and scheduled Submit calls. CreateRuntime defaults to DIRECT when
execution_mode is absent.

- `EXECUTION_MODE_DIRECT = 0`
- `EXECUTION_MODE_SCHEDULED = 1`

## volvoxai.v1.BuildProfile

Which build profile a library was composed with. An inference build does
not register VxTrainingService or VxQuantizationService.

- `BUILD_PROFILE_UNSPECIFIED = 0`
- `BUILD_PROFILE_INFERENCE = 1`
- `BUILD_PROFILE_FULL = 2`

## volvoxai.v1.TransportProfile

IN_PROCESS accepts BorrowedBuffer descriptors in its local address space.
REMOTE accepts inline bytes and server-owned BufferView capabilities, but
rejects local pointers, native resources, access mappings and DLPack. A
network adapter must enforce this boundary before forwarding native calls.

- `TRANSPORT_PROFILE_UNSPECIFIED = 0`
- `TRANSPORT_PROFILE_IN_PROCESS = 1`
- `TRANSPORT_PROFILE_REMOTE = 2`

## volvoxai.v1.NativeStatus



- `NATIVE_STATUS_OK = 0`
- `NATIVE_STATUS_INVALID_ARGUMENT = -1`
- `NATIVE_STATUS_OUT_OF_MEMORY = -2`
- `NATIVE_STATUS_HANDLE_DISPOSED = -3`
- `NATIVE_STATUS_IO_ERROR = -4`
- `NATIVE_STATUS_INVALID_GRAPH = -5`
- `NATIVE_STATUS_BACKEND_UNAVAILABLE = -6`
- `NATIVE_STATUS_BACKEND_UNSUPPORTED = -7`
- `NATIVE_STATUS_BACKEND_REQUIRED = -8`
- `NATIVE_STATUS_OPERATOR_FALLBACK_FORBIDDEN = -9`
- `NATIVE_STATUS_EXECUTION_FAILED = -10`
- `NATIVE_STATUS_RESULT_DISPOSED = -11`
- `NATIVE_STATUS_DEVICE_LOST = -12`
- `NATIVE_STATUS_ABI_UNSUPPORTED = -13`
- `NATIVE_STATUS_NOT_FOUND = -14`
- `NATIVE_STATUS_BUFFER_TOO_SMALL = -15`
- `NATIVE_STATUS_INTERNAL = -16`
- `NATIVE_STATUS_REVISION_CONFLICT = -17`
- `NATIVE_STATUS_BUSY = -18`
- `NATIVE_STATUS_OVERLOADED = -19`
- `NATIVE_STATUS_CANCELLED = -20`
- `NATIVE_STATUS_DEADLINE_EXCEEDED = -21`
- `NATIVE_STATUS_SUPERSEDED = -22`
- `NATIVE_STATUS_CONTEXT_RESET_REQUIRED = -23`
- `NATIVE_STATUS_TRANSPORT_UNSUPPORTED = -24`: A BorrowedBuffer or local interop request reached an incompatible transport.

## volvoxai.v1.OperationCode

Machine-readable detail. Status describes the broad outcome; code preserves
the specific cause independently of human-readable message text.

- `OPERATION_CODE_NONE = 0`
- `OPERATION_CODE_ABI_UNSUPPORTED = 1`
- `OPERATION_CODE_ACTIVATION_CAPACITY_EXCEEDED = 3`
- `OPERATION_CODE_ACTIVATION_PLAN_INVALID = 4`
- `OPERATION_CODE_ACTIVATION_SIGNATURE_MISMATCH = 5`
- `OPERATION_CODE_ADAPTER_NOT_READABLE = 6`
- `OPERATION_CODE_ADAPTER_PACKAGE_REJECTED = 7`
- `OPERATION_CODE_ADAPTER_REVISION_MISSING = 8`
- `OPERATION_CODE_ADAPTER_REVISION_NOT_FOUND = 9`
- `OPERATION_CODE_ADAPTER_UNSUPPORTED = 10`
- `OPERATION_CODE_BACKEND_CONTEXT_CREATE_FAILED = 11`
- `OPERATION_CODE_BACKEND_ENUMERATION_FAILED = 12`
- `OPERATION_CODE_BACKEND_REQUIRED = 13`
- `OPERATION_CODE_BACKEND_UNAVAILABLE = 14`
- `OPERATION_CODE_BACKEND_UNSUPPORTED = 15`
- `OPERATION_CODE_BASE_ADAPTER_MISSING = 16`
- `OPERATION_CODE_BATCH_AXIS_INVALID = 17`
- `OPERATION_CODE_BATCH_AXIS_MISMATCH = 18`
- `OPERATION_CODE_BATCH_CONTRACT_INVALID = 19`
- `OPERATION_CODE_BATCH_EVIDENCE_UNAVAILABLE = 20`
- `OPERATION_CODE_BATCH_OUTPUT_AXIS_INVALID = 21`
- `OPERATION_CODE_BATCH_PROOF_MISMATCH = 22`
- `OPERATION_CODE_BATCH_RESULT_MISSING = 23`
- `OPERATION_CODE_BOUNDED_DOMAIN_UNSUPPORTED = 24`
- `OPERATION_CODE_BUFFER_TOO_SMALL = 25`
- `OPERATION_CODE_BUSY = 26`
- `OPERATION_CODE_CALL_SEQUENCE_POLICY_INVALID = 28`
- `OPERATION_CODE_CALL_SEQUENCE_SELECTION_INVALID = 29`
- `OPERATION_CODE_CANCELLED = 30`
- `OPERATION_CODE_CANONICAL_SHAPE_CONTRACT_UNSUPPORTED = 31`
- `OPERATION_CODE_CLOCK_FAILED = 33`
- `OPERATION_CODE_CONDITION_INIT_FAILED = 35`
- `OPERATION_CODE_CONTEXT_RESET_REQUIRED = 36`
- `OPERATION_CODE_COORDINATOR_INIT_FAILED = 37`
- `OPERATION_CODE_DEADLINE_EXCEEDED = 38`
- `OPERATION_CODE_DECODE_CACHE_CAPACITY = 39`
- `OPERATION_CODE_DECODE_CACHE_REJECTED = 40`
- `OPERATION_CODE_DECODE_MODE_UNSUPPORTED = 41`
- `OPERATION_CODE_DECODE_NOT_ENABLED = 42`
- `OPERATION_CODE_DECODE_PREFILL_REQUIRED = 43`
- `OPERATION_CODE_DECODE_RESET_FAILED = 44`
- `OPERATION_CODE_DECODE_RESOURCE_DOMAIN_UNSUPPORTED = 45`
- `OPERATION_CODE_DECODE_ROW_DOMAIN_UNSUPPORTED = 46`
- `OPERATION_CODE_DECODE_STATE_WRITE_FAILED = 47`
- `OPERATION_CODE_DEVICE_EXECUTION_FAILED = 48`
- `OPERATION_CODE_DEVICE_LOST = 49`
- `OPERATION_CODE_DUPLICATE_DECODE_INPUT = 50`
- `OPERATION_CODE_DUPLICATE_PROVIDER = 52`
- `OPERATION_CODE_DYNAMIC_CUDA_ADAPTER_UNSUPPORTED = 53`
- `OPERATION_CODE_DYNAMIC_CUDA_DECODE_UNSUPPORTED = 54`
- `OPERATION_CODE_DYNAMIC_DOMAIN_OUT_OF_MEMORY = 55`
- `OPERATION_CODE_DYNAMIC_DOMAIN_RESERVATION_FAILED = 56`
- `OPERATION_CODE_EXECUTION_EVIDENCE_UNAVAILABLE = 57`
- `OPERATION_CODE_EXECUTION_FAILED = 58`
- `OPERATION_CODE_FETCH_UNAVAILABLE = 61`
- `OPERATION_CODE_GRAPH_INITIALIZATION_FAILED = 62`
- `OPERATION_CODE_GRAPH_LOWERING_FAILED = 63`
- `OPERATION_CODE_GRAPH_NOT_READABLE = 64`
- `OPERATION_CODE_HANDLE_DISPOSED = 65`
- `OPERATION_CODE_INPUT_BINDING_FAILED = 66`
- `OPERATION_CODE_INPUT_NOT_FOUND = 67`
- `OPERATION_CODE_INPUT_QUANTIZATION_INTROSPECTION_UNSUPPORTED = 68`
- `OPERATION_CODE_INTERNAL = 70`
- `OPERATION_CODE_INVALID_ADAPTER_REVISION = 71`
- `OPERATION_CODE_INVALID_ADAPTER_SOURCE = 72`
- `OPERATION_CODE_INVALID_AFFINE_QUANTIZATION_REQUEST = 73`
- `OPERATION_CODE_INVALID_ARGUMENT = 74`
- `OPERATION_CODE_INVALID_AUTHORING_REVISION = 75`
- `OPERATION_CODE_INVALID_BANK_RESIDENCY = 76`
- `OPERATION_CODE_INVALID_CONTEXT = 80`
- `OPERATION_CODE_INVALID_CONTEXT_OPTIONS = 81`
- `OPERATION_CODE_INVALID_DECODE_DEPENDENCY = 82`
- `OPERATION_CODE_INVALID_DECODE_FEEDBACK = 83`
- `OPERATION_CODE_INVALID_DECODE_INPUT = 84`
- `OPERATION_CODE_INVALID_DECODE_POSITION = 85`
- `OPERATION_CODE_INVALID_GRAPH = 87`
- `OPERATION_CODE_INVALID_GRAPH_CONTRACT = 88`
- `OPERATION_CODE_INVALID_GRAPH_PATH = 89`
- `OPERATION_CODE_INVALID_INPUT_BATCH = 90`
- `OPERATION_CODE_INVALID_INPUT_VALUES = 92`
- `OPERATION_CODE_INVALID_LOGICAL_DOMAIN = 93`
- `OPERATION_CODE_INVALID_MODEL_SOURCE = 94`
- `OPERATION_CODE_INVALID_MODEL_SOURCE_RESOLUTION = 95`
- `OPERATION_CODE_INVALID_OPTIONS = 96`
- `OPERATION_CODE_INVALID_OUTPUT_INFO = 97`
- `OPERATION_CODE_INVALID_POLICY = 98`
- `OPERATION_CODE_INVALID_PREFIX_ROW_COUNT = 99`
- `OPERATION_CODE_INVALID_PROVIDER = 100`
- `OPERATION_CODE_INVALID_READ = 108`
- `OPERATION_CODE_INVALID_REQUEST = 109`
- `OPERATION_CODE_INVALID_REQUEST_INFO = 110`
- `OPERATION_CODE_INVALID_RESULT_POINTER = 111`
- `OPERATION_CODE_INVALID_REVISION_INFO = 112`
- `OPERATION_CODE_INVALID_RUN = 113`
- `OPERATION_CODE_INVALID_RUNTIME = 114`
- `OPERATION_CODE_INVALID_SUBMISSION = 116`
- `OPERATION_CODE_INVALID_TENSOR_SPEC = 118`
- `OPERATION_CODE_INVALID_TRAINER = 119`
- `OPERATION_CODE_INVALID_TRAINING_ENGINE = 122`
- `OPERATION_CODE_INVALID_WEIGHT_REVISION = 124`
- `OPERATION_CODE_IO_ERROR = 125`
- `OPERATION_CODE_LOGICAL_GRAPH_INVALID = 126`
- `OPERATION_CODE_MODEL_SOURCE_RESOLUTION_FAILED = 127`
- `OPERATION_CODE_MUTEX_INIT_FAILED = 128`
- `OPERATION_CODE_NOT_FOUND = 129`
- `OPERATION_CODE_OPERATOR_FALLBACK_FORBIDDEN = 131`
- `OPERATION_CODE_OUTPUT_NOT_FOUND = 132`
- `OPERATION_CODE_OUT_OF_MEMORY = 133`
- `OPERATION_CODE_OVERLOADED = 134`
- `OPERATION_CODE_PACKAGE_BODY_UNAVAILABLE = 135`
- `OPERATION_CODE_PACKAGE_FETCH_FAILED = 136`
- `OPERATION_CODE_PACKAGE_READ_FAILED = 137`
- `OPERATION_CODE_PACKAGE_TOO_LARGE = 138`
- `OPERATION_CODE_PRIVATE_VFS_MOUNT_FAILED = 140`
- `OPERATION_CODE_PROVIDER_ALREADY_REGISTERED = 141`
- `OPERATION_CODE_PROVIDER_RUNTIME_CREATE_FAILED = 142`
- `OPERATION_CODE_QUEUE_FULL = 155`
- `OPERATION_CODE_QUEUE_INIT_FAILED = 156`
- `OPERATION_CODE_REQUEST_INIT_FAILED = 157`
- `OPERATION_CODE_RESULT_BUDGET_INVALID = 158`
- `OPERATION_CODE_RESULT_DISPOSED = 159`
- `OPERATION_CODE_RESULT_PENDING = 160`
- `OPERATION_CODE_REVISION_CONFLICT = 161`
- `OPERATION_CODE_REVISION_MISSING = 162`
- `OPERATION_CODE_REVISION_OVERFLOW = 163`
- `OPERATION_CODE_ROUTE_LEASE_FAILED = 165`
- `OPERATION_CODE_SCHEDULED_MODE_DISABLED = 166`
- `OPERATION_CODE_SCHEDULER_CLOCK_UNAVAILABLE = 167`
- `OPERATION_CODE_SHAPE_DOMAIN_ATTESTATION_INVALID = 168`
- `OPERATION_CODE_SHAPE_DOMAIN_UNSUPPORTED = 169`
- `OPERATION_CODE_STATE_ALLOCATION_FAILED = 170`
- `OPERATION_CODE_STATE_ALLOCATION_OVERFLOW = 171`
- `OPERATION_CODE_SUPERSEDED = 172`
- `OPERATION_CODE_TRANSPORT_UNSUPPORTED = 183`
- `OPERATION_CODE_WAIT_FAILED = 184`
- `OPERATION_CODE_WEIGHTS_NOT_READABLE = 185`
- `OPERATION_CODE_WEIGHT_STORE_INVALID = 186`
- `OPERATION_CODE_WEIGHT_STORE_MISSING = 187`
- `OPERATION_CODE_WORKSPACE_ALLOCATION_FAILED = 188`
- `OPERATION_CODE_TRAINING_LOSS_NONFINITE = 190`: A training loss is not finite. TrainStepResult.gradients names the loss.
- `OPERATION_CODE_TRAINING_GRADIENT_NONFINITE = 191`: A gradient is not finite. TrainStepResult.gradients names the parameter.
- `OPERATION_CODE_TOKENIZER_OPERATION_UNSUPPORTED = 192`: The immutable tokenizer profile does not implement the requested operation.
- `OPERATION_CODE_TOKEN_ID_UNKNOWN = 193`: A ByteLevel decode request contains an ID absent from its immutable table.
- `OPERATION_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH = 194`: An accepted source snapshot disagrees with its declared length or digest.

## volvoxai.v1.OperationStage



- `OPERATION_STAGE_NONE = 0`
- `OPERATION_STAGE_RUNTIME_CREATE = 1`
- `OPERATION_STAGE_MODEL_LOAD = 2`
- `OPERATION_STAGE_COMPILE = 3`
- `OPERATION_STAGE_CONTEXT_CREATE = 4`
- `OPERATION_STAGE_INPUT = 5`
- `OPERATION_STAGE_EXECUTE = 6`
- `OPERATION_STAGE_READBACK = 7`
- `OPERATION_STAGE_CLOSE = 8`
- `OPERATION_STAGE_DECODE = 9`
- `OPERATION_STAGE_ADAPTER = 10`
- `OPERATION_STAGE_SUBMIT = 22`
- `OPERATION_STAGE_ADMISSION = 23`
- `OPERATION_STAGE_GRAPH_PLAN_CREATE = 25`
- `OPERATION_STAGE_GRAPH_PLAN_GET = 26`
- `OPERATION_STAGE_GRAPH_PLAN_RESOLVE = 27`
- `OPERATION_STAGE_GRAPH_PLAN_CLOSE = 28`
- `OPERATION_STAGE_SAFETENSORS_INSPECT = 29`
- `OPERATION_STAGE_TOKENIZER_CREATE = 30`
- `OPERATION_STAGE_TOKENIZE = 31`
- `OPERATION_STAGE_DETOKENIZE = 32`
- `OPERATION_STAGE_GRAPH_PLAN_EDIT = 33`
- `OPERATION_STAGE_GRAPH_PLAN_EXPORT = 34`
- `OPERATION_STAGE_SAFETENSORS_READ = 35`
- `OPERATION_STAGE_SAFETENSORS_WRITE = 36`
- `OPERATION_STAGE_TENSOR_INITIALIZE = 37`
- `OPERATION_STAGE_LORA_AUTHOR = 38`
- `OPERATION_STAGE_WEIGHT_QUANTIZE = 39`
- `OPERATION_STAGE_WEIGHT_DEQUANTIZE = 40`

## volvoxai.v1.NativeResourceKind

Resource representation is independent of physical placement and mapping.

- `NATIVE_RESOURCE_KIND_UNSPECIFIED = 0`
- `NATIVE_RESOURCE_KIND_HOST = 1`
- `NATIVE_RESOURCE_KIND_CUDA = 2`
- `NATIVE_RESOURCE_KIND_VULKAN = 3`
- `NATIVE_RESOURCE_KIND_OPENGL = 4`
- `NATIVE_RESOURCE_KIND_METAL = 5`

## volvoxai.v1.BufferAccessMode



- `BUFFER_ACCESS_MODE_READ = 0`
- `BUFFER_ACCESS_MODE_WRITE = 1`

## volvoxai.v1.DimensionKind

One axis of a logical bounded tensor. A FIXED axis has an empty symbol,
min == max, and multiple_of == 1. A SYMBOLIC axis has a non-empty canonical
symbol and positive min/max/multiple_of values. Mixed states are rejected.

Zero is deliberately UNSPECIFIED rather than FIXED so a zero-initialized
descriptor fails validation instead of silently reading as a fixed axis of
extent zero.

- `DIMENSION_KIND_UNSPECIFIED = 0`
- `DIMENSION_KIND_FIXED = 1`
- `DIMENSION_KIND_SYMBOLIC = 2`

## volvoxai.v1.CandidateOutcome



- `CANDIDATE_OUTCOME_UNSPECIFIED = 0`
- `CANDIDATE_OUTCOME_SELECTED = 1`
- `CANDIDATE_OUTCOME_UNAVAILABLE = 2`
- `CANDIDATE_OUTCOME_DOMAIN_UNSUPPORTED = 3`
- `CANDIDATE_OUTCOME_COMPILE_FAILED = 4`
- `CANDIDATE_OUTCOME_OPERATOR_UNSUPPORTED = 5`
- `CANDIDATE_OUTCOME_NOT_EVALUATED = 6`

## volvoxai.v1.ShapePlanOrigin



- `SHAPE_PLAN_ORIGIN_UNSPECIFIED = 0`
- `SHAPE_PLAN_ORIGIN_CACHE_HIT = 1`
- `SHAPE_PLAN_ORIGIN_COLD = 2`

## volvoxai.v1.DecodeMode



- `DECODE_MODE_UNSPECIFIED = 0`
- `DECODE_MODE_DISABLED = 1`
- `DECODE_MODE_PROVIDER = 2`
- `DECODE_MODE_ROW = 3`
- `DECODE_MODE_DEPENDENCY = 4`

## volvoxai.v1.InputValidationCode



- `INPUT_VALIDATION_CODE_UNSPECIFIED = 0`
- `INPUT_VALIDATION_CODE_MISSING_INPUT = 1`
- `INPUT_VALIDATION_CODE_UNKNOWN_INPUT = 2`
- `INPUT_VALIDATION_CODE_DUPLICATE_INPUT = 3`
- `INPUT_VALIDATION_CODE_INVALID_NAME = 4`
- `INPUT_VALIDATION_CODE_DTYPE_MISMATCH = 5`
- `INPUT_VALIDATION_CODE_RANK_MISMATCH = 6`
- `INPUT_VALIDATION_CODE_DIMENSION_OUT_OF_RANGE = 7`
- `INPUT_VALIDATION_CODE_SYMBOL_MISMATCH = 8`
- `INPUT_VALIDATION_CODE_BYTE_SIZE_MISMATCH = 9`
- `INPUT_VALIDATION_CODE_PAYLOAD_REQUIRED = 10`
- `INPUT_VALIDATION_CODE_LOCATION_UNSUPPORTED = 11`
- `INPUT_VALIDATION_CODE_TRANSPORT_UNSUPPORTED = 12`
- `INPUT_VALIDATION_CODE_DECODE_SHAPE_MISMATCH = 13`
- `INPUT_VALIDATION_CODE_SIZE_OVERFLOW = 14`
- `INPUT_VALIDATION_CODE_INVALID_BUFFER_VIEW = 15`

## volvoxai.v1.ApiEffect



- `API_EFFECT_UNSPECIFIED = 0`
- `API_EFFECT_READ_ONLY = 1`
- `API_EFFECT_CREATE = 2`
- `API_EFFECT_RELEASE = 3`
- `API_EFFECT_MUTATE = 4`
- `API_EFFECT_EXECUTE = 5`

## volvoxai.v1.ApiRuleKind



- `API_RULE_KIND_UNSPECIFIED = 0`
- `API_RULE_KIND_EXACTLY_ONE = 1`
- `API_RULE_KIND_REQUIRES = 2`
- `API_RULE_KIND_LIVE_HANDLE = 3`
- `API_RULE_KIND_SCHEDULED_RUNTIME = 4`
- `API_RULE_KIND_PREFILLED_CONTEXT = 5`
- `API_RULE_KIND_COMPLETE_INPUT_BATCH = 6`
- `API_RULE_KIND_IN_PROCESS_ONLY = 7`
- `API_RULE_KIND_HANDLE_STATE = 8`: The handle named by `fields` must be in one of `states` (enum value names
of the handle's state enum), as reported by its Get method.

## volvoxai.v1.PackedArithmetic

Internal arithmetic of PackedLinear only. Graph boundaries and bias are F32.
Compilation resolves UNSPECIFIED to F32, independent of backend, and pins the
result for the executable lifetime. Backend fallback preserves this contract.
Parallel reductions and FMA may differ in their final FP32 rounding; none of
these values promises bitwise equality across devices or execution shapes.

- `PACKED_ARITHMETIC_UNSPECIFIED = 0`
- `PACKED_ARITHMETIC_DYNAMIC_I8 = 1`: Per-row dynamic I8 activations; exact integer group dots, F32 scale/sum.
- `PACKED_ARITHMETIC_F16 = 2`: Input and reconstructed weight round to IEEE F16, then F32 accumulation.
- `PACKED_ARITHMETIC_F32 = 3`: F32 operands/accumulation. Reassociation/FMA allowed; reduced-precision
operand conversion (including TF32) is not allowed.
- `PACKED_ARITHMETIC_F32_FAST = 4`: Explicitly permits compensated TF32 hi+lo activation decomposition with
integer weight operands and F32 accumulation/group scaling. May use F32
kernels on other hardware or shapes. Never selects F16 or dynamic I8.
- `PACKED_ARITHMETIC_DYNAMIC_I4 = 5`: Reserved selection for future per-row dynamic I4 activation arithmetic.
No backend implements it in this release: CompileModel returns
BACKEND_UNSUPPORTED without creating an executable or selecting another
arithmetic. This value does not change weight storage or static affines.

## volvoxai.v1.ExecutionEffect

Typed model execution metadata is required in the graph document's
"execution" member, using these message field names and enum values. The
same contract survives loading, GetModelInfo, planning, edits and export.

- `EXECUTION_EFFECT_UNSPECIFIED = 0`
- `EXECUTION_EFFECT_EVALUATE = 1`
- `EXECUTION_EFFECT_INITIALIZE = 2`
- `EXECUTION_EFFECT_TRANSITION = 3`

## volvoxai.v1.ExecutionRangeKind



- `EXECUTION_RANGE_KIND_UNSPECIFIED = 0`
- `EXECUTION_RANGE_KIND_FULL = 1`
- `EXECUTION_RANGE_KIND_PREFIX_ROWS = 2`
- `EXECUTION_RANGE_KIND_SLOT_ROWS = 3`
- `EXECUTION_RANGE_KIND_DEPENDENCIES = 4`

## volvoxai.v1.ExecutionKind



- `EXECUTION_KIND_UNSPECIFIED = 0`
- `EXECUTION_KIND_GRAPH = 1`
- `EXECUTION_KIND_AUTOREGRESSIVE = 2`

## volvoxai.v1.ExecutionContextState



- `EXECUTION_CONTEXT_STATE_UNSPECIFIED = 0`
- `EXECUTION_CONTEXT_STATE_READY = 1`
- `EXECUTION_CONTEXT_STATE_QUEUED = 2`
- `EXECUTION_CONTEXT_STATE_RUNNING = 3`
- `EXECUTION_CONTEXT_STATE_COMPLETED = 4`
- `EXECUTION_CONTEXT_STATE_CANCELLED = 5`
- `EXECUTION_CONTEXT_STATE_FAILED = 6`

## volvoxai.v1.ExecutionStopReason



- `EXECUTION_STOP_REASON_UNSPECIFIED = 0`
- `EXECUTION_STOP_REASON_STEP_LIMIT = 1`
- `EXECUTION_STOP_REASON_CANCELLED = 2`
- `EXECUTION_STOP_REASON_FAILED = 3`
- `EXECUTION_STOP_REASON_EOS = 4`
- `EXECUTION_STOP_REASON_ID_STOP = 5`

## volvoxai.v1.SamplingMode



- `SAMPLING_MODE_UNSPECIFIED = 0`
- `SAMPLING_MODE_GREEDY = 1`
- `SAMPLING_MODE_RANDOM = 2`: Seeded categorical selection with the options in AutoregressiveExecution.
Backend admission remains explicit; no hidden host numerical fallback.

## volvoxai.v1.OutputStorage



- `OUTPUT_STORAGE_UNSPECIFIED = 0`
- `OUTPUT_STORAGE_INLINE = 1`
- `OUTPUT_STORAGE_OWNED_BUFFER = 2`

## volvoxai.v1.DecodeCachePolicy



- `DECODE_CACHE_POLICY_PAGED = 0`
- `DECODE_CACHE_POLICY_CONTIGUOUS = 1`

## volvoxai.v1.RequestFreshness



- `REQUEST_FRESHNESS_ALL = 0`: Every admitted request remains eligible until cancelled or completed.
- `REQUEST_FRESHNESS_LATEST = 1`: A request with the same compiled route and nonzero stream key is
superseded by a newer admitted LATEST request. Queued replacement is
transactional: equal payload storage may be reused across dynamic shapes,
while a larger payload requires temporary byte-budget headroom. Already
submitted work remains physical but its result is discarded at
completion.
- `REQUEST_FRESHNESS_DROP_IF_LATE = 2`: Refuse or retire queued/completed work whose monotonic deadline elapsed.

## volvoxai.v1.RequestState



- `REQUEST_STATE_UNSPECIFIED = 0`
- `REQUEST_STATE_QUEUED = 1`
- `REQUEST_STATE_RUNNING = 2`
- `REQUEST_STATE_SUCCEEDED = 3`
- `REQUEST_STATE_CANCELLED = 4`
- `REQUEST_STATE_FAILED = 5`
- `REQUEST_STATE_SUPERSEDED = 6`

## volvoxai.v1.MemorySpace



- `MEMORY_SPACE_UNSPECIFIED = 0`
- `MEMORY_SPACE_HOST = 1`
- `MEMORY_SPACE_JS_HEAP = 2`
- `MEMORY_SPACE_JS_EXTERNAL = 3`
- `MEMORY_SPACE_JS_ARRAY_BUFFER = 4`
- `MEMORY_SPACE_WASM_LINEAR = 5`
- `MEMORY_SPACE_NATIVE_HEAP = 6`
- `MEMORY_SPACE_MAPPED_FILE = 7`
- `MEMORY_SPACE_DEVICE = 8`: An API-visible device allocation whose physical placement is unknown.
- `MEMORY_SPACE_DEVICE_LOCAL = 9`
- `MEMORY_SPACE_HOST_VISIBLE_DEVICE = 10`
- `MEMORY_SPACE_UNIFIED = 11`

## volvoxai.v1.MemoryOwnerKind

The lifetime owner is not necessarily the object currently using a
resource. In particular, contexts can consume one immutable
COMPILED_MODEL-owned weight allocation without owning copies of it.

- `MEMORY_OWNER_KIND_UNSPECIFIED = 0`
- `MEMORY_OWNER_KIND_PROCESS = 1`
- `MEMORY_OWNER_KIND_RUNTIME = 2`
- `MEMORY_OWNER_KIND_MODEL = 3`
- `MEMORY_OWNER_KIND_COMPILED_MODEL = 4`
- `MEMORY_OWNER_KIND_EXECUTION_CONTEXT = 5`
- `MEMORY_OWNER_KIND_RESULT = 6`
- `MEMORY_OWNER_KIND_BACKEND_SHARED = 7`
- `MEMORY_OWNER_KIND_GRAPH_PLAN = 10`: Logical planning runs in every profile; VxPlanningService, which publishes
plan handles, is full-only.
- `MEMORY_OWNER_KIND_MODULE = 11`: Every known owner in one engine module.

## volvoxai.v1.MemoryResourceRole



- `MEMORY_RESOURCE_ROLE_UNSPECIFIED = 0`
- `MEMORY_RESOURCE_ROLE_WEIGHTS = 1`
- `MEMORY_RESOURCE_ROLE_PACKED_WEIGHTS = 2`
- `MEMORY_RESOURCE_ROLE_INPUT = 3`
- `MEMORY_RESOURCE_ROLE_ACTIVATION = 4`
- `MEMORY_RESOURCE_ROLE_KV_CACHE = 5`
- `MEMORY_RESOURCE_ROLE_SCRATCH = 6`
- `MEMORY_RESOURCE_ROLE_RESULT_SNAPSHOT = 7`
- `MEMORY_RESOURCE_ROLE_ARENA = 8`
- `MEMORY_RESOURCE_ROLE_PLAN_CACHE = 9`
- `MEMORY_RESOURCE_ROLE_KERNEL_CACHE = 10`
- `MEMORY_RESOURCE_ROLE_METADATA = 11`
- `MEMORY_RESOURCE_ROLE_RUNTIME_OVERHEAD = 12`
- `MEMORY_RESOURCE_ROLE_ALLOCATOR_OVERHEAD = 13`
- `MEMORY_RESOURCE_ROLE_DRIVER_OVERHEAD = 14`
- `MEMORY_RESOURCE_ROLE_OTHER = 15`
- `MEMORY_RESOURCE_ROLE_PROFILER = 21`
- `MEMORY_RESOURCE_ROLE_STAGING = 22`

## volvoxai.v1.MemoryBoundKind

Each bound kind is an independent number. Bounds of different kinds are
never summed. A decode bound, for example, is an alternative to the
ordinary resident bound rather than an additional allocation.

- `MEMORY_BOUND_KIND_UNSPECIFIED = 0`
- `MEMORY_BOUND_KIND_MAXIMUM_TENSOR = 1`
- `MEMORY_BOUND_KIND_CONTEXT_CAPACITY = 2`
- `MEMORY_BOUND_KIND_ORDINARY_RESIDENT = 3`
- `MEMORY_BOUND_KIND_DECODE_RESIDENT = 4`

## volvoxai.v1.MemoryInventoryKind

Whether all live backing allocations in the stated inventory scope are
represented. COMPLETE concerns inventory coverage, not physical residency,
tensor liveness or other scopes. An empty COMPLETE inventory means no live
allocations in that scope; an empty PARTIAL inventory remains unknown.

- `MEMORY_INVENTORY_KIND_UNSPECIFIED = 0`
- `MEMORY_INVENTORY_KIND_PARTIAL = 1`
- `MEMORY_INVENTORY_KIND_COMPLETE = 2`

## volvoxai.v1.TraceDetail



- `TRACE_DETAIL_BASIC = 0`: Host operations and, when device timing is enabled, existing device passes.
- `TRACE_DETAIL_NODES = 1`: Also collect executable nodes, training work and supported GPU programs.
Actual pass/node/program coverage is reported in TraceInfo.devices.

## volvoxai.v1.TracePhase

The work that actually executed, independently of its host/device clock.
UNSPECIFIED includes outer operations and observations with no known phase.

- `TRACE_PHASE_UNSPECIFIED = 0`
- `TRACE_PHASE_FORWARD = 1`
- `TRACE_PHASE_LOSS = 2`
- `TRACE_PHASE_BACKWARD = 3`
- `TRACE_PHASE_GRADIENT = 4`: Initialization, accumulation, norm and clipping.
- `TRACE_PHASE_OPTIMIZER = 5`

## volvoxai.v1.TraceState



- `TRACE_STATE_UNSPECIFIED = 0`
- `TRACE_STATE_COLLECTING = 1`
- `TRACE_STATE_DRAINING = 2`: Stopped; accepted host operations or device timestamps are still pending.
- `TRACE_STATE_READY = 3`: Immutable and readable.

## volvoxai.v1.TraceTimingSupport

Timestamp support observed on a backend's encountered device paths.
UNSPECIFIED means no device timing path was encountered.

- `TRACE_TIMING_SUPPORT_UNSPECIFIED = 0`
- `TRACE_TIMING_SUPPORT_AVAILABLE = 1`
- `TRACE_TIMING_SUPPORT_UNSUPPORTED = 2`
- `TRACE_TIMING_SUPPORT_MIXED = 3`: The same backend encountered devices/passes with different support.

## volvoxai.v1.TraceClockMethod



- `TRACE_CLOCK_METHOD_UNSPECIFIED = 0`
- `TRACE_CLOCK_METHOD_CALIBRATED = 1`: A device clock sample is related to host time with measured uncertainty.
Mapping is local to this pass; it is not a permanent clock conversion.
- `TRACE_CLOCK_METHOD_BOUNDED = 2`: Causal submission/completion bounds only, not clock calibration.

## volvoxai.v1.TraceActivity



- `TRACE_ACTIVITY_UNSPECIFIED = 0`: Memory events carry no activity.
- `TRACE_ACTIVITY_COMPUTE = 1`: Operator or kernel execution (CUPTI KERNEL, Kineto cpu_op/kernel).
- `TRACE_ACTIVITY_COPY = 2`: CUPTI MEMCPY.
- `TRACE_ACTIVITY_SUBMIT = 3`: Command submission (Vulkan/WebGPU submit, CUDA launch).
- `TRACE_ACTIVITY_SYNCHRONIZE = 4`: Host blocking call, including scheduling overhead (CUPTI SYNCHRONIZATION).
- `TRACE_ACTIVITY_COMPLETION = 5`: Observed asynchronous completion (fence, onSubmittedWorkDone); not CPU
blocking time.
- `TRACE_ACTIVITY_ANNOTATION = 6`: Application range from AnnotateTrace.

## volvoxai.v1.TraceMemoryAction

EXISTING introduces an allocation at the first inventory of its owner; it
does not claim the allocation happened during this trace. Identities are
opaque and unique within the trace, including after address reuse.

- `TRACE_MEMORY_ACTION_UNSPECIFIED = 0`
- `TRACE_MEMORY_ACTION_EXISTING = 1`
- `TRACE_MEMORY_ACTION_ALLOCATE = 2`
- `TRACE_MEMORY_ACTION_FREE = 3`

## volvoxai.v1.CostStatus



- `COST_STATUS_UNSPECIFIED = 0`
- `COST_STATUS_EXACT = 1`: Closed-form count for the concrete shapes.
- `COST_STATUS_ESTIMATED = 2`: An upper or representative count: attention with masked keys, packed
weights, normalization and windowed reductions.
- `COST_STATUS_UNKNOWN = 3`: No model for this operator; the counts are zero, not measured zero.

## volvoxai.v1.TraceSummaryGrouping



- `TRACE_SUMMARY_GROUPING_UNSPECIFIED = 0`
- `TRACE_SUMMARY_GROUPING_NODE = 1`: One row per executable node of a plan ("slow nodes").
- `TRACE_SUMMARY_GROUPING_OPERATOR = 2`: One row per backend, phase and operator type.
- `TRACE_SUMMARY_GROUPING_OPERATION = 3`: One row per public call, such as Execute or TrainStep.
- `TRACE_SUMMARY_GROUPING_ACTIVITY = 4`: Copies, submissions, blocking waits and asynchronous completions.

## volvoxai.v1.TraceTimeDomain



- `TRACE_TIME_DOMAIN_UNSPECIFIED = 0`
- `TRACE_TIME_DOMAIN_HOST = 1`: TraceEvent.host spans
- `TRACE_TIME_DOMAIN_DEVICE = 2`: TraceEvent.device elapsed intervals

## volvoxai.v1.ObservationStatus



- `OBSERVATION_STATUS_UNSPECIFIED = 0`
- `OBSERVATION_STATUS_AVAILABLE = 1`
- `OBSERVATION_STATUS_UNSUPPORTED = 2`
- `OBSERVATION_STATUS_NOT_COLLECTED = 3`
- `OBSERVATION_STATUS_FAILED = 4`
