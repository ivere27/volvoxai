/*
 * Receipt digit reader inference helper built on the public generated API.
 *
 * This helper owns package-manifest validation, the shaped input view, and
 * slot decoding. Compilation, shape proof, strict backend routing, execution,
 * and lifecycle cleanup all flow through the selected profile host plus the
 * generated protobuf clients.
 */

import { prepareReceiptImage } from './ReceiptDigitInput.js';

const PACKAGE_FORMAT = 'volvoxai-receipt-digit-reader-onnx-package-v1';
const MAX_BATCH_SIZE = 32;

function fail(message) {
  throw new Error(`[ReceiptDigitSession] ${message}`);
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function asBigIntShape(shape) {
  return shape.map((extent) => BigInt(extent));
}

function assertManifest(manifest) {
  if (!isRecord(manifest) || manifest.format !== PACKAGE_FORMAT) {
    fail(`package manifest must declare format '${PACKAGE_FORMAT}'.`);
  }
  const { abi, decode, preprocess } = manifest;
  if (!isRecord(abi) || !isRecord(abi.input) || !Array.isArray(abi.input.shape)
      || abi.input.shape.length !== 4 || typeof abi.input.name !== 'string'
      || typeof abi.output !== 'string') {
    fail('manifest.abi must declare a rank-4 input name/shape and one output name.');
  }
  if (abi.input.shape.some((extent) => !Number.isSafeInteger(extent) || extent <= 0)
      || abi.input.shape[0] !== 1) {
    fail('manifest.abi.input.shape must be a positive-integer per-request shape with batch 1.');
  }
  const batch = abi.batch;
  if (!isRecord(batch) || batch.per_request !== 1 || batch.min !== 1
      || batch.multiple_of !== 1 || !Number.isSafeInteger(batch.max)
      || batch.max < 1 || batch.max > MAX_BATCH_SIZE) {
    fail(`manifest.abi.batch must declare per_request=min=multiple_of=1 and max 1..${MAX_BATCH_SIZE}.`);
  }
  if (batch.max > 1 && (typeof batch.symbol !== 'string' || batch.symbol.length === 0)) {
    fail('manifest.abi.batch.symbol must name the dynamic leading axis when max is greater than 1.');
  }
  if (batch.max === 1 && batch.symbol !== null) {
    fail('manifest.abi.batch.symbol must be null for a static batch-1 graph.');
  }
  for (const key of ['slots', 'phone_slots', 'street_slots', 'blank_class', 'num_classes']) {
    if (!Number.isInteger(decode?.[key]) || decode[key] < 0) {
      fail(`manifest.decode.${key} must be a non-negative integer.`);
    }
  }
  if (decode.phone_slots + decode.street_slots !== decode.slots) {
    fail('manifest.decode phone_slots + street_slots must equal slots.');
  }
  if (decode.blank_class >= decode.num_classes) {
    fail('manifest.decode.blank_class must be a valid class index.');
  }
  if (!isRecord(preprocess) || !Number.isInteger(preprocess.width)
      || !Number.isInteger(preprocess.height)) {
    fail('manifest.preprocess must declare integer width and height.');
  }
  return manifest;
}

function buildCreateRuntimeRequest(pb, execution) {
  if (execution === undefined) return new pb.CreateRuntimeRequest();
  if (!isRecord(execution)) fail('execution must be an object when provided.');
  const request = new pb.CreateRuntimeRequest();
  const mode = execution.mode;
  if (mode !== undefined) {
    if (mode === 'direct') {
      request.executionMode = pb.ExecutionMode.EXECUTION_MODE_DIRECT;
    } else if (mode === 'scheduled') {
      request.executionMode = pb.ExecutionMode.EXECUTION_MODE_SCHEDULED;
    } else {
      fail("execution.mode must be 'direct' or 'scheduled'.");
    }
  }
  const scheduler = execution.scheduler;
  const results = execution.results;
  if (scheduler !== undefined || results !== undefined) {
    request.budget = new pb.RuntimeBudget();
  }
  if (isRecord(scheduler)) {
    if (scheduler.maxRequests !== undefined) {
      request.budget.maxScheduledRequests = BigInt(scheduler.maxRequests);
    }
    if (scheduler.maxInputBytes !== undefined) {
      request.budget.maxScheduledInputBytes = BigInt(scheduler.maxInputBytes);
    }
    if (scheduler.maxBatchDelayMs !== undefined) {
      request.budget.maxBatchDelayMilliseconds = scheduler.maxBatchDelayMs;
    }
  }
  if (isRecord(results)) {
    if (results.maxRetainedResults !== undefined) {
      request.budget.maxUnconsumedResults = BigInt(results.maxRetainedResults);
    }
    if (results.maxRetainedOutputBytes !== undefined) {
      request.budget.maxUnconsumedResultBytes = BigInt(results.maxRetainedOutputBytes);
    }
  }
  return request;
}

function strictCompiledBackend(pb, report, requiredBackend) {
  if (!requiredBackend) return report?.backend || '';
  const selected = report?.compilation?.candidates?.filter(
    (candidate) => candidate.outcome === pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED,
  ) ?? [];
  if (report.backend !== requiredBackend
      || report.compilation?.policyMode !== pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE
      || report.compilation?.operatorFallback !== pb.OperatorFallback.OPERATOR_FALLBACK_FORBID
      || selected.length !== 1
      || selected[0].backend !== requiredBackend) {
    fail(`CompileModel did not attest the required '${requiredBackend}' backend.`);
  }
  return requiredBackend;
}

function inputTensor(pb, name, shape, data) {
  if (!(data instanceof Float32Array)) {
    fail(`input '${name}' must be a Float32Array.`);
  }
  return new pb.Tensor({
    name,
    shape: asBigIntShape(shape),
    dtype: pb.DataType.DATA_TYPE_F32,
    inline: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  });
}

function readInlineF32(pb, tensor, outputName) {
  if (tensor?.dtype !== pb.DataType.DATA_TYPE_F32 || tensor.inline === undefined) {
    fail(`output '${outputName}' is not an inline F32 tensor.`);
  }
  const bytes = tensor.inline.slice();
  if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    fail(`output '${outputName}' has an unaligned byte length.`);
  }
  return new Float32Array(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );
}

/**
 * Read one slot row block until the blank class.
 *
 * Decoding stops at the first blank rather than dropping blanks, because the
 * slots are left-aligned by construction: a blank in the middle means the
 * model ended the number there, and skipping it would silently splice unrelated
 * digits together.
 */
export function decodeSlots(logits, decode) {
  const { slots, phone_slots: phoneSlots, blank_class: blank, num_classes: classes } = decode;
  if (!ArrayBuffer.isView(logits) || logits.length !== slots * classes) {
    fail(`logits must hold exactly ${slots} x ${classes} values.`);
  }
  const argmax = [];
  for (let slot = 0; slot < slots; slot++) {
    let best = 0;
    for (let candidate = 1; candidate < classes; candidate++) {
      if (logits[slot * classes + candidate] > logits[slot * classes + best]) best = candidate;
    }
    argmax.push(best);
  }
  const read = (from, to) => {
    let digits = '';
    for (let slot = from; slot < to; slot++) {
      if (argmax[slot] === blank) break;
      digits += String(argmax[slot]);
    }
    return digits;
  };
  return {
    phone: read(0, phoneSlots),
    street: read(phoneSlots, slots),
    slotClasses: argmax,
  };
}

export class ReceiptDigitSession {
  #inference;

  #pb;

  #compiledModelId;

  #manifest;

  #backend;

  #host;

  #closePromise = null;

  constructor({ host, inference, pb, compiledModelId, manifest, backend }) {
    this.#host = host;
    this.#inference = inference;
    this.#pb = pb;
    this.#compiledModelId = compiledModelId;
    this.#manifest = manifest;
    this.#backend = backend;
  }

  get manifest() {
    return this.#manifest;
  }

  get backend() {
    return this.#backend;
  }

  static async open({
    manifest,
    graphUrl,
    weightsUrl,
    backend,
    fetch: fetchImpl,
    wasmUrl,
    execution,
    api,
  }) {
    const resolved = assertManifest(manifest);
    const runtimeApi = api ?? await import('../../ts/index.js');
    const { FullEngineHost, EngineHost, VxInferenceServiceClient, pb } = runtimeApi;
    const Host = FullEngineHost ?? EngineHost;
    const usesFullProfile = typeof FullEngineHost === 'function';
    if (backend === 'webgpu' && !usesFullProfile) {
      fail('the webgpu backend requires the full API profile.');
    }
    if (typeof Host !== 'function'
        || typeof VxInferenceServiceClient !== 'function'
        || !pb) {
      fail('api must expose a profile host, VxInferenceServiceClient, and pb.');
    }
    const host = new Host({
      ...(fetchImpl ? { fetch: fetchImpl } : {}),
      ...(wasmUrl ? { wasmUrl } : {}),
    });
    const inference = new VxInferenceServiceClient(host);
    try {
      const runtime = await inference.createRuntime(
        buildCreateRuntimeRequest(pb, execution),
      );
      const { runtimeId } = runtime;

      const model = await inference.loadModel(new pb.LoadModelRequest({
        runtimeId,
        graphPath: String(graphUrl),
        weightPaths: [String(weightsUrl)],
      }));
      const { modelId } = model;

      const compiled = await inference.compileModel(new pb.CompileModelRequest({
        modelId,
        ...(backend
          ? {
            policy: new pb.BackendPolicy({
              mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
              backends: [backend],
              operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
            }),
          }
          : {}),
      }));
      const { compiledModelId } = compiled;
      const usedBackend = strictCompiledBackend(pb, compiled.report, backend);

      return new ReceiptDigitSession({
        host,
        inference,
        pb,
        compiledModelId,
        manifest: resolved,
        backend: usedBackend,
      });
    } catch (error) {
      try {
        await host.close();
      } catch (cleanupError) {
        throw new AggregateError([error, cleanupError], '[ReceiptDigitSession] open failed.');
      }
      throw error;
    }
  }

  async #executeReceipt(image, measureExecution) {
    if (this.#closePromise !== null) fail('session is closed.');
    const { abi, decode, preprocess } = this.#manifest;
    const data = ArrayBuffer.isView(image) && !(image instanceof DataView)
      ? Float32Array.from(image)
      : prepareReceiptImage(image, { width: preprocess.width, height: preprocess.height });
    const expected = abi.input.shape.reduce((total, value) => total * value, 1);
    if (data.length !== expected) {
      fail(`preprocessed input holds ${data.length} values, expected ${expected}.`);
    }
    const started = measureExecution ? performance.now() : 0;
    const result = await this.#inference.run(new this.#pb.RunRequest({
      compiledModelId: this.#compiledModelId,
      inputs: [inputTensor(this.#pb, abi.input.name, abi.input.shape, data)],
    }));
    try {
      const response = await this.#inference.readOutput(new this.#pb.ReadOutputRequest({
        resultId: result.resultId,
        name: abi.output,
      }));
      const logits = readInlineF32(this.#pb, response.tensor, abi.output);
      const executionMs = measureExecution ? performance.now() - started : null;
      return Object.freeze({
        record: decodeSlots(logits, decode),
        executionMs,
      });
    } finally {
      if (this.#closePromise === null) {
        await this.#inference.releaseResult(new this.#pb.ResultRef({
          resultId: result.resultId,
        }));
      }
    }
  }

  async read(image) {
    return (await this.#executeReceipt(image, false)).record;
  }

  async readForBenchmark(image) {
    return this.#executeReceipt(image, true);
  }

  async close() {
    if (this.#closePromise === null) {
      this.#compiledModelId = 0n;
      this.#closePromise = this.#host.close();
    }
    return this.#closePromise;
  }
}
