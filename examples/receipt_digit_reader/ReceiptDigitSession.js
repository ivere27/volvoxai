/*
 * Receipt digit reader inference session.
 *
 * The graph reads one receipt and emits one record: a phone number and a
 * street number, each as left-aligned digit slots terminated by a blank class.
 * The graph never sees a question -- question handling is regex-only and lives
 * in questionRouter.js -- so several questions about one receipt cost exactly
 * one forward pass.
 *
 * This session owns package-manifest validation, the shaped input view, and
 * slot decoding. Compilation, shape proof, backend selection, and execution
 * stay in VolvoxAI.
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

function assertGraphBatchContract(snapshot, manifest) {
  const graph = snapshot?.graph;
  if (!isRecord(graph) || !isRecord(graph.inputs) || !isRecord(graph.tensors)
      || !isRecord(graph.dimensions) || !Array.isArray(graph.outputs)) {
    fail('loaded model does not expose a logical graph batch contract.');
  }
  const { input, output, batch } = manifest.abi;
  const inputNames = Object.keys(graph.inputs);
  if (inputNames.length !== 1 || inputNames[0] !== input.name) {
    fail(`graph must expose only manifest input '${input.name}'.`);
  }
  const graphInputShape = graph.inputs[input.name]?.shape;
  if (!Array.isArray(graphInputShape) || graphInputShape.length !== input.shape.length
      || graphInputShape.slice(1).some((extent, axis) => extent !== input.shape[axis + 1])) {
    fail(`graph input '${input.name}' shape does not match the manifest per-request ABI.`);
  }
  if (graph.outputs.length !== 1 || graph.outputs[0] !== output) {
    fail(`graph must expose only manifest output '${output}'.`);
  }
  const publicOutputShapes = graph.outputs.map((name) => graph.tensors[name]?.shape);
  if (publicOutputShapes.some((shape) => !Array.isArray(shape) || shape.length === 0)) {
    fail('every graph output must expose a non-scalar logical shape.');
  }

  if (batch.symbol === null) {
    if (graphInputShape[0] !== 1 || publicOutputShapes.some((shape) => shape[0] !== 1)) {
      fail('a static batch-1 manifest requires literal leading extent 1 on every public graph value.');
    }
    return;
  }

  if (graphInputShape[0] !== batch.symbol
      || publicOutputShapes.some((shape) => shape[0] !== batch.symbol)) {
    fail(`dynamic batch symbol '${batch.symbol}' must lead the public graph input and every output.`);
  }
  const domain = graph.dimensions[batch.symbol];
  if (!isRecord(domain) || domain.min !== batch.min || domain.max !== batch.max
      || domain.multiple_of !== batch.multiple_of) {
    fail(`graph dimension '${batch.symbol}' must exactly match manifest.abi.batch.`);
  }
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
  #runtime;

  #compiled;

  #manifest;

  #ownsRuntime;

  #closePromise = null;

  constructor(runtime, compiled, manifest, ownsRuntime) {
    this.#runtime = runtime;
    this.#compiled = compiled;
    this.#manifest = manifest;
    this.#ownsRuntime = ownsRuntime;
  }

  get manifest() {
    return this.#manifest;
  }

  /**
   * Open one session against a published package.
   *
   * `backend` is passed through as a strict requirement by default. A reader
   * deployed on a chosen device should fail loudly rather than silently land
   * on a slower provider, and operator fallback would defeat the point of the
   * byte-domain variants.
   */
  static async open({
    manifest, graphUrl, weightsUrl, runtime, backends, backend, fetch: fetchImpl, wasmUrl,
    execution, api,
  }) {
    const resolved = assertManifest(manifest);
    if (runtime !== undefined && execution !== undefined) {
      fail('execution belongs to Runtime creation; a supplied Runtime owns its execution policy.');
    }
    // `api` lets a host supply an already-loaded runtime module. The fallback
    // import is deliberately dynamic: a static one is resolved whether or not
    // it is reached, and neither Deno nor a browser can resolve the
    // repository's TypeScript entry, so both would fail before `api` was ever
    // consulted.
    const { Model, VolvoxAI } = api ?? await import('../../ts/index.js');
    if (typeof Model?.load !== 'function' || typeof VolvoxAI?.createRuntime !== 'function') {
      fail('api must expose Model.load and VolvoxAI.createRuntime.');
    }
    const ownsRuntime = runtime === undefined;
    const activeRuntime = runtime ?? await VolvoxAI.createRuntime({
      backends: backends ?? ['wasm', 'cpu-js'],
      ...(wasmUrl ? { wasmUrl } : {}),
      ...(execution ? { execution } : {}),
    });
    let compiled;
    try {
      const snapshot = await Model.load(weightsUrl, {
        graphUrl, ...(fetchImpl ? { fetch: fetchImpl } : {}),
      });
      assertGraphBatchContract(snapshot, resolved);
      compiled = await activeRuntime.compile(snapshot, backend
        ? { backend: { mode: 'require', backend, operatorFallback: 'forbid' } }
        : undefined);
    } catch (error) {
      await compiled?.close().catch(() => undefined);
      if (ownsRuntime) await activeRuntime.close().catch(() => undefined);
      throw error;
    }
    return new ReceiptDigitSession(
      activeRuntime, compiled, resolved, ownsRuntime,
    );
  }

  async #executeReceipt(image, runOptions, measureExecution) {
    const { abi, decode, preprocess } = this.#manifest;
    const data = ArrayBuffer.isView(image) && !(image instanceof DataView)
      ? Float32Array.from(image)
      : prepareReceiptImage(image, { width: preprocess.width, height: preprocess.height });
    const expected = abi.input.shape.reduce((total, value) => total * value, 1);
    if (data.length !== expected) {
      fail(`preprocessed input holds ${data.length} values, expected ${expected}.`);
    }
    const started = measureExecution ? performance.now() : 0;
    const result = await this.#compiled.run({
      [abi.input.name]: { data, shape: [...abi.input.shape] },
    }, runOptions);
    try {
      const logits = await result.output(abi.output).read();
      // Match the native and ONNX Runtime benchmark boundary: stop after the
      // owned host output exists, before application slot decoding and result
      // cleanup. Input conversion above is likewise outside this interval.
      const executionMs = measureExecution ? performance.now() - started : null;
      return Object.freeze({
        record: decodeSlots(logits, decode),
        executionMs,
      });
    } finally {
      await result.close();
    }
  }

  /** Run one receipt and return its decoded record. */
  async read(image, runOptions) {
    return (await this.#executeReceipt(image, runOptions, false)).record;
  }

  /**
   * Benchmark-only execution boundary used by the example harness.
   *
   * The returned interval excludes input conversion, slot decoding, and
   * result cleanup so it is comparable to the native and ORT routes.
   */
  async readForBenchmark(image, runOptions) {
    return this.#executeReceipt(image, runOptions, true);
  }

  async close() {
    if (this.#closePromise === null) {
      this.#closePromise = (async () => {
        const tasks = [Promise.resolve().then(() => this.#compiled.close())];
        if (this.#ownsRuntime) {
          tasks.push(Promise.resolve().then(() => this.#runtime.close()));
        }
        const failures = (await Promise.allSettled(tasks))
          .filter((result) => result.status === 'rejected')
          .map((result) => result.reason);
        if (failures.length === 1) throw failures[0];
        if (failures.length > 1) {
          throw new AggregateError(failures, '[ReceiptDigitSession] close failed.');
        }
      })();
    }
    return this.#closePromise;
  }
}
