/**
 * Model-neutral activation calibration primitives.
 *
 * Weights quantize data-free; activation affines can only come from running the
 * graph on real inputs and watching the ranges. This module owns that mechanism
 * and nothing else: it knows about graphs, tensors and ranges, and deliberately
 * knows nothing about what the model is for. Callers supply the input feeds, so
 * preprocessing, tokenization, routing and record selection stay in the
 * application tool that owns those contracts.
 *
 * The generated Debug service observes the retained C model without selecting
 * a quantization policy. Tensor groups bound captured statistics.
 *
 * Ranges accumulate across calls, so a sweep may be split over several batches
 * and several input sets without losing earlier observations.
 */

import { p, ok, tensors } from '../proto_fixture.mjs';
import { VxDebugServiceClient, VxInferenceServiceClient } from '../../runtime/generated/typescript/volvoxai_ffi.js';

/** Every float32 tensor a node produces — the PTQ candidates.
 *
 * Graph outputs are included. They need no `setOutputs` promotion — they are
 * already readable — but they still need *observing*, and excluding them from
 * the sweep is not the same thing: it leaves a model's final projection with no
 * calibrated range, so the byte domain ends one node early.
 */
export function candidateTensorNames(document) {
  const names = [...(document.outputs || [])];
  for (const node of document.nodes || []) {
    for (const output of Object.values(node.outputs || {})) {
      if (typeof output?.tensor === 'string') names.push(output.tensor);
    }
  }
  return [...new Set(names)];
}

/** Merge one tensor's finite min/max into `observations`, rejecting NaN/Inf. */
export function observeFiniteRange(observations, name, values, label = 'tensor') {
  if (!observations || typeof observations !== 'object') {
    throw new Error('finite range observation requires an observations object');
  }
  if (typeof name !== 'string' || name.length === 0
      || values == null || typeof values[Symbol.iterator] !== 'function') {
    throw new Error(`${label} '${name}' has no iterable values to observe`);
  }
  let minimum = Infinity;
  let maximum = -Infinity;
  let count = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error(`${label} '${name}' contains a non-finite value at index ${count}`);
    }
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
    count++;
  }
  if (count === 0) throw new Error(`${label} '${name}' has no values to observe`);

  const previous = observations[name];
  if (previous !== undefined
      && (!Number.isFinite(previous?.min) || !Number.isFinite(previous?.max)
        || previous.min > previous.max
        || !Number.isSafeInteger(previous.samples) || previous.samples <= 0
        || !Number.isSafeInteger(previous.elements) || previous.elements < previous.samples)) {
    throw new Error(`${label} '${name}' has an invalid existing calibration range`);
  }
  const samples = (previous?.samples ?? 0) + 1;
  const elements = (previous?.elements ?? 0) + count;
  if (!Number.isSafeInteger(samples) || !Number.isSafeInteger(elements)) {
    throw new Error(`${label} '${name}' calibration counts exceed safe integers`);
  }
  observations[name] = previous
    ? {
      min: Math.min(previous.min, minimum),
      max: Math.max(previous.max, maximum),
      samples,
      elements,
    }
    : { min: minimum, max: maximum, samples, elements };
  return observations[name];
}

/** Observe the float32 public inputs of one executed sample. */
export function observeInputs(document, inputs, observations, excluded = []) {
  const excludedNames = new Set(excluded);
  for (const [name, descriptor] of Object.entries(document.inputs || {})) {
    if ((descriptor?.dtype ?? 'float32') !== 'float32' || excludedNames.has(name)) continue;
    const values = inputs[name]?.data ?? inputs[name];
    // Accumulate across samples like the sweep does; overwriting would keep
    // only the last sample's range and silently undo the multi-sample pass.
    observeFiniteRange(observations, name, values, 'graph input');
  }
}

/**
 * An immutable Embedding is a finite lookup table, so its complete output
 * domain is knowable without guessing which IDs representative decode happens
 * to visit. Merge the whole table range into the observed output range. This is
 * essential for autoregressive models: a BOS/PAD seed alone otherwise clips
 * legal token rows before the first decoder block.
 */
export function widenImmutableEmbeddingRanges(document, tensorLookup, observations) {
  if (typeof tensorLookup !== 'function' || observations == null
      || typeof observations !== 'object') {
    throw new Error('embedding range widening requires a tensor lookup and observations');
  }
  let widened = 0;
  for (const node of document.nodes || []) {
    if (node?.opType !== 'Embedding') continue;
    const weightName = node.inputs?.weight;
    const outputName = node.outputs?.out?.tensor;
    if (typeof weightName !== 'string' || typeof outputName !== 'string') {
      throw new Error('Embedding calibration requires exact weight and out ports');
    }
    const weight = tensorLookup(weightName);
    const values = weight?.data ?? weight?.buffer;
    if (!weight || weight.dtype !== 'float32'
        || !(values instanceof Float32Array) || values.length === 0) {
      throw new Error(
        `Embedding '${outputName}' requires a finite immutable F32 table '${weightName}'`,
      );
    }
    let minimum = Infinity;
    let maximum = -Infinity;
    for (const value of values) {
      if (!Number.isFinite(value)) {
        throw new Error(`Embedding table '${weightName}' contains non-finite values`);
      }
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    const previous = observations[outputName];
    observations[outputName] = previous
      ? {
        ...previous,
        min: Math.min(previous.min, minimum),
        max: Math.max(previous.max, maximum),
      }
      : {
        min: minimum,
        max: maximum,
        samples: 1,
        elements: values.length,
      };
    widened++;
  }
  return widened;
}

/** Observe arbitrary F32 tensors through bounded C debug statistics.
 * This is independent of quantization eligibility. No raw tensor snapshots are
 * retained, and observations commit only after every requested batch succeeds.
 */
export async function observeActivations({
  host, modelId, document, inputSets, names, observations,
}) {
  const descriptors = new Map(Object.entries(document.inputs));
  for (const node of document.nodes)
    for (const output of Object.values(node.outputs)) descriptors.set(output.tensor, output);
  const selected = [...new Set(names)].filter(name => descriptors.get(name)?.dtype === 'float32');
  if (!selected.length) return 0;
  const inference = new VxInferenceServiceClient(host);
  const debug = new VxDebugServiceClient(host);
  const compiled = ok(await inference.compileModel(new p.CompileModelRequest({modelId,
    preserveNodeBoundaries: true,
    policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm']})})));
  let context;
  const pending = {};
  try {
    context = ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, graph: new p.GraphExecution()})));
    for (const inputs of inputSets) {
      let session;
      try {
        session = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({
          execute: new p.ExecuteRequest({contextId: context.contextId,
            inputs: new p.InputBindings({values: tensors(inputs)}),
            scope: new p.ExecutionScope({entrypoint: 'evaluate', full: new p.Empty()})}),
          capture: new p.DebugCapture({tensorNames: selected, inputs: true, outputs: true,
            constants: false, values: false})})));
        const complete = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({
          debugSessionId: session.debugSessionId, expectedRevision: session.revision})));
        if (complete.state !== p.DebugState.DEBUG_STATE_COMPLETED || !complete.captureComplete)
          throw new Error(complete.failure?.message || 'activation observation did not complete');
        const plan = ok(await debug.getDebugPlan(new p.DebugSessionRef({debugSessionId: session.debugSessionId}))).plan;
        const namesById = new Map(plan.tensors.map(tensor => [tensor.tensorId, tensor.name]));
        const batch = new Map();
        let pageToken = '';
        do {
          const page = ok(await debug.listDebugEvents(new p.ListDebugEventsRequest({
            debugSessionId: session.debugSessionId, pageSize: 4096, pageToken})));
          for (const event of page.events) for (const snapshot of event.snapshots) {
            const name = namesById.get(snapshot.tensorId);
            if (!selected.includes(name) || batch.has(name)) continue;
            const stats = snapshot.statistics;
            if (!stats || !stats.finiteCount || stats.nanCount || stats.positiveInfinityCount || stats.negativeInfinityCount)
              throw new Error(`activation '${name}' has missing or non-finite observations`);
            batch.set(name, {min: stats.minimum, max: stats.maximum, elements: Number(stats.finiteCount)});
          }
          pageToken = page.nextPageToken;
        } while (pageToken);
        for (const name of selected) {
          const value = batch.get(name);
          if (!value) throw new Error(`activation '${name}' was not observed`);
          const previous = pending[name];
          pending[name] = {min: Math.min(previous?.min ?? Infinity, value.min),
            max: Math.max(previous?.max ?? -Infinity, value.max),
            samples: (previous?.samples ?? 0) + 1, elements: (previous?.elements ?? 0) + value.elements};
          if (!Number.isSafeInteger(pending[name].elements)) throw new Error('observation count exceeds safe integers');
        }
      } finally {
        if (session) ok(await debug.releaseDebugSession(new p.DebugSessionRef({debugSessionId: session.debugSessionId})));
      }
    }
    for (const [name, value] of Object.entries(pending)) {
      const previous = observations[name];
      observations[name] = {min: Math.min(previous?.min ?? Infinity, value.min),
        max: Math.max(previous?.max ?? -Infinity, value.max),
        samples: (previous?.samples ?? 0) + value.samples,
        elements: (previous?.elements ?? 0) + value.elements};
    }
    return selected.length;
  } finally {
    if (context) ok(await inference.releaseExecutionContext(new p.ExecutionContextRef({contextId: context.contextId})));
    ok(await inference.releaseCompiledModel(new p.CompiledModelRef({compiledModelId: compiled.compiledModelId})));
  }
}

/** Split `names` into promotion batches of at most `size`. */
export function observationBatches(names, size) {
  if (!Number.isInteger(size) || size <= 0) {
    throw new Error('observation batch size must be a positive integer');
  }
  const batches = [];
  for (let index = 0; index < names.length; index += size) {
    batches.push(names.slice(index, index + size));
  }
  return batches;
}
