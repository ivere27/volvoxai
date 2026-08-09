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
 * The promotion trick is an immutable logical snapshot whose document outputs
 * include the requested intermediates. Tensors are observed in batches because promoting all of them at
 * once keeps every intermediate live for the whole execution.
 *
 * Ranges accumulate across calls, so a sweep may be split over several batches
 * and several input sets without losing earlier observations.
 */

import {
  Model,
  parseGraphDocument,
} from '../../ts/index.js';

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

/**
 * Promote `names` to graph outputs, execute every input set, and merge each
 * observed range into `observations`. Returns how many tensors were observed.
 *
 * The loaded package remains immutable. Non-float32 and unknown tensors are
 * skipped rather than rejected: a candidate list may name tensors a particular
 * graph revision does not contain.
 */
export async function observeActivations({
  logicalPackage, document, runtime, backend, inputSets, names, observations,
  label = 'runtime output', onSample = null,
}) {
  if (!logicalPackage?.graph || !document || !runtime) {
    throw new Error('activation sweep requires a logical package, document, and runtime');
  }
  if (!observations || typeof observations !== 'object') {
    throw new Error('activation sweep requires an observations object');
  }
  const original = [...logicalPackage.graph.outputs];
  const selected = (names || []).filter((name) => {
    const tensor = logicalPackage.graph.tensors[name];
    return tensor && tensor.dtype === 'float32';
  });
  if (selected.length === 0) return 0;

  let compiled;
  let context;
  try {
    const promotedDocument = {
      ...document,
      outputs: [...new Set([...original, ...selected])],
    };
    const weights = Object.values(logicalPackage.weights).map((weight) => ({
      name: weight.name,
      dtype: weight.dtype,
      shape: [...weight.shape],
    }));
    const promotedGraph = parseGraphDocument(promotedDocument, weights);
    const snapshot = Model.capture({
      graph: promotedGraph,
      weights: logicalPackage.weights,
      quantizationByTensor: logicalPackage.quantizationByTensor,
    });
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    context = await compiled.createContext();
    for (const [index, inputs] of inputSets.entries()) {
      let result;
      try {
        result = await context.execute(inputs);
        const activations = {};
        for (const name of selected) {
          const output = result.output(name);
          const values = await output.read();
          observeFiniteRange(observations, name, values, label);
          activations[name] = Object.freeze({
            data: Float32Array.from(values),
            shape: Object.freeze([...output.shape]),
          });
        }
        onSample?.(index, Object.freeze({
          inputs,
          activations: Object.freeze(activations),
        }));
      } finally {
        await result?.close().catch(() => undefined);
      }
    }
    return selected.length;
  } finally {
    await context?.close().catch(() => undefined);
    await compiled?.close().catch(() => undefined);
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
