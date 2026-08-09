import { createBoundExecutionGraph } from '../../ts/core/BoundExecutionGraph.js';
import {
  VOLVOX_AFFINE_QUANTIZATION_FORMAT,
  parseGraphDocument,
} from '../../ts/core/Graph.js';
import { Model } from '../../ts/core/Model.js';
import { importModelCheckpoint } from '../../ts/training/ModelCheckpoint.js';
import { ensureTrainingGraphState } from '../../ts/training/TrainingGraph.js';

const restoredCheckpoints = new WeakMap();

function cloneStorage(dtype, storage, elementCount) {
  const bytes = storage == null
    ? new Uint8Array(elementCount * (dtype === 'float32' || dtype === 'int32' ? 4 : 1))
    : storage instanceof ArrayBuffer
      ? new Uint8Array(storage)
      : new Uint8Array(storage.buffer, storage.byteOffset, storage.byteLength);
  const owned = bytes.slice().buffer;
  if (dtype === 'float32') return new Float32Array(owned);
  if (dtype === 'int32') return new Int32Array(owned);
  if (dtype === 'int8') return new Int8Array(owned);
  if (dtype === 'uint8') return new Uint8Array(owned);
  throw new Error(`Unsupported concrete fixture dtype '${dtype}'.`);
}

function jsonValue(value) {
  if (value == null || typeof value === 'boolean' || typeof value === 'number' ||
      typeof value === 'string') return value;
  if (ArrayBuffer.isView(value)) return Array.from(value, jsonValue);
  if (Array.isArray(value)) return value.map(jsonValue);
  if (typeof value === 'object') {
    return Object.fromEntries(Object.entries(value)
      .filter(([, entry]) => entry !== undefined && typeof entry !== 'function')
      .map(([name, entry]) => [name, jsonValue(entry)]));
  }
  throw new Error(`Concrete fixture parameter '${String(value)}' is not logical JSON.`);
}

function uniqueTensorName(occupied, requested) {
  if (!occupied.has(requested)) {
    occupied.add(requested);
    return requested;
  }
  for (let ordinal = 1; ; ordinal++) {
    const candidate = `${requested}.${ordinal}`;
    if (!occupied.has(candidate)) {
      occupied.add(candidate);
      return candidate;
    }
  }
}

/**
 * Test-only bridge for fixed concrete kernel fixtures. Public examples never
 * receive this adapter: production Trainer entry points accept snapshots only.
 */
export function logicalSnapshotFromTrainingGraph(concrete) {
  if (!concrete?.tensors || !Array.isArray(concrete.nodes)) {
    throw new Error('Concrete training fixture must be a Graph.');
  }
  const occupied = new Set(concrete.tensors.keys());
  const weightSources = [];
  for (const tensor of concrete.tensors.values()) {
    if (!tensor.isWeight) continue;
    const elementCount = tensor.shape.reduce((total, extent) => total * extent, 1);
    weightSources.push({
      name: tensor.name,
      dtype: tensor.dtype,
      shape: [...tensor.shape],
      data: cloneStorage(tensor.dtype, tensor.buffer, elementCount),
    });
  }

  const quantizationReferences = {};
  const quantizationByTensor = {};
  for (const tensor of concrete.tensors.values()) {
    if (!tensor.quantization) continue;
    const quantizationNode = concrete.nodes.find((node) =>
      (node.opType === 'DequantizeLinear' && node.inputs.input === tensor) ||
      (node.opType === 'QuantizeLinear' && Object.values(node.outputs).includes(tensor)));
    const referencedScale = quantizationNode?.inputs.scale;
    const referencedZeroPoint = quantizationNode?.inputs.zero_point;
    const scaleName = referencedScale?.name ??
      uniqueTensorName(occupied, `fixture.quant.${tensor.name}.scale`);
    const zeroPointName = referencedZeroPoint?.name ??
      uniqueTensorName(occupied, `fixture.quant.${tensor.name}.zero_point`);
    const scales = tensor.quantization.scheme === 'per_axis'
      ? Float32Array.from(tensor.quantization.scales)
      : Float32Array.of(tensor.quantization.scale);
    const zeroPoints = tensor.quantization.scheme === 'per_axis'
      ? tensor.quantization.zero_points
      : [tensor.quantization.zero_point];
    const ZeroPointArray = tensor.dtype === 'int8' ? Int8Array : Uint8Array;
    if (!referencedScale) {
      weightSources.push({
        name: scaleName,
        dtype: 'float32',
        shape: [scales.length],
        data: scales,
      });
    }
    if (!referencedZeroPoint) {
      weightSources.push({
        name: zeroPointName,
        dtype: tensor.dtype,
        shape: [zeroPoints.length],
        data: ZeroPointArray.from(zeroPoints),
      });
    }
    quantizationReferences[tensor.name] = tensor.quantization.scheme === 'per_axis'
      ? {
        scheme: 'per_axis',
        axis: tensor.quantization.axis,
        scale_tensor: scaleName,
        zero_point_tensor: zeroPointName,
      }
      : {
        scheme: 'per_tensor',
        scale_tensor: scaleName,
        zero_point_tensor: zeroPointName,
      };
    quantizationByTensor[tensor.name] = tensor.quantization;
  }

  const document = {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: Object.fromEntries([...concrete.tensors.values()]
      .filter((tensor) => tensor.isInput)
      .map((tensor) => [tensor.name, {
        dtype: tensor.dtype,
        shape: [...tensor.shape],
      }])),
    nodes: concrete.nodes.flatMap((node) => {
      const params = jsonValue(node.params ?? {});
      if (node.wLayout) {
        params.weight_layout = node.wLayout === 'dout' ? 'dout_din' : 'din_dout';
      }
      const declaredOutput = node.outputs.out ?? Object.values(node.outputs)[0];
      if ((node.opType === 'Reshape' || node.opType === 'Expand' || node.opType === 'Broadcast') &&
          params.shape === undefined && declaredOutput) {
        params.shape = [...declaredOutput.shape];
      }
      if ((node.opType === 'SDPA' || node.opType === 'CrossSDPA') && params.causal === undefined) {
        params.causal = node.opType === 'SDPA';
      }
      if (node.opType === 'Pad' && Array.isArray(params.pads) && declaredOutput) {
        const rank = node.inputs.input.shape.length;
        if (rank === 4 && params.pads.length === 4) {
          const [top, left, bottom, right] = params.pads;
          params.pads = [0, top, left, 0, 0, bottom, right, 0];
        }
      }
      if (node.opType === 'Split' && params.split === undefined &&
          params.num_outputs === undefined) {
        params.num_outputs = Object.keys(node.outputs).length;
      }
      let logicalInputs = Object.fromEntries(Object.entries(node.inputs)
        .map(([port, tensor]) => [port, tensor.name]));
      if (node.opType === 'Concat') {
        logicalInputs = Object.fromEntries(Object.values(node.inputs)
          .map((tensor, index) => [`input${index}`, tensor.name]));
      }
      if (node.opType === 'Where') {
        logicalInputs = {
          condition: node.inputs.condition?.name ?? node.inputs.cond?.name,
          a: node.inputs.a?.name ?? node.inputs.x?.name,
          b: node.inputs.b?.name ?? node.inputs.y?.name,
        };
      }
      const quantizedTensor = node.opType === 'QuantizeLinear'
        ? declaredOutput
        : node.opType === 'DequantizeLinear'
          ? node.inputs.input
          : null;
      if (quantizedTensor && quantizationReferences[quantizedTensor.name]) {
        const reference = quantizationReferences[quantizedTensor.name];
        logicalInputs.scale = reference.scale_tensor;
        logicalInputs.zero_point = reference.zero_point_tensor;
      }
      const broadcasts = [];
      if ((node.opType === 'Add' || node.opType === 'Mul') && declaredOutput) {
        for (const port of ['a', 'b']) {
          const tensor = node.inputs[port];
          if (!tensor || tensor.shape.length === declaredOutput.shape.length &&
              tensor.shape.every((extent, axis) => extent === declaredOutput.shape[axis])) continue;
          const tensorName = uniqueTensorName(occupied, `${node.id}.fixture_broadcast_${port}`);
          broadcasts.push({
            id: `${node.id}.fixture_broadcast_${port}`,
            opType: 'Broadcast',
            inputs: { input: tensor.name },
            outputs: {
              out: {
                tensor: tensorName,
                dtype: tensor.dtype,
                shape: [...declaredOutput.shape],
              },
            },
            params: { shape: [...declaredOutput.shape] },
          });
          logicalInputs[port] = tensorName;
        }
      }
      return [...broadcasts, {
        id: String(node.id),
        opType: node.opType,
        inputs: logicalInputs,
        outputs: Object.fromEntries(Object.entries(node.outputs)
          .map(([port, tensor]) => [port, {
            tensor: tensor.name,
            dtype: tensor.dtype,
            shape: [...tensor.shape],
        }])),
        params,
      }];
    }),
    outputs: [...concrete.outputNames],
    ...(Object.keys(quantizationReferences).length === 0 ? {} : {
      quantization: {
        format: VOLVOX_AFFINE_QUANTIZATION_FORMAT,
        tensors: quantizationReferences,
      },
    }),
  };
  const descriptors = weightSources.map(({ name, dtype, shape }) => ({ name, dtype, shape }));
  const graph = parseGraphDocument(document, descriptors);
  return Model.capture({
    graph,
    weights: Object.fromEntries(weightSources.map((weight) => [weight.name, weight])),
    quantizationByTensor,
  });
}

export function shapedFixtureInputs(snapshot, inputs = {}) {
  return Object.fromEntries(snapshot.inputNames.map((name) => {
    const value = inputs[name];
    if (value?.data && Array.isArray(value.shape)) return [name, value];
    const shape = snapshot.graph.inputs[name].shape;
    if (!shape.every(Number.isSafeInteger)) {
      throw new Error(`Concrete fixture input '${name}' unexpectedly has a symbolic shape.`);
    }
    return [name, { data: value, shape: [...shape] }];
  }));
}

export function trainingGraphFromCheckpoint(checkpoint) {
  const restored = importModelCheckpoint(checkpoint);
  if (!restored.snapshot.staticShapePlan) {
    throw new Error('Concrete fixture checkpoint must have one static shape plan.');
  }
  const bound = createBoundExecutionGraph(restored.snapshot, restored.snapshot.staticShapePlan);
  const graph = ensureTrainingGraphState(bound.graph);
  graph.trainingStep = restored.trainingStep;
  graph.trainingMetadata = restored.trainingMetadata;
  graph.optimizerDescriptor = restored.optimizerDescriptor;
  restoredCheckpoints.set(graph, checkpoint);
  return graph;
}

export function checkpointForTrainingGraph(graph) {
  return restoredCheckpoints.get(graph) ?? null;
}
