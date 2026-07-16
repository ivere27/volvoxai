import type { TensorLike } from '../types.js';

type ValidationTensor = TensorLike;

interface ValidationNode {
  id: string | number;
  opType: string;
  inputs: Record<string, ValidationTensor>;
  outputs: Record<string, ValidationTensor>;
  params: Record<string, any>;
}

export interface PortableQuantizedGraph {
  nodes: ValidationNode[];
}

export class PortableQuantizedGraphValidator {
  /* The core engines share a canonical W8A8 contract. Reject non-portable
   * QTensor representations early, but admit physical-byte graphs whose
   * scale/zero-point mapping is carried by immutable tensor metadata.
   *
   * This is deliberately a graph-level gate, rather than a collection of
   * best-effort checks in F32 operators.  A byte tensor with a descriptor has
   * a real-value interpretation; sending it to a generic F32 kernel would
   * either reinterpret bytes incorrectly or make backend support depend on
   * incidental implementation details.  Keep the portable W8A8 surface small
   * and fail closed until a new typed operator exists on every core backend. */
  static assert(graph: PortableQuantizedGraph) {
    const byteTensor = (tensor) => tensor && ['int8', 'uint8'].includes(tensor.dtype);
    const quantizedByteTensor = (tensor) => byteTensor(tensor) && tensor.quantization != null;
    const perTensor = (tensor) => byteTensor(tensor) && tensor.quantization?.scheme === 'per_tensor';
    const outputOf = (node: ValidationNode): ValidationTensor | undefined =>
      node.outputs?.out || Object.values(node.outputs || {})[0];
    const sameShape = (left, right) => Array.isArray(left?.shape) && Array.isArray(right?.shape) &&
      left.shape.length === right.shape.length &&
      left.shape.every((dimension, index) => dimension === right.shape[index]);
    const sameQuantization = (left, right) => {
      if (!left || !right || left.scheme !== right.scheme) return false;
      if (left.scheme === 'per_tensor') {
        return left.scale === right.scale && left.zero_point === right.zero_point;
      }
      return left.scheme === 'per_axis' && left.axis === right.axis &&
        left.scales.length === right.scales.length && left.zero_points.length === right.zero_points.length &&
        left.scales.every((value, index) => value === right.scales[index]) &&
        left.zero_points.every((value, index) => value === right.zero_points[index]);
    };
    const sameByteDomain = (left, right) => byteTensor(left) && byteTensor(right) &&
      left.dtype === right.dtype && sameQuantization(left.quantization, right.quantization);
    const label = (node) => `${node.opType} node ${node.id}`;
    const reject = (node: ValidationNode, message: string): never => {
      throw new Error(`[VolvoxAI] ${label(node)} ${message}`);
    };
    const nodeInput = (node: ValidationNode, names: string[], inputLabel: string): ValidationTensor => {
      const matches = names.filter((name) => node.inputs?.[name]);
      if (matches.length !== 1) reject(node, `requires exactly one '${inputLabel}' input (${names.join('/')}).`);
      return node.inputs[matches[0]];
    };
    const requireOnlyInputs = (node, names) => {
      for (const name of Object.keys(node.inputs || {})) {
        if (!names.includes(name)) reject(node, `has unsupported canonical W8A8 input '${name}'.`);
      }
    };
    const requireSingleOutput = (node: ValidationNode): ValidationTensor => {
      const outputs = Object.values(node.outputs || {}).filter(Boolean);
      if (outputs.length !== 1) reject(node, 'requires exactly one canonical W8A8 output.');
      return outputs[0]!;
    };
    const integerPair = (node, name, fallback, minimum, required = false) => {
      const source = node.params?.[name];
      if (source == null) {
        if (required) reject(node, `requires '${name}' for canonical W8A8 execution.`);
        return [fallback, fallback];
      }
      const values = Array.isArray(source) ? source : [source];
      if (values.length < 1 || values.length > 2 || values.some((value) =>
        !Number.isInteger(value) || value < minimum)) {
        reject(node, `requires '${name}' to be one or two integers >= ${minimum}.`);
      }
      return [values[0], values[1] ?? values[0]];
    };
    const pads = (node, padding) => {
      const source = node.params?.pads;
      if (source == null) return [padding[0], padding[1], padding[0], padding[1]];
      if (!Array.isArray(source) || source.length !== 4 || source.some((value) =>
        !Number.isInteger(value) || value < 0)) {
        reject(node, 'requires four non-negative top/left/bottom/right pads.');
      }
      return source;
    };
    const falseOrAbsent = (value) => value == null || value === false || value === 0;
    const requireNearestMapping = (node) => {
      const mode = node.params?.mode;
      if (node.opType === 'Resize') {
        if (mode !== 'nearest') reject(node, 'supports raw I8/U8 Resize only with mode "nearest".');
      } else if (mode != null && mode !== 'nearest') {
        reject(node, 'supports raw I8/U8 ResizeNearest2D only with mode "nearest".');
      }
      for (const key of ['coordinate_transformation_mode', 'coordinate_transform_mode']) {
        const transform = node.params?.[key];
        if (transform != null && transform !== 'asymmetric') {
          reject(node, `supports raw I8/U8 nearest resize only with ${key} "asymmetric".`);
        }
      }
      const nearestMode = node.params?.nearest_mode;
      if (nearestMode != null && nearestMode !== 'floor') {
        reject(node, 'supports raw I8/U8 nearest resize only with nearest_mode "floor".');
      }
      if (!falseOrAbsent(node.params?.align_corners) || !falseOrAbsent(node.params?.antialias)) {
        reject(node, 'does not support align_corners or antialias for raw I8/U8 nearest resize.');
      }
    };
    const requirePerAxisWeight = (node, weight, outputChannels, operation) => {
      const descriptor = weight?.quantization;
      if (!byteTensor(weight) || descriptor?.scheme !== 'per_axis' || descriptor.axis !== 0 ||
          descriptor.scales.length !== outputChannels || descriptor.zero_points.length !== outputChannels) {
        reject(node, `${operation} requires I8/U8 weights with axis-0 per-output-channel quantization metadata.`);
      }
    };
    const canonicalQConv = (node) => {
      const input = nodeInput(node, ['input', 'x'], 'activation');
      const weight = node.inputs.weight;
      const bias = node.inputs.bias || null;
      const output = requireSingleOutput(node);
      const embeddedQuantizationParameters = ['input_scale', 'input_zero_point', 'output_scale', 'output_zero_point'];
      const hasOnlyCanonicalInputs = Object.keys(node.inputs || {}).every((name) =>
        ['input', 'x', 'weight', 'bias'].includes(name));
      if (!hasOnlyCanonicalInputs || !perTensor(input) || !perTensor(output) || !byteTensor(weight) ||
          weight.quantization?.scheme !== 'per_axis' || weight.quantization.axis !== 0 ||
          weight.quantization.scales.length !== weight.shape[0] ||
          weight.quantization.zero_points.length !== weight.shape[0] ||
          (bias && (bias.dtype !== 'int32' || bias.shape.length !== 1 || bias.shape[0] !== weight.shape[0])) ||
          node.inputs.weight_scale || node.inputs.weight_zero_point ||
          embeddedQuantizationParameters.some((name) => node.params?.[name] != null) ||
          (node.params?.data_layout && node.params.data_layout !== 'NHWC') ||
          (node.params?.weight_layout && node.params.weight_layout !== 'OHWI')) {
        reject(node, 'must use canonical physical I8/U8 NHWC/OHWI storage, per-tensor activation metadata, axis-0 per-channel weight metadata, and optional I32 bias.');
      }
      if (input.shape.length !== 4 || weight.shape.length !== 4 || output.shape.length !== 4 ||
          input.shape[0] !== output.shape[0] || weight.shape[0] !== output.shape[3] ||
          (bias && (bias.shape.length !== 1 || bias.shape[0] !== weight.shape[0]))) {
        reject(node, 'has incompatible canonical NHWC/OHWI shapes.');
      }
      requirePerAxisWeight(node, weight, output.shape[3], 'QConv2D');
    };
    const canonicalQLinear = (node) => {
      requireOnlyInputs(node, ['input', 'x', 'a', 'weight', 'bias']);
      const input = nodeInput(node, ['input', 'x', 'a'], 'activation');
      const weight = node.inputs.weight;
      const bias = node.inputs.bias;
      const output = requireSingleOutput(node);
      if (!perTensor(input) || !perTensor(output) || !byteTensor(weight) || !bias ||
          bias.dtype !== 'int32' || input.shape.length < 1 || output.shape.length !== input.shape.length ||
          input.shape.slice(0, -1).some((dimension, index) => dimension !== output.shape[index]) ||
          weight.shape.length !== 2 || bias.shape.length !== 1 ||
          weight.shape[0] !== output.shape.at(-1) || weight.shape[1] !== input.shape.at(-1) ||
          bias.shape[0] !== output.shape.at(-1)) {
        reject(node, 'must use canonical typed [...,d_in] activations, [d_out,d_in] byte weights, I32 bias, and [...,d_out] output.');
      }
      requirePerAxisWeight(node, weight, output.shape.at(-1), node.opType);
    };
    const canonicalQEmbedding = (node) => {
      requireOnlyInputs(node, ['input', 'weight']);
      const input = nodeInput(node, ['input'], 'token IDs');
      const weight = node.inputs.weight;
      const output = requireSingleOutput(node);
      if (!input?.isInput || input.dtype !== 'int32' || input.shape.length < 1 || !byteTensor(weight) ||
          !perTensor(output) || weight.shape.length !== 2 ||
          output.shape.length !== input.shape.length + 1 ||
          output.shape.slice(0, -1).some((dimension, index) => dimension !== input.shape[index]) ||
          output.shape.at(-1) !== weight.shape[1]) {
        reject(node, 'must use a graph-input I32 token tensor, [vocab,hidden] I8/U8 weight with axis-0 row metadata, and a per-tensor I8/U8 [...token,hidden] output.');
      }
      requirePerAxisWeight(node, weight, weight.shape[0], 'QEmbedding');
    };
    const canonicalQAdd = (node) => {
      requireOnlyInputs(node, ['a', 'input', 'x', 'b', 'y']);
      const a = nodeInput(node, ['a', 'input', 'x'], 'left activation');
      const b = nodeInput(node, ['b', 'y'], 'right activation');
      const output = requireSingleOutput(node);
      const relu = node.params?.relu ?? 0;
      if (!perTensor(a) || !perTensor(b) || !perTensor(output) || !sameShape(a, b) || !sameShape(a, output) ||
          !Number.isInteger(relu) || relu < 0 || relu > 2) {
        reject(node, 'must use exact-shape canonical per-tensor I8/U8 edges and relu 0, 1, or 2.');
      }
    };
    const canonicalQSiLU = (node) => {
      requireOnlyInputs(node, ['input', 'x', 'data']);
      const input = nodeInput(node, ['input', 'x', 'data'], 'activation');
      const output = requireSingleOutput(node);
      if (!perTensor(input) || !perTensor(output) || !sameShape(input, output) || input === output ||
          Object.keys(node.params || {}).length !== 0) {
        reject(node, 'must use one same-shape canonical per-tensor I8/U8 activation input/output pair with distinct tensors and no parameters.');
      }
    };
    const canonicalQGELU = (node) => {
      requireOnlyInputs(node, ['input', 'x', 'data']);
      const input = nodeInput(node, ['input', 'x', 'data'], 'activation');
      const output = requireSingleOutput(node);
      const parameterNames = Object.keys(node.params || {});
      const canonicalParameters = parameterNames.length === 0 ||
        (parameterNames.length === 1 && parameterNames[0] === 'approximate' &&
          node.params.approximate === 'none');
      if (!perTensor(input) || !perTensor(output) || !sameShape(input, output) || input === output ||
          !canonicalParameters) {
        reject(node, "must use one same-shape canonical per-tensor I8/U8 activation input/output pair with distinct tensors and only omitted parameters or approximate='none'.");
      }
    };
    const canonicalQGroupNorm = (node) => {
      const inputNames = Object.keys(node.inputs || {}).sort();
      const input = node.inputs?.input;
      const weight = node.inputs?.weight;
      const bias = node.inputs?.bias;
      const output = requireSingleOutput(node);
      const params = node.params;
      const paramsAreObject = params == null || (typeof params === 'object' && !Array.isArray(params));
      const parameterNames = paramsAreObject ? Object.keys(params || {}) : [];
      const allowedParameters = paramsAreObject && parameterNames.every((name) =>
        ['num_groups', 'eps', 'data_layout'].includes(name));
      const numGroups = params?.num_groups;
      const epsilonSource = params?.eps ?? 1e-5;
      const epsilon = Math.fround(epsilonSource);
      const channels = input?.shape?.[3];
      const validNHWC = input?.shape?.length === 4 && output?.shape?.length === 4 &&
        input.shape.every((dimension) => Number.isInteger(dimension) && dimension > 0) &&
        sameShape(input, output);
      const validAffine = (tensor) => tensor?.dtype === 'float32' &&
        Array.isArray(tensor.shape) && tensor.shape.length === 1 && tensor.shape[0] === channels &&
        tensor.sizeBytes === channels * 4;
      if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
          inputNames[2] !== 'weight' || !perTensor(input) || !perTensor(output) || input === output ||
          !validNHWC || !validAffine(weight) || !validAffine(bias) || !allowedParameters ||
          !Number.isInteger(numGroups) || numGroups <= 0 || channels % numGroups !== 0 ||
          (params?.data_layout != null && params.data_layout !== 'NHWC') ||
          typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
          !Number.isFinite(epsilon) || epsilon <= 0) {
        reject(node, 'must use exact input/weight/bias inputs, rank-4 NHWC per-tensor I8/U8 activation edges, F32 [C] affine tensors, and positive num_groups/eps parameters.');
      }
    };
    const canonicalQLayerNorm = (node) => {
      const inputNames = Object.keys(node.inputs || {}).sort();
      const input = node.inputs?.input;
      const weight = node.inputs?.weight;
      const bias = node.inputs?.bias;
      const output = requireSingleOutput(node);
      const params = node.params;
      const paramsAreObject = params == null || (typeof params === 'object' && !Array.isArray(params));
      const parameterNames = paramsAreObject ? Object.keys(params || {}) : [];
      const allowedParameters = paramsAreObject && parameterNames.every((name) =>
        ['eps', 'd_model'].includes(name));
      const epsilonSource = params?.eps ?? 1e-5;
      const epsilon = Math.fround(epsilonSource);
      const dModel = input?.shape?.at(-1);
      const validRows = input?.shape?.length >= 1 && output?.shape?.length >= 1 &&
        input.shape.every((dimension) => Number.isInteger(dimension) && dimension > 0) &&
        sameShape(input, output);
      const validAffine = (tensor) => tensor?.dtype === 'float32' &&
        Array.isArray(tensor.shape) && tensor.shape.length === 1 && tensor.shape[0] === dModel &&
        tensor.sizeBytes === dModel * 4;
      if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
          inputNames[2] !== 'weight' || !perTensor(input) || !perTensor(output) || input === output ||
          output === weight || output === bias || !validRows || !validAffine(weight) || !validAffine(bias) ||
          !allowedParameters || (params?.d_model != null &&
            (!Number.isInteger(params.d_model) || params.d_model !== dModel)) ||
          typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
          !Number.isFinite(epsilon) || epsilon <= 0) {
        reject(node, 'must use exact input/weight/bias inputs, same-shape rank-at-least-1 per-tensor I8/U8 activation edges, F32 [D] affine tensors, and positive eps with optional d_model matching D.');
      }
    };
    const canonicalQMaskedMean = (node) => {
      const inputNames = Object.keys(node.inputs || {}).sort();
      const input = node.inputs?.input;
      const mask = node.inputs?.mask;
      const output = node.outputs?.out;
      const params = node.params || {};
      const validShape = (tensor) => Array.isArray(tensor?.shape) && tensor.shape.every((dimension) =>
        Number.isInteger(dimension) && dimension > 0);
      const count = (shape) => validShape({ shape })
        ? shape.reduce((product, dimension) => product * dimension, 1) : NaN;
      const validByteDescriptor = (tensor) => {
        const descriptor = tensor?.quantization;
        const minimum = tensor?.dtype === 'int8' ? -128 : 0;
        const maximum = tensor?.dtype === 'int8' ? 127 : 255;
        const scale = Math.fround(descriptor?.scale);
        return ['int8', 'uint8'].includes(tensor?.dtype) && descriptor?.scheme === 'per_tensor' &&
          Object.isFrozen(descriptor) && typeof descriptor.scale === 'number' &&
          Number.isFinite(descriptor.scale) && descriptor.scale > 0 && Number.isFinite(scale) && scale > 0 &&
          Number.isInteger(descriptor.zero_point) && descriptor.zero_point >= minimum &&
          descriptor.zero_point <= maximum;
      };
      const inputElements = count(input?.shape);
      const maskElements = count(mask?.shape);
      const outputElements = count(output?.shape);
      const batch = input?.shape?.[0];
      const sequence = input?.shape?.[1];
      const width = input?.shape?.[2];
      const multiplier = Math.fround(Math.fround(input?.quantization?.scale) /
        Math.fround(output?.quantization?.scale));
      const minimum = input?.dtype === 'int8' ? -128 : 0;
      const maximum = input?.dtype === 'int8' ? 127 : 255;
      const maximumCenteredMagnitude = Math.max(
        Math.abs(minimum - input?.quantization?.zero_point),
        Math.abs(maximum - input?.quantization?.zero_point),
      );
      const maximumSum = sequence * maximumCenteredMagnitude;
      const paramsAreEmpty = params != null && typeof params === 'object' && !Array.isArray(params) &&
        Object.keys(params).length === 0;
      if (inputNames.length !== 2 || inputNames[0] !== 'input' || inputNames[1] !== 'mask' ||
          Object.keys(node.outputs || {}).length !== 1 || !output || !paramsAreEmpty ||
          !validByteDescriptor(input) || !validByteDescriptor(output) || input === output ||
          !validShape(input) || !validShape(mask) || !validShape(output) || input.shape.length !== 3 ||
          mask?.dtype !== 'int32' || mask?.quantization != null || mask.shape.length !== 2 ||
          output.shape.length !== 2 || mask.shape[0] !== batch || mask.shape[1] !== sequence ||
          output.shape[0] !== batch || output.shape[1] !== width ||
          !Number.isSafeInteger(inputElements) || !Number.isSafeInteger(maskElements) ||
          !Number.isSafeInteger(outputElements) || input.sizeBytes !== inputElements ||
          mask.sizeBytes !== maskElements * 4 || output.sizeBytes !== outputElements ||
          inputElements !== batch * sequence * width || maskElements !== batch * sequence ||
          outputElements !== batch * width || !Number.isSafeInteger(maximumSum) ||
          maximumSum > 0x7fffffff || !Number.isFinite(multiplier) || multiplier <= 0) {
        reject(node, 'must use exactly { input, mask } -> { out }: immutable per-tensor I8/U8 [B,S,D] and [B,D] edges, an unquantized I32 [B,S] keep mask, no parameters, and an I32-safe centered sum.');
      }
    };
    const canonicalQSDPA = (node) => {
      const inputNames = Object.keys(node.inputs || {}).sort();
      const q = node.inputs?.q;
      const k = node.inputs?.k;
      const v = node.inputs?.v;
      const hasMask = Object.prototype.hasOwnProperty.call(node.inputs || {}, 'mask');
      const mask = node.inputs?.mask;
      const output = requireSingleOutput(node);
      const params = node.params;
      const paramsAreObject = params != null && typeof params === 'object' && !Array.isArray(params);
      const parameterNames = paramsAreObject ? Object.keys(params) : [];
      const allowedParameters = paramsAreObject && parameterNames.every((name) =>
        ['heads', 'causal', 'scale'].includes(name));
      const rank = q?.shape?.length;
      const qShapeValid = Array.isArray(q?.shape) && q.shape.every((dimension) =>
        Number.isInteger(dimension) && dimension > 0);
      const kShapeValid = Array.isArray(k?.shape) && k.shape.every((dimension) =>
        Number.isInteger(dimension) && dimension > 0);
      const vShapeValid = Array.isArray(v?.shape) && v.shape.every((dimension) =>
        Number.isInteger(dimension) && dimension > 0);
      const outputShapeValid = Array.isArray(output?.shape) && output.shape.every((dimension) =>
        Number.isInteger(dimension) && dimension > 0);
      const geometryIsRanked = (rank === 2 || rank === 3) && k?.shape?.length === rank &&
        v?.shape?.length === rank && output?.shape?.length === rank;
      const batch = rank === 2 ? 1 : q?.shape?.[0];
      const queries = q?.shape?.[rank - 2];
      const keys = k?.shape?.[rank - 2];
      const dModel = q?.shape?.[rank - 1];
      const kBatch = rank === 2 ? 1 : k?.shape?.[0];
      const vBatch = rank === 2 ? 1 : v?.shape?.[0];
      const validQuantization = (tensor) => {
        if (!perTensor(tensor)) return false;
        const descriptor = tensor.quantization;
        const minimum = tensor.dtype === 'int8' ? -128 : 0;
        const maximum = tensor.dtype === 'int8' ? 127 : 255;
        const scale = Math.fround(descriptor?.scale);
        return typeof descriptor?.scale === 'number' && Number.isFinite(descriptor.scale) &&
          descriptor.scale > 0 && Number.isFinite(scale) && scale > 0 &&
          Number.isInteger(descriptor.zero_point) && descriptor.zero_point >= minimum &&
          descriptor.zero_point <= maximum;
      };
      const validMask = !hasMask || (mask?.dtype === 'int32' && Array.isArray(mask.shape) &&
        mask.shape.every((dimension) => Number.isInteger(dimension) && dimension > 0) &&
        mask.sizeBytes === mask.shape.reduce((product, dimension) => product * dimension, 1) * 4 &&
        ((mask.shape.length === 1 && mask.shape[0] === keys) ||
          (mask.shape.length === 2 && mask.shape[1] === keys &&
            (mask.shape[0] === batch || mask.shape[0] === queries)) ||
          (mask.shape.length === 3 && mask.shape[0] === batch && mask.shape[1] === queries &&
            mask.shape[2] === keys)));
      const heads = params?.heads;
      const headDim = dModel / heads;
      const scaleSource = params?.scale ?? 1 / Math.sqrt(headDim);
      const scale = Math.fround(scaleSource);
      const maximumCenteredMagnitude = (tensor) => {
        const minimum = tensor?.dtype === 'int8' ? -128 : 0;
        const maximum = tensor?.dtype === 'int8' ? 127 : 255;
        const zeroPoint = tensor?.quantization?.zero_point;
        return Math.max(Math.abs(minimum - zeroPoint), Math.abs(maximum - zeroPoint));
      };
      const qScale = Math.fround(q?.quantization?.scale);
      const kScale = Math.fround(k?.quantization?.scale);
      const scoreMultiplier = Math.fround(Math.fround(qScale * kScale) * scale);
      const maximumScore = Math.fround(Math.fround(headDim * maximumCenteredMagnitude(q || {}) *
        maximumCenteredMagnitude(k || {})) * scoreMultiplier);
      const exactInputs = (inputNames.length === 3 && inputNames[0] === 'k' && inputNames[1] === 'q' &&
        inputNames[2] === 'v') || (inputNames.length === 4 && inputNames[0] === 'k' &&
        inputNames[1] === 'mask' && inputNames[2] === 'q' && inputNames[3] === 'v');
      if (!exactInputs || !validQuantization(q) || !validQuantization(k) || !validQuantization(v) ||
          !validQuantization(output) || output === q || output === k || output === v || output === mask ||
          !qShapeValid || !kShapeValid || !vShapeValid || !outputShapeValid || !geometryIsRanked ||
          !sameShape(q, output) || kBatch !== batch || vBatch !== batch ||
          k?.shape?.[rank - 1] !== dModel || v?.shape?.[rank - 1] !== dModel ||
          v?.shape?.[rank - 2] !== keys || !validMask || !allowedParameters ||
          !Number.isInteger(heads) || heads <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
          dModel % 4 !== 0 || headDim % 4 !== 0 || headDim > 64 ||
          !Object.prototype.hasOwnProperty.call(params || {}, 'heads') ||
          !Object.prototype.hasOwnProperty.call(params || {}, 'causal') || typeof params?.causal !== 'boolean' ||
          typeof scaleSource !== 'number' || !Number.isFinite(scaleSource) || scaleSource <= 0 ||
          !Number.isFinite(scale) || scale <= 0 || !Number.isFinite(scoreMultiplier) ||
          scoreMultiplier <= 0 || !Number.isFinite(maximumScore)) {
        reject(node, 'must use exact q/k/v and optional I32 mask inputs, rank-2/3 per-tensor I8/U8 tensors, D/head dimensions divisible by 4 with head_dim <= 64, and explicit heads/causal plus finite score scaling.');
      }
    };
    const canonicalQArgMax = (node) => {
      const inputNames = Object.keys(node.inputs || {});
      const outputNames = Object.keys(node.outputs || {});
      const input = node.inputs?.input;
      const output = node.outputs?.out;
      const params = node.params;
      const rank = input?.shape?.length;
      const validShape = (tensor) => Array.isArray(tensor?.shape) && tensor.shape.every((dimension) =>
        Number.isInteger(dimension) && dimension > 0);
      const count = (shape) => validShape({ shape })
        ? shape.reduce((product, dimension) => product * dimension, 1) : NaN;
      const inputElements = count(input?.shape);
      const outputElements = count(output?.shape);
      const paramsAreExact = params != null && typeof params === 'object' && !Array.isArray(params) &&
        Object.keys(params).length === 1 && Object.prototype.hasOwnProperty.call(params, 'axis') &&
        Number.isInteger(params.axis);
      let axis = params?.axis;
      if (axis < 0) axis += rank;
      const axisSize = input?.shape?.[axis];
      const outer = Array.isArray(input?.shape) && Number.isInteger(axis)
        ? input.shape.slice(0, axis).reduce((product, dimension) => product * dimension, 1) : NaN;
      const inner = Array.isArray(input?.shape) && Number.isInteger(axis)
        ? input.shape.slice(axis + 1).reduce((product, dimension) => product * dimension, 1) : NaN;
      const expectedOutputShape = Number.isInteger(axis) && Array.isArray(input?.shape)
        ? [...input.shape.slice(0, axis), ...input.shape.slice(axis + 1)] : null;
      const descriptor = input?.quantization;
      const minimum = input?.dtype === 'int8' ? -128 : 0;
      const maximum = input?.dtype === 'int8' ? 127 : 255;
      const f32Scale = Math.fround(descriptor?.scale);
      const immutablePerTensor = ['int8', 'uint8'].includes(input?.dtype) &&
        descriptor?.scheme === 'per_tensor' && Object.isFrozen(descriptor) &&
        typeof descriptor.scale === 'number' && Number.isFinite(descriptor.scale) && descriptor.scale > 0 &&
        Number.isFinite(f32Scale) && f32Scale > 0 && Number.isInteger(descriptor.zero_point) &&
        descriptor.zero_point >= minimum && descriptor.zero_point <= maximum;
      if (inputNames.length !== 1 || inputNames[0] !== 'input' || outputNames.length !== 1 ||
          outputNames[0] !== 'out' || !input || !output || input === output || !immutablePerTensor ||
          !validShape(input) || !validShape(output) || rank < 2 || rank > 8 || !paramsAreExact ||
          !Number.isInteger(axis) || axis < 0 || axis >= rank || !Number.isInteger(axisSize) ||
          axisSize <= 0 || axisSize > 0x7fffffff || !Number.isSafeInteger(outer) || outer <= 0 ||
          !Number.isSafeInteger(inner) || inner <= 0 || outer > 0xffffffff || inner > 0xffffffff ||
          !Number.isSafeInteger(inputElements) || inputElements <= 0 || inputElements > 0xffffffff ||
          !Number.isSafeInteger(outputElements) || outputElements <= 0 || outputElements > 0xffffffff ||
          input.sizeBytes !== inputElements || output.dtype !== 'int32' || output.quantization != null ||
          output.sizeBytes !== outputElements * 4 || outputElements !== outer * inner ||
          inputElements !== outer * axisSize * inner || !sameShape(output, { shape: expectedOutputShape })) {
        reject(node, 'must use exactly { input } -> { out }, immutable per-tensor rank-2..8 I8/U8 input, exact { axis: integer } parameters, and an unquantized I32 output with the reduced axis removed.');
      }
    };
    const canonicalRequantize = (node) => {
      requireOnlyInputs(node, ['input', 'x', 'data']);
      const input = nodeInput(node, ['input', 'x', 'data'], 'activation');
      const output = requireSingleOutput(node);
      if (!perTensor(input) || !perTensor(output) || !sameShape(input, output)) {
        reject(node, 'requires equal-shape canonical per-tensor I8/U8 edges.');
      }
    };
    const canonicalQuantize = (node) => {
      requireOnlyInputs(node, ['input', 'x', 'data', 'scale', 'zero_point']);
      const input = nodeInput(node, ['input', 'x', 'data'], 'F32 activation');
      const scale = node.inputs.scale;
      const zeroPoint = node.inputs.zero_point || null;
      const output = requireSingleOutput(node);
      if (!input || input.dtype !== 'float32' || !scale || scale.dtype !== 'float32' || scale.sizeBytes !== 4 ||
          !perTensor(output) || !sameShape(input, output) ||
          (zeroPoint && (zeroPoint.dtype !== output.dtype || zeroPoint.sizeBytes !== 1))) {
        reject(node, 'requires F32 input, scalar F32 scale, optional scalar matching zero point, and per-tensor I8/U8 output.');
      }
    };
    const canonicalDequantize = (node) => {
      requireOnlyInputs(node, ['input', 'x', 'data', 'scale', 'zero_point']);
      const input = nodeInput(node, ['input', 'x', 'data'], 'activation');
      const scale = node.inputs.scale;
      const zeroPoint = node.inputs.zero_point || null;
      const output = requireSingleOutput(node);
      const supportedInput = input && ['float32', 'int32', 'int8', 'uint8'].includes(input.dtype);
      const supportedZeroPoint = !zeroPoint ||
        (['float32', 'int32'].includes(zeroPoint.dtype) && zeroPoint.sizeBytes === 4) ||
        (['int8', 'uint8'].includes(zeroPoint.dtype) && zeroPoint.sizeBytes === 1);
      if (!supportedInput || !scale || scale.dtype !== 'float32' || scale.sizeBytes !== 4 ||
          !output || output.dtype !== 'float32' || !sameShape(input, output) ||
          !supportedZeroPoint) {
        reject(node, 'requires equal-shape F32/I32/I8/U8 input and F32 output, scalar F32 scale, and an optional scalar F32/I32/I8/U8 zero point.');
      }
    };
    const canonicalShapeCopy = (node) => {
      requireOnlyInputs(node, ['input']);
      const input = nodeInput(node, ['input'], 'activation');
      const output = requireSingleOutput(node);
      if (!perTensor(input) || !perTensor(output) || !sameByteDomain(input, output) ||
          input.sizeBytes !== output.sizeBytes) {
        reject(node, 'requires equal-size I8/U8 storage with identical per-tensor quantization metadata.');
      }
    };
    const canonicalResize = (node) => {
      requireOnlyInputs(node, ['input']);
      const input = nodeInput(node, ['input'], 'activation');
      const output = requireSingleOutput(node);
      if (!perTensor(input) || !perTensor(output) || !sameByteDomain(input, output) ||
          input.shape.length !== 4 || output.shape.length !== 4 || input.shape[0] !== output.shape[0] ||
          input.shape[3] !== output.shape[3]) {
        reject(node, 'requires descriptor-preserving rank-4 NHWC I8/U8 input/output tensors.');
      }
      requireNearestMapping(node);
    };
    const canonicalMaxPool = (node) => {
      requireOnlyInputs(node, ['input']);
      const input = nodeInput(node, ['input'], 'activation');
      const output = requireSingleOutput(node);
      if (!perTensor(input) || !perTensor(output) || !sameByteDomain(input, output) ||
          input.shape.length !== 4 || output.shape.length !== 4 || input.shape[0] !== output.shape[0] ||
          input.shape[3] !== output.shape[3]) {
        reject(node, 'requires descriptor-preserving rank-4 NHWC I8/U8 input/output tensors.');
      }
      if (!falseOrAbsent(node.params?.ceil_mode)) {
        reject(node, 'does not support ceil_mode for raw I8/U8 storage.');
      }
      const kernel = integerPair(node, 'kernel', 1, 1, true);
      const stride = integerPair(node, 'stride', 1, 1);
      const dilation = integerPair(node, 'dilation', 1, 1);
      const padding = integerPair(node, 'padding', 0, 0);
      const fullPads = pads(node, padding);
      if (dilation[0] !== 1 || dilation[1] !== 1) {
        reject(node, 'supports raw I8/U8 MaxPool2D only with unit dilation.');
      }
      const expectedHeight = Math.floor((input.shape[1] + fullPads[0] + fullPads[2] - kernel[0]) / stride[0]) + 1;
      const expectedWidth = Math.floor((input.shape[2] + fullPads[1] + fullPads[3] - kernel[1]) / stride[1]) + 1;
      if (expectedHeight !== output.shape[1] || expectedWidth !== output.shape[2]) {
        reject(node, 'has an output shape incompatible with canonical I8/U8 MaxPool2D parameters.');
      }
    };
    const canonicalConcat = (node: ValidationNode) => {
      const output = requireSingleOutput(node);
      const inputs = Object.values(node.inputs || {}).filter(Boolean);
      let axis = node.params?.axis ?? 0;
      if (axis < 0) axis += output?.shape?.length ?? 0;
      if (!perTensor(output) || inputs.length === 0 || !Number.isInteger(axis) || axis < 0 || axis >= output.shape.length ||
          !falseOrAbsent(node.params?.sigmoid)) {
        reject(node, 'requires canonical descriptor-preserving I8/U8 inputs, a valid axis, and no fused sigmoid.');
      }
      let axisSum = 0;
      for (const input of inputs) {
        if (!perTensor(input) || !sameByteDomain(input, output) || input.shape.length !== output.shape.length ||
            input.shape.some((dimension, index) => index !== axis && dimension !== output.shape[index])) {
          reject(node, 'requires same-dtype inputs with identical per-tensor metadata and compatible shapes.');
        }
        axisSum += input.shape[axis];
      }
      if (axisSum !== output.shape[axis]) reject(node, 'has input axes that do not match its output axis.');
    };

    const typedHandlers: Partial<Record<string, (node: ValidationNode) => void>> = {
      QConv2D: canonicalQConv,
      QLinear: canonicalQLinear,
      QMatMul: canonicalQLinear,
      QGemm: canonicalQLinear,
      QEmbedding: canonicalQEmbedding,
      QAdd: canonicalQAdd,
      QGELU: canonicalQGELU,
      QGroupNorm: canonicalQGroupNorm,
      QLayerNorm: canonicalQLayerNorm,
      QMaskedMean: canonicalQMaskedMean,
      QSDPA: canonicalQSDPA,
      QArgMax: canonicalQArgMax,
      QSiLU: canonicalQSiLU,
      RequantizeLinear: canonicalRequantize,
      QuantizeLinear: canonicalQuantize,
      DequantizeLinear: canonicalDequantize,
      MaxPool2D: canonicalMaxPool,
      Resize: canonicalResize,
      ResizeNearest2D: canonicalResize,
      Reshape: canonicalShapeCopy,
      Flatten: canonicalShapeCopy,
      Squeeze: canonicalShapeCopy,
      Unsqueeze: canonicalShapeCopy,
      Identity: canonicalShapeCopy,
      Concat: canonicalConcat,
    };
    const metadataOnlyInputs = new Set([
      'weight', 'bias', 'scale', 'weight_scale', 'zero_point', 'weight_zero_point',
      'input_scale', 'input_zero_point', 'output_scale', 'output_zero_point',
    ]);
    const explicitCanonicalOperators = new Set([
      'QConv2D', 'QLinear', 'QMatMul', 'QGemm', 'QAdd', 'QGELU', 'QGroupNorm', 'QLayerNorm', 'QMaskedMean', 'QSDPA', 'QArgMax', 'QSiLU',
      'QEmbedding',
      'RequantizeLinear', 'QuantizeLinear', 'DequantizeLinear',
    ]);
    const genericArithmeticOperators = new Set(['Add', 'Mul', 'Sub', 'Div']);
    for (const node of graph.nodes) {
      // A descriptor on a constant weight is the established W8A32 storage
      // form for ordinary F32 MatMul/Conv nodes.  It is not a typed activation
      // edge and remains eligible for the explicit weight-only path.  By contrast, all
      // outputs and non-metadata inputs are live values: a descriptor there
      // means raw W8A8 data must stay inside the portable typed subset.
      const typedActivationTensors = [
        ...Object.entries(node.inputs || {})
          .filter(([name]) => !metadataOnlyInputs.has(name))
          .map(([, tensor]) => tensor),
        ...Object.values(node.outputs || {}),
      ].filter(quantizedByteTensor);
      const byteActivationTensors = [
        ...Object.entries(node.inputs || {})
          .filter(([name]) => !metadataOnlyInputs.has(name))
          .map(([, tensor]) => tensor),
        ...Object.values(node.outputs || {}),
      ].filter(byteTensor);
      const handler = typedHandlers[node.opType];
      if (genericArithmeticOperators.has(node.opType) && byteActivationTensors.length > 0) {
        reject(node, 'routes I8/U8 storage to an unsupported generic operator; use a typed quantized operator or an explicit F32 boundary.');
      }
      if (explicitCanonicalOperators.has(node.opType)) {
        if (!handler) reject(node, 'does not have a canonical typed validation handler.');
        handler!(node);
      } else if (typedActivationTensors.length > 0) {
        if (!handler) reject(node, 'routes canonical raw I8/U8 storage to an unsupported generic operator; insert an explicit DequantizeLinear boundary or implement a typed operator.');
        handler!(node);
      }
      if (['ReLU', 'Sigmoid', 'HardSwish', 'HardSigmoid', 'SiLU', 'Swish', 'Tanh'].includes(node.opType)) {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = outputOf(node);
        if (byteTensor(input) || byteTensor(output)) {
          throw new Error(`[VolvoxAI] ${node.opType} node ${node.id} requires an explicit F32 boundary; it cannot reinterpret raw quantized bytes.`);
        }
      }
      if (node.opType === 'Concat' && node.params?.sigmoid) {
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        if (byteTensor(output)) {
          throw new Error(`[VolvoxAI] Quantized Concat node ${node.id} cannot fuse sigmoid; insert an explicit F32 boundary.`);
        }
      }
    }
  }

}

export function validatePortableQuantizedGraph(graph: PortableQuantizedGraph) {
  return PortableQuantizedGraphValidator.assert(graph);
}
