import {
  assertShapeKernelOutput,
  assertShapeKernelTensor,
  sameShape,
} from './shapeKernelValidation.js';

function product(values) {
  return values.reduce((a, b) => a * b, 1);
}

/**
 * Routed expert linear layer.
 *
 * Contract:
 *   input          [..., d_in]
 *   expert_weight  [num_experts, d_in, d_out]
 *   expert_bias    [num_experts, d_out] (optional)
 *   route_indices  [..., top_k] (F32 integer values)
 *   route_weights  [..., top_k]
 *   out            [..., d_out]
 *
 * When the expert weight is a partially resident bank, `node.residentSlots`
 * carries the ascending global slot ids the context materialized. Route
 * indices are always global, so they are mapped to the local row here and a
 * route to a non-resident expert is an error rather than a silent mismatch.
 */
export function _cpuMoELinear(node) {
  const input = node.inputs.input || node.inputs.x;
  const expertWeight = node.inputs.expert_weight || node.inputs.weight;
  const expertBiasTensor = node.inputs.expert_bias || node.inputs.bias || null;
  const expertBias = expertBiasTensor?.buffer || null;
  const routeIndices = node.inputs.route_indices || node.inputs.indices;
  const routeWeights = node.inputs.route_weights || node.inputs.weights;
  const output = node.outputs.out || Object.values(node.outputs)[0];
  assertShapeKernelTensor(input, 'MoELinear input', {
    dtypes: ['float32'], minimumRank: 1, maximumRank: 8,
  });
  assertShapeKernelTensor(expertWeight, 'MoELinear expert_weight', {
    dtypes: ['float32'], minimumRank: 3, maximumRank: 3,
  });
  assertShapeKernelTensor(routeIndices, 'MoELinear route_indices', {
    dtypes: ['float32'], minimumRank: input.shape.length,
    maximumRank: input.shape.length,
  });
  assertShapeKernelTensor(routeWeights, 'MoELinear route_weights', {
    dtypes: ['float32'], minimumRank: input.shape.length,
    maximumRank: input.shape.length,
  });
  if (expertBiasTensor) {
    assertShapeKernelTensor(expertBiasTensor, 'MoELinear expert_bias', {
      dtypes: ['float32'], minimumRank: 2, maximumRank: 2,
    });
  }

  const dIn = input.shape[input.shape.length - 1];
  const dOut = expertWeight.shape[2];
  const rows = product(input.shape.slice(0, -1));
  const numExperts = expertWeight.shape[0];
  const topK = routeIndices.shape[routeIndices.shape.length - 1];
  if (expertWeight.shape.length !== 3 || expertWeight.shape[1] !== dIn || expertWeight.shape[2] !== dOut) {
    throw new Error(`MoELinear expert_weight must be [experts, ${dIn}, ${dOut}].`);
  }
  const expectedRouteShape = [...input.shape.slice(0, -1), topK];
  if (!Number.isInteger(topK) || topK < 1 || topK > numExperts ||
      !sameShape(routeIndices.shape, expectedRouteShape) ||
      !sameShape(routeWeights.shape, expectedRouteShape) ||
      (expertBias && !sameShape(expertBiasTensor.shape, [numExperts, dOut]))) {
    throw new Error("MoELinear route tensor shape does not match the input rows.");
  }
  assertShapeKernelOutput(
    output, [...input.shape.slice(0, -1), dOut], 'float32', undefined, 'MoELinear',
  );

  // A resident slot table maps global expert ids to rows of the staged slice.
  const residentSlots: readonly number[] | null = node.residentSlots || null;
  if (residentSlots !== null && residentSlots.length !== numExperts) {
    throw new Error(
      `MoELinear residentSlots lists ${residentSlots.length} slots but the staged ` +
      `expert weight has ${numExperts}.`);
  }
  const slotToRow: Map<number, number> | null = residentSlots === null
    ? null
    : new Map(residentSlots.map((slot, row) => [slot, row]));

  // Route indices and weights are value-dependent. Validate the full routing
  // table before the first output write so a bad late route cannot publish a
  // partially updated tensor.
  for (let routeOffset = 0; routeOffset < rows * topK; routeOffset++) {
    const rawExpert = routeIndices.buffer[routeOffset];
    const gate = routeWeights.buffer[routeOffset];
    const valid = Number.isInteger(rawExpert) && rawExpert >= 0 &&
      Number.isFinite(gate);
    const resident = slotToRow === null
      ? rawExpert < numExperts
      : slotToRow.has(rawExpert);
    if (!valid || !resident) {
      const row = Math.floor(routeOffset / topK);
      const slot = routeOffset % topK;
      throw new Error(valid && slotToRow !== null
        ? `MoELinear routed row ${row}, slot ${slot} to expert ${rawExpert}, ` +
          `which is not resident in this context.`
        : `MoELinear received invalid route at row ${row}, slot ${slot}.`);
    }
  }

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < dOut; col++) {
      let sum = 0;
      for (let slot = 0; slot < topK; slot++) {
        const routeOffset = row * topK + slot;
        const rawExpert = routeIndices.buffer[routeOffset];
        const expert = slotToRow === null
          ? Math.trunc(rawExpert)
          : slotToRow.get(Math.trunc(rawExpert))!;
        const gate = routeWeights.buffer[routeOffset];
        let expertValue = expertBias ? expertBias[expert * dOut + col] : 0;
        const expertBase = expert * dIn * dOut;
        for (let d = 0; d < dIn; d++) {
          expertValue += input.buffer[row * dIn + d] * expertWeight.buffer[expertBase + d * dOut + col];
        }
        sum += gate * expertValue;
      }
      output.buffer[row * dOut + col] = sum;
    }
  }
}
