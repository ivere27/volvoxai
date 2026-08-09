import { cpuBroadcastBinary } from './broadcast.js';
import { assertShapeKernelParams } from './shapeKernelValidation.js';

export function _cpuDiv(node) {
  assertShapeKernelParams(node, [], 'Div');
  return cpuBroadcastBinary(node, (left, right) => left / right, 'Div', {
    dtypes: ['float32'], maximumRank: 8,
  });
}
