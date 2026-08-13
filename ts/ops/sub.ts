import { cpuBroadcastBinary } from './broadcast.js';
import { assertShapeKernelParams } from './shapeKernelValidation.js';

export function _cpuSub(node) {
  assertShapeKernelParams(node, [], 'Sub');
  return cpuBroadcastBinary(node, (left, right) => left - right, 'Sub', {
    dtypes: ['float32'], maximumRank: 8,
  });
}
