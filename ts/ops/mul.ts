import { cpuBroadcastBinary } from './broadcast.js';

export function _cpuMul(node) {
  return cpuBroadcastBinary(node, (left, right) => left * right, "Mul");
}
