import { cpuBroadcastBinary } from './broadcast.js';

export function _cpuSub(node) {
  return cpuBroadcastBinary(node, (left, right) => left - right, 'Sub');
}
