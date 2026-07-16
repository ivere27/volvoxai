import { cpuBroadcastBinary } from './broadcast.js';

export function _cpuDiv(node) {
  return cpuBroadcastBinary(node, (left, right) => left / right, 'Div');
}
