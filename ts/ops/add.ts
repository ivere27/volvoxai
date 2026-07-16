import { cpuBroadcastBinary } from './broadcast.js';

export function _cpuAdd(node) {
  const relu = node.params?.relu ?? 0;
  if (!Number.isInteger(relu) || relu < 0 || relu > 2) {
    throw new Error('Add supports relu values 0 (none), 1 (ReLU), or 2 (ReLU6).');
  }
  return cpuBroadcastBinary(node, (left, right) => {
    const sum = left + right;
    if (relu === 1) return sum < 0 ? 0 : sum;
    if (relu === 2) return sum < 0 ? 0 : sum > 6 ? 6 : sum;
    return sum;
  }, "Add");
}
