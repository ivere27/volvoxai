export function _cpuReduceMean(node) {
    const input = node.inputs.input || node.inputs.data;
    const inBuf = input.buffer;
    const outBuf = node.outputs.out.buffer;
    
    const d = input.shape.at(-1);
    const b = input.shape.slice(0, -1).reduce((count, dimension) => count * dimension, 1);
    if (!Number.isSafeInteger(d) || d <= 0 || inBuf.length !== b * d || outBuf.length !== b) {
      throw new Error(`ReduceMean node ${node.id ?? "<unnamed>"} must reduce the last axis into one value per outer row.`);
    }
    
    for (let i = 0; i < b; i++) {
        let sum = 0.0;
        for (let j = 0; j < d; j++) {
            sum += inBuf[i * d + j];
        }
        outBuf[i] = sum / d;
    }
}
