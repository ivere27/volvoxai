export function _cpuSoftmax(node) {
    const input = node.inputs.input || node.inputs.x;
    const outBuf = node.outputs.out.buffer;
    const shape = input.shape;
    let axis = node.params.axis ?? -1;
    if (axis < 0) axis += shape.length;
    if (shape.length === 0 || axis !== shape.length - 1) {
        throw new Error(`CPU Softmax node ${node.id} requires the last axis.`);
    }
    const d = shape[shape.length - 1];
    const b = input.buffer.length / d;
    if (!Number.isInteger(b) || b <= 0 || !Number.isInteger(d) || d <= 0) {
        throw new Error(`CPU Softmax node ${node.id} has invalid reduction dimensions.`);
    }
    for (let i = 0; i < b; i++) {
        let max = -Infinity;
        for (let j = 0; j < d; j++) max = Math.max(max, input.buffer[i * d + j]);
        let sum = 0.0;
        for (let j = 0; j < d; j++) {
            const val = Math.exp(input.buffer[i * d + j] - max);
            outBuf[i * d + j] = val;
            sum += val;
        }
        for (let j = 0; j < d; j++) outBuf[i * d + j] /= sum;
    }
}
