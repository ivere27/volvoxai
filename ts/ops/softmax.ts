export function _cpuSoftmax(node) {
    const input = node.inputs.input || node.inputs.x;
    const outBuf = node.outputs.out.buffer;
    const in_shape = input.shape.length === 2 ? input.shape : [1, input.buffer.length];
    const b = in_shape[0];
    const d = in_shape[1];
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