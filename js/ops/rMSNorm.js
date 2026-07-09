export function _cpuRMSNorm(node) {
    const input = node.inputs.input || node.inputs.x;
    const weight = node.inputs.weight;
    const outBuf = node.outputs.out.buffer;
    const in_shape = input.shape.length === 2 ? input.shape : [1, input.buffer.length];
    const b = in_shape[0], d = in_shape[1];
    const eps = node.params.eps || 1e-5;
    for (let i = 0; i < b; i++) {
        let sq_sum = 0.0;
        for (let j = 0; j < d; j++) sq_sum += input.buffer[i * d + j] * input.buffer[i * d + j];
        const rms = Math.sqrt(sq_sum / d + eps);
        for (let j = 0; j < d; j++) outBuf[i * d + j] = (input.buffer[i * d + j] / rms) * weight.buffer[j];
    }
}