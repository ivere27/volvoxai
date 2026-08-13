export function _cpuRMSNorm(node) {
    const input = node.inputs.input || node.inputs.x;
    const weight = node.inputs.weight;
    const outBuf = node.outputs.out.buffer;
    const d = node.params.d_model ?? input.shape.at(-1);
    if (!Number.isInteger(d) || d <= 0 || input.buffer.length % d !== 0) {
        throw new Error("RMSNorm has an invalid d_model.");
    }
    const b = input.buffer.length / d;
    const eps = node.params.eps ?? 1e-6;
    for (let i = 0; i < b; i++) {
        let sq_sum = 0.0;
        for (let j = 0; j < d; j++) sq_sum += input.buffer[i * d + j] * input.buffer[i * d + j];
        const rms = Math.sqrt(sq_sum / d + eps);
        for (let j = 0; j < d; j++) outBuf[i * d + j] = (input.buffer[i * d + j] / rms) * weight.buffer[j];
    }
}
