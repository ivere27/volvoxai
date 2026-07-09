export function _cpuPReLU(node) {
    const inBuf = (node.inputs.input || node.inputs.x).buffer;
    const slope = (node.inputs.slope || node.inputs.weight).buffer;
    const outBuf = node.outputs.out.buffer;
    const shape = (node.inputs.input || node.inputs.x).shape || [];
    const channels = shape.length === 4 ? shape[3] : slope.length;
    for (let i = 0; i < outBuf.length; i++) {
        const alpha = slope.length === channels ? slope[i % channels] : slope[i % slope.length];
        outBuf[i] = inBuf[i] < 0.0 ? inBuf[i] * alpha : inBuf[i];
    }
}
