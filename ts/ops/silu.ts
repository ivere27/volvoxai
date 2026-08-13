export function _cpuSiLU(node) {
    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    for (let i = 0; i < outBuf.length; i++) {
        const x = inBuf[i];
        outBuf[i] = x / (1.0 + Math.exp(-x));
    }
}