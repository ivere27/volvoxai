export function _cpuTanh(node) {
    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    for (let i = 0; i < outBuf.length; i++) {
        outBuf[i] = Math.tanh(inBuf[i]);
    }
}