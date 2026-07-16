export function _cpuDequantizeLinear(node) {
    const inBuf = node.inputs.input.buffer;
    const scale = node.inputs.scale.buffer[0];
    const zp = node.inputs.zero_point ? node.inputs.zero_point.buffer[0] : 0.0;
    const outBuf = node.outputs.out.buffer;
    for (let i = 0; i < outBuf.length; i++) {
        outBuf[i] = (inBuf[i] - zp) * scale;
    }
}