export function _cpuSub(node) {
    const aBuf = node.inputs.a.buffer;
    const bBuf = node.inputs.b.buffer;
    const outBuf = node.outputs.out.buffer;
    const elements = outBuf.length;
    if (bBuf.length === 1) {
        for (let i = 0; i < elements; i++) outBuf[i] = aBuf[i] - bBuf[0];
    } else {
        for (let i = 0; i < elements; i++) outBuf[i] = aBuf[i] - bBuf[i];
    }
}