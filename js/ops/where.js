export function _cpuWhere(node) {
    const cond = node.inputs.cond || node.inputs.condition;
    const a = node.inputs.x || node.inputs.a;
    const b = node.inputs.y || node.inputs.b;
    const outBuf = node.outputs.out.buffer;
    
    const condBuf = cond.buffer;
    const aBuf = a.buffer;
    const bBuf = b.buffer;
    
    const elements = outBuf.length;
    for (let i = 0; i < elements; i++) {
        outBuf[i] = condBuf[i] !== 0 ? aBuf[i] : bBuf[i];
    }
}
