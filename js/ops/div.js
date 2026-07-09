export function _cpuDiv(node) {

    const a = node.inputs.a;
    const b = node.inputs.b;
    const out = node.outputs.out;
    const aBuf = a.buffer;
    const bBuf = b.buffer;
    const outBuf = out.buffer;
    if (bBuf.length === 1) {
        for (let i = 0; i < aBuf.length; i++) outBuf[i] = aBuf[i] / bBuf[0];
    } else {
        // Broadcast fallback assuming b is same size or inner dimension broadcasting
        for (let i = 0; i < aBuf.length; i++) outBuf[i] = aBuf[i] / bBuf[i % bBuf.length];
    }
  }