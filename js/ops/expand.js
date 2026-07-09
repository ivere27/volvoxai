export function _cpuExpand(node) {
    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    if (inBuf.length === 1) {
        for (let i = 0; i < outBuf.length; i++) outBuf[i] = inBuf[0];
    } else if (inBuf.length === outBuf.length) {
        outBuf.set(inBuf);
    } else {
        const pad4 = (sh) => [1, 1, 1, 1].slice(0, 4 - sh.length).concat(sh);
        const [ib, ih, iw, ic] = pad4(node.inputs.input.shape);
        const [ob, oh, ow, oc] = pad4(node.outputs.out.shape);
        for (let b = 0; b < ob; b++) {
            for (let y = 0; y < oh; y++) {
                for (let x = 0; x < ow; x++) {
                    for (let c = 0; c < oc; c++) {
                        const src = (((b % ib) * ih + (y % ih)) * iw + (x % iw)) * ic + (c % ic);
                        const dst = ((b * oh + y) * ow + x) * oc + c;
                        outBuf[dst] = inBuf[src];
                    }
                }
            }
        }
    }
}
