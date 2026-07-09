export function _cpuPad(node) {
    const input = node.inputs.input || node.inputs.data;
    const outBuf = node.outputs.out.buffer;
    const pads = node.params.pads || [];
    const val = node.params.value || 0.0;

    let pt = 0, pb = 0, pl = 0, pr = 0;
    if (pads.length === 8) {
        pt = pads[1];
        pl = pads[2];
        pb = pads[5];
        pr = pads[6];
    } else if (pads.length === 4) {
        pt = pads[0]; pl = pads[1]; pb = pads[2]; pr = pads[3];
    }
    
    const inShape = input.shape.length === 4 ? input.shape : [1, input.shape[0] || 1, input.shape[1] || 1, 1];
    const [b, in_h, in_w, c] = inShape;
    const out_h = in_h + pt + pb;
    const out_w = in_w + pl + pr;
    
    for (let i = 0; i < outBuf.length; i++) outBuf[i] = val;

    for (let batch = 0; batch < b; batch++) {
        for (let y = 0; y < in_h; y++) {
            for (let x = 0; x < in_w; x++) {
                for (let chan = 0; chan < c; chan++) {
                    const inIdx = ((batch * in_h + y) * in_w + x) * c + chan;
                    const outIdx = ((batch * out_h + (y + pt)) * out_w + (x + pl)) * c + chan;
                    outBuf[outIdx] = input.buffer[inIdx];
                }
            }
        }
    }
}
