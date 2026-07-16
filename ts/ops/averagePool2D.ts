export function _cpuAveragePool2D(node) {
    const input = node.inputs.input || node.inputs.x;
    const inBuf = input.buffer;
    const outBuf = node.outputs.out.buffer;
    
    const [b, in_h, in_w, c] = input.shape;
    const [out_b, out_h, out_w, out_c] = node.outputs.out.shape;
    
    const kh = node.params.kernel[0], kw = node.params.kernel[1];
    const sh = node.params.stride ? node.params.stride[0] : 1;
    const sw = node.params.stride ? node.params.stride[1] : 1;
    const ph = node.params.padding ? node.params.padding[0] : 0;
    const pw = node.params.padding ? node.params.padding[1] : 0;

    for (let batch = 0; batch < b; batch++) {
        for (let y = 0; y < out_h; y++) {
            for (let x = 0; x < out_w; x++) {
                for (let chan = 0; chan < c; chan++) {
                    let sum = 0.0;
                    let count = 0;
                    for (let ky = 0; ky < kh; ky++) {
                        for (let kx = 0; kx < kw; kx++) {
                            const in_y = y * sh - ph + ky;
                            const in_x = x * sw - pw + kx;
                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                const inIdx = ((batch * in_h + in_y) * in_w + in_x) * c + chan;
                                sum += inBuf[inIdx];
                                count++;
                            }
                        }
                    }
                    const outIdx = ((batch * out_h + y) * out_w + x) * out_c + chan;
                    outBuf[outIdx] = count > 0 ? sum / count : 0.0;
                }
            }
        }
    }
}
