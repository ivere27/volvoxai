export function _cpuConvTranspose2D(node) {
    const input = node.inputs.input || node.inputs.x;
    const weight = node.inputs.weight;
    const bias = node.inputs.bias;
    
    const inBuf = input.buffer;
    const wBuf = weight.buffer;
    const bBuf = bias ? bias.buffer : null;
    const outBuf = node.outputs.out.buffer;
    
    const [b, in_h, in_w, in_c] = input.shape;
    const [out_b, out_h, out_w, out_c] = node.outputs.out.shape;
    const kh = node.params.kernel[0], kw = node.params.kernel[1];
    const sh = node.params.stride ? node.params.stride[0] : 1;
    const sw = node.params.stride ? node.params.stride[1] : 1;
    const ph = node.params.padding ? node.params.padding[0] : 0;
    const pw = node.params.padding ? node.params.padding[1] : 0;
    
    // Zero initialize output
    for (let i = 0; i < outBuf.length; i++) {
        outBuf[i] = bBuf ? bBuf[i % out_c] : 0.0;
    }
    
    for (let i_b = 0; i_b < b; i_b++) {
        for (let i_ic = 0; i_ic < in_c; i_ic++) {
            for (let iy = 0; iy < in_h; iy++) {
                for (let ix = 0; ix < in_w; ix++) {
                    const in_val = inBuf[((i_b * in_h + iy) * in_w + ix) * in_c + i_ic];
                    for (let oc = 0; oc < out_c; oc++) {
                        for (let ky = 0; ky < kh; ky++) {
                            for (let kx = 0; kx < kw; kx++) {
                                const oy = iy * sh - ph + ky;
                                const ox = ix * sw - pw + kx;
                                if (oy >= 0 && oy < out_h && ox >= 0 && ox < out_w) {
                                    const w_val = wBuf[i_ic * (out_c * kh * kw) + oc * (kh * kw) + ky * kw + kx];
                                    outBuf[((i_b * out_h + oy) * out_w + ox) * out_c + oc] += in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
