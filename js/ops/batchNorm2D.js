export function _cpuBatchNorm2D(node) {
    const input = node.inputs.input || node.inputs.x;
    const weight = node.inputs.weight || node.inputs.scale;
    const bias = node.inputs.bias || node.inputs.b;
    const running_mean = node.inputs.running_mean || node.inputs.mean;
    const running_var = node.inputs.running_var || node.inputs.var;
    const outBuf = node.outputs.out.buffer;
    
    const [b, h, w, c] = input.shape;
    const eps = node.params.eps || 1e-5;
    
    for (let batch = 0; batch < b; batch++) {
        for (let chan = 0; chan < c; chan++) {
            const w_val = weight.buffer[chan];
            const b_val = bias ? bias.buffer[chan] : 0.0;
            const rm_val = running_mean.buffer[chan];
            const rv_val = running_var.buffer[chan];
            
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const idx = ((batch * h + y) * w + x) * c + chan;
                    const val = input.buffer[idx];
                    outBuf[idx] = ((val - rm_val) / Math.sqrt(rv_val + eps)) * w_val + b_val;
                }
            }
        }
    }
}
