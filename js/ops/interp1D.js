export function _cpuInterp1D(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    const [c, in_l] = input.shape.slice(1);
    const out_l = node.params.size;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    // Half-pixel (align_corners=false) mapping, matching interp1d_f32 and the
    // WebGPU shader (PyTorch F.interpolate default).
    const scale = in_l / out_l;
    for (let ch = 0; ch < c; ch++) {
      for (let x = 0; x < out_l; x++) {
        let pos = (x + 0.5) * scale - 0.5;
        if (pos < 0) pos = 0;
        if (pos > in_l - 1) pos = in_l - 1;
        const x0 = Math.floor(pos);
        const x1 = x0 + 1 < in_l ? x0 + 1 : x0;
        const dx = pos - x0;
        const v0 = inBuf[ch * in_l + x0];
        const v1 = inBuf[ch * in_l + x1];
        outBuf[ch * out_l + x] = v0 + dx * (v1 - v0);
      }
    }
  }