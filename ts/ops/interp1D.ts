export function _cpuInterp1D(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    const [batch, c, in_l] = input.shape;
    const out_l = node.params.size;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    // Half-pixel (align_corners=false) mapping, matching interp1d_f32 and the
    // WebGPU shader (PyTorch F.interpolate default).
    const scale = in_l / out_l;
    if (output.shape.length !== 3 || output.shape[0] !== batch || output.shape[1] !== c || output.shape[2] !== out_l) {
      throw new Error('Interp1D requires matching rank-3 NCL input/output tensors.');
    }
    for (let b = 0; b < batch; b++) for (let ch = 0; ch < c; ch++) {
      for (let x = 0; x < out_l; x++) {
        let pos = (x + 0.5) * scale - 0.5;
        if (pos < 0) pos = 0;
        if (pos > in_l - 1) pos = in_l - 1;
        const x0 = Math.floor(pos);
        const x1 = x0 + 1 < in_l ? x0 + 1 : x0;
        const dx = pos - x0;
        const inputBase = (b * c + ch) * in_l;
        const v0 = inBuf[inputBase + x0];
        const v1 = inBuf[inputBase + x1];
        outBuf[(b * c + ch) * out_l + x] = v0 + dx * (v1 - v0);
      }
    }
  }
