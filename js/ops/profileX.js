export function _cpuProfileX(node) {

    // Column (W) profile: collapse H, keep W -> [1, 2c, w]. Mirrors C profile_x_f32.
    const input = node.inputs.input;
    const output = node.outputs.out;
    const [, h, w, c] = input.shape;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let ch = 0; ch < c; ch++) {
      for (let x = 0; x < w; x++) {
        let max_v = -Infinity;
        let sum_v = 0;
        for (let y = 0; y < h; y++) {
          const v = inBuf[(y * w + x) * c + ch];
          if (v > max_v) max_v = v;
          sum_v += v;
        }
        outBuf[ch * w + x] = max_v;
        outBuf[(c + ch) * w + x] = sum_v / h;
      }
    }
  }
