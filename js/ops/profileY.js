export function _cpuProfileY(node) {

    // Row (H) profile: collapse W, keep H -> [1, 2c, h]. Mirrors C profile_y_f32.
    const input = node.inputs.input;
    const output = node.outputs.out;
    const [, h, w, c] = input.shape;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let ch = 0; ch < c; ch++) {
      for (let y = 0; y < h; y++) {
        let max_v = -Infinity;
        let sum_v = 0;
        for (let x = 0; x < w; x++) {
          const v = inBuf[(y * w + x) * c + ch];
          if (v > max_v) max_v = v;
          sum_v += v;
        }
        outBuf[ch * h + y] = max_v;
        outBuf[(c + ch) * h + y] = sum_v / w;
      }
    }
  }
