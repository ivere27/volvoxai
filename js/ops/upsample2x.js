export function _cpuUpsample2x(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    const [n, h, w, c] = input.shape;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let b = 0; b < n; b++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          for (let ch = 0; ch < c; ch++) {
            const v = inBuf[((b * h + y) * w + x) * c + ch];
            const oy = y * 2;
            const ox = x * 2;
            outBuf[((b * h * 2 + oy) * w * 2 + ox) * c + ch] = v;
            outBuf[((b * h * 2 + oy) * w * 2 + ox + 1) * c + ch] = v;
            outBuf[((b * h * 2 + oy + 1) * w * 2 + ox) * c + ch] = v;
            outBuf[((b * h * 2 + oy + 1) * w * 2 + ox + 1) * c + ch] = v;
          }
        }
      }
    }
  }
