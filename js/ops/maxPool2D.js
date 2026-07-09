export function _cpuMaxPool2D(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    const [n, h, w, c] = input.shape;
    const ky = node.params.kernel[0];
    const kx = node.params.kernel[1];
    const sy = node.params.stride[0];
    const sx = node.params.stride[1];
    const pad_y = node.params.padding ? node.params.padding[0] : 0;
    const pad_x = node.params.padding ? node.params.padding[1] : 0;
    const out_h = output.shape[1];
    const out_w = output.shape[2];
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let b = 0; b < n; b++) {
      for (let oy = 0; oy < out_h; oy++) {
        for (let ox = 0; ox < out_w; ox++) {
          for (let ch = 0; ch < c; ch++) {
            let best = -Infinity;
            for (let dy = 0; dy < ky; dy++) {
              for (let dx = 0; dx < kx; dx++) {
                const ih = oy * sy + dy - pad_y;
                const iw = ox * sx + dx - pad_x;
                if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                  const v = inBuf[((b * h + ih) * w + iw) * c + ch];
                  if (v > best) best = v;
                }
              }
            }
            outBuf[((b * out_h + oy) * out_w + ox) * c + ch] = best;
          }
        }
      }
    }
  }
