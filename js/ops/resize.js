export function _cpuResize(node) {

    const inp = node.inputs.input, out = node.outputs.out;
    const [b, inH, inW, c] = inp.shape;
    const [, outH, outW] = out.shape;
    const src = inp.buffer, dst = out.buffer;
    if (node.opType === "ResizeNearest2D" || node.params.mode === "nearest") {
      for (let n = 0; n < b; n++) {
        for (let y = 0; y < outH; y++) {
          let iy = Math.floor(y * inH / outH);
          if (iy >= inH) iy = inH - 1;
          for (let x = 0; x < outW; x++) {
            let ix = Math.floor(x * inW / outW);
            if (ix >= inW) ix = inW - 1;
            for (let ch = 0; ch < c; ch++) {
              dst[((n * outH + y) * outW + x) * c + ch] = src[((n * inH + iy) * inW + ix) * c + ch];
            }
          }
        }
      }
      return;
    }

    // Bilinear, half-pixel (align_corners=False) to match PyTorch F.interpolate.
    const sy = inH / outH, sx = inW / outW;
    for (let n = 0; n < b; n++) {
      for (let y = 0; y < outH; y++) {
        let iy = (y + 0.5) * sy - 0.5; if (iy < 0) iy = 0;
        const y0 = Math.min(inH - 1, Math.floor(iy)), y1 = Math.min(inH - 1, y0 + 1), dy = iy - y0;
        for (let x = 0; x < outW; x++) {
          let ix = (x + 0.5) * sx - 0.5; if (ix < 0) ix = 0;
          const x0 = Math.min(inW - 1, Math.floor(ix)), x1 = Math.min(inW - 1, x0 + 1), dx = ix - x0;
          for (let ch = 0; ch < c; ch++) {
            const v00 = src[((n * inH + y0) * inW + x0) * c + ch];
            const v01 = src[((n * inH + y0) * inW + x1) * c + ch];
            const v10 = src[((n * inH + y1) * inW + x0) * c + ch];
            const v11 = src[((n * inH + y1) * inW + x1) * c + ch];
            dst[((n * outH + y) * outW + x) * c + ch] =
              v00 * (1 - dy) * (1 - dx) + v01 * (1 - dy) * dx + v10 * dy * (1 - dx) + v11 * dy * dx;
          }
        }
      }
    }
  }
