export function _cpuSpatialSoftargmaxY(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    const h = input.shape[1];
    const w = input.shape[2];
    const c = input.shape[3];
    const n = input.shape[0];
    for (let batch = 0; batch < n; batch++) {
      const inBase = batch * h * w * c;
      const outBatchBase = batch * c * w;
      for (let ch = 0; ch < c; ch++) {
        const outBase = outBatchBase + ch * w;
        for (let x = 0; x < w; x++) {
          let maxLogit = -Infinity;
          for (let y = 0; y < h; y++) {
            const v = inBuf[inBase + (y * w + x) * c + ch];
            if (v > maxLogit) maxLogit = v;
          }
          let denom = 0;
          let weighted = 0;
          for (let y = 0; y < h; y++) {
            const ev = Math.exp(inBuf[inBase + (y * w + x) * c + ch] - maxLogit);
            denom += ev;
            weighted += ev * ((y + 0.5) / h);
          }
          outBuf[outBase + x] = denom > 0 ? weighted / denom : 0;
        }
      }
    }
  }
