import { visionProfileKernelDescriptor } from './visionProfileKernel.js';

export function _cpuProfileY(node) {

    // Row (H) profile: collapse W, keep H -> [N, 2C, H].
    const { input, output, n, h, w, c } = visionProfileKernelDescriptor(
      node, 'ProfileY', 'profile-y',
    );
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let batch = 0; batch < n; batch++) {
      const inBase = batch * h * w * c;
      const outBase = batch * 2 * c * h;
      for (let ch = 0; ch < c; ch++) {
        for (let y = 0; y < h; y++) {
          let max_v = -Infinity;
          let sum_v = 0;
          for (let x = 0; x < w; x++) {
            const v = inBuf[inBase + (y * w + x) * c + ch];
            if (v > max_v) max_v = v;
            sum_v += v;
          }
          outBuf[outBase + ch * h + y] = max_v;
          outBuf[outBase + (c + ch) * h + y] = sum_v / w;
        }
      }
    }
  }
