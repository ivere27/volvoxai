import { visionProfileKernelDescriptor } from './visionProfileKernel.js';

export function _cpuProfileX(node) {

    // Column (W) profile: collapse H, keep W -> [N, 2C, W].
    const { input, output, n, h, w, c } = visionProfileKernelDescriptor(
      node, 'ProfileX', 'profile-x',
    );
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let batch = 0; batch < n; batch++) {
      const inBase = batch * h * w * c;
      const outBase = batch * 2 * c * w;
      for (let ch = 0; ch < c; ch++) {
        for (let x = 0; x < w; x++) {
          let max_v = -Infinity;
          let sum_v = 0;
          for (let y = 0; y < h; y++) {
            const v = inBuf[inBase + (y * w + x) * c + ch];
            if (v > max_v) max_v = v;
            sum_v += v;
          }
          outBuf[outBase + ch * w + x] = max_v;
          outBuf[outBase + (c + ch) * w + x] = sum_v / h;
        }
      }
    }
  }
