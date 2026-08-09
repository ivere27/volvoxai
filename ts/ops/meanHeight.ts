import { visionProfileKernelDescriptor } from './visionProfileKernel.js';

export function _cpuMeanHeight(node) {
    const { input, output, n, h, w, c } = visionProfileKernelDescriptor(
      node, 'MeanHeight', 'mean-height',
    );
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let batch = 0; batch < n; batch++) {
      const inBase = batch * h * w * c;
      const outBase = batch * c * w;
      for (let ch = 0; ch < c; ch++) {
        for (let x = 0; x < w; x++) {
          let sum = 0;
          for (let y = 0; y < h; y++) {
            sum += inBuf[inBase + (y * w + x) * c + ch];
          }
          outBuf[outBase + ch * w + x] = sum / h;
        }
      }
    }
  }
