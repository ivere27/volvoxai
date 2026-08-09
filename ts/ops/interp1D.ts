import {
  assertShapeKernelOutput,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';

export function _cpuInterp1D(node) {

    const input = node.inputs?.input;
    const output = node.outputs?.out;
    assertShapeKernelTensor(input, 'Interpolate1D input', {
      dtypes: ['float32'], minimumRank: 3, maximumRank: 3,
    });
    const [batch, c, in_l] = input.shape;
    const out_l = node.params?.size;
    if (!Number.isSafeInteger(out_l) || out_l <= 0) {
      throw new Error('Interpolate1D size must be a positive safe integer.');
    }
    assertShapeKernelOutput(
      output, [batch, c, out_l], 'float32', undefined, 'Interpolate1D',
    );
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    // Half-pixel (align_corners=false) mapping, matching interp1d_f32 and the
    // WebGPU shader (PyTorch F.interpolate default).
    const scale = in_l / out_l;
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
