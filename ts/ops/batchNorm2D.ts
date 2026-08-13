import {
  assertShapeKernelOutput,
  assertShapeKernelTensor,
  sameShape,
} from './shapeKernelValidation.js';

export function _cpuBatchNorm2D(node) {
    const input = node.inputs.input || node.inputs.x;
    const weight = node.inputs.weight || node.inputs.scale;
    const bias = node.inputs.bias || node.inputs.b;
    const running_mean = node.inputs.running_mean || node.inputs.mean;
    const running_var = node.inputs.running_var || node.inputs.var;
    const output = node.outputs?.out;
    assertShapeKernelTensor(input, 'BatchNorm2D input', {
      dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
    });
    const [b, h, w, c] = input.shape;
    for (const [name, tensor] of [
      ['weight', weight], ['bias', bias], ['running_mean', running_mean],
      ['running_var', running_var],
    ]) {
      assertShapeKernelTensor(tensor, `BatchNorm2D ${name}`, {
        dtypes: ['float32'], minimumRank: 1, maximumRank: 1,
      });
      if (!sameShape(tensor.shape, [c])) {
        throw new Error(`BatchNorm2D ${name} must have shape [${c}].`);
      }
    }
    assertShapeKernelOutput(
      output, input.shape, 'float32', undefined, 'BatchNorm2D',
    );
    const eps = node.params?.eps ?? 1e-5;
    if (typeof eps !== 'number' || !Number.isFinite(eps) || eps <= 0) {
      throw new Error('BatchNorm2D eps must be positive and finite.');
    }
    const outBuf = output.buffer;
    
    for (let batch = 0; batch < b; batch++) {
        for (let chan = 0; chan < c; chan++) {
            const w_val = weight.buffer[chan];
            const b_val = bias.buffer[chan];
            const rm_val = running_mean.buffer[chan];
            const rv_val = running_var.buffer[chan];
            
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const idx = ((batch * h + y) * w + x) * c + chan;
                    const val = input.buffer[idx];
                    outBuf[idx] = ((val - rm_val) / Math.sqrt(rv_val + eps)) * w_val + b_val;
                }
            }
        }
    }
}
