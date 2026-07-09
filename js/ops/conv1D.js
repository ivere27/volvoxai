import { _pair } from '../Tensor.js';

export function _cpuConv1D(node) {

    const input = node.inputs.input;
    const weight = node.inputs.weight;
    const bias = node.inputs.bias ? node.inputs.bias.buffer : null;
    const output = node.outputs.out;
    const [in_c, in_l] = input.shape.slice(1);
    const [out_c, k_c, k] = weight.shape;
    const out_l = output.shape[2];
    const stride = _pair(node.params.stride, 1)[0];
    const padding = _pair(node.params.padding, 0)[0];
    const relu = node.params.relu;
    const inBuf = input.buffer;
    const wBuf = weight.buffer;
    const outBuf = output.buffer;
    for (let oc = 0; oc < out_c; oc++) {
      for (let x = 0; x < out_l; x++) {
        let sum = bias ? bias[oc] : 0;
        for (let ic = 0; ic < in_c; ic++) {
          for (let k_idx = 0; k_idx < k; k_idx++) {
            const ix = x * stride + k_idx - padding;
            if (ix >= 0 && ix < in_l) {
              const in_idx = ic * in_l + ix;
              const w_idx = (oc * in_c + ic) * k + k_idx;
              sum += inBuf[in_idx] * wBuf[w_idx];
            }
          }
        }
        if (relu && sum < 0) sum = 0;
        outBuf[oc * out_l + x] = sum;
      }
    }
  }