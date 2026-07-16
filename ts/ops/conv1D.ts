import { normalizeSpatialPair } from './spatialParameters.js';

export function _cpuConv1D(node) {

    const input = node.inputs.input;
    const weight = node.inputs.weight;
    const bias = node.inputs.bias ? node.inputs.bias.buffer : null;
    const output = node.outputs.out;
    const [batch, in_c, in_l] = input.shape;
    const [out_c, k_c, k] = weight.shape;
    const out_l = output.shape[2];
    const stride = normalizeSpatialPair(node.params.stride, 1)[0];
    const padding = normalizeSpatialPair(node.params.padding, 0)[0];
    const relu = node.params.relu;
    const groups = node.params.groups ?? 1;
    if (!Number.isInteger(groups) || groups <= 0 || in_c % groups || out_c % groups || k_c !== in_c / groups) {
      throw new Error('Conv1D requires [out_channels, in_channels_per_group, kernel] weights compatible with groups.');
    }
    const inBuf = input.buffer;
    const wBuf = weight.buffer;
    const outBuf = output.buffer;
    const groupOut = out_c / groups;
    for (let b = 0; b < batch; b++) {
      const inBase = b * in_c * in_l;
      const outBase = b * out_c * out_l;
      for (let oc = 0; oc < out_c; oc++) {
        const inputStart = Math.floor(oc / groupOut) * k_c;
        for (let x = 0; x < out_l; x++) {
          let sum = bias ? bias[oc] : 0;
          for (let localIc = 0; localIc < k_c; localIc++) {
            const ic = inputStart + localIc;
            for (let k_idx = 0; k_idx < k; k_idx++) {
              const ix = x * stride + k_idx - padding;
              if (ix >= 0 && ix < in_l) {
                const in_idx = inBase + ic * in_l + ix;
                const w_idx = (oc * k_c + localIc) * k + k_idx;
                sum += inBuf[in_idx] * wBuf[w_idx];
              }
            }
          }
          if (relu && sum < 0) sum = 0;
          outBuf[outBase + oc * out_l + x] = sum;
        }
      }
    }
  }
