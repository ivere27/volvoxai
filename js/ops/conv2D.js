import { _pair } from '../Tensor.js';

export function _cpuConv2D(node) {

    const input = node.inputs.input;
    const weight = node.inputs.weight;
    const bias = node.inputs.bias ? node.inputs.bias.buffer : null;
    const output = node.outputs.out;
    const [batch, in_h, in_w, in_c] = input.shape;
    const [k_h, k_w] = weight.shape;
    const out_c = output.shape[3];
    const out_h = output.shape[1];
    const out_w = output.shape[2];
    const [stride_y, stride_x] = _pair(node.params.stride, 1);
    const [pad_y, pad_x] = _pair(node.params.padding, 0);
    const pads = node.params.pads || [pad_y, pad_x, pad_y, pad_x];
    const [dil_y, dil_x] = _pair(node.params.dilation, 1);
    const inBuf = input.buffer;
    const wBuf = weight.buffer;
    const outBuf = output.buffer;
    const groups = node.params.groups || 1;
    const group_out = out_c / groups;
    const group_in = weight.shape[2];
    for (let b = 0; b < batch; b++) {
      for (let oh = 0; oh < out_h; oh++) {
        for (let ow = 0; ow < out_w; ow++) {
          for (let oc = 0; oc < out_c; oc++) {
            let sum = 0;
            if (groups === in_c) {
              const mult = out_c / in_c;
              const ic = Math.floor(oc / mult);
              const m = oc - ic * mult;
              for (let kh = 0; kh < k_h; kh++) {
                for (let kw = 0; kw < k_w; kw++) {
                  const ih = oh * stride_y + kh * dil_y - pads[0];
                  const iw = ow * stride_x + kw * dil_x - pads[1];
                  if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    const in_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic;
                    const w_idx = (((kh * k_w + kw) * in_c + ic) * mult) + m;
                    sum += inBuf[in_idx] * wBuf[w_idx];
                  }
                }
              }
            } else {
              const g = Math.floor(oc / group_out);
              const in_start = g * group_in;
              for (let icl = 0; icl < group_in; icl++) {
                const ic = in_start + icl;
                for (let kh = 0; kh < k_h; kh++) {
                  for (let kw = 0; kw < k_w; kw++) {
                    const ih = oh * stride_y + kh * dil_y - pads[0];
                    const iw = ow * stride_x + kw * dil_x - pads[1];
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                      const in_idx = ((b * in_h + ih) * in_w + iw) * in_c + ic;
                      const w_idx = (((kh * k_w + kw) * group_in + icl) * out_c) + oc;
                      sum += inBuf[in_idx] * wBuf[w_idx];
                    }
                  }
                }
              }
            }
            if (bias) sum += bias[oc];
            if (node.params.relu === 1 && sum < 0) sum = 0;
            else if (node.params.relu >= 2) sum = Math.min(Math.max(sum, 0), 6);
            outBuf[((b * out_h + oh) * out_w + ow) * out_c + oc] = sum;
          }
        }
      }
    }
  }
