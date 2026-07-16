export function _cpuMatMul(node) {

    const input = node.inputs.input;
    const weight = node.inputs.weight;
    const output = node.outputs.out;
    const M = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const K = input.shape[input.shape.length - 1];
    const N = output.shape[output.shape.length - 1];
    const inBuf = input.buffer;
    const wBuf = weight.buffer;
    const outBuf = output.buffer;
    const scaleTensor = node.inputs.scale || node.inputs.weight_scale;
    const zeroPointTensor = node.inputs.zero_point || node.inputs.weight_zero_point;
    const scale = scaleTensor ? scaleTensor.buffer : null;
    const zeroPoint = zeroPointTensor ? zeroPointTensor.buffer : null;
    const bias = node.inputs.bias ? node.inputs.bias.buffer : null;
    // Two portable weight layouts coexist: output-major [d_out,d_in] (out=x·Wᵀ)
    // and input-major [d_in,d_out] (out=x·W). Non-square shapes disambiguate them.
    // INT8-quantized weights are always output-major with per-output scale.
    const w0 = weight.shape.length >= 2 ? weight.shape[0] : N;
    const w1 = weight.shape.length >= 2 ? weight.shape[1] : K;
    // Prefer the per-node layout resolved at load (handles square weights); fall back to
    // shape detection. 'dout' = [d_out, d_in] (x·Wᵀ), 'din' = [d_in, d_out] (x·W).
    const doutFirst = scale ? true : (node.wLayout ? node.wLayout === "dout" : (w0 === N && w1 === K));
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        const zp = zeroPoint ? zeroPoint[zeroPoint.length === 1 ? 0 : j] : 0;
        if (doutFirst) {
          for (let k = 0; k < K; k++) sum += inBuf[i * K + k] * (wBuf[j * K + k] - zp);
        } else {
          for (let k = 0; k < K; k++) sum += inBuf[i * K + k] * (wBuf[k * N + j] - zp);
        }
        if (scale) sum *= scale[scale.length === 1 ? 0 : j];
        if (bias) sum += bias[j];
        outBuf[i * N + j] = sum;
      }
    }
  }
