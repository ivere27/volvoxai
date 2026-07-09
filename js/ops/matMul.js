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
    const scale = node.inputs.scale ? node.inputs.scale.buffer : null;
    const bias = node.inputs.bias ? node.inputs.bias.buffer : null;
    // Two weight layouts coexist: PyTorch Linear stores [d_out, d_in] (out = x·Wᵀ),
    // while GPT-Neo/Conv1D stores [d_in, d_out] (out = x·W). Detect by shape so both
    // model families work. INT8-quantized weights are always [d_out, d_in] + per-out scale.
    const w0 = weight.shape.length >= 2 ? weight.shape[0] : N;
    const w1 = weight.shape.length >= 2 ? weight.shape[1] : K;
    // Prefer the per-node layout resolved at load (handles square weights); fall back to
    // shape detection. 'dout' = [d_out, d_in] (x·Wᵀ), 'din' = [d_in, d_out] (x·W).
    const doutFirst = scale ? true : (node.wLayout ? node.wLayout === "dout" : (w0 === N && w1 === K));
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        if (doutFirst) {
          for (let k = 0; k < K; k++) sum += inBuf[i * K + k] * wBuf[j * K + k];
        } else {
          for (let k = 0; k < K; k++) sum += inBuf[i * K + k] * wBuf[k * N + j];
        }
        if (scale) sum *= scale[j];
        if (bias) sum += bias[j];
        outBuf[i * N + j] = sum;
      }
    }
  }