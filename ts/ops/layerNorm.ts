export function _cpuLayerNorm(node) {

    const inBuf = node.inputs.input.buffer;
    const wBuf = node.inputs.weight.buffer;
    const bBuf = node.inputs.bias?.buffer;
    const outBuf = node.outputs.out.buffer;
    const d_model = node.params.d_model ?? node.inputs.input.shape.at(-1);
    if (d_model !== node.inputs.input.shape.at(-1)) {
      throw new Error(`LayerNorm d_model must match the last input dimension (${node.inputs.input.shape.at(-1)}).`);
    }
    const eps = node.params.eps ?? 1e-6;
    const seq_len = node.inputs.input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
    for (let i = 0; i < seq_len; i++) {
      const offset = i * d_model;
      let sum = 0, sq_sum = 0;
      for (let j = 0; j < d_model; j++) {
        const val = inBuf[offset + j];
        sum += val;
        sq_sum += val * val;
      }
      const mean = sum / d_model;
      const variance = sq_sum / d_model - mean * mean;
      const inv_std = 1 / Math.sqrt(variance + eps);
      for (let j = 0; j < d_model; j++) {
        const norm_val = (inBuf[offset + j] - mean) * inv_std;
        outBuf[offset + j] = norm_val * wBuf[j] + (bBuf ? bBuf[j] : 0);
      }
    }
  }
