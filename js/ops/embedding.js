export function _cpuEmbedding(node) {

    const tokens = node.inputs.input.buffer;
    const wBuf = node.inputs.weight.buffer;
    const outBuf = node.outputs.out.buffer;
    const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
    const seq_len = tokens.length;
    for (let i = 0; i < seq_len; i++) {
      const token_id = tokens[i];
      for (let j = 0; j < d_model; j++) {
        outBuf[i * d_model + j] = wBuf[token_id * d_model + j];
      }
    }
  }