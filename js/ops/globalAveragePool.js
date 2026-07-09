export function _cpuGlobalAveragePool(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    const B = input.shape[0];
    const H = input.shape[1];
    const W = input.shape[2];
    const C = input.shape[3];
    const spatial = H * W;
    for (let b = 0; b < B; b++) {
      for (let c = 0; c < C; c++) {
        let sum = 0;
        for (let i = 0; i < spatial; i++) {
          const y = Math.floor(i / W);
          const x = i - y * W;
          sum += input.buffer[((b * H + y) * W + x) * C + c];
        }
        output.buffer[b * C + c] = sum / spatial;
      }
    }
  }
