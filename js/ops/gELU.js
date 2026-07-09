export function _cpuGELU(node) {

    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    for (let i = 0; i < inBuf.length; i++) {
      const x = inBuf[i];
      outBuf[i] = 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    }
  }