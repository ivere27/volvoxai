export function _cpuSigmoid(node) {

    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    for (let i = 0; i < inBuf.length; i++) outBuf[i] = 1.0 / (1.0 + Math.exp(-inBuf[i]));
  }