export function _cpuReLU(node) {

    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    for (let i = 0; i < inBuf.length; i++) {
        outBuf[i] = inBuf[i] > 0 ? inBuf[i] : 0;
    }
  }