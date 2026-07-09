export function _cpuLeakyReLU(node) {

    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    const alpha = node.params.alpha !== undefined ? node.params.alpha : 0.01;
    for (let i = 0; i < inBuf.length; i++) {
        const v = inBuf[i];
        outBuf[i] = v > 0 ? v : alpha * v;
    }
  }
