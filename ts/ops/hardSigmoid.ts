export function _cpuHardSigmoid(node) {

    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    for (let i = 0; i < inBuf.length; i++) {
        const x = inBuf[i];
        let v = x + 3.0;
        if (v < 0.0) v = 0.0;
        else if (v > 6.0) v = 6.0;
        outBuf[i] = v / 6.0;
    }
  }