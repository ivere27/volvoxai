export function _cpuArgMax(node) {

    // ONNX ArgMax over a single axis. Output holds the winning indices (stored as
    // float32 here); works for any rank and either keepdims setting because the
    // output element count is outerBlock * innerBlock regardless.
    const input = node.inputs.input || node.inputs.data;
    const out = node.outputs.out;
    const inBuf = input.buffer;
    const outBuf = out.buffer;
    const shape = input.shape;
    let axis = node.params.axis !== undefined ? node.params.axis : 0;
    if (axis < 0) axis += shape.length;
    const axisSize = shape[axis];

    let innerBlock = 1;
    for (let i = axis + 1; i < shape.length; i++) innerBlock *= shape[i];
    let outerBlock = 1;
    for (let i = 0; i < axis; i++) outerBlock *= shape[i];

    let o = 0;
    for (let ob = 0; ob < outerBlock; ob++) {
      for (let ib = 0; ib < innerBlock; ib++) {
        const base = ob * axisSize * innerBlock + ib;
        let best = inBuf[base];
        let bestIdx = 0;
        for (let a = 1; a < axisSize; a++) {
          const v = inBuf[base + a * innerBlock];
          if (v > best) { best = v; bestIdx = a; }
        }
        outBuf[o++] = bestIdx;
      }
    }
  }
