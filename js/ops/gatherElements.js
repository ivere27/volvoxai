export function _cpuGatherElements(node) {

    // ONNX GatherElements: output has the same shape as `indices`; each element
    // pulls from `data` at the same coordinates but with the `axis` coordinate
    // replaced by the corresponding index value.
    const data = node.inputs.input || node.inputs.data;
    const indices = node.inputs.indices;
    const out = node.outputs.out;
    const dBuf = data.buffer;
    const iBuf = indices.buffer;
    const oBuf = out.buffer;
    const dShape = data.shape;
    const iShape = indices.shape;
    let axis = node.params.axis !== undefined ? node.params.axis : 0;
    if (axis < 0) axis += dShape.length;

    const dStrides = new Array(dShape.length);
    { let s = 1; for (let k = dShape.length - 1; k >= 0; k--) { dStrides[k] = s; s *= dShape[k]; } }
    const iStrides = new Array(iShape.length);
    { let s = 1; for (let k = iShape.length - 1; k >= 0; k--) { iStrides[k] = s; s *= iShape[k]; } }

    const rank = iShape.length;
    const coord = new Array(rank);
    for (let lin = 0; lin < oBuf.length; lin++) {
      let rem = lin;
      for (let k = 0; k < rank; k++) { coord[k] = Math.floor(rem / iStrides[k]); rem %= iStrides[k]; }
      let idx = iBuf[lin] | 0;
      if (idx < 0) idx += dShape[axis];
      let off = 0;
      for (let k = 0; k < rank; k++) off += (k === axis ? idx : coord[k]) * dStrides[k];
      oBuf[lin] = dBuf[off];
    }
  }
