export function _cpuTranspose(node) {
    const input = node.inputs.input || node.inputs.x || node.inputs.data;
    const output = node.outputs.out || Object.values(node.outputs || {})[0];
    if (!input || !output || !['float32', 'int32', 'int8', 'uint8'].includes(input.dtype) ||
        output.dtype !== input.dtype || !input.buffer || !output.buffer ||
        !Array.isArray(input.shape) || input.shape.length < 1 || input.shape.length > 8) {
      throw new Error(`Transpose node ${node.id ?? '<unnamed>'} requires rank-1..8 same-dtype F32/I32/I8/U8 tensors.`);
    }
    const inBuf = input.buffer;
    const inShape = input.shape;
    const outBuf = output.buffer;
    const perm = node.params?.perm || [...Array(inShape.length).keys()].reverse();
    if (!Array.isArray(perm) || perm.length !== inShape.length ||
        new Set(perm).size !== perm.length ||
        perm.some((axis) => !Number.isInteger(axis) || axis < 0 || axis >= inShape.length)) {
      throw new Error(`Transpose node ${node.id ?? '<unnamed>'} has an invalid permutation.`);
    }
    
    const inStrides = new Array(inShape.length);
    let s = 1;
    for (let i = inShape.length - 1; i >= 0; i--) {
        inStrides[i] = s;
        s *= inShape[i];
    }
    const outShape = perm.map(p => inShape[p]);
    if (output.shape.length !== outShape.length ||
        output.shape.some((dimension, index) => dimension !== outShape[index]) ||
        output.buffer.length !== input.buffer.length) {
      throw new Error(`Transpose node ${node.id ?? '<unnamed>'} output shape/storage is incompatible.`);
    }
    const outStrides = new Array(outShape.length);
    s = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
        outStrides[i] = s;
        s *= outShape[i];
    }
    
    const elements = inBuf.length;
    for (let i = 0; i < elements; i++) {
        let inIdx = 0;
        let temp = i;
        for (let j = 0; j < outShape.length; j++) {
            const outCoord = Math.floor(temp / outStrides[j]);
            temp %= outStrides[j];
            inIdx += outCoord * inStrides[perm[j]];
        }
        outBuf[i] = inBuf[inIdx];
    }
  }
