export function _cpuSplit(node) {
    const input = node.inputs.input || node.inputs.x || node.inputs.data;
    if (!input || !['float32', 'int32'].includes(input.dtype) || !input.buffer ||
        !Array.isArray(input.shape) || input.shape.length < 1) {
      throw new Error(`Split node ${node.id ?? '<unnamed>'} requires an F32 or I32 input tensor.`);
    }
    const inBuf = input.buffer;
    const inShape = input.shape;
    let axis = node.params?.axis ?? 0;
    if (axis < 0) axis += inShape.length;
    if (!Number.isInteger(axis) || axis < 0 || axis >= inShape.length) {
      throw new Error(`Split node ${node.id ?? '<unnamed>'} has an invalid axis.`);
    }
    // The graph's declared output order is Split's slice order. Lexical sorting
    // would incorrectly place out10 before out2.
    const outKeys = Object.keys(node.outputs);
    const numOutputs = outKeys.length;
    if (numOutputs === 0 || inShape[axis] % numOutputs !== 0) {
      throw new Error(`Split node ${node.id ?? '<unnamed>'} requires equal-sized output slices.`);
    }
    const splitSize = inShape[axis] / numOutputs;
    let outerSize = 1;
    for (let i = 0; i < axis; i++) outerSize *= inShape[i];
    let innerSize = 1;
    for (let i = axis + 1; i < inShape.length; i++) innerSize *= inShape[i];
    const chunkSize = splitSize * innerSize;
    for (let o = 0; o < numOutputs; o++) {
        const output = node.outputs[outKeys[o]];
        const expectedShape = [...inShape];
        expectedShape[axis] = splitSize;
        if (!output || output.dtype !== input.dtype || !output.buffer ||
            output.shape.length !== expectedShape.length ||
            output.shape.some((dimension, index) => dimension !== expectedShape[index]) ||
            output.buffer.length !== outerSize * chunkSize) {
          throw new Error(`Split node ${node.id ?? '<unnamed>'} has an incompatible ${input.dtype} output.`);
        }
        const outBuf = output.buffer;
        for (let i = 0; i < outerSize; i++) {
            const inOffset = (i * inShape[axis] + o * splitSize) * innerSize;
            const outOffset = i * chunkSize;
            outBuf.set(inBuf.subarray(inOffset, inOffset + chunkSize), outOffset);
        }
    }
  }
