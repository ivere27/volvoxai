export function _cpuSplit(node) {

    const input = node.inputs.input;
    const inBuf = input.buffer;
    const inShape = input.shape;
    let axis = node.params.axis || 0;
    if (axis < 0) axis += inShape.length;
    const outKeys = Object.keys(node.outputs).sort();
    const numOutputs = outKeys.length;
    const splitSize = inShape[axis] / numOutputs;
    let outerSize = 1;
    for (let i = 0; i < axis; i++) outerSize *= inShape[i];
    let innerSize = 1;
    for (let i = axis + 1; i < inShape.length; i++) innerSize *= inShape[i];
    const chunkSize = splitSize * innerSize;
    for (let o = 0; o < numOutputs; o++) {
        const outBuf = node.outputs[outKeys[o]].buffer;
        for (let i = 0; i < outerSize; i++) {
            const inOffset = (i * inShape[axis] + o * splitSize) * innerSize;
            const outOffset = i * chunkSize;
            outBuf.set(inBuf.subarray(inOffset, inOffset + chunkSize), outOffset);
        }
    }
  }