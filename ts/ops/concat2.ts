import { assertRawQuantizedShapeTensors } from './quantizedShape.js';

export function _cpuConcat2(node) {

    const output = node.outputs.out;
    if (!output || !['float32', 'int32', 'int8', 'uint8'].includes(output.dtype) ||
        !output.buffer) {
      throw new Error(`Concat node ${node.id || "<unnamed>"} requires typed output storage.`);
    }
    const outBuf = output.buffer;
    const rank = output.shape.length;
    let axis = node.params?.axis ?? 0;
    if (axis < 0) axis += rank;
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank) {
      throw new Error(`Concat node ${node.id || "<unnamed>"} has an invalid axis.`);
    }

    const preferred = ['input', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const seen = new Set();
    const entries: Array<[string, any]> = [];
    for (const key of preferred) {
      if (node.inputs[key]) {
        entries.push([key, node.inputs[key]]);
        seen.add(key);
      }
    }
    entries.push(...Object.entries(node.inputs)
      .filter(([key, tensor]) => tensor && !seen.has(key))
      .sort(([left], [right]) => {
        const li = /^input(\d+)$/.exec(left);
        const ri = /^input(\d+)$/.exec(right);
        if (li && ri) return Number(li[1]) - Number(ri[1]);
        return left.localeCompare(right);
      }));

    const inner = output.shape.slice(axis + 1).reduce((count, dim) => count * dim, 1);
    const outer = output.shape.slice(0, axis).reduce((count, dim) => count * dim, 1);
    const outputAxis = output.shape[axis];
    let summedAxis = 0;
    for (const [, tensor] of entries) {
      if (tensor.dtype !== output.dtype || !tensor.buffer ||
          tensor.shape.length !== rank ||
          tensor.shape.some((dim, index) => index !== axis && dim !== output.shape[index])) {
        throw new Error(`Concat node ${node.id || "<unnamed>"} has incompatible input shapes or dtypes.`);
      }
      summedAxis += tensor.shape[axis];
    }
    if (summedAxis !== outputAxis) {
      throw new Error(`Concat node ${node.id || "<unnamed>"} input axes do not match its output.`);
    }
    const quantized = assertRawQuantizedShapeTensors(node,
      [...entries.map(([, tensor]) => tensor), output], 'Concat');
    if (quantized && node.params?.sigmoid) {
      throw new Error(`Concat node ${node.id || '<unnamed>'} cannot fuse sigmoid into quantized byte storage; insert an explicit F32 boundary.`);
    }
    if (output.dtype === 'int32' && node.params?.sigmoid) {
      throw new Error(`Concat node ${node.id || '<unnamed>'} cannot fuse sigmoid into I32 storage.`);
    }

    for (let outerIndex = 0; outerIndex < outer; outerIndex++) {
      let axisOffset = 0;
      for (const [, tensor] of entries) {
        const inputAxis = tensor.shape[axis];
        const block = inputAxis * inner;
        const srcStart = outerIndex * block;
        const dstStart = (outerIndex * outputAxis + axisOffset) * inner;
        if (node.params?.sigmoid) {
          for (let i = 0; i < block; i++) {
            outBuf[dstStart + i] = 1 / (1 + Math.exp(-tensor.buffer[srcStart + i]));
          }
        } else {
          outBuf.set(tensor.buffer.subarray(srcStart, srcStart + block), dstStart);
        }
        axisOffset += inputAxis;
      }
    }
  }
