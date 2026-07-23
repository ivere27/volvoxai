import { assertRawQuantizedShapeTensors } from './quantizedShape.js';

export function _cpuReshape(node) {
    const input = node.inputs.input || node.inputs.x || node.inputs.data;
    const output = node.outputs.out || Object.values(node.outputs || {})[0];
    if (!input || !output || input.dtype !== output.dtype ||
        !['float32', 'int32', 'int8', 'uint8'].includes(input.dtype) ||
        !input.buffer || !output.buffer || input.buffer.length !== output.buffer.length) {
      throw new Error(`${node.opType} node ${node.id ?? '<unnamed>'} requires equal-size same-dtype storage.`);
    }
    assertRawQuantizedShapeTensors(node, [input, output], node.opType);
    output.buffer.set(input.buffer);
  }
