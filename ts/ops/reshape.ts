import { assertRawQuantizedShapeTensors } from './quantizedShape.js';

export function _cpuReshape(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    assertRawQuantizedShapeTensors(node, [input, output], node.opType);
    output.buffer.set(input.buffer);
  }
