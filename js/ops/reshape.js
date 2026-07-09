export function _cpuReshape(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    output.buffer.set(input.buffer);
  }