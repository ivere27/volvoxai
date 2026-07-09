export function _cpuClip(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    let min = node.params.min !== undefined ? node.params.min : -Infinity;
    let max = node.params.max !== undefined ? node.params.max : Infinity;
    if (node.inputs.min) min = node.inputs.min.buffer[0];
    if (node.inputs.max) max = node.inputs.max.buffer[0];
    for (let i = 0; i < input.buffer.length; i++) {
      let v = input.buffer[i];
      if (v < min) v = min;
      if (v > max) v = max;
      output.buffer[i] = v;
    }
  }
