export function _cpuMeanHeight(node) {

    const input = node.inputs.input;
    const output = node.outputs.out;
    const [, h, w, c] = input.shape;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    for (let ch = 0; ch < c; ch++) {
      for (let x = 0; x < w; x++) {
        let sum = 0;
        for (let y = 0; y < h; y++) {
          sum += inBuf[(y * w + x) * c + ch];
        }
        outBuf[ch * w + x] = sum / h;
      }
    }
  }
