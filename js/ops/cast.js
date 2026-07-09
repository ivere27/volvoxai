export function _cpuCast(node) {

    // Activations are Float32 throughout this engine, so a Cast is a plain copy.
    // If the target is an integer ONNX dtype we truncate toward zero to match
    // the numeric result of a real cast (e.g. float indices -> int32).
    const input = node.inputs.input || node.inputs.data;
    const output = node.outputs.out;
    const inBuf = input.buffer;
    const outBuf = output.buffer;
    const to = node.params.to;
    const intCast = to === "int32" || to === "int64" || to === "int8"
                 || to === 3 || to === 5 || to === 6 || to === 7;
    const n = Math.min(inBuf.length, outBuf.length);
    if (intCast) {
      for (let i = 0; i < n; i++) outBuf[i] = Math.trunc(inBuf[i]);
    } else {
      outBuf.set(inBuf.subarray(0, n));
    }
  }
