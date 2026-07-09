export function _cpuConcat2(node) {

    // N-way concat for the blueprint keys emitted by exporters: input0, input1, ...
    // Current model packages concatenate flattened detection heads, so flat append is
    // enough for the supported axis layouts.
    const outBuf = node.outputs.out.buffer;
    let off = 0;
    const entries = Object.entries(node.inputs).sort(([a], [b]) => {
      const ai = /^input(\d+)$/.exec(a);
      const bi = /^input(\d+)$/.exec(b);
      if (ai && bi) return Number(ai[1]) - Number(bi[1]);
      return a.localeCompare(b);
    });
    for (const [, t] of entries) {
      outBuf.set(t.buffer, off);
      off += t.buffer.length;
    }
  }
