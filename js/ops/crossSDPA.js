export function _cpuCrossSDPA(node) {

    const q = node.inputs.q.buffer;
    const k = node.inputs.k.buffer;
    const v = node.inputs.v.buffer;
    const outBuf = node.outputs.out.buffer;
    const seqQ = node.inputs.q.shape[1];
    const seqKV = node.inputs.k.shape[1];
    const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
    const heads = node.params.heads || 8;
    const head_dim = d_model / heads;
    const scale = 1 / Math.sqrt(head_dim);
    for (let h = 0; h < heads; h++) {
      for (let qi = 0; qi < seqQ; qi++) {
        const logits = new Float32Array(seqKV);
        let mx = -Infinity;
        for (let ki = 0; ki < seqKV; ki++) {
          let s = 0;
          for (let d = 0; d < head_dim; d++) {
            s += q[qi * d_model + h * head_dim + d] * k[ki * d_model + h * head_dim + d];
          }
          s *= scale;
          logits[ki] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let ki = 0; ki < seqKV; ki++) {
          const e = Math.exp(logits[ki] - mx);
          logits[ki] = e;
          sum += e;
        }
        for (let d = 0; d < head_dim; d++) {
          let o = 0;
          for (let ki = 0; ki < seqKV; ki++) {
            o += (logits[ki] / sum) * v[ki * d_model + h * head_dim + d];
          }
          outBuf[qi * d_model + h * head_dim + d] = o;
        }
      }
    }
  }