import { _cpuAttentionMask } from './sDPA.js';

export function _cpuCrossSDPA(node, execution: {
  probabilityMultiplier?: (index: number) => number;
} = {}) {
  const qShape = node.inputs.q.shape;
  const kShape = node.inputs.k.shape;
  const vShape = node.inputs.v.shape;
  const outputShape = node.outputs.out.shape;
  const rank = qShape.length;
  const batch = rank === 2 ? 1 : qShape[0];
  const batchOf = (shape) => shape.length === 2 ? 1 : shape[0];
  if ((rank !== 2 && rank !== 3) ||
      [kShape, vShape, outputShape].some((shape) => shape.length !== rank || batchOf(shape) !== batch) ||
      qShape[rank - 2] !== outputShape[rank - 2] || kShape[rank - 2] !== vShape[rank - 2] ||
      qShape[rank - 1] !== kShape[rank - 1] || qShape[rank - 1] !== vShape[rank - 1] ||
      qShape[rank - 1] !== outputShape[rank - 1]) {
    throw new Error(`CrossSDPA node ${node.id} requires compatible [batch?, sequence, d_model] Q/K/V/output tensors.`);
  }
  const q = node.inputs.q.buffer;
  const k = node.inputs.k.buffer;
  const v = node.inputs.v.buffer;
  const outBuf = node.outputs.out.buffer;
  const seqQ = qShape[rank - 2];
  const seqKV = kShape[rank - 2];
  const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
  const heads = node.params.heads || 8;
  const head_dim = d_model / heads;
  if (!Number.isInteger(heads) || heads <= 0 || !Number.isInteger(head_dim) || head_dim <= 0) {
    throw new Error(`CrossSDPA node ${node.id} has incompatible heads and d_model.`);
  }
  const scale = node.params.scale ?? 1 / Math.sqrt(head_dim);
  const causal = node.params.causal === true;
  const keeps = _cpuAttentionMask(node, batch, seqQ, seqKV);
  const probabilityMultiplier = execution.probabilityMultiplier || (() => 1);
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    const qBase = batchIndex * seqQ * d_model;
    const kvBase = batchIndex * seqKV * d_model;
    const outputBase = qBase;
    for (let h = 0; h < heads; h++) {
      for (let qi = 0; qi < seqQ; qi++) {
        const probabilities = new Float64Array(seqKV);
        let mx = -Infinity;
        let visibleKeys = 0;
        for (let ki = 0; ki < seqKV; ki++) {
          if ((causal && ki > qi) || !keeps(batchIndex, qi, ki)) continue;
          let s = 0;
          for (let d = 0; d < head_dim; d++) {
            s += q[qBase + qi * d_model + h * head_dim + d] *
              k[kvBase + ki * d_model + h * head_dim + d];
          }
          s *= scale;
          probabilities[ki] = s;
          if (s > mx) mx = s;
          visibleKeys++;
        }

        if (visibleKeys === 0) {
          for (let d = 0; d < head_dim; d++) {
            outBuf[outputBase + qi * d_model + h * head_dim + d] = 0;
          }
          continue;
        }

        let sum = 0;
        for (let ki = 0; ki < seqKV; ki++) {
          if ((causal && ki > qi) || !keeps(batchIndex, qi, ki)) continue;
          probabilities[ki] = Math.exp(probabilities[ki] - mx);
          sum += probabilities[ki];
        }
        for (let d = 0; d < head_dim; d++) {
          let o = 0;
          for (let ki = 0; ki < seqKV; ki++) {
            if ((causal && ki > qi) || !keeps(batchIndex, qi, ki)) continue;
            const probabilityIndex = (((batchIndex * heads + h) * seqQ + qi) * seqKV + ki);
            o += (probabilities[ki] / sum) * probabilityMultiplier(probabilityIndex) *
              v[kvBase + ki * d_model + h * head_dim + d];
          }
          outBuf[outputBase + qi * d_model + h * head_dim + d] = o;
        }
      }
    }
  }
}
