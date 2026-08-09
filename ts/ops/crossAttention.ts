import {
  assertShapeKernelOutput,
  assertShapeKernelTensor,
  sameShape,
} from './shapeKernelValidation.js';

export function _cpuCrossAttention(node) {
    const qTensor = node.inputs?.q;
    const kvTensor = node.inputs?.kv;
    const weightTensor = node.inputs?.weight;
    const scaleTensor = node.inputs?.scale || null;
    const biasTensor = node.inputs?.bias || null;
    const output = node.outputs?.out;
    assertShapeKernelTensor(qTensor, 'CrossAttention q', {
      dtypes: ['float32'], minimumRank: 2, maximumRank: 3,
    });
    assertShapeKernelTensor(kvTensor, 'CrossAttention kv', {
      dtypes: ['float32'], minimumRank: qTensor.shape.length,
      maximumRank: qTensor.shape.length,
    });
    assertShapeKernelTensor(weightTensor, 'CrossAttention weight', {
      dtypes: ['float32'], minimumRank: 2, maximumRank: 2,
    });

    const qShape = qTensor.shape;
    const kvShape = kvTensor.shape;
    const rank = qShape.length;
    const batch = rank === 3 ? qShape[0] : 1;
    const seq_len_q = qShape[rank - 2];
    const seq_len_kv = kvShape[rank - 2];
    const d_model = qShape[rank - 1];
    const projection = 3 * d_model;
    if (!Number.isSafeInteger(projection) || kvShape[rank - 1] !== d_model ||
        (rank === 3 && kvShape[0] !== batch) ||
        !sameShape(weightTensor.shape, [projection, d_model])) {
      throw new Error('CrossAttention has incompatible batch, feature, or projection dimensions.');
    }
    for (const [name, tensor] of [['scale', scaleTensor], ['bias', biasTensor]]) {
      if (!tensor) continue;
      assertShapeKernelTensor(tensor, `CrossAttention ${name}`, {
        dtypes: ['float32'], minimumRank: 1, maximumRank: 1,
      });
      if (!sameShape(tensor.shape, [projection])) {
        throw new Error(`CrossAttention ${name} must have shape [${projection}].`);
      }
    }
    assertShapeKernelOutput(
      output, qShape, 'float32', undefined, 'CrossAttention',
    );
    const num_heads = node.params?.heads ?? 8;
    if (!Number.isSafeInteger(num_heads) || num_heads <= 0 || d_model % num_heads !== 0) {
      throw new Error(`CrossAttention heads must be a positive divisor of ${d_model}.`);
    }
    const q_in = qTensor.buffer;
    const kv_in = kvTensor.buffer;
    const wBuf = weightTensor.buffer;
    const scale = scaleTensor?.buffer || null;
    const bias = biasTensor?.buffer || null;
    const outBuf = output.buffer;
    const head_dim = d_model / num_heads;
    const scale_factor = 1 / Math.sqrt(head_dim);
    for (let b = 0; b < batch; b++) {
      const qBase = b * seq_len_q * d_model;
      const kvBase = b * seq_len_kv * d_model;
      for (let h = 0; h < num_heads; h++) {
        for (let q = 0; q < seq_len_q; q++) {
        const q_proj = new Float32Array(head_dim);
        for (let d = 0; d < head_dim; d++) {
          let sum = 0;
          const out_col = h * head_dim + d;
          for (let i = 0; i < d_model; i++) {
            sum += q_in[qBase + q * d_model + i] * wBuf[out_col * d_model + i];
          }
          if (scale) sum *= scale[out_col];
          if (bias) sum += bias[out_col];
          q_proj[d] = sum;
        }
        const logits = new Float32Array(seq_len_kv);
        let max_logit = -Infinity;
        for (let k = 0; k < seq_len_kv; k++) {
          let score = 0;
          for (let d = 0; d < head_dim; d++) {
            let k_val = 0;
            const out_col = d_model + h * head_dim + d;
            for (let i = 0; i < d_model; i++) {
              k_val += kv_in[kvBase + k * d_model + i] * wBuf[out_col * d_model + i];
            }
            if (scale) k_val *= scale[out_col];
            if (bias) k_val += bias[out_col];
            score += q_proj[d] * k_val;
          }
          score *= scale_factor;
          logits[k] = score;
          if (score > max_logit) max_logit = score;
        }
        let sum_exp = 0;
        for (let k = 0; k < seq_len_kv; k++) {
          const exp_val = Math.exp(logits[k] - max_logit);
          logits[k] = exp_val;
          sum_exp += exp_val;
        }
        for (let d = 0; d < head_dim; d++) {
          let out_val = 0;
          for (let k = 0; k < seq_len_kv; k++) {
            const w = logits[k] / sum_exp;
            let v_val = 0;
            const out_col = d_model * 2 + h * head_dim + d;
            for (let i = 0; i < d_model; i++) {
              v_val += kv_in[kvBase + k * d_model + i] * wBuf[out_col * d_model + i];
            }
            if (scale) v_val *= scale[out_col];
            if (bias) v_val += bias[out_col];
            out_val += w * v_val;
          }
          outBuf[qBase + q * d_model + h * head_dim + d] = out_val;
        }
      }
    }
    }
  }
