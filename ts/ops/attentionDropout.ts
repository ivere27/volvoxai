import { dropoutEffectiveSeed, dropoutHash, dropoutThreshold } from './dropout.js';

export function attentionDropoutProbability(node) {
  const probability = node?.params?.dropout ?? node?.params?.attention_dropout ?? 0;
  if (typeof probability !== "number" || !Number.isFinite(probability) || probability < 0 || probability >= 1) {
    throw new Error(`Attention node ${node?.id ?? "<unnamed>"} dropout must be finite and in [0, 1).`);
  }
  return probability;
}

export function attentionProbabilityIndex(batch, head, query, key, heads, queries, keys) {
  return (((batch * heads + head) * queries + query) * keys + key);
}

export function attentionDropoutEffectiveSeed(node, context, nodeIndex = 0) {
  const seedNode = {
    params: {
      seed: node?.params?.dropout_seed ?? node?.params?.training_seed ?? node?.params?.seed ?? 0,
    },
  };
  return dropoutEffectiveSeed(seedNode, context, nodeIndex);
}

/**
 * Return the inverted-dropout multiplier for a softmax probability. Inference
 * passes no context, so attention remains the original deterministic operator.
 */
export function attentionDropout(node, context, nodeIndex = 0) {
  const probability = attentionDropoutProbability(node);
  if (!context || probability === 0) return () => 1;
  const seed = attentionDropoutEffectiveSeed(node, context, nodeIndex);
  const threshold = dropoutThreshold(probability);
  const scale = 1 / (1 - probability);
  const counter = context.counter >>> 0;
  return (index) => dropoutHash(index, seed, counter, -1) >= threshold ? scale : 0;
}
