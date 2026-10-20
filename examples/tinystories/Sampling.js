/** Example-owned seeded sampling. Engine execution still uses the generated API. */
export function createPrng(seed) {
  let state = (Math.floor(seed) >>> 0) || 1;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Return sampled token and the actual top-K distribution of this logits row. */
export function sampleToken(logits, row, vocabularySize, temperature, prng, topK = 40) {
  if (!Number.isInteger(row) || row < 0 || logits.length < (row + 1) * vocabularySize)
    throw new Error('Logits do not contain the selected token row.');
  if (!Number.isFinite(temperature) || temperature < 0 || !Number.isInteger(topK) || topK < 1)
    throw new Error('Temperature must be non-negative and top-K must be a positive integer.');
  const offset = row * vocabularySize, count = Math.min(topK, vocabularySize);
  const ids = new Int32Array(count), values = new Float32Array(count);
  values.fill(-Infinity);
  for (let id = 0; id < vocabularySize; id++) {
    const value = logits[offset + id];
    if (!Number.isFinite(value)) throw new Error(`Non-finite logit at token ${id}.`);
    if (value > values[count - 1]) {
      let index = count - 1;
      while (index > 0 && values[index - 1] < value) {
        values[index] = values[index - 1]; ids[index] = ids[index - 1]; index--;
      }
      values[index] = value; ids[index] = id;
    }
  }
  if (temperature <= .01) return {tokenId: ids[0], probability: 1,
    candidates: Array.from(ids, (tokenId, i) => ({tokenId, logit: values[i], probability: i === 0 ? 1 : 0}))};
  const probabilities = new Float32Array(count);
  let total = 0;
  for (let i = 0; i < count; i++) { probabilities[i] = Math.exp((values[i] - values[0]) / temperature); total += probabilities[i]; }
  const draw = prng() * total;
  let sum = 0, selected = 0;
  for (let i = 0; i < count; i++) { sum += probabilities[i]; if (draw <= sum) { selected = i; break; } }
  return {tokenId: ids[selected], probability: probabilities[selected] / total,
    candidates: Array.from(ids, (tokenId, i) => ({tokenId, logit: values[i], probability: probabilities[i] / total}))};
}
