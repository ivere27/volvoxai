function fail(message) {
  throw new Error(`[TinyStories] ${message}`);
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, axis) => dimension === right[axis]);
}

function legalDimensionBounds(descriptor, label) {
  const minimum = descriptor?.min;
  const maximum = descriptor?.max;
  const multipleOf = descriptor?.multiple_of ?? 1;
  if (!Number.isSafeInteger(minimum) || minimum <= 0 ||
      !Number.isSafeInteger(maximum) || maximum < minimum ||
      !Number.isSafeInteger(multipleOf) || multipleOf <= 0) {
    fail(`${label} must have finite positive min, max, and multiple_of bounds.`);
  }
  const minimumLegal = Math.ceil(minimum / multipleOf) * multipleOf;
  const maximumLegal = Math.floor(maximum / multipleOf) * multipleOf;
  if (minimumLegal > maximumLegal) fail(`${label} has no legal concrete extent.`);
  return Object.freeze({ minimumLegal, maximumLegal, multipleOf });
}

/** Validate the model-family ABI and recover its bounded sequence contract. */
export function resolveTinyStoriesSequenceContract(snapshot) {
  const graph = snapshot?.graph;
  const tokens = graph?.inputs?.tokens;
  const positions = graph?.inputs?.positions;
  if (!graph || tokens?.dtype !== 'int32' || positions?.dtype !== 'int32' ||
      !Array.isArray(tokens.shape) || tokens.shape.length !== 2 || tokens.shape[0] !== 1 ||
      typeof tokens.shape[1] !== 'string' || !sameShape(tokens.shape, positions.shape)) {
    fail('tokens and positions must be I32 [1,S] inputs sharing one bounded symbol.');
  }

  const sequenceSymbol = tokens.shape[1];
  const bounds = legalDimensionBounds(
    graph.dimensions?.[sequenceSymbol],
    `sequence dimension '${sequenceSymbol}'`,
  );
  const outputName = snapshot.outputNames?.at(-1);
  const logits = typeof outputName === 'string' ? graph.tensors?.[outputName] : null;
  if (!logits || logits.dtype !== 'float32' ||
      !Array.isArray(logits.shape) || logits.shape.length !== 3 ||
      logits.shape[0] !== 1 || logits.shape[1] !== sequenceSymbol ||
      !Number.isSafeInteger(logits.shape[2]) || logits.shape[2] <= 0) {
    fail('the final output must be F32 logits shaped [1,S,vocabulary].');
  }

  return Object.freeze({
    outputName,
    sequenceSymbol,
    minimumSequenceCapacity: bounds.minimumLegal,
    maximumSequenceCapacity: bounds.maximumLegal,
    sequenceMultipleOf: bounds.multipleOf,
    vocabularySize: logits.shape[2],
  });
}

function shapedSequence(data, capacity) {
  return Object.freeze({
    data,
    shape: Object.freeze([1, capacity]),
  });
}

/**
 * Allocate only the capacity required by this prompt and requested decode.
 * The typed storage remains mutable so the host can append one generated token
 * before each decode step; public logical shape always stays explicit.
 */
export function createTinyStoriesDecodeState(
  promptTokenIds,
  { requestedNewTokens, eosTokenId, contract },
) {
  if (!Array.isArray(promptTokenIds) && !(promptTokenIds instanceof Int32Array)) {
    fail('prompt token IDs must be an array or Int32Array.');
  }
  const prompt = Array.from(promptTokenIds);
  if (prompt.length === 0) fail('the prompt must encode to at least one token.');
  if (!Number.isSafeInteger(requestedNewTokens) || requestedNewTokens < 0) {
    fail('the requested generation length must be a non-negative safe integer.');
  }
  if (!contract || !Number.isSafeInteger(contract.minimumSequenceCapacity) ||
      !Number.isSafeInteger(contract.maximumSequenceCapacity) ||
      !Number.isSafeInteger(contract.sequenceMultipleOf) ||
      !Number.isSafeInteger(contract.vocabularySize)) {
    fail('a validated TinyStories sequence contract is required.');
  }
  if (!Number.isSafeInteger(eosTokenId) || eosTokenId < 0 ||
      eosTokenId >= contract.vocabularySize) {
    fail('the EOS token ID is outside the model vocabulary.');
  }
  if (prompt.some((token) => !Number.isSafeInteger(token) || token < 0 ||
      token >= contract.vocabularySize)) {
    fail('the prompt contains an out-of-vocabulary token ID.');
  }
  if (prompt.length > contract.maximumSequenceCapacity) {
    fail(`the prompt exceeds maximum sequence capacity ${contract.maximumSequenceCapacity}.`);
  }

  const generationLimit = Math.min(
    requestedNewTokens,
    contract.maximumSequenceCapacity - prompt.length,
  );
  const requiredCapacity = Math.max(
    contract.minimumSequenceCapacity,
    prompt.length + generationLimit,
  );
  const sequenceCapacity = Math.ceil(
    requiredCapacity / contract.sequenceMultipleOf,
  ) * contract.sequenceMultipleOf;
  if (sequenceCapacity > contract.maximumSequenceCapacity) {
    fail('the requested prompt and generation do not fit one legal sequence capacity.');
  }

  const tokens = new Int32Array(sequenceCapacity);
  tokens.fill(eosTokenId);
  tokens.set(prompt);
  const positions = Int32Array.from(
    { length: sequenceCapacity },
    (_, position) => position,
  );
  const inputs = Object.freeze({
    tokens: shapedSequence(tokens, sequenceCapacity),
    positions: shapedSequence(positions, sequenceCapacity),
  });
  return Object.freeze({
    tokens,
    positions,
    inputs,
    promptLength: prompt.length,
    generationLimit,
    sequenceCapacity,
  });
}
