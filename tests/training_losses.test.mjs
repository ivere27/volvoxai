import test from 'node:test';
import assert from 'node:assert/strict';

import {
  addGradient,
  crossEntropyGradient,
  normalizeCrossEntropyLosses,
} from '../ts/training/TrainingLosses.js';

test('weighted cross-entropy losses retain independent normalization', () => {
  const descriptors = normalizeCrossEntropyLosses({
    losses: [
      { name: 'tokens', logitsTensor: 'lm', targets: [0, 1], weight: 1 },
      { name: 'router', logitsTensor: 'route', targets: [1], weight: 0.1 },
    ],
  });
  const token = crossEntropyGradient(
    { name: 'lm', dtype: 'float32', shape: [1, 2, 3] },
    new Float32Array(6),
    descriptors[0],
  );
  const router = crossEntropyGradient(
    { name: 'route', dtype: 'float32', shape: [1, 2] },
    new Float32Array(2),
    descriptors[1],
  );

  assert.ok(Math.abs(token.loss - Math.log(3)) < 1e-7);
  assert.ok(Math.abs(router.loss - 0.1 * Math.log(2)) < 1e-7);
  assert.deepEqual([...token.gradient], [
    -1 / 3, 1 / 6, 1 / 6,
    1 / 6, -1 / 3, 1 / 6,
  ].map(Math.fround));
  assert.deepEqual([...router.gradient], [Math.fround(0.05), Math.fround(-0.05)]);
});

test('explicit loss normalizers make microbatch gradient sums exact', () => {
  const [first] = normalizeCrossEntropyLosses({
    losses: [{ name: 'tokens', logitsTensor: 'lm', targets: [0], normalizer: 2 }],
  });
  const [second] = normalizeCrossEntropyLosses({
    losses: [{ name: 'tokens', logitsTensor: 'lm', targets: [1], normalizer: 2 }],
  });
  const logits = { name: 'lm', dtype: 'float32', shape: [1, 2] };
  const left = crossEntropyGradient(logits, new Float32Array(2), first);
  const right = crossEntropyGradient(logits, new Float32Array(2), second);
  const sum = addGradient(left.gradient, right.gradient);

  assert.ok(Math.abs(left.loss + right.loss - Math.log(2)) < 1e-7);
  assert.deepEqual([...sum], [0, 0]);
});

test('loss descriptors reject ambiguous and malformed requests', () => {
  assert.throws(() => normalizeCrossEntropyLosses({
    targets: [0],
    losses: [{ logitsTensor: 'x', targets: [0] }],
  }), /either losses or the single-loss target fields/);
  assert.throws(() => normalizeCrossEntropyLosses({ losses: [] }), /must not be empty/);
  assert.throws(() => normalizeCrossEntropyLosses({
    losses: [{ name: 'x', logitsTensor: 'a', targets: [0] }, { name: 'x', logitsTensor: 'b', targets: [0] }],
  }), /unique/);
  assert.throws(() => normalizeCrossEntropyLosses({
    losses: [{ logitsTensor: 'a', targets: [0], normalizer: 0 }],
  }), /positive/);
});
