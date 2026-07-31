import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SIGNATURE_SCHEMA,
  SIGNATURE_VERSION,
  compareSignature,
  signature,
} from './parity/lib/extract.mjs';

const EXACT = { rtol: 0, atol: 0 };

function clone(value) {
  return structuredClone(value);
}

test('parity signature rejects empty, non-finite, and structurally inconsistent input', () => {
  assert.throws(() => signature([]), /must not be empty/);
  assert.throws(() => signature([1, Number.NaN]), /non-finite.*index 1/);
  assert.throws(() => signature([1, Number.POSITIVE_INFINITY]), /non-finite.*index 1/);
  assert.throws(() => signature([1, 2, 3], { shape: [2, 2] }), /has 4 elements, expected 3/);
  assert.throws(() => signature([1, 2], { shape: [2], sampleAxes: [0, 0] }), /duplicate axis 0/);
});

test('parity signature v2 records shape, explicit sample indices, and finite top-k structure', () => {
  const sig = signature(Float32Array.from([4, 1, 3, 2]), { shape: [2, 2], sampleAxes: [0, 1], topk: 2 });
  assert.equal(sig.schema, SIGNATURE_SCHEMA);
  assert.equal(sig.version, SIGNATURE_VERSION);
  assert.deepEqual(sig.shape, [2, 2]);
  assert.deepEqual(sig.sample.indices, [0, 1, 2, 3]);
  assert.equal(sig.sample.count, 4);
  assert.deepEqual(sig.topk, {
    requested: 2,
    count: 2,
    entries: [{ i: 0, v: 4 }, { i: 2, v: 3 }],
  });
});

test('parity comparator rejects noncanonical, truncated, reshaped, and non-finite signatures', () => {
  const reference = signature(Float32Array.from({ length: 1024 }, (_, i) => i / 10), { shape: [32, 32], topk: 3 });

  const noncanonical = { n: reference.n, stats: reference.stats, sub: { stride: 2, values: [] }, topk: [] };
  let result = compareSignature(noncanonical, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /regenerate parity goldens and artifacts/);

  const truncated = clone(reference);
  truncated.n--;
  result = compareSignature(truncated, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /shape.*elements|element count/);

  const reshaped = signature(Float32Array.from({ length: 1024 }, (_, i) => i / 10), { shape: [16, 64], topk: 3 });
  result = compareSignature(reshaped, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /shape \[16,64\] vs \[32,32\]/);

  const shortSample = clone(reference);
  shortSample.sample.values.pop();
  result = compareSignature(shortSample, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /sample\.values length/);

  const shortTopk = clone(reference);
  shortTopk.topk.entries.pop();
  result = compareSignature(shortTopk, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /topk\.entries length/);

  const changedIndices = clone(reference);
  [changedIndices.sample.indices[0], changedIndices.sample.indices[1]] =
    [changedIndices.sample.indices[1], changedIndices.sample.indices[0]];
  result = compareSignature(changedIndices, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /sampling strategy/);

  const nonFinite = clone(reference);
  nonFinite.topk.entries[0].v = Number.POSITIVE_INFINITY;
  result = compareSignature(nonFinite, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /topk\.entries\[0\]\.v must be finite/);

  const nonFiniteSample = clone(reference);
  nonFiniteSample.sample.values[0] = Number.NaN;
  result = compareSignature(nonFiniteSample, reference, EXACT);
  assert.equal(result.pass, false);
  assert.match(result.problems.join('\n'), /sample\.values\[0\] must be finite/);
});

test('shape-aware detector samples cover every class and box-coordinate axis value', () => {
  const anchors = 1024;
  const classes = 90;
  const scores = signature(new Float32Array(anchors * classes), {
    topk: 0,
    shape: [1, anchors, classes],
    sampleAxes: [1, 2],
  });
  const coveredClasses = new Set(scores.sample.indices.map((index) => index % classes));
  assert.equal(coveredClasses.size, classes);

  const boxWidth = 4;
  const boxes = signature(new Float32Array(anchors * boxWidth), {
    topk: 0,
    shape: [1, anchors, boxWidth],
    sampleAxes: [1, 2],
  });
  const coveredCoordinates = new Set(boxes.sample.indices.map((index) => index % boxWidth));
  assert.deepEqual([...coveredCoordinates].sort((a, b) => a - b), [0, 1, 2, 3]);
});

test('shape-aware sampling detects changes confined to detector residues', () => {
  const anchors = 1024;
  const width = 4;
  const baseline = new Float32Array(anchors * width);
  const changed = baseline.slice();
  for (let i = 0; i < changed.length; i++) {
    if (i % width === 1 || i % width === 3) changed[i] = 100;
  }
  const options = { topk: 0, shape: [1, anchors, width], sampleAxes: [1, 2] };
  const result = compareSignature(signature(changed, options), signature(baseline, options), {
    rtol: 0,
    atol: 0,
    statsGate: false,
  });
  assert.equal(result.pass, false);
  assert.ok(result.numeric.sampleExceed > 0);
  assert.ok(Number.isFinite(result.numeric.maxRel));
});
