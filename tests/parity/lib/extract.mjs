// Turn a large output tensor into a small, committable signature and compare
// two signatures with numeric tolerance + task-level metrics.
//
// The schema records tensor structure and every sampled flat index, so
// truncated or differently-shaped outputs cannot compare successfully.

export const SIGNATURE_SCHEMA = 'volvoxai.parity.signature';
export const SIGNATURE_VERSION = 2;

const SAMPLE_MAX = 512;
const SAMPLE_STRATEGY = 'multiscale-axis-v1';

// flat: a finite, non-empty flat numeric array. `shape` defaults to the flat
// shape [flat.length]; callers that know the logical tensor shape should pass
// it. `sampleAxes` names axes whose coordinates need explicit coverage (for
// example [0, 1] for detector [anchors, classes/box-coordinates] outputs).
export function signature(flat, { topk = 10, shape, sampleAxes = [] } = {}) {
  if (flat == null || !Number.isSafeInteger(flat.length)) {
    throw new TypeError('signature input must be an array-like numeric value');
  }
  const n = flat.length;
  if (n === 0) throw new RangeError('signature input must not be empty');

  const normalizedShape = normalizeShape(shape ?? [n], n);
  const normalizedAxes = normalizeAxes(sampleAxes, normalizedShape.length);
  if (!Number.isSafeInteger(topk) || topk < 0) {
    throw new RangeError(`signature topk must be a non-negative safe integer, got ${topk}`);
  }

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumsq = 0;
  for (let i = 0; i < n; i++) {
    const v = flat[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw new TypeError(`signature input contains a non-finite number at flat index ${i}`);
    }
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumsq += v * v;
  }
  if (!Number.isFinite(sum) || !Number.isFinite(sumsq)) {
    throw new RangeError('signature statistics overflowed; input magnitude is too large');
  }

  const indices = sampleIndices(normalizedShape, normalizedAxes);
  const values = indices.map((i) => round(flat[i]));

  // top-k by value (id = flat index; decode to coordinates is the caller's
  // business). Array.prototype.sort is stable, so equal values prefer the
  // lower flat index on every supported JS runtime.
  const topCount = Math.min(topk, n);
  const entries = topCount === 0 ? [] : Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => flat[b] - flat[a])
    .slice(0, topCount)
    .map((i) => ({ i, v: round(flat[i]) }));

  return {
    schema: SIGNATURE_SCHEMA,
    version: SIGNATURE_VERSION,
    n,
    shape: normalizedShape,
    stats: {
      min: round(min),
      max: round(max),
      mean: round(sum / n),
      sumsq: round(sumsq),
    },
    sample: {
      strategy: SAMPLE_STRATEGY,
      axes: normalizedAxes,
      count: indices.length,
      indices,
      values,
    },
    topk: { requested: topk, count: entries.length, entries },
  };
}

function normalizeShape(shape, n) {
  if (!Array.isArray(shape)) throw new TypeError('signature shape must be an array');
  if (shape.length === 0 && n !== 1) {
    throw new RangeError(`scalar signature shape [] requires exactly 1 element, got ${n}`);
  }
  let elements = 1;
  for (let axis = 0; axis < shape.length; axis++) {
    const dimension = shape[axis];
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      throw new RangeError(`signature shape[${axis}] must be a positive safe integer, got ${dimension}`);
    }
    if (elements > Math.floor(Number.MAX_SAFE_INTEGER / dimension)) {
      throw new RangeError('signature shape element count exceeds Number.MAX_SAFE_INTEGER');
    }
    elements *= dimension;
  }
  if (elements !== n) {
    throw new RangeError(`signature shape [${shape.join(',')}] has ${elements} elements, expected ${n}`);
  }
  return [...shape];
}

function normalizeAxes(sampleAxes, rank) {
  if (!Array.isArray(sampleAxes)) throw new TypeError('signature sampleAxes must be an array');
  const seen = new Set();
  return sampleAxes.map((axis, i) => {
    if (!Number.isSafeInteger(axis) || axis < 0 || axis >= rank) {
      throw new RangeError(`signature sampleAxes[${i}] must be in [0, ${rank}), got ${axis}`);
    }
    if (seen.has(axis)) throw new RangeError(`signature sampleAxes contains duplicate axis ${axis}`);
    seen.add(axis);
    return axis;
  });
}

// Breadth-first interval midpoints form a deterministic multiscale ordering:
// center, quarters, eighths, ... . Unlike a fixed stride this has no permanent
// residue-class blind spot and remains well distributed when only a prefix is
// retained.
function intervalMidpoints(length, limit) {
  const result = [];
  const queue = [[0, length]];
  let head = 0;
  while (result.length < limit && head < queue.length) {
    const [lo, hi] = queue[head++];
    const mid = Math.floor((lo + hi) / 2);
    result.push(mid);
    if (lo < mid) queue.push([lo, mid]);
    if (mid + 1 < hi) queue.push([mid + 1, hi]);
  }
  return result;
}

function flattenIndex(coords, shape) {
  let index = 0;
  for (let axis = 0; axis < shape.length; axis++) index = index * shape[axis] + coords[axis];
  return index;
}

function sampleIndices(shape, axes) {
  const n = shape.reduce((product, dimension) => product * dimension, 1);
  const count = Math.min(n, SAMPLE_MAX);
  if (n <= SAMPLE_MAX) return Array.from({ length: n }, (_, i) => i);

  const axisSequences = shape.map((dimension) => intervalMidpoints(dimension, Math.min(dimension, count)));
  const selected = new Set();

  // Cover requested axes round-robin. Every coordinate of a detector class or
  // box-coordinate axis is therefore represented when that dimension fits in
  // the sample budget, while large anchor axes receive multiscale coverage.
  const cursors = axes.map(() => 0);
  let active = axes.length;
  while (selected.size < count && active > 0) {
    active = 0;
    for (let slot = 0; slot < axes.length && selected.size < count; slot++) {
      const axis = axes[slot];
      const sequence = axisSequences[axis];
      const cursor = cursors[slot];
      if (cursor >= sequence.length) continue;
      active++;
      cursors[slot]++;
      const coords = shape.map((_, otherAxis) => {
        if (otherAxis === axis) return sequence[cursor];
        const other = axisSequences[otherAxis];
        return other[(cursor + axis * 37 + otherAxis * 17) % other.length];
      });
      selected.add(flattenIndex(coords, shape));
    }
  }

  // Fill any remaining budget from the tensor's own multiscale ordering. The
  // first `count` interval midpoints are unique, so the union always reaches
  // the requested count even when axis-derived points overlap.
  for (const index of intervalMidpoints(n, count)) {
    if (selected.size >= count) break;
    selected.add(index);
  }
  return [...selected];
}

function round(x) {
  // 6 significant figures keeps goldens small and stable across platforms.
  const value = Number.parseFloat(x.toPrecision(6));
  return Object.is(value, -0) ? 0 : value;
}

function close(a, b, rtol, atol) {
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

function arraysEqual(a, b) {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}

function validationProblems(sig, label) {
  const problems = [];
  const add = (message) => problems.push(`${label} ${message}`);
  if (sig == null || typeof sig !== 'object' || Array.isArray(sig)) {
    add('signature must be an object');
    return problems;
  }
  if (sig.schema !== SIGNATURE_SCHEMA || sig.version !== SIGNATURE_VERSION) {
    add(`uses an unsupported signature schema; regenerate parity goldens and artifacts (expected ${SIGNATURE_SCHEMA} v${SIGNATURE_VERSION})`);
    return problems;
  }
  if (!Number.isSafeInteger(sig.n) || sig.n <= 0) add(`n must be a positive safe integer, got ${sig.n}`);

  let shapeValid = Number.isSafeInteger(sig.n) && sig.n > 0;
  if (shapeValid) {
    try {
      normalizeShape(sig.shape, sig.n);
    } catch (error) {
      add(error.message);
      shapeValid = false;
    }
  }

  if (sig.stats == null || typeof sig.stats !== 'object') {
    add('stats must be an object');
  } else {
    for (const key of ['min', 'max', 'mean', 'sumsq']) {
      if (!Number.isFinite(sig.stats[key])) add(`stats.${key} must be finite`);
    }
  }

  let axesValid = shapeValid;
  if (sig.sample == null || typeof sig.sample !== 'object') {
    add('sample must be an object');
    axesValid = false;
  } else {
    if (sig.sample.strategy !== SAMPLE_STRATEGY) add(`sample.strategy must be ${SAMPLE_STRATEGY}`);
    if (axesValid) {
      try {
        normalizeAxes(sig.sample.axes, sig.shape.length);
      } catch (error) {
        add(error.message);
        axesValid = false;
      }
    }
    const expectedCount = Number.isSafeInteger(sig.n) && sig.n > 0 ? Math.min(sig.n, SAMPLE_MAX) : null;
    if (!Number.isSafeInteger(sig.sample.count) || sig.sample.count !== expectedCount) {
      add(`sample.count must be ${expectedCount}, got ${sig.sample.count}`);
    }
    if (!Array.isArray(sig.sample.indices)) {
      add('sample.indices must be an array');
    } else {
      if (sig.sample.indices.length !== sig.sample.count) {
        add(`sample.indices length ${sig.sample.indices.length} does not match sample.count ${sig.sample.count}`);
      }
      const seen = new Set();
      for (let i = 0; i < sig.sample.indices.length; i++) {
        const index = sig.sample.indices[i];
        if (!Number.isSafeInteger(index) || index < 0 || index >= sig.n) add(`sample.indices[${i}] is out of range`);
        if (seen.has(index)) add(`sample.indices contains duplicate flat index ${index}`);
        seen.add(index);
      }
      if (shapeValid && axesValid) {
        const expected = sampleIndices(sig.shape, sig.sample.axes);
        if (!arraysEqual(sig.sample.indices, expected)) add('sample.indices do not match the declared shape/axes sampling strategy');
      }
    }
    if (!Array.isArray(sig.sample.values)) {
      add('sample.values must be an array');
    } else {
      if (sig.sample.values.length !== sig.sample.count) {
        add(`sample.values length ${sig.sample.values.length} does not match sample.count ${sig.sample.count}`);
      }
      for (let i = 0; i < sig.sample.values.length; i++) {
        if (!Number.isFinite(sig.sample.values[i])) add(`sample.values[${i}] must be finite`);
      }
    }
  }

  if (sig.topk == null || typeof sig.topk !== 'object') {
    add('topk must be an object');
  } else {
    if (!Number.isSafeInteger(sig.topk.requested) || sig.topk.requested < 0) {
      add(`topk.requested must be a non-negative safe integer, got ${sig.topk.requested}`);
    }
    const expectedTopCount = Number.isSafeInteger(sig.topk.requested) && sig.topk.requested >= 0
      && Number.isSafeInteger(sig.n) && sig.n > 0 ? Math.min(sig.topk.requested, sig.n) : null;
    if (!Number.isSafeInteger(sig.topk.count) || sig.topk.count !== expectedTopCount) {
      add(`topk.count must be ${expectedTopCount}, got ${sig.topk.count}`);
    }
    if (!Array.isArray(sig.topk.entries)) {
      add('topk.entries must be an array');
    } else {
      if (sig.topk.entries.length !== sig.topk.count) {
        add(`topk.entries length ${sig.topk.entries.length} does not match topk.count ${sig.topk.count}`);
      }
      const seen = new Set();
      for (let rank = 0; rank < sig.topk.entries.length; rank++) {
        const entry = sig.topk.entries[rank];
        if (entry == null || typeof entry !== 'object') {
          add(`topk.entries[${rank}] must be an object`);
          continue;
        }
        if (!Number.isSafeInteger(entry.i) || entry.i < 0 || entry.i >= sig.n) {
          add(`topk.entries[${rank}].i is out of range`);
        }
        if (seen.has(entry.i)) add(`topk.entries contains duplicate flat index ${entry.i}`);
        seen.add(entry.i);
        if (!Number.isFinite(entry.v)) add(`topk.entries[${rank}].v must be finite`);
        if (rank > 0 && Number.isFinite(entry.v) && Number.isFinite(sig.topk.entries[rank - 1]?.v)
          && entry.v > sig.topk.entries[rank - 1].v) {
          add('topk.entries values must be in non-increasing order');
        }
      }
    }
  }
  return problems;
}

function failedComparison(problems, notes = []) {
  return {
    pass: false,
    numeric: { maxAbs: 0, maxRel: 0, sampleExceed: 0 },
    task: { top1Match: null, topkOverlap: null },
    problems,
    notes,
  };
}

// Compare candidate signature vs golden. `pass` folds gating checks only.
//
// Gating (always): exact structural compatibility, sampled values within
// tolerance, and top-1 task match when a top-k is present. Global energy stats
// gate only when `statsGate` is true; otherwise they remain diagnostics.
export function compareSignature(cand, gold, tol) {
  const problems = [
    ...validationProblems(cand, 'candidate'),
    ...validationProblems(gold, 'golden'),
  ];
  if (problems.length > 0) return failedComparison(problems);

  if (cand.n !== gold.n) problems.push(`element count ${cand.n} vs ${gold.n}`);
  if (!arraysEqual(cand.shape, gold.shape)) problems.push(`shape [${cand.shape}] vs [${gold.shape}]`);
  if (!arraysEqual(cand.sample.axes, gold.sample.axes)) {
    problems.push(`sample axes [${cand.sample.axes}] vs [${gold.sample.axes}]`);
  }
  if (cand.sample.count !== gold.sample.count) {
    problems.push(`sample count ${cand.sample.count} vs ${gold.sample.count}`);
  }
  if (!arraysEqual(cand.sample.indices, gold.sample.indices)) problems.push('sample indices differ');
  if (cand.topk.requested !== gold.topk.requested || cand.topk.count !== gold.topk.count) {
    problems.push(`top-k structure ${cand.topk.count}/${cand.topk.requested} vs ${gold.topk.count}/${gold.topk.requested}`);
  }
  if (problems.length > 0) return failedComparison(problems);

  const { rtol, atol, statsGate = true } = tol ?? {};
  if (!Number.isFinite(rtol) || rtol < 0 || !Number.isFinite(atol) || atol < 0) {
    return failedComparison([`invalid comparison tolerance rtol=${rtol}, atol=${atol}`]);
  }
  const notes = [];
  const statBucket = statsGate ? problems : notes;

  for (const key of ['min', 'max', 'mean']) {
    if (!close(cand.stats[key], gold.stats[key], rtol, atol)) {
      statBucket.push(`stats.${key} ${cand.stats[key]} vs ${gold.stats[key]}`);
    }
  }
  if (!close(cand.stats.sumsq, gold.stats.sumsq, Math.max(rtol, 5e-3), atol)) {
    statBucket.push(`stats.sumsq ${cand.stats.sumsq} vs ${gold.stats.sumsq}`);
  }

  let maxAbs = 0;
  let maxRel = 0;
  let exceed = 0;
  const count = cand.sample.count;
  for (let i = 0; i < count; i++) {
    const a = cand.sample.values[i];
    const b = gold.sample.values[i];
    const abs = Math.abs(a - b);
    const denominator = Math.max(Math.abs(b), atol);
    const quotient = denominator === 0 ? Number.MAX_VALUE : abs / denominator;
    const rel = abs === 0 ? 0 : Math.min(quotient, Number.MAX_VALUE);
    if (abs > maxAbs) maxAbs = abs;
    if (rel > maxRel) maxRel = rel;
    if (!close(a, b, rtol, atol)) exceed++;
  }
  if (exceed > 0) {
    problems.push(`${exceed}/${count} sampled values exceed tol (maxAbs=${maxAbs.toExponential(2)})`);
  }

  let top1Match = null;
  let topkOverlap = null;
  if (gold.topk.count > 0) {
    top1Match = cand.topk.entries[0].i === gold.topk.entries[0].i;
    const candidateIds = new Set(cand.topk.entries.map((entry) => entry.i));
    topkOverlap = gold.topk.entries.filter((entry) => candidateIds.has(entry.i)).length / gold.topk.count;
    if (!top1Match) problems.push(`top-1 id ${cand.topk.entries[0].i} vs ${gold.topk.entries[0].i}`);
  }

  return {
    pass: problems.length === 0,
    numeric: { maxAbs: round(maxAbs), maxRel: round(maxRel), sampleExceed: exceed },
    task: { top1Match, topkOverlap },
    problems,
    notes,
  };
}
