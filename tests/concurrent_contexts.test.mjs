import test from 'node:test';
import assert from 'node:assert/strict';

import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';
import { CPUShapeExecutionContext } from '../ts/core/CPUShapeExecutionContext.js';

/*
 * One runtime, one model revision, several inferences in flight.
 *
 * This is the shape a server actually runs, and it is the shape that made the
 * per-context weight copy a P0 rather than an inefficiency: VRAM and heap are
 * the ceiling on how many requests can be resident, so a weight copy per
 * context lowers the batch size that concurrency was opened to raise.
 *
 * What is checked here is both halves of that claim — the weights are paid for
 * once, and sharing them did not make the answers wrong.
 */

function makeSnapshot(document, definitions = []) {
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  const weights = Object.fromEntries(definitions.map((definition) => [
    definition.name,
    {
      name: definition.name,
      dtype: definition.dtype,
      shape: [...definition.shape],
      data: Float32Array.from(definition.values),
    },
  ]));
  return Model.capture({ graph, weights });
}

/** Linear -> ReLU over a weight big enough that duplication would show. */
function sharedModel({ width = 64 } = {}) {
  const values = Array.from(
    { length: width * width },
    (_, index) => Math.fround(((index % 13) - 6) * 0.125),
  );
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1, width] } },
    nodes: [
      {
        id: 'dense',
        opType: 'Linear',
        inputs: { input: 'x', weight: 'w' },
        outputs: { out: { tensor: 'h', dtype: 'float32', shape: [1, width] } },
        params: {},
      },
      {
        id: 'act',
        opType: 'ReLU',
        inputs: { input: 'h' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, width] } },
        params: {},
      },
    ],
    outputs: ['y'],
  }, [{ name: 'w', dtype: 'float32', shape: [width, width], values }]);
}

function requestRow(width, seed) {
  return Float32Array.from(
    { length: width }, (_, index) => Math.fround(((index + seed) % 7) - 3),
  );
}

test('several contexts over one revision infer concurrently and pay for the weights once', async () => {
  const width = 64;
  const requests = 6;

  /* The oracle runs over its own revision. Sharing is keyed by revision, so an
   * oracle over the same one would pre-warm the store and the contexts under
   * test would inherit rather than establish the sharing. */
  const oracleContext = new CPUShapeExecutionContext(sharedModel({ width }));
  const oracle = [];
  for (let index = 0; index < requests; index++) {
    const result = await oracleContext.execute(
      { x: { data: requestRow(width, index), shape: [1, width] } });
    oracle.push(new Float32Array(result.get('y').data));
  }
  await oracleContext.close();

  // N contexts over one revision, all in flight at once.
  const snapshot = sharedModel({ width });
  const contexts = Array.from({ length: requests },
    () => new CPUShapeExecutionContext(snapshot));
  const produced = await Promise.all(contexts.map((context, index) =>
    context.execute({ x: { data: requestRow(width, index), shape: [1, width] } })
      .then((result) => new Float32Array(result.get('y').data))));

  for (const [index, row] of produced.entries()) {
    assert.deepEqual([...row], [...oracle[index]],
      `request ${index} must answer the same in flight as it does alone`);
  }

  const reports = contexts.map((context) => context.inspect().invariantWeights);
  const weightBytes = reports[0].sizeBytes;
  assert.equal(weightBytes, width * width * 4);
  const allocated = reports.reduce(
    (total, report) => total + (report.sizeBytes - report.sharedSizeBytes), 0);
  assert.equal(allocated, weightBytes,
    `${requests} concurrent contexts must allocate the weights once, ` +
    `not ${requests * weightBytes} bytes`);
  assert.equal(reports.filter((report) => report.sharedSizeBytes === 0).length, 1,
    'exactly one context materializes and the rest borrow');

  for (const context of contexts) await context.close();
});

test('a context closing mid-flight does not disturb its siblings', async () => {
  const width = 32;
  const snapshot = sharedModel({ width });

  const survivors = [new CPUShapeExecutionContext(snapshot), new CPUShapeExecutionContext(snapshot)];
  const transient = new CPUShapeExecutionContext(snapshot);

  const before = await Promise.all(survivors.map((context, index) =>
    context.execute({ x: { data: requestRow(width, index), shape: [1, width] } })
      .then((result) => new Float32Array(result.get('y').data))));

  await transient.execute({ x: { data: requestRow(width, 9), shape: [1, width] } });
  await transient.close();

  /* The transient context borrowed the weights; closing it must not take them
   * from the contexts still running. */
  const after = await Promise.all(survivors.map((context, index) =>
    context.execute({ x: { data: requestRow(width, index), shape: [1, width] } })
      .then((result) => new Float32Array(result.get('y').data))));

  for (const [index, row] of after.entries()) {
    assert.deepEqual([...row], [...before[index]]);
  }
  for (const context of survivors) await context.close();
});

test('interleaved requests across contexts keep their own answers', async () => {
  const width = 16;
  const snapshot = sharedModel({ width });
  const contexts = [
    new CPUShapeExecutionContext(snapshot),
    new CPUShapeExecutionContext(snapshot),
  ];

  /* Alternate between two contexts without awaiting in order, so a context
   * that leaked state into a shared buffer would answer with its neighbour's
   * row rather than its own. */
  const pending = [];
  for (let round = 0; round < 8; round++) {
    const context = contexts[round % 2];
    const seed = round % 2 === 0 ? 1 : 2;
    pending.push(context
      .execute({ x: { data: requestRow(width, seed), shape: [1, width] } })
      .then((result) => ({ seed, row: new Float32Array(result.get('y').data) })));
  }
  const answers = await Promise.all(pending);

  const bySeed = new Map();
  for (const { seed, row } of answers) {
    const previous = bySeed.get(seed);
    if (previous === undefined) bySeed.set(seed, row);
    else assert.deepEqual([...row], [...previous], `seed ${seed} answered two ways`);
  }
  assert.equal(bySeed.size, 2);
  assert.notDeepEqual([...bySeed.get(1)], [...bySeed.get(2)],
    'the fixture must actually distinguish the two inputs');

  for (const context of contexts) await context.close();
});
