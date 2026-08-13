import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BatchScheduler,
  SchedulerError,
  formatGroupKey,
  copySlice,
  cloneRuntimeArray,
} from '../ts/core/BatchScheduler.js';
import { PagedKVCache } from '../ts/core/PagedKVCache.js';

/*
 * BatchScheduler policy-core verification suite.
 * See ../docs/scheduling-and-dynamic-batching-design.md#internal-batch-scheduler-core.
 *
 * 1. Transparency: batched results bitwise identical to single execution for all kinds.
 * 2. Determinism: identical submission + manual step timing produces identical dispatches.
 * 3. Group separation: model, adapter, shape, and rows_per_lane isolate into distinct batches.
 * 4. Padding isolation: multiple_of padding rows are discarded and never leak to requests.
 * 5. Error containment, cancellation, and draining lifecycle.
 * 6. Backpressure: queue depth cap refuses extra submissions immediately.
 * 7. Telemetry: utilization = device_busy / wall, padding_waste = 1 - useful / dispatched.
 */

test('BatchScheduler transparency: stateless batched execution is bitwise identical to single execution', async () => {
  // Oracle: single execution function
  function singleForward(input) {
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) {
      output[i] = Math.fround(Math.sin(input[i]) * 2.5 + 0.1);
    }
    return output;
  }

  // Generate 4 distinct requests
  const inputs = [
    new Float32Array([1.0, 2.0, 3.0, 4.0]),
    new Float32Array([5.0, 6.0, 7.0, 8.0]),
    new Float32Array([9.0, 10.0, 11.0, 12.0]),
    new Float32Array([13.0, 14.0, 15.0, 16.0]),
  ];

  // Ground truth single-request outputs
  const oracleOutputs = inputs.map(singleForward);

  const scheduler = new BatchScheduler({
    maxLanes: 4,
    tokenBudgetPerDispatch: 1024,
  });

  const ids = inputs.map((data) =>
    scheduler.submitStateless({
      modelId: 'test-model',
      shapeSignatureMinusBatch: 'x:float32:[4]',
      rowsPerLane: 1,
      inputs: { x: data },
    })
  );

  /* One dispatch, one [B, 4] buffer, one pass over it. Calling the oracle once
   * per work item would prove only that the slices were wired up; the claim in
   * The transparency contract says batching is invisible in the *result*, so
   * the batched path has to actually be batched. */
  const runBatchedStep = (batch, metadata) => {
    assert.equal(metadata.kind, 'stateless');
    assert.equal(metadata.usefulCount, 4);
    assert.equal(metadata.groupKey.modelId, 'test-model');
    assert.equal(batch.length, metadata.paddedCount);

    const width = 4;
    const stacked = new Float32Array(batch.length * width);
    for (const [lane, work] of batch.entries()) stacked.set(work.inputs.x, lane * width);

    const computed = new Float32Array(stacked.length);
    for (let i = 0; i < stacked.length; i++) {
      computed[i] = Math.fround(Math.sin(stacked[i]) * 2.5 + 0.1);
    }

    return batch.map((_work, lane) => ({
      outputs: { out: computed.subarray(lane * width, (lane + 1) * width) },
    }));
  };

  await scheduler.step(runBatchedStep, 1000);

  assert.equal(scheduler.results.length, 4);
  for (let i = 0; i < 4; i++) {
    const res = scheduler.result(ids[i]);
    assert.ok(res);
    assert.equal(res.state, 'completed');
    assert.ok(res.outputs?.out);
    // Bitwise identical verification
    assert.deepEqual(res.outputs.out, oracleOutputs[i]);
    /* Per-request result ownership: each request holds its own copy, not a
     * window onto the batch buffer that the next dispatch overwrites. */
    assert.equal(res.outputs.out.buffer.byteLength, 4 * Float32Array.BYTES_PER_ELEMENT);
  }
});

/* The determinism contract asks that the same submission order and the same
 * manual step timing produce the same *dispatch composition*. It deliberately
 * does not ask that two runs take the same number of microseconds:
 * `deviceBusyMicros` and `wallMicros` are measurements of a machine, and
 * pinning them would only be possible by feeding the injected clock into the
 * telemetry — exactly the mixed-clock fiction that made `utilization` read
 * 1.0 for free. */
test('BatchScheduler determinism: fixed submission and manual step timestamps produce identical runs', async () => {
  async function runSimulation() {
    const cache = new PagedKVCache({
      lanes: 2,
      pageTokens: 2,
      laneTokenCapacity: 8,
      maxPages: 8,
      policy: 'paged',
    });
    const scheduler = new BatchScheduler({
      cache,
      maxLanes: 2,
      tokenBudgetPerDispatch: 16,
    });

    const id1 = scheduler.submitLLM({ promptTokens: 2, maxTokens: 2 });
    const id2 = scheduler.submitStateless({
      modelId: 'stateless-m',
      shapeSignatureMinusBatch: 'data:float32:[2]',
      inputs: { data: new Float32Array([1, 2]) },
    });

    // The composition of every dispatch, in the order they were issued.
    const composition = [];
    const runner = (batch, meta) => {
      composition.push({
        group: formatGroupKey(meta.groupKey),
        kind: meta.kind,
        useful: meta.usefulCount,
        padded: meta.paddedCount,
        rows: meta.totalRows,
        members: batch.map((w) => `${w.requestId}@${w.slot}:${w.position}+${w.tokens}`),
      });
      return batch.map((w) => ({
        value: w.position,
        outputs: w.inputs ? { out: cloneRuntimeArray(w.inputs.data) } : undefined,
      }));
    };

    await scheduler.step(runner, 1000);
    await scheduler.step(runner, 2000);
    await scheduler.step(runner, 3000);

    const telem = scheduler.telemetry();
    return {
      composition,
      results: scheduler.results.map((r) => ({ id: r.requestId, state: r.state, values: r.values })),
      deterministicTelemetry: {
        dispatches: telem.dispatches,
        rowsDispatched: telem.rowsDispatched,
        rowsUseful: telem.rowsUseful,
        steps: telem.steps,
        admitted: telem.admitted,
        completed: telem.completed,
        cancelled: telem.cancelled,
        failed: telem.failed,
        queueDepth: telem.queueDepth,
      },
    };
  }

  const runA = await runSimulation();
  const runB = await runSimulation();

  assert.deepEqual(runA, runB);
  assert.ok(runA.composition.length > 0, 'the run must actually have dispatched something');
});


test('BatchScheduler route separation: distinct models, adapters, shapes, and rows_per_lane are isolated', async () => {
  const scheduler = new BatchScheduler({
    maxLanes: 4,
    tokenBudgetPerDispatch: 2048,
  });

  // 4 requests belonging to 4 distinct groups
  const id1 = scheduler.submitStateless({
    modelId: 'model-A',
    adapterRevision: null,
    shapeSignatureMinusBatch: 'x:float32:[4]',
    rowsPerLane: 1,
    inputs: { x: new Float32Array([1, 2, 3, 4]) },
  });

  const id2 = scheduler.submitStateless({
    modelId: 'model-B',
    adapterRevision: null,
    shapeSignatureMinusBatch: 'x:float32:[4]',
    rowsPerLane: 1,
    inputs: { x: new Float32Array([5, 6, 7, 8]) },
  });

  const id3 = scheduler.submitStateless({
    modelId: 'model-A',
    adapterRevision: 'lora_v1',
    shapeSignatureMinusBatch: 'x:float32:[4]',
    rowsPerLane: 1,
    inputs: { x: new Float32Array([9, 10, 11, 12]) },
  });

  const id4 = scheduler.submitStateless({
    modelId: 'model-A',
    adapterRevision: null,
    shapeSignatureMinusBatch: 'x:float32:[8]',
    rowsPerLane: 2,
    inputs: { x: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]) },
  });

  const telemetryInitial = scheduler.telemetry();
  assert.equal(telemetryInitial.groupCount, 4);

  const dispatchedGroups = [];
  const runner = (batch, metadata) => {
    dispatchedGroups.push(formatGroupKey(metadata.groupKey));
    return batch.map((w) => ({ value: w.requestId }));
  };

  // Step 4 times to dispatch each group
  await scheduler.step(runner, 1000);
  await scheduler.step(runner, 2000);
  await scheduler.step(runner, 3000);
  await scheduler.step(runner, 4000);

  assert.equal(dispatchedGroups.length, 4);
  const uniqueDispatched = new Set(dispatchedGroups);
  assert.equal(uniqueDispatched.size, 4, 'All 4 groups must be dispatched separately');

  assert.equal(scheduler.results.length, 4);
  assert.ok(scheduler.results.every((r) => r.state === 'completed'));
});

test('BatchScheduler padding isolation: multiple_of padding rows are discarded and never returned', async () => {
  const scheduler = new BatchScheduler({
    maxLanes: 4,
    multipleOf: 4,
    tokenBudgetPerDispatch: 1024,
  });

  // Submit 3 requests (will be padded to 4)
  const reqInputs = [
    new Float32Array([10, 20]),
    new Float32Array([30, 40]),
    new Float32Array([50, 60]),
  ];

  for (const input of reqInputs) {
    scheduler.submitStateless({
      modelId: 'pad-test',
      shapeSignatureMinusBatch: 'x:float32:[2]',
      rowsPerLane: 1,
      inputs: { x: input },
    });
  }

  let observedMetadata = null;
  const runner = (batch, metadata) => {
    observedMetadata = metadata;
    assert.equal(metadata.usefulCount, 3);
    assert.equal(metadata.paddedCount, 4);
    assert.equal(metadata.usefulRows, 3);
    assert.equal(metadata.totalRows, 4);

    return batch.map((w) => ({
      outputs: { out: new Float32Array([w.inputs.x[0] * 2, w.inputs.x[1] * 2]) },
    }));
  };

  await scheduler.step(runner, 1000);

  assert.ok(observedMetadata);
  const results = scheduler.results;
  assert.equal(results.length, 3, 'Only 3 results published, padding row discarded');

  assert.deepEqual(results[0].outputs.out, new Float32Array([20, 40]));
  assert.deepEqual(results[1].outputs.out, new Float32Array([60, 80]));
  assert.deepEqual(results[2].outputs.out, new Float32Array([100, 120]));

  const telem = scheduler.telemetry();
  assert.equal(telem.rowsUseful, 3);
  assert.equal(telem.rowsDispatched, 4);
  assert.equal(telem.paddingWaste, 1 - 3 / 4); // 0.25
});

test('BatchScheduler error containment, cancellation, and drain lifecycle', async () => {
  const scheduler = new BatchScheduler({
    maxLanes: 4,
  });

  const id1 = scheduler.submitStateless({ inputs: { x: new Float32Array([1]) } });
  const id2 = scheduler.submitStateless({ inputs: { x: new Float32Array([2]) } });

  // Cancel queued id2
  const cancelOk = scheduler.cancel(id2);
  assert.equal(cancelOk, true);
  assert.equal(scheduler.state(id2), 'cancelled');
  assert.equal(scheduler.queueDepth, 1);

  // Dispatch failing batch
  const failingRunner = () => {
    throw new Error('Device out of memory');
  };

  await scheduler.step(failingRunner, 1000);
  assert.equal(scheduler.state(id1), 'failed');
  assert.equal(scheduler.result(id1)?.error?.message, 'Device out of memory');

  // Test drain close on admitted stateful request
  const cache = new PagedKVCache({
    lanes: 2,
    pageTokens: 2,
    laneTokenCapacity: 8,
    maxPages: 8,
    policy: 'paged',
  });
  const statefulSched = new BatchScheduler({ cache, maxLanes: 2 });
  const id3 = statefulSched.submitLLM({ promptTokens: 2, maxTokens: 1 });
  const id4 = statefulSched.submitLLM({ promptTokens: 2, maxTokens: 1 });
  // Step once to admit id3 and id4
  await statefulSched.step((batch) => batch.map((w) => ({ value: w.position })), 2000);
  assert.equal(statefulSched.state(id3), 'decoding');

  // Submit id5 which remains queued
  const id5 = statefulSched.submitLLM({ promptTokens: 2, maxTokens: 1 });
  assert.equal(statefulSched.state(id5), 'queued');

  // Close with drain
  await statefulSched.close((batch) => batch.map((w) => ({ value: w.position })), { drain: true }, 3000);
  assert.equal(statefulSched.state(id3), 'completed');
  assert.equal(statefulSched.state(id4), 'completed');
  assert.equal(statefulSched.state(id5), 'cancelled', 'Queued request is cancelled on draining close');
  assert.equal(statefulSched.closed, true);
});

test('BatchScheduler bounded backpressure: queue depth limit refuses submissions immediately', () => {
  const scheduler = new BatchScheduler({
    maxQueueDepth: 2,
  });

  const id1 = scheduler.submitStateless({ inputs: { x: new Float32Array([1]) } });
  const id2 = scheduler.submitStateless({ inputs: { x: new Float32Array([2]) } });
  assert.equal(scheduler.queueDepth, 2);

  assert.throws(() => {
    scheduler.submitStateless({ inputs: { x: new Float32Array([3]) } });
  }, (err) => err instanceof SchedulerError && err.code === 'QUEUE_FULL');

  assert.equal(scheduler.queueDepth, 2);
});

test('BatchScheduler telemetry: utilization and token budget metrics', async () => {
  const scheduler = new BatchScheduler({
    tokenBudgetPerDispatch: 4,
    maxLanes: 8,
  });

  // Submit 6 requests with rowsPerLane = 1
  for (let i = 0; i < 6; i++) {
    scheduler.submitStateless({
      modelId: 'budget-test',
      rowsPerLane: 1,
      inputs: { x: new Float32Array([i]) },
    });
  }

  // Token budget is 4, so first step will dispatch 4 items, second step will dispatch 2 items
  const runner = (batch, meta) => {
    // Fake busy work
    return batch.map((w) => ({ value: w.requestId }));
  };

  // Step 1: wall time starts at 1000, step takes place at 2000 (wall elapsed = 1000)
  await scheduler.step(runner, 1000);
  await scheduler.step(runner, 2000);

  const telem = scheduler.telemetry();
  assert.equal(telem.dispatches, 2);
  assert.equal(telem.rowsUseful, 6);
  assert.equal(telem.rowsDispatched, 6);
  assert.equal(telem.paddingWaste, 0.0);
  assert.ok(telem.wallMicros > 0);
  /* utilization <= 1 is a fact about the two clocks, not a clamp: wall is the
   * span from the first step to the last, and every dispatch happens inside
   * it. A number above one would mean the two were measured differently. */
  assert.ok(telem.utilization >= 0.0 && telem.utilization <= 1.0);
  assert.ok(telem.deviceBusyMicros <= telem.wallMicros);
});

test('a prefill that bound a shared prefix writes after it, not over it', async () => {
  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 4, laneTokenCapacity: 32, maxPages: 32, policy: 'paged',
  });
  const scheduler = new BatchScheduler({ cache, maxQueueDepth: 8 });

  const prefills = [];
  const runner = (batch) => {
    for (const work of batch) {
      if (work.phase === 'prefill') {
        prefills.push({ id: work.requestId, position: work.position, tokens: work.tokens });
      }
    }
    return batch.map(() => ({}));
  };

  // Publishes an 8-token prefix under 'sys'.
  scheduler.submitLLM({ promptTokens: 8, maxTokens: 1, promptKey: 'sys' });
  await scheduler.runUntilIdle(runner);
  // Binds those 8 tokens and prefills a 4-token continuation on top.
  scheduler.submitLLM({ promptTokens: 12, maxTokens: 1, promptKey: 'sys' });
  await scheduler.runUntilIdle(runner);

  assert.equal(scheduler.telemetry().prefixReuses, 1);
  assert.deepEqual(prefills, [
    { id: 1, position: 0, tokens: 8 },
    /* Not zero. The first eight tokens are pages this lane shares with the
     * publisher, so a prefill starting at zero would rewrite another
     * request's attention state. */
    { id: 2, position: 8, tokens: 4 },
  ]);
});

test('work-conserving dispatches every ready group in the round, not one per round', async () => {
  const scheduler = new BatchScheduler({ maxLanes: 8, maxQueueDepth: 64 });
  const dispatched = [];
  const runner = (batch, meta) => {
    dispatched.push(formatGroupKey(meta.groupKey));
    return batch.map(() => ({}));
  };

  for (let i = 0; i < 4; i++) scheduler.submitStateless({ modelId: 'vision', shapeSignatureMinusBatch: 'A' });
  for (let i = 0; i < 4; i++) scheduler.submitStateless({ modelId: 'audio', shapeSignatureMinusBatch: 'B' });

  await scheduler.step(runner, 1000);

  assert.equal(dispatched.length, 2, 'both ready groups go in the same round');
  assert.equal(new Set(dispatched).size, 2);
  assert.equal(scheduler.telemetry().completed, 8);
  assert.equal(scheduler.queueDepth, 0, 'nothing waits a round for a device that is idle');
});

test('group order is by oldest arrival, so a rare shape is not starved', async () => {
  const scheduler = new BatchScheduler({ maxLanes: 4, maxQueueDepth: 64 });
  const order = [];
  const runner = (batch, meta) => {
    order.push(meta.groupKey.shapeSignatureMinusBatch);
    return batch.map(() => ({}));
  };

  scheduler.submitStateless({ shapeSignatureMinusBatch: 'rare' });
  for (let i = 0; i < 4; i++) scheduler.submitStateless({ shapeSignatureMinusBatch: 'common' });

  await scheduler.step(runner, 1000);
  assert.deepEqual(order, ['rare', 'common'],
    'the group holding the oldest request is dispatched first');
});

test('fill_first holds only its own partial group and never a resident decode', async () => {
  const cache = new PagedKVCache({
    lanes: 4, pageTokens: 4, laneTokenCapacity: 32, maxPages: 32, policy: 'paged',
  });
  let virtualNow = 0;
  const scheduler = new BatchScheduler({
    cache,
    maxLanes: 4,
    maxQueueDepth: 16,
    clock: () => virtualNow,
    dispatchPolicy: { mode: 'fill_first', maxWaitMicros: 1_000_000 },
  });

  const decodeDispatches = [];
  const statelessDispatches = [];
  const runner = (batch, meta) => {
    if (meta.kind === 'decode') decodeDispatches.push(meta.usefulCount);
    if (meta.kind === 'stateless') statelessDispatches.push(meta.usefulCount);
    return batch.map(() => ({}));
  };

  scheduler.submitLLM({ promptTokens: 4, maxTokens: 4 });
  scheduler.submitStateless({ shapeSignatureMinusBatch: 'partial' });

  for (let round = 0; round < 4; round++) {
    virtualNow += 1000;
    await scheduler.step(runner, virtualNow);
  }

  assert.ok(decodeDispatches.length >= 2,
    'an admitted lane already holds its KV; waiting buys no batching and costs a token per round');
  assert.equal(statelessDispatches.length, 0,
    'the partial stateless group is still inside its latency budget');

  // Age the stateless group past the budget; it goes without needing to fill.
  virtualNow += 2_000_000;
  await scheduler.step(runner, virtualNow);
  assert.deepEqual(statelessDispatches, [1]);
});

test('multiple_of pads with a copy of the first row, never with zeros', async () => {
  const scheduler = new BatchScheduler({ maxLanes: 4, multipleOf: 4 });
  let observed = null;
  const runner = (batch, meta) => {
    observed = { batch, meta };
    return batch.map((work) => ({ outputs: { out: cloneRuntimeArray(work.inputs.x) } }));
  };

  scheduler.submitStateless({ inputs: { x: new Float32Array([7, 8]) } });
  scheduler.submitStateless({ inputs: { x: new Float32Array([9, 10]) } });
  await scheduler.step(runner, 1000);

  assert.equal(observed.meta.usefulCount, 2);
  assert.equal(observed.meta.paddedCount, 4);
  assert.equal(observed.batch.length, 4, 'the padding rows are real rows the callback receives');
  for (const padded of observed.batch.slice(2)) {
    assert.equal(padded.padding, true);
    assert.equal(padded.requestId, -1);
    /* Zeros through a normalization produce NaN and poison the diagnosis of
     * the rows that mattered, so a padding row repeats the first one. */
    assert.deepEqual(padded.inputs.x, new Float32Array([7, 8]));
  }
  assert.equal(observed.batch.filter((w) => !w.padding).length, 2);

  assert.equal(scheduler.results.length, 2, 'padding rows reach no request');
  assert.equal(scheduler.telemetry().rowsDispatched, 4);
  assert.equal(scheduler.telemetry().rowsUseful, 2);
});

test('multiple_of keeps the remainder queued rather than padding a full batch', async () => {
  const scheduler = new BatchScheduler({ maxLanes: 8, multipleOf: 4, maxQueueDepth: 16 });
  const counts = [];
  const runner = (batch, meta) => {
    counts.push([meta.usefulCount, meta.paddedCount]);
    return batch.map(() => ({}));
  };
  for (let i = 0; i < 6; i++) scheduler.submitStateless({});

  await scheduler.step(runner, 1000);
  assert.deepEqual(counts, [[4, 4]], 'n >= k sends floor(n/k)*k');
  assert.equal(scheduler.queueDepth, 2);

  await scheduler.step(runner, 2000);
  assert.deepEqual(counts[1], [2, 4], 'the remainder is padded on the round it can no longer fill');
});

test('a callback returning too many outcomes fails the batch', async () => {
  const scheduler = new BatchScheduler({ maxLanes: 4 });
  scheduler.submitStateless({});
  scheduler.submitStateless({});

  await scheduler.step((batch) => [...batch.map(() => ({})), {}], 1000);

  assert.equal(scheduler.results.length, 2);
  for (const result of scheduler.results) {
    assert.equal(result.state, 'failed');
    assert.match(result.error.message, /2-row step returned 3 outcome\(s\)/);
  }
});

test('queue-depth percentiles come from a bounded window', async () => {
  const scheduler = new BatchScheduler({ maxLanes: 1, maxQueueDepth: 64, queueDepthWindow: 4 });
  const runner = (batch) => batch.map(() => ({}));

  // Deep queue for a while, then quiet. The window must forget the deep part.
  for (let i = 0; i < 20; i++) scheduler.submitStateless({});
  for (let round = 0; round < 20; round++) await scheduler.step(runner, 1000 + round * 1000);
  for (let round = 0; round < 8; round++) await scheduler.step(runner, 30000 + round * 1000);

  const telem = scheduler.telemetry();
  assert.equal(telem.queueDepthP50, 0);
  assert.equal(telem.queueDepthP99, 0);
  assert.equal(telem.maxQueueDepthSeen, 20, 'the high-water mark is still reported in full');
});

test('group identity is length-delimited and execution kind cannot collide', async () => {
  const scheduler = new BatchScheduler({ maxLanes: 4 });
  const groups = [];
  scheduler.submitStateless({
    modelId: 'a|b', adapterRevision: null, shapeSignatureMinusBatch: 'c',
  });
  scheduler.submitStateless({
    modelId: 'a', adapterRevision: 'b|', shapeSignatureMinusBatch: 'c',
  });
  await scheduler.step((batch, metadata) => {
    groups.push([metadata.kind, metadata.groupKey.modelId, batch.length]);
    return batch.map(() => ({}));
  });
  assert.deepEqual(groups, [
    ['stateless', 'a|b', 1],
    ['stateless', 'a', 1],
  ]);

  const cache = new PagedKVCache({
    lanes: 2, pageTokens: 1, laneTokenCapacity: 8, maxPages: 16, policy: 'paged',
  });
  const mixed = new BatchScheduler({ cache, maxLanes: 2 });
  const decode = mixed.submitLLM({
    modelId: 'same', shapeSignatureMinusBatch: 'shape', promptTokens: 1, maxTokens: 1,
  });
  const stateless = mixed.submitStateless({
    modelId: 'same', shapeSignatureMinusBatch: 'shape', rowsPerLane: 1,
  });
  await mixed.runUntilIdle((batch) => batch.map(() => ({})));
  assert.equal(mixed.result(decode).state, 'completed');
  assert.equal(mixed.result(stateless).state, 'completed');
});

test('step calls are serialized and cannot submit the same decode lane twice', async () => {
  const cache = new PagedKVCache({
    lanes: 1, pageTokens: 1, laneTokenCapacity: 8, maxPages: 8, policy: 'paged',
  });
  const scheduler = new BatchScheduler({ cache, maxLanes: 1 });
  scheduler.submitLLM({ promptTokens: 1, maxTokens: 2 });
  await scheduler.step((batch) => batch.map(() => ({})));

  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  let active = 0;
  let maximumActive = 0;
  const positions = [];
  const runner = async (batch) => {
    active++;
    maximumActive = Math.max(maximumActive, active);
    positions.push(batch[0].position);
    if (positions.length === 1) await gate;
    active--;
    return batch.map(() => ({}));
  };
  const first = scheduler.step(runner);
  const second = scheduler.step(runner);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(positions, [1]);
  release();
  await Promise.all([first, second]);
  assert.equal(maximumActive, 1);
  assert.deepEqual(positions, [1, 2]);
  assert.equal(cache.kvLength[0], 0, 'completed lane is released only after both callbacks');
});

test('prefill is chunked by the hard token budget', async () => {
  const cache = new PagedKVCache({
    lanes: 1, pageTokens: 2, laneTokenCapacity: 16, maxPages: 8, policy: 'paged',
  });
  const scheduler = new BatchScheduler({
    cache, maxLanes: 1, tokenBudgetPerDispatch: 4,
  });
  scheduler.submitLLM({ promptTokens: 10, maxTokens: 1 });
  const prefillRows = [];
  await scheduler.runUntilIdle((batch, metadata) => {
    if (metadata.kind === 'prefill') prefillRows.push(metadata.totalRows);
    assert.ok(metadata.totalRows <= 4);
    return batch.map(() => ({}));
  });
  assert.deepEqual(prefillRows, [4, 4, 2]);
});

test('illegal padding contracts fail before admission and result retention is bounded', async () => {
  assert.throws(
    () => new BatchScheduler({ maxLanes: 2, multipleOf: 4 }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );
  assert.throws(
    () => new BatchScheduler({
      maxLanes: 4, multipleOf: 4, tokenBudgetPerDispatch: 2,
    }).submitStateless({ rowsPerLane: 1 }),
    (error) => error.code === 'INVALID_ARGUMENT',
  );

  const scheduler = new BatchScheduler({
    maxLanes: 1, maxQueueDepth: 4, maxRetainedResults: 2,
  });
  const ids = [];
  for (let index = 0; index < 4; index++) {
    ids.push(scheduler.submitStateless({ payload: new Uint8Array(1024) }));
    await scheduler.step((batch) => batch.map(() => ({})));
  }
  assert.deepEqual(scheduler.results.map(({ requestId }) => requestId), ids.slice(-2));
  assert.equal(scheduler.state(ids[0]), null);
  assert.equal(scheduler.takeResult(ids[2]).requestId, ids[2]);
  assert.equal(scheduler.results.length, 1);
});
