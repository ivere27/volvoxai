import test from 'node:test';
import assert from 'node:assert/strict';

import { ReceiptDigitSession, decodeSlots } from '../ReceiptDigitSession.js';

const LAYOUT = { slots: 16, phone_slots: 12, street_slots: 4, blank_class: 10, num_classes: 11 };

function logitsFor(classes) {
  const values = new Float32Array(LAYOUT.slots * LAYOUT.num_classes);
  for (let slot = 0; slot < LAYOUT.slots; slot++) {
    const chosen = classes[slot] ?? LAYOUT.blank_class;
    values[slot * LAYOUT.num_classes + chosen] = 1;
  }
  return values;
}

function manifest(overrides = {}) {
  return {
    format: 'volvoxai-receipt-digit-reader-onnx-package-v1',
    variant: 'fp32',
    abi: {
      input: { name: 'input0', shape: [1, 1, 320, 672] },
      output: 'slot_logits',
      batch: { per_request: 1, symbol: null, min: 1, max: 1, multiple_of: 1 },
    },
    decode: { ...LAYOUT },
    preprocess: { width: 672, height: 320 },
    ...overrides,
  };
}

function fakeResult() {
  return {
    report: {},
    output(name) {
      assert.equal(name, 'slot_logits');
      return { read: async () => logitsFor([0, 1, 2, 3, 4, 5, 6, 10, 10, 10, 10, 10, 9, 9, 9, 10]) };
    },
    close: async () => undefined,
  };
}

function fakeSnapshot({
  symbol = null, max = 1, outputLeading = symbol ?? 1, outputs = ['slot_logits'],
} = {}) {
  const inputLeading = symbol ?? 1;
  return {
    revision: 1,
    graph: {
      inputs: {
        input0: { name: 'input0', shape: [inputLeading, 1, 320, 672], dtype: 'float32' },
      },
      outputs,
      tensors: {
        slot_logits: { name: 'slot_logits', shape: [outputLeading, 16, 11], dtype: 'float32' },
        extra: { name: 'extra', shape: [outputLeading, 1], dtype: 'float32' },
      },
      dimensions: symbol === null ? {} : {
        [symbol]: { name: symbol, min: 1, max, multiple_of: 1 },
      },
    },
  };
}

function fakeSessionApi(run, observations, snapshot = fakeSnapshot()) {
  const compiled = {
    run,
    createContext() {
      observations.createContextCalls++;
      throw new Error('ReceiptDigitSession must not create an execution context');
    },
    close: async () => {
      observations.compiledClosed++;
      if (observations.compiledCloseError) throw observations.compiledCloseError;
    },
  };
  const runtime = {
    compile: async (_snapshot, options) => {
      observations.compileOptions.push(options);
      return compiled;
    },
    close: async () => {
      observations.runtimeClosed++;
      if (observations.runtimeCloseError) throw observations.runtimeCloseError;
    },
  };
  return {
    runtime,
    api: {
      Model: { load: async () => snapshot },
      VolvoxAI: {
        createRuntime: async (options) => {
          observations.runtimeOptions.push(options);
          return runtime;
        },
      },
    },
  };
}

test('decodeSlots reads each block until its first blank', () => {
  const record = decodeSlots(
    logitsFor([0, 1, 2, 3, 4, 5, 6, 10, 10, 10, 10, 10, 9, 9, 9, 10]), LAYOUT,
  );
  assert.equal(record.phone, '0123456');
  assert.equal(record.street, '999');
  assert.equal(record.slotClasses.length, LAYOUT.slots);
});

test('decodeSlots stops at an interior blank rather than dropping it', () => {
  // The slots are left-aligned, so a blank means the number ended there.
  // Skipping it would splice unrelated digits together into a plausible wrong
  // number, which is indistinguishable from a right one downstream.
  const record = decodeSlots(
    logitsFor([5, 6, 10, 7, 8, 10, 10, 10, 10, 10, 10, 10, 1, 10, 2, 10]), LAYOUT,
  );
  assert.equal(record.phone, '56');
  assert.equal(record.street, '1');
});

test('decodeSlots yields empty strings for an all-blank read', () => {
  const record = decodeSlots(logitsFor([]), LAYOUT);
  assert.equal(record.phone, '');
  assert.equal(record.street, '');
});

test('decodeSlots rejects a logits block of the wrong size', () => {
  assert.throws(() => decodeSlots(new Float32Array(10), LAYOUT), /16 x 11/);
  assert.throws(() => decodeSlots([1, 2, 3], LAYOUT), /16 x 11/);
});

test('opening a session rejects a foreign package format', async () => {
  await assert.rejects(
    () => ReceiptDigitSession.open({ manifest: manifest({ format: 'something-else' }) }),
    /volvoxai-receipt-digit-reader-onnx-package-v1/,
  );
});

test('opening a session rejects an inconsistent decode layout', async () => {
  await assert.rejects(
    () => ReceiptDigitSession.open({
      manifest: manifest({ decode: { ...LAYOUT, phone_slots: 11 } }),
    }),
    /phone_slots \+ street_slots/,
  );
  await assert.rejects(
    () => ReceiptDigitSession.open({
      manifest: manifest({ decode: { ...LAYOUT, blank_class: 11 } }),
    }),
    /blank_class/,
  );
});

test('opening a session rejects a malformed ABI or preprocess block', async () => {
  await assert.rejects(
    () => ReceiptDigitSession.open({
      manifest: manifest({ abi: { input: { name: 'x', shape: [1, 1] }, output: 'y' } }),
    }),
    /rank-4/,
  );
  await assert.rejects(
    () => ReceiptDigitSession.open({
      manifest: manifest({ preprocess: { width: 672 } }),
    }),
    /preprocess/,
  );
});

test('opening a session rejects a non-B1 per-request shape', async () => {
  await assert.rejects(
    () => ReceiptDigitSession.open({
      manifest: manifest({
        abi: {
          input: { name: 'input0', shape: [4, 1, 320, 672] },
          output: 'slot_logits',
          batch: { per_request: 1, symbol: 'batch', min: 1, max: 4, multiple_of: 1 },
        },
      }),
    }),
    /per-request shape with batch 1/,
  );
});

test('opening a session requires and validates the graph batch domain', async () => {
  const dynamicAbi = {
    input: { name: 'input0', shape: [1, 1, 320, 672] },
    output: 'slot_logits',
    batch: { per_request: 1, symbol: 'batch', min: 1, max: 4, multiple_of: 1 },
  };
  for (const batch of [
    { ...dynamicAbi.batch, per_request: 2 },
    { ...dynamicAbi.batch, min: 0 },
    { ...dynamicAbi.batch, max: 33 },
    { ...dynamicAbi.batch, multiple_of: 2 },
    { ...dynamicAbi.batch, symbol: null },
  ]) {
    await assert.rejects(
      () => ReceiptDigitSession.open({
        manifest: manifest({ abi: { ...dynamicAbi, batch } }),
      }),
      /manifest\.abi\.batch/,
    );
  }
  await assert.rejects(
    () => ReceiptDigitSession.open({
      manifest: manifest({
        abi: { ...dynamicAbi, batch: { ...dynamicAbi.batch, max: 1 } },
      }),
    }),
    /symbol must be null/,
  );
  await assert.rejects(
    () => ReceiptDigitSession.open({
      manifest: manifest({
        abi: { input: dynamicAbi.input, output: dynamicAbi.output },
      }),
    }),
    /manifest\.abi\.batch/,
  );
});

test('opening a session rejects graph and manifest batch-contract mismatches', async () => {
  const dynamicManifest = manifest({
    abi: {
      input: { name: 'input0', shape: [1, 1, 320, 672] },
      output: 'slot_logits',
      batch: { per_request: 1, symbol: 'batch', min: 1, max: 4, multiple_of: 1 },
    },
  });
  const observations = () => ({
    createContextCalls: 0, compiledClosed: 0, runtimeClosed: 0,
    compileOptions: [], runtimeOptions: [], runs: [],
  });
  for (const [snapshot, pattern] of [
    [fakeSnapshot({ symbol: 'batch', max: 8 }), /exactly match manifest\.abi\.batch/],
    [fakeSnapshot({ symbol: 'batch', max: 4, outputLeading: 1 }), /must lead/],
    [fakeSnapshot({ symbol: 'batch', max: 4, outputs: ['slot_logits', 'extra'] }), /only manifest output/],
  ]) {
    const seen = observations();
    const { api } = fakeSessionApi(async () => fakeResult(), seen, snapshot);
    await assert.rejects(() => ReceiptDigitSession.open({
      manifest: dynamicManifest, graphUrl: 'graph.json', weightsUrl: 'weights', api,
    }), pattern);
    assert.equal(seen.compileOptions.length, 0);
    assert.equal(seen.runtimeClosed, 1);
  }

  const seen = observations();
  const { api } = fakeSessionApi(
    async () => fakeResult(), seen, fakeSnapshot({ symbol: 'batch', max: 1 }),
  );
  await assert.rejects(() => ReceiptDigitSession.open({
    manifest: manifest(), graphUrl: 'graph.json', weightsUrl: 'weights', api,
  }), /literal leading extent 1/);
  assert.equal(seen.compileOptions.length, 0);
  assert.equal(seen.runtimeClosed, 1);
});

test('concurrent reads use stateless compiled.run and preserve run options', async () => {
  const observations = {
    createContextCalls: 0,
    compiledClosed: 0,
    runtimeClosed: 0,
    compileOptions: [],
    runtimeOptions: [],
    runs: [],
  };
  const pending = [];
  const { api } = fakeSessionApi((inputs, options) => {
    observations.runs.push({ inputs, options });
    return new Promise((resolve) => {
      pending.push(() => resolve(fakeResult()));
      if (pending.length === 2) queueMicrotask(() => pending.splice(0).forEach((release) => release()));
    });
  }, observations, fakeSnapshot({ symbol: 'batch', max: 4 }));
  const dynamicManifest = manifest({
    abi: {
      input: { name: 'input0', shape: [1, 1, 320, 672] },
      output: 'slot_logits',
      batch: { per_request: 1, symbol: 'batch', min: 1, max: 4, multiple_of: 1 },
    },
  });
  const execution = { mode: 'scheduled', scheduler: { maxBatchSize: 4 } };
  const session = await ReceiptDigitSession.open({
    manifest: dynamicManifest,
    graphUrl: 'graph.json',
    weightsUrl: 'model.safetensors',
    backend: 'cpu-js',
    execution,
    api,
  });
  const pixels = new Float32Array(320 * 672);
  const records = await Promise.all([
    session.read(pixels, { mode: 'scheduled' }),
    session.read(pixels, { mode: 'scheduled' }),
  ]);

  assert.equal(observations.createContextCalls, 0);
  assert.deepEqual(observations.runtimeOptions, [{
    backends: ['wasm', 'cpu-js'], execution,
  }]);
  assert.equal(observations.runs.length, 2);
  assert.deepEqual(observations.runs.map(({ options }) => options), [
    { mode: 'scheduled' }, { mode: 'scheduled' },
  ]);
  assert.deepEqual(records.map(({ phone, street }) => `${phone}/${street}`), [
    '0123456/999', '0123456/999',
  ]);
  await session.close();
  assert.equal(observations.compiledClosed, 1);
  assert.equal(observations.runtimeClosed, 1);
});

test('DIRECT execution uses compiled.run without allocating a context', async () => {
  const observations = {
    createContextCalls: 0,
    compiledClosed: 0,
    runtimeClosed: 0,
    compileOptions: [],
    runtimeOptions: [],
    runs: [],
  };
  const { api } = fakeSessionApi(async (inputs, options) => {
    observations.runs.push({ inputs, options });
    return fakeResult();
  }, observations);
  const session = await ReceiptDigitSession.open({
    manifest: manifest(),
    graphUrl: 'graph.json',
    weightsUrl: 'model.safetensors',
    backend: 'cpu-js',
    execution: { mode: 'direct' },
    api,
  });
  await session.read(new Float32Array(320 * 672), { mode: 'direct' });

  assert.equal(observations.createContextCalls, 0);
  assert.deepEqual(observations.runtimeOptions[0].execution, { mode: 'direct' });
  assert.deepEqual(observations.runs[0].options, { mode: 'direct' });
  await session.close();
});

test('benchmark reads expose an execution interval and preserve decoded output', async () => {
  const observations = {
    createContextCalls: 0,
    compiledClosed: 0,
    runtimeClosed: 0,
    compileOptions: [],
    runtimeOptions: [],
    runs: [],
  };
  const { api } = fakeSessionApi(async (inputs, options) => {
    observations.runs.push({ inputs, options });
    return fakeResult();
  }, observations);
  const session = await ReceiptDigitSession.open({
    manifest: manifest(),
    graphUrl: 'graph.json',
    weightsUrl: 'model.safetensors',
    backend: 'cpu-js',
    api,
  });

  const measured = await session.readForBenchmark(
    new Float32Array(320 * 672), { mode: 'direct' },
  );
  assert.equal(measured.record.phone, '0123456');
  assert.equal(measured.record.street, '999');
  assert.equal(Number.isFinite(measured.executionMs), true);
  assert.equal(measured.executionMs >= 0, true);
  assert.deepEqual(observations.runs[0].options, { mode: 'direct' });
  await session.close();
});

test('close is idempotent and closes an owned Runtime after a compiled close failure', async () => {
  const compiledError = new Error('compiled close failed');
  const observations = {
    createContextCalls: 0,
    compiledClosed: 0,
    runtimeClosed: 0,
    compileOptions: [],
    runtimeOptions: [],
    runs: [],
    compiledCloseError: compiledError,
  };
  const { api } = fakeSessionApi(async () => fakeResult(), observations);
  const session = await ReceiptDigitSession.open({
    manifest: manifest(), graphUrl: 'graph.json', weightsUrl: 'model.safetensors', api,
  });

  await assert.rejects(session.close(), (error) => error === compiledError);
  await assert.rejects(session.close(), (error) => error === compiledError);
  assert.equal(observations.compiledClosed, 1);
  assert.equal(observations.runtimeClosed, 1);
});

test('a provided Runtime rejects a competing execution policy', async () => {
  const observations = {
    createContextCalls: 0,
    compiledClosed: 0,
    runtimeClosed: 0,
    compileOptions: [],
    runtimeOptions: [],
    runs: [],
  };
  const { api, runtime } = fakeSessionApi(async (inputs, options) => {
    observations.runs.push({ inputs, options });
    return fakeResult();
  }, observations);
  await assert.rejects(() => ReceiptDigitSession.open({
    manifest: manifest(), graphUrl: 'graph.json', weightsUrl: 'model.safetensors',
    runtime, execution: { mode: 'scheduled' }, api,
  }), /execution belongs to Runtime creation/);

  assert.deepEqual(observations.runtimeOptions, []);
  assert.equal(observations.runs.length, 0);
});

test('a provided Runtime keeps its own default execution policy', async () => {
  const observations = {
    createContextCalls: 0,
    compiledClosed: 0,
    runtimeClosed: 0,
    compileOptions: [],
    runtimeOptions: [],
    runs: [],
  };
  const { api, runtime } = fakeSessionApi(async (inputs, options) => {
    observations.runs.push({ inputs, options });
    return fakeResult();
  }, observations);
  const session = await ReceiptDigitSession.open({
    manifest: manifest(), graphUrl: 'graph.json', weightsUrl: 'model.safetensors',
    runtime, api,
  });
  await session.read(new Float32Array(320 * 672));

  assert.deepEqual(observations.runtimeOptions, []);
  assert.equal(observations.runs[0].options, undefined);
  await session.close();
  assert.equal(observations.runtimeClosed, 0);
});
