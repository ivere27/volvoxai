import test from 'node:test';
import assert from 'node:assert/strict';

import {
  TinyReceiptCharVocab,
  TinyReceiptW8A8Session,
  preprocessTinyReceiptImage,
  TINY_RECEIPT_W8A8_FAMILY_ORDER,
} from '../TinyReceiptW8A8Session.js';
import { CPUEngine, DecodeSession } from '../../../ts/index.js';

const PACKAGE_URL = 'https://example.test/tiny-receipt/package_manifest.json';

function tensor(name, shape, dtype, isInput = false) {
  const bytes = dtype === 'float32' || dtype === 'int32' ? 4 : 1;
  return { name, shape, dtype, isInput, quantization: null, sizeBytes: shape.reduce((a, b) => a * b, 1) * bytes };
}

function graphForRouter() {
  const qIds = tensor('q_ids', [1, 4], 'int32', true);
  const keep = tensor('router_keep', [1, 4], 'int32', true);
  const output = tensor('router_family', [1], 'int32');
  return {
    kind: 'router',
    tensors: new Map([[qIds.name, qIds], [keep.name, keep], [output.name, output]]),
    nodes: [{ opType: 'QArgMax', outputs: { out: output } }],
    outputNames: ['router_family'],
  };
}

function graphForFamily(family) {
  const image = tensor('image', [1, 2, 2, 1], 'float32', true);
  const qIds = tensor('q_ids', [1, 4], 'int32', true);
  const routerKeep = tensor('router_keep', [1, 4], 'int32', true);
  const memoryKeep = tensor('memory_keep', [1, 7], 'int32', true);
  const yIds = tensor('y_ids', [1, 3], 'int32', true);
  const yKeep = tensor('y_keep', [1, 3], 'int32', true);
  const output = tensor('token_ids', [1, 3], 'int32');
  return {
    kind: 'family', family,
    tensors: new Map([
      [image.name, image], [qIds.name, qIds], [routerKeep.name, routerKeep],
      [memoryKeep.name, memoryKeep], [yIds.name, yIds], [yKeep.name, yKeep], [output.name, output],
    ]),
    nodes: [{ opType: 'QArgMax', outputs: { out: output } }],
    outputNames: ['token_ids'],
  };
}

function packageManifest() {
  const family = {
    config: 'family.config.json',
    interface: {
      inputs: {
        image: 'image', q_ids: 'q_ids', router_keep: 'router_keep', memory_keep: 'memory_keep',
        y_ids: 'y_ids', y_keep: 'y_keep',
      },
      output_name: 'token_ids',
    },
  };
  return {
    format: 'volvoxai-tiny-receipt-vqa-w8a8-materialized-package-v1',
    weights: { file: 'model.safetensors' },
    family_order: [...TINY_RECEIPT_W8A8_FAMILY_ORDER],
    router: {
      config: 'router.config.json', output_name: 'router_family',
      inputs: { q_ids: 'q_ids', router_keep: 'router_keep' },
    },
    explicit_families: Object.fromEntries(TINY_RECEIPT_W8A8_FAMILY_ORDER.map((name) => [name, family])),
    vocab: { file: 'vocab.json', token_ids: { pad: 0, bos: 1, eos: 2, unk: 3 } },
    preprocessing: {
      color_space: 'grayscale', resize: { width: 2, height: 2, resample: 'bilinear' },
      normalization: 'minus-one-one', model_input_layout: 'NHWC', model_input_dtype: 'float32',
    },
  };
}

function fakeFetch(resources) {
  return async (url) => {
    const body = resources.get(String(url));
    if (body == null) return { ok: false, status: 404, statusText: 'not found' };
    return { ok: true, json: async () => body };
  };
}

function cloneInputs(inputs) {
  return Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, value.slice()]));
}

async function syntheticSession({ gpuReadback = false, incrementalSupport = false,
  incrementalDecodeSupport = false, decodeSessionSupport = false,
  deviceFeedbackSupport = false, safetensorsCache = null } = {}) {
  const resources = new Map([
    [PACKAGE_URL, packageManifest()],
    ['https://example.test/tiny-receipt/vocab.json', { itos: ['<pad>', '<bos>', '<eos>', '<unk>', 'A', '😀'] }],
  ]);
  const graphLoads = [];
  const calls = [];
  const decodeSessions = [];
  const readbacks = [];
  const runtime = { loadGraph() {}, compile() {} };
  const session = await TinyReceiptW8A8Session.load({
    runtime,
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    ...(safetensorsCache == null ? {} : { safetensorsCache }),
    graphLoader: async ({ kind, family, configUrl, weightsUrl }) => {
      graphLoads.push({ kind, family, configUrl, weightsUrl });
      return kind === 'router' ? graphForRouter() : graphForFamily(family);
    },
    compileGraph: async (graph) => {
      const outputName = graph.kind === 'router' ? 'router_family' : 'token_ids';
      const resultFor = (inputs) => {
        if (graph.kind === 'router') return Int32Array.of(1); // address
        const step = inputs.y_keep.reduce((count, keep) => count + (keep !== 0 ? 1 : 0), 0) - 1;
        const output = new Int32Array(3);
        output[step] = step === 0 ? 4 : 2; // A, then EOS
        return output;
      };
      const decorate = (engine) => {
        if (!decodeSessionSupport) return engine;
        engine.createDecodeSession = function createDecodeSession(options) {
          const record = { options, seeds: 0, steps: [], closes: 0 };
          const decode = new DecodeSession(this, options);
          const seed = decode.seed.bind(decode);
          const step = decode.step.bind(decode);
          const close = decode.close.bind(decode);
          decode.seed = (...args) => { record.seeds++; return seed(...args); };
          decode.step = (inputs, stepOptions) => {
            record.steps.push(stepOptions);
            return step(inputs, stepOptions);
          };
          decode.close = (...args) => { record.closes++; return close(...args); };
          record.session = decode;
          decodeSessions.push(record);
          return decode;
        };
        return engine;
      };
      if (gpuReadback) {
        const outputBuffer = { graph, outputName };
        const engine = {
          graph,
          supportsIncrementalExecution: incrementalSupport,
          supportsIncrementalRows: incrementalDecodeSupport,
          gpuBuffers: new Map([[outputName, outputBuffer]]),
          async execute(inputs, options) {
            calls.push({ kind: graph.kind, family: graph.family, inputs: cloneInputs(inputs), options, method: 'execute' });
            this.last = resultFor(inputs);
            return outputBuffer;
          },
          async readBuffer(buffer, sizeBytes, dtype) {
            assert.equal(buffer, outputBuffer);
            assert.equal(dtype, 'int32');
            assert.equal(sizeBytes, graph.tensors.get(outputName).sizeBytes);
            readbacks.push({ kind: graph.kind, mode: 'full', sizeBytes });
            return this.last;
          },
          async readBufferRange(buffer, byteOffset, sizeBytes, dtype) {
            assert.equal(buffer, outputBuffer);
            assert.equal(dtype, 'int32');
            assert.equal(byteOffset % Int32Array.BYTES_PER_ELEMENT, 0);
            assert.equal(sizeBytes % Int32Array.BYTES_PER_ELEMENT, 0);
            readbacks.push({ kind: graph.kind, mode: 'range', byteOffset, sizeBytes });
            const begin = byteOffset / Int32Array.BYTES_PER_ELEMENT;
            return this.last.slice(begin, begin + sizeBytes / Int32Array.BYTES_PER_ELEMENT);
          },
        };
        if (deviceFeedbackSupport) {
          engine.executeDeviceFeedbackDecode = async function executeDeviceFeedbackDecode(inputs, options) {
            calls.push({
              kind: graph.kind,
              family: graph.family,
              inputs: inputs == null ? null : cloneInputs(inputs),
              options,
              method: 'device-feedback',
            });
            this.last = Int32Array.of(4, 4, 2);
            return outputBuffer;
          };
        }
        return decorate(engine);
      }
      return decorate({
        graph,
        supportsIncrementalExecution: incrementalSupport,
        supportsIncrementalRows: incrementalDecodeSupport,
        async execute(inputs, options) {
          calls.push({ kind: graph.kind, family: graph.family, inputs: cloneInputs(inputs), options, method: 'execute' });
          return { [outputName]: resultFor(inputs) };
        },
      });
    },
  });
  return { session, graphLoads, calls, decodeSessions, readbacks };
}

test('TinyReceipt CharVocab keeps source clean-text, code-point, EOS, and special-token semantics', () => {
  const vocab = new TinyReceiptCharVocab(['<pad>', '<bos>', '<eos>', '<unk>', 'A', '😀'], {
    pad: 0, bos: 1, eos: 2, unk: 3,
  });
  // Whitespace is normalized to one ordinary space. This synthetic vocabulary
  // deliberately has no space character, so it becomes the source <unk> ID.
  assert.deepEqual(vocab.encodeQuestion('  A\t😀\n', 4), [4, 3, 5, 2]);
  assert.deepEqual(vocab.encodeQuestion('AAAA', 3), [4, 4, 2]);
  assert.equal(vocab.decode([1, 4, 0, 5, 2, 4]), 'A😀');
});

test('TinyReceipt preprocessing creates grayscale, bilinear, minus-one-one F32 NHWC', async () => {
  const output = await preprocessTinyReceiptImage({
    // Pillow converts/resizes an 8-bit L plane. The 127.5 bilinear result
    // rounds to 128 before the evaluator's minus-one-one normalization.
    data: Uint8Array.of(0, 255, 0, 255), width: 2, height: 2, channels: 1,
  }, { width: 1, height: 1 });
  assert.ok(output instanceof Float32Array);
  assert.equal(output.length, 1);
  assert.ok(Math.abs(output[0] - ((128 / 255) * 2 - 1)) < 1e-7);

  // Source order matters: RGB -> rounded 8-bit L -> BILINEAR. Red and blue
  // become 76 and 29, which reduce to source-compatible L=53.
  const rgb = await preprocessTinyReceiptImage({
    data: Uint8Array.of(255, 0, 0, 0, 0, 255), width: 2, height: 1, channels: 3,
  }, { width: 1, height: 1 });
  assert.ok(Math.abs(rgb[0] - ((53 / 255) * 2 - 1)) < 1e-7);
});

test('TinyReceipt accepts the read-only safetensors cache exported by a separate bundle', async () => {
  const cache = { size: 0, async load() {}, clear() {} };
  const { session } = await syntheticSession({ safetensorsCache: cache });
  assert.ok(session instanceof TinyReceiptW8A8Session);
});

test('TinyReceipt session runs router then selected family with raw I32/QArgMax IDs and ordinary forward', async () => {
  const { session, graphLoads, calls } = await syntheticSession();
  const result = await session.generate({
    prompt: 'A😀',
    image: { data: Uint8Array.of(0, 255, 128, 64), width: 2, height: 2, channels: 1 },
    maxNewTokens: 3,
  });

  assert.deepEqual(graphLoads.map(({ kind, family }) => [kind, family]), [
    ['router', null], ['explicit-family', 'address'],
  ]);
  assert.equal(result.execution, 'ordinary-forward');
  assert.equal(result.routerFamily, 'address');
  assert.equal(result.family, 'address');
  assert.deepEqual(result.questionTokenIds, [4, 5, 2]);
  assert.deepEqual(result.tokenIds, [4, 2]);
  assert.equal(result.text, 'A');
  assert.equal(result.stoppedAtEos, true);
  assert.equal(calls.length, 3); // router, A, EOS
  assert.equal(calls.every((call) => call.method === 'execute'), true);
  assert.deepEqual([...calls[0].inputs.q_ids], [4, 5, 2, 0]);
  assert.deepEqual([...calls[0].inputs.router_keep], [1, 1, 1, 0]);
  assert.deepEqual([...calls[1].inputs.memory_keep], [1, 1, 1, 1, 1, 1, 0]);
  assert.deepEqual([...calls[1].inputs.y_ids], [1, 0, 0]);
  assert.deepEqual([...calls[1].inputs.y_keep], [1, 0, 0]);
  assert.deepEqual([...calls[2].inputs.y_ids], [1, 4, 0]);
  assert.deepEqual([...calls[2].inputs.y_keep], [1, 1, 0]);
  assert.ok(calls[1].inputs.image instanceof Float32Array);
  assert.equal(calls[1].inputs.image.length, 4);
});

test('TinyReceipt session reads terminal QArgMax IDs through the WebGPU readback surface', async () => {
  const { session } = await syntheticSession({ gpuReadback: true });
  const routed = await session.route('A');
  assert.equal(routed.family, 'address');
  assert.deepEqual([...routed.qIds], [4, 2, 0, 0]);
  assert.deepEqual([...routed.routerKeep], [1, 1, 0, 0]);
});

test('TinyReceipt preload assembles router and selected family graphs exactly once', async () => {
  const { session, graphLoads } = await syntheticSession();
  const prepared = await session.preload({ families: ['address', 'phone', 'address'] });
  assert.deepEqual(prepared, { router: true, families: ['address', 'phone'] });
  assert.deepEqual(graphLoads.map(({ kind, family }) => [kind, family]), [
    ['router', null],
    ['explicit-family', 'address'],
    ['explicit-family', 'phone'],
  ]);

  const result = await session.generate({
    prompt: 'A', preprocessed: true, image: new Float32Array(4), maxNewTokens: 3,
  });
  assert.equal(result.family, 'address');
  assert.equal(graphLoads.length, 3, 'generation reuses the preloaded router and family graph');
  await assert.rejects(
    session.preload({ families: ['invalid'] }),
    /preload families must be 'all'/,
  );
});

test('TinyReceipt incremental mode resets once per image and marks only decoder inputs changed', async () => {
  const { session, calls } = await syntheticSession({ incrementalSupport: true });
  const result = await session.generate({
    prompt: 'A', incremental: true,
    image: { data: Uint8Array.of(0, 255, 128, 64), width: 2, height: 2, channels: 1 },
    maxNewTokens: 3,
  });
  assert.equal(result.text, 'A');
  assert.equal(result.execution, 'incremental-dependency-cache');
  const familyCalls = calls.filter((call) => call.kind === 'family');
  assert.equal(familyCalls.length, 2);
  assert.equal(familyCalls[0].options.incremental, true);
  assert.equal(familyCalls[0].options.incrementalReset, true);
  assert.deepEqual(familyCalls[1].options.changedInputs, ['y_ids', 'y_keep']);
  assert.equal(familyCalls[1].options.incrementalReset, false);
  assert.equal(Object.prototype.hasOwnProperty.call(familyCalls[1].options, 'incrementalRowPosition'), false);
});

test('TinyReceipt negotiates CPU/WASM row decode after the full incremental seed', async () => {
  const { session, calls } = await syntheticSession({
    incrementalSupport: true,
    incrementalDecodeSupport: true,
  });
  const result = await session.generate({
    prompt: 'A', incremental: true,
    image: { data: Uint8Array.of(0, 255, 128, 64), width: 2, height: 2, channels: 1 },
    maxNewTokens: 3,
  });
  assert.equal(result.text, 'A');
  assert.equal(result.execution, 'incremental-row-kv-cache');
  const familyCalls = calls.filter((call) => call.kind === 'family');
  assert.equal(Object.prototype.hasOwnProperty.call(familyCalls[0].options, 'incrementalRowPosition'), false);
  assert.equal(familyCalls[1].options.incrementalRowPosition, 1);
  assert.deepEqual(familyCalls[1].options.changedInputs, ['y_ids', 'y_keep']);
});

test('TinyReceipt reads only the current I32 token from WebGPU row decode', async () => {
  const { session, readbacks } = await syntheticSession({
    gpuReadback: true,
    incrementalSupport: true,
    incrementalDecodeSupport: true,
  });
  const result = await session.generate({
    prompt: 'A', incremental: true,
    image: { data: Uint8Array.of(0, 255, 128, 64), width: 2, height: 2, channels: 1 },
    maxNewTokens: 3,
  });
  assert.equal(result.text, 'A');
  assert.equal(result.execution, 'incremental-row-kv-cache');
  assert.deepEqual(readbacks, [
    { kind: 'router', mode: 'full', sizeBytes: Int32Array.BYTES_PER_ELEMENT },
    { kind: 'family', mode: 'range', byteOffset: 0, sizeBytes: Int32Array.BYTES_PER_ELEMENT },
    { kind: 'family', mode: 'range', byteOffset: Int32Array.BYTES_PER_ELEMENT,
      sizeBytes: Int32Array.BYTES_PER_ELEMENT },
  ]);
});

test('TinyReceipt batches WebGPU token feedback and EOS checks entirely on-device', async () => {
  const { session, calls, readbacks } = await syntheticSession({
    gpuReadback: true,
    incrementalSupport: true,
    incrementalDecodeSupport: true,
    deviceFeedbackSupport: true,
  });
  const result = await session.generate({
    prompt: 'A', incremental: true, deviceFeedbackChunkSize: 2,
    image: { data: Uint8Array.of(0, 255, 128, 64), width: 2, height: 2, channels: 1 },
    maxNewTokens: 3,
  });
  assert.equal(result.text, 'AA');
  assert.deepEqual(result.tokenIds, [4, 4, 2]);
  assert.equal(result.execution, 'device-feedback-row-kv-cache');
  const familyCalls = calls.filter((call) => call.kind === 'family');
  assert.equal(familyCalls.length, 2);
  assert.equal(familyCalls.every((call) => call.method === 'device-feedback'), true);
  assert.ok(familyCalls[0].inputs);
  assert.equal(familyCalls[1].inputs, null);
  assert.deepEqual(familyCalls.map(({ options }) => options), [
    {
      tokenInput: 'y_ids', keepInput: 'y_keep', output: 'token_ids',
      startPosition: 0, endPosition: 2, rowsPerSubmission: 1,
    },
    {
      tokenInput: 'y_ids', keepInput: 'y_keep', output: 'token_ids',
      startPosition: 2, endPosition: 3, rowsPerSubmission: 1,
    },
  ]);
  assert.deepEqual(readbacks, [
    { kind: 'router', mode: 'full', sizeBytes: Int32Array.BYTES_PER_ELEMENT },
    { kind: 'family', mode: 'range', byteOffset: 0,
      sizeBytes: 2 * Int32Array.BYTES_PER_ELEMENT },
    { kind: 'family', mode: 'range', byteOffset: 2 * Int32Array.BYTES_PER_ELEMENT,
      sizeBytes: Int32Array.BYTES_PER_ELEMENT },
  ]);
});

test('TinyReceipt consumes the backend DecodeSession facade when available', async () => {
  const { session, calls, decodeSessions } = await syntheticSession({
    incrementalSupport: true,
    incrementalDecodeSupport: true,
    decodeSessionSupport: true,
  });
  const result = await session.generate({
    prompt: 'A', incremental: true,
    image: { data: Uint8Array.of(0, 255, 128, 64), width: 2, height: 2, channels: 1 },
    maxNewTokens: 3,
  });
  assert.equal(result.execution, 'incremental-row-kv-cache');
  assert.equal(decodeSessions.length, 1);
  assert.deepEqual(decodeSessions[0].options, {
    changedInputs: ['y_ids', 'y_keep'], rowMode: 'auto',
  });
  assert.equal(decodeSessions[0].seeds, 1);
  assert.deepEqual(decodeSessions[0].steps, [{
    changedInputs: ['y_ids', 'y_keep'], position: 1,
  }]);
  assert.equal(decodeSessions[0].closes, 1);
  const familyCalls = calls.filter((call) => call.kind === 'family');
  assert.equal(familyCalls[0].options.incrementalReset, true);
  assert.equal(familyCalls[1].options.incrementalRowPosition, 1);
});

test('CPU incremental executor invalidates a failed reset before retrying', async () => {
  const staticInput = tensor('static', [1], 'float32', true);
  const dynamicInput = tensor('dynamic', [1], 'float32', true);
  const staticValue = tensor('static_value', [1], 'float32');
  const dynamicValue = tensor('dynamic_value', [1], 'float32');
  const output = tensor('output', [1], 'float32');
  const graph = {
    tensors: new Map([staticInput, dynamicInput, staticValue, dynamicValue, output].map((value) => [value.name, value])),
    nodes: [
      { inputs: { input: staticInput }, outputs: { out: staticValue } },
      { inputs: { input: dynamicInput }, outputs: { out: dynamicValue } },
      { inputs: { a: staticValue, b: dynamicValue }, outputs: { out: output } },
    ],
    outputNames: ['output'],
  };
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const calls = [];
  let fail = false;
  engine._runNode = (_node, { nodeIndex }) => {
    calls.push(nodeIndex);
    if (fail) throw new Error('synthetic reset failure');
  };
  const inputs = { static: Float32Array.of(1), dynamic: Float32Array.of(2) };
  await engine.execute(inputs, { incremental: true, incrementalReset: true });
  assert.equal(engine._incrementalCacheValid, true);
  calls.length = 0;
  await engine.execute(inputs, { incremental: true, changedInputs: ['dynamic'] });
  assert.deepEqual(calls, [1, 2]);
  fail = true;
  await assert.rejects(engine.execute(inputs, { incremental: true, incrementalReset: true }), /synthetic reset failure/);
  assert.equal(engine._incrementalCacheValid, false);
  fail = false;
  calls.length = 0;
  await engine.execute(inputs, { incremental: true, changedInputs: ['dynamic'] });
  assert.deepEqual(calls, [0, 1, 2]);
});
