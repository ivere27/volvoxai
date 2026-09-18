import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
/** Native and WASM enter the same C authoring implementation through proto.
 * The fixtures compare canonical decimal rendering, escaping and metadata. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const MODULE_PATH = process.env.VOLVOXAI_PTQ_WASM
  || join(ROOT, 'dist', packageVersion(), 'volvoxai.wasm');
const INFERENCE_MODULE_PATH = join(
  ROOT, 'dist', packageVersion(), 'volvoxai.lite.wasm',
);
const NATIVE_PATH = firstExisting([
  // Prefer the current Makefile output over an older native-local build.
  join(ROOT, 'build', 'cmake', 'native', 'test_ptq_authoring'),
  join(ROOT, 'native', 'build', 'native', 'test_ptq_authoring'),
]);

function packageVersion() {
  return JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).version;
}

function firstExisting(candidates) {
  return candidates.find((candidate) => existsSync(candidate)) || null;
}

assert.ok(existsSync(MODULE_PATH),
  'volvoxai.wasm is not built (make build_wasm)');
assert.ok(existsSync(INFERENCE_MODULE_PATH),
  'volvoxai.lite.wasm is not built (make build_wasm)');
assert.ok(NATIVE_PATH, 'test_ptq_authoring is not built');

// --- the fixture ------------------------------------------------------------
//
// A transformer block reduced to its quantizable skeleton, matching
// python/tests/ptq_fixture.py: LayerNorm, a dense projection with a bias,
// GELU, and a residual Add. The LayerNorm's `eps` is what makes this
// interesting — it is stored as F32 and written back as the shortest decimal
// that round-trips as a double, sixteen significant digits of it.

const D_MODEL = 8;
const D_FF = 16;
const SEQUENCE = 4;

// JSON.stringify renders 1.5e-5 as "0.000015", which exercises neither the
// exponent nor — for a rounder value — the fractional half of number parsing.
// A source graph from the exporter is full of exponent-form numbers, so the
// fixture is written as one, with a fraction, so that a module mis-parsing
// either half authors visibly differently rather than identically.
const EPS = 1.5e-5;
const EPS_TEXT = '1.5e-05';

function graphText() {
  return JSON.stringify(fixtureGraph(), null, 1)
    .replace(`"eps": ${JSON.stringify(EPS)}`, `"eps": ${EPS_TEXT}`);
}

function fixtureGraph() {
  const shape = (width) => [1, SEQUENCE, width];
  return {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { hidden: { shape: shape(D_MODEL), dtype: 'float32' } },
    nodes: [
      {
        id: 'node_0',
        opType: 'LayerNorm',
        inputs: { input: 'hidden', weight: 'ln.weight', bias: 'ln.bias' },
        outputs: { out: { tensor: 'normed', shape: shape(D_MODEL), dtype: 'float32' } },
        // Written through EPS_TEXT below, in exponent form.
        params: { eps: EPS, d_model: D_MODEL },
      },
      {
        id: 'node_1',
        opType: 'Linear',
        inputs: { input: 'normed', weight: 'fc.weight', bias: 'fc.bias' },
        outputs: { out: { tensor: 'wide', shape: shape(D_FF), dtype: 'float32' } },
        params: { weight_layout: 'dout_din' },
      },
      {
        id: 'node_2',
        opType: 'GELU',
        inputs: { input: 'wide' },
        outputs: { out: { tensor: 'activated', shape: shape(D_FF), dtype: 'float32' } },
        params: {},
      },
    ],
    outputs: ['activated'],
  };
}

/** A minimal safetensors file, written here so the test needs no exporter. */
function safetensors(entries) {
  const header = {};
  const payloads = [];
  let offset = 0;
  for (const [name, { shape, values }] of Object.entries(entries)) {
    const payload = new Uint8Array(Float32Array.from(values).buffer);
    header[name] = { dtype: 'F32', shape, data_offsets: [offset, offset + payload.length] };
    payloads.push(payload);
    offset += payload.length;
  }
  let headerBytes = new TextEncoder().encode(JSON.stringify(header));
  // The header length is 8-byte aligned; safetensors pads with spaces.
  const padding = (8 - (headerBytes.length % 8)) % 8;
  if (padding) {
    const padded = new Uint8Array(headerBytes.length + padding);
    padded.set(headerBytes);
    padded.fill(0x20, headerBytes.length);
    headerBytes = padded;
  }
  const out = new Uint8Array(8 + headerBytes.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  out.set(headerBytes, 8);
  let cursor = 8 + headerBytes.length;
  for (const payload of payloads) { out.set(payload, cursor); cursor += payload.length; }
  return out;
}

/** Deterministic values, so both sides see identical input. */
function ramp(count, start) {
  return Array.from({ length: count }, (_, index) => (start + index) / 64);
}

function fixtureWeights() {
  return safetensors({
    'ln.weight': { shape: [D_MODEL], values: ramp(D_MODEL, 64) },
    'ln.bias': { shape: [D_MODEL], values: ramp(D_MODEL, 1) },
    'fc.weight': { shape: [D_FF, D_MODEL], values: ramp(D_FF * D_MODEL, 3) },
    'fc.bias': { shape: [D_FF], values: ramp(D_FF, 7) },
  });
}

async function authoringHostWithMemory() {
  const { loadModelControlWasmDispatchFactory, ModelControlWasmDispatchFactory } = await import('../ts/core/ModelControlWasm.js');
  const { VxQuantizationServiceClient } = await import('../runtime/generated/typescript/volvoxai_ffi.js');
  const pb = await import('../runtime/generated/typescript/volvoxai_lite.js');
  const { REFUSING_GPU_BRIDGE } = await import('../ts/backends/WebGPUHostBridge.js');
  const bridge = { imports: REFUSING_GPU_BRIDGE, attach() {}, async waitForCompletion() {} };
  const source = await loadModelControlWasmDispatchFactory(MODULE_PATH, bridge);
  let memory;
  class ObservedFactory extends ModelControlWasmDispatchFactory {
    instantiate(wakeup) { const instance = super.instantiate(wakeup); memory = instance.exports.memory; return instance; }
  }
  const owner = new ObservedFactory(source.module, bridge).create();
  const api = new VxQuantizationServiceClient(reportTransport(owner));
  return {
    memory,
    host: {
      async author(graph, weightShards = []) {
        const result = (await api.authorPtqTemplate(new pb.AuthorPtqTemplateRequest({ sourceGraph: graph, weightShards })));
        assert.equal(result.report.status, pb.NativeStatus.NATIVE_STATUS_OK, result.report.message);
        return { ...result, quantizedNodes: Number(result.quantizedNodes), retainedFloatNodes: Number(result.retainedFloatNodes) };
      },
      release: async () => (await owner.close()),
    },
  };
}
async function authorShardsInWasm(graph, shards) {
  const { host } = await authoringHostWithMemory();
  try { return (await host.author(graph, shards)); } finally { (await host.release()); }
}
async function authorInWasm(graph, weights) { return authorShardsInWasm(graph, [weights]); }

function geluChainGraph(nodeCount) {
  let input = 'input';
  const nodes = [];
  for (let index = 0; index < nodeCount; index++) {
    const output = `hidden_${index}`;
    nodes.push({
      id: `gelu_${index}`,
      opType: 'GELU',
      inputs: { input },
      outputs: { out: { tensor: output, shape: [1, D_MODEL], dtype: 'float32' } },
      params: {},
    });
    input = output;
  }
  return new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { input: { shape: [1, D_MODEL], dtype: 'float32' } },
    nodes,
    outputs: [input],
  }));
}

/** The layout a C library gives a double, from JavaScript's digits. */
function authorNatively(directory, graph, weights) {
  const graphPath = join(directory, 'graph.json');
  const weightsPath = join(directory, 'model.safetensors');
  const templatePath = join(directory, 'native.json');
  writeFileSync(graphPath, graph);
  writeFileSync(weightsPath, weights);
  execFileSync(NATIVE_PATH, [graphPath, weightsPath, templatePath], { stdio: 'pipe' });
  return readFileSync(templatePath, 'utf8');
}

test('WebAssembly and native author the identical template', async () => {
  const graph = new TextEncoder().encode(graphText());
  const weights = fixtureWeights();
  const directory = mkdtempSync(join(tmpdir(), 'volvoxai-ptq-wasm-'));
  try {
    const authored = await authorInWasm(graph, weights);
    const native = authorNatively(directory, graph, weights);

    const fromWasm = new TextDecoder().decode(authored.templateGraph);
    // The native path appends a trailing newline when it writes the file;
    // that is the file, not the template.
    assert.equal(fromWasm, native.replace(/\n$/, ''),
      'the two builds rendered different template bytes');
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('the float parameter survives the crossing exactly', async () => {
  // eps is stored as F32 and written back as the shortest decimal that reads
  // as the same double — sixteen digits of it. A formatter that is merely
  // close produces a template that parses to a different number, and this is
  // the value that catches it.
  const graph = new TextEncoder().encode(graphText());
  const authored = await authorInWasm(graph, fixtureWeights());
  const template = JSON.parse(new TextDecoder().decode(authored.templateGraph));
  const layerNorm = template.nodes.find((node) => node.id === 'node_0');

  assert.equal(layerNorm.opType, 'QLayerNorm');
  assert.equal(layerNorm.params.eps, Math.fround(EPS));
  assert.equal(layerNorm.params.d_model, D_MODEL);
});

test('the module reports what it authored', async () => {
  const graph = new TextEncoder().encode(graphText());
  const authored = await authorInWasm(graph, fixtureWeights());

  assert.equal(authored.quantizedNodes, 3);
  assert.equal(authored.retainedFloatNodes, 0);
  assert.deepEqual(authored.observers.map((observer) => observer.tensorName),
    ['hidden', 'normed', 'wide', 'activated']);
  assert.equal(authored.layers.length, 1);
  assert.equal(authored.layers[0].nodeId, 'node_1');
  assert.match(authored.layers[0].packedWeightName, /^__ptq__\.[0-9a-f]{20}\.weight$/);
});

test('the browser path adds no fixed weight-shard limit', async () => {
  const graph = new TextEncoder().encode(graphText());
  const emptyShard = safetensors({});
  const shards = Array.from({ length: 16 }, () => emptyShard);
  shards.push(fixtureWeights());

  const authored = await authorShardsInWasm(graph, shards);
  assert.equal(authored.quantizedNodes, 3);
  assert.equal(authored.layers.length, 1);
});

test('repeated authoring reuses the full C owner memory', async () => {
  const { host, memory } = await authoringHostWithMemory();
  const graph = geluChainGraph(100);
  let firstCallBytes = 0;
  for (let call = 0; call < 50; call++) {
    const authored = (await host.author(graph));
    assert.equal(authored.quantizedNodes, 100);
    if (call === 0) firstCallBytes = memory.buffer.byteLength;
    assert.ok(
      memory.buffer.byteLength <= firstCallBytes + 65_536,
      `C owner memory grew from ${firstCallBytes} to ${memory.buffer.byteLength} bytes`,
    );
  }
  (await host.release());
});

test('non-finite F32 parameters are refused before template emission', async () => {
  const graph = new TextEncoder().encode(graphText().replace(EPS_TEXT, '-1e400'));
  await assert.rejects(
    authorInWasm(graph, fixtureWeights()),
    /must be finite and representable as F32/,
  );

  const directory = mkdtempSync(join(tmpdir(), 'volvoxai-ptq-nonfinite-'));
  try {
    let nativeFailure;
    try {
      authorNatively(directory, graph, fixtureWeights());
    } catch (error) {
      nativeFailure = error;
    }
    assert.ok(nativeFailure, 'native authoring unexpectedly accepted -1e400');
    assert.match(
      String(nativeFailure.stderr),
      /must be finite and representable as F32/,
    );
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('every float parameter renders the same as natively', async () => {
  // One value proves the path works; a spread of them is what catches a
  // formatter that is right in the common case and off by one somewhere else.
  // These are the magnitudes quantization parameters take — epsilons, scales,
  // and a few deliberately awkward mantissas — and each is rendered as the
  // shortest decimal that reads back as the same double, which is where a
  // hand-written renderer disagrees with a C library if it is going to.
  const values = [
    '1e-05', '1.5e-05', '1e-06', '1e-12', '9.999999e-06', '0.125',
    '0.1', '0.3', '1.0000000000000002', '3.141592653589793',
    '2.2250738585072014e-08', '6.103515625e-05', '1.234567890123456e-07',
    '7.7e-07', '5e-324',
  ];
  const directory = mkdtempSync(join(tmpdir(), 'volvoxai-ptq-eps-'));
  try {
    for (const text of values) {
      const document = fixtureGraph();
      document.nodes[0].params.eps = Number(text);
      const graph = new TextEncoder().encode(
        JSON.stringify(document, null, 1)
          .replace(`"eps": ${JSON.stringify(Number(text))}`, `"eps": ${text}`));

      const authored = await authorInWasm(graph, fixtureWeights());
      const native = authorNatively(directory, graph, fixtureWeights());
      assert.equal(
        new TextDecoder().decode(authored.templateGraph),
        native.replace(/\n$/, ''),
        `eps ${text} rendered differently in WebAssembly`);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('retained cJSON numbers preserve native 15-then-17 digit rendering', async () => {
  // The first value is one native cJSON accepts at its 15-digit pass. The old
  // freestanding formatter rounded its last digit down instead. The other two
  // cover the large-integer division and extreme-exponent fallback paths that
  // used to accumulate rounding error before cJSON reached its 17-digit pass.
  const values = [
    '-4.62054322728846e-156',
    '1.234567890123456e20',
    '1e-100',
  ];
  const directory = mkdtempSync(join(tmpdir(), 'volvoxai-ptq-retained-number-'));
  try {
    for (const text of values) {
      const document = fixtureGraph();
      document.nodes[2].opType = 'RetainedNumericMetadata';
      document.nodes[2].params = { retained_number: Number(text) };
      const renderedByJson = JSON.stringify(Number(text));
      const graph = new TextEncoder().encode(
        JSON.stringify(document, null, 1).replace(
          `"retained_number": ${renderedByJson}`,
          `"retained_number": ${text}`,
        ),
      );

      const authored = await authorInWasm(graph, fixtureWeights());
      const native = authorNatively(directory, graph, fixtureWeights());
      const browserTemplate = new TextDecoder().decode(authored.templateGraph);
      assert.equal(
        browserTemplate,
        native.replace(/\n$/, ''),
        `retained number ${text} rendered differently in WebAssembly`,
      );
      const retained = JSON.parse(browserTemplate).nodes
        .find(({ id }) => id === 'node_2');
      assert.equal(retained.params.retained_number, Number(text));
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('control characters are escaped as valid JSON and remain byte-identical', async () => {
  const document = fixtureGraph();
  document.nodes[2].id = 'gelu\u001fnode';
  const graph = new TextEncoder().encode(JSON.stringify(document, null, 1));
  const weights = fixtureWeights();
  const directory = mkdtempSync(join(tmpdir(), 'volvoxai-ptq-control-'));
  try {
    const authored = await authorInWasm(graph, weights);
    const rendered = new TextDecoder().decode(authored.templateGraph);
    assert.ok(
      JSON.parse(rendered).nodes.some(({ id }) => id === document.nodes[2].id),
      'the escaped node id must survive authoring',
    );
    assert.equal(rendered, authorNatively(directory, graph, weights).replace(/\n$/, ''));
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('a refused graph says why', async () => {
  const document = fixtureGraph();
  document.nodes[1].opType = 'Embedding';
  const graph = new TextEncoder().encode(JSON.stringify(document, null, 1));
  await assert.rejects(
    async () => authorInWasm(graph, fixtureWeights()),
    (error) => /Embedding/.test(String(error.message || error)),
  );
});
