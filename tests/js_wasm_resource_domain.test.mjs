import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { runInNewContext } from 'node:vm';

import { createBackendCompileInput } from '../ts/backends/BackendProvider.js';
import { WasmBackendProvider } from '../ts/backends/WasmBackendProvider.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';
const allocatorLimitBytes = 0x7ffffff0;
const planCacheBytes = 1024 * 1024;
const growthHeadroomBytes = 64 * 1024 * 1024;
const wasmPageBytes = 64 * 1024;

async function buildForwardWasm(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot, maxBuffer: 1024 * 1024 });
}

function identitySnapshot(extent) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [extent] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [extent] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function addSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 2 }, S: { min: 1, max: 3 } },
    inputs: {
      a: { dtype: 'float32', shape: ['B', 'S'] },
      b: { dtype: 'float32', shape: ['B', 'S'] },
    },
    nodes: [{
      id: 'add', opType: 'Add', inputs: { a: 'a', b: 'b' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 'S'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function weightedMultiOutputSnapshot() {
  const weight = { name: 'w', dtype: 'float32', shape: [2] };
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [2] } },
    nodes: [{
      id: 'add', opType: 'Add', inputs: { a: 'x', b: 'w' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2] } },
      params: {},
    }, {
      id: 'identity', opType: 'Identity', inputs: { input: 'y' },
      outputs: { out: { tensor: 'z', dtype: 'float32', shape: [2] } },
      params: {},
    }],
    outputs: ['y', 'z'],
  }, [weight]);
  return Model.capture({
    graph,
    weights: { w: { ...weight, data: Float32Array.of(1, 2) } },
  });
}

function ssmPairSnapshot() {
  const definitions = [
    { name: 'A', dtype: 'float32', shape: [4, 5] },
    { name: 'scan_b', dtype: 'float32', shape: [5] },
    { name: 'scan_c', dtype: 'float32', shape: [5] },
  ];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 }, S: { min: 1, max: 8 } },
    inputs: {
      input: { dtype: 'float32', shape: ['B', 'S', 4] },
      delta: { dtype: 'float32', shape: ['B', 'S', 4] },
    },
    nodes: ['SSMScan', 'SelectiveScan'].map((opType, index) => ({
      id: `scan${index}`,
      opType,
      inputs: { input: 'input', delta: 'delta', A: 'A', B: 'scan_b', C: 'scan_c' },
      outputs: {
        out: { tensor: `out${index}`, dtype: 'float32', shape: ['B', 'S', 4] },
        state: { tensor: `state${index}`, dtype: 'float32', shape: ['B', 4, 5] },
      },
      params: {},
    })),
    outputs: ['out0', 'out1'],
  }, definitions);
  const weights = Object.fromEntries(definitions.map((definition) => [definition.name, {
    ...definition,
    data: new Float32Array(definition.shape.reduce((product, value) => product * value, 1)),
  }]));
  return Model.capture({ graph, weights });
}

function moeBankSnapshot() {
  const descriptor = { name: 'experts', dtype: 'float32', shape: [4, 2, 1] };
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { F: { min: 1, max: 8 } },
    banks: { experts: 'F' },
    inputs: {
      x: { dtype: 'float32', shape: [1, 2] },
      route_indices: { dtype: 'float32', shape: [1, 1] },
      route_weights: { dtype: 'float32', shape: [1, 1] },
    },
    nodes: [{
      id: 'mix', opType: 'MoELinear',
      inputs: {
        input: 'x', expert_weight: 'experts',
        route_indices: 'route_indices', route_weights: 'route_weights',
      },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 1] } },
      params: {},
    }],
    outputs: ['y'],
  }, [descriptor]);
  return Model.capture({
    graph,
    weights: {
      experts: { ...descriptor, data: new Float32Array(8) },
    },
  });
}

function orderedBankSnapshot() {
  const definitions = [
    { name: 'a_large', dtype: 'float32', shape: [5, 1, 1] },
    { name: 'z_small', dtype: 'float32', shape: [2, 1, 1] },
  ];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { F: { min: 1, max: 5 } },
    banks: { a_large: 'F', z_small: 'F' },
    inputs: {
      x: { dtype: 'float32', shape: [1, 1] },
      route_indices: { dtype: 'float32', shape: [1, 1] },
      route_weights: { dtype: 'float32', shape: [1, 1] },
    },
    nodes: definitions.map((definition, index) => ({
      id: `mix${index}`,
      opType: 'MoELinear',
      inputs: {
        input: 'x', expert_weight: definition.name,
        route_indices: 'route_indices', route_weights: 'route_weights',
      },
      outputs: {
        out: { tensor: `y${index}`, dtype: 'float32', shape: [1, 1] },
      },
      params: {},
    })),
    outputs: ['y0', 'y1'],
  }, definitions);
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [definition.name, {
      ...definition,
      data: new Float32Array(definition.shape[0]),
    }])),
  });
}

function typedData(dtype, values) {
  if (dtype === 'float32') return Float32Array.from(values);
  if (dtype === 'int32') return Int32Array.from(values);
  if (dtype === 'int8') return Int8Array.from(values);
  return Uint8Array.from(values);
}

function parallelQConvSnapshot() {
  const definitions = [
    { name: 'w', dtype: 'int8', shape: [8, 1, 1, 1], values: new Array(8).fill(1) },
    { name: 'bias', dtype: 'int32', shape: [8], values: new Array(8).fill(0) },
    { name: 'sx', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zx', dtype: 'int8', shape: [1], values: [0] },
    { name: 'sw', dtype: 'float32', shape: [8], values: new Array(8).fill(1) },
    { name: 'zw', dtype: 'int8', shape: [8], values: new Array(8).fill(0) },
    { name: 'sy', dtype: 'float32', shape: [1], values: [1] },
    { name: 'zy', dtype: 'int8', shape: [1], values: [0] },
  ];
  const quantizedOutput = (tensor) => ({
    tensor, dtype: 'int8', shape: ['B', 'H', 'W', 8],
  });
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 2 }, H: { min: 1, max: 4 }, W: { min: 1, max: 5 },
    },
    inputs: { x: { dtype: 'int8', shape: ['B', 'H', 'W', 1] } },
    nodes: [0, 1].map((index) => ({
      id: `conv${index}`, opType: 'QConv2D',
      inputs: { input: 'x', weight: 'w', bias: 'bias' },
      outputs: { out: quantizedOutput(`y${index}`) },
      params: {
        data_layout: 'NHWC', weight_layout: 'OHWI', stride: [1, 1],
        dilation: [1, 1], pads: [0, 0, 0, 0],
      },
    })),
    outputs: ['y0', 'y1'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        x: { scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx' },
        w: { scheme: 'per_axis', axis: 0, scale_tensor: 'sw', zero_point_tensor: 'zw' },
        y0: { scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy' },
        y1: { scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy' },
      },
    },
  }, definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })));
  const weights = Object.fromEntries(definitions.map((definition) => [definition.name, {
    name: definition.name, dtype: definition.dtype, shape: definition.shape,
    data: typedData(definition.dtype, definition.values),
  }]));
  return Model.capture({
    graph,
    weights,
    quantizationByTensor: {
      x: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      w: {
        scheme: 'per_axis', axis: 0,
        scales: new Array(8).fill(1), zero_points: new Array(8).fill(0),
      },
      y0: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      y1: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    },
  });
}

function syntheticRuntimeEngine({
  heapBase = new WebAssembly.Global({ value: 'i32', mutable: false }, 16),
  heapMark = heapBase?.value,
  memoryByteLength = wasmPageBytes,
  omitHeapBase = false,
  exports: extraExports = {},
} = {}) {
  const exports = {
    memory: memoryByteLength === wasmPageBytes
      ? new WebAssembly.Memory({ initial: 1 })
      : { buffer: { byteLength: memoryByteLength } },
    wasm_runtime_abi_version: () => 1,
    alloc_bytes: () => 0,
    reset_heap: () => {},
    heap_mark: () => heapMark,
    heap_rewind: () => 1,
    ...extraExports,
  };
  if (!omitHeapBase) exports.__heap_base = heapBase;
  return new WasmEngine({ instance: { exports } });
}

test('WASM runtime ABI requires a positive aligned pristine linker heap base', async () => {
  for (const [options, expected] of [
    [{ omitHeapBase: true }, /invalid positive aligned __heap_base/],
    [{ heapBase: new WebAssembly.Global({ value: 'i32', mutable: false }, 0) },
      /invalid positive aligned __heap_base/],
    [{ heapBase: new WebAssembly.Global({ value: 'i32', mutable: false }, 3) },
      /invalid positive aligned __heap_base/],
    [{ heapBase: new WebAssembly.Global({ value: 'i32', mutable: false }, 16), heapMark: 32 },
      /allocator does not start at its exported __heap_base/],
  ]) {
    const source = syntheticRuntimeEngine(options);
    const provider = new WasmBackendProvider(source);
    try {
      await assert.rejects(
        provider.compile(createBackendCompileInput(identitySnapshot(1)), {
          operatorFallback: 'forbid',
        }),
        expected,
      );
    } finally {
      await provider.close();
      source.dispose();
    }
  }
});

test('WASM ABI validation accepts a genuine heap-base Global from another realm', async () => {
  const heapBase = runInNewContext(
    "new WebAssembly.Global({ value: 'i32', mutable: false }, 16)",
  );
  assert.equal(heapBase instanceof WebAssembly.Global, false);
  const source = syntheticRuntimeEngine({ heapBase, heapMark: 16 });
  const provider = new WasmBackendProvider(source);
  let compiled;
  try {
    compiled = await provider.compile(createBackendCompileInput(identitySnapshot(1)), {
      operatorFallback: 'forbid',
    });
    assert.equal(compiled.compilationEvidence.shapeDomain.staticPrefixBytes, 16);
  } finally {
    await compiled?.close();
    await provider.close();
    source.dispose();
  }
});

test('WASM resource proof accounts initial memory and rejects an oversized initial extent', async () => {
  const initialMemoryBytes = 256 * 1024 * 1024;
  const source = syntheticRuntimeEngine({ memoryByteLength: initialMemoryBytes });
  const provider = new WasmBackendProvider(source);
  let compiled;
  try {
    compiled = await provider.compile(createBackendCompileInput(identitySnapshot(1)), {
      operatorFallback: 'forbid',
    });
    const resources = compiled.compilationEvidence.shapeDomain;
    assert.equal(resources.initialMemoryBytes, initialMemoryBytes);
    assert.equal(resources.maximumLinearResidentBytes, initialMemoryBytes);
    assert.equal(resources.maximumInputBytes, 4);
    assert.equal(resources.maximumResultBytes, 4);
    assert.equal(resources.maximumResidentBytes, initialMemoryBytes + planCacheBytes + 4,
      'linear memory, one result snapshot, and the host plan cache are additive');
    assert.equal(resources.maximumDecodeResidentBytes,
      initialMemoryBytes + planCacheBytes + 12,
      'decode also charges old and replacement public-input snapshots');
  } finally {
    await compiled?.close();
    await provider.close();
    source.dispose();
  }

  const oversized = syntheticRuntimeEngine({ memoryByteLength: 0x80000000 });
  const oversizedProvider = new WasmBackendProvider(oversized);
  try {
    await assert.rejects(
      oversizedProvider.compile(createBackendCompileInput(identitySnapshot(1)), {
        operatorFallback: 'forbid',
      }),
      /bounded-domain resources require.*exceeding/,
    );
  } finally {
    await oversizedProvider.close();
    oversized.dispose();
  }
});

test('WASM resource proof includes host weights, exact bank staging, and results', async () => {
  const initialMemoryBytes = 256 * 1024 * 1024;
  const weightedSource = syntheticRuntimeEngine({ memoryByteLength: initialMemoryBytes });
  const weightedProvider = new WasmBackendProvider(weightedSource);
  let weightedCompiled;
  try {
    weightedCompiled = await weightedProvider.compile(
      createBackendCompileInput(weightedMultiOutputSnapshot()),
      { operatorFallback: 'forbid' },
    );
    const resources = weightedCompiled.compilationEvidence.shapeDomain;
    assert.equal(resources.maximumLinearResidentBytes, initialMemoryBytes);
    assert.equal(resources.weightBytes, 8);
    assert.equal(resources.maximumBankResidentBytes, 0);
    assert.equal(resources.maximumBankStagingTransactionBytes, 0);
    assert.equal(resources.maximumInputBytes, 8);
    assert.equal(resources.maximumResultBytes, 16);
    assert.equal(resources.maximumResidentBytes,
      initialMemoryBytes + planCacheBytes + 2 * 8 + 16,
      'snapshot/context weights and both result snapshots coexist');
    assert.equal(resources.maximumDecodeResidentBytes,
      initialMemoryBytes + planCacheBytes + 2 * 8 + 16 + 2 * 8);
  } finally {
    await weightedCompiled?.close();
    await weightedProvider.close();
    weightedSource.dispose();
  }

  const source = syntheticRuntimeEngine({ memoryByteLength: initialMemoryBytes });
  const provider = new WasmBackendProvider(source);
  let compiled;
  try {
    compiled = await provider.compile(createBackendCompileInput(orderedBankSnapshot()), {
      operatorFallback: 'forbid',
    });
    const resources = compiled.compilationEvidence.shapeDomain;
    assert.equal(resources.maximumLinearResidentBytes, initialMemoryBytes);
    assert.equal(resources.weightBytes, 28);
    assert.equal(resources.maximumBankResidentBytes, 28);
    assert.equal(resources.maximumBankStagingTransactionBytes, 40,
      'canonical a_large then z_small staging peaks at max(2*20, 20 + 2*8)');
    assert.equal(resources.maximumInputBytes, 12);
    assert.equal(resources.maximumResultBytes, 8,
      'both public result snapshots remain live together');
    assert.equal(resources.maximumResidentBytes,
      initialMemoryBytes + planCacheBytes + 2 * 28 + 40,
      'the initial-memory-dominant bank staging transaction exceeds steady B + R');
    assert.equal(resources.maximumDecodeResidentBytes,
      initialMemoryBytes + planCacheBytes + 2 * 28 + 28 + 8 + 2 * 12);
    assert.equal(resources.decodeContextSupported, true);
    assert.equal(resources.residentLimitBytes, allocatorLimitBytes);
  } finally {
    await compiled?.close();
    await provider.close();
    source.dispose();
  }
});

test('WASM host-result ceiling rejects compile and decode ceiling rejects before fork', async () => {
  const resultBoundaryExtent = 180_000_000;
  const resultBytes = resultBoundaryExtent * 4;
  const oldLinearOnlyBound = 16 + 2 * resultBytes + growthHeadroomBytes + planCacheBytes;
  assert.ok(oldLinearOnlyBound < allocatorLimitBytes);
  assert.ok(oldLinearOnlyBound + resultBytes > allocatorLimitBytes);
  const resultSource = syntheticRuntimeEngine();
  const resultProvider = new WasmBackendProvider(resultSource);
  try {
    await assert.rejects(
      resultProvider.compile(createBackendCompileInput(identitySnapshot(resultBoundaryExtent)), {
        operatorFallback: 'forbid',
      }),
      /bounded-domain resources require.*exceeding/,
    );
  } finally {
    await resultProvider.close();
    resultSource.dispose();
  }

  const decodeBoundaryExtent = 125_000_000;
  const decodeSource = syntheticRuntimeEngine();
  const decodeProvider = new WasmBackendProvider(decodeSource);
  let compiled;
  try {
    compiled = await decodeProvider.compile(
      createBackendCompileInput(identitySnapshot(decodeBoundaryExtent)),
      { operatorFallback: 'forbid' },
    );
    const resources = compiled.compilationEvidence.shapeDomain;
    assert.ok(resources.maximumResidentBytes < resources.residentLimitBytes);
    assert.ok(resources.maximumDecodeResidentBytes > resources.residentLimitBytes);
    assert.equal(resources.decodeContextSupported, false);

    let forkCalls = 0;
    decodeSource.fork = async () => {
      forkCalls++;
      throw new Error('decode rejection must precede engine fork');
    };
    await assert.rejects(
      compiled.createContext({ decode: {} }),
      /retained decode resources require.*resident\/transaction limit/,
    );
    assert.equal(forkCalls, 0);
  } finally {
    await compiled?.close();
    await decodeProvider.close();
    decodeSource.dispose();
  }
});

test('WASM resource proof sums exact per-node persistent metadata maxima', async () => {
  for (const [snapshot, expected, reason] of [
    [addSnapshot(), 64, 'Add retains two separately aligned rank-two broadcast descriptors'],
    [ssmPairSnapshot(), 640, 'two scans retain independent maximum-domain F32 states'],
    [moeBankSnapshot(), 16, 'a four-slot bank admits a separately aligned global slot table'],
  ]) {
    const source = syntheticRuntimeEngine();
    const provider = new WasmBackendProvider(source);
    let compiled;
    try {
      compiled = await provider.compile(createBackendCompileInput(snapshot), {
        operatorFallback: 'forbid',
      });
      assert.equal(
        compiled.compilationEvidence.shapeDomain.maximumPersistentMetadataBytes,
        expected,
        reason,
      );
      assert.equal(compiled.compilationEvidence.shapeDomain.maximumSharedScratchBytes, 0);
    } finally {
      await compiled?.close();
      await provider.close();
      source.dispose();
    }
  }
});

test('WASM resource proof charges the linker static prefix exactly once', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-resource-proof-'));
  let source;
  let provider;
  let compiled;
  try {
    const wasmPath = join(directory, 'volvoxai.wasm');
    await buildForwardWasm(wasmPath);
    source = await WasmEngine.init(wasmPath);
    assert.ok(source);

    assert.ok(source.api.__heap_base instanceof WebAssembly.Global,
      'runtime ABI sidecars expose the linker-defined heap base');
    const staticPrefixBytes = Number(source.api.__heap_base.value);
    const initialMemoryBytes = source.mem.buffer.byteLength;
    assert.ok(Number.isSafeInteger(staticPrefixBytes) && staticPrefixBytes > 0);
    assert.equal(staticPrefixBytes % 16, 0);
    assert.equal(source.api.heap_mark(), staticPrefixBytes,
      'a fresh allocator starts immediately after the complete static/stack prefix');

    provider = new WasmBackendProvider(source);
    compiled = await provider.compile(createBackendCompileInput(identitySnapshot(4)), {
      operatorFallback: 'forbid',
    });
    const resources = compiled.compilationEvidence.shapeDomain;
    assert.equal(resources.staticPrefixBytes, staticPrefixBytes,
      'ordinary compile evidence reports the queried sidecar prefix');
    assert.equal(resources.initialMemoryBytes, initialMemoryBytes);
    assert.equal(resources.maximumTensorBytes, 16);
    assert.equal(resources.maximumPersistentMetadataBytes, 0);
    assert.equal(resources.maximumSharedScratchBytes, 0);
    assert.equal(resources.weightBytes, 0);
    assert.equal(resources.maximumBankResidentBytes, 0);
    assert.equal(resources.maximumBankStagingTransactionBytes, 0);
    assert.equal(resources.maximumInputBytes, 16);
    assert.equal(resources.maximumResultBytes, 16);
    const maximumLinearResidentBytes = Math.max(
      initialMemoryBytes,
      staticPrefixBytes + 32 + growthHeadroomBytes,
    );
    assert.equal(resources.maximumLinearResidentBytes, maximumLinearResidentBytes);
    assert.equal(resources.maximumResidentBytes,
      maximumLinearResidentBytes + planCacheBytes + 16,
      'linear memory, the two aligned tensors, one result, and the host plan cache are charged');
    assert.equal(resources.maximumDecodeResidentBytes,
      maximumLinearResidentBytes + planCacheBytes + 48);
    assert.equal(resources.resourceLimitBytes, allocatorLimitBytes);
    assert.equal(resources.residentLimitBytes, allocatorLimitBytes);

    await compiled.close();
    compiled = null;

    compiled = await provider.compile(createBackendCompileInput(parallelQConvSnapshot()), {
      operatorFallback: 'forbid',
    });
    assert.equal(compiled.compilationEvidence.shapeDomain.maximumPersistentMetadataBytes, 128,
      'two QConv nodes each retain two independently aligned per-axis arrays');
    assert.equal(compiled.compilationEvidence.shapeDomain.maximumSharedScratchBytes, 48,
      'parallel QConv nodes share one aligned maximum-domain im2col allocation');
    await compiled.close();
    compiled = null;

    /* Two aligned F32 tensors plus their result snapshot leave only 32 bytes
     * beneath the signed resident ceiling before the module prefix is counted.
     * Every real instance necessarily starts at __heap_base, so compilation
     * must reject the domain before context creation. */
    const nearLimitExtent = 173_277_180;
    const alignedTensorBytes = Math.ceil((nearLimitExtent * 4) / 16) * 16;
    const suffixWithoutStaticPrefix = alignedTensorBytes * 2 +
      nearLimitExtent * 4 + growthHeadroomBytes + planCacheBytes;
    assert.equal(allocatorLimitBytes - suffixWithoutStaticPrefix, 32);
    assert.ok(staticPrefixBytes > allocatorLimitBytes - suffixWithoutStaticPrefix);
    await assert.rejects(
      provider.compile(createBackendCompileInput(identitySnapshot(nearLimitExtent)), {
        operatorFallback: 'forbid',
      }),
      /bounded-domain resources require.*exceeding/,
    );
  } finally {
    await compiled?.close();
    await provider?.close();
    source?.dispose();
    await rm(directory, { recursive: true, force: true });
  }
});
