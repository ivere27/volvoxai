// Run a model through the built JS bundle (pure-JS 'cpu-js' or 'wasm' tier) in Node,
// and return named output tensors + wall-clock timing. Uses the same file://
// fetch shim as bin/volvox.js so the browser bundle runs headless.
import fs from 'fs';
import path from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import { createRuntimeEvidence } from './artifact.mjs';
import { readTensor, seededUint8, seededFloat32 } from './tensorio.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

function installFileFetchShim() {
  if (globalThis.__volvoxParityShim) return;
  const nf = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const href = typeof url === 'string' ? url : url?.url;
    if (href && href.startsWith('file://')) {
      const data = await fs.promises.readFile(fileURLToPath(href));
      return new Response(data, { status: 200 });
    }
    return nf(url, init);
  };
  globalThis.__volvoxParityShim = true;
}

export function version() {
  return JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
}

function assertRequestedBackend(compiled, requested) {
  if (compiled?.backend !== requested) {
    throw new Error(`requested backend '${requested}' compiled on '${compiled?.backend || 'unknown'}'`);
  }
}

function checkedHostOutput(graph, name, value, backend, concreteDescriptor) {
  const tensor = typeof graph.getTensor === 'function'
    ? graph.getTensor(name)
    : graph.tensors?.[name];
  if (!tensor) throw new Error(`${backend}: graph has no output tensor "${name}"`);
  if (!ArrayBuffer.isView(value)) {
    throw new Error(`${backend}: output "${name}" is not a host typed array`);
  }
  if (!concreteDescriptor || concreteDescriptor.dtype !== tensor.dtype ||
      !Array.isArray(concreteDescriptor.shape) ||
      concreteDescriptor.shape.some((dimension) =>
        !Number.isSafeInteger(dimension) || dimension <= 0)) {
    throw new Error(`${backend}: output "${name}" has no valid concrete result descriptor`);
  }
  const dtypeBytes = { float32: 4, int32: 4, int8: 1, uint8: 1 }[
    concreteDescriptor.dtype
  ];
  if (!dtypeBytes) throw new Error(`${backend}: output "${name}" has unsupported dtype`);
  const expectedBytes = concreteDescriptor.shape.reduce(
    (size, dimension) => size * dimension,
    dtypeBytes,
  );
  if (!Number.isSafeInteger(expectedBytes) || value.byteLength !== expectedBytes) {
    throw new Error(
      `${backend}: output "${name}" has ${value.byteLength} bytes, expected ${expectedBytes}`,
    );
  }
  return value instanceof Float32Array ? value : Float32Array.from(value);
}

function sameBytes(left, right) {
  if (!ArrayBuffer.isView(left) || !ArrayBuffer.isView(right) ||
      left.byteLength !== right.byteLength) return false;
  const a = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
  const b = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) return false;
  }
  return true;
}

export function concreteExecutionInputs(snapshot, values) {
  const shaped = {};
  for (const name of snapshot.inputNames) {
    const value = values[name];
    if (value && typeof value === 'object' && ArrayBuffer.isView(value.data) &&
        Array.isArray(value.shape)) {
      shaped[name] = value;
      continue;
    }
    if (!ArrayBuffer.isView(value)) {
      throw new Error(`input '${name}' must provide typed storage`);
    }
    const descriptor = snapshot.staticShapePlan?.tensors?.[name];
    if (!descriptor || !Array.isArray(descriptor.shape)) {
      throw new Error(
        `dynamic input '${name}' requires an explicit { data, shape } fixture`,
      );
    }
    shaped[name] = { data: value, shape: [...descriptor.shape] };
  }
  return shaped;
}

export function runtimeFailureMessage(error) {
  const summary = String(error?.message || error);
  const candidates = Array.isArray(error?.report?.candidates)
    ? error.report.candidates
    : [];
  const details = candidates.flatMap((candidate) => {
    if (!candidate || typeof candidate !== 'object' ||
        typeof candidate.message !== 'string' || !candidate.message.trim()) return [];
    const backend = typeof candidate.backend === 'string' ? candidate.backend : 'backend';
    const outcome = typeof candidate.outcome === 'string' ? candidate.outcome : 'failed';
    return [`${backend} ${outcome}: ${candidate.message.trim()}`];
  });
  return details.length === 0 ? summary : `${summary} ${details.join(' | ')}`;
}

/**
 * Read every declared output, prove each read is a fresh copy, close the context,
 * and prove the still-open result remains readable before closing it.
 */
export async function captureStableResult(graph, result, context, backend, outputNames) {
  const names = [...outputNames];
  if (names.length === 0) throw new Error(`${backend}: graph has no declared outputs`);
  if (result?.report?.contextId !== context?.id || result?.report?.backend !== backend) {
    throw new Error(`${backend}: execution report is not bound to the selected context`);
  }
  const baselines = new Map();
  const outputs = {};
  const descriptors = [];
  let resultClosed = false;
  try {
    for (const name of names) {
      const tensorResult = result.output(name);
      const first = await tensorResult.read();
      outputs[name] = checkedHostOutput(graph, name, first, backend, tensorResult);
      baselines.set(name, first);
      descriptors.push({
        name,
        shape: [...tensorResult.shape],
        dtype: tensorResult.dtype,
        location: tensorResult.location,
        byteLength: first.byteLength,
      });

      const second = await tensorResult.read();
      if (second === first || second.buffer === first.buffer || !sameBytes(first, second)) {
        throw new Error(`${backend}: output "${name}" did not return a fresh stable read`);
      }
    }

    await context.close();
    for (const name of names) {
      const baseline = baselines.get(name);
      const afterClose = await result.output(name).read();
      if (afterClose === baseline || afterClose.buffer === baseline.buffer ||
          !sameBytes(baseline, afterClose)) {
        throw new Error(`${backend}: output "${name}" changed after context close`);
      }
    }
    await result.close();
    resultClosed = true;
    return {
      outputs,
      stableResult: {
        outputs: descriptors,
        freshCallerOwnedReads: true,
        readableAfterContextClose: true,
        contextClosedBeforeResult: true,
        resultClosedAfterVerification: true,
      },
    };
  } finally {
    if (!resultClosed) await result.close();
  }
}

// Build explicit shaped views for policy inputs. Seeded inputs are also written
// to `fixtureDir` so native tiers can read the exact same physical bytes.
export function buildInputs(model, fixtureDir) {
  const inputs = {};
  for (const inp of model.inputs || []) {
    if (!Array.isArray(inp.shape) || inp.shape.length === 0 ||
        inp.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) {
      throw new Error(`input ${inp.name}: policy requires one explicit positive shape`);
    }
    const elements = inp.shape.reduce((product, dimension) => product * dimension, 1);
    if (!Number.isSafeInteger(elements)) {
      throw new Error(`input ${inp.name}: policy shape element count is not safely representable`);
    }
    let data;
    if (inp.file) {
      data = readTensor(path.join(ROOT, inp.file), inp.dtype);
    } else if (inp.gen === 'seededU8' || inp.gen === 'seededF32') {
      const isF32 = inp.gen === 'seededF32';
      const arr = isF32
        ? seededFloat32(elements, inp.seed, inp.lo ?? 0, inp.hi ?? 1)
        : seededUint8(elements, inp.seed);
      data = arr;
      if (fixtureDir) {
        fs.mkdirSync(fixtureDir, { recursive: true });
        fs.writeFileSync(path.join(fixtureDir, `${inp.name}.${isF32 ? 'f32' : 'u8'}`),
          Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
      }
    } else {
      throw new Error(`input ${inp.name}: need "file" or "gen"`);
    }
    if (data.length !== elements) {
      throw new Error(
        `input ${inp.name}: ${data.length} stored elements do not match shape [${inp.shape}]`,
      );
    }
    inputs[inp.name] = Object.freeze({
      data,
      shape: Object.freeze([...inp.shape]),
    });
  }
  return inputs;
}

export async function runJs(model, backend, fixtureDir) {
  installFileFetchShim();
  const ver = version();
  const module = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
  const wasmPath = path.join(ROOT, 'dist', ver, 'volvoxai.wasm');
  const modelUrl = pathToFileURL(path.join(ROOT, model.dir, 'model.safetensors')).href;
  const snapshot = module.Model.capture(
    await module.ModelLoader.load(modelUrl),
  );
  const runtime = await module.VolvoxAI.createRuntime({ backends: [backend], wasmUrl: wasmPath });
  let compiled;
  let context;
  try {
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    assertRequestedBackend(compiled, backend);
    context = await compiled.createContext();
    const inputs = concreteExecutionInputs(snapshot, buildInputs(model, fixtureDir));

    const warmup = await context.execute(inputs);
    await warmup.close();
    const t0 = performance.now();
    const result = await context.execute(inputs);
    const ms = performance.now() - t0;
    const execution = result.report;
    const { outputs, stableResult } = await captureStableResult(
      snapshot.graph,
      result,
      context,
      backend,
      model.outputs.map((output) => output.name),
    );
    const runtimeEvidence = createRuntimeEvidence({
      compilation: compiled.report,
      execution,
      stableResult,
    });
    return { outputs, ms, runtimeEvidence };
  } finally {
    await context?.close();
    await compiled?.close();
    await runtime.close();
  }
}

// Run a synthetic single-node package (a directory with graph.json +
// model.safetensors) through a JS backend and return the output tensor map.
export async function runPackage(pkgDir, backend, inputs) {
  installFileFetchShim();
  const ver = version();
  const module = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
  const wasmUrl = path.join(ROOT, 'dist', ver, 'volvoxai.wasm');
  const url = pathToFileURL(path.join(pkgDir, 'model.safetensors')).href;
  const snapshot = module.Model.capture(
    await module.ModelLoader.load(url),
  );
  const runtime = await module.VolvoxAI.createRuntime({ backends: [backend], wasmUrl });
  let compiled;
  let context;
  try {
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    assertRequestedBackend(compiled, backend);
    context = await compiled.createContext();
    const result = await context.execute(concreteExecutionInputs(snapshot, inputs));
    const execution = result.report;
    const { outputs, stableResult } = await captureStableResult(
      snapshot.graph,
      result,
      context,
      backend,
      snapshot.outputNames,
    );
    return {
      outputs,
      runtimeEvidence: createRuntimeEvidence({
        compilation: compiled.report,
        execution,
        stableResult,
      }),
    };
  } finally {
    await context?.close();
    await compiled?.close();
    await runtime.close();
  }
}

export { ROOT };
