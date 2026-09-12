#!/usr/bin/env -S deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

function fail(message) {
  throw new Error(`[onnx-oracle/webgpu] ${message}`);
}

function parseArguments(argv) {
  const options = { requirePhysical: false };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    if (flag === '--require-physical') {
      if (options.requirePhysical) fail('--require-physical may be passed only once');
      options.requirePhysical = true;
      continue;
    }
    if (![
      '--bundle', '--wasm', '--package', '--inputs', '--out',
      '--next-inputs', '--invalid-inputs', '--changed-inputs',
    ].includes(flag)) {
      fail(`unknown argument ${flag}`);
    }
    const value = argv[++index];
    if (!value || value.startsWith('--')) fail(`${flag} requires a value`);
    const property = ({
      '--next-inputs': 'nextInputs',
      '--invalid-inputs': 'invalidInputs',
      '--changed-inputs': 'changedInputs',
    })[flag] ?? flag.slice(2);
    options[property] = path.resolve(value);
  }
  for (const name of ['bundle', 'wasm', 'package', 'inputs', 'out']) {
    if (!options[name]) fail(`--${name} is required`);
  }
  const lifecycleInputs = ['nextInputs', 'invalidInputs', 'changedInputs'];
  const lifecycleCount = lifecycleInputs.filter((name) => options[name]).length;
  if (lifecycleCount !== 0 && lifecycleCount !== lifecycleInputs.length) {
    fail('--next-inputs, --invalid-inputs, and --changed-inputs must be passed together');
  }
  return options;
}

const DTYPE = Object.freeze({
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
});

const SOFTWARE_ADAPTER = /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software rasterizer|microsoft basic render|cpu)\b/i;

function exactArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

function exactBytes(value) {
  return new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
}

function bytesEqual(left, right) {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index++) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function operationSucceeded(api, report, label) {
  if (report?.status !== api.pb.NativeStatus.NATIVE_STATUS_OK) {
    fail(`${label} failed with ${report?.code || report?.status}: ${report?.message || ''}`);
  }
}

function strictRoute(report, label) {
  if (report?.compilation?.tierFallbackUsed === true ||
      report?.route?.attested !== true ||
      report?.fallback?.operatorFallbackUsed === true) {
    fail(`${label} does not attest a strict no-fallback route`);
  }
}

function normalizedAdapterIdentity(info, required) {
  if (typeof info === 'string') {
    info = info.trim()
      ? Object.fromEntries(info.split(',').map((field) => {
          const separator = field.indexOf('=');
          return separator < 0
            ? ['description', field]
            : [field.slice(0, separator), field.slice(separator + 1)];
        }).filter(([, value]) => value))
      : null;
  }
  if (info == null || typeof info !== 'object' || Array.isArray(info)) {
    if (required) fail('physical WebGPU adapter identity is unavailable');
    return null;
  }
  const identity = Object.fromEntries(Object.entries(info)
    .filter(([, value]) => typeof value === 'string' && value.trim())
    .map(([key, value]) => [key, value.trim()]));
  const description = Object.values(identity).join(' ');
  if (required && !description) fail('physical WebGPU adapter identity is empty');
  const genericIdentity = Object.keys(identity).every((key) =>
    key === 'backend' || key === 'device') &&
    identity.backend?.toLowerCase() === 'webgpu' &&
    identity.device?.toLowerCase() === 'gpu';
  const hardwareIdentity = ['vendor', 'architecture', 'device', 'description', 'deviceType', 'driver']
    .some((key) => {
      const value = identity[key]?.toLowerCase();
      return value && !['gpu', 'webgpu', 'unknown', 'unavailable'].includes(value);
    });
  if (required && (genericIdentity || !hardwareIdentity)) {
    fail(`physical WebGPU adapter identity is unavailable (${description})`);
  }
  if (required && SOFTWARE_ADAPTER.test(description)) {
    fail(`software WebGPU adapter rejected (${description})`);
  }
  const requiredIdentity = String(
    globalThis.process?.env?.VOLVOXAI_PARITY_WEBGPU_ADAPTER ?? '',
  ).trim();
  if (required && requiredIdentity &&
      !description.toLowerCase().includes(requiredIdentity.toLowerCase())) {
    fail(`adapter '${description}' does not match '${requiredIdentity}'`);
  }
  return Object.freeze(identity);
}

function validateCompilation(api, report, requirePhysical) {
  operationSucceeded(api, report, 'WebGPU compilation');
  const candidate = report.compilation?.candidates?.[0];
  if (report.backend !== 'webgpu' ||
      report.compilation?.policyMode !==
        api.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE ||
      report.compilation?.operatorFallback !==
        api.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID ||
      report.compilation?.candidates?.length !== 1 ||
      candidate?.backend !== 'webgpu' ||
      candidate?.outcome !== api.pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED) {
    fail('compilation report did not strictly select required WebGPU');
  }
  strictRoute(report, 'compilation');
  const adapter = normalizedAdapterIdentity(report.device, requirePhysical);
  return adapter;
}

function validateExecution(api, report, selectedDevice) {
  operationSucceeded(api, report, 'WebGPU execution');
  if (report.backend !== 'webgpu') {
    fail(`required WebGPU execution reported '${report.backend || 'no backend'}'`);
  }
  strictRoute(report, 'execution');
  if (report.device !== selectedDevice) {
    fail('execution WebGPU device differs from compilation evidence');
  }
}

async function loadInput(api, descriptor) {
  const Constructor = DTYPE[descriptor.dtype];
  if (!Constructor) fail(`input '${descriptor.name}' has unsupported dtype '${descriptor.dtype}'`);
  const bytes = await readFile(descriptor.file);
  if (bytes.byteLength % Constructor.BYTES_PER_ELEMENT !== 0) {
    fail(`input '${descriptor.name}' byte length is not aligned to ${descriptor.dtype}`);
  }
  const data = new Constructor(exactArrayBuffer(bytes));
  const elements = descriptor.shape.reduce((total, extent) => total * extent, 1);
  if (data.length !== elements) {
    fail(`input '${descriptor.name}' has ${data.length} elements, expected ${elements}`);
  }
  const dtype = ({
    float32: api.pb.DataType.DATA_TYPE_F32,
    int32: api.pb.DataType.DATA_TYPE_I32,
    int8: api.pb.DataType.DATA_TYPE_I8,
    uint8: api.pb.DataType.DATA_TYPE_U8,
  })[descriptor.dtype];
  return new api.pb.Tensor({
    name: descriptor.name,
    dtype,
    shape: descriptor.shape.map(BigInt),
    inline: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  });
}

async function loadInputs(api, manifestPath, graph) {
  const inputManifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (!Array.isArray(inputManifest.inputs)) fail(`${manifestPath} has no inputs array`);
  const graphInputNames = Object.keys(graph.inputs ?? {});
  const manifestInputNames = inputManifest.inputs.map((entry) => entry.name);
  if (JSON.stringify(graphInputNames) !== JSON.stringify(manifestInputNames)) {
    fail(`input manifest order ${JSON.stringify(manifestInputNames)} differs from graph ${JSON.stringify(graphInputNames)}`);
  }
  const inputs = [];
  for (const descriptor of inputManifest.inputs) {
    inputs.push(await loadInput(api, descriptor));
  }
  return inputs;
}

const DTYPE_NAME = Object.freeze({
  DATA_TYPE_F32: 'float32',
  DATA_TYPE_I32: 'int32',
  DATA_TYPE_I8: 'int8',
  DATA_TYPE_U8: 'uint8',
});

function dtypeName(api, value) {
  for (const [enumName, name] of Object.entries(DTYPE_NAME)) {
    if (api.pb.DataType[enumName] === value) return name;
  }
  return null;
}

async function readOutputs(api, inference, resultId, graph) {
  const info = await inference.getResult(new api.pb.ResultRef({ resultId }));
  operationSucceeded(api, info.report, 'GetResult');
  const actualNames = info.outputs.map(({ name }) => name);
  if (JSON.stringify(actualNames) !== JSON.stringify(graph.outputs)) {
    fail(`runtime outputs ${JSON.stringify(actualNames)} differ from graph ${JSON.stringify(graph.outputs)}`);
  }
  const outputs = new Map();
  for (const descriptor of info.outputs) {
    const response = await inference.readOutput(new api.pb.ReadOutputRequest({
      resultId,
      name: descriptor.name,
    }));
    operationSucceeded(api, response.report, `ReadOutput(${descriptor.name})`);
    const tensor = response.tensor;
    const dtype = dtypeName(api, tensor?.dtype);
    if (!tensor || !dtype || tensor.inline === undefined) {
      fail(`output '${descriptor.name}' has no supported inline payload`);
    }
    outputs.set(descriptor.name, {
      name: tensor.name,
      dtype,
      shape: tensor.shape.map(Number),
      byteLength: Number(response.requiredBytes),
      bytes: exactBytes(tensor.inline),
    });
  }
  return outputs;
}

async function main() {
  const options = parseArguments(globalThis.Deno?.args ?? globalThis.process.argv.slice(2));
  globalThis.window ??= globalThis;
  globalThis.self ??= globalThis;
  globalThis.document ??= { createElement: () => ({}), querySelector: () => null };
  const nativeFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (source, init) => {
    const href = typeof source === 'string' ? source : source?.url;
    if (href?.startsWith('file://')) {
      return new Response(await readFile(fileURLToPath(href)));
    }
    return nativeFetch(source, init);
  };

  const [api, graphText, weights] = await Promise.all([
    import(pathToFileURL(options.bundle).href),
    readFile(path.join(options.package, 'graph.json'), 'utf8'),
    readFile(path.join(options.package, 'model.safetensors')),
  ]);
  const graph = JSON.parse(graphText);
  if (typeof api.FullEngineHost !== 'function' ||
      typeof api.VxInferenceServiceClient !== 'function' || !api.pb) {
    fail('full bundle does not expose the generated proto client and FullEngineHost');
  }

  const fetchPackage = async (source) => {
    const value = String(source);
    if (value === 'graph.json') {
      return { ok: true, json: async () => JSON.parse(graphText), text: async () => graphText };
    }
    if (value === 'model.safetensors') {
      return { ok: true, arrayBuffer: async () => exactArrayBuffer(weights) };
    }
    return { ok: false, status: 404, statusText: `unexpected package source ${value}` };
  };

  const inputs = await loadInputs(api, options.inputs, graph);
  const lifecycleInputs = options.nextInputs ? {
    next: await loadInputs(api, options.nextInputs, graph),
    invalid: await loadInputs(api, options.invalidInputs, graph),
    changed: await loadInputs(api, options.changedInputs, graph),
  } : null;

  const host = new api.FullEngineHost({
    wasmUrl: pathToFileURL(options.wasm),
    fetch: fetchPackage,
    resolveModelSource: (graphPath, weightPaths) => ({
      graphUrl: graphPath,
      weightSources: weightPaths,
    }),
  });
  const inference = new api.VxInferenceServiceClient(host);
  let runtimeId = 0n;
  let modelId = 0n;
  let compiledModelId = 0n;
  let contextId = 0n;
  let firstResultId = 0n;
  let secondResultId = 0n;
  let changedResultId = 0n;
  let lifecycleOutputSets = null;
  try {
    const runtime = await inference.createRuntime(new api.pb.CreateRuntimeRequest());
    operationSucceeded(api, runtime.report, 'CreateRuntime');
    runtimeId = runtime.runtimeId;
    const model = await inference.loadModel(new api.pb.LoadModelRequest({
      runtimeId,
      graphPath: 'graph.json',
      weightPaths: ['model.safetensors'],
    }));
    operationSucceeded(api, model.report, 'LoadModel');
    modelId = model.modelId;
    const compiled = await inference.compileModel(new api.pb.CompileModelRequest({
      modelId,
      policy: new api.pb.BackendPolicy({
        mode: api.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: ['webgpu'],
        operatorFallback: api.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
      }),
    }));
    const adapter = validateCompilation(api, compiled.report, options.requirePhysical);
    const selectedDevice = compiled.report.device;
    compiledModelId = compiled.compiledModelId;
    const context = await inference.createExecutionContext(
      new api.pb.CreateExecutionContextRequest({ compiledModelId }),
    );
    operationSucceeded(api, context.report, 'CreateExecutionContext');
    contextId = context.contextId;
    const first = await inference.execute(new api.pb.ExecuteRequest({ contextId, inputs }));
    validateExecution(api, first.report, selectedDevice);
    firstResultId = first.resultId;
    const firstOutputs = await readOutputs(api, inference, firstResultId, graph);

    const second = await inference.execute(new api.pb.ExecuteRequest({
      contextId,
      inputs: lifecycleInputs?.next ?? inputs,
    }));
    validateExecution(api, second.report, selectedDevice);
    secondResultId = second.resultId;
    const secondOutputs = await readOutputs(api, inference, secondResultId, graph);
    if (lifecycleInputs) {
      lifecycleOutputSets = { next: secondOutputs };
      if (second.report.route?.shapePlan?.signature ===
          first.report.route?.shapePlan?.signature) {
        fail('min and max executions published the same shape signature');
      }
    } else {
      for (const name of graph.outputs) {
        if (!bytesEqual(firstOutputs.get(name).bytes, secondOutputs.get(name).bytes)) {
          fail(`output '${name}' changed across same-context replay`);
        }
      }
    }
    operationSucceeded(api, await inference.releaseResult(
      new api.pb.ResultRef({ resultId: secondResultId })), 'Release second result');
    secondResultId = 0n;

    let invalidCode = null;
    let changedInputObserved = false;
    if (lifecycleInputs) {
      const invalid = await inference.execute(new api.pb.ExecuteRequest({
        contextId,
        inputs: lifecycleInputs.invalid,
      }));
      if (invalid.report?.status === api.pb.NativeStatus.NATIVE_STATUS_OK) {
        if (invalid.resultId !== 0n) {
          await inference.releaseResult(new api.pb.ResultRef({ resultId: invalid.resultId }));
        }
        fail('out-of-domain dynamic input unexpectedly executed');
      }
      invalidCode = invalid.report?.code || '';
      if (invalidCode !== 'INVALID_ARGUMENT') {
        fail(`out-of-domain dynamic input rejected with '${invalidCode || 'untyped error'}'`);
      }

      const changed = await inference.execute(new api.pb.ExecuteRequest({
        contextId,
        inputs: lifecycleInputs.changed,
      }));
      validateExecution(api, changed.report, selectedDevice);
      changedResultId = changed.resultId;
      if (changed.report.route?.shapePlan?.signature !==
          first.report.route?.shapePlan?.signature) {
        fail('changed min execution did not return to the original shape signature');
      }
      const changedOutputs = await readOutputs(api, inference, changedResultId, graph);
      lifecycleOutputSets.changed = changedOutputs;
      changedInputObserved = graph.outputs.some((name) =>
        !bytesEqual(firstOutputs.get(name).bytes, changedOutputs.get(name).bytes));
      if (!changedInputObserved) {
        fail('changed min input produced byte-identical outputs; stale replay is not excluded');
      }
      operationSucceeded(api, await inference.releaseResult(
        new api.pb.ResultRef({ resultId: changedResultId })), 'Release changed result');
      changedResultId = 0n;
    }

    operationSucceeded(api, await inference.releaseExecutionContext(
      new api.pb.ExecutionContextRef({ contextId })), 'Release execution context');
    contextId = 0n;
    const stableOutputs = await readOutputs(api, inference, firstResultId, graph);
    for (const name of graph.outputs) {
      if (!bytesEqual(firstOutputs.get(name).bytes, stableOutputs.get(name).bytes)) {
        fail(`output '${name}' changed after execution-context close`);
      }
    }

    await mkdir(options.out, { recursive: true });
    const outputs = [];
    for (const name of graph.outputs) {
      const descriptor = firstOutputs.get(name);
      const filename = `${name}.raw`;
      await writeFile(path.join(options.out, filename), descriptor.bytes);
      outputs.push({
        name: descriptor.name,
        dtype: descriptor.dtype,
        shape: descriptor.shape,
        byteLength: descriptor.byteLength,
        file: filename,
      });
    }
    const lifecycleOutputs = {};
    for (const [phase, outputSet] of Object.entries(lifecycleOutputSets ?? {})) {
      const descriptors = [];
      for (const name of graph.outputs) {
        const descriptor = outputSet.get(name);
        const filename = `lifecycle-${phase}-${name}.raw`;
        await writeFile(path.join(options.out, filename), descriptor.bytes);
        descriptors.push({
          name: descriptor.name,
          dtype: descriptor.dtype,
          shape: descriptor.shape,
          byteLength: descriptor.byteLength,
          file: filename,
        });
      }
      lifecycleOutputs[phase] = descriptors;
    }
    await writeFile(path.join(options.out, 'outputs.json'), `${JSON.stringify({
      schema: 'volvoxai.onnx-oracle-output',
      version: 1,
      backend: 'webgpu',
      adapter,
      executionEvidence: {
        provider: first.report.route?.provider ?? '',
        attested: first.report.route?.attested === true,
      },
      stableResult: {
        sameContextReplay: lifecycleInputs === null,
        readableAfterContextClose: true,
        ...(lifecycleInputs ? {
          dynamicLifecycle: {
            minToMax: true,
            invalidRejected: true,
            invalidCode,
            changedMinObserved: changedInputObserved,
            returnedToMinShape: true,
          },
        } : {}),
      },
      ...(lifecycleOutputSets ? { lifecycleOutputs } : {}),
      outputs,
    }, null, 2)}\n`);
  } finally {
    const cleanupFailures = [];
    for (const [label, id, release] of [
      ['second result', secondResultId, async (value) => (await inference.releaseResult(
        new api.pb.ResultRef({ resultId: value })))],
      ['changed result', changedResultId, async (value) => (await inference.releaseResult(
        new api.pb.ResultRef({ resultId: value })))],
      ['first result', firstResultId, async (value) => (await inference.releaseResult(
        new api.pb.ResultRef({ resultId: value })))],
      ['context', contextId, async (value) => (await inference.releaseExecutionContext(
        new api.pb.ExecutionContextRef({ contextId: value })))],
      ['compiled model', compiledModelId, async (value) => (await inference.releaseCompiledModel(
        new api.pb.CompiledModelRef({ compiledModelId: value })))],
      ['model', modelId, async (value) => (await inference.releaseModel(new api.pb.ModelRef({ modelId: value })))],
      ['runtime', runtimeId, async (value) => (await inference.releaseRuntime(new api.pb.RuntimeRef({ runtimeId: value })))],
    ]) {
      if (id === 0n) continue;
      try {
        operationSucceeded(api, await release(id), `Release ${label}`);
      } catch (error) {
        cleanupFailures.push(new Error(`failed to close ${label}`, { cause: error }));
      }
    }
    if (cleanupFailures.length) {
      throw new AggregateError(cleanupFailures, 'WebGPU oracle resource cleanup failed');
    }
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  globalThis.process.exitCode = 1;
});
