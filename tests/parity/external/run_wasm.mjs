#!/usr/bin/env node

import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function fail(message) {
  throw new Error(`[onnx-oracle/wasm] ${message}`);
}

function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    if (!['--bundle', '--wasm', '--package', '--inputs', '--out'].includes(flag)) {
      fail(`unknown argument ${flag}`);
    }
    const value = argv[++index];
    if (!value || value.startsWith('--')) fail(`${flag} requires a value`);
    options[flag.slice(2)] = path.resolve(value);
  }
  for (const name of ['bundle', 'wasm', 'package', 'inputs', 'out']) {
    if (!options[name]) fail(`--${name} is required`);
  }
  return options;
}

const DTYPE = Object.freeze({
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
});

function exactArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
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

function validateCompilation(api, report) {
  operationSucceeded(api, report, 'WASM compilation');
  const candidate = report.compilation?.candidates?.[0];
  if (report.backend !== 'wasm' ||
      report.compilation?.policyMode !==
        api.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE ||
      report.compilation?.operatorFallback !==
        api.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID ||
      report.compilation?.candidates?.length !== 1 ||
      candidate?.backend !== 'wasm' ||
      candidate?.outcome !== api.pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED) {
    fail('compilation report did not strictly select required WASM');
  }
  strictRoute(report, 'compilation');
}

function validateExecution(api, report) {
  operationSucceeded(api, report, 'WASM execution');
  if (report.backend !== 'wasm') {
    fail(`required WASM execution reported '${report.backend || 'no backend'}'`);
  }
  strictRoute(report, 'execution');
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
  const outputs = [];
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
    outputs.push({
      name: tensor.name,
      dtype,
      shape: tensor.shape.map(Number),
      byteLength: Number(response.requiredBytes),
      bytes: tensor.inline,
    });
  }
  return outputs;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const [api, graphText, weights, inputManifestText] = await Promise.all([
    import(pathToFileURL(options.bundle).href),
    readFile(path.join(options.package, 'graph.json'), 'utf8'),
    readFile(path.join(options.package, 'model.safetensors')),
    readFile(options.inputs, 'utf8'),
  ]);
  const graph = JSON.parse(graphText);
  const inputManifest = JSON.parse(inputManifestText);
  if (typeof api.EngineHost !== 'function' ||
      typeof api.VxInferenceServiceClient !== 'function' || !api.pb) {
    fail('inference bundle does not expose the generated proto client and EngineHost');
  }
  if (!Array.isArray(inputManifest.inputs)) fail('input manifest has no inputs array');
  const graphInputNames = Object.keys(graph.inputs ?? {});
  const manifestInputNames = inputManifest.inputs.map((entry) => entry.name);
  if (JSON.stringify(graphInputNames) !== JSON.stringify(manifestInputNames)) {
    fail(
      `input manifest order ${JSON.stringify(manifestInputNames)} `
      + `differs from graph ${JSON.stringify(graphInputNames)}`,
    );
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

  const inputs = [];
  for (const descriptor of inputManifest.inputs) {
    inputs.push(await loadInput(api, descriptor));
  }

  const host = new api.EngineHost({
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
  let resultId = 0n;
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
        backends: ['wasm'],
        operatorFallback: api.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
      }),
    }));
    validateCompilation(api, compiled.report);
    compiledModelId = compiled.compiledModelId;
    const context = await inference.createExecutionContext(
      new api.pb.CreateExecutionContextRequest({ compiledModelId }),
    );
    operationSucceeded(api, context.report, 'CreateExecutionContext');
    contextId = context.contextId;
    const result = await inference.execute(new api.pb.ExecuteRequest({ contextId, inputs }));
    validateExecution(api, result.report);
    resultId = result.resultId;
    const descriptors = await readOutputs(api, inference, resultId, graph);

    await mkdir(options.out, { recursive: true });
    const outputs = [];
    for (const descriptor of descriptors) {
      const filename = `${descriptor.name}.raw`;
      const bytes = Buffer.from(
        descriptor.bytes.buffer,
        descriptor.bytes.byteOffset,
        descriptor.bytes.byteLength,
      );
      await writeFile(path.join(options.out, filename), bytes);
      outputs.push({
        name: descriptor.name,
        dtype: descriptor.dtype,
        shape: descriptor.shape,
        byteLength: descriptor.byteLength,
        file: filename,
      });
    }
    await writeFile(path.join(options.out, 'outputs.json'), `${JSON.stringify({
      schema: 'volvoxai.onnx-oracle-output',
      version: 1,
      backend: 'wasm',
      outputs,
    }, null, 2)}\n`);
  } finally {
    const cleanupFailures = [];
    for (const [label, id, release] of [
      ['result', resultId, async (value) => (await inference.releaseResult(new api.pb.ResultRef({ resultId: value })))],
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
      throw new AggregateError(cleanupFailures, 'WASM oracle resource cleanup failed');
    }
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exitCode = 1;
});
