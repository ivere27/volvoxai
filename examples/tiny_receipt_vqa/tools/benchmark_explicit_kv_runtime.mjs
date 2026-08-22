#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

import { preprocessTinyReceiptImage } from '../TinyReceiptInput.js';
import { TinyReceiptSplitSession } from '../TinyReceiptSplitSession.js';
import {
  executionPhaseBreakdown,
  finiteMilliseconds,
  summarizeDecoderSteps,
  validateExplicitKVAnswer,
} from './benchmark_explicit_kv_contract.mjs';
import {
  DYNAMIC_REBIND_QUALIFICATION_SCHEMA,
  qualifyDynamicRebind,
} from './qualify_dynamic_rebind.mjs';
import {
  requireTinyReceiptPhysicalAdapterIdentity,
} from '../TinyReceiptSplitE2E.js';

export {
  executionPhaseBreakdown,
  finiteMilliseconds,
  summarizeDecoderSteps,
  validateExplicitKVAnswer,
} from './benchmark_explicit_kv_contract.mjs';

export const SAMPLE_SCHEMA = 'volvoxai.tiny-receipt-explicit-kv-runtime-sample/v1';
export const KV_PACKAGE_FORMAT = 'volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2';
export const SAMPLE_PREFIX = 'TINYRECEIPT_EXPLICIT_KV_SAMPLE ';
export const DYNAMIC_QUALIFICATION_PREFIX =
  'TINYRECEIPT_DYNAMIC_REBIND_QUALIFICATION ';

const DYNAMIC_QUALIFICATION_MODE = 'dynamic-rebind';
const DYNAMIC_QUALIFICATION_PROMPT = 'phone number last one';
const CANONICAL_IMAGE_WIDTH = 672;
const CANONICAL_IMAGE_HEIGHT = 320;
export const CANONICAL_PIXEL_SHA256 =
  'a0cd5bef32c7c56946a9487979213b0f41f0810c98b5b28771684c1d02b7c19d';
export const CANONICAL_INPUT_F32_SHA256 =
  '7a6f7eb153434868b1685c4fd96fc63f1004cae356d15bd58404dccdc2c15963';

const repository = fileURLToPath(new URL('../../../', import.meta.url));

function fail(message) {
  throw new Error(`[benchmark_explicit_kv_runtime] ${message}`);
}

export function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const argument = values[index];
    if (!argument.startsWith('--')) fail(`unexpected positional argument '${argument}'.`);
    const separator = argument.indexOf('=');
    const name = separator < 0 ? argument.slice(2) : argument.slice(2, separator);
    if (!name || Object.hasOwn(result, name)) fail(`duplicate option '--${name}'.`);
    if (separator >= 0) {
      result[name] = argument.slice(separator + 1);
    } else if (index + 1 < values.length && !values[index + 1].startsWith('--')) {
      result[name] = values[++index];
    } else {
      result[name] = true;
    }
  }
  const allowed = new Set([
    'adapter', 'api', 'backend', 'canonical-image', 'execution-warmup',
    'family', 'image', 'max-new', 'package', 'prompt', 'qualification',
    'require-adapter', 'wasm',
  ]);
  for (const name of Object.keys(result)) {
    if (!allowed.has(name)) fail(`unknown option '--${name}'.`);
  }
  if (typeof result.package !== 'string' || result.package.length === 0) {
    fail('--package is required.');
  }
  const canonicalImage = result['canonical-image'] === true;
  if (result['canonical-image'] !== undefined && !canonicalImage) {
    fail('--canonical-image does not accept a value.');
  }
  const hasImage = typeof result.image === 'string' && result.image.length > 0;
  if (Boolean(result.image) !== hasImage || hasImage === canonicalImage) {
    fail('pass exactly one of --image <png> or --canonical-image.');
  }
  const qualification = result.qualification ?? null;
  if (qualification !== null && qualification !== DYNAMIC_QUALIFICATION_MODE) {
    fail(`--qualification must be '${DYNAMIC_QUALIFICATION_MODE}'.`);
  }
  if (result.prompt !== undefined &&
      (typeof result.prompt !== 'string' || result.prompt.length === 0)) {
    fail('--prompt must be a non-empty string.');
  }
  if (qualification !== null && result.prompt !== undefined &&
      result.prompt.normalize('NFC') !== DYNAMIC_QUALIFICATION_PROMPT) {
    fail(`dynamic-rebind qualification requires the canonical prompt ` +
      `'${DYNAMIC_QUALIFICATION_PROMPT}'.`);
  }
  if (qualification === null && typeof result.prompt !== 'string') {
    fail('--prompt is required.');
  }
  const backend = result.backend ?? 'wasm';
  if (!['wasm', 'webgpu'].includes(backend)) {
    fail("--backend must be 'wasm' or 'webgpu'; JS CPU is outside this benchmark.");
  }
  if (qualification !== null && backend !== 'wasm') {
    fail('dynamic-rebind qualification currently requires --backend=wasm.');
  }
  const maxNewTokens = Number(result['max-new'] ?? 4);
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 2 || maxNewTokens > 191) {
    fail('--max-new must be an integer in [2, 191].');
  }
  if (qualification !== null && maxNewTokens !== 4) {
    fail('dynamic-rebind qualification requires --max-new=4.');
  }
  const executionWarmup = Number(result['execution-warmup'] ?? 0);
  if (!Number.isInteger(executionWarmup) || executionWarmup < 0 || executionWarmup > 20) {
    fail('--execution-warmup must be an integer in [0, 20].');
  }
  if (qualification !== null && executionWarmup !== 0) {
    fail('dynamic-rebind qualification does not accept --execution-warmup.');
  }
  const adapterPreference = result.adapter ?? 'high-performance';
  if (!['default', 'high-performance', 'low-power'].includes(adapterPreference)) {
    fail("--adapter must be 'default', 'high-performance', or 'low-power'.");
  }
  const requiredAdapter = result['require-adapter'] ?? null;
  if (backend === 'webgpu') {
    if (typeof requiredAdapter !== 'string' || requiredAdapter.length === 0) {
      fail('--require-adapter is required for a physical WebGPU benchmark.');
    }
    if (executionWarmup < 1) {
      fail('WebGPU measurement requires --execution-warmup >= 1 for KV qualification.');
    }
  } else if (result.adapter !== undefined || requiredAdapter !== null) {
    fail('--adapter and --require-adapter apply only to --backend=webgpu.');
  }
  const prompt = (result.prompt ?? DYNAMIC_QUALIFICATION_PROMPT).normalize('NFC');
  const family = result.family ?? 'phone';
  if (canonicalImage &&
      (prompt !== DYNAMIC_QUALIFICATION_PROMPT || family !== 'phone' || maxNewTokens !== 4)) {
    fail(`--canonical-image requires --prompt '${DYNAMIC_QUALIFICATION_PROMPT}', ` +
      "--family phone, and --max-new 4.");
  }
  return Object.freeze({
    packageDir: resolve(result.package),
    imagePath: hasImage ? resolve(result.image) : null,
    canonicalImage,
    prompt,
    family,
    backend,
    maxNewTokens,
    qualification,
    executionWarmup,
    adapterPreference,
    requiredAdapter,
    apiPath: resolve(result.api ?? join(repository, 'dist/0.4.0/volvoxai.js')),
    wasmPath: resolve(result.wasm ?? join(repository, 'dist/0.4.0/volvoxai.wasm')),
  });
}

function strictRoute(route, label) {
  if (route?.tierFallback !== false || route?.operator?.attestation !== 'none' ||
      route.operator.used !== false || route.operator.offendingNode != null) {
    fail(`${label} does not attest a strict no-fallback route.`);
  }
}

export function validateCompilation(report, backend, label) {
  const candidate = report?.candidates?.[0];
  if (report?.requestedPolicy?.mode !== 'require' ||
      report.requestedPolicy.backend !== backend ||
      report.requestedPolicy.operatorFallback !== 'forbid' ||
      report.selectedBackend !== backend || !Array.isArray(report.candidates) ||
      report.candidates.length !== 1 || candidate?.backend !== backend ||
      candidate.outcome !== 'selected' || report.selectedDevice == null ||
      canonicalJson(candidate.device) !== canonicalJson(report.selectedDevice)) {
    fail(`${label} compilation did not strictly select ${backend}.`);
  }
  strictRoute(report.routeEvidence, `${label} compilation`);
  strictRoute(candidate.routeEvidence, `${label} compilation candidate`);
}

function validateExecution(report, backend, label) {
  if (report?.backend !== backend || report.outcome !== 'success' ||
      report.operatorFallback !== 'none') {
    fail(`${label} did not execute successfully on strict ${backend}.`);
  }
  strictRoute(report.routeEvidence, label);
}

/**
 * Validate that each qualification request used the same encoder and decoder
 * contexts, with one encoder execution and one decoder execution per token.
 */
export function validateDynamicExecutionGroups(groups, runs, backend) {
  if (!Array.isArray(groups) || !Array.isArray(runs) || groups.length !== runs.length ||
      groups.length !== 4) {
    fail('dynamic qualification must contain four matching execution groups.');
  }
  let encoderContextId = null;
  let decoderContextId = null;
  const attestations = groups.map((group, runIndex) => {
    const run = runs[runIndex];
    if (group?.label !== run?.label || !Array.isArray(group?.reports) ||
        !Array.isArray(run?.tokenIds)) {
      fail(`dynamic qualification run ${runIndex} does not match its execution group.`);
    }
    if (group.reports.length !== run.tokenIds.length + 1) {
      fail(`dynamic qualification run ${runIndex} must contain one encoder and ` +
        'one decoder execution per token.');
    }
    const encoder = group.reports[0];
    const decoder = group.reports.slice(1);
    if (typeof encoder?.contextId !== 'string' || encoder.contextId.length === 0 ||
        decoder.some((report) => typeof report?.contextId !== 'string' ||
          report.contextId.length === 0)) {
      fail(`dynamic qualification run ${runIndex} lacks context identities.`);
    }
    if (runIndex === 0) {
      encoderContextId = encoder.contextId;
      decoderContextId = decoder[0]?.contextId;
      if (!decoderContextId || decoderContextId === encoderContextId) {
        fail('dynamic qualification does not identify distinct encoder and decoder contexts.');
      }
    }
    if (encoder.contextId !== encoderContextId ||
        decoder.some((report) => report.contextId !== decoderContextId)) {
      fail(`dynamic qualification run ${runIndex} did not reuse the same contexts.`);
    }
    validateExecution(encoder, backend, `dynamic run ${runIndex} encoder`);
    decoder.forEach((report, step) =>
      validateExecution(report, backend, `dynamic run ${runIndex} decoder step ${step}`));
    return Object.freeze({
      label: run.label,
      encoder,
      decoder: Object.freeze(decoder),
    });
  });
  return Object.freeze({
    encoderContextId,
    decoderContextId,
    attestations: Object.freeze(attestations),
  });
}

async function sha256File(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

function sha256Float32LE(values) {
  if (!(values instanceof Float32Array)) fail('preprocessed image must be Float32Array.');
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return createHash('sha256').update(bytes).digest('hex');
}

async function fileFetch(input) {
  try {
    const source = input instanceof Request ? input.url : String(input);
    const url = new URL(source);
    if (url.protocol !== 'file:') {
      return new Response('Only local package assets are accepted.', { status: 403 });
    }
    return new Response(await readFile(fileURLToPath(url)), { status: 200 });
  } catch (error) {
    return new Response(String(error), { status: 404 });
  }
}

function decodeImage(path) {
  const program = [
    'from PIL import Image',
    'import struct,sys',
    "im=Image.open(sys.argv[1]).convert('RGB')",
    "sys.stdout.buffer.write(struct.pack('<II', *im.size))",
    'sys.stdout.buffer.write(im.tobytes())',
  ].join('\n');
  const decoded = execFileSync('python3', ['-c', program, path], {
    maxBuffer: 32 * 1024 * 1024,
  });
  if (decoded.byteLength < 8) fail('image decoder returned a truncated payload.');
  const width = decoded.readUInt32LE(0);
  const height = decoded.readUInt32LE(4);
  const pixels = decoded.byteLength - 8;
  if (width <= 0 || height <= 0 || pixels !== width * height * 3) {
    fail('image decoder returned invalid RGB dimensions.');
  }
  return Object.freeze({
    data: new Uint8Array(decoded.buffer, decoded.byteOffset + 8, pixels),
    width,
    height,
    channels: 3,
  });
}

export function createCanonicalTinyReceiptImage() {
  const data = new Uint8Array(CANONICAL_IMAGE_WIDTH * CANONICAL_IMAGE_HEIGHT);
  for (let y = 0; y < CANONICAL_IMAGE_HEIGHT; y++) {
    for (let x = 0; x < CANONICAL_IMAGE_WIDTH; x++) {
      data[y * CANONICAL_IMAGE_WIDTH + x] = (17 * x + 29 * y + 7 * (x ^ y)) & 255;
    }
  }
  const digest = createHash('sha256').update(data).digest('hex');
  if (digest !== CANONICAL_PIXEL_SHA256) {
    fail(`canonical pixel generator produced '${digest}'.`);
  }
  return Object.freeze({
    data,
    width: CANONICAL_IMAGE_WIDTH,
    height: CANONICAL_IMAGE_HEIGHT,
    channels: 1,
  });
}

async function prepareBenchmarkInput(manifest, options) {
  const resize = manifest?.preprocessing?.resize;
  if (!Number.isInteger(resize?.width) || !Number.isInteger(resize?.height)) {
    fail('package preprocessing must declare integer resize dimensions.');
  }
  if (options.canonicalImage &&
      (resize.width !== CANONICAL_IMAGE_WIDTH || resize.height !== CANONICAL_IMAGE_HEIGHT)) {
    fail(`canonical image requires package resize ${CANONICAL_IMAGE_WIDTH}x${CANONICAL_IMAGE_HEIGHT}.`);
  }
  const image = options.canonicalImage
    ? createCanonicalTinyReceiptImage()
    : decodeImage(options.imagePath);
  const inputTensor = await preprocessTinyReceiptImage(image, {
    width: resize.width,
    height: resize.height,
  });
  const inputTensorSha256 = sha256Float32LE(inputTensor);
  if (options.canonicalImage && inputTensorSha256 !== CANONICAL_INPUT_F32_SHA256) {
    fail(`canonical F32 input produced '${inputTensorSha256}'.`);
  }
  return Object.freeze({ inputTensor, inputTensorSha256 });
}

function precisionFromManifest(manifest) {
  if (manifest?.format !== KV_PACKAGE_FORMAT ||
      manifest?.generation?.strategy !== 'greedy-autoregressive-explicit-kv') {
    fail(`package must use '${KV_PACKAGE_FORMAT}'.`);
  }
  if (manifest?.source?.variant === 'fp32') return 'fp32';
  if (manifest?.source?.variant === 'int8-w8a8') return 'int8';
  fail('package source.variant must be fp32 or int8-w8a8.');
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

export function overrideWebGPUAdapterPreference(gpu, preference) {
  if (preference === 'default') return () => {};
  const originalDescriptor = Object.getOwnPropertyDescriptor(gpu, 'requestAdapter');
  const originalRequestAdapter = gpu?.requestAdapter;
  if (typeof originalRequestAdapter !== 'function') {
    fail('WebGPU requestAdapter is unavailable.');
  }
  const bound = originalRequestAdapter.bind(gpu);
  gpu.requestAdapter = (request = {}) => bound({
    ...request,
    powerPreference: preference,
  });
  return () => {
    if (originalDescriptor) {
      Object.defineProperty(gpu, 'requestAdapter', originalDescriptor);
    } else if (!delete gpu.requestAdapter) {
      fail('could not restore inherited WebGPU requestAdapter.');
    }
  };
}

function runtimeIdentity(backend) {
  const deno = typeof Deno === 'undefined' ? null : Deno;
  if (!deno) {
    return Object.freeze({ name: 'node', version: process.version });
  }
  const identity = {
    name: 'deno',
    version: deno.version.deno,
    v8: deno.version.v8,
    typescript: deno.version.typescript,
    target: deno.build.target,
  };
  if (backend === 'webgpu') {
    let configuredBackend;
    try {
      configuredBackend = deno.env.get('DENO_WEBGPU_BACKEND');
    } catch {
      fail('Deno WebGPU measurement requires --allow-env=DENO_WEBGPU_BACKEND.');
    }
    if (configuredBackend !== 'vulkan') {
      fail("Deno WebGPU measurement requires DENO_WEBGPU_BACKEND=vulkan.");
    }
    identity.webgpuBackend = configuredBackend;
  }
  return Object.freeze(identity);
}

function routeAttestation(report) {
  return Object.freeze({
    executionId: report.executionId,
    contextId: report.contextId,
    backend: report.backend,
    device: report.device,
    outcome: report.outcome,
    operatorFallback: report.operatorFallback,
    routeEvidence: report.routeEvidence,
  });
}

function compilationAttestation(label, report) {
  const candidate = report.candidates[0];
  return Object.freeze({
    label,
    compilationId: report.compilationId,
    requestedPolicy: report.requestedPolicy,
    selectedBackend: report.selectedBackend,
    selectedDevice: report.selectedDevice,
    routeEvidence: report.routeEvidence,
    candidate: Object.freeze({
      backend: candidate.backend,
      outcome: candidate.outcome,
      device: candidate.device,
      operatorFallback: candidate.operatorFallback,
      routeEvidence: candidate.routeEvidence,
    }),
  });
}

export function selectComponentTiming(backend, synchronized, encoderPhases, decoderPhases) {
  if (backend === 'webgpu') {
    return Object.freeze({
      boundary: 'required-small-output-readback',
      encoderMs: synchronized.encoderMs,
      decoder: synchronized.decoder,
    });
  }
  return Object.freeze({
    boundary: 'runtime-execution-diagnostic',
    encoderMs: encoderPhases.executionMs,
    decoder: summarizeDecoderSteps(decoderPhases.map(({ executionMs }) => executionMs)),
  });
}

function answerSignature(answer, tokens) {
  return canonicalJson({
    family: answer.family,
    familyId: answer.familyId,
    requestedFamily: answer.requestedFamily,
    questionTokenIds: [...answer.questionTokenIds],
    tokenIds: [...tokens],
    stoppedAtEos: answer.stoppedAtEos,
    activeShape: answer.activeShape,
    logicalShape: answer.logicalShape,
    cacheShape: answer.cacheShape,
    decodeReports: answer.decodeReports.map(
      ({ operation, position, pastLength, presentLength, sentinelMaskValue }) => ({
        operation, position, pastLength, presentLength, sentinelMaskValue,
      }),
    ),
  });
}

function validateCanonicalBenchmarkAnswer(answer, tokens, options) {
  if (!options.canonicalImage) return;
  const expectedQuestion = [1038, 54, 1124, 54, 1181, 54, 1031, 2];
  const expectedTokens = [4, 1038, 5, 6];
  if (answer.family !== 'phone' || answer.familyId !== 0 ||
      answer.requestedFamily !== 'phone' ||
      canonicalJson([...answer.questionTokenIds]) !== canonicalJson(expectedQuestion) ||
      canonicalJson([...tokens]) !== canonicalJson(expectedTokens) ||
      canonicalJson(answer.activeShape) !== canonicalJson({ B: 1, Q: 8, M: 218, T: 5 })) {
    fail('canonical benchmark answer does not match the fixed family/token/shape contract.');
  }
}

function validateExecutionGroup(reports, tokens, backend, label) {
  if (!Array.isArray(reports) || reports.length !== tokens.length + 1) {
    fail(`${label} diagnostics do not contain one encoder and one decoder call per token.`);
  }
  const encoder = reports[0];
  const decoder = reports.slice(1);
  if (typeof encoder?.contextId !== 'string' || !encoder.contextId ||
      typeof decoder[0]?.contextId !== 'string' || !decoder[0].contextId ||
      decoder[0].contextId === encoder.contextId ||
      decoder.some((report) => report.contextId !== decoder[0].contextId)) {
    fail(`${label} diagnostics do not identify one encoder and one decoder context.`);
  }
  validateExecution(encoder, backend, `${label} encoder`);
  decoder.forEach((report, index) =>
    validateExecution(report, backend, `${label} decoder step ${index}`));
  return Object.freeze({
    encoder,
    decoder: Object.freeze(decoder),
    encoderContextId: encoder.contextId,
    decoderContextId: decoder[0].contextId,
  });
}

export function validateSynchronizedTiming(answer, reports, backend, label) {
  const timing = answer?.synchronizedTiming;
  const expectedCompletion = backend === 'webgpu'
    ? 'required-small-output-readback' : 'required-output-readback';
  if (timing?.completion !== expectedCompletion ||
      !Array.isArray(timing.decoderStepMs) ||
      timing.decoderStepMs.length !== reports.decoder.length) {
    fail(`${label} lacks the required synchronized ${backend} timing boundary.`);
  }
  if (backend === 'webgpu' &&
      (timing.applicationValidationIncluded !== false ||
       timing.cacheQualificationReadbackIncluded !== false)) {
    fail(`${label} WebGPU timing includes application or cache qualification readback.`);
  }
  const encoderMs = finiteMilliseconds(timing.encoderExecutionMs, `${label} encoder execution`);
  const decoder = summarizeDecoderSteps(
    timing.decoderStepMs.map((value, index) =>
      finiteMilliseconds(value, `${label} decoder step ${index}`)),
  );
  const diagnosticValues = [reports.encoder.executionTimeMs,
    ...reports.decoder.map((report) => report.executionTimeMs)].map((value, index) =>
    finiteMilliseconds(value, `${label} execution diagnostic ${index}`));
  const synchronizedValues = [encoderMs, ...decoder.steps];
  if (diagnosticValues.some((value, index) => value > synchronizedValues[index] + 0.001)) {
    fail(`${label} synchronized timing is shorter than its execution diagnostic.`);
  }
  return Object.freeze({ encoderMs, decoder });
}

export function validateWebGPUCacheEvidence(answer, tokenCount, qualified, label) {
  const resident = answer?.gpuResidentKv;
  const expectedDecoderHandoffs = Math.max(0, tokenCount - 1) * 8;
  if (resident?.enabled !== true || resident.crossCacheOutputs !== 8 ||
      resident.presentCacheOutputsPerStep !== 8 ||
      resident.encoderCrossCacheHandoffs !== tokenCount * 8 ||
      resident.decoderCacheHandoffs !== expectedDecoderHandoffs ||
      resident.runtimeValidatedDeviceInputs !== true ||
      resident.encoderResultRetainedThroughDecode !== true ||
      resident.decoderResultRetainedUntilSuccessorExecution !== true) {
    fail(`${label} lacks exact device-resident KV handoff evidence.`);
  }
  if (qualified) {
    if (resident.mode !== 'device-qualified' ||
        resident.encoderMemoryReadback !== true ||
        resident.encoderMemoryReadbackValidated !== true ||
        resident.encoderCrossCacheReadbackValidated !== true ||
        resident.cachePrefixReadbackValidated !== true ||
        resident.appendedCacheReadbackValidated !== true ||
        resident.cacheReadbackFree !== false) {
      fail(`${label} did not qualify device-resident KV cache values.`);
    }
  } else if (resident.mode !== 'device-resident' ||
      resident.encoderMemoryReadback !== false ||
      resident.encoderMemoryReadbackValidated !== false ||
      resident.encoderCrossCacheReadbackValidated !== false ||
      resident.cachePrefixReadbackValidated !== false ||
      resident.appendedCacheReadbackValidated !== false ||
      resident.cacheReadbackFree !== true) {
    fail(`${label} performed a KV cache readback.`);
  }
}

export async function closeRuntimeSampleResources({
  session,
  runtime,
  restoreAdapter,
  operationError = null,
}) {
  const cleanupErrors = [];
  if (session) {
    try {
      await session.close();
    } catch (error) {
      cleanupErrors.push(error);
    }
  }
  if (runtime) {
    try {
      await runtime.close();
    } catch (error) {
      cleanupErrors.push(error);
    }
  }
  try {
    restoreAdapter();
  } catch (error) {
    cleanupErrors.push(error);
  }
  if (cleanupErrors.length > 0) {
    throw new AggregateError(
      operationError ? [operationError, ...cleanupErrors] : cleanupErrors,
      operationError
        ? '[benchmark_explicit_kv_runtime] operation and lifecycle cleanup failed.'
        : '[benchmark_explicit_kv_runtime] lifecycle cleanup failed.',
    );
  }
}

export async function runRuntimeSample(options) {
  const manifestPath = join(options.packageDir, 'package_manifest.json');
  const manifestBytes = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const precision = precisionFromManifest(manifest);
  const api = await import(pathToFileURL(options.apiPath).href);
  const { inputTensor, inputTensorSha256 } = await prepareBenchmarkInput(manifest, options);
  const runtimeInfo = runtimeIdentity(options.backend);
  const compilations = [];
  let activeExecutions = null;
  let runtime;
  let session;
  let restoreAdapter = () => {};
  let operationError = null;
  if (options.backend === 'webgpu') {
    const gpu = typeof navigator === 'undefined' ? null : navigator.gpu;
    if (!gpu) fail('WebGPU is unavailable.');
    restoreAdapter = overrideWebGPUAdapterPreference(gpu, options.adapterPreference);
  }
  try {
    runtime = await api.VolvoxAI.createRuntime({
      backends: [options.backend],
      ...(options.backend === 'wasm' ? { wasmUrl: pathToFileURL(options.wasmPath) } : {}),
      onDiagnostic(event) {
        if (event.kind === 'compilation') {
          compilations.push(event.report);
        } else if (activeExecutions && event.kind === 'execution') {
          activeExecutions.push(event.report);
        }
      },
    });
    session = await TinyReceiptSplitSession.load({
      runtime,
      packageUrl: pathToFileURL(manifestPath),
      fetch: fileFetch,
      compileOptions: {
        backend: {
          mode: 'require', backend: options.backend, operatorFallback: 'forbid',
        },
      },
      snapshotLoader: async ({ weightsUrl, graphUrl }) => {
        const logicalPackage = await api.ModelLoader.load(weightsUrl, {
          graphUrl,
          fetch: fileFetch,
        });
        return api.Model.capture(logicalPackage);
      },
    });
    await session.preload();

    const executeRun = async (label, qualified) => {
      const executions = [];
      activeExecutions = executions;
      const started = performance.now();
      let answer;
      try {
        answer = await session.generate({
          image: inputTensor,
          prompt: options.prompt,
          family: options.family,
          maxNewTokens: options.maxNewTokens,
          shapeMode: 'active',
          preprocessed: true,
          kvTransferMode: options.backend === 'webgpu'
            ? (qualified ? 'device-qualified' : 'device-resident')
            : 'host-validated',
        });
      } finally {
        activeExecutions = null;
      }
      const endToEndInferenceMs = performance.now() - started;
      const tokens = validateExplicitKVAnswer(answer, options.maxNewTokens);
      validateCanonicalBenchmarkAnswer(answer, tokens, options);
      const reports = validateExecutionGroup(executions, tokens, options.backend, label);
      const synchronized = validateSynchronizedTiming(answer, reports, options.backend, label);
      if (options.backend === 'webgpu') {
        validateWebGPUCacheEvidence(answer, tokens.length, qualified, label);
      }
      return Object.freeze({
        label,
        qualified,
        answer,
        tokens: Object.freeze([...tokens]),
        reports,
        synchronized,
        endToEndInferenceMs: finiteMilliseconds(endToEndInferenceMs, `${label} end-to-end inference`),
        signature: answerSignature(answer, tokens),
      });
    };

    const warmupRuns = [];
    for (let index = 0; index < options.executionWarmup; index++) {
      warmupRuns.push(await executeRun(`warmup ${index}`, options.backend === 'webgpu' && index === 0));
    }
    const measured = await executeRun('measured', false);

    if (warmupRuns.some((run) => run.signature !== measured.signature)) {
      fail('same-context warmup changed family, tokens, shape, or cache transitions.');
    }

    if (compilations.length !== 2) fail(`expected two compilations, received ${compilations.length}.`);
    validateCompilation(compilations[0], options.backend, 'encoder');
    validateCompilation(compilations[1], options.backend, 'decoder');
    const allRuns = [...warmupRuns, measured];
    if (allRuns.some((run) =>
      run.reports.encoderContextId !== measured.reports.encoderContextId ||
      run.reports.decoderContextId !== measured.reports.decoderContextId)) {
      fail('warmup and measured requests did not reuse the same encoder/decoder contexts.');
    }
    const selectedDevice = compilations[0].selectedDevice;
    const selectedDeviceKey = canonicalJson(selectedDevice);
    if (selectedDevice == null || canonicalJson(compilations[1].selectedDevice) !== selectedDeviceKey ||
        allRuns.some((run) => [run.reports.encoder, ...run.reports.decoder]
          .some((report) => canonicalJson(report.device) !== selectedDeviceKey))) {
      fail('compilation and execution did not retain one identical device identity.');
    }
    if (options.backend === 'webgpu') {
      requireTinyReceiptPhysicalAdapterIdentity(selectedDevice, options.requiredAdapter);
    }
    const encoderPhases = executionPhaseBreakdown(measured.reports.encoder, 'measured encoder');
    const decoderPhases = measured.reports.decoder.map((report, index) =>
      executionPhaseBreakdown(report, `measured decoder step ${index}`));
    if (options.executionWarmup > 0 &&
        (encoderPhases.specializationCacheHit !== true ||
         decoderPhases.some((phase) => phase.specializationCacheHit !== true))) {
      fail('measured request missed a warmed shape specialization.');
    }
    const answer = measured.answer;
    const tokens = measured.tokens;
    const componentTiming = selectComponentTiming(
      options.backend,
      measured.synchronized,
      encoderPhases,
      decoderPhases,
    );
    const decoderTiming = componentTiming.decoder;
    return Object.freeze({
      schema: SAMPLE_SCHEMA,
      engine: options.backend === 'webgpu' ? 'volvoxai-webgpu' : 'volvoxai-wasm',
      backend: options.backend,
      precision,
      provider: options.backend,
      strictNoFallback: true,
      freshProcess: true,
      family: answer.family,
      familyId: answer.familyId,
      requestedFamily: answer.requestedFamily,
      questionTokenIds: Object.freeze([...answer.questionTokenIds]),
      inputTensorSha256,
      tokenIds: tokens,
      stoppedAtEos: answer.stoppedAtEos,
      shape: {
        mode: answer.shapeMode,
        B: answer.activeShape.B,
        Q: answer.activeShape.Q,
        M: answer.activeShape.M,
        T: answer.activeShape.T,
      },
      cache: {
        initialPastLength: 1,
        finalPastLength: tokens.length + 1,
        sentinelMaskValue: 1,
        transitions: answer.decodeReports.map(({ position, pastLength, presentLength }) => ({
          position, pastLength, presentLength,
        })),
      },
      executionWarmup: {
        runs: options.executionWarmup,
        sameRuntime: true,
        sameSession: true,
        sameEncoderContext: true,
        sameDecoderContext: true,
        stateResetToSentinel: true,
        tokenCacheTransitionParity: true,
        ...(options.backend === 'webgpu' ? {
          deviceResidentKVQualified: true,
          qualificationRun: 'warmup-0',
        } : {}),
      },
      device: selectedDevice,
      ...(options.backend === 'webgpu' ? {
        packedDot4: navigator.gpu.wgslLanguageFeatures
          ?.has('packed_4x8_integer_dot_product') === true,
      } : {}),
      timing: {
        boundary: componentTiming.boundary,
        requiredCompletion: answer.synchronizedTiming.completion,
        encoderExecutionMs: componentTiming.encoderMs,
        decoderSeedMs: decoderTiming.seedMs,
        decoderSteadySteps: decoderTiming.steadySteps,
        decoderSteadyTotalMs: decoderTiming.steadyTotalMs,
        decoderSteadyMeanMs: decoderTiming.steadyMeanMs,
        decoderSteadyTokensPerSecond: decoderTiming.steadyTokensPerSecond,
        decoderExecutionTotalMs: decoderTiming.totalMs,
        decoderStepMs: decoderTiming.steps,
        endToEndInferenceMs: measured.endToEndInferenceMs,
      },
      executionPhases: {
        encoder: encoderPhases,
        decoder: decoderPhases,
      },
      attestation: {
        compilation: Object.freeze([
          compilationAttestation('encoder', compilations[0]),
          compilationAttestation('decoder', compilations[1]),
        ]),
        execution: Object.freeze(allRuns.map((run) => Object.freeze({
          label: run.label,
          qualified: run.qualified,
          signature: run.signature,
          encoder: routeAttestation(run.reports.encoder),
          decoder: Object.freeze(run.reports.decoder.map(routeAttestation)),
        }))),
      },
      ...(options.backend === 'webgpu' ? {
        webgpu: {
          adapterPreference: options.adapterPreference,
          requiredAdapter: options.requiredAdapter,
          backend: runtimeInfo.webgpuBackend ?? null,
          warmupQualification: warmupRuns[0].answer.gpuResidentKv,
          measuredResidency: measured.answer.gpuResidentKv,
          synchronizedTiming: {
            completion: measured.answer.synchronizedTiming.completion,
            applicationValidationIncluded:
              measured.answer.synchronizedTiming.applicationValidationIncluded,
            cacheQualificationReadbackIncluded:
              measured.answer.synchronizedTiming.cacheQualificationReadbackIncluded,
          },
        },
      } : {}),
      artifact: {
        packageManifestSha256: createHash('sha256').update(manifestBytes).digest('hex'),
        api: { name: basename(options.apiPath), sha256: await sha256File(options.apiPath) },
        ...(options.backend === 'wasm'
          ? { wasm: { name: basename(options.wasmPath), sha256: await sha256File(options.wasmPath) } }
          : {}),
      },
      runtime: runtimeInfo,
    });
  } catch (error) {
    operationError = error;
    throw error;
  } finally {
    activeExecutions = null;
    await closeRuntimeSampleResources({ session, runtime, restoreAdapter, operationError });
  }
}

export async function runRuntimeDynamicQualification(options) {
  if (options.qualification !== DYNAMIC_QUALIFICATION_MODE) {
    fail(`dynamic qualification requires --qualification=${DYNAMIC_QUALIFICATION_MODE}.`);
  }
  const manifestPath = join(options.packageDir, 'package_manifest.json');
  const manifestBytes = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const precision = precisionFromManifest(manifest);
  const api = await import(pathToFileURL(options.apiPath).href);
  const { inputTensor, inputTensorSha256 } = await prepareBenchmarkInput(manifest, options);
  const compilations = [];
  const executions = [];
  const executionGroups = [];
  let recording = false;
  let activeRun = null;
  let runtime;
  let session;
  let operationError = null;
  try {
    runtime = await api.VolvoxAI.createRuntime({
      backends: [options.backend],
      ...(options.backend === 'wasm' ? { wasmUrl: pathToFileURL(options.wasmPath) } : {}),
      onDiagnostic(event) {
        if (event.kind === 'compilation') {
          compilations.push(event.report);
        } else if (recording && event.kind === 'execution') {
          executions.push(event.report);
        }
      },
    });
    session = await TinyReceiptSplitSession.load({
      runtime,
      packageUrl: pathToFileURL(manifestPath),
      fetch: fileFetch,
      compileOptions: {
        backend: {
          mode: 'require', backend: options.backend, operatorFallback: 'forbid',
        },
      },
      snapshotLoader: async ({ weightsUrl, graphUrl }) => {
        const logicalPackage = await api.ModelLoader.load(weightsUrl, {
          graphUrl,
          fetch: fileFetch,
        });
        return api.Model.capture(logicalPackage);
      },
    });
    await session.preload();
    recording = true;
    let qualification;
    try {
      qualification = await qualifyDynamicRebind({
        session,
        image: inputTensor,
        prompt: options.prompt,
        family: options.family,
        maximumNewTokens: options.maxNewTokens,
        beforeRun({ label }) {
          if (activeRun !== null) {
            fail('dynamic qualification execution groups overlap.');
          }
          activeRun = Object.freeze({ label, start: executions.length });
        },
        afterRun({ label }) {
          if (activeRun?.label !== label) {
            fail('dynamic qualification execution group is not active.');
          }
          executionGroups.push(Object.freeze({
            label,
            reports: Object.freeze(executions.slice(activeRun.start)),
          }));
          activeRun = null;
        },
      });
    } finally {
      recording = false;
    }
    if (activeRun !== null) fail('dynamic qualification left an execution group open.');
    if (qualification.schema !== DYNAMIC_REBIND_QUALIFICATION_SCHEMA ||
        qualification.timed !== false || qualification.sameSession !== true) {
      fail('dynamic qualification orchestration returned an invalid contract.');
    }
    if (compilations.length !== 2) {
      fail(`expected two compilations, received ${compilations.length}.`);
    }
    validateCompilation(compilations[0], options.backend, 'encoder');
    validateCompilation(compilations[1], options.backend, 'decoder');
    const executionEvidence = validateDynamicExecutionGroups(
      executionGroups,
      qualification.runs,
      options.backend,
    );
    return Object.freeze({
      schema: DYNAMIC_REBIND_QUALIFICATION_SCHEMA,
      engine: 'volvoxai-wasm',
      backend: options.backend,
      precision,
      provider: options.backend,
      timed: false,
      freshProcess: true,
      sameSession: true,
      sameRuntime: true,
      sameEncoderContext: true,
      sameDecoderContext: true,
      strictNoFallback: true,
      boundedDecoderMaximumNewTokens: qualification.boundedDecoderMaximumNewTokens,
      maximumLegalEncoderBinding: qualification.maximumLegalEncoderBinding,
      inputTensorSha256,
      prompt: options.prompt,
      family: options.family,
      checks: qualification.checks,
      runs: qualification.runs,
      routeEvidence: Object.freeze({
        compilation: Object.freeze([...compilations]),
        execution: executionEvidence.attestations,
      }),
      artifact: Object.freeze({
        packageManifestSha256: createHash('sha256').update(manifestBytes).digest('hex'),
        api: Object.freeze({
          name: basename(options.apiPath), sha256: await sha256File(options.apiPath),
        }),
        wasm: Object.freeze({
          name: basename(options.wasmPath), sha256: await sha256File(options.wasmPath),
        }),
      }),
    });
  } catch (error) {
    operationError = error;
    throw error;
  } finally {
    recording = false;
    await closeRuntimeSampleResources({
      session,
      runtime,
      restoreAdapter: () => {},
      operationError,
    });
  }
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArguments(argv);
  const qualification = options.qualification === DYNAMIC_QUALIFICATION_MODE;
  const result = qualification
    ? await runRuntimeDynamicQualification(options)
    : await runRuntimeSample(options);
  const prefix = qualification ? DYNAMIC_QUALIFICATION_PREFIX : SAMPLE_PREFIX;
  process.stdout.write(`${prefix}${JSON.stringify(result)}\n`);
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (invokedPath === import.meta.url) {
  main().catch((error) => {
    console.error(error?.stack || String(error));
    process.exitCode = 1;
  });
}
