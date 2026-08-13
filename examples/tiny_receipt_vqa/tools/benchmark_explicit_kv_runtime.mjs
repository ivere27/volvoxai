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

export {
  executionPhaseBreakdown,
  finiteMilliseconds,
  summarizeDecoderSteps,
  validateExplicitKVAnswer,
} from './benchmark_explicit_kv_contract.mjs';

export const SAMPLE_SCHEMA = 'volvoxai.tiny-receipt-explicit-kv-runtime-sample/v1';
export const KV_PACKAGE_FORMAT = 'volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v1';
export const SAMPLE_PREFIX = 'TINYRECEIPT_EXPLICIT_KV_SAMPLE ';
export const DYNAMIC_QUALIFICATION_PREFIX =
  'TINYRECEIPT_DYNAMIC_REBIND_QUALIFICATION ';

const DYNAMIC_QUALIFICATION_MODE = 'dynamic-rebind';
const DYNAMIC_QUALIFICATION_PROMPT = 'phone number last one';

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
    'api', 'backend', 'family', 'image', 'max-new', 'package', 'prompt',
    'qualification', 'wasm',
  ]);
  for (const name of Object.keys(result)) {
    if (!allowed.has(name)) fail(`unknown option '--${name}'.`);
  }
  for (const name of ['package', 'image']) {
    if (typeof result[name] !== 'string' || result[name].length === 0) {
      fail(`--${name} is required.`);
    }
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
  if (backend !== 'wasm') fail("--backend must be 'wasm'; JS CPU is outside this benchmark.");
  const maxNewTokens = Number(result['max-new'] ?? 4);
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 2 || maxNewTokens > 191) {
    fail('--max-new must be an integer in [2, 191].');
  }
  if (qualification !== null && maxNewTokens !== 4) {
    fail('dynamic-rebind qualification requires --max-new=4.');
  }
  return Object.freeze({
    packageDir: resolve(result.package),
    imagePath: resolve(result.image),
    prompt: (result.prompt ?? DYNAMIC_QUALIFICATION_PROMPT).normalize('NFC'),
    family: result.family ?? 'phone',
    backend,
    maxNewTokens,
    qualification,
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

function validateCompilation(report, backend, label) {
  if (report?.requestedPolicy?.mode !== 'require' ||
      report.requestedPolicy.backend !== backend ||
      report.requestedPolicy.operatorFallback !== 'forbid' ||
      report.selectedBackend !== backend || !Array.isArray(report.candidates) ||
      report.candidates.length !== 1 || report.candidates[0]?.backend !== backend ||
      report.candidates[0].outcome !== 'selected') {
    fail(`${label} compilation did not strictly select ${backend}.`);
  }
  strictRoute(report.routeEvidence, `${label} compilation`);
  strictRoute(report.candidates[0].routeEvidence, `${label} compilation candidate`);
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

function precisionFromManifest(manifest) {
  if (manifest?.format !== KV_PACKAGE_FORMAT ||
      manifest?.generation?.strategy !== 'greedy-autoregressive-explicit-kv') {
    fail(`package must use '${KV_PACKAGE_FORMAT}'.`);
  }
  if (manifest?.source?.variant === 'fp32') return 'fp32';
  if (manifest?.source?.variant === 'int8-w8a8') return 'int8';
  fail('package source.variant must be fp32 or int8-w8a8.');
}

export async function runRuntimeSample(options) {
  const manifestPath = join(options.packageDir, 'package_manifest.json');
  const manifestBytes = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const precision = precisionFromManifest(manifest);
  const api = await import(pathToFileURL(options.apiPath).href);
  const image = decodeImage(options.imagePath);
  const resize = manifest?.preprocessing?.resize;
  if (!Number.isInteger(resize?.width) || !Number.isInteger(resize?.height)) {
    fail('package preprocessing must declare integer resize dimensions.');
  }
  const inputTensor = await preprocessTinyReceiptImage(image, {
    width: resize.width,
    height: resize.height,
  });
  const inputTensorSha256 = sha256Float32LE(inputTensor);
  const compilations = [];
  const executions = [];
  let measuring = false;
  let runtime;
  let session;
  try {
    runtime = await api.VolvoxAI.createRuntime({
      backends: [options.backend],
      ...(options.backend === 'wasm' ? { wasmUrl: pathToFileURL(options.wasmPath) } : {}),
      onDiagnostic(event) {
        if (event.kind === 'compilation') {
          compilations.push(event.report);
        } else if (measuring && event.kind === 'execution') {
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
    measuring = true;
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
      });
    } finally {
      measuring = false;
    }
    const endToEndInferenceMs = performance.now() - started;
    const tokens = validateExplicitKVAnswer(answer, options.maxNewTokens);

    if (compilations.length !== 2) fail(`expected two compilations, received ${compilations.length}.`);
    validateCompilation(compilations[0], options.backend, 'encoder');
    validateCompilation(compilations[1], options.backend, 'decoder');
    const contextIds = [...new Set(executions.map((report) => report.contextId))];
    if (contextIds.length !== 2 || executions[0]?.contextId !== contextIds[0]) {
      fail('execution diagnostics do not identify one encoder and one decoder context.');
    }
    const encoder = executions.filter((report) => report.contextId === contextIds[0]);
    const decoder = executions.filter((report) => report.contextId === contextIds[1]);
    if (encoder.length !== 1 || decoder.length !== tokens.length ||
        executions.length !== tokens.length + 1) {
      fail('execution diagnostics do not contain one encoder and one decoder call per token.');
    }
    validateExecution(encoder[0], options.backend, 'encoder');
    decoder.forEach((report, index) =>
      validateExecution(report, options.backend, `decoder step ${index}`));
    const encoderPhases = executionPhaseBreakdown(encoder[0], 'encoder');
    const decoderPhases = decoder.map((report, index) =>
      executionPhaseBreakdown(report, `decoder step ${index}`));
    const decoderTiming = summarizeDecoderSteps(
      decoder.map((report) => report.executionTimeMs),
    );
    return Object.freeze({
      schema: SAMPLE_SCHEMA,
      engine: 'volvoxai-wasm',
      backend: options.backend,
      precision,
      provider: options.backend,
      strictNoFallback: true,
      family: answer.family,
      familyId: answer.familyId,
      requestedFamily: answer.requestedFamily,
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
      timing: {
        encoderExecutionMs: finiteMilliseconds(
          encoder[0].executionTimeMs,
          'encoder execution',
        ),
        decoderSeedMs: decoderTiming.seedMs,
        decoderSteadySteps: decoderTiming.steadySteps,
        decoderSteadyTotalMs: decoderTiming.steadyTotalMs,
        decoderSteadyMeanMs: decoderTiming.steadyMeanMs,
        decoderSteadyTokensPerSecond: decoderTiming.steadyTokensPerSecond,
        decoderExecutionTotalMs: decoderTiming.totalMs,
        decoderStepMs: decoderTiming.steps,
        endToEndInferenceMs: finiteMilliseconds(endToEndInferenceMs, 'end-to-end inference'),
      },
      executionPhases: {
        encoder: encoderPhases,
        decoder: decoderPhases,
      },
      artifact: {
        packageManifestSha256: createHash('sha256').update(manifestBytes).digest('hex'),
        api: { name: basename(options.apiPath), sha256: await sha256File(options.apiPath) },
        ...(options.backend === 'wasm'
          ? { wasm: { name: basename(options.wasmPath), sha256: await sha256File(options.wasmPath) } }
          : {}),
      },
    });
  } finally {
    measuring = false;
    if (session) await session.close();
    if (runtime) await runtime.close();
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
  const image = decodeImage(options.imagePath);
  const resize = manifest?.preprocessing?.resize;
  if (!Number.isInteger(resize?.width) || !Number.isInteger(resize?.height)) {
    fail('package preprocessing must declare integer resize dimensions.');
  }
  const inputTensor = await preprocessTinyReceiptImage(image, {
    width: resize.width,
    height: resize.height,
  });
  const inputTensorSha256 = sha256Float32LE(inputTensor);
  const compilations = [];
  const executions = [];
  const executionGroups = [];
  let recording = false;
  let activeRun = null;
  let runtime;
  let session;
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
  } finally {
    recording = false;
    if (session) await session.close();
    if (runtime) await runtime.close();
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
