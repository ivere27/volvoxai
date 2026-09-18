#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rmdir, stat, unlink, writeFile } from 'node:fs/promises';
import { platform } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const SCRIPT = fileURLToPath(import.meta.url);
const SCHEMA = 'volvoxai.native-w8a8-release-performance/v3';
const FINALIZATION_SCHEMA = 'volvoxai-native-release-finalization/v1';
const HISTORICAL_ANCHOR = '478e82a78eaba2c8f4112476b325727b66a92420';
const PERFORMANCE_TARGET = 'volvoxai_native_w8a8_performance';
const PERFORMANCE_LOCK_DIRECTORY = path.join(
  ROOT, 'build', 'performance', '.native-w8a8-release.lock',
);
const TARGETS = Object.freeze({
  baseline: 'volvoxai_native_w8a8_baseline',
  candidate: 'volvoxai_native_w8a8_candidate',
});
const CLI_ANCHOR_SYMBOL = 'volvoxai_native_w8a8_release_cli_anchor';
const EXPECTED_HOT_SPAN_COUNT = 47;
const GRAYSCALE_SINGLE_FIXED_WORK = Object.freeze({
  formulaVersion: 'shared-fastest-warmup-v1',
  metric: 'grayscaleStemQconvSingle',
  spanLabel: 'grayscale-qconv-single',
  targetMilliseconds: 200,
  headroom: 1.10,
  maximumIterations: 100_000_000,
});

export function grayscaleSingleFixedWorkPolicy(minimumTimedMs) {
  if (!Number.isFinite(minimumTimedMs) || minimumTimedMs <= 0) {
    throw new Error('minimum timed span must be positive');
  }
  return {
    ...GRAYSCALE_SINGLE_FIXED_WORK,
    targetMilliseconds: Math.max(
      GRAYSCALE_SINGLE_FIXED_WORK.targetMilliseconds,
      minimumTimedMs,
    ),
  };
}
const REQUIRED_OUTPUT_MARKERS = Object.freeze([
  'Timing policy: gate=on',
  'Timing affinity: single-cpu=',
  'CPU features:',
  'TinyReceipt weighted dense total:',
  'TinyReceipt full-prefill weighted base-dense subset:',
  'TinyReceipt quantized activation proxy',
  'TinyReceipt grayscale stem QConv2D',
  'QConv2D W8A8:',
]);
const HOT_METRIC_PATTERNS = Object.freeze({
  decoderRowRaw:
    /TinyReceipt weighted dense total:.*?W8A8-current-M1-policy\(raw\)=([0-9]+(?:\.[0-9]+)?) ms/mu,
  decoderRowPacked:
    /TinyReceipt weighted dense total:.*?W8A8-all-packed=([0-9]+(?:\.[0-9]+)?) ms/mu,
  decoderPrefillPacked:
    /TinyReceipt full-prefill weighted base-dense subset: packed=([0-9]+(?:\.[0-9]+)?) ms/mu,
  decoderPrefillRaw1t:
    /TinyReceipt full-prefill weighted base-dense subset:.*?raw\[1t\]=([0-9]+(?:\.[0-9]+)?) ms/mu,
  decoderPrefillRaw4t:
    /TinyReceipt full-prefill weighted base-dense subset:.*?raw\[4t\]=([0-9]+(?:\.[0-9]+)?) ms/mu,
  activationRouterKernel:
    /^  QGELU router .*?kernel=([0-9]+(?:\.[0-9]+)?) ms/mu,
  activationStemKernel:
    /^  QSiLU stem .*?kernel=([0-9]+(?:\.[0-9]+)?) ms/mu,
  activationEncoderKernel:
    /^  QGELU encoder .*?kernel=([0-9]+(?:\.[0-9]+)?) ms/mu,
  activationDecoderPrefillKernel:
    /^  QGELU decoder prefill .*?kernel=([0-9]+(?:\.[0-9]+)?) ms/mu,
  activationDecoderRowKernel:
    /^  QGELU decoder row .*?kernel=([0-9]+(?:\.[0-9]+)?) ms/mu,
  grayscaleStemQconvSingle:
    /TinyReceipt grayscale stem QConv2D .*?single\[[^\]]+\]=([0-9]+(?:\.[0-9]+)?) ms/mu,
  grayscaleStemQconvTiled4t:
    /TinyReceipt grayscale stem QConv2D .*?tiled\[[^\]]+,4t\]=([0-9]+(?:\.[0-9]+)?) ms/mu,
  genericQconvSingle:
    /^QConv2D W8A8:.*?single\[[^\]]+\]=([0-9]+(?:\.[0-9]+)?) ms/mu,
  genericQconvTiled4t:
    /^QConv2D W8A8:.*?tiled\[[^\]]+,4t\]=([0-9]+(?:\.[0-9]+)?) ms/mu,
});
export const HARD_METRICS = Object.freeze(Object.keys(HOT_METRIC_PATTERNS));

function stringArgument(name, fallback) {
  const prefix = `--${name}=`;
  const matches = process.argv.slice(2).filter((value) => value.startsWith(prefix));
  if (matches.length > 1) throw new Error(`--${name} may be specified only once`);
  const value = matches.length === 0 ? fallback : matches[0].slice(prefix.length);
  if (!value) throw new Error(`--${name} must not be empty`);
  return value;
}

function numberArgument(name, fallback, minimum, maximum, integer = false) {
  const value = Number(stringArgument(name, String(fallback)));
  const valid = Number.isFinite(value) && value >= minimum && value <= maximum &&
    (!integer || Number.isSafeInteger(value));
  if (!valid) {
    const kind = integer ? 'an integer' : 'a number';
    throw new Error(`--${name} must be ${kind} from ${minimum} through ${maximum}`);
  }
  return value;
}

function rejectUnknownArguments() {
  const known = new Set([
    'native-build-dir', 'blocks', 'max-blocks', 'min-timed-ms',
    'max-regression-percent', 'bootstrap-resamples', 'timeout-ms',
    'build-jobs', 'output', 'cpu-list',
  ]);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/u.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function parseOptions() {
  rejectUnknownArguments();
  const blocks = numberArgument('blocks', 4, 2, 8, true);
  const maxBlocks = numberArgument('max-blocks', 40, blocks, 40, true);
  if (blocks % 2 !== 0 || maxBlocks % 2 !== 0) {
    throw new Error('--blocks and --max-blocks must be even for balanced order');
  }
  return {
    buildDirectory: path.resolve(stringArgument(
      'native-build-dir',
      process.env.CMAKE_BUILD_DIR ?? 'build/cmake',
    )),
    blocks,
    maxBlocks,
    minimumTimedMs: numberArgument('min-timed-ms', 50, 10, 1000),
    maxRegressionPercent: numberArgument('max-regression-percent', 2, 0, 100),
    bootstrapResamples: numberArgument('bootstrap-resamples', 10_000, 1000, 100_000, true),
    timeoutMs: numberArgument('timeout-ms', 300_000, 1000, 900_000, true),
    buildJobs: numberArgument('build-jobs', 2, 1, 256, true),
    output: path.resolve(stringArgument(
      'output',
      'build/performance/native-w8a8-release.json',
    )),
    cpuListOverride: stringArgument('cpu-list', 'auto'),
  };
}

function checkedSpawn(command, args, options, label) {
  const child = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    ...options,
  });
  if (child.error) {
    const detail = child.error.code === 'ETIMEDOUT' ? ' (timed out)' : '';
    throw new Error(`${label} failed${detail}: ${child.error.message}`);
  }
  if (child.status !== 0) {
    const diagnostics = [
      child.stderr?.trim() ? `stderr:\n${child.stderr.trim()}` : null,
      child.stdout?.trim() ? `stdout:\n${child.stdout.trim()}` : null,
    ].filter(Boolean);
    const detail = diagnostics.join('\n') ||
      `exit ${child.status}${child.signal ? `, signal ${child.signal}` : ''}`;
    throw new Error(`${label} failed: ${detail}`);
  }
  return child;
}

function sha256(contents) {
  return createHash('sha256').update(contents).digest('hex');
}

async function fileEvidence(filename) {
  const [contents, metadata] = await Promise.all([
    readFile(filename),
    stat(filename, { bigint: true }),
  ]);
  return {
    bytes: contents.byteLength,
    sha256: sha256(contents),
    modifiedAt: metadata.mtime.toISOString(),
    modifiedTimeNs: metadata.mtimeNs.toString(),
  };
}

function displayPath(filename) {
  const relative = path.relative(ROOT, filename);
  if (relative === '') return '.';
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    return filename;
  }
  return relative.split(path.sep).join('/');
}

async function atomicWriteJson(filename, value) {
  await mkdir(path.dirname(filename), { recursive: true });
  const temporary = `${filename}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  await rename(temporary, filename);
}

export async function acquirePerformanceLock(options, lockDirectory = PERFORMANCE_LOCK_DIRECTORY) {
  const token = `${process.pid}-${Date.now()}-${randomUUID()}`;
  const ownerPath = path.join(lockDirectory, 'owner.json');
  const owner = {
    token,
    pid: process.pid,
    startedAt: new Date().toISOString(),
    buildDirectory: displayPath(options.buildDirectory),
    output: displayPath(options.output),
  };
  await mkdir(path.dirname(lockDirectory), { recursive: true });
  try {
    await mkdir(lockDirectory);
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    let existing = 'owner metadata unavailable';
    try {
      existing = (await readFile(ownerPath, 'utf8')).trim();
    } catch {
      // The owner may still be writing its metadata. The directory itself is
      // the atomic lock authority, so absence of this diagnostic is harmless.
    }
    throw new Error(
      `native W8A8 performance gate is already locked at ` +
      `${displayPath(lockDirectory)} (${existing})`,
    );
  }
  try {
    await writeFile(ownerPath, `${JSON.stringify(owner, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
  } catch (error) {
    await rmdir(lockDirectory);
    throw error;
  }
  let released = false;
  return {
    evidence: {
      path: displayPath(lockDirectory),
      pid: owner.pid,
      startedAt: owner.startedAt,
    },
    async release() {
      if (released) throw new Error('native W8A8 performance lock was released twice');
      const recorded = JSON.parse(await readFile(ownerPath, 'utf8'));
      if (recorded.token !== token) {
        throw new Error('native W8A8 performance lock ownership changed unexpectedly');
      }
      await unlink(ownerPath);
      await rmdir(lockDirectory);
      released = true;
    },
  };
}

function tokenizeCommand(command, source) {
  const tokens = (command.match(/[^\s"']+|"[^"]*"|'[^']*'/gu) || []).map(
    (token) => (/^(["']).*\1$/u.test(token) ? token.slice(1, -1) : token),
  );
  if (tokens.length === 0 || tokens.some((token) => /["']/u.test(token))) {
    throw new Error(`link command '${source}' uses unsupported quoting`);
  }
  if (tokens.some((token) => token.startsWith('@'))) {
    throw new Error(`link command '${source}' uses an unauditable response file`);
  }
  return tokens;
}

function targetFromObject(token) {
  return /CMakeFiles\/([^/]+)\.dir\//u.exec(token)?.[1] ?? null;
}

export function logicalObjectSource(token) {
  const marker = '.dir/';
  const index = token.indexOf(marker);
  if (index < 0 || !token.endsWith('.o')) return null;
  return token.slice(index + marker.length).replaceAll('\\', '/');
}

export function classifyPerformanceObject(token, variant) {
  const target = targetFromObject(token);
  if (!target) return 'unknown';
  if (target === 'volvoxai_native_w8a8_benchmark_main') return 'benchmark-main';
  if (variant === 'candidate' &&
      target === 'volvoxai_native_w8a8_candidate_cli_anchor') {
    return 'candidate-cli-anchor';
  }
  if (variant === 'baseline' &&
      target === 'volvoxai_native_w8a8_baseline_cli_anchor') {
    return 'baseline-cli-anchor';
  }
  if (target === 'volvox_release_arm_dotprod') return 'arm-dotprod';
  if (target === 'volvox_release_arm_i8mm') return 'arm-i8mm';
  if (variant === 'candidate' &&
      /^volvoxai_inference_release_.*_hot_objects$/u.test(target)) return 'hot';
  if (variant === 'candidate' &&
      /^volvoxai_inference_release_.*_cold_objects$/u.test(target)) return 'cold';
  if (variant === 'baseline' &&
      /^volvoxai_native_w8a8_baseline_(?:engine|shader)_objects$/u.test(target)) {
    return 'baseline-o3';
  }
  return 'unknown';
}

async function inspectFlags(nativeBuildDirectory, classifiedObjects) {
  const targetCategories = new Map();
  for (const object of classifiedObjects) {
    const target = targetFromObject(object.token);
    if (!target) throw new Error(`cannot identify target for '${object.token}'`);
    const previous = targetCategories.get(target);
    if (previous && previous !== object.category) {
      throw new Error(`target '${target}' spans multiple performance categories`);
    }
    targetCategories.set(target, object.category);
  }
  const evidence = {};
  for (const [target, category] of targetCategories) {
    const filename = path.join(nativeBuildDirectory, 'CMakeFiles', `${target}.dir`, 'flags.make');
    const contents = await readFile(filename, 'utf8');
    const definitions = /^C_DEFINES\s*=\s*(.*)$/mu.exec(contents)?.[1] ?? '';
    const includes = /^C_INCLUDES\s*=\s*(.*)$/mu.exec(contents)?.[1] ?? '';
    const flags = /^C_FLAGS\s*=\s*(.*)$/mu.exec(contents)?.[1] ?? '';
    const optimizations = flags.match(/(?:^|\s)(-O(?:0|1|2|3|s|z|fast))(?=\s|$)/gu)
      ?.map((value) => value.trim()) ?? [];
    const effective = optimizations.at(-1) ?? null;
    const expected = category === 'cold' || category === 'candidate-cli-anchor'
      ? new Set(['-Os', '-Oz'])
      : category === 'benchmark-main' ? new Set(['-O2'])
        : new Set(['-O3']);
    if (!effective || !expected.has(effective)) {
      throw new Error(
        `performance target '${target}' has ${effective ?? 'no optimization'}, ` +
        `expected ${[...expected].join(' or ')}`,
      );
    }
    if (!/(?:^|\s)-ffunction-sections(?:\s|$)/u.test(flags) ||
        !/(?:^|\s)-fdata-sections(?:\s|$)/u.test(flags)) {
      throw new Error(`performance target '${target}' lacks function/data sections`);
    }
    const isAnchor = category === 'candidate-cli-anchor' ||
      category === 'baseline-cli-anchor';
    const hasAnchorRename = definitions.split(/\s+/u)
      .includes(`-Dmain=${CLI_ANCHOR_SYMBOL}`);
    if (isAnchor !== hasAnchorRename) {
      throw new Error(
        `performance target '${target}' has an invalid CLI-anchor main definition`,
      );
    }
    evidence[target] = {
      category,
      path: displayPath(filename),
      effectiveOptimization: effective,
      definitions,
      includes,
      flags,
      ...(await fileEvidence(filename)),
    };
  }
  return evidence;
}

async function inspectPerformanceGraph(buildDirectory, variant) {
  const target = TARGETS[variant];
  const nativeBuildDirectory = path.join(buildDirectory, 'native');
  const linkFile = path.join(nativeBuildDirectory, 'CMakeFiles', `${target}.dir`, 'link.txt');
  const linkCommand = (await readFile(linkFile, 'utf8')).trim();
  const tokens = tokenizeCommand(linkCommand, linkFile);
  const compiler = tokens[0];
  const linkArguments = tokens.slice(1);
  if (linkArguments.filter((token) => token === '-o').length !== 1) {
    throw new Error(`performance link '${linkFile}' must have exactly one output`);
  }
  if (!linkArguments.includes('-Wl,--gc-sections') ||
      !linkArguments.includes('-Wl,--build-id=sha1')) {
    throw new Error(`performance link '${linkFile}' lacks GC or deterministic build ID`);
  }
  if (linkArguments.some((token) => /^-Wl,(?:-Map|--Map)/u.test(token))) {
    throw new Error(`performance link '${linkFile}' must not write a release map`);
  }
  const anchorRootOption = `-Wl,-u,${CLI_ANCHOR_SYMBOL}`;
  if (linkArguments.filter((token) => token === anchorRootOption).length !== 1) {
    throw new Error(`${variant} performance link must root exactly one CLI anchor`);
  }
  const outputIndex = linkArguments.indexOf('-o');
  const outputToken = linkArguments[outputIndex + 1];
  const executable = path.resolve(nativeBuildDirectory, outputToken);
  const objects = linkArguments.filter((token) => token.endsWith('.o'));
  const classifiedObjects = objects.map((token) => ({
    token,
    category: classifyPerformanceObject(token, variant),
    logicalSource: logicalObjectSource(token),
  }));
  const unknown = classifiedObjects.filter(({ category }) => category === 'unknown');
  if (unknown.length !== 0) {
    throw new Error(
      `${variant} performance graph has unclassified objects: ` +
      unknown.map(({ token }) => token).join(', '),
    );
  }
  if (classifiedObjects.at(-1)?.category !== 'benchmark-main' ||
      classifiedObjects.filter(({ category }) => category === 'benchmark-main').length !== 1) {
    throw new Error(`${variant} performance graph must keep the shared benchmark main last`);
  }
  const anchorCategory = `${variant}-cli-anchor`;
  if (classifiedObjects.at(-2)?.category !== anchorCategory ||
      classifiedObjects.filter(({ category }) => category === anchorCategory).length !== 1 ||
      !classifiedObjects.at(-2)?.logicalSource?.endsWith('cli/main.c.o')) {
    throw new Error(
      `${variant} performance graph must keep exactly one ${anchorCategory} before main`,
    );
  }
  if (variant === 'candidate' &&
      (!classifiedObjects.some(({ category }) => category === 'hot') ||
       !classifiedObjects.some(({ category }) => category === 'cold'))) {
    throw new Error('candidate performance graph is not the selective hot/cold release graph');
  }
  if (variant === 'baseline' &&
      !classifiedObjects.some(({ category }) => category === 'baseline-o3')) {
    throw new Error('baseline performance graph has no all-O3 reference objects');
  }
  const [flags, executableEvidence, linkEvidence, objectEvidence] = await Promise.all([
    inspectFlags(nativeBuildDirectory, classifiedObjects),
    fileEvidence(executable),
    fileEvidence(linkFile),
    Promise.all(classifiedObjects.map(async (object) => ({
      ...object,
      ...(await fileEvidence(path.resolve(nativeBuildDirectory, object.token))),
    }))),
  ]);
  const digest = createHash('sha256');
  for (const object of objectEvidence) {
    digest.update(
      `${object.logicalSource}\0${object.category}\0${object.bytes}\0${object.sha256}\n`,
    );
  }
  return {
    variant,
    target,
    compiler,
    nativeBuildDirectory,
    linkFile,
    linkCommand,
    linkInterfaceArguments: linkArguments.filter((token, index) =>
      !token.endsWith('.o') && token !== '-o' && linkArguments[index - 1] !== '-o'),
    link: { path: displayPath(linkFile), ...linkEvidence },
    executable,
    executableEvidence: { path: displayPath(executable), ...executableEvidence },
    logicalSources: classifiedObjects.map(({ logicalSource }) => logicalSource),
    objectGraphSha256: digest.digest('hex'),
    objects: objectEvidence,
    flags,
  };
}

function performanceGraphIntegritySnapshot(graph) {
  return {
    variant: graph.variant,
    target: graph.target,
    compiler: graph.compiler,
    nativeBuildDirectory: graph.nativeBuildDirectory,
    linkFile: graph.linkFile,
    linkCommand: graph.linkCommand,
    linkInterfaceArguments: graph.linkInterfaceArguments,
    link: graph.link,
    executable: graph.executable,
    executableEvidence: graph.executableEvidence,
    logicalSources: graph.logicalSources,
    objectGraphSha256: graph.objectGraphSha256,
    objects: graph.objects,
    flags: graph.flags,
  };
}

export function performanceGraphIntegritySha256(graph) {
  return sha256(JSON.stringify(performanceGraphIntegritySnapshot(graph)));
}

export function assertPerformanceGraphUnchanged(before, after, stage) {
  const beforeSha256 = performanceGraphIntegritySha256(before);
  const afterSha256 = performanceGraphIntegritySha256(after);
  if (beforeSha256 !== afterSha256) {
    throw new Error(
      `${before.variant} performance graph changed ${stage}: ` +
      `${beforeSha256} -> ${afterSha256}`,
    );
  }
  return beforeSha256;
}

function assertMatchingLogicalOrder(baseline, candidate) {
  if (baseline.compiler !== candidate.compiler ||
      JSON.stringify(baseline.linkInterfaceArguments) !==
        JSON.stringify(candidate.linkInterfaceArguments)) {
    throw new Error('baseline/candidate linker interface differs beyond objects and output');
  }
  const baselineOrder = JSON.stringify(baseline.logicalSources);
  const candidateOrder = JSON.stringify(candidate.logicalSources);
  if (baselineOrder !== candidateOrder) {
    const mismatch = baseline.logicalSources.findIndex(
      (source, index) => source !== candidate.logicalSources[index],
    );
    throw new Error(
      `baseline/candidate logical object order differs at index ${mismatch}: ` +
      `${baseline.logicalSources[mismatch] ?? '<missing>'} vs ` +
      `${candidate.logicalSources[mismatch] ?? '<missing>'}`,
    );
  }
  const mainSource = baseline.logicalSources.at(-1);
  if (!mainSource?.endsWith('benchmark_w8a8_native.c.o')) {
    throw new Error('shared benchmark main is not the final logical source');
  }
  if (!baseline.logicalSources.at(-2)?.endsWith('cli/main.c.o')) {
    throw new Error('CLI anchor is not immediately before the shared benchmark main');
  }
  const digest = createHash('sha256');
  for (const source of baseline.logicalSources) digest.update(`${source}\n`);
  return digest.digest('hex');
}

async function inspectReleaseProjection(buildDirectory, candidate) {
  const nativeBuildDirectory = path.join(buildDirectory, 'native');
  const linkFile = path.join(
    nativeBuildDirectory,
    'CMakeFiles',
    'volvoxai.dir',
    'link.txt',
  );
  const linkCommand = (await readFile(linkFile, 'utf8')).trim();
  const objects = tokenizeCommand(linkCommand, linkFile).filter(
    (token) => token.endsWith('.o'),
  );
  const cliObjects = objects.filter(
    (token) => targetFromObject(token) === 'volvoxai_inference_release_cli_object',
  );
  if (cliObjects.length !== 1 || objects.at(-1) !== cliObjects[0]) {
    throw new Error('inference release link must keep exactly one CLI object last');
  }
  const releaseCoreObjects = objects.slice(0, -1);
  const candidateCoreObjects = candidate.objects.slice(0, -2).map(({ token }) => token);
  if (JSON.stringify(releaseCoreObjects) !== JSON.stringify(candidateCoreObjects)) {
    const mismatch = releaseCoreObjects.findIndex(
      (token, index) => token !== candidateCoreObjects[index],
    );
    throw new Error(
      `candidate graph differs from the release core projection at object ${mismatch}: ` +
      `${releaseCoreObjects[mismatch] ?? '<missing>'} vs ` +
      `${candidateCoreObjects[mismatch] ?? '<missing>'}`,
    );
  }
  const candidateAnchor = candidate.objects.at(-2);
  const releaseCliSource = logicalObjectSource(cliObjects[0]);
  if (candidateAnchor?.category !== 'candidate-cli-anchor' ||
      candidateAnchor.logicalSource !== releaseCliSource) {
    throw new Error('candidate CLI anchor does not project the production release CLI source');
  }
  const releaseCliFlagsByTarget = await inspectFlags(nativeBuildDirectory, [{
    token: cliObjects[0],
    category: 'cold',
  }]);
  const releaseCliTarget = targetFromObject(cliObjects[0]);
  const candidateAnchorTarget = targetFromObject(candidateAnchor.token);
  const releaseCliFlags = releaseCliFlagsByTarget[releaseCliTarget];
  const candidateAnchorFlags = candidate.flags[candidateAnchorTarget];
  const normalizedDefinitions = (value) => value.split(/\s+/u)
    .filter((item) => item && item !== `-Dmain=${CLI_ANCHOR_SYMBOL}`)
    .sort();
  if (candidateAnchorFlags.flags !== releaseCliFlags.flags ||
      candidateAnchorFlags.includes !== releaseCliFlags.includes ||
      JSON.stringify(normalizedDefinitions(candidateAnchorFlags.definitions)) !==
        JSON.stringify(normalizedDefinitions(releaseCliFlags.definitions))) {
    throw new Error('candidate CLI anchor compile profile differs from the release CLI');
  }
  return {
    link: { path: displayPath(linkFile), ...(await fileEvidence(linkFile)) },
    coreObjectCount: releaseCoreObjects.length,
    coreLogicalSources: releaseCoreObjects.map(logicalObjectSource),
    cliObject: {
      token: cliObjects[0],
      logicalSource: releaseCliSource,
      flags: releaseCliFlags,
    },
    candidateCliAnchor: {
      token: candidateAnchor.token,
      logicalSource: candidateAnchor.logicalSource,
      bytes: candidateAnchor.bytes,
      sha256: candidateAnchor.sha256,
      flags: candidateAnchorFlags,
    },
    candidateCoreProjectionVerified: true,
    cliReachabilityProjectionVerified: true,
  };
}

function canonicalNmHex(value, label) {
  if (!/^[0-9a-f]+$/iu.test(value)) {
    throw new Error(`nm reported invalid ${label} '${value}'`);
  }
  return value.toLowerCase().replace(/^0+(?=[0-9a-f])/u, '');
}

export function parsePosixNmTextSymbols(output) {
  const symbols = [];
  for (const line of output.split(/\r?\n/u)) {
    if (line.trim() === '') continue;
    const fields = line.trim().split(/\s+/u);
    if (fields.length < 2 || !/^[tTwW]$/u.test(fields[1])) continue;
    if (fields.length < 3) throw new Error(`nm text symbol has no address: '${line}'`);
    symbols.push({
      name: fields[0],
      type: fields[1],
      address: canonicalNmHex(fields[2], 'code address'),
      size: fields[3] == null ? null : canonicalNmHex(fields[3], 'code size'),
    });
  }
  if (symbols.length === 0) throw new Error('nm reported no defined text symbols');
  return symbols;
}

export function parsePosixNmSymbols(output, requiredNames) {
  const required = new Set(requiredNames);
  if (required.size === 0 || required.size !== requiredNames.length) {
    throw new Error('required nm symbol names must be a non-empty unique list');
  }
  const matches = new Map([...required].map((name) => [name, []]));
  for (const line of output.split(/\r?\n/u)) {
    const fields = line.trim().split(/\s+/u);
    if (fields.length < 3 || !required.has(fields[0])) continue;
    matches.get(fields[0]).push({
      name: fields[0],
      type: fields[1],
      address: canonicalNmHex(fields[2], 'code address'),
      size: fields[3] == null ? null : canonicalNmHex(fields[3], 'code size'),
    });
  }
  return Object.fromEntries([...required].map((name) => {
    const entries = matches.get(name);
    if (entries.length !== 1) {
      throw new Error(
        `nm must report exactly one defined '${name}' symbol; found ${entries.length}`,
      );
    }
    const entry = entries[0];
    if (!/^[tTwW]$/u.test(entry.type)) {
      throw new Error(`nm reported invalid code symbol '${name}'`);
    }
    return [name, entry];
  }));
}

function nmEntryKey(symbol, name = symbol.name) {
  return `${name}\0${symbol.type}\0${symbol.address}\0${symbol.size ?? ''}`;
}

function normalizedNmDigest(symbols) {
  const digest = createHash('sha256');
  for (const value of symbols.map((symbol) => nmEntryKey(symbol)).sort()) {
    digest.update(`${value}\n`);
  }
  return digest.digest('hex');
}

function takeNmMatch(available, releaseSymbol, ignoreAddress = false) {
  const index = available.findIndex((candidateSymbol) =>
    candidateSymbol.name === releaseSymbol.name &&
    candidateSymbol.type === releaseSymbol.type &&
    candidateSymbol.size === releaseSymbol.size &&
    (ignoreAddress || candidateSymbol.address === releaseSymbol.address));
  if (index < 0) return null;
  return available.splice(index, 1)[0];
}

function exactlyOneNamed(symbols, name, label) {
  const matches = symbols.filter((symbol) => symbol.name === name);
  if (matches.length !== 1) {
    throw new Error(`${label} must contain exactly one '${name}' text symbol; found ${matches.length}`);
  }
  return matches[0];
}

export function compareReleaseTextLayout(releaseOutput, candidateOutput) {
  const releaseSymbols = parsePosixNmTextSymbols(releaseOutput);
  const candidateSymbols = parsePosixNmTextSymbols(candidateOutput);
  const releaseMain = exactlyOneNamed(releaseSymbols, 'main', 'selected release');
  const candidateAnchor = exactlyOneNamed(
    candidateSymbols, CLI_ANCHOR_SYMBOL, 'candidate executable',
  );
  const candidateMain = exactlyOneNamed(candidateSymbols, 'main', 'candidate executable');
  if (releaseMain.address !== candidateAnchor.address ||
      releaseMain.type !== candidateAnchor.type ||
      releaseMain.size !== candidateAnchor.size) {
    throw new Error(
      `candidate CLI anchor differs from release main: ` +
      `${candidateAnchor.type}/0x${candidateAnchor.address}/${candidateAnchor.size} vs ` +
      `${releaseMain.type}/0x${releaseMain.address}/${releaseMain.size}`,
    );
  }
  if (BigInt(`0x${candidateMain.address}`) <= BigInt(`0x${candidateAnchor.address}`)) {
    throw new Error('candidate benchmark main must be after the release CLI anchor');
  }

  const available = candidateSymbols.filter((symbol) => symbol !== candidateAnchor);
  const stableReleaseProjection = [{ ...releaseMain, name: CLI_ANCHOR_SYMBOL }];
  const stableCandidateProjection = [candidateAnchor];
  const relocatableTail = [];
  const benchmarkBoundary = BigInt(`0x${candidateMain.address}`);
  for (const releaseSymbol of releaseSymbols) {
    if (releaseSymbol === releaseMain) continue;
    let candidateSymbol = takeNmMatch(available, releaseSymbol);
    const relocatable = BigInt(`0x${releaseSymbol.address}`) >= benchmarkBoundary;
    if (!candidateSymbol && relocatable) {
      candidateSymbol = takeNmMatch(available, releaseSymbol, true);
    }
    if (!candidateSymbol) {
      throw new Error(
        `candidate is missing production text symbol ` +
        `'${releaseSymbol.name}' (${releaseSymbol.type}/0x${releaseSymbol.address}/` +
        `${releaseSymbol.size ?? 'unknown-size'})`,
      );
    }
    if (candidateSymbol.address !== releaseSymbol.address) {
      if (!relocatable) {
        throw new Error(
          `candidate production text symbol '${releaseSymbol.name}' moved: ` +
          `0x${candidateSymbol.address} vs release 0x${releaseSymbol.address}`,
        );
      }
      if (BigInt(`0x${candidateSymbol.address}`) <= benchmarkBoundary) {
        throw new Error(
          `relocatable linker tail symbol '${releaseSymbol.name}' is not after main`,
        );
      }
      relocatableTail.push({
        name: releaseSymbol.name,
        type: releaseSymbol.type,
        sizeHex: releaseSymbol.size,
        releaseAddress: `0x${releaseSymbol.address}`,
        candidateAddress: `0x${candidateSymbol.address}`,
      });
      continue;
    }
    stableReleaseProjection.push(releaseSymbol);
    stableCandidateProjection.push(candidateSymbol);
  }

  const candidateOnlyBeforeAnchor = available.filter((symbol) =>
    BigInt(`0x${symbol.address}`) <= BigInt(`0x${candidateAnchor.address}`));
  if (candidateOnlyBeforeAnchor.length !== 0) {
    const sample = candidateOnlyBeforeAnchor.slice(0, 5)
      .map((symbol) => `${symbol.name}@0x${symbol.address}`).join(', ');
    throw new Error(
      `candidate adds ${candidateOnlyBeforeAnchor.length} text symbol(s) at or before ` +
      `the release CLI anchor: ${sample}`,
    );
  }
  const candidateOnlyBeforeBenchmarkMain = available.filter((symbol) =>
    BigInt(`0x${symbol.address}`) < benchmarkBoundary);
  if (candidateOnlyBeforeBenchmarkMain.length !== 0) {
    const sample = candidateOnlyBeforeBenchmarkMain.slice(0, 5)
      .map((symbol) => `${symbol.name}@0x${symbol.address}`).join(', ');
    throw new Error(
      `candidate adds ${candidateOnlyBeforeBenchmarkMain.length} text symbol(s) before ` +
      `the benchmark object boundary: ${sample}`,
    );
  }
  const releaseDigest = normalizedNmDigest(stableReleaseProjection);
  const candidateDigest = normalizedNmDigest(stableCandidateProjection);
  if (releaseDigest !== candidateDigest) {
    throw new Error('candidate production text projection digest differs from release');
  }
  return {
    releaseTextSymbolCount: releaseSymbols.length,
    releaseTextSymbolNameCount: new Set(releaseSymbols.map(({ name }) => name)).size,
    candidateTextSymbolCount: candidateSymbols.length,
    candidateTextSymbolNameCount: new Set(candidateSymbols.map(({ name }) => name)).size,
    exactProductionTextSymbolCount: stableReleaseProjection.length,
    productionTextProjectionSha256: releaseDigest,
    candidateOnlyAfterAnchorCount: available.length,
    candidateOnlyAfterAnchorSha256: normalizedNmDigest(available),
    candidateOnlyAtOrBeforeAnchorCount: 0,
    candidateOnlyBeforeBenchmarkMainCount: 0,
    releaseMainMappedToCandidateCliAnchor: true,
    releaseMainAddress: `0x${releaseMain.address}`,
    candidateCliAnchorAddress: `0x${candidateAnchor.address}`,
    candidateBenchmarkMainAddress: `0x${candidateMain.address}`,
    relocatableImplicitLinkerTailSymbols: relocatableTail,
  };
}

async function cmakeFileTool(buildDirectory, name) {
  const cacheFile = path.join(buildDirectory, 'CMakeCache.txt');
  const cache = await readFile(cacheFile, 'utf8');
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
  const value = new RegExp(`^${escaped}(?::[^=]*)?=(.+)$`, 'mu')
    .exec(cache)?.[1]?.trim();
  if (!value || value.endsWith('-NOTFOUND')) {
    throw new Error(`CMake cache has no usable ${name} tool`);
  }
  return value;
}

function assertFingerprint(label, expected, actual) {
  if (!expected || expected.rawBytes !== actual.bytes || expected.sha256 !== actual.sha256) {
    throw new Error(`${label} differs from native finalization evidence`);
  }
}

function resolvedEvidencePath(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`native finalization evidence has no ${label} path`);
  }
  return path.resolve(ROOT, value);
}

function readElfBuildId(readelf, filename, label) {
  const output = checkedSpawn(
    readelf,
    ['-n', filename],
    { cwd: ROOT },
    `${label} build-ID query`,
  ).stdout;
  const matches = [...output.matchAll(/Build ID:\s*([0-9a-f]+)/giu)]
    .map((match) => match[1].toLowerCase());
  if (matches.length !== 1) {
    throw new Error(`${label} must contain exactly one GNU build ID`);
  }
  return matches[0];
}

async function validateReleaseLinkEvidence(buildDirectory, releaseBundle, document) {
  const nativeBuildDirectory = path.join(buildDirectory, 'native');
  const expectedLinkCommand = path.join(
    nativeBuildDirectory, 'CMakeFiles', 'volvoxai.dir', 'link.txt',
  );
  const link = document.build?.link;
  if (!link || !Array.isArray(link.objects) || link.objects.length === 0) {
    throw new Error('native finalization evidence has no ordered release link inputs');
  }
  const linkCommandEvidence = await fileEvidence(expectedLinkCommand);
  const workingDirectory = nativeBuildDirectory;
  const command = await readFile(expectedLinkCommand, 'utf8');
  const tokens = tokenizeCommand(command, expectedLinkCommand);
  const outputIndices = tokens.flatMap((token, index) => token === '-o' ? [index] : []);
  if (outputIndices.length !== 1 || outputIndices[0] + 1 >= tokens.length) {
    throw new Error('release link command must contain exactly one output');
  }
  const output = path.resolve(workingDirectory, tokens[outputIndices[0] + 1]);
  if (output !== releaseBundle['private release'].filename) {
    throw new Error('release link command output differs from the selected private release');
  }
  const objectPaths = tokens.filter((token) => token.endsWith('.o'))
    .map((token) => path.resolve(workingDirectory, token));
  if (objectPaths.length !== link.objects.length || new Set(objectPaths).size !== objectPaths.length) {
    throw new Error('ordered release link input count differs from finalization evidence');
  }
  const digest = createHash('sha256');
  for (let index = 0; index < objectPaths.length; index++) {
    const filename = objectPaths[index];
    const expected = link.objects[index];
    const evidencePath = resolvedEvidencePath(expected?.path, `link input ${index}`);
    if (evidencePath !== filename) {
      throw new Error(`release link input ${index} differs from finalization evidence`);
    }
    const actual = await fileEvidence(filename);
    assertFingerprint(`release link input ${index}`, expected, actual);
    digest.update(`${expected.path}\0${actual.bytes}\0${actual.sha256}\n`);
  }
  return {
    passed: true,
    command: {
      path: displayPath(expectedLinkCommand),
      bytes: linkCommandEvidence.bytes,
      sha256: linkCommandEvidence.sha256,
    },
    workingDirectory: displayPath(workingDirectory),
    orderedObjectCount: objectPaths.length,
    orderedObjectGraphSha256: digest.digest('hex'),
    fingerprintsMatchFinalizationEvidence: true,
  };
}

async function validateMatchingReleaseBundle(buildDirectory, releaseBundle) {
  const privateEvidence = releaseBundle['private evidence'];
  const publishedEvidence = releaseBundle['published evidence'];
  const [privateBytes, publishedBytes] = await Promise.all([
    readFile(privateEvidence.filename),
    readFile(publishedEvidence.filename),
  ]);
  if (!privateBytes.equals(publishedBytes)) {
    throw new Error('published finalization evidence differs from the selected build tree');
  }
  let document;
  try {
    document = JSON.parse(privateBytes.toString('utf8'));
  } catch (error) {
    throw new Error(`invalid native finalization evidence: ${error.message}`);
  }
  if (document?.format !== FINALIZATION_SCHEMA) {
    throw new Error('unexpected native finalization evidence format');
  }
  if (resolvedEvidencePath(document.build?.directory, 'build directory') !==
      path.resolve(buildDirectory)) {
    throw new Error('native finalization evidence belongs to another CMake build tree');
  }

  const privateRelease = releaseBundle['private release'];
  const privateDebug = releaseBundle['private debug'];
  const releaseMap = releaseBundle['release map'];
  if (resolvedEvidencePath(document.build?.artifact?.path, 'build artifact') !==
        privateRelease.filename ||
      resolvedEvidencePath(document.build?.linkMap?.path, 'link map') !==
        releaseMap.filename) {
    throw new Error('native finalization evidence names another private artifact or map');
  }
  if (document.artifact !== 'native/volvoxai' ||
      document.debugArtifact !== 'native/.debug/volvoxai.debug') {
    throw new Error('native finalization evidence has an unexpected logical artifact');
  }
  assertFingerprint('build-private release', document.build.artifact, privateRelease);
  assertFingerprint('release linker map', document.build.linkMap, releaseMap);
  assertFingerprint('finalized release', document.release, privateRelease);
  assertFingerprint('finalized debug sidecar', document.debug, privateDebug);

  const publishedRelease = releaseBundle['published release'];
  const publishedDebug = releaseBundle['published debug'];
  if (publishedRelease.bytes !== privateRelease.bytes ||
      publishedRelease.sha256 !== privateRelease.sha256) {
    throw new Error('published release differs from the selected build tree');
  }
  if (publishedDebug.bytes !== privateDebug.bytes ||
      publishedDebug.sha256 !== privateDebug.sha256) {
    throw new Error('published debug sidecar differs from the selected build tree');
  }

  const expectedBuildId = document.after?.buildId;
  if (typeof expectedBuildId !== 'string' || !/^[0-9a-f]+$/u.test(expectedBuildId) ||
      document.before?.buildId !== expectedBuildId ||
      document.debug?.buildId !== expectedBuildId) {
    throw new Error('native finalization evidence has inconsistent build IDs');
  }
  const readelf = await cmakeFileTool(buildDirectory, 'CMAKE_READELF');
  const privateReleaseBuildId = readElfBuildId(
    readelf, privateRelease.filename, 'selected private release',
  );
  const privateDebugBuildId = readElfBuildId(
    readelf, privateDebug.filename, 'selected private debug sidecar',
  );
  if (privateReleaseBuildId !== expectedBuildId || privateDebugBuildId !== expectedBuildId) {
    throw new Error('selected release/debug build ID differs from finalization evidence');
  }
  const releaseLink = await validateReleaseLinkEvidence(
    buildDirectory, releaseBundle, document,
  );
  return {
    passed: true,
    format: FINALIZATION_SCHEMA,
    buildDirectory: document.build.directory,
    buildId: expectedBuildId,
    finalizationEvidenceSha256: privateEvidence.sha256,
    privateAndPublishedBytesMatch: true,
    fingerprintsMatchFinalizationEvidence: true,
    releaseLink,
  };
}

async function verifyReleaseLayoutSymbols(buildDirectory, candidate, releaseBundle) {
  const nm = await cmakeFileTool(buildDirectory, 'CMAKE_NM');
  const releaseDebug = releaseBundle['private debug'];
  const [candidateOutput, releaseOutput] = [
    [candidate.executable, 'candidate performance executable'],
    [releaseDebug.filename, 'selected release debug sidecar'],
  ].map(([filename, label]) => checkedSpawn(
    nm,
    ['-P', '--defined-only', filename],
    { cwd: ROOT },
    `${label} symbol query`,
  ).stdout);
  const textLayout = compareReleaseTextLayout(releaseOutput, candidateOutput);
  const nmVersion = checkedSpawn(nm, ['--version'], { cwd: ROOT }, 'nm version query')
    .stdout.trim().split(/\r?\n/u)[0];
  return {
    passed: true,
    nm: { path: nm, version: nmVersion },
    candidateExecutable: candidate.executableEvidence,
    selectedReleaseDebug: {
      path: releaseDebug.path,
      bytes: releaseDebug.bytes,
      sha256: releaseDebug.sha256,
    },
    textLayout,
  };
}

function immutableReleaseFiles(buildDirectory) {
  const privateDebug = path.join(buildDirectory, 'native', 'release-link', '.debug');
  return [
    ['release link command', path.join(buildDirectory, 'native', 'CMakeFiles', 'volvoxai.dir', 'link.txt')],
    ['release map', path.join(buildDirectory, 'size-maps', 'volvoxai.map')],
    ['private release', path.join(buildDirectory, 'native', 'release-link', 'volvoxai')],
    ['private debug', path.join(privateDebug, 'volvoxai.debug')],
    ['private evidence', path.join(privateDebug, 'volvoxai.debug.json')],
    ['published release', path.join(ROOT, 'native', 'volvoxai')],
    ['published debug', path.join(ROOT, 'native', '.debug', 'volvoxai.debug')],
    ['published evidence', path.join(ROOT, 'native', '.debug', 'volvoxai.debug.json')],
  ];
}

async function snapshotReleaseBundle(buildDirectory) {
  return Object.fromEntries(await Promise.all(
    immutableReleaseFiles(buildDirectory).map(async ([label, filename]) => [
      label,
      { path: displayPath(filename), filename, ...(await fileEvidence(filename)) },
    ]),
  ));
}

async function assertFileEvidenceUnchanged(filename, before, stage) {
  const after = await fileEvidence(filename);
  if (after.bytes !== before.bytes || after.sha256 !== before.sha256 ||
      after.modifiedTimeNs !== before.modifiedTimeNs) {
    throw new Error(`${displayPath(filename)} changed ${stage}`);
  }
}

async function assertReleaseBundleUnchanged(before, stage) {
  for (const [label, evidence] of Object.entries(before)) {
    try {
      await assertFileEvidenceUnchanged(evidence.filename, evidence, stage);
    } catch (error) {
      throw new Error(`${label} ${error.message}: '${evidence.filename}'`);
    }
  }
}

export function parseCpuList(value) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error('CPU list must not be empty');
  }
  const result = [];
  const seen = new Set();
  for (const part of value.split(',')) {
    const match = /^(\d+)(?:-(\d+))?$/u.exec(part.trim());
    if (!match) throw new Error(`invalid CPU-list component '${part}'`);
    const first = Number(match[1]);
    const last = Number(match[2] ?? match[1]);
    if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || first > last ||
        last > 1_048_575) {
      throw new Error(`invalid CPU-list range '${part}'`);
    }
    for (let cpu = first; cpu <= last; cpu++) {
      if (!seen.has(cpu)) {
        seen.add(cpu);
        result.push(cpu);
      }
    }
  }
  return result;
}

async function readProcStatus() {
  const status = await readFile('/proc/self/status', 'utf8');
  const value = (name) => new RegExp(`^${name}:\\s*(.+)$`, 'mu').exec(status)?.[1] ?? null;
  return {
    cpuAllowedList: value('Cpus_allowed_list'),
    memoryNodeAllowedList: value('Mems_allowed_list'),
  };
}

async function topologyForCpu(cpu) {
  const root = `/sys/devices/system/cpu/cpu${cpu}/topology`;
  try {
    const [packageId, coreId, siblings] = await Promise.all([
      readFile(path.join(root, 'physical_package_id'), 'utf8'),
      readFile(path.join(root, 'core_id'), 'utf8'),
      readFile(path.join(root, 'thread_siblings_list'), 'utf8'),
    ]);
    return {
      cpu,
      packageId: packageId.trim(),
      coreId: coreId.trim(),
      threadSiblingsList: siblings.trim(),
    };
  } catch {
    return { cpu, packageId: null, coreId: null, threadSiblingsList: null };
  }
}

async function selectAffinity(cpuListOverride) {
  const status = await readProcStatus();
  if (!status.cpuAllowedList) throw new Error('/proc does not report an allowed CPU list');
  const allowed = parseCpuList(status.cpuAllowedList);
  const requested = cpuListOverride === 'auto' ? allowed : parseCpuList(cpuListOverride);
  const allowedSet = new Set(allowed);
  if (requested.some((cpu) => !allowedSet.has(cpu))) {
    throw new Error(`requested affinity '${cpuListOverride}' is outside '${status.cpuAllowedList}'`);
  }
  if (requested.length < 4) {
    throw new Error(`native W8A8 4-thread gate requires four allowed CPUs; found ${requested.length}`);
  }
  const topology = await Promise.all(requested.map(topologyForCpu));
  const uniqueCore = [];
  const coreKeys = new Set();
  for (const item of topology) {
    const key = item.packageId == null || item.coreId == null
      ? null
      : `${item.packageId}:${item.coreId}`;
    if (key != null && !coreKeys.has(key)) {
      coreKeys.add(key);
      uniqueCore.push(item.cpu);
    }
  }
  if (uniqueCore.length < 4) {
    throw new Error(
      `native W8A8 4-thread gate requires four topology-proven physical cores; ` +
      `found ${uniqueCore.length}`,
    );
  }
  const selected = uniqueCore.slice(0, 4);
  return {
    selected,
    tasksetList: selected.join(','),
    allowed,
    requested,
    topology,
    cpuAllowedList: status.cpuAllowedList,
    memoryNodeAllowedList: status.memoryNodeAllowedList,
    uniquePhysicalCoreCount: uniqueCore.length,
  };
}

function parseIsa(stdout) {
  const match = /CPU features: avx2=(yes|no) avx-vnni=(yes|no) avx512-vnni=(yes|no) neon=(yes|no) arm-sdot=(yes|no)/u.exec(
    stdout,
  );
  if (!match) throw new Error('benchmark did not report its CPU ISA selection');
  return {
    avx2: match[1] === 'yes',
    avxVnni: match[2] === 'yes',
    avx512Vnni: match[3] === 'yes',
    neon: match[4] === 'yes',
    armSdot: match[5] === 'yes',
  };
}

export function parseBenchmarkAffinity(stdout) {
  const match = /Timing affinity: single-cpu=(\d+) pool-cpus=(\d+)/u.exec(stdout);
  if (!match) throw new Error('benchmark did not report its timing affinity policy');
  const singleCpu = Number(match[1]);
  const poolCpuCount = Number(match[2]);
  if (!Number.isSafeInteger(singleCpu) || !Number.isSafeInteger(poolCpuCount) ||
      poolCpuCount < 4) {
    throw new Error('benchmark reported an invalid timing affinity policy');
  }
  return { singleCpu, poolCpuCount };
}

function parseHotMetrics(stdout) {
  return Object.fromEntries(Object.entries(HOT_METRIC_PATTERNS).map(([name, pattern]) => {
    const match = pattern.exec(stdout);
    const value = Number(match?.[1]);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`benchmark did not report positive hot metric '${name}'`);
    }
    return [name, value];
  }));
}

export function assertFixedHotWorkSpan(span, fixedWork) {
  if (span.label !== fixedWork.spanLabel ||
      !Number.isFinite(span.milliseconds) ||
      span.milliseconds + 0.0005 < fixedWork.targetMilliseconds ||
      !Number.isSafeInteger(span.iterations) ||
      span.iterations !== fixedWork.fixedIterations) {
    throw new Error(
      `fixed hot timing '${fixedWork.spanLabel}' requires at least ` +
      `${fixedWork.targetMilliseconds} ms and exactly ` +
      `${fixedWork.fixedIterations} iterations; observed ` +
      `${span.milliseconds} ms / ${span.iterations} iterations`,
    );
  }
}

function parseHotSpans(stdout, minimumTimedMs, fixedWork = null) {
  const spans = [...stdout.matchAll(
    /hot-span\[([^\]]+)\]=([0-9]+(?:\.[0-9]+)?)\/(\d+)/gu,
  )].map((match) => ({
    label: match[1],
    milliseconds: Number(match[2]),
    iterations: Number(match[3]),
  }));
  if (spans.length !== EXPECTED_HOT_SPAN_COUNT) {
    throw new Error(
      `benchmark reported ${spans.length} hot timing spans; expected ${EXPECTED_HOT_SPAN_COUNT}`,
    );
  }
  for (const span of spans) {
    const requiredMilliseconds = fixedWork?.spanLabel === span.label
      ? fixedWork.targetMilliseconds : minimumTimedMs;
    if (!Number.isFinite(span.milliseconds) ||
        span.milliseconds + 0.0005 < requiredMilliseconds ||
        !Number.isSafeInteger(span.iterations) || span.iterations <= 0) {
      throw new Error(
        `hot timing '${span.label}' did not satisfy ${requiredMilliseconds} ms: ` +
        `${span.milliseconds} ms / ${span.iterations} iterations`,
      );
    }
    if (fixedWork?.spanLabel === span.label) assertFixedHotWorkSpan(span, fixedWork);
  }
  return spans;
}

function uniqueHotSpan(spans, label) {
  const matches = spans.filter((span) => span.label === label);
  if (matches.length !== 1) {
    throw new Error(`benchmark must report exactly one hot timing '${label}'`);
  }
  return matches[0];
}

export function assertMetricMatchesSpan(metric, value, span) {
  const average = span.milliseconds / span.iterations;
  const roundingTolerance = 0.0005 / span.iterations + 0.0000005 + 1e-9;
  if (Math.abs(value - average) > roundingTolerance) {
    throw new Error(
      `hot metric '${metric}' ${value} ms does not match ` +
      `${span.milliseconds} ms / ${span.iterations} iterations`,
    );
  }
}

export function calibrateGrayscaleSingleFixedWork(
  warmups,
  policy = GRAYSCALE_SINGLE_FIXED_WORK,
) {
  if (!Array.isArray(warmups) || warmups.length !== 4) {
    throw new Error('grayscale fixed-work calibration requires exactly four warmup runs');
  }
  const variantCounts = { baseline: 0, candidate: 0 };
  const observations = warmups.map((run) => {
    if (!Object.hasOwn(variantCounts, run.variant) || run.warmup !== true || run.block !== 0) {
      throw new Error('grayscale fixed-work calibration received an invalid warmup run');
    }
    variantCounts[run.variant] += 1;
    const span = uniqueHotSpan(run.hotSpans ?? [], policy.spanLabel);
    const averageMilliseconds = span.milliseconds / span.iterations;
    if (!(averageMilliseconds > 0)) {
      throw new Error('grayscale fixed-work calibration observed a non-positive average');
    }
    return {
      ordinal: run.ordinal,
      variant: run.variant,
      spanMilliseconds: span.milliseconds,
      iterations: span.iterations,
      averageMilliseconds,
    };
  });
  if (variantCounts.baseline !== 2 || variantCounts.candidate !== 2) {
    throw new Error('grayscale fixed-work calibration requires two warmups per variant');
  }
  const fastestWarmupAverageMilliseconds = Math.min(
    ...observations.map(({ averageMilliseconds }) => averageMilliseconds),
  );
  const fixedIterations = Math.ceil(
    policy.headroom * policy.targetMilliseconds / fastestWarmupAverageMilliseconds,
  );
  if (!Number.isSafeInteger(fixedIterations) || fixedIterations <= 0 ||
      fixedIterations > policy.maximumIterations) {
    throw new Error('grayscale fixed-work calibration produced an invalid iteration count');
  }
  const calibration = {
    formulaVersion: policy.formulaVersion,
    metric: policy.metric,
    spanLabel: policy.spanLabel,
    targetMilliseconds: policy.targetMilliseconds,
    headroom: policy.headroom,
    observations,
    fastestWarmupAverageMilliseconds,
    fixedIterations,
    designedMinimumSpanMilliseconds:
      fastestWarmupAverageMilliseconds * fixedIterations,
  };
  return {
    ...calibration,
    sha256: sha256(JSON.stringify(calibration)),
  };
}

function correctnessSignature(stdout) {
  const checksums = [...stdout.matchAll(/(?:checksum|i8)=0x([0-9a-f]{8})/giu)]
    .map((match) => match[1].toLowerCase());
  const checkPassCount = [...stdout.matchAll(/check=pass/gu)].length;
  if (checksums.length === 0 || checkPassCount !== 10) {
    throw new Error('benchmark correctness signature is incomplete');
  }
  return { checksums, checkPassCount };
}

export function buildBenchmarkArguments(executable, affinity, options, fixedWork = null) {
  const arguments_ = [
    '-c', affinity.tasksetList,
    executable,
    '--gate',
    `--min-timed-ms=${options.minimumTimedMs}`,
    `--single-cpu=${affinity.selected[0]}`,
  ];
  if (fixedWork != null) {
    arguments_.push(`--grayscale-single-iterations=${fixedWork.fixedIterations}`);
  }
  return arguments_;
}

function runVariant(
  variant,
  executable,
  affinity,
  options,
  ordinal,
  warmup,
  block,
  fixedWork = null,
) {
  const started = performance.now();
  const arguments_ = buildBenchmarkArguments(executable, affinity, options, fixedWork);
  let child;
  try {
    child = checkedSpawn(
      'taskset',
      arguments_,
      { cwd: ROOT, timeout: options.timeoutMs },
      `${variant} W8A8 benchmark run ${ordinal}`,
    );
  } catch (error) {
    error.benchmarkRun = { variant, ordinal, warmup, block };
    throw error;
  }
  const wallMs = Number((performance.now() - started).toFixed(3));
  const missing = REQUIRED_OUTPUT_MARKERS.filter((marker) => !child.stdout.includes(marker));
  if (missing.length !== 0) {
    throw new Error(`${variant} benchmark output is missing: ${missing.join(', ')}`);
  }
  const hotMetricsMs = parseHotMetrics(child.stdout);
  const hotSpans = parseHotSpans(child.stdout, options.minimumTimedMs, fixedWork);
  const grayscaleSingleSpan = uniqueHotSpan(
    hotSpans,
    GRAYSCALE_SINGLE_FIXED_WORK.spanLabel,
  );
  assertMetricMatchesSpan(
    GRAYSCALE_SINGLE_FIXED_WORK.metric,
    hotMetricsMs[GRAYSCALE_SINGLE_FIXED_WORK.metric],
    grayscaleSingleSpan,
  );
  const timingAffinity = parseBenchmarkAffinity(child.stdout);
  if (timingAffinity.singleCpu !== affinity.selected[0] ||
      timingAffinity.poolCpuCount !== affinity.selected.length) {
    throw new Error(
      `benchmark timing affinity ${JSON.stringify(timingAffinity)} differs from ` +
      `runner selection ${JSON.stringify(affinity.selected)}`,
    );
  }
  return {
    variant,
    ordinal,
    warmup,
    block,
    wallMs,
    hotMetricsMs,
    hotSpans,
    timingAffinity,
    isa: parseIsa(child.stdout),
    correctnessSignature: correctnessSignature(child.stdout),
    stdoutSha256: sha256(child.stdout),
    stderrLines: child.stderr.trim() ? child.stderr.trim().split(/\r?\n/u) : [],
  };
}

export function balancedBlockSchedule(blockNumber) {
  if (!Number.isSafeInteger(blockNumber) || blockNumber < 1) {
    throw new Error('block number must be a positive integer');
  }
  return blockNumber % 2 === 1
    ? ['baseline', 'candidate', 'candidate', 'baseline']
    : ['candidate', 'baseline', 'baseline', 'candidate'];
}

export function median(values) {
  if (!Array.isArray(values) || values.length === 0 ||
      values.some((value) => !Number.isFinite(value))) {
    throw new Error('median requires finite values');
  }
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1] + ordered[middle]) / 2
    : ordered[middle];
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.max(0, Math.ceil(ordered.length * fraction) - 1)];
}

function seededGenerator(seed) {
  let state = seed >>> 0 || 0x564f4c58;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

export function bootstrapMedianInterval(values, resamples = 10_000, seed = 0x564f4c58) {
  if (!Number.isSafeInteger(resamples) || resamples < 1) {
    throw new Error('bootstrap resamples must be a positive integer');
  }
  const random = seededGenerator(seed);
  const estimates = [];
  for (let sample = 0; sample < resamples; sample++) {
    const selected = [];
    for (let index = 0; index < values.length; index++) {
      selected.push(values[Math.floor(random() * values.length)]);
    }
    estimates.push(median(selected));
  }
  return {
    lower95: percentile(estimates, 0.05),
    upper95: percentile(estimates, 0.95),
  };
}

function metricSeed(metric, blockCount) {
  return Number.parseInt(sha256(`${metric}\0${blockCount}`).slice(0, 8), 16) >>> 0;
}

function deltaPercentFromLog(value) {
  return Math.expm1(value) * 100;
}

function robustSigma(values) {
  const center = median(values);
  return median(values.map((value) => Math.abs(value - center))) * 1.4826;
}

export function summarizeMetricBlocks(blocks, metric, policy = {}) {
  const resamples = policy.bootstrapResamples ?? 10_000;
  const thresholdPercent = policy.maxRegressionPercent ?? 2;
  const paired = [];
  const blockLogs = [];
  const baselineSamples = [];
  const candidateSamples = [];
  for (const block of blocks) {
    if (!Array.isArray(block.runs) || block.runs.length !== 4) {
      throw new Error('each measured block must have exactly four runs');
    }
    const blockPairs = [block.runs.slice(0, 2), block.runs.slice(2, 4)];
    const pairLogs = [];
    for (const pair of blockPairs) {
      const baseline = pair.find((run) => run.variant === 'baseline');
      const candidate = pair.find((run) => run.variant === 'candidate');
      const baselineValue = metric === 'processWall' ? baseline?.wallMs
        : baseline?.hotMetricsMs?.[metric];
      const candidateValue = metric === 'processWall' ? candidate?.wallMs
        : candidate?.hotMetricsMs?.[metric];
      if (!(baselineValue > 0) || !(candidateValue > 0)) {
        throw new Error(`block is missing positive values for '${metric}'`);
      }
      const logRatio = Math.log(candidateValue / baselineValue);
      pairLogs.push(logRatio);
      paired.push({
        candidateAfterBaseline: pair.indexOf(candidate) > pair.indexOf(baseline),
        logRatio,
        deltaPercent: deltaPercentFromLog(logRatio),
      });
      baselineSamples.push(baselineValue);
      candidateSamples.push(candidateValue);
    }
    blockLogs.push((pairLogs[0] + pairLogs[1]) / 2);
  }
  const estimateLog = median(blockLogs);
  const interval = bootstrapMedianInterval(
    blockLogs,
    resamples,
    metricSeed(metric, blockLogs.length),
  );
  const after = paired.filter((item) => item.candidateAfterBaseline).map((item) => item.logRatio);
  const before = paired.filter((item) => !item.candidateAfterBaseline).map((item) => item.logRatio);
  const orderGapLog = Math.abs(median(after) - median(before));
  const robustSigmaLog = robustSigma(blockLogs);
  const unstable = robustSigmaLog > Math.log1p(0.05) || orderGapLog > Math.log1p(0.03);
  const thresholdLog = Math.log1p(thresholdPercent / 100);
  let status;
  if (unstable) status = 'unstable';
  else if (interval.upper95 <= thresholdLog + 1e-12) status = 'pass';
  else if (interval.lower95 > thresholdLog + 1e-12) status = 'regression';
  else status = 'inconclusive';
  return {
    status,
    unit: metric === 'processWall' ? 'ms/process' : 'ms',
    baselineSamples,
    candidateSamples,
    baselineMedian: median(baselineSamples),
    candidateMedian: median(candidateSamples),
    ratioOfMediansDeltaPercent:
      (median(candidateSamples) / median(baselineSamples) - 1) * 100,
    paired,
    blockLogRatios: blockLogs,
    estimateDeltaPercent: deltaPercentFromLog(estimateLog),
    lower95DeltaPercent: deltaPercentFromLog(interval.lower95),
    upper95DeltaPercent: deltaPercentFromLog(interval.upper95),
    robustSigmaPercent: deltaPercentFromLog(robustSigmaLog),
    orderStratumGapPercent: deltaPercentFromLog(orderGapLog),
  };
}

function summarizeBlocks(blocks, options) {
  const policy = {
    bootstrapResamples: options.bootstrapResamples,
    maxRegressionPercent: options.maxRegressionPercent,
  };
  const metrics = Object.fromEntries(HARD_METRICS.map((metric) => [
    metric,
    summarizeMetricBlocks(blocks, metric, policy),
  ]));
  const processWall = summarizeMetricBlocks(blocks, 'processWall', policy);
  const statuses = Object.values(metrics).map(({ status }) => status);
  const status = statuses.every((value) => value === 'pass') ? 'pass'
    : statuses.includes('regression') ? 'regression'
      : statuses.includes('unstable') ? 'unstable'
        : 'inconclusive';
  return { status, metrics, processWall };
}

function assertRunContract(run, expected) {
  const isa = JSON.stringify(run.isa);
  const signature = JSON.stringify(run.correctnessSignature);
  if (expected.isa == null) expected.isa = isa;
  if (expected.signature == null) expected.signature = signature;
  if (isa !== expected.isa) throw new Error('CPU ISA selection changed between A/B runs');
  if (signature !== expected.signature) {
    throw new Error('baseline/candidate correctness checksums differ');
  }
}

async function executeGate(options) {
  if (platform() !== 'linux') throw new Error('native W8A8 release gate requires Linux');
  const runnerEvidence = {
    path: displayPath(SCRIPT),
    ...(await fileEvidence(SCRIPT)),
  };
  options.runnerEvidence = runnerEvidence;
  const releaseBundle = await snapshotReleaseBundle(options.buildDirectory);
  const releaseBundleValidation = await validateMatchingReleaseBundle(
    options.buildDirectory,
    releaseBundle,
  );
  let performanceGraphs = null;
  try {
  const affinity = await selectAffinity(options.cpuListOverride);
  const tasksetVersion = checkedSpawn('taskset', ['--version'], { cwd: ROOT }, 'taskset query')
    .stdout.trim().split(/\r?\n/u)[0];
  checkedSpawn('cmake', [
    '--build', options.buildDirectory,
    '--target', PERFORMANCE_TARGET,
    '-j', String(options.buildJobs),
  ], { cwd: ROOT, timeout: 900_000 }, 'native W8A8 performance target build');
  await assertReleaseBundleUnchanged(releaseBundle, 'while building performance targets');
  await validateMatchingReleaseBundle(options.buildDirectory, releaseBundle);

  const [baseline, candidate] = await Promise.all([
    inspectPerformanceGraph(options.buildDirectory, 'baseline'),
    inspectPerformanceGraph(options.buildDirectory, 'candidate'),
  ]);
  performanceGraphs = { baseline, candidate };
  const performanceGraphIntegrity = {
    baselineSha256: performanceGraphIntegritySha256(baseline),
    candidateSha256: performanceGraphIntegritySha256(candidate),
  };
  const orderedSourceSha256 = assertMatchingLogicalOrder(baseline, candidate);
  const sharedMainBaseline = baseline.objects.at(-1);
  const sharedMainCandidate = candidate.objects.at(-1);
  if (sharedMainBaseline.token !== sharedMainCandidate.token ||
      sharedMainBaseline.sha256 !== sharedMainCandidate.sha256) {
    throw new Error('baseline and candidate do not share one identical benchmark-main object');
  }
  const releaseProjection = await inspectReleaseProjection(options.buildDirectory, candidate);
  const releaseLayoutSymbols = await verifyReleaseLayoutSymbols(
    options.buildDirectory,
    candidate,
    releaseBundle,
  );

  const compilerVersion = checkedSpawn(
    candidate.compiler,
    ['--version'],
    { cwd: ROOT },
    'native compiler version query',
  ).stdout.trim().split(/\r?\n/u)[0];
  const expected = { isa: null, signature: null };
  const warmups = [];
  let ordinal = 0;
  for (const variant of balancedBlockSchedule(1)) {
    ordinal += 1;
    const graph = variant === 'baseline' ? baseline : candidate;
    const run = runVariant(
      variant, graph.executable, affinity, options, ordinal, true, 0,
    );
    assertRunContract(run, expected);
    warmups.push(run);
  }
  const grayscaleSingleFixedWork = calibrateGrayscaleSingleFixedWork(
    warmups,
    grayscaleSingleFixedWorkPolicy(options.minimumTimedMs),
  );
  options.fixedWorkCalibration = grayscaleSingleFixedWork;

  const blocks = [];
  const runThroughBlock = (limit) => {
    while (blocks.length < limit) {
      const blockNumber = blocks.length + 1;
      const runs = [];
      for (const variant of balancedBlockSchedule(blockNumber)) {
        ordinal += 1;
        const graph = variant === 'baseline' ? baseline : candidate;
        const run = runVariant(
          variant, graph.executable, affinity, options, ordinal, false, blockNumber,
          grayscaleSingleFixedWork,
        );
        assertRunContract(run, expected);
        runs.push(run);
      }
      blocks.push({ block: blockNumber, order: runs.map(({ variant }) => variant), runs });
    }
  };
  // The final sample count is fixed before measurement. Intermediate summaries
  // are diagnostics only: using them to accept the candidate early would make
  // the nominal one-sided 95% interval invalid through optional stopping.
  runThroughBlock(options.maxBlocks);
  const checkpointDiagnostics = [];
  for (let count = options.blocks; count < options.maxBlocks; count += 4) {
    const checkpoint = summarizeBlocks(blocks.slice(0, count), options);
    checkpointDiagnostics.push({
      blocks: count,
      status: checkpoint.status,
      decision: false,
    });
  }
  const summary = summarizeBlocks(blocks, options);
  const initialStatus = checkpointDiagnostics[0]?.status ?? summary.status;
  return {
    schema: SCHEMA,
    recordedAt: new Date().toISOString(),
    gate: {
      status: summary.status,
      passed: summary.status === 'pass',
      initialStatus,
      initialBlocks: options.blocks,
      measuredBlocks: blocks.length,
      maximumBlocks: options.maxBlocks,
      extended: blocks.length > options.blocks,
      extensionReason: blocks.length > options.blocks ? 'precommitted-fixed-sample' : null,
      checkpointDiagnostics,
      failedMetrics: Object.entries(summary.metrics)
        .filter(([, result]) => result.status !== 'pass')
        .map(([metric, result]) => ({ metric, status: result.status })),
    },
    policy: {
      baselineKind: 'current-source-all-O3',
      historicalAnchorCommit: HISTORICAL_ANCHOR,
      historicalAnchorRole: 'size-investigation start; not the performance baseline',
      maximumRegressionPercent: options.maxRegressionPercent,
      confidence: 'deterministic one-sided 95% block bootstrap; one final decision at the precommitted sample count',
      bootstrapResamples: options.bootstrapResamples,
      bootstrapSeed: 'first 32 bits of SHA-256(metric + NUL + measured block count)',
      schedule: 'alternating BCCB/CBBC ABBA blocks',
      warmupRunsPerVariant: 2,
      minimumHotTimedSpanMs: options.minimumTimedMs,
      hotSpanCountPerRun: EXPECTED_HOT_SPAN_COUNT,
      fixedHotWork: {
        metric: grayscaleSingleFixedWork.metric,
        spanLabel: grayscaleSingleFixedWork.spanLabel,
        targetMilliseconds: grayscaleSingleFixedWork.targetMilliseconds,
        headroom: grayscaleSingleFixedWork.headroom,
        formulaVersion: grayscaleSingleFixedWork.formulaVersion,
        sharedAcrossVariantsAndMeasuredRuns: true,
        recalibrationDuringMeasurement: false,
      },
      outlierRemoval: 'none',
      diagnosticTiming:
        'gate mode keeps correctness calls but limits non-gated timing loops to one iteration',
      instability: {
        maximumRobustSigmaPercent: 5,
        maximumOrderStratumGapPercent: 3,
      },
      hardMetrics: HARD_METRICS,
      diagnostics: ['processWall', 'ratioOfMediansDeltaPercent'],
    },
    correctness: {
      passed: true,
      signature: JSON.parse(expected.signature),
      detectedIsa: JSON.parse(expected.isa),
      warmupRuns: warmups.length,
      measuredRuns: blocks.length * 4,
    },
    timing: {
      calibration: grayscaleSingleFixedWork,
      metrics: summary.metrics,
      processWall: summary.processWall,
      warmups,
      blocks,
    },
    environment: {
      platform: platform(),
      nodeVersion: process.version,
      affinity,
      tasksetVersion,
      exclusiveLock: options.performanceLockEvidence,
    },
    provenance: {
      buildDirectory: displayPath(options.buildDirectory),
      compiler: candidate.compiler,
      compilerVersion,
      runner: {
        ...runnerEvidence,
        nodeExecutable: process.execPath,
        arguments: process.argv.slice(2),
        verifiedUnchangedAtExit: true,
      },
      orderedLogicalSourceSha256: orderedSourceSha256,
      sharedBenchmarkMain: sharedMainCandidate,
      baseline,
      candidate,
      privateComparisonGraphIntegrity: {
        ...performanceGraphIntegrity,
        verifiedUnchangedAtExit: true,
      },
      inferenceReleaseProjection: releaseProjection,
      releaseLayoutSymbols,
      immutableReleaseBundle: Object.fromEntries(
        Object.entries(releaseBundle).map(([label, evidence]) => [label, {
          path: evidence.path,
          bytes: evidence.bytes,
          sha256: evidence.sha256,
          modifiedAt: evidence.modifiedAt,
          modifiedTimeNs: evidence.modifiedTimeNs,
        }]),
      ),
      immutableReleaseBundleVerified: true,
      matchingReleaseBundle: releaseBundleValidation,
      matchingReleaseBundleVerifiedAtStartAfterBuildAndExit: true,
    },
  };
  } finally {
    const checks = [
      assertReleaseBundleUnchanged(releaseBundle, 'before leaving the performance gate'),
      validateMatchingReleaseBundle(options.buildDirectory, releaseBundle),
      assertFileEvidenceUnchanged(
        SCRIPT,
        runnerEvidence,
        'before leaving the performance gate',
      ),
    ];
    if (performanceGraphs != null) {
      checks.push((async () => {
        const [baselineAfter, candidateAfter] = await Promise.all([
          inspectPerformanceGraph(options.buildDirectory, 'baseline'),
          inspectPerformanceGraph(options.buildDirectory, 'candidate'),
        ]);
        assertPerformanceGraphUnchanged(
          performanceGraphs.baseline,
          baselineAfter,
          'during measurement',
        );
        assertPerformanceGraphUnchanged(
          performanceGraphs.candidate,
          candidateAfter,
          'during measurement',
        );
      })());
    }
    await Promise.all(checks);
  }
}

async function cliMain() {
  let options;
  let performanceLock;
  try {
    options = parseOptions();
    performanceLock = await acquirePerformanceLock(options);
    options.performanceLockEvidence = performanceLock.evidence;
    const evidence = await executeGate(options);
    await atomicWriteJson(options.output, evidence);
    process.stdout.write(
      `Native W8A8 release performance: ${evidence.gate.status}; ` +
      `${evidence.gate.measuredBlocks} blocks; evidence ${displayPath(options.output)}\n`,
    );
    if (!evidence.gate.passed) process.exitCode = 1;
  } catch (error) {
    // A contender that did not acquire the lock must never overwrite the
    // active owner's evidence path with its own concurrency error.
    if (options?.output && performanceLock) {
      const evidence = {
        schema: SCHEMA,
        recordedAt: new Date().toISOString(),
        gate: { status: 'error', passed: false },
        error: {
          message: error.message,
          stack: error.stack ?? null,
          ...(error.benchmarkRun == null ? {} : { benchmarkRun: error.benchmarkRun }),
        },
        ...(options.runnerEvidence == null ? {} : {
          provenance: { runner: options.runnerEvidence },
        }),
        ...(options.fixedWorkCalibration == null ? {} : {
          timing: { calibration: options.fixedWorkCalibration },
        }),
      };
      try {
        await atomicWriteJson(options.output, evidence);
      } catch (writeError) {
        process.stderr.write(`could not write failure evidence: ${writeError.message}\n`);
      }
    }
    process.stderr.write(`${error.stack || error.message}\n`);
    process.exitCode = 1;
  } finally {
    if (performanceLock) {
      try {
        await performanceLock.release();
      } catch (error) {
        process.stderr.write(`could not release performance lock: ${error.message}\n`);
        process.exitCode = 1;
      }
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(SCRIPT)) {
  await cliMain();
}
