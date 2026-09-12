#!/usr/bin/env node

import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { constants as fsConstants } from 'node:fs';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import {
  RELEASE_PROFILES,
  WASM_BUILD_EVIDENCE_FORMAT,
  WASM_PTQ_AUTHORING_SECTION,
  WASM_RELAXED_SIMD_SECTION,
  orderedWasmCustomSections,
  sha256,
  stableJSON,
  validateWasmArtifact,
  validateWasmModuleStructure,
  withoutWasmCustomSection,
  wasmProfile} from './release_profiles.mjs';

export { WASM_BUILD_EVIDENCE_FORMAT } from './release_profiles.mjs';

const executeFile = promisify(execFile);
const modulePath = fileURLToPath(import.meta.url);
const defaultRepositoryRoot = path.resolve(path.dirname(modulePath), '..');

const COMPONENT_PROPERTIES = Object.freeze({
  parent: 'parent',
  relaxed: 'relaxed',
  ptqAuthoring: 'ptqAuthoring',
});
const COMPONENT_ORDER = Object.freeze(['parent', 'relaxed', 'ptqAuthoring']);
const WASM_MAGIC = Buffer.from([0x00, 0x61, 0x73, 0x6d]);
const OPTIMIZATION_FLAG = /^-O(?:[0-3]|s|z|fast)$/u;
const toolFingerprintCache = new Map();
const runtimeFingerprintCache = new Map();

function normalizedPath(value) {
  return value.split(path.sep).join('/');
}

function requireNonemptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${label} must be a nonempty string.`);
  }
  return value;
}

function requireCanonicalAbsolutePath(value, label) {
  const selected = requireNonemptyString(value, label);
  if (!path.posix.isAbsolute(selected) || path.posix.normalize(selected) !== selected) {
    throw new Error(`${label} must be a canonical absolute path.`);
  }
  return selected;
}

function requireStringArray(value, label) {
  if (!Array.isArray(value) || value.some((entry) =>
    typeof entry !== 'string' || entry.length === 0)) {
    throw new Error(`${label} must be an array of nonempty strings.`);
  }
  return [...value];
}

function safeComponentId(value, label) {
  const id = requireNonemptyString(value, label);
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/u.test(id)) {
    throw new Error(`${label} '${id}' is not safe for an object filename.`);
  }
  return id;
}

function profileFrom(value) {
  if (typeof value === 'string') return wasmProfile(value);
  if (value === null || typeof value !== 'object') {
    throw new Error('WASM profile must be a profile id, filename, or profile object.');
  }
  return value;
}

function canonicalKind(value) {
  if (value === 'ptq-authoring') return 'ptqAuthoring';
  return value;
}

function componentFor(profile, kindValue) {
  const kind = canonicalKind(kindValue);
  const property = COMPONENT_PROPERTIES[kind];
  if (property === undefined) {
    throw new Error(
      `Unknown WASM component kind '${kindValue}'; expected parent, relaxed, or ptqAuthoring.`,
    );
  }
  const component = profile.recipe?.[property];
  if (component === null || component === undefined) {
    throw new Error(`WASM profile '${profile.id}' has no '${kind}' component.`);
  }
  return Object.freeze({ kind, component });
}

function validateToolRuntimeDependencies(value, label) {
  const dependencies = value === undefined ? [] : value;
  if (!Array.isArray(dependencies)) throw new Error(`${label} must be an array.`);
  const paths = [];
  const validated = dependencies.map((dependency, index) => {
    if (dependency === null || typeof dependency !== 'object' ||
        Array.isArray(dependency)) {
      throw new Error(`${label} dependency ${index} must be an object.`);
    }
    const dependencyPath = requireNonemptyString(
      dependency.path,
      `${label} dependency ${index} path`,
    );
    if (!path.posix.isAbsolute(dependencyPath) ||
        path.posix.normalize(dependencyPath) !== dependencyPath) {
      throw new Error(`${label} dependency ${index} path must be canonical and absolute.`);
    }
    exactKeys(
      dependency,
      ['path', 'rawBytes', 'sha256'],
      `${label} dependency ${index}`,
    );
    paths.push(dependencyPath);
    return Object.freeze({
      path: dependencyPath,
      rawBytes: evidenceBytes(dependency.rawBytes, `${label} dependency ${index} rawBytes`),
      sha256: evidenceSha(dependency.sha256, `${label} dependency ${index} sha256`),
    });
  });
  const sorted = [...new Set(paths)].sort((left, right) => left.localeCompare(right, 'en'));
  if (stableJSON(paths) !== stableJSON(sorted)) {
    throw new Error(`${label} dependency paths must be unique and sorted.`);
  }
  return Object.freeze(validated);
}

function validateToolchain(profile) {
  const toolchain = profile.recipe?.toolchain;
  if (toolchain === null || typeof toolchain !== 'object' || Array.isArray(toolchain)) {
    throw new Error(`WASM profile '${profile.id}' has no object-recipe toolchain.`);
  }
  const resourceDependencyRoot = requireNonemptyString(
    toolchain.resourceDependencyRoot,
    `WASM profile '${profile.id}' resourceDependencyRoot`,
  );
  if (!path.posix.isAbsolute(resourceDependencyRoot) ||
      path.posix.normalize(resourceDependencyRoot) !== resourceDependencyRoot ||
      resourceDependencyRoot === '/') {
    throw new Error(
      `WASM profile '${profile.id}' resourceDependencyRoot must be a canonical ` +
      'absolute directory.',
    );
  }
  return Object.freeze({
    compiler: requireNonemptyString(
      toolchain.compiler,
      `WASM profile '${profile.id}' compiler`,
    ),
    compilerInvocationPath: requireCanonicalAbsolutePath(
      toolchain.compilerInvocationPath,
      `WASM profile '${profile.id}' compilerInvocationPath`,
    ),
    compilerRealPath: requireCanonicalAbsolutePath(
      toolchain.compilerRealPath,
      `WASM profile '${profile.id}' compilerRealPath`,
    ),
    compilerVersion: requireNonemptyString(
      toolchain.compilerVersion,
      `WASM profile '${profile.id}' compilerVersion`,
    ),
    compilerSha256: evidenceSha(
      toolchain.compilerSha256,
      `WASM profile '${profile.id}' compilerSha256`,
    ),
    compilerRuntimeDependencies: validateToolRuntimeDependencies(
      toolchain.compilerRuntimeDependencies,
      `WASM profile '${profile.id}' compilerRuntimeDependencies`,
    ),
    resourceDependencyRoot,
    linker: requireNonemptyString(
      toolchain.linker,
      `WASM profile '${profile.id}' linker`,
    ),
    linkerInvocationPath: requireCanonicalAbsolutePath(
      toolchain.linkerInvocationPath,
      `WASM profile '${profile.id}' linkerInvocationPath`,
    ),
    linkerRealPath: requireCanonicalAbsolutePath(
      toolchain.linkerRealPath,
      `WASM profile '${profile.id}' linkerRealPath`,
    ),
    linkerVersion: requireNonemptyString(
      toolchain.linkerVersion,
      `WASM profile '${profile.id}' linkerVersion`,
    ),
    linkerSha256: evidenceSha(
      toolchain.linkerSha256,
      `WASM profile '${profile.id}' linkerSha256`,
    ),
    linkerRuntimeDependencies: validateToolRuntimeDependencies(
      toolchain.linkerRuntimeDependencies,
      `WASM profile '${profile.id}' linkerRuntimeDependencies`,
    ),
  });
}

function validateComponent(profile, kind, component) {
  if (component === null || typeof component !== 'object' || Array.isArray(component)) {
    throw new Error(`WASM profile '${profile.id}' component '${kind}' is not an object.`);
  }
  if (!Array.isArray(component.objects) || component.objects.length === 0) {
    throw new Error(`WASM profile '${profile.id}' component '${kind}' has no objects.`);
  }
  const ids = new Set();
  const objects = component.objects.map((object, index) => {
    if (object === null || typeof object !== 'object' || Array.isArray(object)) {
      throw new Error(
        `WASM profile '${profile.id}' component '${kind}' object ${index} is invalid.`,
      );
    }
    const id = safeComponentId(
      object.id,
      `WASM profile '${profile.id}' component '${kind}' object ${index} id`,
    );
    if (ids.has(id)) {
      throw new Error(
        `WASM profile '${profile.id}' component '${kind}' repeats object id '${id}'.`,
      );
    }
    ids.add(id);
    const source = requireNonemptyString(
      object.source,
      `WASM profile '${profile.id}' component '${kind}' object '${id}' source`,
    );
    if (path.isAbsolute(source) || normalizedPath(source).split('/').includes('..')) {
      throw new Error(
        `WASM profile '${profile.id}' component '${kind}' object '${id}' ` +
        `source must be repository-relative: '${source}'.`,
      );
    }
    const category = requireNonemptyString(
      object.category,
      `WASM profile '${profile.id}' component '${kind}' object '${id}' category`,
    );
    if (object.optimization !== '-O3' && object.optimization !== '-Oz') {
      throw new Error(
        `WASM profile '${profile.id}' component '${kind}' object '${id}' ` +
        `optimization must be '-O3' or '-Oz'.`,
      );
    }
    const flags = requireStringArray(
      object.flags,
      `WASM profile '${profile.id}' component '${kind}' object '${id}' flags`,
    );
    const forbidden = flags.find((flag) =>
      OPTIMIZATION_FLAG.test(flag) ||
      ['-c', '-o', '-MD', '-MMD', '-MF', '-MT'].includes(flag));
    if (forbidden !== undefined) {
      throw new Error(
        `WASM profile '${profile.id}' component '${kind}' object '${id}' flags ` +
        `contain builder-owned option '${forbidden}'.`,
      );
    }
    return Object.freeze({
      id,
      source: normalizedPath(source),
      category,
      optimization: object.optimization,
      flags: Object.freeze(flags),
    });
  });
  const linkFlags = requireStringArray(
    component.linkFlags,
    `WASM profile '${profile.id}' component '${kind}' linkFlags`,
  );
  const invalidLinkFlag = linkFlags.find((flag) =>
    flag === '-o' || flag.startsWith('-Wl,'));
  if (invalidLinkFlag !== undefined) {
    throw new Error(
      `WASM profile '${profile.id}' component '${kind}' linkFlags contain ` +
      `invalid direct-linker option '${invalidLinkFlag}'.`,
    );
  }
  const requiredInstructions = component.requiredInstructions === undefined
    ? []
    : requireStringArray(
      component.requiredInstructions,
      `WASM profile '${profile.id}' component '${kind}' requiredInstructions`,
    );
  if (new Set(requiredInstructions).size !== requiredInstructions.length) {
    throw new Error(
      `WASM profile '${profile.id}' component '${kind}' repeats a required instruction.`,
    );
  }
  return Object.freeze({
    objects: Object.freeze(objects),
    linkFlags: Object.freeze(linkFlags),
    requiredInstructions: Object.freeze(requiredInstructions),
  });
}

async function command(executable, arguments_, repositoryRoot) {
  try {
    return await executeFile(executable, arguments_, {
      cwd: repositoryRoot,
      encoding: 'utf8',
      env: {
        PATH: process.env.PATH ?? '',
        LANG: 'C',
        LC_ALL: 'C',
        TZ: 'UTC',
        SOURCE_DATE_EPOCH: '0',
        ...(process.platform === 'win32' && process.env.SystemRoot !== undefined
          ? { SystemRoot: process.env.SystemRoot }
          : {}),
      },
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (error) {
    const detail = [error?.stdout, error?.stderr].filter(Boolean).join('\n').trim();
    throw new Error(
      `${executable} ${arguments_.join(' ')} failed${detail ? `:\n${detail}` : '.'}`,
      { cause: error },
    );
  }
}

async function resolveExecutable(commandName, repositoryRoot = defaultRepositoryRoot) {
  const candidates = commandName.includes(path.sep)
    ? [path.resolve(repositoryRoot, commandName)]
    : (process.env.PATH ?? '').split(path.delimiter)
      .filter((entry) => entry.length > 0)
      .map((entry) => path.resolve(entry, commandName));
  for (const candidate of candidates) {
    try {
      await fs.access(candidate, fsConstants.X_OK);
      return Object.freeze({
        invocationPath: candidate,
        realPath: await fs.realpath(candidate),
      });
    } catch (error) {
      if (!['ENOENT', 'EACCES', 'ENOTDIR'].includes(error?.code)) throw error;
    }
  }
  throw new Error(`WASM release tool '${commandName}' was not found on PATH.`);
}

async function sha256File(filePath) {
  const hasher = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) hasher.update(chunk);
  return hasher.digest('hex');
}

function statIdentity(info) {
  return Object.freeze({
    device: info.dev.toString(),
    inode: info.ino.toString(),
    mode: info.mode.toString(),
    size: info.size.toString(),
    mtimeNs: info.mtimeNs.toString(),
    ctimeNs: info.ctimeNs.toString(),
  });
}

async function executableFingerprint(binding) {
  const [invocationInfo, targetInfo] = await Promise.all([
    fs.lstat(binding.invocationPath, { bigint: true }),
    fs.stat(binding.realPath, { bigint: true }),
  ]);
  if (!targetInfo.isFile() || targetInfo.size <= 0n ||
      targetInfo.size > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(
      `WASM release tool is not a non-empty regular file: '${binding.realPath}'.`,
    );
  }
  const symlinkTarget = invocationInfo.isSymbolicLink()
    ? await fs.readlink(binding.invocationPath)
    : null;
  const identity = Object.freeze({
    invocationPath: binding.invocationPath,
    invocation: statIdentity(invocationInfo),
    symlinkTarget,
    realPath: binding.realPath,
    target: statIdentity(targetInfo),
  });
  const key = stableJSON(identity);
  let digest = toolFingerprintCache.get(key);
  if (digest === undefined) {
    digest = await sha256File(binding.realPath);
    toolFingerprintCache.set(key, digest);
  }
  return Object.freeze({
    identity,
    rawBytes: Number(targetInfo.size),
    sha256: digest,
  });
}

async function runtimeDependencyFingerprint(specification, label) {
  let realPath;
  try {
    realPath = await fs.realpath(specification.path);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(`${label} is missing: '${specification.path}'.`);
    }
    throw error;
  }
  if (realPath !== specification.path) {
    throw new Error(
      `${label} path is not its canonical realpath: '${specification.path}' -> '${realPath}'.`,
    );
  }
  const info = await fs.lstat(realPath, { bigint: true });
  if (!info.isFile() || info.isSymbolicLink() || info.size <= 0n ||
      info.size > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} is not a non-empty regular file: '${realPath}'.`);
  }
  const identity = statIdentity(info);
  const key = `${realPath}\0${stableJSON(identity)}`;
  let digest = runtimeFingerprintCache.get(key);
  if (digest === undefined) {
    digest = await sha256File(realPath);
    runtimeFingerprintCache.set(key, digest);
  }
  const fingerprint = Object.freeze({
    identity,
    path: realPath,
    rawBytes: Number(info.size),
    sha256: digest,
  });
  if (fingerprint.rawBytes !== specification.rawBytes ||
      fingerprint.sha256 !== specification.sha256) {
    throw new Error(
      `${label} differs from the release recipe: '${realPath}' is ` +
      `${fingerprint.rawBytes}/${fingerprint.sha256}, expected ` +
      `${specification.rawBytes}/${specification.sha256}.`,
    );
  }
  return fingerprint;
}

async function runtimeClosureFingerprint(specifications, label) {
  const records = [];
  for (let index = 0; index < specifications.length; index += 1) {
    records.push(await runtimeDependencyFingerprint(
      specifications[index],
      `${label} runtime dependency ${index}`,
    ));
  }
  return Object.freeze(records);
}

async function exactTool(
  commandName,
  expectedInvocationPath,
  expectedRealPath,
  expectedVersion,
  expectedSha256,
  expectedRuntimeDependencies,
  repositoryRoot,
  label,
) {
  const binding = await resolveExecutable(commandName, repositoryRoot);
  if (binding.invocationPath !== expectedInvocationPath ||
      binding.realPath !== expectedRealPath) {
    throw new Error(
      `WASM ${label} '${commandName}' resolves to ` +
      `'${binding.invocationPath}' -> '${binding.realPath}'; expected ` +
      `'${expectedInvocationPath}' -> '${expectedRealPath}'.`,
    );
  }
  const [before, runtimeBefore] = await Promise.all([
    executableFingerprint(binding),
    runtimeClosureFingerprint(expectedRuntimeDependencies, `WASM ${label}`),
  ]);
  // Preserve the invocation basename. LLVM's multicall `lld` binary selects
  // the WebAssembly driver from the `wasm-ld-*` symlink name.
  const result = await command(binding.invocationPath, ['--version'], repositoryRoot);
  const actual = `${result.stdout ?? ''}${result.stderr ?? ''}`
    .split(/\r?\n/u, 1)[0]
    .trim();
  if (actual !== expectedVersion) {
    throw new Error(
      `WASM ${label} '${commandName}' reports ${JSON.stringify(actual)}; ` +
      `the release recipe requires ${JSON.stringify(expectedVersion)}.`,
    );
  }
  if (before.sha256 !== expectedSha256) {
    throw new Error(
      `WASM ${label} '${binding.invocationPath}' has SHA-256 ${before.sha256}; ` +
      `the release recipe requires ${expectedSha256}.`,
    );
  }
  const [after, runtimeAfter] = await Promise.all([
    executableFingerprint(binding),
    runtimeClosureFingerprint(expectedRuntimeDependencies, `WASM ${label}`),
  ]);
  if (stableJSON(before) !== stableJSON(after) ||
      stableJSON(runtimeBefore) !== stableJSON(runtimeAfter)) {
    throw new Error(
      `WASM ${label} process image changed during version validation.`,
    );
  }
  return Object.freeze({
    executable: binding.invocationPath,
    binding,
    repositoryRoot: path.resolve(repositoryRoot),
    evidence: Object.freeze({
      command: commandName,
      path: normalizedPath(binding.invocationPath),
      realPath: normalizedPath(binding.realPath),
      rawBytes: before.rawBytes,
      sha256: before.sha256,
      version: actual,
      runtimeDependencies: Object.freeze(runtimeBefore.map(({ path, rawBytes, sha256 }) =>
        Object.freeze({ path, rawBytes, sha256 }))),
    }),
    fingerprint: before,
    runtimeFingerprints: runtimeBefore,
  });
}

async function assertToolUnchanged(tool, label) {
  const binding = await resolveExecutable(tool.evidence.command, tool.repositoryRoot);
  if (stableJSON(binding) !== stableJSON(tool.binding)) {
    throw new Error(
      `WASM ${label} PATH resolution changed from '${tool.executable}' to ` +
      `'${binding.invocationPath}'.`,
    );
  }
  const current = await executableFingerprint(binding);
  const runtimeCurrent = await runtimeClosureFingerprint(
    tool.evidence.runtimeDependencies,
    `WASM ${label}`,
  );
  if (stableJSON(current) !== stableJSON(tool.fingerprint) ||
      stableJSON(runtimeCurrent) !== stableJSON(tool.runtimeFingerprints)) {
    throw new Error(`WASM ${label} process image changed during the build.`);
  }
}

async function assertRepositorySource(repositoryRoot, relative, label) {
  const absolute = path.resolve(repositoryRoot, relative);
  const repositoryReal = await fs.realpath(repositoryRoot);
  let sourceReal;
  try {
    sourceReal = await fs.realpath(absolute);
  } catch (error) {
    if (error?.code === 'ENOENT') throw new Error(`${label} does not exist: '${relative}'.`);
    throw error;
  }
  const within = path.relative(repositoryReal, sourceReal);
  if (within === '..' || within.startsWith(`..${path.sep}`) || path.isAbsolute(within)) {
    throw new Error(`${label} escapes the repository through a symlink: '${relative}'.`);
  }
  const info = await fs.stat(sourceReal);
  if (!info.isFile()) throw new Error(`${label} is not a file: '${relative}'.`);
  return absolute;
}

function depfileWords(contents, label) {
  const normalized = contents.replace(/\\\r?\n[ \t]*/gu, ' ');
  const colon = normalized.indexOf(':');
  if (colon < 0) throw new Error(`${label} has no dependency target.`);
  const words = [];
  let word = '';
  let escaped = false;
  for (const character of normalized.slice(colon + 1)) {
    if (escaped) {
      word += character;
      escaped = false;
    } else if (character === '\\') {
      escaped = true;
    } else if (/\s/u.test(character)) {
      if (word.length > 0) {
        words.push(word);
        word = '';
      }
    } else {
      word += character;
    }
  }
  if (escaped) throw new Error(`${label} ends with an incomplete escape.`);
  if (word.length > 0) words.push(word);
  if (words.length === 0) throw new Error(`${label} contains no dependencies.`);
  return words;
}

async function dependencyEvidence(
  repositoryRoot,
  dependencyPath,
  label,
  {
    dependencyRoot = repositoryRoot,
    repositoryProjectionRoot = repositoryRoot,
  } = {},
) {
  const absolute = path.isAbsolute(dependencyPath)
    ? path.resolve(dependencyPath)
    : path.resolve(dependencyRoot, dependencyPath);
  const repositoryReal = await fs.realpath(repositoryProjectionRoot);
  let dependencyReal;
  try {
    dependencyReal = await fs.realpath(absolute);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(`${label} no longer exists: '${dependencyPath}'.`);
    }
    throw error;
  }
  const within = path.relative(repositoryReal, dependencyReal);
  const artifact = await readArtifact(dependencyReal, label);
  const repositoryFile = within !== '' && within !== '..' &&
    !within.startsWith(`..${path.sep}`) && !path.isAbsolute(within);
  return Object.freeze({
    scope: repositoryFile ? 'repository' : 'toolchain',
    path: normalizedPath(repositoryFile ? within : dependencyReal),
    rawBytes: artifact.bytes,
    sha256: artifact.sha256,
  });
}

async function readDependencyEvidence(repositoryRoot, depfilePath, label, options = {}) {
  const contents = await fs.readFile(depfilePath, 'utf8');
  const dependencies = new Map();
  for (const dependencyPath of depfileWords(contents, label)) {
    const dependency = await dependencyEvidence(
      repositoryRoot,
      dependencyPath,
      `${label} dependency`,
      options,
    );
    dependencies.set(`${dependency.scope}\0${dependency.path}`, dependency);
  }
  return Object.freeze([...dependencies.values()]
    .sort((left, right) => `${left.scope}\0${left.path}`
      .localeCompare(`${right.scope}\0${right.path}`, 'en')));
}

async function compileObjectFromSnapshot({
  repositoryRoot,
  componentDirectory,
  objectsDirectory,
  compiler,
  profile,
  kind,
  object,
  index,
  objectPath,
  requiredInstructions,
}) {
  const label = `WASM ${profile.id}/${kind} object '${object.id}'`;
  await assertRepositorySource(repositoryRoot, object.source, `${label} source`);
  const prefix = `${String(index).padStart(3, '0')}-${object.id}`;
  const discoveryDepfile = path.join(objectsDirectory, `${prefix}.discovery.d`);
  const compileDepfile = path.join(objectsDirectory, `${prefix}.d`);
  const compileFlags = [...object.flags, object.optimization];
  await assertSafePrivateOutputs(
    [discoveryDepfile, compileDepfile, objectPath],
    `${label} private output`,
  );
  await command(compiler.executable, [
    ...compileFlags,
    '-MD',
    '-MF',
    discoveryDepfile,
    '-MT',
    object.id,
    '-E',
    object.source,
    '-o',
    '/dev/null',
  ], repositoryRoot);
  const discoveredDependencies = await readDependencyEvidence(
    repositoryRoot,
    discoveryDepfile,
    `${label} dependency discovery`,
  );
  if (!discoveredDependencies.some((dependency) =>
    dependency.scope === 'repository' && dependency.path === object.source)) {
    throw new Error(`${label} dependency discovery omits its source.`);
  }

  const snapshotRoot = await fs.mkdtemp(
    path.join(componentDirectory, `.snapshot-${object.id}-`),
  );
  try {
    for (const dependency of discoveredDependencies) {
      if (dependency.scope !== 'repository') continue;
      const sourcePath = path.resolve(repositoryRoot, dependency.path);
      const contents = await fs.readFile(sourcePath);
      if (contents.byteLength !== dependency.rawBytes ||
          sha256(contents) !== dependency.sha256) {
        throw new Error(`${label} dependency '${dependency.path}' changed before snapshot.`);
      }
      const snapshotPath = path.resolve(snapshotRoot, dependency.path);
      if (!pathIsWithin(snapshotRoot, snapshotPath)) {
        throw new Error(`${label} has unsafe snapshot dependency '${dependency.path}'.`);
      }
      await fs.mkdir(path.dirname(snapshotPath), { recursive: true });
      await fs.writeFile(snapshotPath, contents, { flag: 'wx', mode: 0o444 });
    }

    const actualArguments = [
      ...compileFlags,
      '-MD',
      '-MF',
      compileDepfile,
      '-MT',
      object.id,
      '-c',
      object.source,
      '-o',
      objectPath,
    ];
    await command(compiler.executable, actualArguments, snapshotRoot);
    const artifact = await readArtifact(objectPath, label);
    const compiledDependencies = await readDependencyEvidence(
      repositoryRoot,
      compileDepfile,
      `${label} depfile`,
      {
        dependencyRoot: snapshotRoot,
        repositoryProjectionRoot: snapshotRoot,
      },
    );
    if (stableJSON(compiledDependencies) !== stableJSON(discoveredDependencies)) {
      throw new Error(`${label} snapshot dependency graph differs from discovery.`);
    }

    let assembly = null;
    if (requiredInstructions.length > 0) {
      const result = await command(compiler.executable, [
        ...compileFlags,
        '-S',
        object.source,
        '-o',
        '-',
      ], snapshotRoot);
      const contents = result.stdout ?? '';
      const occurrences = Object.fromEntries(
        requiredInstructions.map((instruction) => [instruction, 0]),
      );
      for (const line of contents.split(/\r?\n/u)) {
        const instruction = line.trim().split(/\s+/u, 1)[0];
        if (Object.hasOwn(occurrences, instruction)) occurrences[instruction] += 1;
      }
      assembly = Object.freeze({
        sha256: sha256(contents),
        occurrences: Object.freeze(occurrences),
      });
    }
    return Object.freeze({
      artifact,
      dependencies: compiledDependencies,
      assembly,
    });
  } finally {
    await fs.rm(snapshotRoot, { recursive: true, force: true });
  }
}

async function readArtifact(artifactPath, label, { wasm = false } = {}) {
  const bytes = await fs.readFile(artifactPath);
  if (bytes.length === 0) throw new Error(`${label} is empty: '${artifactPath}'.`);
  if (wasm && (bytes.length < WASM_MAGIC.length ||
      !bytes.subarray(0, WASM_MAGIC.length).equals(WASM_MAGIC))) {
    throw new Error(`${label} is not a WebAssembly module: '${artifactPath}'.`);
  }
  return Object.freeze({ bytes: bytes.length, sha256: sha256(bytes), contents: bytes });
}

function canonicalEvidenceArtifact(evidence) {
  const contents = Buffer.from(`${stableJSON(evidence)}\n`);
  return Object.freeze({
    contents,
    bytes: contents.length,
    sha256: sha256(contents),
  });
}

function componentBuildDirectory(buildRoot, profileId, kind) {
  const safeProfile = safeComponentId(profileId, 'WASM profile id');
  const safeKind = safeComponentId(kind, 'WASM component kind');
  const root = path.resolve(buildRoot);
  const directory = path.resolve(root, 'objects', safeProfile, safeKind);
  const relative = path.relative(root, directory);
  if (relative === '' || relative === '..' || relative.startsWith(`..${path.sep}`) ||
      path.isAbsolute(relative)) {
    throw new Error(`Unsafe WASM component build directory '${directory}'.`);
  }
  return directory;
}

function pathIsWithin(directory, candidate) {
  const relative = path.relative(path.resolve(directory), path.resolve(candidate));
  return relative === '' ||
    (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}

async function canonicalPlannedPath(candidate, label) {
  const absolute = path.resolve(candidate);
  const suffix = [];
  let cursor = absolute;
  for (;;) {
    try {
      await fs.lstat(cursor);
      let existingReal;
      try {
        existingReal = await fs.realpath(cursor);
      } catch (error) {
        if (error?.code === 'ENOENT') {
          throw new Error(`${label} contains a dangling symbolic link: '${cursor}'.`);
        }
        throw error;
      }
      return path.resolve(existingReal, ...suffix);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      const parent = path.dirname(cursor);
      if (parent === cursor) {
        throw new Error(`${label} has no existing filesystem ancestor: '${absolute}'.`);
      }
      suffix.unshift(path.basename(cursor));
      cursor = parent;
    }
  }
}

async function distinctPathIdentities(entries) {
  const selected = entries.filter(([, value]) => value !== null && value !== undefined);
  const identities = await Promise.all(selected.map(async ([label, selectedPath]) =>
    Object.freeze({
      label,
      path: selectedPath,
      canonical: await canonicalPlannedPath(selectedPath, label),
    })));
  for (let left = 0; left < identities.length; left += 1) {
    for (let right = left + 1; right < identities.length; right += 1) {
      if (identities[left].canonical === identities[right].canonical) {
        throw new Error(
          `${identities[left].label} and ${identities[right].label} must use different paths.`,
        );
      }
    }
  }
  return identities;
}

async function assertDistinctBuildPaths(entries, componentDirectory, sourcePaths = []) {
  const selected = await distinctPathIdentities(entries);
  const componentCanonical = await canonicalPlannedPath(
    componentDirectory,
    'WASM private object directory',
  );
  const sources = await Promise.all(sourcePaths.map(async (sourcePath) =>
    canonicalPlannedPath(sourcePath, 'WASM release-recipe source')));
  for (let left = 0; left < selected.length; left += 1) {
    const { label: leftLabel, canonical: leftPath } = selected[left];
    if (pathIsWithin(componentCanonical, leftPath)) {
      throw new Error(`${leftLabel} must be outside the private object directory.`);
    }
    if (sources.includes(leftPath)) {
      throw new Error(`${leftLabel} must not overwrite a release-recipe source file.`);
    }
  }
}

async function ensurePrivateBuildDirectory(buildRoot, directory) {
  const root = path.resolve(buildRoot);
  const target = path.resolve(directory);
  const relative = path.relative(root, target);
  if (relative === '' || relative === '..' || relative.startsWith(`..${path.sep}`) ||
      path.isAbsolute(relative)) {
    throw new Error(`Private WASM directory escapes its build root: '${target}'.`);
  }
  const plannedRoot = await canonicalPlannedPath(root, 'Private WASM build root');
  if (plannedRoot !== root) {
    throw new Error(
      `Private WASM build root contains a symbolic-link alias: '${root}' -> '${plannedRoot}'.`,
    );
  }
  await fs.mkdir(root, { recursive: true });
  const rootInfo = await fs.lstat(root);
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) {
    throw new Error(`Private WASM build root is not a real directory: '${root}'.`);
  }
  const rootReal = await fs.realpath(root);
  if (rootReal !== root) {
    throw new Error(
      `Private WASM build root contains a symbolic-link alias: '${root}' -> '${rootReal}'.`,
    );
  }
  let cursor = root;
  for (const segment of relative.split(path.sep)) {
    cursor = path.join(cursor, segment);
    try {
      const info = await fs.lstat(cursor);
      if (info.isSymbolicLink() || !info.isDirectory()) {
        throw new Error(`Private WASM build path is not a real directory: '${cursor}'.`);
      }
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      await fs.mkdir(cursor);
    }
    const currentReal = await fs.realpath(cursor);
    if (!pathIsWithin(rootReal, currentReal)) {
      throw new Error(`Private WASM build path escapes its build root: '${cursor}'.`);
    }
  }
}

let publishTransactionSequence = 0;

function publishSiblingPath(destination, role, index) {
  publishTransactionSequence += 1;
  return path.join(
    path.dirname(destination),
    `.${path.basename(destination)}.volvoxai-${role}-${process.pid}-` +
      `${Date.now()}-${publishTransactionSequence}-${index}`,
  );
}

async function destinationState(destination, label) {
  try {
    const info = await fs.lstat(destination, { bigint: true });
    if (!info.isFile() && !info.isSymbolicLink()) {
      throw new Error(`${label} is neither a regular file nor a symbolic link.`);
    }
    return Object.freeze({
      exists: true,
      identity: statIdentity(info),
      symlinkTarget: info.isSymbolicLink() ? await fs.readlink(destination) : null,
    });
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    return Object.freeze({ exists: false });
  }
}

async function assertPublicationInputAliases(entries, protectedInputs) {
  const destinations = await distinctPathIdentities(entries.map(({ label, destination }) =>
    [label, destination]));
  const inputs = new Set(await Promise.all(protectedInputs.map((inputPath) =>
    canonicalPlannedPath(inputPath, 'WASM protected build input'))));
  for (const destination of destinations) {
    if (inputs.has(destination.canonical)) {
      throw new Error(`${destination.label} must not overwrite a protected build input.`);
    }
  }
}

async function capturePublicationDestination(entry) {
  const requestedDestination = path.resolve(entry.destination);
  const requestedParent = path.dirname(requestedDestination);
  const canonicalParent = await canonicalPlannedPath(
    requestedParent,
    `${entry.label} publication directory`,
  );

  // Create through the canonical path captured above. If requestedParent is a
  // symlink and is retargeted concurrently, mkdir must not follow the new
  // target and create directories outside the destination observed here.
  await fs.mkdir(canonicalParent, { recursive: true });
  const realizedParent = await fs.realpath(canonicalParent);
  const requestedParentNow = await canonicalPlannedPath(
    requestedParent,
    `${entry.label} publication directory`,
  );
  if (realizedParent !== canonicalParent || requestedParentNow !== canonicalParent) {
    throw new Error(`${entry.label} publication directory changed during initialization.`);
  }

  const destination = path.join(canonicalParent, path.basename(requestedDestination));
  const [requestedCanonical, canonicalDestination, previous] = await Promise.all([
    canonicalPlannedPath(requestedDestination, entry.label),
    canonicalPlannedPath(destination, entry.label),
    destinationState(destination, entry.label),
  ]);
  if (requestedCanonical !== canonicalDestination) {
    throw new Error(`${entry.label} destination changed during initialization.`);
  }
  return {
    ...entry,
    requestedDestination,
    canonicalParent,
    canonicalDestination,
    destination,
    previous,
  };
}

async function assertPublicationDestinationUnchanged(entry) {
  const [requestedParent, fixedParent, requestedDestination, fixedDestination, current] =
    await Promise.all([
      canonicalPlannedPath(
        path.dirname(entry.requestedDestination),
        `${entry.label} publication directory`,
      ),
      canonicalPlannedPath(
        path.dirname(entry.destination),
        `${entry.label} fixed publication directory`,
      ),
      canonicalPlannedPath(entry.requestedDestination, entry.label),
      canonicalPlannedPath(entry.destination, entry.label),
      destinationState(entry.destination, entry.label),
    ]);
  if (requestedParent !== entry.canonicalParent || fixedParent !== entry.canonicalParent) {
    throw new Error(`${entry.label} publication directory changed before commit.`);
  }
  if (requestedDestination !== entry.canonicalDestination ||
      fixedDestination !== entry.canonicalDestination) {
    throw new Error(`${entry.label} destination changed before commit.`);
  }
  if (stableJSON(current) !== stableJSON(entry.previous)) {
    throw new Error(`${entry.label} changed before the publish transaction committed.`);
  }
}

/**
 * Stage every publication before touching a destination. If any rename or
 * post-publish validation fails, restore the complete previous file set.
 * The evidence file is deliberately committed last and acts as the snapshot
 * marker for readers that validate an artifact/evidence pair.
 */
async function publishFileTransaction(entries, {
  protectedInputs = [],
  precommit = null,
} = {}) {
  if (!Array.isArray(protectedInputs)) {
    throw new Error('WASM publish protectedInputs must be an array.');
  }
  if (precommit !== null && typeof precommit !== 'function') {
    throw new Error('WASM publish precommit must be a function or null.');
  }
  const requested = entries.filter(({ destination }) =>
    destination !== null && destination !== undefined);
  if (requested.length === 0) return;
  // Freeze the real publication parent before staging. All stage, backup,
  // rename, validation, rollback, and cleanup paths below use this fixed
  // identity rather than a caller-visible symlink that can be retargeted.
  const selected = [];
  for (const entry of requested) {
    selected.push(await capturePublicationDestination(entry));
  }
  await assertPublicationInputAliases(selected, protectedInputs);

  const prepared = [];
  let preserveBackups = false;
  try {
    for (let index = 0; index < selected.length; index += 1) {
      const entry = selected[index];
      const contents = Buffer.from(entry.contents);
      if (contents.length === 0) throw new Error(`${entry.label} publication is empty.`);
      const stage = publishSiblingPath(entry.destination, 'stage', index);
      const preparedEntry = {
        ...entry,
        contents,
        stage,
        stageCreated: false,
        backup: null,
        installed: false,
      };
      prepared.push(preparedEntry);
      const stageHandle = await fs.open(stage, 'wx', 0o644);
      preparedEntry.stageCreated = true;
      try {
        await stageHandle.writeFile(contents);
      } finally {
        await stageHandle.close();
      }
      const artifact = await readArtifact(stage, `${entry.label} staged publication`, {
        wasm: entry.wasm === true,
      });
      if (artifact.bytes !== contents.length || artifact.sha256 !== sha256(contents)) {
        throw new Error(`${entry.label} changed while it was staged.`);
      }
    }

    // Staging may take long enough for an earlier observation to become stale.
    // Rehash the caller's complete graph, then repeat alias/destination checks
    // immediately before the first public destination can change.
    if (precommit !== null) await precommit();
    await assertPublicationInputAliases(prepared, protectedInputs);
    for (const entry of prepared) {
      await assertPublicationDestinationUnchanged(entry);
    }

    try {
      for (let index = 0; index < prepared.length; index += 1) {
        const entry = prepared[index];
        // Repeat immediately before this particular destination changes; a
        // transaction with several artifacts may spend time committing its
        // earlier entries.
        await assertPublicationDestinationUnchanged(entry);
        if (entry.previous.exists) {
          const backup = publishSiblingPath(entry.destination, 'backup', index);
          await fs.rename(entry.destination, backup);
          entry.backup = backup;
        }
        await fs.rename(entry.stage, entry.destination);
        entry.installed = true;
      }
      for (const entry of prepared) {
        const published = await readArtifact(
          entry.destination,
          `${entry.label} published artifact`,
          { wasm: entry.wasm === true },
        );
        if (published.bytes !== entry.contents.length ||
            published.sha256 !== sha256(entry.contents)) {
          throw new Error(`${entry.label} changed while it was published.`);
        }
      }
    } catch (error) {
      const rollbackErrors = [];
      for (const entry of [...prepared].reverse()) {
        try {
          if (entry.installed) await fs.rm(entry.destination, { force: true });
          if (entry.backup !== null) {
            await fs.rename(entry.backup, entry.destination);
            entry.backup = null;
          }
        } catch (rollbackError) {
          rollbackErrors.push(rollbackError);
        }
      }
      if (rollbackErrors.length > 0) {
        preserveBackups = true;
        throw new AggregateError(
          [error, ...rollbackErrors],
          'WASM publish transaction failed and could not completely restore its inputs.',
        );
      }
      throw error;
    }

    for (const entry of prepared) {
      if (entry.backup !== null) {
        await fs.rm(entry.backup, { force: true });
        entry.backup = null;
      }
    }
  } finally {
    await Promise.all(prepared.flatMap((entry) => [
      ...(entry.stageCreated ? [entry.stage] : []),
      ...(preserveBackups ? [] : [entry.backup]),
    ]
      .filter((candidate) => candidate !== null)
      .map((candidate) => fs.rm(candidate, { force: true }))));
  }
}

async function coalesceProfilePublications(entries) {
  const selected = [];
  const byCanonicalPath = new Map();
  for (const entry of entries) {
    const canonical = await canonicalPlannedPath(entry.destination, entry.label);
    const previous = byCanonicalPath.get(canonical);
    if (previous === undefined) {
      byCanonicalPath.set(canonical, entry);
      selected.push(entry);
      continue;
    }
    const sameSharedArtifact = entry.sharedKey !== undefined &&
      entry.sharedKey === previous.sharedKey &&
      entry.wasm === true && previous.wasm === true &&
      Buffer.from(entry.contents).equals(Buffer.from(previous.contents));
    if (!sameSharedArtifact) {
      throw new Error(`${previous.label} and ${entry.label} must use different paths.`);
    }
  }
  return Object.freeze([
    ...selected.filter(({ commitMarker }) => commitMarker !== true),
    ...selected.filter(({ commitMarker }) => commitMarker === true),
  ]);
}

async function assertSafePrivateOutputs(paths, label) {
  for (const outputPath of paths) {
    try {
      const info = await fs.lstat(outputPath, { bigint: true });
      if (!info.isFile() || info.isSymbolicLink() || info.nlink !== 1n) {
        throw new Error(`${label} is not a private regular file: '${outputPath}'.`);
      }
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }
}

export function wasmBuildObjectPath(buildRoot, profileValue, kindValue, index, idValue) {
  const profile = profileFrom(profileValue);
  const kind = canonicalKind(kindValue);
  if (!Number.isSafeInteger(index) || index < 0) {
    throw new Error(`WASM object index must be a nonnegative safe integer; found '${index}'.`);
  }
  const id = safeComponentId(idValue, 'WASM object id');
  return path.join(
    componentBuildDirectory(buildRoot, profile.id, kind),
    'objects',
    `${String(index).padStart(3, '0')}-${id}.o`,
  );
}

function buildEvidenceDocument(profile, toolchain, components) {
  return Object.freeze({
    format: WASM_BUILD_EVIDENCE_FORMAT,
    profile: profile.id,
    artifact: profile.filename,
    recipeSha256: sha256(stableJSON(profile.recipe)),
    toolchain,
    components: Object.freeze([...components]),
  });
}

function exactKeys(value, expected, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (stableJSON(actual) !== stableJSON(wanted)) {
    throw new Error(
      `${label} fields are ${actual.join(', ')}; expected ${wanted.join(', ')}.`,
    );
  }
}

function evidenceSha(value, label) {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/u.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

function evidenceBytes(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return value;
}

function deepFreeze(value) {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

/** Validate canonical build evidence against the central data-only recipe. */
export function validateWasmBuildEvidence(
  document,
  profileValue,
  { allowPartial = false } = {},
) {
  const profile = profileFrom(profileValue);
  exactKeys(document, [
    'format', 'profile', 'artifact', 'recipeSha256', 'toolchain', 'components',
  ], 'WASM build evidence');
  if (document.format !== WASM_BUILD_EVIDENCE_FORMAT) {
    throw new Error(`Unsupported WASM build evidence format '${document.format}'.`);
  }
  if (document.profile !== profile.id || document.artifact !== profile.filename) {
    throw new Error(
      `WASM build evidence identifies '${document.profile}/${document.artifact}'; ` +
      `expected '${profile.id}/${profile.filename}'.`,
    );
  }
  const expectedRecipeSha256 = sha256(stableJSON(profile.recipe));
  if (document.recipeSha256 !== expectedRecipeSha256) {
    throw new Error(
      `WASM ${profile.id} build evidence recipe is '${document.recipeSha256}'; ` +
      `expected '${expectedRecipeSha256}'.`,
    );
  }

  const recipeToolchain = validateToolchain(profile);
  exactKeys(document.toolchain, ['compiler', 'linker'], 'WASM build evidence toolchain');
  for (const [
    kind,
    expectedCommand,
    expectedInvocationPath,
    expectedRealPath,
    expectedVersion,
    expectedSha256,
    expectedRuntimeDependencies,
  ] of [
    [
      'compiler',
      recipeToolchain.compiler,
      recipeToolchain.compilerInvocationPath,
      recipeToolchain.compilerRealPath,
      recipeToolchain.compilerVersion,
      recipeToolchain.compilerSha256,
      recipeToolchain.compilerRuntimeDependencies,
    ],
    [
      'linker',
      recipeToolchain.linker,
      recipeToolchain.linkerInvocationPath,
      recipeToolchain.linkerRealPath,
      recipeToolchain.linkerVersion,
      recipeToolchain.linkerSha256,
      recipeToolchain.linkerRuntimeDependencies,
    ],
  ]) {
    const tool = document.toolchain[kind];
    exactKeys(
      tool,
      [
        'command', 'path', 'realPath', 'rawBytes', 'sha256', 'version',
        'runtimeDependencies',
      ],
      `WASM build evidence ${kind}`,
    );
    if (tool.command !== expectedCommand || tool.version !== expectedVersion ||
        tool.path !== expectedInvocationPath || tool.realPath !== expectedRealPath) {
      throw new Error(`WASM build evidence ${kind} differs from the release toolchain.`);
    }
    evidenceBytes(tool.rawBytes, `WASM build evidence ${kind} rawBytes`);
    evidenceSha(tool.sha256, `WASM build evidence ${kind} sha256`);
    if (tool.sha256 !== expectedSha256) {
      throw new Error(`WASM build evidence ${kind} digest differs from the release toolchain.`);
    }
    const actualRuntimeDependencies = validateToolRuntimeDependencies(
      tool.runtimeDependencies,
      `WASM build evidence ${kind} runtimeDependencies`,
    );
    if (stableJSON(actualRuntimeDependencies) !== stableJSON(expectedRuntimeDependencies)) {
      throw new Error(
        `WASM build evidence ${kind} runtime dependencies differ from the release toolchain.`,
      );
    }
  }

  const expectedKinds = COMPONENT_ORDER.filter((kind) =>
    profile.recipe?.[COMPONENT_PROPERTIES[kind]] !== null &&
    profile.recipe?.[COMPONENT_PROPERTIES[kind]] !== undefined);
  const actualKinds = Array.isArray(document.components)
    ? document.components.map(({ kind }) => kind)
    : [];
  const expectedActualKinds = allowPartial
    ? expectedKinds.filter((kind) => actualKinds.includes(kind))
    : expectedKinds;
  if (!Array.isArray(document.components) || actualKinds.length === 0 ||
      stableJSON(actualKinds) !== stableJSON(expectedActualKinds)) {
    throw new Error(
      `WASM ${profile.id} build evidence component order differs from the recipe.`,
    );
  }
  for (let componentIndex = 0; componentIndex < actualKinds.length; componentIndex += 1) {
    const kind = actualKinds[componentIndex];
    const actual = document.components[componentIndex];
    const expected = validateComponent(
      profile,
      kind,
      profile.recipe[COMPONENT_PROPERTIES[kind]],
    );
    exactKeys(
      actual,
      ['kind', 'artifact', 'orderedObjects', 'link', 'instructionAudit'],
      `WASM ${profile.id}/${kind} build evidence`,
    );
    if (actual.kind !== kind) {
      throw new Error(`WASM build evidence component ${componentIndex} must be '${kind}'.`);
    }
    exactKeys(actual.artifact, ['rawBytes', 'sha256'], `WASM ${profile.id}/${kind} artifact`);
    evidenceBytes(actual.artifact.rawBytes, `WASM ${profile.id}/${kind} artifact rawBytes`);
    evidenceSha(actual.artifact.sha256, `WASM ${profile.id}/${kind} artifact sha256`);
    if (!Array.isArray(actual.orderedObjects) ||
        actual.orderedObjects.length !== expected.objects.length) {
      throw new Error(`WASM ${profile.id}/${kind} object count differs from the recipe.`);
    }
    for (let index = 0; index < expected.objects.length; index += 1) {
      const actualObject = actual.orderedObjects[index];
      const expectedObject = expected.objects[index];
      exactKeys(actualObject, [
        'index', 'id', 'source', 'category', 'optimization', 'flags',
        'flagsSha256', 'dependencies', 'rawBytes', 'sha256',
      ], `WASM ${profile.id}/${kind} object ${index}`);
      for (const field of ['id', 'source', 'category', 'optimization']) {
        if (actualObject[field] !== expectedObject[field]) {
          throw new Error(
            `WASM ${profile.id}/${kind} object ${index} ${field} differs from the recipe.`,
          );
        }
      }
      if (actualObject.index !== index ||
          stableJSON(actualObject.flags) !== stableJSON(expectedObject.flags) ||
          actualObject.flagsSha256 !== sha256(stableJSON(expectedObject.flags))) {
        throw new Error(`WASM ${profile.id}/${kind} object ${index} flags/order are invalid.`);
      }
      evidenceBytes(
        actualObject.rawBytes,
        `WASM ${profile.id}/${kind} object ${index} rawBytes`,
      );
      evidenceSha(actualObject.sha256, `WASM ${profile.id}/${kind} object ${index} sha256`);
      if (!Array.isArray(actualObject.dependencies) ||
          actualObject.dependencies.length === 0) {
        throw new Error(`WASM ${profile.id}/${kind} object ${index} has no dependencies.`);
      }
      const dependencyPaths = [];
      for (let dependencyIndex = 0;
        dependencyIndex < actualObject.dependencies.length;
        dependencyIndex += 1) {
        const dependency = actualObject.dependencies[dependencyIndex];
        exactKeys(
          dependency,
          ['scope', 'path', 'rawBytes', 'sha256'],
          `WASM ${profile.id}/${kind} object ${index} dependency ${dependencyIndex}`,
        );
        const repositoryDependency = dependency.scope === 'repository';
        const toolchainDependency = dependency.scope === 'toolchain';
        if ((!repositoryDependency && !toolchainDependency) ||
            typeof dependency.path !== 'string' || dependency.path.length === 0 ||
            dependency.path.includes('\\') || dependency.path.includes('\0') ||
            path.posix.normalize(dependency.path) !== dependency.path ||
            (repositoryDependency && (
              path.posix.isAbsolute(dependency.path) ||
              dependency.path.split('/').includes('..')
            )) ||
            (toolchainDependency && !path.posix.isAbsolute(dependency.path))) {
          throw new Error(
            `WASM ${profile.id}/${kind} object ${index} has unsafe dependency ` +
            `'${String(dependency.path)}'.`,
          );
        }
        if (toolchainDependency &&
            dependency.path !== recipeToolchain.resourceDependencyRoot &&
            !dependency.path.startsWith(`${recipeToolchain.resourceDependencyRoot}/`)) {
          throw new Error(
            `WASM ${profile.id}/${kind} object ${index} has dependency outside the ` +
            `pinned compiler resource root: '${dependency.path}'.`,
          );
        }
        evidenceBytes(
          dependency.rawBytes,
          `WASM ${profile.id}/${kind} dependency '${dependency.path}' rawBytes`,
        );
        evidenceSha(
          dependency.sha256,
          `WASM ${profile.id}/${kind} dependency '${dependency.path}' sha256`,
        );
        dependencyPaths.push(`${dependency.scope}\0${dependency.path}`);
      }
      const sortedDependencies = [...new Set(dependencyPaths)].sort((left, right) =>
        left.localeCompare(right, 'en'));
      if (stableJSON(dependencyPaths) !== stableJSON(sortedDependencies) ||
          !dependencyPaths.includes(`repository\0${expectedObject.source}`)) {
        throw new Error(
          `WASM ${profile.id}/${kind} object ${index} dependency order/source is invalid.`,
        );
      }
      const repositoryDependencyPaths = actualObject.dependencies
        .filter(({ scope }) => scope === 'repository')
        .map(({ path: dependencyPath }) => dependencyPath);
      const forbiddenPatterns = profile.forbiddenDependencyPatterns ?? [];
      const forbiddenDependency = repositoryDependencyPaths.find((dependency) =>
        profile.forbiddenDependencyPrefixes.some((prefix) => dependency.startsWith(prefix)) ||
        profile.forbiddenDependencyFiles.includes(dependency) ||
        forbiddenPatterns.some((pattern) => new RegExp(pattern, 'u').test(dependency)));
      if (forbiddenDependency !== undefined) {
        throw new Error(
          `WASM ${profile.id}/${kind} object ${index} depends on forbidden profile source ` +
          `'${forbiddenDependency}'.`,
        );
      }
    }
    exactKeys(actual.link, ['flags', 'flagsSha256', 'orderedObjectIds'],
      `WASM ${profile.id}/${kind} link`);
    if (stableJSON(actual.link.flags) !== stableJSON(expected.linkFlags) ||
        actual.link.flagsSha256 !== sha256(stableJSON(expected.linkFlags)) ||
        stableJSON(actual.link.orderedObjectIds) !==
          stableJSON(expected.objects.map(({ id }) => id))) {
      throw new Error(`WASM ${profile.id}/${kind} link evidence differs from the recipe.`);
    }
    if (expected.requiredInstructions.length === 0) {
      if (actual.instructionAudit !== null) {
        throw new Error(`WASM ${profile.id}/${kind} has unexpected instruction evidence.`);
      }
    } else {
      exactKeys(actual.instructionAudit, [
        'requiredInstructions', 'occurrences', 'assemblySha256',
      ], `WASM ${profile.id}/${kind} instruction evidence`);
      if (stableJSON(actual.instructionAudit.requiredInstructions) !==
          stableJSON(expected.requiredInstructions)) {
        throw new Error(`WASM ${profile.id}/${kind} required instructions differ.`);
      }
      exactKeys(
        actual.instructionAudit.occurrences,
        expected.requiredInstructions,
        `WASM ${profile.id}/${kind} instruction occurrences`,
      );
      for (const instruction of expected.requiredInstructions) {
        if (!Number.isSafeInteger(actual.instructionAudit.occurrences[instruction]) ||
            actual.instructionAudit.occurrences[instruction] <= 0) {
          throw new Error(
            `WASM ${profile.id}/${kind} does not prove instruction '${instruction}'.`,
          );
        }
      }
      evidenceSha(
        actual.instructionAudit.assemblySha256,
        `WASM ${profile.id}/${kind} instruction assemblySha256`,
      );
    }
  }
  return deepFreeze(JSON.parse(stableJSON(document)));
}

export async function readWasmBuildEvidenceRecord(
  evidencePath,
  profileValue,
  options = {},
) {
  const contents = await fs.readFile(evidencePath);
  let document;
  try {
    document = JSON.parse(contents.toString('utf8'));
  } catch (error) {
    throw new Error(`Invalid WASM build evidence JSON '${evidencePath}'.`, { cause: error });
  }
  const validated = validateWasmBuildEvidence(document, profileValue, options);
  const canonical = Buffer.from(`${stableJSON(validated)}\n`);
  if (!contents.equals(canonical)) {
    throw new Error(`WASM build evidence '${evidencePath}' is not canonical JSON.`);
  }
  return Object.freeze({
    document: validated,
    rawBytes: contents.byteLength,
    sha256: sha256(contents),
  });
}

export async function readWasmBuildEvidence(evidencePath, profileValue, options = {}) {
  return (await readWasmBuildEvidenceRecord(evidencePath, profileValue, options)).document;
}

export async function validateWasmBuildEvidenceTools(
  document,
  profileValue,
  options = {},
) {
  const profile = profileFrom(profileValue);
  const { repositoryRoot = defaultRepositoryRoot, ...validationOptions } = options;
  const evidence = validateWasmBuildEvidence(document, profile, validationOptions);
  const recipeToolchain = validateToolchain(profile);
  const actual = await Promise.all([
    exactTool(
      recipeToolchain.compiler,
      recipeToolchain.compilerInvocationPath,
      recipeToolchain.compilerRealPath,
      recipeToolchain.compilerVersion,
      recipeToolchain.compilerSha256,
      recipeToolchain.compilerRuntimeDependencies,
      repositoryRoot,
      'compiler',
    ),
    exactTool(
      recipeToolchain.linker,
      recipeToolchain.linkerInvocationPath,
      recipeToolchain.linkerRealPath,
      recipeToolchain.linkerVersion,
      recipeToolchain.linkerSha256,
      recipeToolchain.linkerRuntimeDependencies,
      repositoryRoot,
      'linker',
    ),
  ]);
  for (const [index, kind] of ['compiler', 'linker'].entries()) {
    if (stableJSON(actual[index].evidence) !== stableJSON(evidence.toolchain[kind])) {
      throw new Error(`WASM build evidence ${kind} executable fingerprint is stale.`);
    }
  }
  return Object.freeze({
    compiler: actual[0].evidence,
    linker: actual[1].evidence,
  });
}

export async function validateWasmBuildEvidenceObjects(
  buildRoot,
  document,
  profileValue,
  options = {},
) {
  const profile = profileFrom(profileValue);
  const {
    repositoryRoot = defaultRepositoryRoot,
    verifyToolchainInputs = true,
    ...validationOptions
  } = options;
  if (typeof verifyToolchainInputs !== 'boolean') {
    throw new Error('verifyToolchainInputs must be a boolean.');
  }
  const evidence = validateWasmBuildEvidence(document, profile, validationOptions);
  if (verifyToolchainInputs) {
    await validateWasmBuildEvidenceTools(evidence, profile, {
      repositoryRoot,
      ...validationOptions,
    });
  }
  const records = [];
  let objectCount = 0;
  let dependencyCount = 0;
  let repositoryDependencyCount = 0;
  let toolchainDependencyCount = 0;
  for (const component of evidence.components) {
    for (const object of component.orderedObjects) {
      const objectPath = wasmBuildObjectPath(
        buildRoot,
        profile,
        component.kind,
        object.index,
        object.id,
      );
      const actual = await readArtifact(
        objectPath,
        `WASM ${profile.id}/${component.kind} object '${object.id}'`,
      );
      if (actual.bytes !== object.rawBytes || actual.sha256 !== object.sha256) {
        throw new Error(
          `WASM ${profile.id}/${component.kind} object '${object.id}' ` +
          'does not match its build evidence.',
        );
      }
      records.push(`${component.kind}\0${object.index}\0${object.id}\0` +
        `${actual.bytes}\0${actual.sha256}\n`);
      objectCount += 1;
      for (const dependency of object.dependencies) {
        const current = dependency.scope === 'toolchain' && !verifyToolchainInputs
          ? dependency
          : await dependencyEvidence(
            repositoryRoot,
            dependency.path,
            `WASM ${profile.id}/${component.kind} dependency '${dependency.path}'`,
          );
        if (current.scope !== dependency.scope || current.path !== dependency.path ||
            current.rawBytes !== dependency.rawBytes ||
            current.sha256 !== dependency.sha256) {
          throw new Error(
            `WASM ${profile.id}/${component.kind} dependency '${dependency.path}' ` +
            'does not match its build evidence.',
          );
        }
        records.push(`${component.kind}\0${object.index}\0dependency\0` +
          `${dependency.scope}\0${dependency.path}\0` +
          `${current.rawBytes}\0${current.sha256}\n`);
        dependencyCount += 1;
        if (dependency.scope === 'repository') repositoryDependencyCount += 1;
        else toolchainDependencyCount += 1;
      }
    }
  }
  return Object.freeze({
    objectCount,
    dependencyCount,
    repositoryDependencyCount,
    toolchainDependencyCount,
    toolchainInputsVerified: verifyToolchainInputs,
    objectGraphSha256: sha256(records.join('')),
  });
}

/**
 * Bind parent/child component hashes to one final release module. Object bytes
 * are checked separately because their deterministic build-root paths are not
 * part of canonical evidence.
 */
export function validateWasmBuildEvidenceAgainstArtifact(
  input,
  document,
  profileValue,
) {
  const profile = profileFrom(profileValue);
  const evidence = validateWasmBuildEvidence(document, profile);
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const customSections = orderedWasmCustomSections(bytes);
  const releaseOwnedNames = customSections
    .filter(({ name }) => name.startsWith('volvoxai.'))
    .map(({ name }) => name);
  if (stableJSON(releaseOwnedNames) !== stableJSON(profile.recipe.customSections)) {
    throw new Error(
      `WASM ${profile.id} release-owned custom sections differ from the recipe: ` +
      releaseOwnedNames.join(', '),
    );
  }
  const releaseSections = [];
  for (const name of profile.recipe.customSections) {
    const matches = customSections.filter((section) => section.name === name);
    if (matches.length !== 1) {
      throw new Error(
        `WASM ${profile.id} artifact contains ${matches.length} '${name}' custom sections; ` +
        'expected exactly one.',
      );
    }
    releaseSections.push(matches[0]);
  }
  for (let index = 1; index < releaseSections.length; index += 1) {
    if (releaseSections[index - 1].index >= releaseSections[index].index) {
      throw new Error(`WASM ${profile.id} release custom-section order is invalid.`);
    }
  }

  let parent = bytes;
  for (const name of profile.recipe.customSections) {
    parent = withoutWasmCustomSection(parent, name);
  }
  const components = new Map(evidence.components.map((component) =>
    [component.kind, component]));
  const parentEvidence = components.get('parent');
  const parentBytes = Buffer.from(parent);
  if (parentEvidence.artifact.rawBytes !== parentBytes.length ||
      parentEvidence.artifact.sha256 !== sha256(parentBytes)) {
    throw new Error(`WASM ${profile.id} parent payload does not match build evidence.`);
  }

  const childSections = new Map([
    ['relaxed', WASM_RELAXED_SIMD_SECTION],
    ['ptqAuthoring', WASM_PTQ_AUTHORING_SECTION],
  ]);
  const results = [{
    kind: 'parent',
    rawBytes: parentBytes.length,
    sha256: sha256(parentBytes),
  }];
  for (const [kind, sectionName] of childSections) {
    const component = components.get(kind);
    if (component === undefined) continue;
    const section = customSections.find((candidate) => candidate.name === sectionName);
    if (section === undefined || component.artifact.rawBytes !== section.payload.length ||
        component.artifact.sha256 !== sha256(section.payload)) {
      throw new Error(`WASM ${profile.id} ${kind} child does not match build evidence.`);
    }
    results.push({
      kind,
      rawBytes: section.payload.length,
      sha256: sha256(section.payload),
    });
  }
  return deepFreeze({
    profile: profile.id,
    artifact: profile.filename,
    components: results,
  });
}

/**
 * Validate one published WASM artifact, its canonical build evidence, retained
 * objects, and repository dependencies as one stable read-only snapshot. The
 * strict default additionally rehashes current tool executables and toolchain
 * dependencies; host-portable package/report callers may disable only that
 * build-image-dependent portion. The closing pass catches an overlapping
 * build or source edit instead of combining two generations.
 */
export async function validateWasmReleaseBuildSnapshot({
  repositoryRoot = defaultRepositoryRoot,
  artifact: artifactValue,
  evidence: evidenceValue,
  buildRoot: buildRootValue,
  profile: profileValue,
  packageVersion,
  verifyToolchainInputs = true,
} = {}) {
  const root = path.resolve(repositoryRoot);
  const profile = profileFrom(profileValue);
  const artifactPath = path.resolve(
    root,
    requireNonemptyString(artifactValue, 'WASM release artifact'),
  );
  const evidencePath = path.resolve(
    root,
    requireNonemptyString(evidenceValue, 'WASM build evidence'),
  );
  const buildRoot = buildRootValue === undefined
    ? wasmReleaseBuildRoot(root)
    : path.resolve(root, requireNonemptyString(buildRootValue, 'WASM build root'));
  requireNonemptyString(packageVersion, 'WASM package version');

  const readSnapshot = async () => {
    const [artifact, recorded] = await Promise.all([
      readArtifact(artifactPath, `WASM ${profile.id} release artifact`, { wasm: true }),
      readWasmBuildEvidenceRecord(evidencePath, profile),
    ]);
    const objectGraph = await validateWasmBuildEvidenceObjects(
      buildRoot,
      recorded.document,
      profile,
      { repositoryRoot: root, verifyToolchainInputs },
    );
    const componentBinding = validateWasmBuildEvidenceAgainstArtifact(
      artifact.contents,
      recorded.document,
      profile,
    );
    return Object.freeze({ artifact, recorded, objectGraph, componentBinding });
  };

  const before = await readSnapshot();
  const validatedArtifact = await validateWasmArtifact(
    root,
    artifactPath,
    packageVersion,
    { buildEvidenceSha256: before.recorded.sha256 },
  );
  const after = await readSnapshot();
  const snapshotIdentity = (snapshot) => stableJSON({
    artifact: {
      rawBytes: snapshot.artifact.bytes,
      sha256: snapshot.artifact.sha256,
    },
    evidence: {
      rawBytes: snapshot.recorded.rawBytes,
      sha256: snapshot.recorded.sha256,
    },
    objectGraph: snapshot.objectGraph,
    componentBinding: snapshot.componentBinding,
  });
  if (snapshotIdentity(before) !== snapshotIdentity(after) ||
      validatedArtifact.artifactSha256 !== after.artifact.sha256) {
    throw new Error(
      `WASM ${profile.id} artifact, evidence, object graph, or dependencies changed ` +
      'while the release snapshot was validated.',
    );
  }
  return Object.freeze({
    artifactBytes: after.artifact.contents,
    artifactSha256: after.artifact.sha256,
    artifactRawBytes: after.artifact.bytes,
    recorded: after.recorded,
    objectGraph: after.objectGraph,
    componentBinding: after.componentBinding,
    validatedArtifact,
  });
}

/**
 * Compile and link one central WASM release component.
 *
 * The exported API deliberately takes one options object so provenance replay
 * can redirect output/build paths without changing the authoritative recipe.
 */
export async function buildWasmComponent({
  repositoryRoot = defaultRepositoryRoot,
  profile: profileValue,
  kind,
  output: outputValue,
  buildRoot: buildRootValue,
  evidence: evidenceValue,
  publish = true,
} = {}) {
  if (typeof publish !== 'boolean') {
    throw new Error('WASM component publish option must be a boolean.');
  }
  const profile = profileFrom(profileValue);
  const selected = componentFor(profile, kind);
  const canonicalComponentKind = selected.kind;
  const component = validateComponent(
    profile,
    canonicalComponentKind,
    selected.component,
  );
  const toolchain = validateToolchain(profile);
  const root = path.resolve(repositoryRoot);
  const outputPath = path.resolve(root, requireNonemptyString(outputValue, 'WASM output'));
  const buildRoot = path.resolve(root, requireNonemptyString(buildRootValue, 'WASM build root'));
  const evidencePath = evidenceValue === undefined || evidenceValue === null
    ? null
    : path.resolve(root, requireNonemptyString(evidenceValue, 'WASM evidence path'));

  const [compiler, linker] = await Promise.all([
    exactTool(
      toolchain.compiler,
      toolchain.compilerInvocationPath,
      toolchain.compilerRealPath,
      toolchain.compilerVersion,
      toolchain.compilerSha256,
      toolchain.compilerRuntimeDependencies,
      root,
      'compiler',
    ),
    exactTool(
      toolchain.linker,
      toolchain.linkerInvocationPath,
      toolchain.linkerRealPath,
      toolchain.linkerVersion,
      toolchain.linkerSha256,
      toolchain.linkerRuntimeDependencies,
      root,
      'linker',
    ),
  ]);

  const componentDirectory = componentBuildDirectory(
    buildRoot,
    profile.id,
    canonicalComponentKind,
  );
  const objectsDirectory = path.join(componentDirectory, 'objects');
  await assertDistinctBuildPaths(
    [
      ['WASM component output', outputPath],
      ['WASM component evidence', evidencePath],
    ],
    path.join(buildRoot, 'objects'),
    [
      ...component.objects.map(({ source }) => path.resolve(root, source)),
      compiler.binding.invocationPath,
      compiler.binding.realPath,
      ...compiler.evidence.runtimeDependencies.map(({ path: inputPath }) => inputPath),
      linker.binding.invocationPath,
      linker.binding.realPath,
      ...linker.evidence.runtimeDependencies.map(({ path: inputPath }) => inputPath),
    ],
  );
  // Every expected object and depfile is overwritten below. Do not recursively
  // clear this predictable path: a hostile intermediate symlink must never
  // turn a release build into deletion outside its build root.
  await ensurePrivateBuildDirectory(buildRoot, objectsDirectory);

  const objectEvidence = [];
  const objectPaths = [];
  const assemblyEvidence = [];
  for (let index = 0; index < component.objects.length; index += 1) {
    const object = component.objects[index];
    const objectPath = wasmBuildObjectPath(
      buildRoot,
      profile,
      canonicalComponentKind,
      index,
      object.id,
    );
    const compiled = await compileObjectFromSnapshot({
      repositoryRoot: root,
      componentDirectory,
      objectsDirectory,
      compiler,
      profile,
      kind: canonicalComponentKind,
      object,
      index,
      objectPath,
      requiredInstructions: component.requiredInstructions,
    });
    objectEvidence.push(Object.freeze({
      index,
      id: object.id,
      source: object.source,
      category: object.category,
      optimization: object.optimization,
      flags: Object.freeze([...object.flags]),
      flagsSha256: sha256(stableJSON(object.flags)),
      dependencies: compiled.dependencies,
      rawBytes: compiled.artifact.bytes,
      sha256: compiled.artifact.sha256,
    }));
    assemblyEvidence.push(compiled.assembly);
    objectPaths.push(objectPath);
  }

  let instructionAudit = null;
  if (component.requiredInstructions.length > 0) {
    const occurrences = Object.fromEntries(
      component.requiredInstructions.map((instruction) => [instruction, 0]),
    );
    const assemblyRecords = [];
    for (let index = 0; index < component.objects.length; index += 1) {
      const assembly = assemblyEvidence[index];
      if (assembly === null) {
        throw new Error(
          `WASM ${profile.id}/${canonicalComponentKind} lacks compiler assembly evidence.`,
        );
      }
      assemblyRecords.push(`${component.objects[index].id}\0${assembly.sha256}\n`);
      for (const instruction of component.requiredInstructions) {
        occurrences[instruction] += assembly.occurrences[instruction];
      }
    }
    for (const instruction of component.requiredInstructions) {
      if (occurrences[instruction] === 0) {
        throw new Error(
          `WASM ${profile.id}/${canonicalComponentKind} compiler assembly lacks ` +
          `required instruction '${instruction}'.`,
        );
      }
    }
    instructionAudit = Object.freeze({
      requiredInstructions: Object.freeze([...component.requiredInstructions]),
      occurrences: Object.freeze(occurrences),
      assemblySha256: sha256(assemblyRecords.join('')),
    });
  }

  const linkedOutput = path.join(componentDirectory, 'linked.wasm');
  await assertSafePrivateOutputs(
    [linkedOutput],
    `WASM ${profile.id}/${canonicalComponentKind} linked output`,
  );
  const linkArguments = [...component.linkFlags, '-o', linkedOutput, ...objectPaths];
  await command(linker.executable, linkArguments, root);
  const linked = await readArtifact(
    linkedOutput,
    `WASM ${profile.id}/${canonicalComponentKind} linked artifact`,
    { wasm: true },
  );
  await validateWasmModuleStructure(linked.contents, {
    relaxedSimd: canonicalComponentKind === 'relaxed',
    label: `WASM ${profile.id}/${canonicalComponentKind} linked artifact`,
  });
  for (let index = 0; index < objectPaths.length; index += 1) {
    const actual = await readArtifact(
      objectPaths[index],
      `WASM ${profile.id}/${canonicalComponentKind} object ` +
        `'${objectEvidence[index].id}' after link`,
    );
    if (actual.bytes !== objectEvidence[index].rawBytes ||
        actual.sha256 !== objectEvidence[index].sha256) {
      throw new Error(
        `WASM ${profile.id}/${canonicalComponentKind} object ` +
        `'${objectEvidence[index].id}' changed while the component was linked.`,
      );
    }
    for (const dependency of objectEvidence[index].dependencies) {
      const current = await dependencyEvidence(
        root,
        dependency.path,
        `WASM ${profile.id}/${canonicalComponentKind} dependency '${dependency.path}'`,
      );
      if (stableJSON(current) !== stableJSON(dependency)) {
        throw new Error(
          `WASM ${profile.id}/${canonicalComponentKind} dependency ` +
          `'${dependency.path}' changed while the component was linked.`,
        );
      }
    }
  }
  await Promise.all([
    assertToolUnchanged(compiler, 'compiler'),
    assertToolUnchanged(linker, 'linker'),
  ]);
  const toolInputPaths = [
    compiler.binding.invocationPath,
    compiler.binding.realPath,
    ...compiler.evidence.runtimeDependencies.map(({ path: inputPath }) => inputPath),
    linker.binding.invocationPath,
    linker.binding.realPath,
    ...linker.evidence.runtimeDependencies.map(({ path: inputPath }) => inputPath),
  ];
  const componentBuildInputPaths = [
    ...objectEvidence.flatMap(({ dependencies }) => dependencies
      .map(({ scope, path: dependencyPath }) => scope === 'repository'
        ? path.resolve(root, dependencyPath)
        : path.resolve(dependencyPath))),
    ...toolInputPaths,
  ];
  await assertDistinctBuildPaths(
    [
      ['WASM component output', outputPath],
      ['WASM component evidence', evidencePath],
    ],
    path.join(buildRoot, 'objects'),
    componentBuildInputPaths,
  );
  const componentEvidence = Object.freeze({
    kind: canonicalComponentKind,
    instructionAudit,
    artifact: Object.freeze({
      rawBytes: linked.bytes,
      sha256: linked.sha256,
    }),
    orderedObjects: Object.freeze(objectEvidence),
    link: Object.freeze({
      flags: Object.freeze([...component.linkFlags]),
      flagsSha256: sha256(stableJSON(component.linkFlags)),
      orderedObjectIds: Object.freeze(objectEvidence.map(({ id }) => id)),
    }),
  });
  const toolchainEvidence = Object.freeze({
    compiler: compiler.evidence,
    linker: linker.evidence,
  });
  const evidence = validateWasmBuildEvidence(
    buildEvidenceDocument(profile, toolchainEvidence, [componentEvidence]),
    profile,
    { allowPartial: true },
  );
  const canonical = canonicalEvidenceArtifact(evidence);
  if (publish) {
    await publishFileTransaction([
      {
        label: `WASM ${profile.id}/${canonicalComponentKind} output`,
        destination: outputPath,
        contents: linked.contents,
        wasm: true,
      },
      {
        label: `WASM ${profile.id}/${canonicalComponentKind} evidence`,
        destination: evidencePath,
        contents: canonical.contents,
      },
    ], {
      protectedInputs: componentBuildInputPaths,
      precommit: async () => validateWasmBuildEvidenceObjects(
        buildRoot,
        evidence,
        profile,
        { repositoryRoot: root, allowPartial: true },
      ),
    });
  }
  return Object.freeze({
    profile,
    kind: canonicalComponentKind,
    outputPath,
    artifactContents: Buffer.from(linked.contents),
    component: componentEvidence,
    toolchain: toolchainEvidence,
    evidence,
    evidencePath,
    evidenceBytes: canonical.bytes,
    evidenceSha256: canonical.sha256,
    toolInputPaths: Object.freeze(toolInputPaths),
  });
}

function defaultChildOutput(buildRoot, section) {
  return path.join(buildRoot, `${section}.wasm`);
}

export function wasmBuildEvidencePath(evidenceDirectory, profileValue) {
  const profile = profileFrom(profileValue);
  return path.join(path.resolve(evidenceDirectory), `${profile.filename}.build.json`);
}

export function wasmReleaseBuildRoot(repositoryRoot = defaultRepositoryRoot) {
  return path.join(path.resolve(repositoryRoot), 'build', 'wasm', 'release');
}

export function wasmReleaseEvidenceDirectory(repositoryRoot = defaultRepositoryRoot) {
  return path.join(path.resolve(repositoryRoot), 'build', 'wasm', 'provenance');
}

export function wasmReleaseEvidencePath(repositoryRoot, profileValue) {
  return wasmBuildEvidencePath(
    wasmReleaseEvidenceDirectory(repositoryRoot),
    profileValue,
  );
}

/** Build every component of one inference or full WASM profile. */
export async function buildWasmProfile({
  repositoryRoot = defaultRepositoryRoot,
  profile: profileValue,
  outputDirectory: outputDirectoryValue,
  buildRoot: buildRootValue,
  evidenceDirectory: evidenceDirectoryValue,
  relaxedOutput: relaxedOutputValue,
  ptqOutput: ptqOutputValue,
  publish = true,
} = {}) {
  if (typeof publish !== 'boolean') {
    throw new Error('WASM profile publish option must be a boolean.');
  }
  const profile = profileFrom(profileValue);
  const root = path.resolve(repositoryRoot);
  const outputDirectory = path.resolve(
    root,
    requireNonemptyString(outputDirectoryValue, 'WASM output directory'),
  );
  const buildRoot = path.resolve(
    root,
    requireNonemptyString(buildRootValue, 'WASM build root'),
  );
  const evidenceDirectory = path.resolve(
    root,
    requireNonemptyString(evidenceDirectoryValue, 'WASM evidence directory'),
  );
  const relaxedOutput = relaxedOutputValue === undefined
    ? defaultChildOutput(buildRoot, WASM_RELAXED_SIMD_SECTION)
    : path.resolve(root, relaxedOutputValue);
  const ptqOutput = ptqOutputValue === undefined
    ? defaultChildOutput(buildRoot, WASM_PTQ_AUTHORING_SECTION)
    : path.resolve(root, ptqOutputValue);
  const outputs = Object.freeze({
    parent: path.join(outputDirectory, profile.filename),
    relaxed: relaxedOutput,
    ptqAuthoring: ptqOutput,
  });
  const aggregatePath = wasmBuildEvidencePath(evidenceDirectory, profile);
  await assertDistinctBuildPaths([
    ['WASM parent output', outputs.parent],
    ['WASM Relaxed-SIMD output', outputs.relaxed],
    ...(profile.recipe.ptqAuthoring === null
      ? []
      : [['WASM PTQ authoring output', outputs.ptqAuthoring]]),
    ['WASM aggregate evidence', aggregatePath],
  ], path.join(buildRoot, 'objects'));

  const components = [];
  for (const kind of COMPONENT_ORDER) {
    const property = COMPONENT_PROPERTIES[kind];
    if (profile.recipe?.[property] === null || profile.recipe?.[property] === undefined) continue;
    components.push(await buildWasmComponent({
      repositoryRoot: root,
      profile,
      kind,
      output: outputs[kind],
      buildRoot,
      publish: false,
    }));
  }

  const toolchains = new Set(components.map(({ toolchain }) => stableJSON(toolchain)));
  if (toolchains.size !== 1) {
    throw new Error(`WASM profile '${profile.id}' components used different toolchains.`);
  }
  const aggregate = validateWasmBuildEvidence(
    buildEvidenceDocument(
      profile,
      components[0].toolchain,
      components.map(({ component }) => component),
    ),
    profile,
  );
  // Earlier components may have been idle while later components compiled.
  // Rehash the complete retained object/dependency/tool graph immediately
  // before this profile is eligible for publication.
  await validateWasmBuildEvidenceObjects(buildRoot, aggregate, profile, {
    repositoryRoot: root,
  });
  const canonical = canonicalEvidenceArtifact(aggregate);
  // A destination that is harmless relative to one component may alias a
  // source consumed by a later component. Recheck the complete profile's
  // discovered repository dependency union immediately before publication.
  const publicationPathEntries = [
    ...components.map(({ kind, outputPath }) =>
      [`WASM ${profile.id}/${kind} output`, outputPath]),
    ['WASM aggregate evidence', aggregatePath],
  ];
  const buildInputPaths = components.flatMap(({ component }) =>
    component.orderedObjects.flatMap(({ dependencies }) => dependencies
      .map(({ scope, path: dependencyPath }) => scope === 'repository'
        ? path.resolve(root, dependencyPath)
        : path.resolve(dependencyPath))))
    .concat(components.flatMap(({ toolInputPaths }) => toolInputPaths));
  await assertDistinctBuildPaths(
    publicationPathEntries,
    path.join(buildRoot, 'objects'),
    buildInputPaths,
  );
  const publications = Object.freeze([
    ...components.map(({ kind, outputPath, artifactContents }) => ({
      label: `WASM ${profile.id}/${kind} output`,
      destination: outputPath,
      contents: artifactContents,
      wasm: true,
      ...(kind === 'relaxed' ? { sharedKey: 'relaxed-simd-child' } : {}),
    })),
    {
      label: `WASM ${profile.id} aggregate evidence`,
      destination: aggregatePath,
      contents: canonical.contents,
      commitMarker: true,
    },
  ]);
  if (publish) {
    await publishFileTransaction(publications, {
      protectedInputs: buildInputPaths,
      precommit: async () => validateWasmBuildEvidenceObjects(
        buildRoot,
        aggregate,
        profile,
        { repositoryRoot: root },
      ),
    });
  }
  return Object.freeze({
    profile,
    components: Object.freeze(components),
    evidence: aggregate,
    evidencePath: aggregatePath,
    evidenceBytes: canonical.bytes,
    evidenceSha256: canonical.sha256,
    publications,
    buildInputPaths: Object.freeze(buildInputPaths),
  });
}

/** Build and publish the complete inference/full set as one transaction. */
export async function buildWasmReleaseProfiles({
  repositoryRoot = defaultRepositoryRoot,
  profiles = RELEASE_PROFILES.wasm,
  outputDirectory,
  buildRoot,
  evidenceDirectory,
  relaxedOutput,
  ptqOutput,
} = {}) {
  if (!Array.isArray(profiles) || profiles.length === 0) {
    throw new Error('WASM release profile set must be a non-empty array.');
  }
  const root = path.resolve(repositoryRoot);
  const resolvedBuildRoot = path.resolve(
    root,
    requireNonemptyString(buildRoot, 'WASM build root'),
  );
  const results = [];
  for (const profile of profiles) {
    results.push(await buildWasmProfile({
      repositoryRoot: root,
      profile,
      outputDirectory,
      buildRoot: resolvedBuildRoot,
      evidenceDirectory,
      ...(relaxedOutput === undefined ? {} : { relaxedOutput }),
      ...(ptqOutput === undefined ? {} : { ptqOutput }),
      publish: false,
    }));
  }
  const relaxedHashes = new Set(results.flatMap(({ components }) => components
    .filter(({ kind }) => kind === 'relaxed')
    .map(({ component }) => component.artifact.sha256)));
  if (relaxedHashes.size > 1) {
    throw new Error('Inference/full relaxed-SIMD component recipes produced different bytes.');
  }
  const publications = await coalesceProfilePublications(
    results.flatMap(({ publications: profilePublications }) => profilePublications),
  );
  // Building the later profile can outlive the earlier profile's final local
  // check. Repeat every complete graph before the cross-profile commit.
  for (const result of results) {
    await validateWasmBuildEvidenceObjects(
      resolvedBuildRoot,
      result.evidence,
      result.profile,
      { repositoryRoot: root },
    );
  }
  const buildInputPaths = results.flatMap(({ buildInputPaths: profileInputs }) =>
    profileInputs);
  await assertDistinctBuildPaths(
    publications.map(({ label, destination }) => [label, destination]),
    path.join(resolvedBuildRoot, 'objects'),
    buildInputPaths,
  );
  await publishFileTransaction(publications, {
    protectedInputs: buildInputPaths,
    precommit: async () => {
      for (const result of results) {
        await validateWasmBuildEvidenceObjects(
          resolvedBuildRoot,
          result.evidence,
          result.profile,
          { repositoryRoot: root },
        );
      }
    },
  });
  return Object.freeze({
    profiles: Object.freeze(results),
    publications,
  });
}

function parseOptions(arguments_) {
  const options = new Map();
  for (let index = 0; index < arguments_.length; index += 2) {
    const name = arguments_[index];
    const value = arguments_[index + 1];
    if (!name?.startsWith('--') || value === undefined || value.startsWith('--')) {
      throw new Error(`Expected '--name value'; found '${name ?? ''} ${value ?? ''}'.`);
    }
    if (options.has(name)) throw new Error(`Duplicate option '${name}'.`);
    options.set(name, value);
  }
  return options;
}

function takeOption(options, name, { required = false } = {}) {
  const value = options.get(name);
  options.delete(name);
  if (required && value === undefined) throw new Error(`Missing required ${name} option.`);
  return value;
}

async function main(arguments_) {
  const [commandName, ...rest] = arguments_;
  const options = parseOptions(rest);
  if (commandName === 'component') {
    const result = await buildWasmComponent({
      profile: takeOption(options, '--profile', { required: true }),
      kind: takeOption(options, '--kind', { required: true }),
      output: takeOption(options, '--output', { required: true }),
      buildRoot: takeOption(options, '--build-root', { required: true }),
      evidence: takeOption(options, '--evidence'),
    });
    if (options.size > 0) throw new Error(`Unknown options: ${[...options.keys()].join(', ')}.`);
    console.log(
      `Built ${result.profile.id}/${result.kind} ` +
      `(${result.component.artifact.rawBytes} bytes, ${result.component.artifact.sha256}).`,
    );
    return;
  }
  if (commandName === 'all') {
    const outputDirectory = takeOption(options, '--output-dir', { required: true });
    const buildRoot = takeOption(options, '--build-root', { required: true });
    const evidenceDirectory = takeOption(options, '--evidence-dir', { required: true });
    const relaxedOutput = takeOption(options, '--relaxed-output');
    const ptqOutput = takeOption(options, '--ptq-output');
    if (options.size > 0) throw new Error(`Unknown options: ${[...options.keys()].join(', ')}.`);
    const result = await buildWasmReleaseProfiles({
      outputDirectory,
      buildRoot,
      evidenceDirectory,
      ...(relaxedOutput === undefined ? {} : { relaxedOutput }),
      ...(ptqOutput === undefined ? {} : { ptqOutput }),
    });
    console.log(
      `Built ${result.profiles.length} WASM profiles with ` +
      `${result.profiles.reduce((sum, profileResult) =>
        sum + profileResult.components.length, 0)} components.`,
    );
    return;
  }
  throw new Error(
    'Use `all --output-dir <dir> --build-root <dir> --evidence-dir <dir> ' +
    '[--relaxed-output <path>] [--ptq-output <path>]` or ' +
    '`component --profile <inference|full> --kind <parent|relaxed|ptq-authoring> ' +
    '--output <path> --build-root <dir> [--evidence <path>]`.',
  );
}

if (process.argv[1] !== undefined && path.resolve(process.argv[1]) === modulePath) {
  await main(process.argv.slice(2));
}
