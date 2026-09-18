#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PROFILES = Object.freeze([
  Object.freeze({ label: 'inference', target: 'volvoxai-lite' }),
  Object.freeze({ label: 'full', target: 'volvoxai' }),
]);

function nativeBuildDirectoryArgument() {
  let value = 'build/cmake';
  for (const argument of process.argv.slice(2)) {
    if (!argument.startsWith('--native-build-dir=')) {
      throw new Error(`unknown argument '${argument}'`);
    }
    value = argument.slice('--native-build-dir='.length);
    if (!value) throw new Error('--native-build-dir must not be empty');
  }
  return path.resolve(ROOT, value);
}

function tokenize(command, filename) {
  const tokens = (command.match(/[^\s"']+|"[^"]*"|'[^']*'/gu) || []).map(
    (token) => (/^(["']).*\1$/u.test(token) ? token.slice(1, -1) : token),
  );
  if (!tokens.length || tokens.some((token) => /["']/u.test(token))) {
    throw new Error(`unsupported generated command in '${filename}'`);
  }
  return tokens;
}

function checkedSpawn(command, args, options, label) {
  const child = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
    ...options,
  });
  if (child.error) throw child.error;
  if (child.status !== 0) {
    const output = child.stderr?.trim() || child.stdout?.trim() ||
      `exit ${child.status}`;
    throw new Error(`${label} failed:\n${output}`);
  }
  if (child.stdout) process.stdout.write(child.stdout);
  if (child.stderr) process.stderr.write(child.stderr);
  return child;
}

function graphDocument() {
  const shape = [1, 4, 2];
  return {
    format: 'volvox-graph/v1',
    dimensions: {},
    /* Declaration order intentionally differs from canonical name order. */
    inputs: {
      zeta: { shape, dtype: 'float32' },
      alpha: { shape, dtype: 'float32' },
    },
    nodes: [
      {
        id: 'zeta_identity',
        opType: 'Identity',
        inputs: { input: 'zeta' },
        outputs: {
          out: { tensor: 'zeta_path', shape, dtype: 'float32' },
        },
        params: {},
      },
      {
        id: 'alpha_identity',
        opType: 'Identity',
        inputs: { input: 'alpha' },
        outputs: {
          out: { tensor: 'alpha_path', shape, dtype: 'float32' },
        },
        params: {},
      },
      {
        id: 'merge',
        opType: 'Add',
        inputs: { a: 'zeta_path', b: 'alpha_path' },
        outputs: { out: { tensor: 'sum', shape, dtype: 'float32' } },
        params: {},
      },
    ],
    outputs: ['sum'],
  };
}

async function readProfile(buildDirectory, profile) {
  const nativeBuildDirectory = path.join(buildDirectory, 'native');
  const targetDirectory = path.join(
    nativeBuildDirectory, 'CMakeFiles', `${profile.target}.dir`,
  );
  const linkFile = path.join(targetDirectory, 'link.txt');
  let linkCommand;
  try {
    linkCommand = await readFile(linkFile, 'utf8');
  } catch (error) {
    throw new Error(
      `cannot read ${profile.label} CMake link metadata under ` +
      `'${targetDirectory}'; build both native profiles first: ${error.message}`,
    );
  }
  const tokens = tokenize(linkCommand.trim(), linkFile);
  const compiler = tokens.shift();
  const cliObjects = tokens.filter((token) => token.endsWith('/cli/main.c.o'));
  const cliTarget = /^(CMakeFiles\/[^/]+\.dir)\//u.exec(cliObjects[0] ?? '')?.[1];
  if (cliObjects.length !== 1 || !cliTarget) {
    throw new Error(
      `${profile.label} link command '${linkFile}' must contain exactly one ` +
      'CMake object-library CLI entry',
    );
  }
  const flagsFile = path.join(nativeBuildDirectory, cliTarget, 'flags.make');
  let flags;
  try {
    flags = await readFile(flagsFile, 'utf8');
  } catch (error) {
    throw new Error(
      `cannot read ${profile.label} release CLI flags '${flagsFile}': ${error.message}`,
    );
  }
  const definitionsLine = /^C_DEFINES\s*=\s*(.*)$/mu.exec(flags)?.[1] || '';
  const definitions = tokenize(definitionsLine, flagsFile).filter(
    (token) => /^-DVOLVOXAI_ENABLE_[A-Z0-9_]+=.+$/u.test(token),
  );
  const expectedTraining =
    `-DVOLVOXAI_ENABLE_TRAINING=${profile.label === 'full' ? 1 : 0}`;
  const trainingDefinitions = definitions.filter(
    (token) => token.startsWith('-DVOLVOXAI_ENABLE_TRAINING='),
  );
  if (!compiler || trainingDefinitions.length !== 1 ||
      trainingDefinitions[0] !== expectedTraining) {
    throw new Error(
      `incomplete or mismatched ${profile.label} metadata in ` +
      `'${flagsFile}', expected ${expectedTraining}`,
    );
  }
  return {
    ...profile,
    compiler,
    definitions,
    linkTokens: tokens,
    nativeBuildDirectory,
    linkFile,
  };
}

function linkStandaloneFixture(profile, object, executable) {
  checkedSpawn(
    profile.compiler,
    [object, '-o', executable],
    { cwd: ROOT },
    `${profile.label} ${path.basename(executable)} standalone link`,
  );
}

function compileFixture(profile, source, object, privateHeaders) {
  const includes = [
    path.join(ROOT, 'native', 'include'),
    path.join(ROOT, 'native', 'src'),
    path.join(ROOT, 'native', 'src', 'runtime'),
  ];
  if (privateHeaders) {
    includes.push(
      path.join(ROOT, 'native', 'src', 'kernels'),
      path.join(ROOT, 'native', 'src', 'backends'),
      path.join(ROOT, 'native', 'third_party'),
    );
  }
  const argumentsList = [
    '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
    ...profile.definitions,
  ];
  for (const include of includes) argumentsList.push('-I', include);
  argumentsList.push('-c', source, '-o', object);
  checkedSpawn(
    profile.compiler,
    argumentsList,
    { cwd: ROOT },
    `${profile.label} ${path.basename(source)} compilation`,
  );
}

function linkFixture(profile, object, executable, wrapValidator) {
  // CMake's release command carries a fixed -Map output used by size
  // provenance. Reusing that token for this temporary relink would overwrite
  // the release map with evidence for the fixture executable.
  const releaseMapTokens = profile.linkTokens.filter((token) =>
    /^-Wl,(?:-Map,|-Map=|--Map,|--Map=)/u.test(token));
  if (releaseMapTokens.length !== 1) {
    throw new Error(
      `${profile.label} link command '${profile.linkFile}' must contain exactly ` +
      'one removable release map output',
    );
  }
  const tokens = profile.linkTokens.filter((token) =>
    !releaseMapTokens.includes(token));
  const mainObjectIndex = tokens.findIndex(
    (token) => token.endsWith('/cli/main.c.o'),
  );
  const outputIndex = tokens.indexOf('-o');
  if (mainObjectIndex < 0 || outputIndex < 0 ||
      outputIndex + 1 >= tokens.length) {
    throw new Error(
      `${profile.label} link command '${profile.linkFile}' has no replaceable CLI entry`,
    );
  }
  tokens.splice(mainObjectIndex, 1, object);
  tokens[tokens.indexOf('-o') + 1] = executable;
  if (wrapValidator) {
    tokens.push(
      '-Wl,--wrap=vx_call_sequence_policy_validate_response_v1',
    );
  }
  checkedSpawn(
    profile.compiler,
    tokens,
    { cwd: profile.nativeBuildDirectory },
    `${profile.label} ${path.basename(executable)} relink`,
  );
}

async function main() {
  if (process.platform !== 'linux') {
    throw new Error(
      'native CPU selection relink tests require the pinned Linux CMake ' +
      'toolchain because validator interception uses the ELF --wrap option',
    );
  }
  const buildDirectory = nativeBuildDirectoryArgument();
  const temporaryDirectory = await mkdtemp(
    path.join(tmpdir(), 'volvox-native-cpu-selection-'),
  );
  try {
    const graphPath = path.join(temporaryDirectory, 'selection.graph.json');
    await writeFile(
      graphPath,
      `${JSON.stringify(graphDocument(), null, 2)}\n`,
      'utf8',
    );
    const profiles = await Promise.all(
      PROFILES.map((profile) => readProfile(buildDirectory, profile)),
    );
    const resourceSource = path.join(
      ROOT, 'native', 'tests', 'call_sequence_resource_accounting_test.c',
    );
    const resourceObject = path.join(
      temporaryDirectory, 'call-sequence-resource-accounting.o',
    );
    const resourceExecutable = path.join(
      temporaryDirectory, 'call-sequence-resource-accounting',
    );
    compileFixture(profiles[0], resourceSource, resourceObject, false);
    linkStandaloneFixture(profiles[0], resourceObject, resourceExecutable);
    checkedSpawn(
      resourceExecutable,
      [],
      { cwd: ROOT },
      'call-sequence resource accounting',
    );
    for (const profile of profiles) {

      const publicSource = path.join(
        ROOT, 'native', 'tests', 'native_cpu_call_sequence_selection_test.c',
      );
      const publicObject = path.join(
        temporaryDirectory, `${profile.target}-selection.o`,
      );
      const publicExecutable = path.join(
        temporaryDirectory, `${profile.target}-selection`,
      );
      compileFixture(profile, publicSource, publicObject, false);
      linkFixture(profile, publicObject, publicExecutable, true);
      checkedSpawn(
        publicExecutable,
        [graphPath, profile.label],
        { cwd: ROOT },
        `${profile.label} public portable-C selection integration`,
      );

      const ownerSource = path.join(
        ROOT, 'native', 'tests', 'native_cpu_selection_owner_projection_test.c',
      );
      const ownerObject = path.join(
        temporaryDirectory, `${profile.target}-owner-projection.o`,
      );
      const ownerExecutable = path.join(
        temporaryDirectory, `${profile.target}-owner-projection`,
      );
      compileFixture(profile, ownerSource, ownerObject, true);
      linkFixture(profile, ownerObject, ownerExecutable, false);
      checkedSpawn(
        ownerExecutable,
        [profile.label],
        { cwd: ROOT },
        `${profile.label} physical owner projection integration`,
      );
    }
    process.stdout.write(
      'Native portable-C inference-plan and CPU selection passed in inference and full profiles.\n',
    );
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}

await main();
