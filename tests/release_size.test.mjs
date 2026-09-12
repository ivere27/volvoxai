import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  NATIVE_RELEASE_FINALIZATION_FORMAT,
  RELEASE_SIZE_BUDGET_FORMAT,
  aggregateBrowserMetafile,
  describeWasmExportOwnership,
  deterministicGzip,
  evaluateSizeBudgets,
  formatBudgetFailure,
  inspectWasmBytes,
  parseLlvmNmSymbols,
  parseLlvmReadobjBuildId,
  parseLlvmReadobjSections,
  parseNativeLinkMapObjects,
  regressionAllowance,
  renderReleaseSizeMarkdown,
  validateNativeDebugSections,
  validateNativeLinkMapTarget,
  validateNativeReleaseFinalization,
  validateReleaseSizeBudgetInventory,
  validateStrippedNativeSections} from '../tools/release_size.mjs';
import {
  WASM_BUILD_EVIDENCE_FORMAT,
  WASM_PROVENANCE_SECTION,
  WASM_RELAXED_SIMD_SECTION,
  createWasmProvenance,
  inspectWasmInterface,
  parseCanonicalWasmProvenance,
  sha256,
  stableJSON,
  wasmProfile,
  withoutWasmCustomSection} from '../tools/release_profiles.mjs';
import {
  buildWasmComponent,
  buildWasmProfile,
  buildWasmReleaseProfiles,
  readWasmBuildEvidence,
  validateWasmBuildEvidence,
  validateWasmBuildEvidenceAgainstArtifact,
  validateWasmBuildEvidenceObjects,
  validateWasmBuildEvidenceTools,
  wasmBuildObjectPath} from '../tools/build_wasm_release.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('release provenance refuses retained debug names and DWARF sections',async()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'volvoxai-debug-section-'));
  const artifact=path.join(directory,'volvoxai.wasm');
  const {version}=JSON.parse(fs.readFileSync(path.join(ROOT,'package.json'),'utf8'));
  try {
    for(const name of ['name','.debug_info']) {
      fs.writeFileSync(artifact,wasmWithCustomSections([[name,new Uint8Array()]]));
      await assert.rejects(createWasmProvenance(ROOT,artifact,version,{
        buildEvidenceSha256:'0'.repeat(64),
      }),/retains debug sections/);
    }
  } finally {
    fs.rmSync(directory,{recursive:true,force:true});
  }
});

function unsignedLeb128(value) {
  const bytes = [];
  do {
    const remainder = value & 0x7f;
    value >>>= 7;
    bytes.push(remainder | (value === 0 ? 0 : 0x80));
  } while (value !== 0);
  return bytes;
}

function wasmString(value) {
  const bytes = [...new TextEncoder().encode(value)];
  return [...unsignedLeb128(bytes.length), ...bytes];
}

function wasmSection(id, payload) {
  return [id, ...unsignedLeb128(payload.length), ...payload];
}

function syntheticWasm() {
  const type = wasmSection(1, [1, 0x60, 0, 1, 0x7f]);
  const importedFunction = wasmSection(2, [
    1,
    ...wasmString('env'),
    ...wasmString('host'),
    0,
    0,
  ]);
  const functions = wasmSection(3, [1, 0]);
  const memory = wasmSection(5, [1, 0, 1]);
  const exports = wasmSection(7, [
    2,
    ...wasmString('answer'), 0, 1,
    ...wasmString('memory'), 2, 0,
  ]);
  const body = [0, 0x41, 7, 0x0b];
  const code = wasmSection(10, [1, ...unsignedLeb128(body.length), ...body]);
  const custom = wasmSection(0, [...wasmString('evidence'), 1, 2, 3]);
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...type,
    ...importedFunction,
    ...functions,
    ...memory,
    ...exports,
    ...code,
    ...custom,
  ]);
}

function syntheticImportedMemoryWasm() {
  const type = wasmSection(1, [1, 0x60, 0, 0]);
  const importedMemory = wasmSection(2, [
    1,
    ...wasmString('env'),
    ...wasmString('memory'),
    2,
    0,
    1,
  ]);
  const functions = wasmSection(3, [1, 0]);
  const exports = wasmSection(7, [
    1,
    ...wasmString('qlinear_i8u8_relaxed'), 0, 0,
  ]);
  const body = [0, 0x0b];
  const code = wasmSection(10, [1, ...unsignedLeb128(body.length), ...body]);
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...type,
    ...importedMemory,
    ...functions,
    ...exports,
    ...code,
  ]);
}

function wasmWithCustomSections(sections) {
  const customSections = sections.flatMap(([sectionName, contents]) =>
    wasmSection(0, [
      ...wasmString(sectionName),
      ...contents,
    ]));
  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ...customSections,
  ]);
}

function wasmEvidenceComponent(kind, recipe, artifactBytes) {
  const requiredInstructions = [...(recipe.requiredInstructions ?? [])];
  return {
    artifact: {
      rawBytes: artifactBytes.byteLength,
      sha256: sha256(artifactBytes),
    },
    instructionAudit: requiredInstructions.length === 0 ? null : {
      requiredInstructions,
      occurrences: Object.fromEntries(requiredInstructions.map((instruction) => [instruction, 1])),
      assemblySha256: sha256(`${kind}/fixture-assembly`),
    },
    kind,
    link: {
      flags: [...recipe.linkFlags],
      flagsSha256: sha256(stableJSON(recipe.linkFlags)),
      orderedObjectIds: recipe.objects.map(({ id }) => id),
    },
    orderedObjects: recipe.objects.map((object, index) => ({
      category: object.category,
      dependencies: [{
        scope: 'repository',
        path: object.source,
        rawBytes: object.source.length,
        sha256: sha256(`dependency/${object.source}`),
      }],
      flags: [...object.flags],
      flagsSha256: sha256(stableJSON(object.flags)),
      id: object.id,
      index,
      optimization: object.optimization,
      rawBytes: index + 1,
      sha256: sha256(`${kind}/${object.id}/fixture-object`),
      source: object.source,
    })),
  };
}

function inferenceWasmBuildEvidenceFixture() {
  const base = wasmProfile('inference');
  // A synthetic two-component profile tests generic evidence/publication code.
  // Shipping profiles have only the C parent.
  const profile = { ...base, recipe: { ...base.recipe, relaxed: {
    objects: [{ id: 'relaxed', source: 'native/src/kernels/qlinear_w8a8_wasm_relaxed.c',
      category: 'numerical-kernel', optimization: '-O3', flags: ['--target=wasm32', '-msimd128', '-mrelaxed-simd', '-ffreestanding'] }],
    linkFlags: ['-m', 'wasm32', '--no-entry', '--import-memory'],
    requiredInstructions: ['i32x4.relaxed_dot_i8x16_i7x16_add_s'],
  }, customSections: [WASM_RELAXED_SIMD_SECTION, WASM_PROVENANCE_SECTION] } };
  const relaxedBytes = new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  ]);
  const artifactBytes = wasmWithCustomSections([
    [WASM_RELAXED_SIMD_SECTION, relaxedBytes],
    [WASM_PROVENANCE_SECTION, new TextEncoder().encode('{}\n')],
  ]);
  let parentBytes = artifactBytes;
  for (const sectionName of profile.recipe.customSections) {
    parentBytes = withoutWasmCustomSection(parentBytes, sectionName);
  }
  const evidence = {
    artifact: profile.filename,
    components: [
      wasmEvidenceComponent('parent', profile.recipe.parent, parentBytes),
      wasmEvidenceComponent('relaxed', profile.recipe.relaxed, relaxedBytes),
    ],
    format: WASM_BUILD_EVIDENCE_FORMAT,
    profile: profile.id,
    recipeSha256: sha256(stableJSON(profile.recipe)),
    toolchain: {
      compiler: {
        command: profile.recipe.toolchain.compiler,
        path: profile.recipe.toolchain.compilerInvocationPath,
        realPath: profile.recipe.toolchain.compilerRealPath,
        rawBytes: 1,
        sha256: profile.recipe.toolchain.compilerSha256,
        version: profile.recipe.toolchain.compilerVersion,
        runtimeDependencies: structuredClone(
          profile.recipe.toolchain.compilerRuntimeDependencies,
        ),
      },
      linker: {
        command: profile.recipe.toolchain.linker,
        path: profile.recipe.toolchain.linkerInvocationPath,
        realPath: profile.recipe.toolchain.linkerRealPath,
        rawBytes: 1,
        sha256: profile.recipe.toolchain.linkerSha256,
        version: profile.recipe.toolchain.linkerVersion,
        runtimeDependencies: structuredClone(
          profile.recipe.toolchain.linkerRuntimeDependencies,
        ),
      },
    },
  };
  return { artifactBytes, evidence, profile, relaxedBytes };
}

function budgetFixture(rawActual = 108_192) {
  const report = {
    scope: 'full',
    artifacts: [{
      id: 'dist/<package-version>/fixture.js',
      rawBytes: rawActual,
      gzipBytes: 50_000,
    }],
    browserProfiles: [{
      id: 'fixture',
      pair: { rawBytes: 200_000, gzipBytes: 80_000 },
    }],
  };
  const budgets = {
    format: RELEASE_SIZE_BUDGET_FORMAT,
    policy: { relativeIncrease: 0.005, absoluteIncreaseBytes: 8192 },
    baselines: {
      artifacts: {
        'dist/<package-version>/fixture.js': {
          rawBytes: 100_000,
          gzipBytes: 50_000,
        },
      },
      browserPairs: {
        fixture: { rawBytes: 200_000, gzipBytes: 80_000 },
      },
    },
    plannedTargets: {
      'fixture/planned': {
        artifact: 'dist/<package-version>/fixture.js',
        limits: { rawBytes: 90_000 },
        source: 'tests/release_size.test.mjs',
      },
      'extension/inference-only': {
        status: 'deferred',
        limits: { rawBytes: 1_450_000 },
        reason: 'Separate extension packaging is deferred.',
      },
    },
  };
  return { report, budgets };
}

function nativeFinalizationFixture() {
  const buildId = 'ab'.repeat(20);
  const profile = {
    filename: 'native/volvoxai',
    debugArtifact: 'native/.debug/volvoxai.debug',
    debugEvidence: 'native/.debug/volvoxai.debug.json',
  };
  const releaseArtifact = {
    path: profile.filename,
    rawBytes: 101,
    sha256: '1'.repeat(64),
  };
  const debugArtifact = {
    path: profile.debugArtifact,
    rawBytes: 202,
    sha256: '2'.repeat(64),
  };
  const evidenceArtifact = {
    path: profile.debugEvidence,
    rawBytes: 303,
    sha256: '3'.repeat(64),
  };
  const buildDirectory = 'build/cmake';
  const buildArtifact = {
    ...releaseArtifact,
    path: `${buildDirectory}/native/release-link/volvoxai`,
  };
  const buildDebugArtifact = {
    ...debugArtifact,
    path: `${buildDirectory}/native/release-link/.debug/volvoxai.debug`,
  };
  const buildEvidenceArtifact = {
    ...evidenceArtifact,
    path: `${buildDirectory}/native/release-link/.debug/volvoxai.debug.json`,
  };
  const linkMap = {
    path: `${buildDirectory}/size-maps/volvoxai.map`,
    rawBytes: 404,
    mapSha256: '5'.repeat(64),
  };
  const snapshot = {
    buildId,
    loader: { interpreter: '/lib64/ld-linux-x86-64.so.2', needed: ['libc.so.6'] },
    loadSegments: [{ flags: 'R E', fileSize: 101 }],
    loadableSectionsSha256: '4'.repeat(64),
    version: 'volvoxai 0.4.0',
  };
  const evidence = {
    format: NATIVE_RELEASE_FINALIZATION_FORMAT,
    artifact: profile.filename,
    debugArtifact: profile.debugArtifact,
    build: {
      directory: buildDirectory,
      artifact: {
        path: buildArtifact.path,
        rawBytes: buildArtifact.rawBytes,
        sha256: buildArtifact.sha256,
      },
      linkMap: {
        path: linkMap.path,
        rawBytes: linkMap.rawBytes,
        sha256: linkMap.mapSha256,
      },
    },
    before: structuredClone(snapshot),
    after: structuredClone(snapshot),
    release: {
      rawBytes: releaseArtifact.rawBytes,
      sha256: releaseArtifact.sha256,
      hasGnuDebuglink: true,
      definedDynamicExports: [],
    },
    debug: {
      rawBytes: debugArtifact.rawBytes,
      sha256: debugArtifact.sha256,
      buildId,
    },
    symbolization: {
      symbol: 'main',
      address: '0x1000',
      release: '??:0',
      debug: 'main at cli/main.c:1',
    },
  };
  return {
    profile,
    releaseArtifact,
    debugArtifact,
    evidenceArtifact,
    buildArtifact,
    buildDebugArtifact,
    buildEvidenceArtifact,
    buildDirectory,
    linkMap,
    evidence,
    releaseBuildId: buildId,
    debugBuildId: buildId,
  };
}

test('deterministic gzip repeats byte-for-byte without wall-clock metadata', () => {
  const input = Buffer.from('volvoxai-size-evidence\n'.repeat(200));
  const first = deterministicGzip(input);
  const second = deterministicGzip(input);
  assert.deepEqual(first, second);
  assert.deepEqual([...first.subarray(4, 8)], [0, 0, 0, 0]);
});

test('synthetic WebAssembly reports standard, import/export, and custom sections', () => {
  const result = inspectWasmBytes(syntheticWasm());
  assert.equal(result.importCount, 1);
  assert.deepEqual(result.imports, ['env.host:function']);
  assert.equal(result.exportCount, 2);
  assert.deepEqual(result.exports, ['answer:function', 'memory:memory']);
  assert.ok(result.sectionBytes.code > 0);
  assert.deepEqual(result.customSections, [{
    index: 6,
    name: 'evidence',
    contentBytes: 3,
    payloadBytes: 12,
    totalBytes: 14,
    contentSha256: sha256(new Uint8Array([1, 2, 3])),
  }]);
  const ownership = describeWasmExportOwnership(result, {
    authority: {
      kind: 'generated-abi-manifest',
      source: 'tools/generated/example.mjs',
      manifestSha256: 'abc123',
    },
    exports: ['memory', 'missing'],
    exportKinds: { memory: 'memory', missing: 'function' },
  });
  assert.deepEqual(ownership.missingRequired, ['missing:function']);
  assert.deepEqual(ownership.unmanaged.entries, ['answer:function']);
  assert.equal(ownership.linked.setSha256, result.exportSetSha256);
  assert.equal(ownership.authority.kind, 'generated-abi-manifest');
});

test('feature-independent WASM interface inspection reads synthetic imports and exports', () => {
  assert.deepEqual(inspectWasmInterface(syntheticWasm(), 'synthetic fixture'), {
    imports: ['env.host:function'],
    exports: ['answer:function', 'memory:memory'],
  });
  const relaxedFixture = syntheticImportedMemoryWasm();
  assert.equal(WebAssembly.validate(relaxedFixture), true);
  assert.deepEqual(inspectWasmInterface(relaxedFixture, 'relaxed fixture'), {
    imports: ['env.memory:memory'],
    exports: ['qlinear_i8u8_relaxed:function'],
  });
});

test('embedded WASM provenance accepts only canonical stable JSON with one newline', async (t) => {
  const provenance = {
    artifact: 'fixture.wasm',
    buildEvidenceSha256: '1'.repeat(64),
    contractSha256: '2'.repeat(64),
    format: 'volvoxai-wasm-provenance/v1',
    packageVersion: '0.4.0',
    payloadSha256: '3'.repeat(64),
    profile: 'inference',
    recipeSha256: '4'.repeat(64),
    sourceFileCount: 1,
    sourceSha256: '5'.repeat(64),
  };
  const canonical = Buffer.from(`${stableJSON(provenance)}\n`);
  assert.deepEqual(
    parseCanonicalWasmProvenance(canonical, 'fixture.wasm'),
    provenance,
  );

  const mutations = new Map([
    ['reordered', Buffer.from(`${JSON.stringify(
      Object.fromEntries(Object.entries(provenance).reverse()),
    )}\n`)],
    ['pretty', Buffer.from(`${JSON.stringify(provenance, null, 2)}\n`)],
    ['no-newline', Buffer.from(stableJSON(provenance))],
    [
      'duplicate-key',
      Buffer.from(`{"artifact":"shadow",${stableJSON(provenance).slice(1)}\n`),
    ],
  ]);
  for (const [name, contents] of mutations) {
    await t.test(`rejects ${name}`, () => {
      assert.throws(
        () => parseCanonicalWasmProvenance(contents, 'fixture.wasm'),
        /release provenance is not canonical JSON/u,
      );
    });
  }
});

test('WASM build evidence binds the object recipe and linked component bytes', async (t) => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'volvoxai-wasm-evidence-'));
  try {
    const { artifactBytes, evidence, profile, relaxedBytes } =
      inferenceWasmBuildEvidenceFixture();
    const artifactPath = path.join(temporaryRoot, profile.filename);
    const validEvidencePath = path.join(temporaryRoot, `${profile.filename}.build.json`);
    const canonicalEvidence = Buffer.from(`${stableJSON(evidence)}\n`);
    fs.writeFileSync(artifactPath, artifactBytes);
    fs.writeFileSync(validEvidencePath, canonicalEvidence);

    assert.doesNotThrow(() => new WebAssembly.Module(artifactBytes));
    assert.doesNotThrow(() => new WebAssembly.Module(relaxedBytes));
    const validated = validateWasmBuildEvidence(evidence, profile);
    assert.deepEqual(validated, evidence);
    assert.deepEqual(
      await readWasmBuildEvidence(validEvidencePath, profile),
      evidence,
    );
    assert.equal(fs.readFileSync(validEvidencePath, 'utf8'), `${stableJSON(evidence)}\n`);
    const binding = validateWasmBuildEvidenceAgainstArtifact(
      artifactBytes,
      evidence,
      profile,
    );
    assert.deepEqual(binding.components.map(({ kind }) => kind), ['parent', 'relaxed']);
    assert.deepEqual(
      binding.components.map(({ sha256: digest }) => digest),
      evidence.components.map(({ artifact }) => artifact.sha256),
    );

    function mutatedEvidence(mutate) {
      const mutated = structuredClone(evidence);
      mutate(mutated);
      return mutated;
    }

    await t.test('rejects an object optimization mutation', () => {
      const mutated = mutatedEvidence((document) => {
        document.components[0].orderedObjects[0].optimization = '-Oz';
      });
      assert.throws(
        () => validateWasmBuildEvidence(mutated, profile),
        /object 0 optimization differs from the recipe/u,
      );
    });

    await t.test('rejects an object flags digest mutation', () => {
      const mutated = mutatedEvidence((document) => {
        document.components[0].orderedObjects[0].flagsSha256 = '0'.repeat(64);
      });
      assert.throws(
        () => validateWasmBuildEvidence(mutated, profile),
        /object 0 flags\/order are invalid/u,
      );
    });

    await t.test('rejects a linked parent digest mutation', () => {
      const mutated = mutatedEvidence((document) => {
        document.components[0].artifact.sha256 = '0'.repeat(64);
      });
      assert.throws(
        () => validateWasmBuildEvidenceAgainstArtifact(artifactBytes, mutated, profile),
        /parent payload does not match build evidence/u,
      );
    });

    await t.test('rejects an embedded child digest mutation', () => {
      const mutated = mutatedEvidence((document) => {
        document.components[1].artifact.sha256 = '0'.repeat(64);
      });
      assert.throws(
        () => validateWasmBuildEvidenceAgainstArtifact(artifactBytes, mutated, profile),
        /relaxed child does not match build evidence/u,
      );
    });

    await t.test('rejects missing Relaxed-SIMD instruction evidence', () => {
      const mutated = mutatedEvidence((document) => {
        const audit = document.components[1].instructionAudit;
        audit.occurrences[audit.requiredInstructions[0]] = 0;
      });
      assert.throws(
        () => validateWasmBuildEvidence(mutated, profile),
        /does not prove instruction/u,
      );
    });

    await t.test('rejects inference WebGPU, shader and training/PTQ dependencies', () => {
      for (const dependencyPath of [
        'native/src/backends/webgpu_backend.c',
        'native/src/backends/gpu_bridge.h',
        'native/src/backends/shader_catalog_inference.h',
        'native/src/backends/shader_catalog.h',
        'native/src/kernels/training_kernels.h',
        'native/src/backends/cuda_training_kernels.cu',
        'native/src/training/ptq_authoring.h',
        'native/src/api/vx_api_quantization.c',
        'runtime/generated/c/volvoxai_ffi.h',
      ]) {
        const mutated = mutatedEvidence((document) => {
          document.components[0].orderedObjects[0].dependencies.push({
            scope: 'repository',
            path: dependencyPath,
            rawBytes: 1,
            sha256: '1'.repeat(64),
          });
          document.components[0].orderedObjects[0].dependencies.sort((left, right) =>
            `${left.scope}\0${left.path}`.localeCompare(`${right.scope}\0${right.path}`, 'en'));
        });
        assert.throws(
          () => validateWasmBuildEvidence(mutated, profile),
          /depends on forbidden profile source/u,
          dependencyPath,
        );
      }
    });

    await t.test('rejects compiler and linker version mutations', () => {
      for (const kind of ['compiler', 'linker']) {
        const mutated = mutatedEvidence((document) => {
          document.toolchain[kind].version += ' mutated';
        });
        assert.throws(
          () => validateWasmBuildEvidence(mutated, profile),
          new RegExp(`${kind} differs from the release toolchain`, 'u'),
        );
      }
    });

    await t.test('rejects a compiler runtime-library mutation', () => {
      const mutated = mutatedEvidence((document) => {
        document.toolchain.compiler.runtimeDependencies[0].sha256 = '0'.repeat(64);
      });
      assert.throws(
        () => validateWasmBuildEvidence(mutated, profile),
        /runtime dependencies differ from the release toolchain/u,
      );
    });

    await t.test('rejects non-canonical key order and a missing final newline', async () => {
      const reversed = Object.fromEntries(Object.entries(evidence).reverse());
      const reorderedPath = path.join(temporaryRoot, 'reordered.json');
      fs.writeFileSync(reorderedPath, `${JSON.stringify(reversed)}\n`);
      await assert.rejects(
        readWasmBuildEvidence(reorderedPath, profile),
        /not canonical JSON/u,
      );

      const noNewlinePath = path.join(temporaryRoot, 'no-newline.json');
      fs.writeFileSync(noNewlinePath, stableJSON(evidence));
      await assert.rejects(
        readWasmBuildEvidence(noNewlinePath, profile),
        /not canonical JSON/u,
      );
    });
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
});

test('WASM object and tool digest validators rehash their on-disk inputs', {
  skip: process.platform === 'win32',
}, async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'volvoxai-wasm-rehash-'));
  const resourceRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'volvoxai-toolchain-resource-'));
  const previousPath = process.env.PATH;
  try {
    const toolsDirectory = path.join(temporaryRoot, 'tools');
    const source = 'src/fixture.c';
    const sourcePath = path.join(temporaryRoot, source);
    const resourceHeader = path.join(resourceRoot, 'fixture.h');
    const compilerVersion = 'fixture clang version 1.0';
    const linkerVersion = 'fixture LLD version 1.0';
    const toolSpecifications = [
      ['fixture-clang', compilerVersion],
      ['fixture-wasm-ld', linkerVersion],
    ];
    fs.mkdirSync(toolsDirectory, { recursive: true });
    fs.mkdirSync(path.dirname(sourcePath), { recursive: true });
    fs.mkdirSync(resourceRoot, { recursive: true });
    fs.writeFileSync(sourcePath, 'int fixture(void) { return 1; }\n');
    fs.writeFileSync(resourceHeader, '#define FIXTURE_RESOURCE 1\n');
    for (const [command, version] of toolSpecifications.slice(0, 1)) {
      const executable = path.join(toolsDirectory, command);
      fs.writeFileSync(
        executable,
        `#!/bin/sh\nprintf '%s\\n' ${JSON.stringify(version)}\n`,
      );
      fs.chmodSync(executable, 0o755);
    }
    const linkerDriver = path.join(toolsDirectory, 'fixture-lld-driver');
    fs.writeFileSync(
      linkerDriver,
      '#!/bin/sh\ncase "$0" in\n' +
        `  */fixture-wasm-ld) printf '%s\\n' ${JSON.stringify(linkerVersion)} ;;\n` +
        "  *) printf '%s\\n' 'generic linker driver' ;;\n" +
        'esac\n',
    );
    fs.chmodSync(linkerDriver, 0o755);
    fs.symlinkSync(path.basename(linkerDriver), path.join(toolsDirectory, 'fixture-wasm-ld'));
    process.env.PATH = `${toolsDirectory}${path.delimiter}${previousPath ?? ''}`;

    const profile = {
      id: 'fixture',
      filename: 'fixture.wasm',
      forbiddenDependencyFiles: [],
      forbiddenDependencyPrefixes: [],
      recipe: {
        toolchain: {
          compiler: toolSpecifications[0][0],
          compilerInvocationPath: path.join(toolsDirectory, toolSpecifications[0][0])
            .split(path.sep).join('/'),
          compilerRealPath: fs.realpathSync(
            path.join(toolsDirectory, toolSpecifications[0][0]),
          ).split(path.sep).join('/'),
          compilerVersion,
          compilerSha256: sha256(fs.readFileSync(fs.realpathSync(
            path.join(toolsDirectory, toolSpecifications[0][0]),
          ))),
          compilerRuntimeDependencies: [],
          resourceDependencyRoot: resourceRoot.split(path.sep).join('/'),
          linker: toolSpecifications[1][0],
          linkerInvocationPath: path.join(toolsDirectory, toolSpecifications[1][0])
            .split(path.sep).join('/'),
          linkerRealPath: fs.realpathSync(
            path.join(toolsDirectory, toolSpecifications[1][0]),
          ).split(path.sep).join('/'),
          linkerVersion,
          linkerSha256: sha256(fs.readFileSync(fs.realpathSync(
            path.join(toolsDirectory, toolSpecifications[1][0]),
          ))),
          linkerRuntimeDependencies: [],
        },
        parent: {
          objects: [{
            id: 'fixture-object',
            source,
            category: 'portable-control',
            optimization: '-Oz',
            flags: ['--target=wasm32'],
          }],
          linkFlags: ['-m', 'wasm32'],
        },
        relaxed: null,
        ptqAuthoring: null,
      },
    };
    const toolEvidence = Object.fromEntries(toolSpecifications.map(
      ([command, version], index) => {
        const invocation = path.join(toolsDirectory, command);
        const executable = fs.realpathSync(invocation);
        const bytes = fs.readFileSync(executable);
        return [index === 0 ? 'compiler' : 'linker', {
          command,
          path: invocation.split(path.sep).join('/'),
          realPath: executable.split(path.sep).join('/'),
          rawBytes: bytes.byteLength,
          sha256: sha256(bytes),
          version,
          runtimeDependencies: [],
        }];
      },
    ));
    const sourceBytes = fs.readFileSync(sourcePath);
    const objectBytes = Buffer.from('fixture object bytes\n');
    const buildRoot = path.join(temporaryRoot, 'build');
    const objectPath = wasmBuildObjectPath(
      buildRoot,
      profile,
      'parent',
      0,
      profile.recipe.parent.objects[0].id,
    );
    fs.mkdirSync(path.dirname(objectPath), { recursive: true });
    fs.writeFileSync(objectPath, objectBytes);
    const linkedBytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    ]);
    const component = wasmEvidenceComponent('parent', profile.recipe.parent, linkedBytes);
    component.orderedObjects[0].dependencies = [{
      scope: 'repository',
      path: source,
      rawBytes: sourceBytes.byteLength,
      sha256: sha256(sourceBytes),
    }, {
      scope: 'toolchain',
      path: resourceHeader.split(path.sep).join('/'),
      rawBytes: fs.statSync(resourceHeader).size,
      sha256: sha256(fs.readFileSync(resourceHeader)),
    }];
    component.orderedObjects[0].rawBytes = objectBytes.byteLength;
    component.orderedObjects[0].sha256 = sha256(objectBytes);
    const evidence = {
      artifact: profile.filename,
      components: [component],
      format: WASM_BUILD_EVIDENCE_FORMAT,
      profile: profile.id,
      recipeSha256: sha256(stableJSON(profile.recipe)),
      toolchain: toolEvidence,
    };

    assert.deepEqual(await validateWasmBuildEvidenceTools(
      evidence,
      profile,
      { repositoryRoot: temporaryRoot },
    ), toolEvidence);
    const graph = await validateWasmBuildEvidenceObjects(
      buildRoot,
      evidence,
      profile,
      { repositoryRoot: temporaryRoot },
    );
    assert.equal(graph.objectCount, 1);
    assert.equal(graph.dependencyCount, 2);
    assert.equal(graph.repositoryDependencyCount, 1);
    assert.equal(graph.toolchainDependencyCount, 1);
    assert.equal(graph.toolchainInputsVerified, true);

    process.env.PATH = '';
    const portableGraph = await validateWasmBuildEvidenceObjects(
      buildRoot,
      evidence,
      profile,
      { repositoryRoot: temporaryRoot, verifyToolchainInputs: false },
    );
    assert.equal(portableGraph.objectGraphSha256, graph.objectGraphSha256);
    assert.equal(portableGraph.toolchainInputsVerified, false);
    fs.rmSync(resourceHeader);
    assert.equal((await validateWasmBuildEvidenceObjects(
      buildRoot,
      evidence,
      profile,
      { repositoryRoot: temporaryRoot, verifyToolchainInputs: false },
    )).toolchainInputsVerified, false);
    process.env.PATH = `${toolsDirectory}${path.delimiter}${previousPath ?? ''}`;
    await assert.rejects(
      validateWasmBuildEvidenceObjects(
        buildRoot,
        evidence,
        profile,
        { repositoryRoot: temporaryRoot },
      ),
      /no longer exists/u,
    );
    fs.writeFileSync(resourceHeader, '#define FIXTURE_RESOURCE 1\n');

    const shadowTools = path.join(temporaryRoot, 'shadow-tools');
    fs.mkdirSync(shadowTools);
    fs.copyFileSync(
      path.join(toolsDirectory, 'fixture-clang'),
      path.join(shadowTools, 'fixture-clang'),
    );
    fs.chmodSync(path.join(shadowTools, 'fixture-clang'), 0o755);
    fs.copyFileSync(linkerDriver, path.join(shadowTools, 'fixture-lld-driver'));
    fs.chmodSync(path.join(shadowTools, 'fixture-lld-driver'), 0o755);
    fs.symlinkSync('fixture-lld-driver', path.join(shadowTools, 'fixture-wasm-ld'));
    process.env.PATH = `${shadowTools}${path.delimiter}${toolsDirectory}` +
      `${path.delimiter}${previousPath ?? ''}`;
    await assert.rejects(
      validateWasmBuildEvidenceTools(evidence, profile, { repositoryRoot: temporaryRoot }),
      /resolves to .* expected/u,
    );
    process.env.PATH = `${toolsDirectory}${path.delimiter}${previousPath ?? ''}`;

    const collisionPath = path.join(temporaryRoot, 'collision-output');
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: collisionPath,
        evidence: collisionPath,
        buildRoot: path.join(temporaryRoot, 'collision-build'),
      }),
      /must use different paths/u,
    );

    const foreignObjectBuildRoot = path.join(temporaryRoot, 'foreign-object-build');
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: path.join(
          foreignObjectBuildRoot,
          'objects',
          'another-profile',
          'parent',
          'linked.wasm',
        ),
        buildRoot: foreignObjectBuildRoot,
      }),
      /must be outside the private object directory/u,
    );

    const realPublishDirectory = path.join(temporaryRoot, 'real-publish');
    const aliasPublishDirectory = path.join(temporaryRoot, 'alias-publish');
    fs.mkdirSync(realPublishDirectory);
    fs.symlinkSync(realPublishDirectory, aliasPublishDirectory);
    const aliasedCollision = path.join(realPublishDirectory, 'same-file');
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: path.join(aliasPublishDirectory, 'same-file'),
        evidence: aliasedCollision,
        buildRoot: path.join(temporaryRoot, 'aliased-collision-build'),
      }),
      /must use different paths/u,
    );
    assert.equal(fs.existsSync(aliasedCollision), false);

    const compilerBefore = fs.readFileSync(path.join(toolsDirectory, 'fixture-clang'));
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: path.join(toolsDirectory, 'fixture-clang'),
        buildRoot: path.join(temporaryRoot, 'tool-collision-build'),
      }),
      /must not overwrite a release-recipe source file/u,
    );
    assert.deepEqual(
      fs.readFileSync(path.join(toolsDirectory, 'fixture-clang')),
      compilerBefore,
    );

    const rootSymlinkVictim = path.join(temporaryRoot, 'root-symlink-victim');
    const rootSymlinkBuildRoot = path.join(temporaryRoot, 'root-symlink-build');
    fs.mkdirSync(rootSymlinkVictim);
    fs.symlinkSync(rootSymlinkVictim, rootSymlinkBuildRoot);
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: path.join(temporaryRoot, 'root-symlink-output.wasm'),
        buildRoot: rootSymlinkBuildRoot,
      }),
      /build root contains a symbolic-link alias/u,
    );
    assert.deepEqual(fs.readdirSync(rootSymlinkVictim), []);

    const symlinkBuildRoot = path.join(temporaryRoot, 'symlink-build');
    const symlinkVictim = path.join(temporaryRoot, 'symlink-victim');
    fs.mkdirSync(symlinkBuildRoot, { recursive: true });
    fs.mkdirSync(symlinkVictim, { recursive: true });
    fs.symlinkSync(symlinkVictim, path.join(symlinkBuildRoot, 'objects'));
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: path.join(temporaryRoot, 'symlink-output.wasm'),
        buildRoot: symlinkBuildRoot,
      }),
      /not a real directory/u,
    );
    assert.deepEqual(fs.readdirSync(symlinkVictim), []);

    const objectSymlinkBuildRoot = path.join(temporaryRoot, 'object-symlink-build');
    const objectSymlinkPath = wasmBuildObjectPath(
      objectSymlinkBuildRoot,
      profile,
      'parent',
      0,
      profile.recipe.parent.objects[0].id,
    );
    const objectSymlinkVictim = path.join(temporaryRoot, 'object-symlink-victim');
    fs.mkdirSync(path.dirname(objectSymlinkPath), { recursive: true });
    fs.writeFileSync(objectSymlinkVictim, 'must remain intact\n');
    fs.symlinkSync(objectSymlinkVictim, objectSymlinkPath);
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: path.join(temporaryRoot, 'object-symlink-output.wasm'),
        buildRoot: objectSymlinkBuildRoot,
      }),
      /not a private regular file/u,
    );
    assert.equal(fs.readFileSync(objectSymlinkVictim, 'utf8'), 'must remain intact\n');

    const dependencyMutation = structuredClone(evidence);
    dependencyMutation.components[0].orderedObjects[0].dependencies[0].sha256 =
      '0'.repeat(64);
    await assert.rejects(
      validateWasmBuildEvidenceObjects(
        buildRoot,
        dependencyMutation,
        profile,
        { repositoryRoot: temporaryRoot },
      ),
      /dependency 'src\/fixture\.c' does not match its build evidence/u,
    );

    const toolMutation = structuredClone(evidence);
    toolMutation.toolchain.compiler.sha256 = '0'.repeat(64);
    await assert.rejects(
      validateWasmBuildEvidenceTools(
        toolMutation,
        profile,
        { repositoryRoot: temporaryRoot },
      ),
      /compiler digest differs from the release toolchain/u,
    );
  } finally {
    if (previousPath === undefined) delete process.env.PATH;
    else process.env.PATH = previousPath;
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
    fs.rmSync(resourceRoot, { recursive: true, force: true });
  }
});

test('WASM profile build stages every component before publishing', {
  skip: process.platform === 'win32',
}, async () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'volvoxai-wasm-transaction-'));
  const previousPath = process.env.PATH;
  const previousCpath = process.env.CPATH;
  try {
    const toolsDirectory = path.join(temporaryRoot, 'tools');
    const sourcesDirectory = path.join(temporaryRoot, 'src');
    const resourceDirectory = path.join(temporaryRoot, 'toolchain-resources');
    fs.mkdirSync(toolsDirectory, { recursive: true });
    fs.mkdirSync(sourcesDirectory, { recursive: true });
    fs.mkdirSync(resourceDirectory, { recursive: true });
    fs.writeFileSync(path.join(sourcesDirectory, 'parent.c'), 'int parent(void) { return 1; }\n');
    fs.writeFileSync(path.join(sourcesDirectory, 'relaxed.c'), 'int relaxed(void) { return 2; }\n');
    fs.writeFileSync(path.join(sourcesDirectory, 'ptq.c'), 'int ptq(void) { return 3; }\n');

    const compiler = path.join(toolsDirectory, 'transaction-clang');
    fs.writeFileSync(
      compiler,
        '#!/usr/bin/env node\n' +
        "const fs = require('node:fs');\n" +
        "const path = require('node:path');\n" +
        'const args = process.argv.slice(2);\n' +
        "if (process.env.CPATH) process.exit(41);\n" +
        "if (args.includes('--version')) {\n" +
        "  const marker = path.join(process.cwd(), 'retarget-publication.json');\n" +
        "  if (fs.existsSync(marker)) {\n" +
        "    const config = JSON.parse(fs.readFileSync(marker, 'utf8'));\n" +
        "    const count = fs.existsSync(config.counter) ? " +
          "Number(fs.readFileSync(config.counter, 'utf8')) + 1 : 1;\n" +
        "    fs.writeFileSync(config.counter, String(count));\n" +
        "    if (count === 2) {\n" +
        "      const prefix = `.${config.basename}.volvoxai-stage-`;\n" +
        "      const stages = fs.readdirSync(config.targetA).filter((name) => " +
          "name.startsWith(prefix));\n" +
        "      if (stages.length !== 1) process.exit(42);\n" +
        "      fs.rmSync(config.alias);\n" +
        "      fs.symlinkSync(config.targetB, config.alias);\n" +
        "      fs.copyFileSync(path.join(config.targetA, stages[0]), " +
          "path.join(config.targetB, stages[0]));\n" +
        "    }\n" +
        "  }\n" +
        "  console.log('transaction clang 1.0'); process.exit(0);\n" +
        "}\n" +
        "const value = (flag) => args[args.indexOf(flag) + 1];\n" +
        "const source = args.find((arg) => arg.endsWith('.c'));\n" +
        "const depfile = value('-MF');\n" +
        "fs.mkdirSync(path.dirname(depfile), { recursive: true });\n" +
        "fs.writeFileSync(depfile, `${value('-MT')}: ${source}\\n`);\n" +
        "if (args.includes('-c')) { const output = value('-o'); " +
          "fs.writeFileSync(output, `object:${source}\\n`); }\n",
    );
    fs.chmodSync(compiler, 0o755);

    const linker = path.join(toolsDirectory, 'transaction-wasm-ld');
    fs.writeFileSync(
      linker,
      '#!/usr/bin/env node\n' +
        "const fs = require('node:fs');\n" +
        "const path = require('node:path');\n" +
        'const args = process.argv.slice(2);\n' +
        "if (args.includes('--version')) { console.log('transaction LLD 1.0'); process.exit(0); }\n" +
        "const output = args[args.indexOf('-o') + 1];\n" +
        "if (fs.existsSync(path.join(process.cwd(), 'fail-relaxed')) && " +
          "output.includes('/relaxed/')) process.exit(23);\n" +
        'fs.writeFileSync(output, Buffer.from([0,97,115,109,1,0,0,0]));\n',
    );
    fs.chmodSync(linker, 0o755);
    process.env.PATH = `${toolsDirectory}${path.delimiter}${previousPath ?? ''}`;
    process.env.CPATH = path.join(temporaryRoot, 'hostile-include');

    const object = (id, source) => ({
      id,
      source,
      category: 'portable-control',
      optimization: '-Oz',
      flags: ['--target=wasm32'],
    });
    const profile = {
      id: 'transaction',
      filename: 'transaction.wasm',
      forbiddenDependencyFiles: [],
      forbiddenDependencyPrefixes: [],
      recipe: {
        toolchain: {
          compiler: path.basename(compiler),
          compilerInvocationPath: compiler.split(path.sep).join('/'),
          compilerRealPath: fs.realpathSync(compiler).split(path.sep).join('/'),
          compilerVersion: 'transaction clang 1.0',
          compilerSha256: sha256(fs.readFileSync(compiler)),
          compilerRuntimeDependencies: [],
          resourceDependencyRoot: resourceDirectory.split(path.sep).join('/'),
          linker: path.basename(linker),
          linkerInvocationPath: linker.split(path.sep).join('/'),
          linkerRealPath: fs.realpathSync(linker).split(path.sep).join('/'),
          linkerVersion: 'transaction LLD 1.0',
          linkerSha256: sha256(fs.readFileSync(linker)),
          linkerRuntimeDependencies: [],
        },
        parent: {
          objects: [object('parent', 'src/parent.c')],
          linkFlags: ['-m', 'wasm32'],
        },
        relaxed: {
          objects: [object('relaxed', 'src/relaxed.c')],
          linkFlags: ['-m', 'wasm32'],
        },
        ptqAuthoring: null,
      },
    };
    const outputDirectory = path.join(temporaryRoot, 'dist');
    const evidenceDirectory = path.join(temporaryRoot, 'evidence');
    const buildRoot = path.join(temporaryRoot, 'build');
    const parentOutput = path.join(outputDirectory, profile.filename);
    const relaxedOutput = path.join(temporaryRoot, 'relaxed.wasm');
    const aggregateEvidence = path.join(
      evidenceDirectory,
      `${profile.filename}.build.json`,
    );
    fs.mkdirSync(outputDirectory);
    fs.mkdirSync(evidenceDirectory);
    fs.writeFileSync(parentOutput, 'old parent\n');
    fs.writeFileSync(relaxedOutput, 'old relaxed\n');
    fs.writeFileSync(aggregateEvidence, 'old evidence\n');

    const relaxedFailureMarker = path.join(temporaryRoot, 'fail-relaxed');
    fs.writeFileSync(relaxedFailureMarker, 'fail\n');
    await assert.rejects(
      buildWasmProfile({
        repositoryRoot: temporaryRoot,
        profile,
        outputDirectory,
        buildRoot,
        evidenceDirectory,
        relaxedOutput,
      }),
      /transaction-wasm-ld .* failed/u,
    );
    assert.equal(fs.readFileSync(parentOutput, 'utf8'), 'old parent\n');
    assert.equal(fs.readFileSync(relaxedOutput, 'utf8'), 'old relaxed\n');
    assert.equal(fs.readFileSync(aggregateEvidence, 'utf8'), 'old evidence\n');

    fs.rmSync(relaxedFailureMarker);
    const relaxedSource = path.join(sourcesDirectory, 'relaxed.c');
    const relaxedSourceBefore = fs.readFileSync(relaxedSource, 'utf8');
    fs.rmSync(parentOutput);
    fs.symlinkSync(relaxedSource, parentOutput);
    await assert.rejects(
      buildWasmProfile({
        repositoryRoot: temporaryRoot,
        profile,
        outputDirectory,
        buildRoot,
        evidenceDirectory,
        relaxedOutput,
      }),
      /must not overwrite a release-recipe source file/u,
    );
    assert.equal(fs.readFileSync(relaxedSource, 'utf8'), relaxedSourceBefore);
    assert.equal(fs.lstatSync(parentOutput).isSymbolicLink(), true);
    fs.rmSync(parentOutput);
    fs.writeFileSync(parentOutput, 'old parent\n');

    const result = await buildWasmProfile({
      repositoryRoot: temporaryRoot,
      profile,
      outputDirectory,
      buildRoot,
      evidenceDirectory,
      relaxedOutput,
    });
    assert.deepEqual([...fs.readFileSync(parentOutput)], [0, 97, 115, 109, 1, 0, 0, 0]);
    assert.deepEqual([...fs.readFileSync(relaxedOutput)], [0, 97, 115, 109, 1, 0, 0, 0]);
    assert.equal(result.evidencePath, aggregateEvidence);
    assert.deepEqual(
      (await readWasmBuildEvidence(aggregateEvidence, profile)).components
        .map(({ kind }) => kind),
      ['parent', 'relaxed'],
    );

    const raceTargetA = path.join(temporaryRoot, 'publication-target-a');
    const raceTargetB = path.join(temporaryRoot, 'publication-target-b');
    const raceAlias = path.join(temporaryRoot, 'publication-alias');
    const raceOutput = path.join(raceAlias, 'out.wasm');
    const raceMarker = path.join(temporaryRoot, 'retarget-publication.json');
    const raceCounter = path.join(temporaryRoot, 'retarget-publication.count');
    fs.mkdirSync(raceTargetA);
    fs.mkdirSync(raceTargetB);
    fs.symlinkSync(raceTargetA, raceAlias);
    fs.writeFileSync(raceMarker, JSON.stringify({
      alias: raceAlias,
      basename: path.basename(raceOutput),
      counter: raceCounter,
      targetA: raceTargetA,
      targetB: raceTargetB,
    }));
    await assert.rejects(
      buildWasmComponent({
        repositoryRoot: temporaryRoot,
        profile,
        kind: 'parent',
        output: raceOutput,
        buildRoot: path.join(temporaryRoot, 'publication-race-build'),
      }),
      /publication directory changed before commit/u,
    );
    fs.rmSync(raceMarker);
    assert.equal(fs.existsSync(path.join(raceTargetA, 'out.wasm')), false);
    assert.equal(fs.existsSync(path.join(raceTargetB, 'out.wasm')), false);
    assert.deepEqual(
      fs.readdirSync(raceTargetA).filter((name) => name.includes('.volvoxai-')),
      [],
    );

    const fullProfile = structuredClone(profile);
    fullProfile.id = 'transaction-full';
    fullProfile.filename = 'transaction.full.wasm';
    fullProfile.recipe.ptqAuthoring = {
      objects: [object('ptq', 'src/ptq.c')],
      linkFlags: ['-m', 'wasm32'],
    };
    const parentBeforeAll = fs.readFileSync(parentOutput);
    const evidenceBeforeAll = fs.readFileSync(aggregateEvidence);
    const fullParentOutput = path.join(outputDirectory, fullProfile.filename);
    const sharedRelaxedOutput = path.join(temporaryRoot, 'all-relaxed.wasm');
    await assert.rejects(
      buildWasmReleaseProfiles({
        repositoryRoot: temporaryRoot,
        profiles: [profile, fullProfile],
        outputDirectory,
        buildRoot: path.join(temporaryRoot, 'all-build-collision'),
        evidenceDirectory,
        relaxedOutput: sharedRelaxedOutput,
        // Cross-profile collision: this is the inference parent, but it is
        // only the full profile's PTQ child destination.
        ptqOutput: parentOutput,
      }),
      /must use different paths/u,
    );
    assert.deepEqual(fs.readFileSync(parentOutput), parentBeforeAll);
    assert.deepEqual(fs.readFileSync(aggregateEvidence), evidenceBeforeAll);
    assert.equal(fs.existsSync(fullParentOutput), false);
    assert.equal(fs.existsSync(sharedRelaxedOutput), false);

    const ptqOutput = path.join(temporaryRoot, 'ptq.wasm');
    const allResult = await buildWasmReleaseProfiles({
      repositoryRoot: temporaryRoot,
      profiles: [profile, fullProfile],
      outputDirectory,
      buildRoot: path.join(temporaryRoot, 'all-build'),
      evidenceDirectory,
      relaxedOutput: sharedRelaxedOutput,
      ptqOutput,
    });
    assert.equal(allResult.profiles.length, 2);
    assert.deepEqual([...fs.readFileSync(sharedRelaxedOutput)], [0, 97, 115, 109, 1, 0, 0, 0]);
    assert.deepEqual([...fs.readFileSync(fullParentOutput)], [0, 97, 115, 109, 1, 0, 0, 0]);
    assert.deepEqual([...fs.readFileSync(ptqOutput)], [0, 97, 115, 109, 1, 0, 0, 0]);
  } finally {
    if (previousPath === undefined) delete process.env.PATH;
    else process.env.PATH = previousPath;
    if (previousCpath === undefined) delete process.env.CPATH;
    else process.env.CPATH = previousCpath;
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
});

test('regression gate uses max(0.5%, 8192 B) and keeps targets informational', () => {
  assert.equal(regressionAllowance(100_000, {
    relativeIncrease: 0.005,
    absoluteIncreaseBytes: 8192,
  }), 8192);

  const atLimit = budgetFixture(108_192);
  const passing = evaluateSizeBudgets(atLimit.report, atLimit.budgets);
  assert.equal(passing.passed, true);
  assert.equal(passing.plannedTargets[0].status, 'informational');
  assert.equal(passing.plannedTargets[0].metrics.rawBytes.meetsPlannedTarget, false);
  assert.equal(passing.plannedTargets[1].status, 'deferred');

  const aboveLimit = budgetFixture(108_193);
  const failing = evaluateSizeBudgets(aboveLimit.report, aboveLimit.budgets);
  assert.equal(failing.passed, false);
  assert.equal(failing.failures.length, 1);
  assert.equal(failing.failures[0].limitBytes, 108_192);
});

test('linked WebAssembly import/export drift is an exact gate without size allowance', () => {
  const inspection = inspectWasmBytes(syntheticWasm());
  const id = 'dist/<package-version>/fixture.wasm';
  const report = {
    scope: 'web-only',
    artifacts: [{
      id,
      wasm: inspection,
    }],
    browserProfiles: [],
  };
  const budgets = {
    format: RELEASE_SIZE_BUDGET_FORMAT,
    policy: { relativeIncrease: 0.005, absoluteIncreaseBytes: 8192 },
    baselines: {
      wasmContracts: {
        [id]: {
          importCount: inspection.importCount,
          importSetSha256: inspection.importSetSha256,
          exportCount: inspection.exportCount,
          exportSetSha256: inspection.exportSetSha256,
        },
      },
    },
  };
  assert.equal(evaluateSizeBudgets(report, budgets).passed, true);

  report.artifacts[0].wasm = {
    ...inspection,
    exportCount: inspection.exportCount + 1,
    exportSetSha256: '0'.repeat(64),
  };
  const result = evaluateSizeBudgets(report, budgets);
  assert.equal(result.passed, false);
  assert.deepEqual(result.failures.map(({ metric }) => metric), [
    'exportCount',
    'exportSetSha256',
  ]);
  assert.equal(result.failures[0].comparison, 'exact');
  assert.equal('allowanceBytes' in result.failures[0], false);
  assert.match(formatBudgetFailure(result.failures[0]), /exact baseline/);
});

test('committed budgets exactly cover every fixed artifact, pair, and WASM contract', () => {
  const budgets = JSON.parse(fs.readFileSync(
    new URL('../tools/release_size_budgets.json', import.meta.url),
    'utf8',
  ));
  assert.doesNotThrow(() => validateReleaseSizeBudgetInventory(budgets));
  for (const group of ['artifacts', 'browserPairs', 'wasmContracts']) {
    const incomplete = structuredClone(budgets);
    delete incomplete.baselines[group][Object.keys(incomplete.baselines[group])[0]];
    assert.throws(
      () => validateReleaseSizeBudgetInventory(incomplete),
      /must exactly cover the fixed release inventory; missing:/,
    );
  }
  const missingMetric = structuredClone(budgets);
  delete missingMetric.baselines.artifacts['native/volvoxai'].gzipBytes;
  assert.throws(
    () => validateReleaseSizeBudgetInventory(missingMetric),
    /metrics must exactly cover the fixed release inventory; missing: gzipBytes/,
  );
  const invalidHash = structuredClone(budgets);
  invalidHash.baselines.wasmContracts[
    'dist/<package-version>/volvoxai.wasm'
  ].exportSetSha256 = 'not-a-sha256';
  assert.throws(
    () => validateReleaseSizeBudgetInventory(invalidHash),
    /invalid exportSetSha256 value/,
  );
});

test('release composition authority imports before node dependencies are installed', () => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'volvoxai-release-profile-'));
  try {
    fs.mkdirSync(path.join(temporaryRoot, 'tools', 'generated'), { recursive: true });
    for (const relative of [
      'tools/release_profiles.mjs',
      'tools/wasm_host_imports.mjs',
      'tools/generated/wasmInternalAbi.mjs',
    ]) {
      fs.copyFileSync(path.join(ROOT, relative), path.join(temporaryRoot, relative));
    }
    const moduleUrl = pathToFileURL(
      path.join(temporaryRoot, 'tools', 'release_profiles.mjs'),
    ).href;
    const result = spawnSync(
      process.execPath,
      ['--input-type=module', '--eval', `await import(${JSON.stringify(moduleUrl)});`],
      { encoding: 'utf8', env: { ...process.env, NODE_PATH: '' } },
    );
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
});

test('browser metafile categories include unattributed bundler overhead', () => {
  const result = aggregateBrowserMetafile({
    outputs: {
      'dist/fixture.js': {
        inputs: {
          'runtime/generated/typescript/volvoxai_lite.ts': { bytesInOutput: 40 },
          'ts/core/ModelControlWasm.ts': { bytesInOutput: 35 },
          'ts/backends/WebGPUHostBridge.ts': { bytesInOutput: 10 },
          'ts/index.ts': { bytesInOutput: 5 },
        },
      },
    },
  }, 110);
  assert.deepEqual(result.categories, {
    'bundler-overhead': 20,
    generated: 40,
    other: 5,
    'webgpu-wgsl': 10,
    'wasm-portable-control': 35,
  });
});

test('LLVM text parsers preserve ELF section sizes and largest symbols', () => {
  const sections = parseLlvmReadobjSections(`
Section Headers:
  [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
  [ 1] .text             PROGBITS        0000000000001000 001000 000020 00  AX  0   0 16
  [ 2] .rodata           PROGBITS        0000000000002000 002000 000010 00   A  0   0 16
`);
  assert.deepEqual(sections, [
    { name: '.text', sizeBytes: 32 },
    { name: '.rodata', sizeBytes: 16 },
  ]);
  const symbols = parseLlvmNmSymbols('small t 16 8\nlarge T 32 64\n');
  assert.equal(symbols[0].name, 'large');
  assert.equal(symbols[0].sizeBytes, 64);
  assert.equal(parseLlvmReadobjBuildId(`
    Note {
      Owner: GNU
      Build ID: ABCDEF0123456789
    }
  `), 'abcdef0123456789');
});

test('native release and debug section policies keep analysis symbols out of release ELF', () => {
  const releaseSections = [
    '.text', '.dynsym', '.dynstr', '.gnu_debuglink',
  ].map((name) => ({ name, sizeBytes: 1 }));
  const debugSections = [
    '.text', '.symtab', '.strtab', '.debug_info',
  ].map((name) => ({ name, sizeBytes: 1 }));
  assert.equal(validateStrippedNativeSections(releaseSections).stripped, true);
  assert.equal(validateNativeDebugSections(debugSections).symbolized, true);

  assert.throws(
    () => validateStrippedNativeSections([...releaseSections, {
      name: '.symtab', sizeBytes: 1,
    }]),
    /retains stripped symbol\/debug sections: \.symtab/,
  );
  assert.throws(
    () => validateStrippedNativeSections(releaseSections
      .filter(({ name }) => name !== '.gnu_debuglink')),
    /missing required runtime\/debuglink sections: \.gnu_debuglink/,
  );
  assert.throws(
    () => validateNativeDebugSections(debugSections
      .filter(({ name }) => name !== '.debug_info')),
    /missing required symbol\/debug sections: \.debug_info/,
  );
});

test('native finalization evidence binds release, debug sidecar, hashes, and build IDs', () => {
  const fixture = nativeFinalizationFixture();
  const provenance = validateNativeReleaseFinalization(fixture);
  assert.equal(provenance.format, NATIVE_RELEASE_FINALIZATION_FORMAT);
  assert.equal(provenance.path, fixture.profile.debugEvidence);
  assert.equal(provenance.rawBytes, fixture.evidenceArtifact.rawBytes);
  assert.equal(provenance.sha256, fixture.evidenceArtifact.sha256);
  assert.equal(provenance.buildId, fixture.releaseBuildId);
  assert.equal(provenance.beforeEqualsAfter, true);
  assert.deepEqual(provenance.release.definedDynamicExports, []);

  const changedAfter = structuredClone(fixture);
  changedAfter.evidence.after.version = 'volvoxai changed';
  assert.throws(
    () => validateNativeReleaseFinalization(changedAfter),
    /pre\/post snapshots differ/,
  );

  const wrongReleaseFingerprint = structuredClone(fixture);
  wrongReleaseFingerprint.evidence.release.rawBytes += 1;
  assert.throws(
    () => validateNativeReleaseFinalization(wrongReleaseFingerprint),
    /release size\/SHA-256 does not match/,
  );

  const wrongDebugFingerprint = structuredClone(fixture);
  wrongDebugFingerprint.evidence.debug.sha256 = '9'.repeat(64);
  assert.throws(
    () => validateNativeReleaseFinalization(wrongDebugFingerprint),
    /debug size\/SHA-256 does not match/,
  );

  const wrongBuildId = structuredClone(fixture);
  wrongBuildId.debugBuildId = 'cd'.repeat(20);
  assert.throws(
    () => validateNativeReleaseFinalization(wrongBuildId),
    /build IDs differ/,
  );

  const dynamicExport = structuredClone(fixture);
  dynamicExport.evidence.release.definedDynamicExports = ['main'];
  assert.throws(
    () => validateNativeReleaseFinalization(dynamicExport),
    /definedDynamicExports=\[\]/,
  );

  const wrongBuildTree = structuredClone(fixture);
  wrongBuildTree.evidence.build.directory = 'build/another-tree';
  assert.throws(
    () => validateNativeReleaseFinalization(wrongBuildTree),
    /belongs to another build directory/,
  );

  const wrongMap = structuredClone(fixture);
  wrongMap.evidence.build.linkMap.sha256 = '6'.repeat(64);
  assert.throws(
    () => validateNativeReleaseFinalization(wrongMap),
    /linker map does not match the selected build tree/,
  );

  const overwrittenPublishedArtifact = structuredClone(fixture);
  overwrittenPublishedArtifact.releaseArtifact.sha256 = '7'.repeat(64);
  overwrittenPublishedArtifact.evidence.release.sha256 = '7'.repeat(64);
  assert.throws(
    () => validateNativeReleaseFinalization(overwrittenPublishedArtifact),
    /Published native release bundle differs from the selected build tree/,
  );
});

test('GNU linker-map parser attributes linked bytes to objects and groups', () => {
  const result = parseNativeLinkMapObjects(`
Discarded input sections
 .text.dead     0x0000000000000000       0x80 CMakeFiles/volvoxai.dir/dead.c.o
Linker script and memory map
 .rela.data.rel.ro.synthetic
                0x0000000000000200      0x400 /lib/x86_64-linux-gnu/Scrt1.o
 .dynsym        0x0000000000000600      0x200 /lib/x86_64-linux-gnu/Scrt1.o
 .bss.workspace 0x0000000000000800      0x800 CMakeFiles/volvoxai.dir/src/kernels/k.c.o
 .text.kernel   0x0000000000001000       0x20 CMakeFiles/volvoxai.dir/src/kernels/k.c.o
 .text.train
                0x0000000000001020       0x10 CMakeFiles/volvoxai-full.dir/src/training/t.c.o
 .rodata.train  0x0000000000001030        0x8 CMakeFiles/volvoxai-full.dir/src/training/t.c.o
 .gnu.linkonce.t.helper
                0x0000000000001038        0x4 CMakeFiles/volvoxai.dir/src/kernels/k.c.o
OUTPUT(native/volvoxai elf64-x86-64)
 .text.after-output 0x000000000000103c       0x40 CMakeFiles/volvoxai.dir/src/kernels/k.c.o
`);
  assert.equal(result.objectCount, 2);
  assert.equal(result.attributedBytes, 60);
  assert.equal(result.groups.kernels, 36);
  assert.equal(result.groups['training-ptq'], 24);
  assert.equal(result.topObjects.some(({ object }) => object.endsWith('/Scrt1.o')), false);
});

test('LLD linker-map parser ignores synthetic and NOBITS ownership', () => {
  const result = parseNativeLinkMapObjects(`
             VMA              LMA     Size Align Out     In      Symbol
            1000             1000       20    16         CMakeFiles/volvoxai.dir/src/runtime/r.c.o:(.text.run)
            1020             1020       40     8         /lib/Scrt1.o:(.rela.dyn)
            1060             1060       80    16         CMakeFiles/volvoxai.dir/src/runtime/r.c.o:(.bss.cache)
            10e0             10e0       10     8         libfixture.a(k.o):(.data.table)
  `);
  assert.equal(result.objectCount, 2);
  assert.equal(result.attributedBytes, 48);
  assert.equal(result.groups['runtime-control'], 32);
  assert.equal(result.groups.other, 16);
});

test('native linker-map OUTPUT must resolve to the inspected release artifact', () => {
  const linkDirectory = '/workspace/build/cmake/native';
  const artifact = '/workspace/native/volvoxai';
  assert.equal(
    validateNativeLinkMapTarget(
      'OUTPUT(../../../native/volvoxai elf64-x86-64)\n',
      linkDirectory,
      artifact,
    ),
    artifact,
  );
  assert.throws(
    () => validateNativeLinkMapTarget(
      'OUTPUT(/tmp/fixture/volvoxai-owner-projection elf64-x86-64)\n',
      linkDirectory,
      artifact,
    ),
    /not release artifact/,
  );
});

test('Markdown renders the gate, planned target disclaimer, and deferred archive', () => {
  const fixture = budgetFixture();
  const nativeFixture = nativeFinalizationFixture();
  const nativeProvenance = validateNativeReleaseFinalization(nativeFixture);
  const wasmFixture = inferenceWasmBuildEvidenceFixture();
  const wasmInspection = inspectWasmBytes(wasmFixture.artifactBytes);
  const wasmEvidenceBytes = Buffer.from(`${stableJSON(wasmFixture.evidence)}\n`);
  const wasmEvidenceSha256 = sha256(wasmEvidenceBytes);
  const wasmObjectCount = wasmFixture.evidence.components.reduce(
    (total, component) => total + component.orderedObjects.length,
    0,
  );
  const wasmDependencyCount = wasmFixture.evidence.components.reduce(
    (componentTotal, component) => componentTotal + component.orderedObjects.reduce(
      (objectTotal, object) => objectTotal + object.dependencies.length,
      0,
    ),
    0,
  );
  const wasmObjectGraphSha256 = sha256('fixture-object-graph');
  const wasmProvenance = {
    sourceSha256: '1'.repeat(64),
    recipeSha256: wasmFixture.evidence.recipeSha256,
    contractSha256: '2'.repeat(64),
    payloadSha256: '3'.repeat(64),
    buildEvidenceSha256: wasmEvidenceSha256,
  };
  const budgetEvaluation = evaluateSizeBudgets(fixture.report, fixture.budgets);
  const document = {
    format: 'volvoxai-release-size-report/v1',
    generatedAt: '2026-09-01T00:00:00.000Z',
    scope: 'full',
    packageVersion: '0.4.0',
    source: { commit: 'deadbeef', dirty: false },
    environment: {
      toolchain: {
        node: 'v20.0.0',
        esbuild: '0.28.1',
        gzip: 'gzip 1.12',
        wasmCompiler: {
          status: 'available',
          ...wasmFixture.evidence.toolchain.compiler,
        },
        wasmLinker: {
          status: 'available',
          ...wasmFixture.evidence.toolchain.linker,
        },
      },
    },
    buildProvenance: {
      authority: 'tools/release_profiles.mjs',
      wasmProfiles: [{
        id: wasmFixture.profile.id,
        artifact: 'dist/<package-version>/volvoxai.wasm',
        capability: wasmFixture.profile.capability,
        sourceEntries: [...wasmFixture.profile.sourceEntries],
        recipe: structuredClone(wasmFixture.profile.recipe),
        recipeSha256: wasmFixture.evidence.recipeSha256,
      }],
      native: { status: 'out-of-scope' },
    },
    artifacts: [
      {
        id: 'dist/<package-version>/fixture.js',
        kind: 'browser',
        rawBytes: 108_192,
        gzipBytes: 50_000,
      },
      {
        id: 'native/volvoxai',
        kind: 'native',
        rawBytes: nativeFixture.releaseArtifact.rawBytes,
        gzipBytes: 90,
        native: {
          sectionBytes: { text: 32, symtab: 0, strtab: 0 },
          releaseElf: {
            ...nativeFixture.releaseArtifact,
            buildId: nativeFixture.releaseBuildId,
          },
          debugSidecar: {
            ...nativeFixture.debugArtifact,
            buildId: nativeFixture.debugBuildId,
          },
          finalizationEvidence: nativeProvenance,
          objectAttribution: { status: 'unavailable', reason: 'fixture' },
          topSymbolSource: nativeFixture.profile.debugArtifact,
          topSymbols: [{ name: 'main', type: 'T', sizeBytes: 16 }],
        },
      },
      {
        id: 'dist/<package-version>/volvoxai.wasm',
        kind: 'wasm',
        rawBytes: wasmFixture.artifactBytes.byteLength,
        gzipBytes: 64,
        wasm: {
          ...wasmInspection,
          exportOwnership: {
            authority: {
              source: 'tools/generated/wasmInternalAbi.mjs',
              manifestSha256: '4'.repeat(64),
            },
            manifestRequired: { count: 0 },
            unmanaged: { count: 0 },
          },
          provenance: wasmProvenance,
          buildEvidence: {
            path: 'build/wasm/provenance/volvoxai.wasm.build.json',
            rawBytes: wasmEvidenceBytes.byteLength,
            evidenceSha256: wasmEvidenceSha256,
            objectGraph: {
              objectCount: wasmObjectCount,
              dependencyCount: wasmDependencyCount,
              objectGraphSha256: wasmObjectGraphSha256,
            },
            componentBinding: validateWasmBuildEvidenceAgainstArtifact(
              wasmFixture.artifactBytes,
              wasmFixture.evidence,
              wasmFixture.profile,
            ),
            evidence: wasmFixture.evidence,
          },
        },
      },
    ],
    browserProfiles: [{
      id: 'fixture',
      minifiedArtifact: 'dist/<package-version>/fixture.js',
      sidecarArtifact: 'dist/<package-version>/fixture.wasm',
      pair: { rawBytes: 200_000, gzipBytes: 80_000 },
      metafile: { categories: { generated: 50, 'bundler-overhead': 10 } },
    }],
    extensionArchive: {
      status: 'deferred',
      reason: 'M3 owns the extension artifact.',
    },
    budgetEvaluation,
  };
  const markdown = renderReleaseSizeMarkdown(document);
  assert.match(markdown, /Regression gate: \*\*PASS\*\*/);
  assert.match(markdown, /Planned targets \(informational only\)/);
  assert.match(markdown, /extension\/inference-only/);
  assert.match(markdown, /Status: \*\*deferred\*\*/);
  assert.match(markdown, /Release ELF \/ section source: `native\/volvoxai`/);
  assert.match(markdown, /Debug sidecar \/ top-symbol source: `native\/\.debug\/volvoxai\.debug`/);
  assert.match(markdown, /Finalization evidence: `native\/\.debug\/volvoxai\.debug\.json`/);
  assert.match(markdown, /pre\/post identical; GNU debuglink present; 0 defined dynamic exports/);
  assert.match(markdown, /Top symbols are read from `native\/\.debug\/volvoxai\.debug`/);
  assert.match(markdown, /WASM compiler: `Ubuntu clang version 17\.0\.6/);
  assert.match(markdown, /WASM linker: `Ubuntu LLD 17\.0\.6` \(`\/usr\/bin\/wasm-ld-17`\)/);
  assert.match(markdown, /WASM inference: `clang-17` \+ `wasm-ld-17`/);
  const objects = wasmFixture.profile.recipe.parent.objects;
  const o3Count = objects.filter(object => object.optimization === '-O3').length;
  const ozCount = objects.filter(object => object.optimization === '-Oz').length;
  assert.match(markdown, new RegExp(`${o3Count} O3, ${ozCount} Oz`));
  assert.match(markdown, /Object build evidence: `build\/wasm\/provenance\/volvoxai\.wasm\.build\.json`/);
  assert.match(markdown, new RegExp(wasmEvidenceSha256, 'u'));
  for (const digest of Object.values(wasmProvenance)) {
    assert.match(markdown, new RegExp(digest, 'u'));
  }
  assert.ok(markdown.includes(
    `Verified object graph: ${wasmObjectCount} objects, ` +
      `${wasmDependencyCount} dependency records, SHA-256 ` +
      `\`${wasmObjectGraphSha256}\``,
  ));
  for (const customSection of wasmInspection.customSections) {
    assert.ok(markdown.includes(
      `| ${customSection.index} | \`${customSection.name}\` | ` +
        `${customSection.contentBytes} | \`${customSection.contentSha256}\` |`,
    ));
  }
  const parent = wasmFixture.evidence.components[0];
  assert.match(markdown, new RegExp(parent.link.flagsSha256, 'u'));
  assert.match(markdown, new RegExp(parent.orderedObjects[0].flagsSha256, 'u'));
  assert.match(markdown, /0: `kernels` \(`native\/src\/kernels\/kernels\.c`\).*`-O3`/u);
  assert.match(
    markdown,
    /1: `portable-control` \(`native\/src\/runtime\/portable_control_wasm\.c`\).*`-Oz`/u,
  );
});
