import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import {
  createTinyReceiptSplitE2EBindings,
  createTinyReceiptSplitE2EFixtureManifest,
  createTinyReceiptSplitE2EImage,
  createTinyReceiptSplitE2EReference,
  requireTinyReceiptPhysicalAdapterIdentity,
  runTinyReceiptSplitE2E,
  tinyReceiptSplitE2ERawBytes,
  validateTinyReceiptSplitE2EReference,
} from '../TinyReceiptSplitE2E.js';

const SHA = 'a'.repeat(64);

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function vocabulary() {
  const result = ['<pad>', '<bos>', '<eos>', '<unk>'];
  while (result.length < 760) result.push(`token-${result.length}`);
  for (const [id, character] of [
    [4, ' '], [5, 'p'], [6, 'h'], [7, 'o'], [8, 'n'], [9, 'e'],
    [10, 'u'], [11, 'm'], [12, 'b'], [13, 'r'], [14, 'l'], [15, 'a'],
    [16, 's'], [17, 't'],
  ]) result[id] = character;
  return result;
}

function route(overrides = {}) {
  return {
    tierFallback: false,
    operator: {
      attestation: 'none',
      used: false,
      offendingNode: null,
      ...(overrides.operator || {}),
    },
    ...overrides,
  };
}

function fixtureDevice(backend, device = 'fixture') {
  return { backend, device };
}

function compilation(backend, index, {
  device = fixtureDevice(backend),
  candidateDevice = device,
  ...overrides
} = {}) {
  return {
    requestedPolicy: {
      mode: 'require',
      backend,
      operatorFallback: 'forbid',
    },
    selectedBackend: backend,
    selectedDevice: device,
    routeEvidence: route(),
    candidates: [{
      backend,
      outcome: 'selected',
      device: candidateDevice,
      routeEvidence: route(),
    }],
    compilationId: `compilation-${index}`,
    ...overrides,
  };
}

function execution(backend, contextId, index, overrides = {}) {
  return {
    executionId: `execution-${index}`,
    contextId,
    backend,
    device: { backend, device: 'fixture' },
    outcome: 'success',
    operatorFallback: 'none',
    routeEvidence: route(),
    decodeState: { operation: 'execute' },
    executionTimeMs: index + 0.5,
    ...overrides,
  };
}

function fakeEnvironment({
  fallback = false,
  candidateDeviceMismatch = false,
  compilationDeviceMismatch = false,
  executionDeviceMismatch = false,
  cleanupFailure = false,
  tokenizerHash = null,
} = {}) {
  const state = {
    sessionClosed: 0,
    runtimeClosed: 0,
    cacheCleared: 0,
    compileOptions: null,
  };
  let diagnostic;
  const runtime = {
    close: async () => { state.runtimeClosed++; },
  };
  const api = {
    VolvoxAI: {
      async createRuntime(options) {
        assert.deepEqual(options.backends, ['cpu']);
        diagnostic = options.onDiagnostic;
        return runtime;
      },
    },
    Graph: class Graph {},
    GraphLoader: { async load() { throw new Error('fake session must not load graphs'); } },
    ReadOnlySafetensorsCache: class Cache {
      clear() { state.cacheCleared++; }
    },
  };
  const itos = vocabulary();
  class Session {
    static async load(options) {
      state.compileOptions = options.compileOptions;
      return new Session();
    }

    constructor() {
      this.package = {
        encoder: {
          graph: { sha256: SHA },
          weights: { sha256: 'b'.repeat(64) },
        },
        decoder: {
          graph: { sha256: 'c'.repeat(64) },
          weights: { sha256: 'd'.repeat(64) },
        },
      };
      this.vocab = { itos, ...(tokenizerHash == null ? {} : { tokenizerHash }) };
    }

    async generate(options) {
      assert.equal(options.prompt, 'phone number last one');
      assert.equal(options.maxNewTokens, 4);
      assert.equal(options.family, 'auto');
      assert.equal(options.preprocessed, true);
      assert.equal(sha256(new Uint8Array(options.image.buffer)),
        '028acedd12b13cfcd706b8c364f41e82dd80fd34218e61612e31c0c9ad5474fa');
      diagnostic({
        kind: 'compilation',
        report: compilation('cpu', 0, {
          ...(fallback ? { routeEvidence: route({ tierFallback: true }) } : {}),
          ...(candidateDeviceMismatch
            ? { candidateDevice: fixtureDevice('cpu', 'different-candidate') }
            : {}),
        }),
      });
      diagnostic({ kind: 'execution', report: execution('cpu', 'encoder-context', 0) });
      diagnostic({
        kind: 'compilation',
        report: compilation('cpu', 1, compilationDeviceMismatch
          ? { device: fixtureDevice('cpu', 'different-compilation') }
          : {}),
      });
      for (let index = 0; index < 2; index++) {
        diagnostic({
          kind: 'execution',
          report: execution(
            'cpu',
            'decoder-context',
            index + 1,
            executionDeviceMismatch && index === 0
              ? { device: fixtureDevice('cpu', 'different-execution') }
              : {},
          ),
        });
      }
      return {
        family: 'phone',
        familyId: 0,
        requestedFamily: 'auto',
        requestedFamilyId: -1,
        questionTokenIds: [5, 6, 7, 8, 9, 4, 8, 10, 11, 12, 9, 13, 4, 14, 15, 16, 17, 4,
          7, 8, 9, 2],
        tokenIds: [20, 21],
        text: 'xy',
        stoppedAtEos: false,
        routerLogits: Float32Array.of(2, 1, 0, -1, -2, -3, -4, -5),
      };
    }

    async close() {
      state.sessionClosed++;
      if (cleanupFailure) throw new Error('fixture session cleanup failed');
    }
  }
  return { api, Session, state };
}

test('fixed workload is byte-exact and emits native raw binding metadata', async () => {
  const image = createTinyReceiptSplitE2EImage();
  assert.equal(image.length, 320 * 672);
  assert.equal(sha256(new Uint8Array(image.buffer)),
    '028acedd12b13cfcd706b8c364f41e82dd80fd34218e61612e31c0c9ad5474fa');
  assert.deepEqual([...image.slice(0, 4)], [-1, -0.8125, -0.625, -0.4375]);

  const bindings = createTinyReceiptSplitE2EBindings({ itos: vocabulary() });
  assert.deepEqual([...bindings.familyIds], [-1]);
  assert.equal(bindings.decoderInputIds[0], 1);
  assert.ok(bindings.decoderInputIds.slice(1).every((value) => value === 0));
  assert.equal(bindings.questionIds.at(bindings.questionTokenIds.length - 1), 2);
  assert.ok(bindings.questionIds.slice(bindings.questionTokenIds.length)
    .every((value) => value === 0));

  const raw = tinyReceiptSplitE2ERawBytes(bindings);
  const paths = {
    image: 'image.f32',
    question_ids: 'question_ids.i32',
    family_ids: 'family_ids.i32',
    decoder_input_ids: 'decoder_input_ids.i32',
  };
  const records = Object.fromEntries(Object.entries(raw).map(([name, bytes]) => [
    name,
    { path: paths[name], bytes: bytes.byteLength, sha256: sha256(bytes) },
  ]));
  const manifest = await createTinyReceiptSplitE2EFixtureManifest(bindings, records);
  assert.equal(manifest.schema, 'volvoxai.tiny-receipt-split-e2e-fixture/v1');
  assert.deepEqual(manifest.tensors.image.shape, [1, 1, 320, 672]);
  assert.equal(manifest.tensors.decoder_input_ids.byteOrder, 'little');
  assert.equal(JSON.stringify(manifest).includes('/home/'), false);
});

test('fixed workload bindings accept deployed BPE tokenization semantics', () => {
  const vocab = {
    itos: Array.from({ length: 1536 }, (_unused, index) => `token-${index}`),
    pad: 0,
    bos: 1,
    encodeQuestion(text, capacity) {
      assert.equal(text, 'phone number last one');
      assert.equal(capacity, 192);
      return [1038, 54, 1124, 54, 1181, 54, 1031, 2];
    },
  };
  const bindings = createTinyReceiptSplitE2EBindings({ vocab });
  assert.deepEqual(bindings.questionTokenIds,
    [1038, 54, 1124, 54, 1181, 54, 1031, 2]);
  assert.deepEqual([...bindings.questionIds.slice(0, 10)],
    [1038, 54, 1124, 54, 1181, 54, 1031, 2, 0, 0]);
});

test('BPE package identity uses the semantic tokenizer hash', async () => {
  const tokenizerHash = '5'.repeat(64);
  const fixture = fakeEnvironment({ tokenizerHash });
  const report = await runTinyReceiptSplitE2E({
    api: fixture.api,
    backend: 'cpu',
    packageUrl: 'https://example.test/package_manifest.json',
    reference: null,
    sessionClass: fixture.Session,
    verifyPackageAssets: false,
  });
  assert.equal(report.package.vocabularySha256, tokenizerHash);
});

test('checked-in split INT8 reference validates as an independent ORT oracle', async () => {
  const reference = JSON.parse(await readFile(
    new URL('../references/split_int8_e2e_ort_cpu.json', import.meta.url),
    'utf8',
  ));
  assert.equal(await validateTinyReceiptSplitE2EReference(reference), reference);
  assert.equal(reference.provenance.kind, 'onnx-runtime-oracle');
  assert.deepEqual(reference.expected.tokenIds, [29, 66, 69, 65]);
  assert.equal(reference.workload.image.sha256,
    '028acedd12b13cfcd706b8c364f41e82dd80fd34218e61612e31c0c9ad5474fa');
  assert.equal(JSON.stringify(reference).includes('/home/'), false);
});

test('reference integrity rejects corrupted router bytes and numeric summaries', async () => {
  const reference = JSON.parse(await readFile(
    new URL('../references/split_int8_e2e_ort_cpu.json', import.meta.url),
    'utf8',
  ));
  const corruptedDigest = structuredClone(reference);
  corruptedDigest.expected.routerLogits.sha256 = '0'.repeat(64);
  await assert.rejects(
    validateTinyReceiptSplitE2EReference(corruptedDigest),
    /hashes or numeric summaries are internally inconsistent/,
  );

  const corruptedSummary = structuredClone(reference);
  corruptedSummary.expected.routerLogits.summary.sum += 0.25;
  await assert.rejects(
    validateTinyReceiptSplitE2EReference(corruptedSummary),
    /hashes or numeric summaries are internally inconsistent/,
  );
});

test('physical adapter gate rejects software identities and matches case-insensitively', () => {
  assert.deepEqual(requireTinyReceiptPhysicalAdapterIdentity({
    vendor: '4318',
    description: 'NVIDIA GeForce RTX 3090',
  }, 'nvidia geforce rtx 3090'), {
    vendor: '4318',
    description: 'NVIDIA GeForce RTX 3090',
  });
  assert.throws(
    () => requireTinyReceiptPhysicalAdapterIdentity({
      description: 'Google SwiftShader Vulkan',
    }, 'swiftshader'),
    /software device/,
  );
  assert.throws(
    () => requireTinyReceiptPhysicalAdapterIdentity({
      description: 'Intel UHD Graphics',
    }, 'RTX 3090'),
    /required identity/,
  );
  assert.throws(
    () => requireTinyReceiptPhysicalAdapterIdentity({
      description: 'NVIDIA GeForce RTX 3090',
    }, '   '),
    /must be non-empty/,
  );
});

test('runner proves strict provider routing, compares a reference, and closes every owner', async () => {
  const first = fakeEnvironment();
  const candidate = await runTinyReceiptSplitE2E({
    api: first.api,
    backend: 'cpu',
    packageUrl: 'https://example.test/package_manifest.json',
    fetch: async () => { throw new Error('fake session owns loading'); },
    sessionClass: first.Session,
    verifyPackageAssets: false,
  });
  assert.deepEqual(first.state, {
    sessionClosed: 1,
    runtimeClosed: 1,
    cacheCleared: 1,
    compileOptions: {
      backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
    },
  });
  const reference = createTinyReceiptSplitE2EReference(candidate, {
    provenance: {
      kind: 'onnx-runtime-oracle',
      sourceFormat: 'tiny_receipt_vqa_split_onnx_v1',
      sourceVariant: 'int8-w8a8',
      provider: 'fixture oracle',
      description: 'Independent fixture values.',
    },
    routerAtol: 0,
    routerRtol: 0,
  });

  const second = fakeEnvironment();
  const verified = await runTinyReceiptSplitE2E({
    api: second.api,
    backend: 'cpu',
    packageUrl: 'https://example.test/package_manifest.json',
    fetch: async () => { throw new Error('fake session owns loading'); },
    sessionClass: second.Session,
    reference,
    verifyPackageAssets: false,
  });
  assert.equal(verified.reference.matched, true);
  assert.equal(verified.provider.decoderExecutions, 2);
  assert.deepEqual(verified.output.tokenIds, [20, 21]);
  assert.deepEqual(second.state, {
    sessionClosed: 1,
    runtimeClosed: 1,
    cacheCleared: 1,
    compileOptions: {
      backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
    },
  });
});

test('runner fails closed on fallback evidence and still closes every owner', async () => {
  const fixture = fakeEnvironment({ fallback: true });
  await assert.rejects(runTinyReceiptSplitE2E({
    api: fixture.api,
    backend: 'cpu',
    packageUrl: 'https://example.test/package_manifest.json',
    fetch: async () => { throw new Error('fake session owns loading'); },
    sessionClass: fixture.Session,
    verifyPackageAssets: false,
  }), /strict no-fallback route/);
  assert.equal(fixture.state.sessionClosed, 1);
  assert.equal(fixture.state.runtimeClosed, 1);
  assert.equal(fixture.state.cacheCleared, 1);
});

test('runner fails closed on mismatched compilation and execution device identities', async () => {
  const cases = [
    [{ candidateDeviceMismatch: true }, /did not strictly select/],
    [{ compilationDeviceMismatch: true }, /selected different device identities/],
    [{ executionDeviceMismatch: true }, /execution device identity differs/],
  ];
  for (const [options, pattern] of cases) {
    const fixture = fakeEnvironment(options);
    await assert.rejects(runTinyReceiptSplitE2E({
      api: fixture.api,
      backend: 'cpu',
      packageUrl: 'https://example.test/package_manifest.json',
      fetch: async () => { throw new Error('fake session owns loading'); },
      sessionClass: fixture.Session,
      verifyPackageAssets: false,
    }), pattern);
    assert.equal(fixture.state.sessionClosed, 1);
    assert.equal(fixture.state.runtimeClosed, 1);
    assert.equal(fixture.state.cacheCleared, 1);
  }
});

test('operation and cleanup failures are both preserved', async () => {
  const fixture = fakeEnvironment({ fallback: true, cleanupFailure: true });
  await assert.rejects(runTinyReceiptSplitE2E({
    api: fixture.api,
    backend: 'cpu',
    packageUrl: 'https://example.test/package_manifest.json',
    fetch: async () => { throw new Error('fake session owns loading'); },
    sessionClass: fixture.Session,
    verifyPackageAssets: false,
  }), (error) => {
    assert.ok(error instanceof AggregateError);
    assert.equal(error.errors.length, 2);
    assert.match(error.errors[0].message, /strict no-fallback route/);
    assert.match(error.errors[1].message, /fixture session cleanup failed/);
    return true;
  });
  assert.equal(fixture.state.sessionClosed, 1);
  assert.equal(fixture.state.runtimeClosed, 1);
  assert.equal(fixture.state.cacheCleared, 1);
});

test('reference mismatch is fatal and cleanup still completes', async () => {
  const seed = fakeEnvironment();
  const candidate = await runTinyReceiptSplitE2E({
    api: seed.api,
    backend: 'cpu',
    packageUrl: 'https://example.test/package_manifest.json',
    fetch: async () => { throw new Error('fake session owns loading'); },
    sessionClass: seed.Session,
    verifyPackageAssets: false,
  });
  const reference = createTinyReceiptSplitE2EReference(candidate, {
    provenance: {
      kind: 'onnx-runtime-oracle',
      sourceFormat: 'tiny_receipt_vqa_split_onnx_v1',
      sourceVariant: 'int8-w8a8',
      provider: 'fixture oracle',
      description: 'Independent fixture values.',
    },
    routerAtol: 0,
    routerRtol: 0,
  });
  reference.expected.tokenIds[1] = 22;
  const mutatedTokenBytes = Buffer.alloc(reference.expected.tokenIds.length * 4);
  reference.expected.tokenIds.forEach((value, index) =>
    mutatedTokenBytes.writeInt32LE(value, index * 4));
  reference.expected.tokenIdsSha256 = sha256(mutatedTokenBytes);
  const fixture = fakeEnvironment();
  await assert.rejects(runTinyReceiptSplitE2E({
    api: fixture.api,
    backend: 'cpu',
    packageUrl: 'https://example.test/package_manifest.json',
    fetch: async () => { throw new Error('fake session owns loading'); },
    sessionClass: fixture.Session,
    reference,
    verifyPackageAssets: false,
  }), /tokenIdsSha256 does not match/);
  assert.equal(fixture.state.sessionClosed, 1);
  assert.equal(fixture.state.runtimeClosed, 1);
  assert.equal(fixture.state.cacheCleared, 1);
});
