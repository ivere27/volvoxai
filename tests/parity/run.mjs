#!/usr/bin/env node
// Cross-tier parity orchestration. Transient results are valid only through a
// finalized producer manifest whose policy/model/input/build/source fingerprint
// still matches the current checkout. Producers invalidate their selected old
// manifest and artifacts before doing work, so a failed rerun cannot reuse them.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';

import { readTensor } from './lib/tensorio.mjs';
import { compareSignature, signature } from './lib/extract.mjs';
import { buildInputs, ROOT, runJs, runPackage, version } from './lib/runmodel.mjs';
import {
  atomicWriteJsonSync,
  createParityFingerprintSync,
  createRunManifest,
  finalizeRunManifest,
  readRunArtifactSync,
  readRunManifestSync,
  recordRunResult,
  removeParityOutputsSync,
  sha256FileSync,
  validateNativeRuntimeEvidence,
  validateParityFingerprint,
  writeRunArtifactSync,
  writeRunManifestSync,
} from './lib/artifact.mjs';
import {
  requirePhysicalAdapterIdentity,
  requirePhysicalNativeAdapterIdentity,
} from './lib/backend.mjs';
import {
  GPU_CONSENSUS_MODELS,
  REQUIRED_GPU_TIERS,
  gpuConsensusFingerprint,
  kvCacheCampaignFingerprint,
  requireExactConsensusModels,
  requireFiniteFloat32Buffers,
  requireExactPhysicalAdapterSet,
} from './lib/gpu_campaign.mjs';
import { authorCase, cases } from './ops/cases.mjs';
import { authorGraph, cases as graphCases } from './graphs/cases.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const GOLD = path.join(HERE, 'goldens');
const OUT = path.join(HERE, 'out');
const MANIFESTS = path.join(OUT, 'manifests');
const POLICY_FILE = path.join(HERE, 'policy.json');
const policy = JSON.parse(fs.readFileSync(POLICY_FILE, 'utf8'));
const MATRIX_TIERS = ['cpu', 'wasm', 'native-cpu', 'webgpu', 'native-vulkan', 'native-opengl', 'torch'];

function product(shape) {
  if (!Array.isArray(shape) || shape.length === 0) throw new Error(`invalid logical shape: ${shape}`);
  return shape.reduce((n, dim) => n * dim, 1);
}

function selectedPolicyModels(only) {
  if (only && !policy.models[only]) throw new Error(`unknown parity model: ${only}`);
  return Object.entries(policy.models).filter(([name]) => !only || name === only);
}

function fixtureFile(name, input) {
  const ext = input.dtype === 'f32' ? 'f32' : input.dtype;
  return path.join(OUT, 'fixtures', name, `${input.name}.${ext}`);
}

function generatedFixtureFiles(names) {
  return names.flatMap((name) => (policy.models[name].inputs || [])
    .filter((input) => input.gen)
    .map((input) => fixtureFile(name, input)));
}

function materializeFixtures(names) {
  for (const name of names) {
    buildInputs(policy.models[name], path.join(OUT, 'fixtures', name));
  }
  return generatedFixtureFiles(names);
}

function distFile(name) {
  return path.join(ROOT, 'dist', version(), name);
}

function wholeBuildFiles(tier) {
  if (tier === 'cpu') return [distFile('volvoxai.js')];
  if (tier === 'wasm') return [distFile('volvoxai.js'), distFile('volvoxai.wasm')];
  if (tier === 'webgpu') return [distFile('volvoxai.js')];
  if (tier.startsWith('native-')) {
    const binary = path.join(ROOT, 'native', 'volvoxai');
    return fs.existsSync(binary) ? [binary] : [];
  }
  throw new Error(`unknown whole-model build tier: ${tier}`);
}

function wholeFingerprint(tier, names) {
  return createParityFingerprintSync({
    selectedCases: names,
    fixtureFiles: generatedFixtureFiles(names),
    buildFiles: wholeBuildFiles(tier),
  });
}

function referenceFingerprint(name) {
  return createParityFingerprintSync({
    selectedCases: [name],
    fixtureFiles: generatedFixtureFiles([name]),
    source: { revision: null, state: 'reference-inputs', statusSha256: null },
  });
}

function wholeManifestFile(tier) {
  return path.join(MANIFESTS, `whole-${tier}.json`);
}

function wholeArtifactFile(name, tier) {
  return path.join(OUT, `${name}.${tier}.json`);
}

function goldenManifestFile(name) {
  return path.join(GOLD, 'manifests', `${name}.json`);
}

function goldenArtifactFile(name) {
  return path.join(GOLD, `${name}.json`);
}

function atomicWriteText(file, content) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temporary, content, { encoding: 'utf8', flag: 'wx', mode: 0o600 });
    fs.renameSync(temporary, file);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

function errorText(error) {
  return String(error?.message || error).split('\n')[0].slice(0, 500);
}

function recordError(manifest, caseId, tier, error) {
  recordRunResult(manifest, { case: caseId, tier, status: 'error', error: errorText(error) });
}

function finishManifest(manifest, file, options = {}) {
  finalizeRunManifest(manifest);
  writeRunManifestSync(file, manifest, options);
  return manifest.outcome;
}

function assertSignatureValid(sig, label) {
  const result = compareSignature(sig, sig, { rtol: 0, atol: 0, statsGate: true });
  if (!result.pass) throw new Error(`${label}: ${result.problems.join('; ')}`);
}

function sigForOutput(spec, flat) {
  if (!ArrayBuffer.isView(flat)) throw new Error(`${spec.name}: output is not a typed array`);
  let values = flat;
  if (spec.kind === 'row') {
    const expectedFull = spec.rows * spec.cols;
    if (flat.length !== expectedFull) {
      throw new Error(`${spec.name}: output has ${flat.length} elements, expected ${expectedFull}`);
    }
    values = flat.subarray(spec.row * spec.cols, (spec.row + 1) * spec.cols);
  }
  const expected = product(spec.shape);
  if (values.length !== expected) {
    throw new Error(`${spec.name}: selected output has ${values.length} elements, expected ${expected}`);
  }
  return signature(values, {
    topk: spec.topk ?? 10,
    shape: spec.shape,
    sampleAxes: spec.sampleAxes ?? [],
  });
}

async function runTierJs(name, model, backend) {
  const fixtureDir = path.join(OUT, 'fixtures', name);
  const { outputs, ms, runtimeEvidence } = await runJs(model, backend, fixtureDir);
  const sigs = {};
  for (const spec of model.outputs) sigs[spec.name] = sigForOutput(spec, outputs[spec.name]);
  return { model: name, backend, ms: Number(ms.toFixed(2)), sigs, runtimeEvidence };
}

async function cmdGolden(only) {
  const entries = selectedPolicyModels(only);
  materializeFixtures(entries.map(([name]) => name));
  let failures = 0;
  for (const [name, model] of entries) {
    const artifactFile = goldenArtifactFile(name);
    const manifestFile = goldenManifestFile(name);
    removeParityOutputsSync([artifactFile, manifestFile], { outputRoot: GOLD });
    const manifest = createRunManifest({
      command: 'golden',
      fingerprint: wholeFingerprint('cpu', [name]),
      jobs: [{ case: name, tier: 'cpu-reference', expectation: 'required' }],
      producer: {
        kind: 'node', backend: 'cpu', strictBackend: true, runtimeEvidenceRequired: true,
      },
    });
    process.stdout.write(`golden ${name} (strict cpu reference)… `);
    try {
      const result = await runTierJs(name, model, 'cpu');
      writeRunArtifactSync({
        manifest,
        file: artifactFile,
        case: name,
        tier: 'cpu-reference',
        payload: {
          model: name,
          dtype: model.dtype,
          referenceFingerprint: referenceFingerprint(name),
          sigs: result.sigs,
          runtimeEvidence: result.runtimeEvidence,
        },
        outputRoot: GOLD,
        durationMs: result.ms,
        metadata: { runtimeEvidence: result.runtimeEvidence },
      });
      finishManifest(manifest, manifestFile, { outputRoot: GOLD });
      console.log('ok');
    } catch (error) {
      failures++;
      console.log(`ERROR ${errorText(error)}`);
    }
  }
  if (failures) throw new Error(`${failures} golden model(s) failed; old selected goldens were invalidated`);
}

async function cmdProduce(tier, only) {
  if (!['cpu', 'wasm'].includes(tier)) throw new Error('produce handles only cpu|wasm; use native-sig for native tiers');
  const entries = selectedPolicyModels(only).filter(([, model]) => model.tiers.includes(tier));
  if (entries.length === 0) throw new Error(`no policy models selected for tier ${tier}`);
  const names = entries.map(([name]) => name);
  materializeFixtures(names);
  const manifestFile = wholeManifestFile(tier);
  removeParityOutputsSync([manifestFile, ...names.map((name) => wholeArtifactFile(name, tier))]);
  const manifest = createRunManifest({
    command: `produce ${tier}`,
    fingerprint: wholeFingerprint(tier, names),
    jobs: entries.map(([name, model]) => ({
      case: name,
      tier,
      expectation: model.gate.includes(tier) ? 'required' : 'expected-skip',
    })),
    producer: {
      kind: 'node', backend: tier, strictBackend: true, runtimeEvidenceRequired: true,
    },
  });
  for (const [name, model] of entries) {
    process.stdout.write(`produce ${tier} ${name}… `);
    try {
      const result = await runTierJs(name, model, tier);
      writeRunArtifactSync({
        manifest,
        file: wholeArtifactFile(name, tier),
        case: name,
        tier,
        payload: result,
        durationMs: result.ms,
        metadata: { runtimeEvidence: result.runtimeEvidence },
      });
      console.log(`ok (${result.ms} ms)`);
    } catch (error) {
      recordError(manifest, name, tier, error);
      console.log(`ERROR ${errorText(error)}`);
    }
  }
  if (finishManifest(manifest, manifestFile) !== 'success') {
    throw new Error(`${tier} producer did not complete every required job`);
  }
}

function nativeExpectedBackend(tier) {
  return { 'native-cpu': 'cpu', 'native-vulkan': 'vulkan', 'native-opengl': 'opengl' }[tier];
}

function readNativeRuntimeEvidence(file, expectedBackend) {
  if (!fs.existsSync(file)) throw new Error(`missing native runtime evidence: ${file}`);
  const evidence = JSON.parse(fs.readFileSync(file, 'utf8'));
  return validateNativeRuntimeEvidence(evidence, expectedBackend);
}

function cmdNativeSig(args) {
  const requireSelected = args[0] === '--required';
  const tiers = requireSelected ? args.slice(1) : args;
  if (tiers.some((tier) => tier.startsWith('-'))) {
    throw new Error('native-sig usage: native-sig [--required] [tier ...]');
  }
  if (requireSelected && tiers.length === 0) {
    throw new Error('native-sig --required needs at least one explicit native tier');
  }
  const selectedTiers = tiers.length ? [...new Set(tiers)] : ['native-cpu'];
  const selections = selectedTiers.map((tier) => {
    const expectedBackend = nativeExpectedBackend(tier);
    if (!expectedBackend) throw new Error(`unsupported native signature tier: ${tier}`);
    const entries = Object.entries(policy.models).filter(([, model]) => model.tiers.includes(tier));
    if (entries.length === 0) throw new Error(`no policy models declare ${tier}`);
    return { tier, expectedBackend, entries };
  });
  let failed = 0;
  for (const { tier, expectedBackend, entries } of selections) {
    const names = entries.map(([name]) => name);
    const manifestFile = wholeManifestFile(tier);
    removeParityOutputsSync([manifestFile, ...names.map((name) => wholeArtifactFile(name, tier))]);
    const manifest = createRunManifest({
      command: `native-sig${requireSelected ? ' --required' : ''} ${tier}`,
      fingerprint: wholeFingerprint(tier, names),
      jobs: entries.map(([name, model]) => ({
        case: name,
        tier,
        expectation: requireSelected || model.gate.includes(tier) ? 'required' : 'expected-skip',
      })),
      producer: {
        kind: 'native-fold',
        backend: tier,
        selectedBackend: expectedBackend,
        selectedTierRequired: requireSelected,
        strictBackend: true,
        runtimeEvidenceRequired: true,
        physicalAdapterRequired: tier !== 'native-cpu',
      },
    });
    const base = path.join(OUT, 'native', tier);
    for (const [name, model] of entries) {
      const dir = path.join(base, name);
      const job = manifest.selection.jobs.find((candidate) => candidate.case === name && candidate.tier === tier);
      if (!fs.existsSync(dir)) {
        if (job.expectation === 'expected-skip') {
          recordRunResult(manifest, { case: name, tier, status: 'expected-skip', reason: 'current native producer emitted no model directory' });
        } else {
          recordError(manifest, name, tier, new Error('current native producer emitted no model directory'));
        }
        continue;
      }
      try {
        const runtimeEvidence = readNativeRuntimeEvidence(
          path.join(dir, 'runtime-evidence.json'),
          expectedBackend,
        );
        let adapterInfo = null;
        if (tier !== 'native-cpu') {
          const adapterFile = path.join(dir, 'adapter.json');
          if (!fs.existsSync(adapterFile)) throw new Error('missing native physical adapter evidence');
          adapterInfo = requirePhysicalNativeAdapterIdentity(
            JSON.parse(fs.readFileSync(adapterFile, 'utf8')),
            expectedBackend,
            `whole-model ${name}/${tier}`,
          );
        }
        const sigs = {};
        for (const spec of model.outputs) {
          const file = path.join(dir, `${spec.name}.f32`);
          if (!fs.existsSync(file)) throw new Error(`missing native output ${spec.name}.f32`);
          sigs[spec.name] = sigForOutput(spec, readTensor(file, 'f32'));
        }
        const msFile = path.join(dir, 'ms.txt');
        const ms = fs.existsSync(msFile) ? Number(fs.readFileSync(msFile, 'utf8').trim()) : null;
        if (ms != null && (!Number.isFinite(ms) || ms < 0)) throw new Error(`invalid native duration: ${ms}`);
        writeRunArtifactSync({
          manifest,
          file: wholeArtifactFile(name, tier),
          case: name,
          tier,
          payload: {
            model: name,
            backend: tier,
            ms,
            sigs,
            runtimeEvidence,
            ...(adapterInfo ? { adapterInfo } : {}),
          },
          durationMs: ms ?? undefined,
          metadata: {
            selectedBackend: expectedBackend,
            runtimeEvidence,
            ...(adapterInfo ? { adapterInfo } : {}),
          },
        });
        console.log(`native-sig ${tier} ${name}: ok`);
      } catch (error) {
        recordError(manifest, name, tier, error);
        console.error(`native-sig ${tier} ${name}: ERROR ${errorText(error)}`);
      }
    }
    if (finishManifest(manifest, manifestFile) !== 'success') failed++;
  }
  if (failed) throw new Error(`${failed} native tier manifest(s) contain required errors`);
}

function readRawManifestSelection(file) {
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (!Array.isArray(raw?.selection?.cases)) throw new Error(`manifest has no selected cases: ${file}`);
  return raw.selection.cases;
}

function loadWholeManifest(tier) {
  const file = wholeManifestFile(tier);
  if (!fs.existsSync(file)) return null;
  const names = readRawManifestSelection(file);
  return readRunManifestSync(file, {
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: wholeFingerprint(tier, names),
  });
}

function loadGolden(name) {
  const manifest = readRunManifestSync(goldenManifestFile(name), {
    outputRoot: GOLD,
    requireComplete: true,
    requireFinalized: true,
  });
  if (manifest.outcome !== 'success') throw new Error(`golden manifest failed for ${name}`);
  const payload = readRunArtifactSync(goldenArtifactFile(name), manifest, {
    case: name,
    tier: 'cpu-reference',
    outputRoot: GOLD,
  });
  if (payload.referenceFingerprint?.digest !== referenceFingerprint(name).digest) {
    throw new Error(`golden reference fingerprint is stale for ${name}; run: make parity_goldens`);
  }
  return payload;
}

function externalEntries() {
  const entries = [];
  for (const [name, model] of Object.entries(policy.models)) {
    if (model.external?.onnx) entries.push({ name, model, tier: 'onnx', file: path.join(OUT, `${name}.onnx.json`) });
    else if (model.external?.source === 'pytorch') entries.push({ name, model, tier: 'torch', file: path.join(OUT, `${name}.torch.json`) });
  }
  return entries;
}

function externalFingerprint() {
  const entries = externalEntries();
  const buildFiles = [
    path.join(HERE, 'external', 'onnx_oracle.py'),
    path.join(HERE, 'external', 'tinystories_torch_oracle.py'),
    path.join(HERE, 'external', 'sigutil.py'),
  ];
  const extraModelFiles = entries
    .map(({ model }) => model.external?.onnx)
    .filter(Boolean)
    .map((file) => path.join(ROOT, file));
  return createParityFingerprintSync({
    selectedCases: entries.map(({ name }) => name),
    fixtureFiles: generatedFixtureFiles(entries.map(({ name }) => name)),
    buildFiles,
    extraModelFiles,
  });
}

const EXTERNAL_MANIFEST = path.join(MANIFESTS, 'whole-external.json');
const EXTERNAL_PENDING = path.join(OUT, 'external.pending.json');

function cmdExternalBegin() {
  const entries = externalEntries();
  materializeFixtures(entries.map(({ name }) => name));
  removeParityOutputsSync([EXTERNAL_MANIFEST, EXTERNAL_PENDING, ...entries.map(({ file }) => file)]);
  atomicWriteJsonSync(EXTERNAL_PENDING, {
    schema: 'volvoxai.parity-pending',
    version: 1,
    fingerprint: externalFingerprint().digest,
    startedAt: new Date().toISOString(),
  });
  console.log('external oracle outputs invalidated; producer campaign started');
}

function cmdExternalSig() {
  if (!fs.existsSync(EXTERNAL_PENDING)) throw new Error('external-sig requires a fresh external-begin campaign');
  const pending = JSON.parse(fs.readFileSync(EXTERNAL_PENDING, 'utf8'));
  const fingerprint = externalFingerprint();
  if (pending?.schema !== 'volvoxai.parity-pending' || pending.fingerprint !== fingerprint.digest) {
    throw new Error('external pending fingerprint is stale; rerun external-begin and the oracles');
  }
  const entries = externalEntries();
  removeParityOutputsSync([EXTERNAL_MANIFEST]);
  const manifest = createRunManifest({
    command: 'external-sig',
    fingerprint,
    jobs: entries.map(({ name, tier }) => ({ case: name, tier, expectation: 'expected-skip' })),
    producer: { kind: 'external-fold', optional: true },
  });
  for (const entry of entries) {
    if (!fs.existsSync(entry.file)) {
      recordRunResult(manifest, { case: entry.name, tier: entry.tier, status: 'expected-skip', reason: 'optional oracle did not publish a current artifact' });
      continue;
    }
    try {
      const payload = JSON.parse(fs.readFileSync(entry.file, 'utf8'));
      if (payload?.model !== entry.name || payload?.backend !== entry.tier) {
        throw new Error(`oracle identity is ${payload?.model}/${payload?.backend}, expected ${entry.name}/${entry.tier}`);
      }
      for (const spec of entry.model.outputs) {
        if (!payload.sigs?.[spec.name]) throw new Error(`oracle omitted signature ${spec.name}`);
        assertSignatureValid(payload.sigs[spec.name], `${entry.name}/${spec.name}`);
      }
      writeRunArtifactSync({ manifest, file: entry.file, case: entry.name, tier: entry.tier, payload });
      console.log(`external-sig ${entry.name}/${entry.tier}: ok`);
    } catch (error) {
      recordError(manifest, entry.name, entry.tier, error);
      console.error(`external-sig ${entry.name}/${entry.tier}: ERROR ${errorText(error)}`);
    }
  }
  finishManifest(manifest, EXTERNAL_MANIFEST);
  removeParityOutputsSync([EXTERNAL_PENDING]);
  if (manifest.outcome !== 'success') throw new Error('external oracle folding failed');
}

function loadExternalManifest() {
  if (!fs.existsSync(EXTERNAL_MANIFEST)) return null;
  return readRunManifestSync(EXTERNAL_MANIFEST, {
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: externalFingerprint(),
  });
}

function comparePayload(name, model, tier, candidate, gold) {
  const baseTolerance = ['onnx', 'torch'].includes(tier)
    ? policy.tolerances.external
    : policy.tolerances[model.dtype];
  let worstAbs = 0;
  let worstRel = 0;
  let top1 = true;
  let pass = true;
  const notes = [];
  if (candidate?.model !== name || candidate?.backend !== tier) {
    return { pass: false, worstAbs, worstRel, top1: false, notes: [`artifact identity is ${candidate?.model}/${candidate?.backend}`] };
  }
  for (const spec of model.outputs) {
    const tolerance = {
      ...baseTolerance,
      ...(model.toleranceOverrides?.[tier]?.[spec.name] ?? {}),
    };
    const result = compareSignature(candidate.sigs?.[spec.name], gold.sigs?.[spec.name], tolerance);
    worstAbs = Math.max(worstAbs, result.numeric.maxAbs);
    worstRel = Math.max(worstRel, result.numeric.maxRel);
    if (result.task.top1Match === false) top1 = false;
    if (!result.pass) {
      pass = false;
      notes.push(`${spec.name}: ${result.problems.join('; ')}`);
    }
    if (result.notes?.length) notes.push(`${spec.name} (diag): ${result.notes.join('; ')}`);
  }
  return { pass, worstAbs, worstRel, top1, notes };
}

function cmdCompare() {
  const rows = [];
  const bench = [];
  let failed = 0;
  let ran = 0;
  const manifestCache = new Map();
  let externalManifest = null;
  try { externalManifest = loadExternalManifest(); } catch (error) {
    failed++;
    rows.push({ name: 'external', tier: 'manifest', status: `INVALID: ${errorText(error)}`, gate: false });
  }
  for (const [name, model] of Object.entries(policy.models)) {
    let gold;
    try { gold = loadGolden(name); } catch (error) {
      failed++;
      rows.push({ name, tier: 'golden', status: `INVALID: ${errorText(error)}`, gate: true });
      continue;
    }
    for (const tier of model.tiers) {
      const policyRequired = model.gate.includes(tier);
      if (!manifestCache.has(tier)) {
        try { manifestCache.set(tier, loadWholeManifest(tier)); } catch (error) {
          manifestCache.set(tier, error);
        }
      }
      const manifest = manifestCache.get(tier);
      const selectedJob = manifest && !(manifest instanceof Error)
        ? manifest.selection.jobs.find((job) => job.case === name && job.tier === tier)
        : null;
      const required = policyRequired || selectedJob?.expectation === 'required';
      if (!manifest || manifest instanceof Error) {
        const status = manifest instanceof Error ? `INVALID: ${errorText(manifest)}` : 'not run';
        rows.push({ name, tier, status, gate: required });
        if (required || manifest instanceof Error) failed++;
        continue;
      }
      const result = manifest.results.find((entry) => entry.case === name && entry.tier === tier);
      if (!result) {
        rows.push({ name, tier, status: 'missing from current manifest', gate: required });
        if (required) failed++;
        continue;
      }
      if (result.status !== 'success') {
        rows.push({ name, tier, status: result.status === 'expected-skip' ? `SKIP: ${result.reason}` : `ERROR: ${result.error}`, gate: required });
        if (required || result.status === 'error') failed++;
        continue;
      }
      try {
        const candidate = readRunArtifactSync(wholeArtifactFile(name, tier), manifest, { case: name, tier });
        const compared = comparePayload(name, model, tier, candidate, gold);
        const diagnostic = model.diagnosticTiers?.includes(tier) === true;
        ran++;
        if (candidate.ms != null) bench.push({ name, tier, ms: candidate.ms });
        if (!compared.pass && !diagnostic) failed++;
        rows.push({
          name,
          tier,
          status: compared.pass ? 'PASS' : diagnostic ? 'DIAGNOSTIC' : 'FAIL',
          maxAbs: compared.worstAbs,
          maxRel: compared.worstRel,
          top1: compared.top1,
          gate: required,
          notes: compared.notes,
        });
      } catch (error) {
        failed++;
        rows.push({ name, tier, status: `INVALID: ${errorText(error)}`, gate: required });
      }
    }
    const ext = externalEntries().find((entry) => entry.name === name);
    if (ext && externalManifest) {
      const result = externalManifest.results.find((entry) => entry.case === name && entry.tier === ext.tier);
      if (result?.status === 'success') {
        try {
          const candidate = readRunArtifactSync(ext.file, externalManifest, { case: name, tier: ext.tier });
          const compared = comparePayload(name, model, ext.tier, candidate, gold);
          const diagnostic = model.diagnosticTiers?.includes(ext.tier) === true;
          ran++;
          if (!compared.pass && !diagnostic) failed++;
          rows.push({
            name,
            tier: ext.tier,
            status: compared.pass ? 'PASS' : diagnostic ? 'DIAGNOSTIC' : 'FAIL',
            maxAbs: compared.worstAbs,
            maxRel: compared.worstRel,
            top1: compared.top1,
            gate: false,
            notes: compared.notes,
          });
        } catch (error) {
          failed++;
          rows.push({ name, tier: ext.tier, status: `INVALID: ${errorText(error)}`, gate: false });
        }
      } else if (result) {
        rows.push({ name, tier: ext.tier, status: `SKIP: ${result.reason || result.error}`, gate: false });
      }
    }
  }

  const lines = [
    '# Cross-tier parity report',
    '',
    `oracle: strict pure-JS CPU golden. compared ${ran} current tier-runs.`,
    '',
    '| model | tier | status | maxAbs | maxRel | top1 | required |',
    '|---|---|---|---|---|---|---|',
  ];
  for (const row of rows) {
    lines.push(`| ${row.name} | ${row.tier} | ${row.status} | ${fmt(row.maxAbs)} | ${fmt(row.maxRel)} | ${row.top1 ?? ''} | ${row.gate ? 'yes' : ''} |`);
  }
  if (bench.length) {
    lines.push('', '## Benchmark (warm forward, best-effort)', '', '| model | tier | ms |', '|---|---|---|');
    for (const item of bench) lines.push(`| ${item.name} | ${item.tier} | ${item.ms} |`);
  }
  const details = rows.filter((row) => row.notes?.length);
  if (details.length) {
    lines.push('', '## Details');
    for (const row of details) lines.push(`- **${row.name}/${row.tier}** — ${row.notes.join(' | ')}`);
  }
  atomicWriteText(path.join(OUT, 'report.md'), `${lines.join('\n')}\n`);
  atomicWriteJsonSync(path.join(OUT, 'report.json'), { rows, bench, failed });
  console.log(lines.join('\n'));
  if (ran === 0) throw new Error('no current tier artifacts were compared');
  if (failed) throw new Error(`${failed} parity integrity or numeric check(s) failed`);
  console.log('\nparity OK');
}

function fmt(value) {
  return value == null ? '' : (typeof value === 'number' ? value.toExponential(2) : value);
}

function matrixConfig(kind) {
  if (kind === 'ops') return { kind, list: cases, author: authorCase, packages: path.join(OUT, 'opcases'), signatures: path.join(OUT, 'ops'), report: path.join(OUT, 'op_matrix.md') };
  if (kind === 'graphs') return { kind, list: graphCases, author: authorGraph, packages: path.join(OUT, 'graphcases'), signatures: path.join(OUT, 'graphsigs'), report: path.join(OUT, 'graph_matrix.md') };
  throw new Error(`matrix kind must be ops|graphs, got ${kind}`);
}

function matrixManifestFile(kind, tier) {
  return path.join(MANIFESTS, `${kind}-${tier}.json`);
}

function matrixArtifactFile(config, id, tier) {
  return path.join(config.signatures, `${id}.${tier}.json`);
}

function walkFiles(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...walkFiles(file));
    else if (entry.isFile()) files.push(file);
  }
  return files.sort();
}

function matrixBuildFiles(tier) {
  if (tier === 'cpu') return [distFile('volvoxai.js')];
  if (tier === 'wasm') return [distFile('volvoxai.js'), distFile('volvoxai.wasm')];
  if (tier === 'webgpu') return [distFile('volvoxai.js')];
  if (tier.startsWith('native-')) {
    const binary = path.join(ROOT, 'native', 'volvoxai');
    return fs.existsSync(binary) ? [binary] : [];
  }
  if (tier === 'torch') return [path.join(HERE, 'graphs', 'torch_interp.py'), path.join(HERE, 'external', 'sigutil.py')];
  throw new Error(`unknown matrix tier: ${tier}`);
}

function matrixFingerprint(config, tier) {
  if (!fs.existsSync(config.packages)) throw new Error(`matrix packages missing: ${config.packages}`);
  // Fingerprint only the authored package contract. Native runners place final
  // and staging readbacks beside it; neither is a model/input component.
  const files = walkFiles(config.packages).filter((file) => {
    const parts = path.relative(config.packages, file).split(path.sep);
    return parts.length >= 2 && (
      ['graph.json', 'meta.json', 'model.safetensors'].includes(parts[1]) ||
      parts[1] === 'inputs' || parts[1] === 'weights'
    );
  });
  return createParityFingerprintSync({
    selectedCases: [],
    fixtureFiles: files.filter((file) => file.split(path.sep).includes('inputs')),
    extraModelFiles: files.filter((file) => !file.split(path.sep).includes('inputs')),
    buildFiles: matrixBuildFiles(tier),
  });
}

function matrixJobs(config, tier, { nativeAvailable = true } = {}) {
  return config.list.map((item) => {
    const declared = item.skip?.includes(tier);
    const optional = tier === 'torch' || (tier === 'native-cpu' && !nativeAvailable);
    return { case: item.id, tier, expectation: declared || optional ? 'expected-skip' : 'required' };
  });
}

function matrixShape(config, item) {
  const meta = JSON.parse(fs.readFileSync(path.join(config.packages, item.id, 'meta.json'), 'utf8'));
  if (!Array.isArray(meta.outputShape)) throw new Error(`${item.id}: package metadata has no outputShape`);
  return meta.outputShape;
}

function assertNativeRoute(stdout, expected) {
  const routes = String(stdout).split(/\r?\n/).filter((line) => line.startsWith('Backend:'));
  if (routes.length !== 1 || routes[0].trim() !== `Backend: ${expected}`) {
    throw new Error(`native route evidence is '${routes.join(', ') || 'missing'}', expected 'Backend: ${expected}'`);
  }
}

async function cmdMatrix(kind) {
  const config = matrixConfig(kind);
  const manifestFiles = MATRIX_TIERS.map((tier) => matrixManifestFile(kind, tier));
  removeParityOutputsSync([config.packages, config.signatures, ...manifestFiles], { allowDirectories: true });
  const authored = new Map();
  for (const item of config.list) authored.set(item.id, config.author(path.join(config.packages, item.id), item).inputs);
  const nativeBin = path.join(ROOT, 'native', 'volvoxai');
  const haveNative = fs.existsSync(nativeBin);
  const tiers = ['cpu', 'wasm', 'native-cpu'];
  const manifests = new Map(tiers.map((tier) => [tier, createRunManifest({
    command: `${kind}-matrix ${tier}`,
    fingerprint: matrixFingerprint(config, tier),
    jobs: matrixJobs(config, tier, { nativeAvailable: haveNative }),
    producer: tier === 'native-cpu'
      ? {
        kind: 'native',
        backend: 'native-cpu',
        selectedBackend: 'cpu',
        strictBackend: true,
        runtimeEvidenceRequired: true,
      }
      : {
        kind: 'node', backend: tier, strictBackend: true, runtimeEvidenceRequired: true,
      },
  })]));

  for (const item of config.list) {
    const packageDir = path.join(config.packages, item.id);
    const shape = matrixShape(config, item);
    for (const tier of ['cpu', 'wasm']) {
      const manifest = manifests.get(tier);
      if (item.skip?.includes(tier)) {
        recordRunResult(manifest, { case: item.id, tier, status: 'expected-skip', reason: 'declared unsupported in case policy' });
        continue;
      }
      try {
        const { outputs, runtimeEvidence } = await runPackage(
          packageDir,
          tier,
          authored.get(item.id),
        );
        writeRunArtifactSync({
          manifest,
          file: matrixArtifactFile(config, item.id, tier),
          case: item.id,
          tier,
          payload: {
            id: item.id,
            tier,
            sig: signature(outputs.y, { topk: 0, shape }),
            runtimeEvidence,
          },
          metadata: { runtimeEvidence },
        });
      } catch (error) {
        recordError(manifest, item.id, tier, error);
        console.error(`  ${kind} ${item.id} ${tier}: ${errorText(error)}`);
      }
    }

    const nativeManifest = manifests.get('native-cpu');
    let nativeEvidenceFile = null;
    if (!haveNative || item.skip?.includes('native-cpu')) {
      recordRunResult(nativeManifest, {
        case: item.id,
        tier: 'native-cpu',
        status: 'expected-skip',
        reason: haveNative ? 'declared unsupported in case policy' : 'native binary not built',
      });
    } else {
      try {
        const meta = JSON.parse(fs.readFileSync(path.join(packageDir, 'meta.json'), 'utf8'));
        const args = ['run', packageDir];
        for (const input of meta.inputs) args.push('--input', `${input}=${findMatrixInput(packageDir, input)}`);
        const outputFile = path.join(packageDir, 'native-cpu_y.f32');
        nativeEvidenceFile = path.join(
          packageDir,
          `.native-cpu-runtime-${process.pid}-${randomUUID()}.json`,
        );
        fs.rmSync(outputFile, { force: true });
        args.push(
          '--output', `y=${outputFile}`,
          '--cpu',
          '--report-json', nativeEvidenceFile,
        );
        const stdout = execFileSync(nativeBin, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
        assertNativeRoute(stdout, 'cpu');
        const runtimeEvidence = readNativeRuntimeEvidence(nativeEvidenceFile, 'cpu');
        const values = readTensor(outputFile, 'f32');
        if (values.length !== product(shape)) throw new Error(`native output has ${values.length} elements, expected ${product(shape)}`);
        writeRunArtifactSync({
          manifest: nativeManifest,
          file: matrixArtifactFile(config, item.id, 'native-cpu'),
          case: item.id,
          tier: 'native-cpu',
          payload: {
            id: item.id,
            tier: 'native-cpu',
            sig: signature(values, { topk: 0, shape }),
            runtimeEvidence,
          },
          metadata: { selectedBackend: 'cpu', runtimeEvidence },
        });
      } catch (error) {
        recordError(nativeManifest, item.id, 'native-cpu', error);
        console.error(`  ${kind} ${item.id} native-cpu: ${errorText(error)}`);
      } finally {
        if (nativeEvidenceFile) fs.rmSync(nativeEvidenceFile, { force: true });
      }
    }
    console.log(`${kind} ${item.id}: produced`);
  }

  let failed = 0;
  for (const [tier, manifest] of manifests) {
    if (finishManifest(manifest, matrixManifestFile(kind, tier)) !== 'success') failed++;
  }
  if (failed) throw new Error(`${failed} ${kind} producer manifest(s) contain required errors`);
}

function findMatrixInput(packageDir, name) {
  for (const ext of ['f32', 'i32', 'i8', 'u8']) {
    const file = path.join(packageDir, 'inputs', `${name}.${ext}`);
    if (fs.existsSync(file)) return file;
  }
  throw new Error(`no input file for ${name}`);
}

function cmdMatrixBegin(kind, tier) {
  if (!MATRIX_TIERS.includes(tier) || ['cpu', 'wasm', 'native-cpu'].includes(tier)) {
    throw new Error('matrix-begin tier must be webgpu|native-vulkan|native-opengl|torch');
  }
  const config = matrixConfig(kind);
  if (!fs.existsSync(config.packages)) throw new Error(`run ${kind === 'ops' ? 'opmatrix' : 'graphmatrix'} first`);
  removeParityOutputsSync([
    matrixManifestFile(kind, tier),
    ...config.list.map((item) => matrixArtifactFile(config, item.id, tier)),
  ]);
  console.log(`${kind}/${tier}: previous selected outputs invalidated`);
}

function cmdMatrixImport(kind, tier) {
  if (!MATRIX_TIERS.includes(tier) || ['cpu', 'wasm', 'native-cpu'].includes(tier)) {
    throw new Error('matrix-import tier must be webgpu|native-vulkan|native-opengl|torch');
  }
  const config = matrixConfig(kind);
  removeParityOutputsSync([matrixManifestFile(kind, tier)]);
  const manifest = createRunManifest({
    command: `${kind}-matrix import ${tier}`,
    fingerprint: matrixFingerprint(config, tier),
    jobs: matrixJobs(config, tier),
    producer: tier === 'torch'
      ? { kind: 'python', backend: 'torch' }
      : tier === 'webgpu'
        ? {
          kind: 'deno',
          backend: 'webgpu',
          strictBackend: true,
          physicalAdapterRequired: true,
          runtimeEvidenceRequired: true,
        }
        : {
          kind: 'native',
          backend: tier,
          selectedBackend: nativeExpectedBackend(tier),
          strictBackend: true,
          physicalAdapterRequired: true,
          runtimeEvidenceRequired: true,
        },
  });
  for (const item of config.list) {
    const file = matrixArtifactFile(config, item.id, tier);
    const declaredSkip = item.skip?.includes(tier);
    if (!fs.existsSync(file)) {
      if (declaredSkip || tier === 'torch') {
        recordRunResult(manifest, { case: item.id, tier, status: 'expected-skip', reason: declaredSkip ? 'declared unsupported in case policy' : 'optional Torch interpreter did not map this case' });
      } else {
        recordError(manifest, item.id, tier, new Error('requested producer emitted no current artifact'));
      }
      continue;
    }
    try {
      const payload = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (payload?.schema === 'volvoxai.parity-artifact') throw new Error('artifact was not regenerated after matrix-begin');
      if (payload?.id !== item.id || payload?.tier !== tier) throw new Error(`artifact identity is ${payload?.id}/${payload?.tier}`);
      if (tier === 'webgpu' && payload.backend !== 'webgpu') {
        throw new Error(`WebGPU artifact backend identity is '${payload.backend || 'missing'}'`);
      }
      if (tier.startsWith('native-') && payload.backend !== nativeExpectedBackend(tier)) {
        throw new Error(`native artifact backend identity is '${payload.backend || 'missing'}', expected '${nativeExpectedBackend(tier)}'`);
      }
      if (tier === 'webgpu') {
        requirePhysicalAdapterIdentity(payload.adapterInfo, `${kind} matrix ${item.id}/webgpu`);
      } else if (tier === 'native-vulkan' || tier === 'native-opengl') {
        requirePhysicalNativeAdapterIdentity(
          payload.adapterInfo,
          nativeExpectedBackend(tier),
          `${kind} matrix ${item.id}/${tier}`,
        );
        validateNativeRuntimeEvidence(
          payload.runtimeEvidence,
          nativeExpectedBackend(tier),
        );
      }
      assertSignatureValid(payload.sig, `${kind}/${item.id}/${tier}`);
      writeRunArtifactSync({
        manifest,
        file,
        case: item.id,
        tier,
        payload,
        ...(tier === 'webgpu' || tier.startsWith('native-')
          ? { metadata: { runtimeEvidence: payload.runtimeEvidence } }
          : {}),
      });
    } catch (error) {
      recordError(manifest, item.id, tier, error);
    }
  }
  finishManifest(manifest, matrixManifestFile(kind, tier));
  if (manifest.outcome !== 'success') throw new Error(`${kind}/${tier} import is incomplete or invalid`);
  console.log(`${kind}/${tier}: current artifacts sealed by manifest`);
}

function loadMatrixManifest(config, tier) {
  const file = matrixManifestFile(config.kind, tier);
  if (!fs.existsSync(file)) return null;
  return readRunManifestSync(file, {
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: matrixFingerprint(config, tier),
  });
}

function cmdMatrixCompare(kind) {
  const config = matrixConfig(kind);
  const manifests = new Map();
  let failed = 0;
  for (const tier of MATRIX_TIERS) {
    try { manifests.set(tier, loadMatrixManifest(config, tier)); } catch (error) {
      manifests.set(tier, error);
      failed++;
    }
  }
  for (const required of ['cpu', 'wasm', 'native-cpu']) {
    if (!manifests.get(required)) {
      manifests.set(required, new Error(`missing current ${required} manifest`));
      failed++;
    }
  }
  const rows = [];
  let ran = 0;
  for (const item of config.list) {
    const row = { label: kind === 'ops' ? item.op : item.id, cells: {} };
    const cpuManifest = manifests.get('cpu');
    let reference = null;
    if (!(cpuManifest instanceof Error) && cpuManifest) {
      try {
        reference = readRunArtifactSync(matrixArtifactFile(config, item.id, 'cpu'), cpuManifest, { case: item.id, tier: 'cpu' }).sig;
      } catch (error) {
        failed++;
        row.cells.cpu = `INVALID ${errorText(error)}`;
      }
    }
    for (const tier of MATRIX_TIERS.filter((value) => value !== 'cpu')) {
      const manifest = manifests.get(tier);
      if (manifest instanceof Error) { row.cells[tier] = `INVALID ${errorText(manifest)}`; continue; }
      if (!manifest) { row.cells[tier] = '—'; continue; }
      const result = manifest.results.find((entry) => entry.case === item.id && entry.tier === tier);
      if (!result || result.status !== 'success') {
        row.cells[tier] = result?.status === 'expected-skip' ? 'n/a' : `ERROR ${result?.error || 'missing result'}`;
        if (result?.status === 'error' || (!result && tier !== 'torch')) failed++;
        continue;
      }
      if (!reference) { row.cells[tier] = 'no cpu reference'; failed++; continue; }
      try {
        const candidate = readRunArtifactSync(matrixArtifactFile(config, item.id, tier), manifest, { case: item.id, tier }).sig;
        const approximate = item.approxTol && tier !== 'torch';
        const tolerance = tier === 'torch'
          ? policy.tolerances.external
          : approximate ? { ...policy.tolerances.fp32, ...item.approxTol } : policy.tolerances.fp32;
        const compared = compareSignature(candidate, reference, tolerance);
        ran++;
        row.cells[tier] = `${compared.pass ? (approximate ? '≈' : 'ok') : 'FAIL'} ${compared.numeric.maxAbs.toExponential(1)}`;
        if (!compared.pass) failed++;
      } catch (error) {
        failed++;
        row.cells[tier] = `INVALID ${errorText(error)}`;
      }
    }
    rows.push(row);
  }
  const columns = ['wasm', 'native-cpu', 'webgpu', 'native-vulkan', 'native-opengl', 'torch'];
  const lines = [
    kind === 'ops' ? '# Per-op cross-tier matrix (Level 1)' : '# Mixed-graph cross-tier matrix (Level 2)',
    '',
    'oracle: strict pure-JS CPU. Every requested non-Torch tier is a numeric gate; Torch is optional but gates when present.',
    '',
    `| ${kind === 'ops' ? 'op' : 'graph'} | wasm | native-cpu | webgpu | native-vk | native-gl | torch (ext) |`,
    '|---|---|---|---|---|---|---|',
  ];
  for (const row of rows) lines.push(`| ${row.label} | ${columns.map((tier) => row.cells[tier] ?? '—').join(' | ')} |`);
  atomicWriteText(config.report, `${lines.join('\n')}\n`);
  console.log(lines.join('\n'));
  if (ran === 0) throw new Error(`no current ${kind} tier results were compared`);
  if (failed) throw new Error(`${failed} ${kind} integrity or numeric check(s) failed`);
  console.log(`\n${kind} parity OK`);
}

function cmdCoverage() {
  const registryFile = path.join(ROOT, 'docs', 'generated', 'kernel-registry.md');
  const markdown = fs.readFileSync(registryFile, 'utf8');
  const operatorHeading = /^## Operator matrix[ \t]*\r?$/m.exec(markdown);
  if (!operatorHeading) throw new Error(`${registryFile}: missing Operator matrix section`);
  const nextHeading = /^## [^\r\n]+[ \t]*\r?$/gm;
  nextHeading.lastIndex = operatorHeading.index + operatorHeading[0].length;
  const operatorMatrix = markdown.slice(
    nextHeading.lastIndex,
    nextHeading.exec(markdown)?.index ?? markdown.length,
  );
  const allOps = [...new Set([...operatorMatrix.matchAll(/^\| `([A-Za-z0-9_]+)`/gm)].map((match) => match[1]))].sort();
  if (allOps.length === 0) throw new Error(`${registryFile}: Operator matrix contains no operator rows`);
  const level1 = new Set(cases.map((item) => item.op));
  const level2 = new Set(graphCases.flatMap((item) => item.nodes.map((node) => node.opType)));
  const level3 = new Set();
  for (const model of Object.values(policy.models)) {
    const graph = path.join(ROOT, model.dir, 'graph.json');
    if (fs.existsSync(graph)) {
      for (const node of JSON.parse(fs.readFileSync(graph, 'utf8')).nodes || []) {
        if (node.opType) level3.add(node.opType);
      }
    }
  }
  const covered = new Set([...level1, ...level2, ...level3]);
  const isolated = new Set([...level1, ...level2]);
  const uncovered = allOps.filter((op) => !covered.has(op));
  const percent = ((allOps.filter((op) => covered.has(op)).length / allOps.length) * 100).toFixed(0);
  const lines = [
    '# Operator parity coverage',
    '',
    `**${allOps.filter((op) => covered.has(op)).length} / ${allOps.length} operators (${percent}%)** exercised by an isolated parity case (L1/L2) or a whole-model test (L3).`,
    '',
    `- isolated parity cases (L1 op + L2 graph): ${[...isolated].sort().join(', ') || '—'}`,
    `- additionally covered only via whole-model tests (L3): ${allOps.filter((op) => level3.has(op) && !isolated.has(op)).join(', ') || '—'}`,
    '',
    `## Not yet covered (${uncovered.length})`,
    '',
    uncovered.join(', ') || '(none)',
  ];
  atomicWriteText(path.join(OUT, 'coverage.md'), `${lines.join('\n')}\n`);
  console.log(lines.join('\n'));
}

function assertExactSuccessfulJobs(manifest, expectedJobs, label) {
  if (!manifest || manifest.outcome !== 'success') throw new Error(`${label}: current manifest did not succeed`);
  const expected = new Map(expectedJobs.map((job) => [`${job.case}\0${job.tier}`, job]));
  const selected = new Map(manifest.selection.jobs.map((job) => [`${job.case}\0${job.tier}`, job]));
  if (selected.size !== expected.size) {
    throw new Error(`${label}: selected ${selected.size} jobs, expected ${expected.size}`);
  }
  for (const [key, job] of expected) {
    const actual = selected.get(key);
    if (!actual || actual.expectation !== job.expectation) {
      throw new Error(`${label}: selection mismatch for ${job.case}/${job.tier}`);
    }
    const result = manifest.results.find((candidate) =>
      candidate.case === job.case && candidate.tier === job.tier);
    const requiredStatus = job.expectation === 'required' ? 'success' : 'expected-skip';
    if (result?.status !== requiredStatus) {
      throw new Error(`${label}: ${job.case}/${job.tier} is '${result?.status || 'missing'}', expected '${requiredStatus}'`);
    }
  }
}

function cmdGpuBegin() {
  const selectedModels = Object.keys(policy.models);
  const wholeOutputs = REQUIRED_GPU_TIERS.flatMap((tier) => [
    wholeManifestFile(tier),
    ...selectedModels.map((name) => wholeArtifactFile(name, tier)),
  ]);
  const matrixOutputs = ['ops', 'graphs'].flatMap((kind) => {
    const config = matrixConfig(kind);
    return REQUIRED_GPU_TIERS.flatMap((tier) => [
      matrixManifestFile(kind, tier),
      ...config.list.map((item) => matrixArtifactFile(config, item.id, tier)),
    ]);
  });
  removeParityOutputsSync([
    path.join(OUT, 'gpu_required_summary.json'),
    path.join(OUT, 'gpu_consensus'),
    path.join(OUT, 'native', 'native-vulkan'),
    path.join(OUT, 'native', 'native-opengl'),
    ...wholeOutputs,
    ...matrixOutputs,
  ], { allowDirectories: true });

  const kvOut = path.join(HERE, 'kvcache', 'out');
  removeParityOutputsSync([
    'reference.json', 'cpu.json', 'wasm.json', 'webgpu.json', 'run.json',
    'kvcache_matrix.md', 'kvcache_matrix.json',
  ], { outputRoot: kvOut });
  console.log('required GPU campaign: previous selected evidence invalidated');
}

function validateConsensusSummary() {
  const root = path.join(OUT, 'gpu_consensus');
  const file = path.join(root, 'summary.json');
  if (!fs.existsSync(file)) throw new Error('GPU consensus summary is missing');
  const summary = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (summary?.schema !== 'volvoxai.gpu-consensus-summary' || summary.version !== 3 || summary.passed !== true) {
    throw new Error('GPU consensus summary is invalid or did not pass');
  }
  const fingerprint = validateParityFingerprint(summary.fingerprint);
  const expectedFingerprint = gpuConsensusFingerprint();
  if (fingerprint.digest !== expectedFingerprint.digest) {
    throw new Error('GPU consensus summary fingerprint is stale');
  }
  requireExactConsensusModels(summary.models);
  for (const entry of summary.models) {
    requireExactPhysicalAdapterSet(entry.adapters, `GPU consensus ${entry.model}`);
    const expected = [];
    for (const [backend, prefix] of [['webgpu', 'wg'], ['native-opengl', 'gl'], ['native-vulkan', 'vk']]) {
      for (const output of ['scores', 'boxes']) {
        expected.push({ backend, output, path: `${entry.model}/${prefix}_${output}.f32` });
      }
    }
    if (!Array.isArray(entry.outputs) || entry.outputs.length !== expected.length) {
      throw new Error(`GPU consensus ${entry.model}: wrong output selection`);
    }
    const buffers = new Map();
    const expectedElements = { scores: 19206 * 90, boxes: 19206 * 4 };
    for (const wanted of expected) {
      const record = entry.outputs.find((candidate) =>
        candidate.backend === wanted.backend && candidate.output === wanted.output);
      if (!record || record.path !== wanted.path) {
        throw new Error(`GPU consensus ${entry.model}: missing ${wanted.backend}/${wanted.output}`);
      }
      const actual = sha256FileSync(path.join(root, wanted.path));
      if (record.sha256 !== actual.sha256 || record.size !== actual.size) {
        throw new Error(`GPU consensus ${entry.model}: output digest mismatch for ${wanted.backend}/${wanted.output}`);
      }
      if (actual.size !== expectedElements[wanted.output] * 4) {
        throw new Error(`GPU consensus ${entry.model}: wrong logical size for ${wanted.backend}/${wanted.output}`);
      }
      buffers.set(`${wanted.backend}\0${wanted.output}`, fs.readFileSync(path.join(root, wanted.path)));
    }
    for (const output of ['scores', 'boxes']) {
      const backends = ['webgpu', 'native-opengl', 'native-vulkan'];
      const selected = backends
        .map((backend) => buffers.get(`${backend}\0${output}`));
      const values = requireFiniteFloat32Buffers(
        selected,
        backends,
        `GPU consensus ${entry.model}/${output}`,
      );
      if (entry.tolerance === 0) {
        if (!selected[0].equals(selected[1]) || !selected[0].equals(selected[2])) {
          throw new Error(`GPU consensus ${entry.model}: exact ${output} bytes differ`);
        }
        continue;
      }
      for (const [left, right] of [[0, 1], [0, 2], [1, 2]]) {
        for (let i = 0; i < values[left].length; i++) {
          if (Math.abs(values[left][i] - values[right][i]) > entry.tolerance) {
            throw new Error(`GPU consensus ${entry.model}: ${output} exceeds tolerance at ${i}`);
          }
        }
      }
    }
  }
  return {
    digest: sha256FileSync(file),
    fingerprint: fingerprint.digest,
    models: GPU_CONSENSUS_MODELS.map((entry) => entry.model),
  };
}

function validateKvCacheCampaign() {
  const outputRoot = path.join(HERE, 'kvcache', 'out');
  const manifestFile = path.join(outputRoot, 'run.json');
  const manifest = readRunManifestSync(manifestFile, {
    outputRoot,
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: kvCacheCampaignFingerprint(),
  });
  const jobs = ['reference', 'cpu', 'wasm', 'webgpu'].map((tier) => ({
    case: 'w8a8-kvcache', tier, expectation: 'required',
  }));
  assertExactSuccessfulJobs(manifest, jobs, 'KV-cache GPU campaign');
  const payloads = {};
  for (const tier of ['reference', 'cpu', 'wasm', 'webgpu']) {
    const payload = readRunArtifactSync(`${tier}.json`, manifest, {
      case: 'w8a8-kvcache', tier, outputRoot,
    });
    if (tier !== 'reference' && payload?.backend !== tier) {
      throw new Error(`KV-cache ${tier}: artifact backend identity mismatch`);
    }
    if (payload?.cache == null || typeof payload.cache !== 'object') {
      throw new Error(`KV-cache ${tier}: cache payload is missing`);
    }
    for (const name of ['self.k', 'self.v']) {
      const values = payload.cache[name];
      if (!Array.isArray(values) || values.length !== 12 ||
          values.some((value) => !Number.isInteger(value) || value < -128 || value > 127)) {
        throw new Error(`KV-cache ${tier}: ${name} is not an exact 12-value int8 cache`);
      }
    }
    if (tier === 'webgpu') {
      payload.adapterInfo = requirePhysicalAdapterIdentity(payload.adapterInfo, 'KV-cache WebGPU artifact');
    }
    payloads[tier] = payload;
  }
  const reference = ['self.k', 'self.v'].flatMap((name) => payloads.reference.cache[name]);
  for (const tier of ['cpu', 'wasm', 'webgpu']) {
    const actual = ['self.k', 'self.v'].flatMap((name) => payloads[tier].cache[name]);
    if (!actual.every((value, index) => value === reference[index])) {
      throw new Error(`KV-cache ${tier}: cache differs from full recompute`);
    }
  }
  const matrix = JSON.parse(fs.readFileSync(path.join(outputRoot, 'kvcache_matrix.json'), 'utf8'));
  if (matrix?.runId !== manifest.runId || matrix.passed !== true ||
      matrix.fingerprint !== manifest.fingerprint.digest ||
      JSON.stringify(matrix.requiredBackends) !== JSON.stringify(['cpu', 'wasm', 'webgpu'])) {
    throw new Error('KV-cache GPU comparison summary is missing or does not require cpu/wasm/webgpu');
  }
  const matrixAdapter = requirePhysicalAdapterIdentity(
    matrix.webgpuAdapterInfo,
    'KV-cache comparison WebGPU adapter',
  );
  if (JSON.stringify(matrixAdapter) !== JSON.stringify(payloads.webgpu.adapterInfo)) {
    throw new Error('KV-cache WebGPU adapter evidence differs between artifact and comparison summary');
  }
  return {
    runId: manifest.runId,
    fingerprint: manifest.fingerprint.digest,
    matrixDigest: sha256FileSync(path.join(outputRoot, 'kvcache_matrix.json')),
    webgpuAdapterInfo: payloads.webgpu.adapterInfo,
  };
}

function cmdGpuVerify() {
  const summaryFile = path.join(OUT, 'gpu_required_summary.json');
  removeParityOutputsSync([summaryFile]);
  const whole = {};
  for (const tier of REQUIRED_GPU_TIERS) {
    const entries = Object.entries(policy.models).filter(([, model]) => model.tiers.includes(tier));
    const manifest = loadWholeManifest(tier);
    assertExactSuccessfulJobs(manifest, entries.map(([name]) => ({
      case: name, tier, expectation: 'required',
    })), `whole-model ${tier}`);
    const adapters = [];
    for (const [name] of entries) {
      const payload = readRunArtifactSync(wholeArtifactFile(name, tier), manifest, { case: name, tier });
      if (payload?.model !== name || payload?.backend !== tier) {
        throw new Error(`whole-model ${name}/${tier}: artifact identity mismatch`);
      }
      adapters.push(tier === 'webgpu'
        ? requirePhysicalAdapterIdentity(payload.adapterInfo, `whole-model ${name}/webgpu`)
        : requirePhysicalNativeAdapterIdentity(
          payload.adapterInfo,
          nativeExpectedBackend(tier),
          `whole-model ${name}/${tier}`,
        ));
    }
    whole[tier] = {
      runId: manifest.runId,
      fingerprint: manifest.fingerprint.digest,
      requiredJobs: entries.length,
      adapters,
    };
  }

  const matrices = {};
  for (const kind of ['ops', 'graphs']) {
    const config = matrixConfig(kind);
    matrices[kind] = {};
    for (const tier of REQUIRED_GPU_TIERS) {
      const manifest = loadMatrixManifest(config, tier);
      const jobs = matrixJobs(config, tier);
      assertExactSuccessfulJobs(manifest, jobs, `${kind} matrix ${tier}`);
      let requiredJobs = 0;
      let declaredSkips = 0;
      const adapters = [];
      for (const job of jobs) {
        if (job.expectation === 'expected-skip') { declaredSkips++; continue; }
        requiredJobs++;
        const payload = readRunArtifactSync(matrixArtifactFile(config, job.case, tier), manifest, {
          case: job.case, tier,
        });
        adapters.push(tier === 'webgpu'
          ? requirePhysicalAdapterIdentity(payload.adapterInfo, `${kind} matrix ${job.case}/webgpu`)
          : requirePhysicalNativeAdapterIdentity(
            payload.adapterInfo,
            nativeExpectedBackend(tier),
            `${kind} matrix ${job.case}/${tier}`,
          ));
      }
      matrices[kind][tier] = {
        runId: manifest.runId,
        fingerprint: manifest.fingerprint.digest,
        requiredJobs,
        declaredSkips,
        adapters,
      };
    }
  }

  const consensus = validateConsensusSummary();
  const kvcache = validateKvCacheCampaign();
  atomicWriteJsonSync(summaryFile, {
    schema: 'volvoxai.required-gpu-campaign',
    version: 1,
    passed: true,
    generatedAt: new Date().toISOString(),
    requiredAdapter: globalThis.process?.env?.VOLVOXAI_PARITY_GPU_ADAPTER ||
      globalThis.process?.env?.VOLVOXAI_PARITY_WEBGPU_ADAPTER || null,
    whole,
    matrices,
    consensus,
    kvcache,
  });
  console.log(`required GPU campaign verified -> ${path.relative(ROOT, summaryFile)}`);
}

const [command, ...args] = process.argv.slice(2);
try {
  if (command === 'golden') await cmdGolden(args[0]);
  else if (command === 'produce') await cmdProduce(args[0], args[1]);
  else if (command === 'fixtures') {
    materializeFixtures(Object.keys(policy.models));
    console.log('generated input fixtures under tests/parity/out/fixtures/');
  } else if (command === 'native-sig') cmdNativeSig(args);
  else if (command === 'external-begin') cmdExternalBegin();
  else if (command === 'external-sig') cmdExternalSig();
  else if (command === 'compare') cmdCompare();
  else if (command === 'opmatrix') await cmdMatrix('ops');
  else if (command === 'graphmatrix') await cmdMatrix('graphs');
  else if (command === 'matrix-begin') cmdMatrixBegin(args[0], args[1]);
  else if (command === 'matrix-import') cmdMatrixImport(args[0], args[1]);
  else if (command === 'opmatrix-compare') cmdMatrixCompare('ops');
  else if (command === 'graphmatrix-compare') cmdMatrixCompare('graphs');
  else if (command === 'coverage') cmdCoverage();
  else if (command === 'gpu-begin') cmdGpuBegin();
  else if (command === 'gpu-verify') cmdGpuVerify();
  else {
    console.error('usage: run.mjs golden [model] | produce <cpu|wasm> [model] | fixtures | native-sig [--required] [tier ...] | external-begin|external-sig|compare | opmatrix|graphmatrix | matrix-begin <ops|graphs> <tier> | matrix-import <ops|graphs> <tier> | opmatrix-compare|graphmatrix-compare|coverage | gpu-begin|gpu-verify');
    process.exit(2);
  }
} catch (error) {
  console.error('parity harness error:', error?.stack || error);
  process.exit(2);
}
