#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

import {
  atomicWriteJsonSync,
  PARITY_NATIVE_CAPABILITY_EVIDENCE_SCHEMA,
  PARITY_SCHEMA_VERSION,
  sha256FileSync,
} from './lib/artifact.mjs';
import {
  nativeCapabilityEvidenceExpectation,
  physicalAdapterFromCapabilityLog,
  requireExpectedNativeCapabilityEvidence,
} from './lib/gpu_campaign.mjs';

function usage() {
  console.error(
    'usage: seal_native_capability.mjs <tier> <model> <exit-code> ' +
    '<native.log> <runtime-failure-evidence.json> <capability-skip.json>\n' +
    '   or: seal_native_capability.mjs verify-tier <tier> <native-tier-output>',
  );
}

function seal(argv) {
  if (argv.length !== 6) {
    usage();
    return 2;
  }
  const [
    tier,
    modelName,
    exitCodeText,
    nativeLog,
    runtimeFailureEvidenceFile,
    output,
  ] = argv;
  const exitCode = Number(exitCodeText);
  if (!Number.isSafeInteger(exitCode) || exitCode === 0) {
    throw new Error(`native capability probe exit code is invalid: ${exitCodeText}`);
  }

  const expected = nativeCapabilityEvidenceExpectation(modelName, tier);
  const logText = fs.readFileSync(nativeLog, 'utf8');
  const runtimeFailureEvidence = JSON.parse(
    fs.readFileSync(runtimeFailureEvidenceFile, 'utf8'),
  );
  const adapter = physicalAdapterFromCapabilityLog(
    logText,
    expected.request.backend,
    `${modelName}/${tier} capability producer`,
  );
  const evidence = {
    schema: PARITY_NATIVE_CAPABILITY_EVIDENCE_SCHEMA,
    version: PARITY_SCHEMA_VERSION,
    campaign: expected.campaign,
    authority: expected.authority,
    model: expected.model,
    request: expected.request,
    adapter,
    result: {
      ...expected.result,
      exitCode,
      offendingNode: runtimeFailureEvidence?.report?.offendingNode ?? null,
      message: runtimeFailureEvidence?.report?.message,
    },
    runtimeFailureEvidence,
    runtimeFailureEvidenceSha256: sha256FileSync(runtimeFailureEvidenceFile).sha256,
    nativeLogSha256: sha256FileSync(nativeLog).sha256,
  };

  requireExpectedNativeCapabilityEvidence(evidence, {
    modelName,
    tier,
    nativeLog,
    runtimeFailureEvidenceFile,
  });
  atomicWriteJsonSync(path.resolve(output), evidence);
  console.log(
    `native ${tier} ${modelName}: sealed ${evidence.result.status}/` +
    `${evidence.result.reason} on ${adapter.device}`,
  );
  return 0;
}

function verifyTier(argv) {
  if (argv.length !== 2) {
    usage();
    return 2;
  }
  const [tier, outputRoot] = argv;
  const policy = JSON.parse(fs.readFileSync(new URL('./policy.json', import.meta.url), 'utf8'));
  const names = Object.entries(policy.models ?? {})
    .filter(([, model]) => model.tiers?.includes(tier))
    .map(([name]) => name);
  if (names.length === 0) throw new Error(`no parity models declare tier ${tier}`);
  for (const modelName of names) {
    const dir = path.resolve(outputRoot, modelName);
    const evidence = JSON.parse(fs.readFileSync(path.join(dir, 'capability-skip.json'), 'utf8'));
    requireExpectedNativeCapabilityEvidence(evidence, {
      modelName,
      tier,
      names,
      nativeLog: path.join(dir, 'native.log'),
      runtimeFailureEvidenceFile: path.join(dir, 'runtime-failure-evidence.json'),
    });
  }
  console.log(`native ${tier}: ${names.length} current capability artifacts verified`);
  return 0;
}

function main(argv) {
  if (argv[0] === 'verify-tier') return verifyTier(argv.slice(1));
  return seal(argv);
}

try {
  process.exitCode = main(process.argv.slice(2));
} catch (error) {
  console.error(error?.stack || error);
  process.exitCode = 1;
}
