// Browser- and Node-neutral timing and explicit-cache evidence checks shared by
// the reproducible TinyReceipt benchmark runners.

function fail(message) {
  throw new Error(`[benchmark_explicit_kv_contract] ${message}`);
}

export function finiteMilliseconds(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    fail(`${label} is not a finite non-negative millisecond duration.`);
  }
  return value;
}

export function summarizeDecoderSteps(values) {
  if (!Array.isArray(values) || values.length < 2) {
    fail('a KV-cache benchmark requires at least two decoder executions.');
  }
  const steps = values.map((value, index) => finiteMilliseconds(value, `decoder step ${index}`));
  const steady = steps.slice(1);
  const steadyTotalMs = steady.reduce((sum, value) => sum + value, 0);
  return Object.freeze({
    seedMs: steps[0],
    steadySteps: steady.length,
    steadyTotalMs,
    steadyMeanMs: steadyTotalMs / steady.length,
    steadyTokensPerSecond: steadyTotalMs > 0 ? 1000 * steady.length / steadyTotalMs : null,
    totalMs: steps.reduce((sum, value) => sum + value, 0),
    steps: Object.freeze([...steps]),
  });
}

export function executionPhaseBreakdown(report, label) {
  const executionMs = finiteMilliseconds(report?.executionTimeMs, `${label} execution`);
  const shapeBindMs = finiteMilliseconds(report?.shapeBindTimeMs, `${label} shape bind`);
  const providerMs = finiteMilliseconds(report?.providerTimeMs, `${label} provider`);
  const frameworkMs = executionMs - shapeBindMs - providerMs;
  if (frameworkMs < -0.05) {
    fail(`${label} phase timings exceed total execution time.`);
  }
  const specializationFields = {
    writeCount: report?.backendReport?.specializationWriteCount,
    writeBytes: report?.backendReport?.specializationWriteBytes,
    skipCount: report?.backendReport?.specializationWriteSkipCount,
    skipBytes: report?.backendReport?.specializationWriteSkipBytes,
    contentBytes: report?.backendReport?.specializationContentBytes,
  };
  const specializationValues = Object.values(specializationFields);
  const hasSpecializationEvidence = specializationValues.some((value) => value !== undefined);
  if (hasSpecializationEvidence && specializationValues.some((value) =>
    !Number.isSafeInteger(value) || value < 0)) {
    fail(`${label} has incomplete or invalid specialization-write evidence.`);
  }
  return Object.freeze({
    executionMs,
    shapeBindMs,
    providerMs,
    frameworkMs: Math.max(0, frameworkMs),
    specializationCacheHit: report?.backendReport?.specializationCacheHit === true,
    ...(hasSpecializationEvidence
      ? { specializationWrites: Object.freeze(specializationFields) }
      : {}),
  });
}

export function validateDynamicRebindAnswer(answer, maximumNewTokens, expectedShapeMode) {
  if (expectedShapeMode !== 'active' && expectedShapeMode !== 'maximum-padded') {
    fail("expected shape mode must be 'active' or 'maximum-padded'.");
  }
  if (answer?.execution !== 'explicit-kv-cache' || answer.decodeMode !== 'explicit-kv-cache') {
    fail('session did not use the explicit KV-cache v2 execution path.');
  }
  const tokens = answer.tokenIds;
  if (!Array.isArray(tokens) || tokens.length < 2 || tokens.length > maximumNewTokens ||
      tokens.some((value) => !Number.isInteger(value) || value < 0)) {
    fail('session did not emit a valid two-or-more-step token sequence.');
  }
  if (answer.decoderSeedExecutions !== 1 ||
      answer.decoderOrdinaryExecutions !== tokens.length ||
      answer.decoderCacheStepExecutions !== tokens.length - 1) {
    fail('session execution counters do not prove ordinary one-token explicit-cache calls.');
  }
  const activeShape = answer.activeShape;
  const logicalShape = answer.logicalShape;
  if (answer.shapeMode !== expectedShapeMode || logicalShape?.B !== 1 ||
      !Number.isInteger(logicalShape.Q) || logicalShape.Q < 1 || logicalShape.Q > 192 ||
      logicalShape.M !== logicalShape.Q + 210 || logicalShape.T !== maximumNewTokens + 1 ||
      !Array.isArray(answer.questionTokenIds) ||
      answer.questionTokenIds.length !== logicalShape.Q) {
    fail('session did not expose a bounded logical Q/M/T shape.');
  }
  if (expectedShapeMode === 'active') {
    if (activeShape?.B !== logicalShape.B || activeShape.Q !== logicalShape.Q ||
        activeShape.M !== logicalShape.M || activeShape.T !== logicalShape.T) {
      fail('session did not bind the exact active Q/M/T shape.');
    }
  } else if (activeShape?.B !== 1 || activeShape.Q !== 192 || activeShape.M !== 402 ||
             activeShape.T !== logicalShape.T) {
    fail('session did not bind the maximum legal encoder Q/M shape.');
  }
  if (answer.cacheShape?.initialPastLength !== 1 ||
      answer.cacheShape.finalPastLength !== tokens.length + 1 ||
      answer.cacheShape.sentinelSlots !== 1 ||
      !Array.isArray(answer.decodeReports) || answer.decodeReports.length !== tokens.length) {
    fail('session cache-shape evidence is incomplete.');
  }
  for (const [index, report] of answer.decodeReports.entries()) {
    if (report?.operation !== (index === 0 ? 'explicit-kv-seed' : 'explicit-kv-step') ||
        report.position !== index || report.pastLength !== index + 1 ||
        report.presentLength !== index + 2 || report.sentinelMaskValue !== 1) {
      fail(`session cache transition ${index} is not P=${index + 1} to R=${index + 2}.`);
    }
  }
  return Object.freeze([...tokens]);
}

export function validateExplicitKVAnswer(answer, maximumNewTokens) {
  return validateDynamicRebindAnswer(answer, maximumNewTokens, 'active');
}
