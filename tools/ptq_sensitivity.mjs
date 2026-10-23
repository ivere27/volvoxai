const sqnr = (signal, noise) => noise > 0 && signal > 0 ? 10 * Math.log10(signal / noise) : null;

function floats(tensor, p) {
  if (tensor.dtype !== p.DataType.DATA_TYPE_F32) throw new Error(`output ${tensor.name} is not F32`);
  const bytes = tensor.inline;
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
function check(value, what) {
  const report = value?.report ?? value;
  if (report?.status) throw new Error(`${what}: ${report.message || `status ${report.status}`}`);
  return value;
}

/** Runs each evaluation batch and returns {outputName: [Float32Array per batch]}. */
async function evaluate(inference, p, runtimeId, pkg, evaluation, outputs, packedArithmetic) {
  const model = check(await inference.loadModel(new p.LoadModelRequest({runtimeId,
    package: new p.ModelPackage({graphDocument: pkg.graph, weightShards: pkg.weights})})), 'LoadModel');
  let compiled, context;
  const values = Object.fromEntries(outputs.map(name => [name, []]));
  try {
    compiled = check(await inference.compileModel(new p.CompileModelRequest({
      modelId: model.modelId, packedArithmetic})), 'CompileModel');
    context = check(await inference.createExecutionContext(new p.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, graph: new p.GraphExecution()})), 'CreateExecutionContext');
    for (const inputs of evaluation) {
      const result = check(await inference.execute(new p.ExecuteRequest({contextId: context.contextId,
        inputs: new p.InputBindings({values: inputs}),
        scope: new p.ExecutionScope({entrypoint: 'evaluate', full: new p.Empty()}),
        outputs: new p.OutputOptions({selection: new p.TensorOutputSelection({names: outputs})}),
      })), 'Execute');
      for (const name of outputs) values[name].push(floats(result.outputs.find(tensor => tensor.name === name), p));
    }
  } finally {
    try {
      if (context) await inference.releaseExecutionContext(new p.ExecutionContextRef({contextId: context.contextId}));
    } finally {
      try {
        if (compiled) await inference.releaseCompiledModel(new p.CompiledModelRef({compiledModelId: compiled.compiledModelId}));
      } finally { await inference.releaseModel(new p.ModelRef({modelId: model.modelId})); }
    }
  }
  return values;
}

/** SQNR of every output over all evaluation batches, and the worst output. */
function compare(reference, candidate) {
  const perOutput = {};
  for (const [name, batches] of Object.entries(reference)) {
    let signal = 0, noise = 0;
    batches.forEach((x, batch) => {
      const y = candidate[name][batch];
      for (let i = 0; i < x.length; i++) {
        const d = x[i] - y[i];
        signal += x[i] * x[i];
        noise += Number.isFinite(d) ? d * d : Infinity;
      }
    });
    perOutput[name] = sqnr(signal, noise);
  }
  const values = Object.values(perOutput).filter(value => value !== null);
  return {perOutput, sqnrDb: values.length ? Math.min(...values) : null};
}

async function quantizedPackage(quantization, p, source, config, calibration, allowUnusedCalibration = false) {
  const plan = check(await quantization.createQuantizationPlan(new p.CreateQuantizationPlanRequest({config: config, bytes: new p.ModelBytes({graph: source.graph, weightShards: source.weights})})), 'CreateQuantizationPlan');
  try {
    if (plan.info.requiresCalibration) {
      if (!Array.isArray(calibration) || !calibration.length)
        throw new Error('the resolved quantization plan requires calibration batches');
      for (let i = 0; i < calibration.length; i++)
        check(await quantization.calibrateQuantizationPlan(new p.CalibrateQuantizationPlanRequest({
          quantizationPlanId: plan.quantizationPlanId, sampleName: `sample-${i}`, sampleCount: 1n,
          inputs: calibration[i],
        })), 'CalibrateQuantizationPlan');
    } else if (calibration != null && !allowUnusedCalibration) {
      throw new Error('calibration batches are unnecessary for the resolved plan');
    }
    if (!plan.info.quantized.some(item => item.nodes > 0n)) return {info: plan.info, pkg: null};
    const written = await quantization.exportQuantizedModel(new p.ExportQuantizedModelRequest(plan));
    // A selected region without a materializable layer remains an unmeasured row.
    if (written.report?.code === p.OperationCode.OPERATION_CODE_PTQ_PACKAGE_WRITE_FAILED)
      return {info: plan.info, pkg: null, note: 'the package writer refused this selection'};
    check(written, 'ExportQuantizedModel');
    return {info: written.plan, pkg: {graph: written.bytes.graph, weights: written.bytes.weightShards}};
  } finally {
    await quantization.releaseQuantizationPlan(new p.QuantizationPlanRef(plan));
  }
}

/**
 * source: {graph: Uint8Array, weights: Uint8Array[]}; evaluation and static
 * calibration: arrays of Tensor lists; outputs: F32 graph output names.
 * config: QuantizationConfig, including explicit weight and activation policies.
 */
export async function ptqSensitivity({inference, quantization, p, runtimeId, source, calibration, evaluation,
  outputs, config, packedArithmetic = p.PackedArithmetic.PACKED_ARITHMETIC_F32, modes = ['isolated', 'leave-one-float'],
  higherWeightBits = 4}) {
  if (!Array.isArray(modes) || !modes.length || new Set(modes).size !== modes.length ||
      modes.some(mode => !['isolated', 'leave-one-float', 'leave-one-higher'].includes(mode)))
    throw new Error('modes must select distinct supported sensitivity sweeps');
  if (![3, 4, 8].includes(higherWeightBits)) throw new Error('higherWeightBits must be 3, 4 or 8');
  const make = extra => new p.QuantizationConfig({...config, ...extra});
  const full = await quantizedPackage(quantization, p, source, make({}), calibration);
  if (!full.pkg) throw new Error('authoring quantized no node');
  const reference = await evaluate(inference, p, runtimeId, source, evaluation, outputs, packedArithmetic);
  const baseline = compare(reference, await evaluate(inference, p, runtimeId, full.pkg, evaluation, outputs, packedArithmetic));
  const resolved = new Map(full.info.resolvedNodes.filter(item => item.weights).map(item => [item.nodeId, item]));
  const nodes = [...resolved.keys()];
  const rows = [];
  for (const node of nodes) {
    const row = {node};
    for (const mode of modes) {
      const weights = resolved.get(node).weights;
      const higher = new p.WeightQuantization({...weights,
        format: new p.WeightFormat({...weights.format, bits: higherWeightBits})});
      const extra = {rules: mode === 'isolated'
        ? [new p.QuantizationRule({nodeId: node, weights: resolved.get(node).weights}),
           new p.QuantizationRule({nodeGlob: '*', keepFloat: new p.Empty()})]
        : [new p.QuantizationRule(mode === 'leave-one-higher' ? {nodeId: node, weights: higher}
          : {nodeId: node, keepFloat: new p.Empty()}), ...(config.rules ?? [])]};
      const {pkg, note} = mode === 'leave-one-higher' && weights.format.bits >= higherWeightBits
        ? {pkg: null, note: 'node already uses this weight precision or higher'}
        : mode === 'leave-one-float' && nodes.length === 1
        ? {pkg: null, note: 'nothing left to quantize'}
        : await quantizedPackage(quantization, p, source, make(extra), calibration, true);
      const measured = pkg ? compare(reference, await evaluate(inference, p, runtimeId, pkg, evaluation, outputs, packedArithmetic))
        : {sqnrDb: null, perOutput: {}, note: note ?? 'nothing left to quantize'};
      row[{'isolated': 'isolated', 'leave-one-float': 'leaveOneFloat', 'leave-one-higher': 'leaveOneHigher'}[mode]] = measured;
    }
    if (row.leaveOneFloat?.sqnrDb != null && baseline.sqnrDb != null)
      row.recoveryDb = row.leaveOneFloat.sqnrDb - baseline.sqnrDb;
    if (row.leaveOneHigher?.sqnrDb != null && baseline.sqnrDb != null)
      row.higherRecoveryDb = row.leaveOneHigher.sqnrDb - baseline.sqnrDb;
    rows.push(row);
  }
  const byIsolated = [...rows].filter(row => row.isolated?.sqnrDb != null)
    .sort((a, b) => a.isolated.sqnrDb - b.isolated.sqnrDb).map(row => row.node);
  const byRecovery = [...rows].filter(row => row.recoveryDb != null)
    .sort((a, b) => b.recoveryDb - a.recoveryDb).map(row => row.node);
  const report = {format: 'volvoxai-ptq-sensitivity/v1', outputs, evaluationBatches: evaluation.length,
    calibrationBatches: full.info.requiresCalibration ? calibration.length : 0,
    configuration: p.QuantizationConfig.fromBinary(make({}).toBinary()).toJson(), packedArithmetic: p.PackedArithmetic[packedArithmetic], baseline, rows,
    mostSensitive: byIsolated, bestToKeepFloat: byRecovery,
    interpretation: 'SQNR of graph outputs against the float model, worst output per package. ' +
      'Isolated ranks the error a node adds alone; recoveryDb is the SQNR regained by keeping ' +
      'that node in F32 while everything else is quantized. Use keep_float rules for the chosen node IDs.'};
  if (modes.includes('leave-one-higher')) {
    report.higherWeightBits = higherWeightBits;
    report.bestToKeepHigher = rows.filter(row => row.higherRecoveryDb != null)
      .sort((a, b) => b.higherRecoveryDb - a.higherRecoveryDb).map(row => row.node);
    report.interpretation += ' higherRecoveryDb measures raising one node to higherWeightBits; ' +
      'use explicit weights rules and validate the combined selection on separate task data.';
  }
  return report;
}
