#!/usr/bin/env node
/**
 * Per-layer PTQ sensitivity, like Polygraphy's `debug precision` or the
 * layer-wise analyses of AIMET and NNCF. Accumulated error hides which layer
 * causes a loss: this sweep authors one package per quantizable node and
 * compares its outputs with the float model on evaluation inputs.
 *
 *   isolated         quantize only that node (selected_nodes = [node])
 *   leave-one-float  quantize everything except that node (float_nodes = [node])
 *
 * Isolated ranks nodes by the error they add alone; leave-one-float ranks
 * them by how much keeping them in F32 recovers. Both only compose existing
 * AuthorPtqTemplate / CreatePtqPlan / CalibratePtqPlan / WritePtqPackage calls.
 */

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
async function evaluate(inference, p, runtimeId, pkg, evaluation, outputs) {
  const model = check(await inference.loadModel(new p.LoadModelRequest({runtimeId,
    package: new p.ModelPackage({graphDocument: pkg.graph, weightShards: pkg.weights})})), 'LoadModel');
  const compiled = check(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId})), 'CompileModel');
  const context = check(await inference.createExecutionContext(
    new p.CreateExecutionContextRequest({compiledModelId: compiled.compiledModelId})), 'CreateExecutionContext');
  const values = Object.fromEntries(outputs.map(name => [name, []]));
  try {
    for (const inputs of evaluation) {
      const result = check(await inference.execute(new p.ExecuteRequest({contextId: context.contextId, inputs})), 'Execute');
      try {
        for (const name of outputs)
          values[name].push(floats(check(await inference.readOutput(
            new p.ReadOutputRequest({resultId: result.resultId, name})), 'ReadOutput').tensor, p));
      } finally { await inference.releaseResult(new p.ResultRef({resultId: result.resultId})); }
    }
  } finally {
    await inference.releaseExecutionContext(new p.ExecutionContextRef({contextId: context.contextId}));
    await inference.releaseCompiledModel(new p.CompiledModelRef({compiledModelId: compiled.compiledModelId}));
    await inference.releaseModel(new p.ModelRef({modelId: model.modelId}));
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

async function quantizedPackage(quantization, p, floatModelId, source, config, calibration) {
  const authored = check(await quantization.authorPtqTemplate(new p.AuthorPtqTemplateRequest({
    sourceGraph: source.graph, weightShards: source.weights, config})), 'AuthorPtqTemplate');
  if (!authored.quantizedNodes) return {authored, pkg: null};
  const plan = check(await quantization.createPtqPlan(new p.CreatePtqPlanRequest({modelId: floatModelId,
    templateGraph: authored.templateGraph, observers: authored.observers, layers: authored.layers,
    profileNames: ['sensitivity']})), 'CreatePtqPlan');
  try {
    for (let i = 0; i < calibration.length; i++)
      check(await quantization.calibratePtqPlan(new p.CalibratePtqPlanRequest({ptqPlanId: plan.ptqPlanId,
        profileName: 'sensitivity', sampleName: `sample-${i}`, sampleCount: 1n, inputs: calibration[i]})),
        'CalibratePtqPlan');
    const written = await quantization.writePtqPackage(new p.WritePtqPackageRequest({ptqPlanId: plan.ptqPlanId}));
    // Some selections author but cannot be written (for example a region
    // without a dense layer); report them instead of failing the sweep.
    if (written.report?.code === p.OperationCode.OPERATION_CODE_PTQ_PACKAGE_WRITE_FAILED)
      return {authored, pkg: null, note: 'the package writer refused this selection'};
    check(written, 'WritePtqPackage');
    return {authored, pkg: {graph: written.graph, weights: [written.weights]}};
  } finally {
    await quantization.releasePtqPlan(new p.PtqPlanRef({ptqPlanId: plan.ptqPlanId}));
  }
}

/** Source node IDs a full authoring pass quantizes, in graph order. */
function quantizedNodeIds(templateGraph, sourceGraph) {
  const source = new Map(JSON.parse(new TextDecoder().decode(sourceGraph)).nodes.map(node => [node.id, node.opType]));
  return JSON.parse(new TextDecoder().decode(templateGraph)).nodes
    .filter(node => source.has(node.id) && source.get(node.id) !== node.opType).map(node => node.id);
}

/**
 * source: {graph: Uint8Array, weights: Uint8Array[]}; calibration and
 * evaluation: arrays of Tensor lists; outputs: F32 graph output names.
 * config: PtqAuthoringConfig fields for every package (dtype, scheme, ...).
 */
export async function ptqSensitivity({inference, quantization, p, runtimeId, source, calibration, evaluation,
  outputs, config = {}, modes = ['isolated', 'leave-one-float']}) {
  const floatModel = check(await inference.loadModel(new p.LoadModelRequest({runtimeId,
    package: new p.ModelPackage({graphDocument: source.graph, weightShards: source.weights})})), 'LoadModel');
  try {
    const reference = await evaluate(inference, p, runtimeId, source, evaluation, outputs);
    const make = extra => new p.PtqAuthoringConfig({...config, ...extra});
    const full = await quantizedPackage(quantization, p, floatModel.modelId, source, make({}), calibration);
    if (!full.pkg) throw new Error('authoring quantized no node');
    const baseline = compare(reference, await evaluate(inference, p, runtimeId, full.pkg, evaluation, outputs));
    const nodes = quantizedNodeIds(full.authored.templateGraph, source.graph);
    const rows = [];
    for (const node of nodes) {
      const row = {node};
      for (const mode of modes) {
        const extra = mode === 'isolated' ? {selectedNodes: [node]} : {floatNodes: [...(config.floatNodes ?? []), node]};
        const {pkg, note} = await quantizedPackage(quantization, p, floatModel.modelId, source, make(extra), calibration);
        const measured = pkg ? compare(reference, await evaluate(inference, p, runtimeId, pkg, evaluation, outputs))
          : {sqnrDb: null, perOutput: {}, note: note ?? 'nothing left to quantize'};
        row[mode === 'isolated' ? 'isolated' : 'leaveOneFloat'] = measured;
      }
      if (row.leaveOneFloat?.sqnrDb != null && baseline.sqnrDb != null)
        row.recoveryDb = row.leaveOneFloat.sqnrDb - baseline.sqnrDb;
      rows.push(row);
    }
    const byIsolated = [...rows].filter(row => row.isolated?.sqnrDb != null)
      .sort((a, b) => a.isolated.sqnrDb - b.isolated.sqnrDb).map(row => row.node);
    const byRecovery = [...rows].filter(row => row.recoveryDb != null)
      .sort((a, b) => b.recoveryDb - a.recoveryDb).map(row => row.node);
    return {format: 'volvoxai-ptq-sensitivity/v1', outputs, evaluationBatches: evaluation.length,
      calibrationBatches: calibration.length, baseline, rows,
      mostSensitive: byIsolated, bestToKeepFloat: byRecovery,
      interpretation: 'SQNR of graph outputs against the float model, worst output per package. ' +
        'Isolated ranks the error a node adds alone; recoveryDb is the SQNR regained by keeping ' +
        'that node in F32 while everything else is quantized. Pass the chosen ids as float_nodes.'};
  } finally {
    await inference.releaseModel(new p.ModelRef({modelId: floatModel.modelId}));
  }
}
