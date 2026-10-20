import {startEngineTrace} from '../common/EngineTrace.js';
import {efficientDetInputViewFromRgba} from './image_input.js';

function checked(value) {
  const report = value.report ?? value;
  if (report.status !== 0) throw new Error(report.message || 'EfficientDet request failed');
  return value;
}

function backendPolicy(pb, backend) {
  return new pb.BackendPolicy({mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
    backends: [backend], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID});
}

function checkBackend(pb, report, backend) {
  checked(report);
  const selected = report.compilation?.candidates?.filter(
    candidate => candidate.outcome === pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED) ?? [];
  if (report.backend !== backend || report.compilation?.policyMode !== pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE
      || report.compilation?.operatorFallback !== pb.OperatorFallback.OPERATOR_FALLBACK_FORBID
      || selected.length !== 1 || selected[0].backend !== backend)
    throw new Error(`CompileModel did not attest the required '${backend}' backend`);
}

function inputDescriptor(pb, spec, affine) {
  if (!spec) throw new Error('Model is missing image input tensor input0');
  const dtype = new Map([[pb.DataType.DATA_TYPE_F32, 'float32'],
    [pb.DataType.DATA_TYPE_I8, 'int8'], [pb.DataType.DATA_TYPE_U8, 'uint8']]).get(spec.dtype);
  if (!dtype) throw new Error(`input0 has unsupported dtype ${spec.dtype}`);
  const shape = spec.dimensions.map(axis => {
    if (axis.kind !== pb.DimensionKind.DIMENSION_KIND_FIXED || axis.min !== axis.max
        || axis.min <= 0n || axis.min > BigInt(Number.MAX_SAFE_INTEGER))
      throw new Error('EfficientDet input0 requires fixed, positive dimensions');
    return Number(axis.min);
  });
  return {shape, dtype, ...(affine?.defined ? {quantization: {
    scheme: 'per_tensor', scale: affine.scale, zero_point: affine.zeroPoint,
  }} : {})};
}

/** One ordinary context; explicit debug captures own an independent context. */
export class EfficientDetSession {
  #api; #host; #inference; #runtimeId; #modelId; #compiledModelId; #contextId; #closePromise = null;
  constructor({api, host, inference, runtimeId, modelId, compiledModelId, contextId, backend,
    compileReport, input, fallbackReason}) {
    this.#api = api; this.#host = host; this.#inference = inference; this.#runtimeId = runtimeId;
    this.#modelId = modelId; this.#compiledModelId = compiledModelId; this.#contextId = contextId;
    this.backend = backend; this.compileReport = compileReport; this.input = input;
    this.fallbackReason = fallbackReason;
  }

  static async open({api, wasmUrl, graphUrl, weightsUrl, backend}) {
    const {pb} = api, host = new api.FullEngineHost({wasmUrl});
    const inference = new api.VxInferenceServiceClient(host);
    try {
      const {runtimeId} = checked(await inference.createRuntime(new pb.CreateRuntimeRequest()));
      const {modelId} = checked(await inference.loadModel(new pb.LoadModelRequest({runtimeId,
        graphPath: String(graphUrl), weightPaths: [String(weightsUrl)]})));
      let compiled, fallbackReason = '';
      try {
        compiled = await inference.compileModel(new pb.CompileModelRequest({modelId, policy: backendPolicy(pb, backend)}));
        checkBackend(pb, compiled.report, backend);
      } catch (error) {
        if (backend !== 'webgpu') throw error;
        // Retain the original demo's explicit whole-model CPU fallback.
        if (compiled?.compiledModelId) checked(await inference.releaseCompiledModel(
          new pb.CompiledModelRef({compiledModelId: compiled.compiledModelId})));
        fallbackReason = `WebGPU unavailable: ${error.message}. Using WASM CPU.`;
        backend = 'wasm';
        compiled = await inference.compileModel(new pb.CompileModelRequest({modelId, policy: backendPolicy(pb, backend)}));
        checkBackend(pb, compiled.report, backend);
      }
      const {compiledModelId} = compiled;
      const context = checked(await inference.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId, graph: new pb.GraphExecution()})));
      const {contextId} = context, spec = context.inputs.find(input => input.name === 'input0');
      let affine;
      if (spec?.dtype === pb.DataType.DATA_TYPE_I8 || spec?.dtype === pb.DataType.DATA_TYPE_U8)
        affine = checked(await inference.getInputAffineQuantization(new pb.GetInputAffineQuantizationRequest({contextId, name: 'input0'})));
      const input = inputDescriptor(pb, spec, affine);
      if (JSON.stringify(input.shape) !== '[1,320,320,3]')
        throw new Error('EfficientDet Lite0 requires image input [1, 320, 320, 3]');
      return new EfficientDetSession({api, host, inference, runtimeId, modelId, compiledModelId, contextId,
        backend, compileReport: compiled.report, input, fallbackReason});
    } catch (error) {
      try { await host.close(); }
      catch (cleanupError) { throw new AggregateError([error, cleanupError], 'EfficientDet initialization failed'); }
      throw error;
    }
  }

  #prepare(rgba) {
    if (this.#closePromise) throw new Error('EfficientDet session is closed');
    const {pb} = this.#api, input = efficientDetInputViewFromRgba(rgba, this.input);
    const dtype = {float32: pb.DataType.DATA_TYPE_F32, int8: pb.DataType.DATA_TYPE_I8,
      uint8: pb.DataType.DATA_TYPE_U8}[this.input.dtype];
    return new pb.Tensor({name: 'input0', shape: input.shape.map(BigInt), dtype,
      inline: new Uint8Array(input.data.buffer, input.data.byteOffset, input.data.byteLength)});
  }

  #output(outputs, name) {
    const {pb} = this.#api, tensor = outputs.find(value => value.name === name);
    if (tensor?.dtype !== pb.DataType.DATA_TYPE_F32 || !tensor.inline || tensor.inline.byteLength % 4
        || tensor.shape.reduce((size, axis) => size * axis, 1n) * 4n !== BigInt(tensor.inline.byteLength))
      throw new Error(`Output '${name}' is not a complete inline float32 tensor`);
    const bytes = tensor.inline.slice();
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  async #execute(rgba) {
    const {pb} = this.#api, preparedAt = performance.now(), input = this.#prepare(rgba);
    const preprocessMs = performance.now() - preparedAt, started = performance.now();
    const result = checked(await this.#inference.execute(new pb.ExecuteRequest({
      contextId: this.#contextId, inputs: new pb.InputBindings({values: [input]}),
      scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()}),
      outputs: new pb.OutputOptions({selection: new pb.TensorOutputSelection({names: ['scores', 'boxes']})}),
    })));
    const scores = this.#output(result.outputs, 'scores'), boxes = this.#output(result.outputs, 'boxes');
    return {scores, boxes, preprocessMs, executionMs: performance.now() - started,
      contextId: this.#contextId, modelId: this.#modelId, executionId: result.executionId,
      compiledModelId: this.#compiledModelId};
  }

  detect(rgba) { return this.#execute(rgba); }

  async detectProfiled(rgba) {
    const capture = await startEngineTrace(this.#api, this.#host, this.#runtimeId);
    try {
      const started = performance.now(), result = await this.#execute(rgba);
      const wallMs = performance.now() - started;
      return {...result, wallMs, trace: await capture.finish()};
    } finally { await capture.release(); }
  }

  async createDebugSession(rgba, {preserveNodeBoundaries = true, capture, limits} = {}) {
    if (!this.#api.VxDebugServiceClient) throw new Error('Node debugging requires the full profile');
    const {pb} = this.#api, inputs = [this.#prepare(rgba)];
    const {EngineDebugSession} = await import('../common/EngineDebugSession.js');
    let compiledModelId = this.#compiledModelId;
    try {
      if (preserveNodeBoundaries) {
        const compiled = await this.#inference.compileModel(new pb.CompileModelRequest({modelId: this.#modelId,
          preserveNodeBoundaries: true, policy: backendPolicy(pb, this.backend)}));
        compiledModelId = compiled.compiledModelId || this.#compiledModelId;
        checkBackend(pb, compiled.report, this.backend);
      }
      return await EngineDebugSession.createEvaluation(this.#api, this.#host, {
        compiledModelId, inputs,
        capture: new pb.DebugCapture(capture), limits: new pb.DebugLimits(limits),
      });
    } finally {
      if (compiledModelId !== this.#compiledModelId) checked(await this.#inference.releaseCompiledModel(
        new pb.CompiledModelRef({compiledModelId})));
    }
  }

  close() { this.#closePromise ??= this.#host.close(); return this.#closePromise; }
}
