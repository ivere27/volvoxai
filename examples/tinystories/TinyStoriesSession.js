import {createTinyStoriesDecodeState, resolveTinyStoriesSequenceContract} from './browser_session.js';
import {createPrng, sampleToken} from './Sampling.js';
import {EngineDebugSession} from '../common/EngineDebugSession.js';
import {startEngineTrace} from '../common/EngineTrace.js';

function checked(value) {
  const report = value.report ?? value;
  if (report.status !== 0) throw new Error(report.message || 'TinyStories engine request failed');
  return value;
}

/** Own model policy and isolated generation/debug contexts, over generated RPCs. */
export class TinyStoriesSession {
  #api; #host; #inference; #platform; #modelId; #compiledModelId; #runtimeId; #closed = false;
  #children = new Set(); #closePromise = null;
  constructor({api, host, inference, platform, modelId, compiledModelId, runtimeId, graph, tokenizer, backend, compileReport, modelUrl}) {
    Object.assign(this, {graph, tokenizer, backend, compileReport, modelUrl});
    this.#api = api; this.#host = host; this.#inference = inference; this.#platform = platform;
    this.#modelId = modelId; this.#compiledModelId = compiledModelId; this.#runtimeId = runtimeId;
    const tensors = Object.fromEntries(graph.nodes.flatMap(node => Object.values(node.outputs).map(output => [output.tensor, output])));
    this.contract = resolveTinyStoriesSequenceContract({graph: {...graph, tensors}, outputNames: graph.outputs});
    this.eosTokenId = tokenizer.findTokenByText('<|endoftext|>');
    if (this.eosTokenId < 0 || this.eosTokenId >= this.contract.vocabularySize) throw new Error('Tokenizer has no valid EOS token.');
  }

  static async open({api, wasmUrl, graph, graphUrl, weightsUrl, tokenizer, backend = 'wasm'}) {
    const host = new api.FullEngineHost({wasmUrl});
    const inference = new api.VxInferenceServiceClient(host), platform = new api.VxPlatformServiceClient(host);
    try {
      const {runtimeId} = checked(await inference.createRuntime(new api.pb.CreateRuntimeRequest()));
      const {modelId} = checked(await inference.loadModel(new api.pb.LoadModelRequest({runtimeId,
        graphPath: String(graphUrl), weightPaths: [String(weightsUrl)]})));
      const compiled = checked(await inference.compileModel(new api.pb.CompileModelRequest({modelId,
        policy: TinyStoriesSession.policy(api.pb, backend)})));
      TinyStoriesSession.attest(api.pb, compiled.report, backend);
      return new TinyStoriesSession({api, host, inference, platform, modelId, runtimeId,
        compiledModelId: compiled.compiledModelId, compileReport: compiled.report, graph, tokenizer, backend, modelUrl: String(graphUrl)});
    } catch (error) { await host.close(); throw error; }
  }

  static policy(pb, backend) { return new pb.BackendPolicy({mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
    backends: [backend], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID}); }
  static attest(pb, report, backend) {
    const selected = report.compilation?.candidates?.filter(candidate => candidate.outcome === pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED) ?? [];
    if (report.backend !== backend || report.compilation?.policyMode !== pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE ||
        report.compilation?.operatorFallback !== pb.OperatorFallback.OPERATOR_FALLBACK_FORBID || selected.length !== 1 || selected[0].backend !== backend)
      throw new Error(`Compilation did not attest required backend '${backend}'.`);
  }

  prepare({prompt, requestedNewTokens, seed = 42, temperature = .7, topK = 40}) {
    if (this.#closed) throw new Error('TinyStories session is closed.');
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff || !Number.isFinite(temperature) || temperature < 0 ||
        !Number.isInteger(topK) || topK < 1 || topK > this.contract.vocabularySize)
      throw new Error('Use a uint32 seed, non-negative temperature and top-K within the vocabulary.');
    // Read one beyond the legal bound, so an overlong prompt is rejected rather
    // than silently truncated by the tokenizer's maximum-token parameter.
    const promptTokenIds = this.tokenizer.encode(prompt, this.contract.maximumSequenceCapacity + 1);
    const state = createTinyStoriesDecodeState(promptTokenIds, {requestedNewTokens, eosTokenId: this.eosTokenId, contract: this.contract});
    return {prompt, promptTokenIds, requestedNewTokens, seed, temperature, topK, state};
  }

  #tensor(name, shaped) {
    const {pb} = this.#api;
    return new pb.Tensor({name, dtype: pb.DataType.DATA_TYPE_I32, shape: shaped.shape.map(BigInt),
      inline: new Uint8Array(shaped.data.buffer, shaped.data.byteOffset, shaped.data.byteLength)});
  }
  #request(contextId, state, index, readValues = true) {
    const {pb} = this.#api, initialize = index === 0;
    const contract = this.graph.execution;
    const effect = initialize ? 'EXECUTION_EFFECT_INITIALIZE' : 'EXECUTION_EFFECT_TRANSITION';
    const entry = contract.entrypoints.find(entry => entry.effect === effect || entry.effect === (initialize ? 2 : 3));
    if (!entry || !contract.incremental?.active_lengths_input) throw new Error('TinyStories requires an explicit incremental execution contract.');
    const position = state.promptLength - 1 + index;
    const values = [this.#tensor('tokens', state.inputs.tokens)];
    if (initialize) {
      values.push(this.#tensor('positions', state.inputs.positions));
      values.push(this.#tensor(contract.incremental.active_lengths_input,
        {data: Int32Array.of(state.promptLength), shape: [1]}));
    }
    const keepName = contract.generation?.keep_input;
    if (keepName && this.graph.inputs[keepName]) {
      const keep = new Int32Array(state.sequenceCapacity); keep.fill(1, 0, position + 1);
      values.push(this.#tensor(keepName, {data: keep, shape: [1, state.sequenceCapacity]}));
    }
    const supplied = new Set(values.map(value => value.name));
    return new pb.ExecuteRequest({contextId,
      inputs: new pb.InputBindings({values, reuse: initialize ? [] : Object.keys(this.graph.inputs).filter(name => !supplied.has(name))}),
      scope: new pb.ExecutionScope({entrypoint: entry.name,
        ...(initialize ? {full: new pb.Empty()} : {slotRows: new pb.SlotRows({slots: [new pb.SlotRow({position})]})})}),
      outputs: new pb.OutputOptions({selection: new pb.TensorOutputSelection({names: readValues ? [this.contract.outputName] : []})}),
    });
  }
  async #context(compiledModelId) {
    const {pb} = this.#api;
    return checked(await this.#inference.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId,
      graph: new pb.GraphExecution({rowMode: pb.DecodeRowMode.DECODE_ROW_MODE_AUTO, requireIncremental: true})}))).contextId;
  }
  async #execute(contextId, state, index, {readValues = true} = {}) {
    const {pb} = this.#api;
    const result = checked(await this.#inference.execute(this.#request(contextId, state, index, readValues)));
    if (result.report.backend !== this.backend) throw new Error(`Execution ran on unexpected backend '${result.report.backend}'.`);
    let logits = null;
    if (readValues) {
      const tensor = result.outputs.find(output => output.name === this.contract.outputName);
      if (!tensor || tensor.dtype !== pb.DataType.DATA_TYPE_F32 || tensor.inline.byteLength % 4)
        throw new Error('Logits output must contain inline F32 storage.');
      logits = new Float32Array(tensor.inline.slice().buffer);
    }
    return {logits, contextId: String(contextId), executionId: String(result.executionId ?? 0n), modelId: String(this.#modelId)};
  }

  async generate(settings, {onToken = () => {}, shouldStop = () => false, profiled = false} = {}) {
    const prepared = this.prepare(settings), {state, promptTokenIds, prompt, seed, temperature, topK} = prepared;
    const contextId = await this.#context(this.#compiledModelId), {pb} = this.#api;
    let capture = null;
    const output = {...prepared, state: undefined, backend: this.backend, sequenceCapacity: state.sequenceCapacity,
      promptLength: state.promptLength, generationLimit: state.generationLimit, generatedTokenIds: [], records: [], phaseRecords: [],
      stopReason: 'length', firstTokenMs: null, prefillMs: null, decodeMs: null, wallMs: null, text: prompt};
    try {
      if (profiled) capture = await startEngineTrace(this.#api, this.#host, this.#runtimeId);
      const random = createPrng(seed), started = performance.now();
      for (let index = 0; index < state.generationLimit; index++) {
        if (shouldStop()) { output.stopReason = 'stopped'; break; }
        const phase = index === 0 ? 'prefill' : 'decode', position = state.promptLength - 1 + index;
        const startNs = capture ? (await this.#platform.getMonotonicTime(new pb.Empty())).nanoseconds : null;
        const start = performance.now(), executed = await this.#execute(contextId, state, index), wallMs = performance.now() - start;
        const sample = sampleToken(executed.logits, position, this.contract.vocabularySize, temperature, random, topK);
        const record = {...sample, index, phase, position, wallMs, tokenText: this.tokenizer.decodeToken(sample.tokenId), eos: sample.tokenId === this.eosTokenId,
          contextId: executed.contextId, executionId: executed.executionId};
        output.records.push(record);
        if (index === 0) { output.prefillMs = wallMs; output.firstTokenMs = performance.now() - started; }
        if (capture) {
          const endNs = (await this.#platform.getMonotonicTime(new pb.Empty())).nanoseconds;
          const rangeName = `TinyStories ${phase} · prediction ${index} · position ${position}`;
          await capture.annotate(rangeName, startNs, endNs);
          output.phaseRecords.push({phase, index, position, contextId: executed.contextId, executionId: executed.executionId,
            modelId: executed.modelId, wallMs, label: `${phase === 'prefill' ? 'Prefill' : 'Decode'} ${index} · position ${position}`,
            startNs: String(startNs), endNs: String(endNs), rangeName});
        }
        if (!record.eos) { output.generatedTokenIds.push(record.tokenId); output.text = prompt + this.tokenizer.decode(output.generatedTokenIds); }
        output.wallMs = performance.now() - started;
        const decode = output.records.filter(value => value.phase === 'decode');
        output.decodeMs = decode.length ? decode.reduce((total, value) => total + value.wallMs, 0) / decode.length : null;
        // The optional observer can inspect this transient logits buffer; the
        // adapter retains candidate scores rather than every vocabulary row.
        await onToken({...output}, record, executed.logits);
        if (record.eos) { output.stopReason = 'eos'; break; }
        state.tokens[position + 1] = record.tokenId;
        // Browser input/rendering can progress between synchronous WASM calls.
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      output.wallMs = performance.now() - started;
      if (capture) output.trace = await capture.finish();
      return output;
    } finally {
      try { await capture?.release(); }
      finally { await this.#inference.releaseExecutionContext(new pb.ExecutionContextRef({contextId})); }
    }
  }

  /** Replay the chosen prefix in an isolated context, then attach a real decode debugger. */
  async createDebugSession(settings, {prediction = 0, run = null, preserveNodeBoundaries = true, capture, limits} = {}) {
    const prepared = run ? this.prepare({...run, requestedNewTokens: run.generationLimit}) : this.prepare(settings);
    const {state} = prepared, {pb} = this.#api;
    if (!Number.isInteger(prediction) || prediction < 0 || prediction >= Math.max(1, state.generationLimit) ||
        (prediction > 0 && (!run || prediction > run.generatedTokenIds.length)))
      throw new Error('Generate the selected token prefix before debugging its decode step.');
    let compiledModelId = this.#compiledModelId, contextId = null, debug = null;
    const owned = preserveNodeBoundaries;
    const cleanup = async () => {
      try { if (contextId !== null) await this.#inference.releaseExecutionContext(new pb.ExecutionContextRef({contextId})); }
      finally { if (owned && compiledModelId !== this.#compiledModelId) await this.#inference.releaseCompiledModel(new pb.CompiledModelRef({compiledModelId})); }
    };
    try {
      if (owned) {
        const compiled = checked(await this.#inference.compileModel(new pb.CompileModelRequest({modelId: this.#modelId,
          preserveNodeBoundaries: true, policy: TinyStoriesSession.policy(pb, this.backend)})));
        compiledModelId = compiled.compiledModelId; TinyStoriesSession.attest(pb, compiled.report, this.backend);
      }
      contextId = await this.#context(compiledModelId);
      for (let index = 0; index < prediction; index++) {
        if (index > 0) state.tokens[state.promptLength - 1 + index] = run.generatedTokenIds[index - 1];
        await this.#execute(contextId, state, index, {readValues: false});
      }
      if (prediction > 0) state.tokens[state.promptLength - 1 + prediction] = run.generatedTokenIds[prediction - 1];
      debug = await EngineDebugSession.create(this.#api, this.#host, {
        execute: this.#request(contextId, state, prediction),
        capture: new pb.DebugCapture(capture), limits: new pb.DebugLimits(limits)});
      const release = debug.release.bind(debug); let released = null;
      debug.release = () => released ??= (async () => { try { await release(); } finally { try { await cleanup(); } finally { this.#children.delete(debug); } } })();
      this.#children.add(debug);
      return debug;
    } catch (error) { await cleanup(); throw error; }
  }

  close() {
    if (this.#closePromise === null) {
      this.#closed = true;
      this.#closePromise = (async () => { try { await Promise.all([...this.#children].map(session => session.release())); } finally { await this.#host.close(); } })();
    }
    return this.#closePromise;
  }
}
