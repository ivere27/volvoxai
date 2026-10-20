import {preprocessTinyReceiptImage} from './TinyReceiptInput.js';
import {TinyReceiptByteFallbackBPEVocab} from './TinyReceiptByteFallbackBPEVocab.js';
import {EngineDebugSession} from '../common/EngineDebugSession.js';
import {startEngineTrace} from '../common/EngineTrace.js';

const PACKAGE_FORMAT = 'volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v2';
const roles = ['encoder', 'decoder'];
const fail = message => { throw new Error(`[TinyReceiptSession] ${message}`); };
function checked(value) {
  const report = value.report ?? value;
  if (report.status !== 0) fail(report.message || 'Engine request failed');
  return value;
}
function attest(pb, report, backend, compilation = false) {
  checked(report);
  const route = report.route;
  if (report.backend !== backend || route?.provider !== backend || !route.attested ||
      route.activeNodes !== route.selectedNodes || route.fallbackNodes || route.missingNodes ||
      report.fallback?.operatorFallbackUsed || report.compilation?.tierFallbackUsed)
    fail(`Engine did not attest the required '${backend}' route without fallback.`);
  if (compilation) {
    const proof = report.compilation;
    const selected = proof?.candidates?.filter(item => item.outcome === pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED);
    if (proof?.policyMode !== pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE ||
        proof.operatorFallback !== pb.OperatorFallback.OPERATOR_FALLBACK_FORBID ||
        selected?.length !== 1 || selected[0].backend !== backend)
      fail(`Compilation did not prove the required '${backend}' policy.`);
  }
}
const i32 = (values, shape) => ({data: Int32Array.from(values), shape, dtype: 'int32'});
const f32 = (values, shape) => ({data: values instanceof Float32Array ? values : Float32Array.from(values), shape, dtype: 'float32'});

export function validateTinyReceiptManifest(manifest) {
  if (manifest?.format !== PACKAGE_FORMAT) fail(`Expected ${PACKAGE_FORMAT}.`);
  const geometry = manifest.shape_contract?.fixed_geometry, dimensions = manifest.shape_contract?.dimensions;
  if (!Array.isArray(geometry?.image) || geometry.image.length !== 4 || geometry.image[0] !== 1 || geometry.image[1] !== 1 ||
      !geometry.image.every(value => Number.isSafeInteger(value) && value > 0)) fail('Expected a positive batch-1 grayscale image shape.');
  for (const name of ['Q', 'P', 'R']) {
    const domain = dimensions?.[name];
    if (!Number.isSafeInteger(domain?.min) || !Number.isSafeInteger(domain?.max) || domain.min <= 0 || domain.min > domain.max)
      fail(`Missing positive bounded ${name} domain.`);
  }
  if (manifest.cache_contract?.format !== 'masked-zero-sentinel-v1' || manifest.cache_contract.initial_past_length !== 1 ||
      manifest.cache_contract.sentinel_mask_value !== 1) fail('Expected the masked-zero P=1 cache sentinel contract.');
  for (const key of ['decoder_layers', 'attention_heads', 'attention_head_width'])
    if (!Number.isSafeInteger(geometry[key]) || geometry[key] < 1) fail(`Invalid ${key}.`);
  for (const role of roles) {
    const graph = manifest.graphs?.[role];
    if (!graph?.graph?.path || !graph.weights?.path || !graph.inputs || !graph.outputs) fail(`Missing ${role} graph assets / ABI.`);
    if ([...Object.values(graph.inputs), ...Object.values(graph.outputs)].some(name => typeof name !== 'string' || !name))
      fail(`Invalid ${role} ABI names.`);
  }
  return manifest;
}

/** Sampling is example-owned token policy. Greedy keeps the first-index tie rule. */
export function chooseReceiptToken(logits, {sampling = 'greedy', temperature = 1, topK = 40} = {}, random = Math.random) {
  if (!logits.length || !logits.every(Number.isFinite)) fail('Decoder logits must be finite and nonempty.');
  let best = 0;
  for (let index = 1; index < logits.length; index++) if (logits[index] > logits[best]) best = index;
  if (sampling === 'greedy') return best;
  if (sampling !== 'sample' || !Number.isFinite(temperature) || temperature <= 0 ||
      !Number.isSafeInteger(topK) || topK < 0 || topK > logits.length) fail('Invalid sampling options.');
  const indices = Array.from(logits, (_, index) => index).sort((a, b) => logits[b] - logits[a] || a - b).slice(0, topK || logits.length);
  const weights = indices.map(index => Math.exp((logits[index] - logits[best]) / temperature));
  let draw = random() * weights.reduce((total, weight) => total + weight, 0);
  for (let index = 0; index < indices.length; index++) { draw -= weights[index]; if (draw < 0) return indices[index]; }
  return indices.at(-1);
}
function seededRandom(seed) {
  let state = seed >>> 0;
  return () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
}

export function receiptDecoderFeed(manifest, encoded, bos) {
  const geometry = manifest.shape_contract.fixed_geometry;
  const feed = Object.fromEntries(Object.entries(encoded).filter(([name]) => name.startsWith('cross_') || name === 'memory_padding_mask'));
  Object.assign(feed, {decoder_input_ids: i32([bos], [1, 1]), position_ids: i32([0], [1]),
    family_ids: encoded.selected_family_ids, past_padding_mask: i32([1], [1, 1])});
  for (let layer = 0; layer < geometry.decoder_layers; layer++) for (const kind of ['k', 'v'])
    feed[`past_${kind}_${layer}`] = f32(new Float32Array(geometry.attention_heads * geometry.attention_head_width),
      [1, geometry.attention_heads, 1, geometry.attention_head_width]);
  return feed;
}
export function advanceReceiptDecoder(manifest, feed, output, token, position) {
  for (let layer = 0; layer < manifest.shape_contract.fixed_geometry.decoder_layers; layer++) for (const kind of ['k', 'v'])
    feed[`past_${kind}_${layer}`] = output[`present_${kind}_${layer}`];
  feed.past_padding_mask = output.present_padding_mask;
  feed.decoder_input_ids = i32([token], [1, 1]); feed.position_ids = i32([position + 1], [1]);
}
export function parseReceiptAnswer(vocab, tokens) {
  const raw = vocab.decode(tokens), fields = {raw};
  for (const name of ['field', 'value', 'op', 'answer']) {
    const match = new RegExp(`<${name}>([^<]*)</${name}>`).exec(raw);
    fields[name] = match ? match[1].trim() : null;
  }
  return fields;
}

/** Application-owned split-model orchestration over generated inference/debug/profiling calls. */
export class TinyReceiptSession {
  #api; #host; #inference; #platform; #closed = null; #running = false; #lastInput = null;
  models = {}; compiled = {}; contexts = {}; compileReports = {}; graphs = {}; lastRun = null;

  constructor(api, host, inference, manifest, vocab, backend, runtimeId) {
    this.#api = api; this.#host = host; this.#inference = inference; this.manifest = manifest; this.vocab = vocab;
    this.backend = backend; this.runtimeId = runtimeId; this.#platform = new api.VxPlatformServiceClient(host);
  }
  get maxTokens() { return Math.min(this.manifest.generation.maximum_new_tokens,
    this.manifest.shape_contract.dimensions.P.max, this.manifest.shape_contract.dimensions.R.max - 1); }
  get questionCapacity() { return this.manifest.shape_contract.dimensions.Q.max; }
  get families() { return this.manifest.families.ordered_names; }
  #policy() {
    const {pb} = this.#api;
    return new pb.BackendPolicy({mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: [this.backend], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID});
  }
  static async open({api, wasmUrl, packageUrl, backend = 'wasm', fetch: fetchImpl = globalThis.fetch}) {
    if (!['wasm', 'webgpu'].includes(backend) || typeof api.FullEngineHost !== 'function') fail('A full WASM/WebGPU profile is required.');
    const base = new URL(packageUrl);
    const readJson = async path => {
      const response = await fetchImpl(new URL(path, base)); if (!response.ok) fail(`Asset unavailable: ${new URL(path, base)}`);
      return response.json();
    };
    const manifest = validateTinyReceiptManifest(await readJson('package_manifest.json'));
    const tok = manifest.tokenizer;
    const vocab = await TinyReceiptByteFallbackBPEVocab.fromJSON(await readJson(manifest.assets?.vocab?.path || 'vocab.json'), tok.token_ids,
      {vocabSize: tok.vocab_size, normalization: tok.normalization, tokenizerHash: tok.tokenizer_hash});
    const host = new api.FullEngineHost({wasmUrl, fetch: fetchImpl}), inference = new api.VxInferenceServiceClient(host);
    try {
      const {pb} = api, runtime = checked(await inference.createRuntime(new pb.CreateRuntimeRequest()));
      const session = new TinyReceiptSession(api, host, inference, manifest, vocab, backend, runtime.runtimeId);
      for (const role of roles) {
        const contract = manifest.graphs[role];
        const graphUrl = new URL(contract.graph.path, base);
        const graph = await fetchImpl(graphUrl); if (!graph.ok) fail(`Graph unavailable: ${graphUrl}`);
        session.graphs[role] = await graph.json();
        const model = checked(await inference.loadModel(new pb.LoadModelRequest({runtimeId: runtime.runtimeId,
          graphPath: graphUrl.href, weightPaths: [new URL(contract.weights.path, base).href]})));
        session.models[role] = model.modelId;
        const compiled = checked(await inference.compileModel(new pb.CompileModelRequest({modelId: model.modelId, policy: session.#policy()})));
        attest(pb, compiled.report, backend, true); session.compiled[role] = compiled.compiledModelId; session.compileReports[role] = compiled.report;
        session.contexts[role] = checked(await inference.createExecutionContext(
          new pb.CreateExecutionContextRequest({compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()}))).contextId;
      }
      return session;
    } catch (error) { await host.close(); throw error; }
  }
  #inputs(role, feed) {
    const {pb} = this.#api, expected = this.manifest.graphs[role].inputs;
    if (Object.keys(expected).sort().join('\0') !== Object.keys(feed).sort().join('\0')) fail(`Incomplete ${role} input batch.`);
    return Object.entries(feed).map(([name, value]) => new pb.Tensor({name: expected[name], shape: value.shape.map(BigInt),
      dtype: value.dtype === 'float32' ? pb.DataType.DATA_TYPE_F32 : pb.DataType.DATA_TYPE_I32,
      inline: new Uint8Array(value.data.buffer, value.data.byteOffset, value.data.byteLength)}));
  }
  #scope(role, feed) {
    const {pb} = this.#api;
    const effect = role === 'encoder' ? 1 : feed.position_ids.data[0] === 0 ? 2 : 3;
    const entry = this.graphs[role].execution?.entrypoints.find(item => Number(item.effect) === effect);
    if (!entry) fail(`Missing ${role} execution entrypoint for effect ${effect}.`);
    return new pb.ExecutionScope({entrypoint: entry.name, full: new pb.Empty()});
  }
  #bindings(role, feed) {
    const {pb} = this.#api, values = this.#inputs(role, feed);
    const reuse = role === 'decoder' && feed.position_ids.data[0] !== 0
      ? this.graphs[role].execution.invariant_inputs ?? [] : [];
    return new pb.InputBindings({values: values.filter(value => !reuse.includes(value.name)), reuse});
  }
  async #execute(role, feed, contextId = this.contexts[role]) {
    const {pb} = this.#api;
    const result = checked(await this.#inference.execute(new pb.ExecuteRequest({contextId,
      inputs: this.#bindings(role, feed), scope: this.#scope(role, feed)})));
    attest(pb, result.report, this.backend);
    const output = {};
    for (const [semantic, name] of Object.entries(this.manifest.graphs[role].outputs)) {
      const tensor = result.outputs.find(value => value.name === name);
      if (!tensor || ![pb.DataType.DATA_TYPE_F32, pb.DataType.DATA_TYPE_I32].includes(tensor.dtype)) fail(`Unsupported output: ${name}.`);
      const bytes = tensor.inline.slice(), shape = tensor.shape.map(Number);
      if (bytes.byteLength !== shape.reduce((a, b) => a * b, 1) * 4) fail(`Inconsistent output size: ${name}.`);
      output[semantic] = {shape, dtype: tensor.dtype === pb.DataType.DATA_TYPE_F32 ? 'float32' : 'int32',
        data: tensor.dtype === pb.DataType.DATA_TYPE_F32 ? new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
          : new Int32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)};
    }
    return {output, result};
  }
  async #prepare(image, question, {questionCapacity = this.questionCapacity, family = -1} = {}) {
    if (this.#closed) fail('Session is closed.');
    if (!String(question).trim()) fail('Enter a question.');
    if (!Number.isSafeInteger(questionCapacity) || questionCapacity < 1 || questionCapacity > this.questionCapacity) fail('Question token capacity is outside the package domain.');
    if (!Number.isInteger(family) || family < -1 || family >= this.families.length) fail('Invalid family selection.');
    const geometry = this.manifest.shape_contract.fixed_geometry;
    const questionIds = this.vocab.encodeQuestion(question, questionCapacity);
    const pixels = await preprocessTinyReceiptImage(image, {width: geometry.image[3], height: geometry.image[2]});
    return {image: f32(pixels, geometry.image), question_ids: i32(questionIds, [1, questionIds.length]),
      question_position_ids: i32(questionIds.map((_, index) => index), [1, questionIds.length]), family_ids: i32([family], [1])};
  }
  async #phase(role, feed, position, capture, phases) {
    const startNs = capture ? (await this.#platform.getMonotonicTime(new this.#api.pb.Empty())).nanoseconds : undefined;
    const started = performance.now(), {output, result} = await this.#execute(role, feed), wallMs = performance.now() - started;
    const endNs = capture ? (await this.#platform.getMonotonicTime(new this.#api.pb.Empty())).nanoseconds : undefined;
    const phase = {role, phase: role === 'encoder' ? 'encoder' : position === 0 ? 'prefill' : 'decode', position,
      contextId: this.contexts[role], modelId: this.models[role], executionId: result.executionId, wallMs, startNs, endNs,
      ...(role === 'decoder' ? {inputToken: feed.decoder_input_ids.data[0], pastLength: feed.past_padding_mask.shape[1]} : {})};
    phases.push(phase);
    if (capture) await capture.annotate(role === 'encoder' ? 'VQA / encoder' :
      `VQA / ${phase.phase} ${position} / token ${phase.inputToken}`, startNs, endNs);
    return {output, phase};
  }
  async run(image, question, options = {}) {
    if (this.#running) fail('A generation is already running.');
    this.#running = true; this.clearGeneration();
    let capture;
    try {
      const {maxTokens = this.maxTokens, sampling = 'greedy', temperature = 1, topK = 40, seed = 1,
        profile = false, onProgress = () => {}} = options;
      if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || maxTokens > this.maxTokens) fail('Generation limit is outside the package domain.');
      if (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295) fail('Seed must be a uint32 integer.');
      const prepareStarted = performance.now(), feed = await this.#prepare(image, question, options);
      const preprocessMs = performance.now() - prepareStarted;
      if (profile) capture = await startEngineTrace(this.#api, this.#host, this.runtimeId);
      const started = performance.now(), phases = [], tokens = [], tokenRows = [], random = seededRandom(seed);
      onProgress({phase: 'encoder', position: 0, maxTokens});
      const {output: encoded} = await this.#phase('encoder', feed, null, capture, phases);
      const {pb} = this.#api;
      checked(await this.#inference.releaseExecutionContext(new pb.ExecutionContextRef({contextId: this.contexts.decoder})));
      this.contexts.decoder = checked(await this.#inference.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: this.compiled.decoder, graph: new pb.GraphExecution(),
      }))).contextId;
      const decoder = receiptDecoderFeed(this.manifest, encoded, this.vocab.bos);
      for (let position = 0; position < maxTokens; position++) {
        onProgress({phase: position === 0 ? 'prefill' : 'decode', position, maxTokens});
        const {output, phase} = await this.#phase('decoder', decoder, position, capture, phases);
        const token = chooseReceiptToken(output.logits.data, {sampling, temperature, topK}, random); phase.outputToken = token;
        tokens.push(token); tokenRows.push({...phase, literal: this.vocab.itos[token] ?? String(token), presentLength: output.present_padding_mask.shape[1]});
        advanceReceiptDecoder(this.manifest, decoder, output, token, position);
        if (token === this.vocab.eos) break;
      }
      const wallMs = performance.now() - started, trace = capture ? await capture.finish() : null;
      const result = {tokens, tokenRows, phases, wallMs, preprocessMs, encoderMs: phases[0].wallMs,
        decoderMs: tokenRows.reduce((total, row) => total + row.wallMs, 0), family: encoded.selected_family_ids.data[0],
        questionTokens: feed.question_ids.shape[1], memoryTokens: encoded.memory_padding_mask.shape[1],
        reachedEos: tokens.at(-1) === this.vocab.eos, parsed: parseReceiptAnswer(this.vocab, tokens), trace,
        provenance: {question, questionCapacity: options.questionCapacity ?? this.questionCapacity, requestedFamily: options.family ?? -1,
          sampling, temperature, topK, seed, maxTokens}};
      this.lastRun = result; this.#lastInput = feed; return result;
    } finally { this.#running = false; if (capture) await capture.release(); }
  }
  clearGeneration() { this.lastRun = null; this.#lastInput = null; }

  /** Reconstruct the selected call using independent contexts; never overwrite the normal KV sequence. */
  async createDebugSession(image, question, {target = 'encoder', position = 1, preserveNodeBoundaries = true,
    capture, limits, ...options} = {}) {
    if (this.#running) fail('Wait for generation to finish.');
    if (!['encoder', 'prefill', 'decode'].includes(target)) fail('Unknown debug target.');
    const encoderFeed = await this.#prepare(image, question, options), role = target === 'encoder' ? 'encoder' : 'decoder';
    const {pb} = this.#api, contexts = [], run = this.lastRun;
    if (target === 'decode' && (!Number.isSafeInteger(position) || position < 1 || !run || position >= run.tokens.length))
      fail('Choose a reached decode position after running the current input.');
    if (target === 'decode' && Object.entries(encoderFeed).some(([name, value]) => {
      const prior = this.#lastInput?.[name];
      return !prior || value.data.length !== prior.data.length || value.shape.join(',') !== prior.shape.join(',') ||
        !value.data.every((number, index) => number === prior.data[index]);
    })) fail('The image/question/family inputs changed; run them before debugging a decode token.');
    let compiledModelId = this.compiled[role], feed = encoderFeed;
    try {
      if (preserveNodeBoundaries) {
        const compiled = checked(await this.#inference.compileModel(new pb.CompileModelRequest({modelId: this.models[role],
          preserveNodeBoundaries: true, policy: this.#policy()})));
        compiledModelId = compiled.compiledModelId; attest(pb, compiled.report, this.backend, true);
      }
      if (role === 'decoder') {
        for (const name of roles) contexts.push(checked(await this.#inference.createExecutionContext(
          new pb.CreateExecutionContextRequest({compiledModelId: name === role ? compiledModelId : this.compiled[name], graph: new pb.GraphExecution()}))).contextId);
        const {output: encoded} = await this.#execute('encoder', encoderFeed, contexts[0]);
        feed = receiptDecoderFeed(this.manifest, encoded, this.vocab.bos);
        const targetPosition = target === 'prefill' ? 0 : position;
        for (let index = 0; index < targetPosition; index++) {
          const {output} = await this.#execute('decoder', feed, contexts[1]);
          advanceReceiptDecoder(this.manifest, feed, output, run.tokens[index], index);
        }
      }
      if (role === 'encoder') contexts.push(checked(await this.#inference.createExecutionContext(
        new pb.CreateExecutionContextRequest({compiledModelId, graph: new pb.GraphExecution()}))).contextId);
      const debug = await EngineDebugSession.create(this.#api, this.#host, {
        execute: new pb.ExecuteRequest({contextId: contexts.at(-1),
          inputs: this.#bindings(role, feed), scope: this.#scope(role, feed)}),
        capture: new pb.DebugCapture(capture), limits: new pb.DebugLimits(limits)});
      debug.target = {role, target, position: role === 'encoder' ? null : target === 'prefill' ? 0 : position,
        ...(role === 'decoder' ? {inputToken: feed.decoder_input_ids.data[0], pastLength: feed.past_padding_mask.shape[1],
          caches: Object.entries(feed).filter(([name]) => name.startsWith('past_') || name.startsWith('cross_'))
            .map(([name, value]) => ({semantic: name, name: this.manifest.graphs.decoder.inputs[name], shape: value.shape, dtype: value.dtype}))} : {})};
      debug.sourceDetails = {...debug.target, question, questionCapacity: options.questionCapacity ?? this.questionCapacity,
        questionTokens: encoderFeed.question_ids.shape[1], questionTokenIds: Array.from(encoderFeed.question_ids.data),
        requestedFamily: options.family ?? -1, prefixTokens: role === 'decoder' ? run?.tokens.slice(0, debug.target.position) ?? [] : [],
        inputs: Object.entries(feed).map(([semantic, value]) => ({semantic, name: this.manifest.graphs[role].inputs[semantic],
          shape: value.shape, dtype: value.dtype})), cacheContract: this.manifest.cache_contract,
        ...(role === 'decoder' ? {memoryTokens: feed.memory_padding_mask.shape[1]} : {})};
      return debug;
    } finally {
      if (compiledModelId !== this.compiled[role]) checked(await this.#inference.releaseCompiledModel(new pb.CompiledModelRef({compiledModelId})));
      for (const contextId of contexts.reverse()) checked(await this.#inference.releaseExecutionContext(new pb.ExecutionContextRef({contextId})));
    }
  }
  phaseForProfile(row) {
    const lineage = row?.lineage;
    if (lineage?.modelId === this.models.encoder || lineage?.contextId === this.contexts.encoder) return {target: 'encoder', position: 0};
    const phase = this.lastRun?.phases.find(item => item.role === 'decoder' && item.executionId === lineage?.executionId);
    const position = phase?.position ?? 0;
    return {target: position === 0 ? 'prefill' : 'decode', position};
  }
  close() { this.#closed ??= this.#host.close(); return this.#closed; }
}
