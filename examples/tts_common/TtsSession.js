/** Prepared TTS requests over the generated inference API. */
export function validateManifest(manifest) {
  if (manifest?.format !== 'volvoxai-tts-example/v1') throw new Error('Unsupported TTS manifest format.');
  if (!Number.isInteger(manifest.sample_rate) || manifest.sample_rate <= 0) throw new Error('Invalid audio sample rate.');
  if (!Array.isArray(manifest.cases) || !manifest.cases.length) throw new Error('The example has no prepared requests.');
  if (!['ready', 'unsupported'].includes(manifest.volvox?.status)) throw new Error('Missing VolvoxAI export status.');
  return manifest;
}

export function tensorValues(bytes, descriptor, narrow = false) {
  const {dtype, shape} = descriptor;
  if (!Array.isArray(shape) || !shape.every(n => Number.isSafeInteger(n) && n > 0)) throw new Error('Invalid tensor shape.');
  const width = dtype === 'int64' ? 8 : 4;
  if (!['float32', 'int32', 'int64'].includes(dtype) || bytes.byteLength !== shape.reduce((a, b) => a * b, 1) * width)
    throw new Error(`Invalid ${descriptor.name} tensor bytes or dtype.`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const result = dtype === 'float32' ? new Float32Array(bytes.byteLength / width)
    : dtype === 'int64' && !narrow ? new BigInt64Array(bytes.byteLength / width) : new Int32Array(bytes.byteLength / width);
  for (let i = 0; i < result.length; i++) {
    if (dtype === 'float32') result[i] = view.getFloat32(i * width, true);
    else if (dtype === 'int32') result[i] = view.getInt32(i * width, true);
    else {
      const value = view.getBigInt64(i * width, true);
      if (narrow && (value < -2147483648n || value > 2147483647n)) throw new Error(`${descriptor.name} cannot be represented as I32.`);
      result[i] = narrow ? Number(value) : value;
    }
  }
  return result;
}

export function waveformDifference(actual, expected, {atol = 1e-5, rtol = 1e-4, waveform} = {}) {
  if (actual.length !== expected.length || !actual.length) throw new Error('Waveform lengths differ or are empty.');
  let maxAbs = 0, squaredError = 0, squaredReference = 0, mismatches = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i])) throw new Error('Waveform contains non-finite samples.');
    const error = Math.abs(actual[i] - expected[i]);
    maxAbs = Math.max(maxAbs, error); squaredError += error * error; squaredReference += expected[i] * expected[i];
    if (error > atol + rtol * Math.abs(expected[i])) mismatches++;
  }
  const rmse = Math.sqrt(squaredError / actual.length);
  let pass = mismatches === 0;
  if (waveform) {
    if (![waveform.max_absolute_error, waveform.max_rmse].every(x => Number.isFinite(x) && x >= 0))
      throw new Error('Invalid declared waveform criterion.');
    pass = maxAbs <= waveform.max_absolute_error && rmse <= waveform.max_rmse;
  }
  return {pass, criterion: waveform ? 'declared_waveform_error' : 'elementwise',
    strict_elementwise_pass: mismatches === 0, max_abs: maxAbs, rmse,
    snr_db: squaredError === 0 || squaredReference === 0 ? null : 10 * Math.log10(squaredReference / squaredError),
    mismatched_samples: mismatches, samples: actual.length, atol, rtol,
    ...(waveform ? {waveform_criterion: waveform} : {})};
}

export function pcm16Wav(values, sampleRate) {
  if (!Number.isInteger(sampleRate) || sampleRate < 1 || sampleRate > 384000 || !values.length ||
      values.length > (0xffffffff - 36) / 2) throw new Error('Invalid WAV geometry.');
  const bytes = new Uint8Array(44 + values.length * 2), view = new DataView(bytes.buffer);
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, bytes.length - 8, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, values.length * 2, true);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new Error('Cannot encode non-finite audio.');
    const value = Math.max(-1, Math.min(1, values[i]));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return bytes;
}

export function timingSummary(samples, audioSeconds) {
  if (!samples.length || !samples.every(n => Number.isFinite(n) && n >= 0) || !(audioSeconds > 0))
    throw new Error('Invalid timing samples.');
  const sorted = [...samples].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  const position = .9 * (sorted.length - 1), lower = Math.floor(position);
  const p90 = sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * (position - lower);
  return {samples_ms: samples, median_ms: median, p90_ms: p90,
    min_ms: sorted[0], max_ms: sorted.at(-1), real_time_factor: median / (audioSeconds * 1000)};
}

/** Check every completed call outside its measured interval, including warmup. */
export async function benchmarkPrepared(session, fixture, {warmup = 2, repeats = 5, onProgress = () => {}, reportWaveformDifferences = false} = {}) {
  if (fixture.validation?.deterministic === false) throw new Error('Waveform parity requires a deterministic prepared request.');
  if (fixture.validation?.reference_engine !== 'onnxruntime')
    throw new Error('ONNX comparison requires an explicit ONNX Runtime reference; prepare with validation enabled.');
  const descriptor = fixture.output_files?.find(value => value.name === (fixture.audio_output ?? 'audio'));
  if (!descriptor || descriptor.dtype !== 'float32') throw new Error('Missing F32 ONNX reference waveform.');
  const reference = tensorValues(await session.readBytes(descriptor.path), descriptor);
  const request = await session.prepare(fixture);
  let last, comparison, maximumError = 0, maximumRmse = 0, maximumMismatches = 0, allPass = true, allStrictPass = true;
  async function execute(phase) {
    last = await session.execute(request, fixture.audio_output);
    if (JSON.stringify(last.shape) !== JSON.stringify(descriptor.shape)) throw new Error(`${phase}: output shape differs from ONNX.`);
    comparison = waveformDifference(last.audio, reference, {...session.manifest.tolerance, waveform: fixture.validation?.waveform});
    if (!comparison.pass && !reportWaveformDifferences) throw new Error(`${phase}: numerical comparison failed (max absolute error ${comparison.max_abs}).`);
    maximumError = Math.max(maximumError, comparison.max_abs);
    maximumRmse = Math.max(maximumRmse, comparison.rmse);
    maximumMismatches = Math.max(maximumMismatches, comparison.mismatched_samples);
    allPass &&= comparison.pass; allStrictPass &&= comparison.strict_elementwise_pass;
    return last.elapsed_ms;
  }
  const firstMs = await execute('first execution'), warmups = [], samples = [];
  onProgress('warmup');
  for (let i = 0; i < warmup; i++) warmups.push(await execute(`warmup ${i + 1}`));
  onProgress('measurement');
  for (let i = 0; i < repeats; i++) samples.push(await execute(`measurement ${i + 1}`));
  const seconds = last.audio.length / (fixture.sample_rate ?? session.manifest.sample_rate);
  return {last, summary: {id: fixture.id, first_ms: firstMs, warmup_ms: warmups, audio_seconds: seconds,
    timing: timingSummary(samples, seconds), comparison: {...comparison, pass: allPass,
      strict_elementwise_pass: allStrictPass, max_abs_all_executions: maximumError,
      max_rmse_all_executions: maximumRmse, max_mismatched_samples_all_executions: maximumMismatches}, route: last.report.route}};
}

/** Cleanup failure invalidates a completed benchmark receipt. */
export async function closeBenchmarkSession(session, report) {
  try { if (session) await session.close(); }
  catch (error) {
    report.status = 'failed';
    report.error = [report.error, `Session cleanup failed: ${String(error)}`].filter(Boolean).join('; ');
  }
}

function checked(response) {
  if ((response.report?.status ?? response.status) !== 0) throw new Error(response.report?.message || 'Inference request failed.');
  return response;
}
function attest(report, backend) {
  const route = report.route;
  if (report.backend !== backend || !route?.attested || !route.builtin || route.provider !== backend ||
      !Number.isSafeInteger(route.activeNodes) || route.activeNodes <= 0 ||
      route.activeNodes !== route.selectedNodes || route.fallbackNodes || route.missingNodes ||
      report.fallback?.operatorFallbackUsed || report.compilation?.tierFallbackUsed)
    throw new Error(`The engine did not attest complete ${backend} execution without fallback.`);
}

export class TtsSession {
  static async open({manifestUrl, api, wasmUrl, backend = 'wasm', fetch: read = globalThis.fetch,
    host: sharedHost, onDiagnostic}) {
    if (!['wasm', 'webgpu'].includes(backend)) throw new Error('Expected wasm or webgpu.');
    const base = new URL(manifestUrl);
    const readBytes = async relative => {
      const response = await read(new URL(relative, base));
      if (!response.ok) throw new Error(`Could not load ${relative}: HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    };
    const manifest = validateManifest(JSON.parse(new TextDecoder().decode(await readBytes(base.href))));
    if (manifest.volvox.status !== 'ready') throw new Error(`VolvoxAI export unavailable: ${manifest.volvox.reason}`);
    if (manifest.volvox.pipeline) return TtsPipeline.open({manifest, base, api, wasmUrl, backend, read, readBytes, onDiagnostic});
    const host = sharedHost ?? new (api.FullEngineHost ?? api.EngineHost)({wasmUrl, fetch: read, onDiagnostic});
    const client = new api.VxInferenceServiceClient(host), {pb} = api;
    const session = Object.assign(new TtsSession(), {api, host, client, ownsHost: !sharedHost,
      manifest, backend, readBytes, resources: []});
    try {
      const runtime = checked(await client.createRuntime(new pb.CreateRuntimeRequest()));
      session.resources.push(['releaseRuntime', 'RuntimeRef', {runtimeId: runtime.runtimeId}]);
      const graphDocument = await readBytes(manifest.volvox.graph);
      const graph = JSON.parse(new TextDecoder().decode(graphDocument));
      // Large sources use the schema's streamed path contract. ModelPackage
      // is intentionally limited to 64 MiB on every public transport.
      const integrity = async bytes => new pb.ModelSourceIntegrity({byteLength: BigInt(bytes.byteLength),
        sha256: new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))});
      const graphIntegrity = await integrity(graphDocument), weightIntegrity = [];
      let totalBytes = graphDocument.byteLength, maxBytes = graphDocument.byteLength;
      for (const weight of manifest.volvox.weights) {
        const bytes = await readBytes(weight);
        weightIntegrity.push(await integrity(bytes));
        totalBytes += bytes.byteLength; maxBytes = Math.max(maxBytes, bytes.byteLength);
      }
      const model = checked(await client.loadModel(new pb.LoadModelRequest({runtimeId: runtime.runtimeId,
        graphDocument, weightPaths: manifest.volvox.weights.map(weight => new URL(weight, base).href),
        sourceOptions: new pb.ModelSourceOptions({maxTotalBytes: BigInt(totalBytes), maxShardBytes: BigInt(maxBytes),
          graph: graphIntegrity, weightShards: weightIntegrity})})));
      session.resources.push(['releaseModel', 'ModelRef', {modelId: model.modelId}]);
      const compiled = checked(await client.compileModel(new pb.CompileModelRequest({modelId: model.modelId,
        policy: new pb.BackendPolicy({mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: [backend],
          operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
      session.resources.push(['releaseCompiledModel', 'CompiledModelRef', {compiledModelId: compiled.compiledModelId}]);
      attest(compiled.report, backend);
      const context = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()})));
      session.resources.push(['releaseExecutionContext', 'ExecutionContextRef', {contextId: context.contextId}]);
      return Object.assign(session, {graph,
        contextId: context.contextId, compileReport: compiled.report});
    } catch (error) {
      try { await session.close(); }
      catch (cleanupError) { throw new AggregateError([error, cleanupError], `${error}; cleanup: ${cleanupError}`); }
      throw error;
    }
  }

  async prepare(request) {
    return this.prepareValues(await readPreparedInputs(request, this.readBytes));
  }

  prepareValues(inputs) {
    const {pb} = this.api, mapping = this.manifest.volvox.inputs;
    if (Object.keys(inputs).length !== Object.keys(mapping).length) throw new Error('Incomplete prepared inputs.');
    const values = Object.entries(inputs).map(([semantic, {data, shape}]) => {
      const name = mapping[semantic], spec = this.graph.inputs[name];
      if (!name || !spec) throw new Error(`Unknown input ${semantic}.`);
      if (spec.dtype !== (data instanceof Float32Array ? 'float32' : 'int32'))
        throw new Error(`Input dtype differs from the exported contract: ${name}.`);
      return new pb.Tensor({name, shape: shape.map(BigInt),
        dtype: data instanceof Float32Array ? pb.DataType.DATA_TYPE_F32 : pb.DataType.DATA_TYPE_I32,
        inline: new Uint8Array(data.buffer, data.byteOffset, data.byteLength)});
    });
    const entrypoint = this.manifest.volvox.entrypoint ?? 'evaluate';
    return new pb.ExecuteRequest({contextId: this.contextId, inputs: new pb.InputBindings({values}),
      scope: new pb.ExecutionScope({entrypoint, full: new pb.Empty()}),
      outputs: new pb.OutputOptions({storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE})});
  }

  async executeTensors(request) {
    const started = performance.now(), result = checked(await this.client.execute(request));
    attest(result.report, this.backend);
    const outputs = {};
    for (const [semantic, name] of Object.entries(this.manifest.volvox.outputs)) {
      const tensor = result.outputs.find(t => t.name === name);
      if (!tensor) throw new Error(`Missing output ${semantic}.`);
      const dtype = tensor.dtype === this.api.pb.DataType.DATA_TYPE_F32 ? 'float32'
        : tensor.dtype === this.api.pb.DataType.DATA_TYPE_I32 ? 'int32' : null;
      if (!dtype) throw new Error(`Unsupported output dtype: ${semantic}.`);
      const shape = tensor.shape.map(Number);
      outputs[semantic] = {shape, data: tensorValues(tensor.inline, {name: semantic, dtype, shape})};
    }
    return {outputs, elapsed_ms: performance.now() - started, report: result.report};
  }

  async execute(request, audioOutput = 'audio') {
    const {outputs, elapsed_ms, report} = await this.executeTensors(request);
    const tensor = outputs[audioOutput];
    if (!(tensor?.data instanceof Float32Array)) throw new Error('Missing F32 waveform output.');
    return {audio: tensor.data, shape: tensor.shape, elapsed_ms, report};
  }

  async close() {
    if (this.closed) return;
    this.closed = true;
    const errors = [];
    for (const [method, type, handle] of this.resources.splice(0).reverse()) {
      try { checked(await this.client[method](new this.api.pb[type](handle))); }
      catch (error) { errors.push(error); }
    }
    if (this.ownsHost) try { await this.host.close(); } catch (error) { errors.push(error); }
    if (errors.length) throw new AggregateError(errors, errors.map(String).join('; '));
  }
}

async function readPreparedInputs(request, readBytes) {
  if (!request.input_files?.length) throw new Error('Prepare the example to create binary input files.');
  const values = {};
  for (const descriptor of request.input_files) {
    if (values[descriptor.name]) throw new Error(`Duplicate input ${descriptor.name}.`);
    const bytes = await readBytes(descriptor.path);
    if (descriptor.sha256) {
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const actual = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
      if (actual !== descriptor.sha256) throw new Error(`Prepared input hash mismatch: ${descriptor.name}.`);
    }
    values[descriptor.name] = {data: tensorValues(bytes, descriptor, true), shape: descriptor.shape};
  }
  return values;
}

/** All stages execute on the requested Volvox backend on every invocation.
 * Only declared length metadata controls admission of the prepared next shape. */
export class TtsPipeline {
  static async open({manifest, base, api, wasmUrl, backend, read, readBytes, onDiagnostic}) {
    const entries = manifest.volvox.pipeline;
    if (!Array.isArray(entries) || entries.length < 2 || new Set(entries.map(x => x.name)).size !== entries.length)
      throw new Error('Invalid TTS pipeline stages.');
    const stages = [];
    // One host owns one WASM instance and one WebGPU device. Separate models,
    // contexts and runtimes still isolate the stages' mutable engine state.
    const host = new (api.FullEngineHost ?? api.EngineHost)({wasmUrl, fetch: read, onDiagnostic});
    try {
      for (const entry of entries) {
        const url = new URL(entry.manifest, base);
        const document = validateManifest(JSON.parse(new TextDecoder().decode(await readBytes(url.href))));
        if (document.volvox.pipeline) throw new Error('Nested TTS pipelines are unsupported.');
        stages.push({name: entry.name, session: await TtsSession.open({manifestUrl: url, api, wasmUrl, backend, fetch: read, host, onDiagnostic})});
      }
      return Object.assign(new TtsPipeline(), {manifest, backend, readBytes, stages, host});
    } catch (error) {
      const results = await Promise.allSettled(stages.reverse().map(stage => stage.session.close()));
      const failures = results.filter(result => result.status === 'rejected').map(result => result.reason);
      try { await host.close(); } catch (cleanupError) { failures.push(cleanupError); }
      if (failures.length) throw new AggregateError([error, ...failures], `${error}; cleanup: ${failures.join('; ')}`);
      throw error;
    }
  }

  async prepare(fixture) {
    const values = await readPreparedInputs(fixture, this.readBytes), available = new Set(), required = new Set();
    for (const {session} of this.stages) {
      for (const key of Object.keys(session.manifest.volvox.inputs)) if (!available.has(key)) required.add(key);
      for (const key of Object.keys(session.manifest.volvox.outputs)) available.add(key);
    }
    if (required.size !== Object.keys(values).length || [...required].some(key => !values[key]))
      throw new Error('Pipeline inputs must contain exactly the original request tensors.');
    return values;
  }

  async execute(inputs, audioOutput = 'audio') {
    const started = performance.now(), values = {...inputs}, routes = {};
    for (const {name, session} of this.stages) {
      const required = Object.keys(session.manifest.volvox.inputs);
      const request = session.prepareValues(Object.fromEntries(required.map(key => [key, values[key]])));
      const result = await session.executeTensors(request);
      Object.assign(values, result.outputs); routes[name] = result.report.route;
      const feedback = this.manifest.volvox.shape_feedback;
      if (feedback?.stage === name) {
        const tensor = result.outputs[feedback.output];
        const count = tensor?.data.at(feedback.index);
        if (!Number.isSafeInteger(count) || count <= 0 || count !== feedback.expected)
          throw new Error(`Duration is ${count}; this package admits ${feedback.expected} frames. Prepare the request for its new duration.`);
      }
    }
    const tensor = values[audioOutput];
    if (!(tensor?.data instanceof Float32Array)) throw new Error('Missing pipeline waveform.');
    return {audio: tensor.data, shape: tensor.shape, elapsed_ms: performance.now()-started,
      report: {backend: this.backend, route: {stages: routes}}};
  }

  async close() {
    if (this.closed) return;
    this.closed = true;
    const stages = this.stages.splice(0).reverse();
    const results = await Promise.allSettled(stages.map(stage => stage.session.close()));
    const failures = results.filter(result => result.status === 'rejected').map(result => result.reason);
    try { await this.host.close(); } catch (error) { failures.push(error); }
    if (failures.length) throw new AggregateError(failures, failures.map(String).join('; '));
  }
}
