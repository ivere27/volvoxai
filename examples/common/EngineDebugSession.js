/** Example adapter over VxDebugService. Observations remain append-only.
 * `request` contains the same Execute request used by ordinary graph calls. */
function checked(value) {
  const report = value.report ?? value;
  if (report.status !== 0) throw new Error(report.message || 'Debug request failed');
  return value;
}

export class EngineDebugSession {
  #client;
  #pb;
  #ref;
  #releasePromise = null;
  /** DebugEvent records; each carries its tensor snapshots. */
  events = [];
  /** Every snapshot, in event order, for lookups by tensor or snapshot ID. */
  tensors = [];

  constructor(client, pb, info) {
    this.#client = client;
    this.#pb = pb;
    this.#ref = new pb.DebugSessionRef({debugSessionId: info.debugSessionId});
    this.info = info;
  }

  static async create(api, host, request) {
    const client = new api.VxDebugServiceClient(host);
    const info = checked(await client.createDebugSession(new api.pb.CreateDebugSessionRequest(request)));
    const session = new EngineDebugSession(client, api.pb, info);
    try {
      session.plan = checked(await client.getDebugPlan(session.#ref)).plan;
      return session;
    } catch (error) {
      await session.release();
      throw error;
    }
  }

  static async createEvaluation(api, host, {compiledModelId, inputs, capture, limits}) {
    const inference = new api.VxInferenceServiceClient(host), {pb} = api;
    const {contextId} = checked(await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId, graph: new pb.GraphExecution(),
    })));
    try {
      return await EngineDebugSession.create(api, host, {
        execute: new pb.ExecuteRequest({contextId, inputs: new pb.InputBindings({values: inputs}),
          scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()})}), capture, limits,
      });
    } finally {
      checked(await inference.releaseExecutionContext(new pb.ExecutionContextRef({contextId})));
    }
  }

  /** A failing model is session state: state FAILED plus info.failure. */
  get failure() { return this.info.failure ?? null; }

  async step() {
    this.info = checked(await this.#client.stepDebugSession(new this.#pb.StepDebugSessionRequest({
      ...this.#ref, expectedRevision: this.info.revision,
    })));
    await this.refresh();
  }

  /** Runs to completion, before a listed source node, or after a NaN/Inf output. */
  async continue({breakBeforeNodes = [], breakOnNonfinite = false} = {}) {
    this.info = checked(await this.#client.continueDebugSession(new this.#pb.ContinueDebugSessionRequest({
      ...this.#ref, expectedRevision: this.info.revision, breakBeforeNodes, breakOnNonfinite,
    })));
    await this.refresh();
  }

  async cancel() {
    this.info = checked(await this.#client.cancelDebugSession(this.#ref));
  }

  /** Re-lists the published events; pages are repeatable and never consumed. */
  async refresh() {
    this.info = checked(await this.#client.getDebugSession(this.#ref));
    const events = [];
    let pageToken = '';
    do {
      const page = checked(await this.#client.listDebugEvents(new this.#pb.ListDebugEventsRequest({
        ...this.#ref, pageToken, pageSize: 1024,
      })));
      events.push(...page.events);
      pageToken = page.nextPageToken;
    } while (pageToken);
    this.events = events;
    this.tensors = events.flatMap(event => event.snapshots);
  }

  /** Decode targets: slot cursors and KV caches at the current stop. */
  async decodeState() {
    return checked(await this.#client.getDebugDecodeState(this.#ref));
  }

  /** Read a bounded live cache page; callers choose the slot and logical token range. */
  async readKVCachePage(cacheId, slot = 0, tokenOffset = 0, tokenLimit = 1) {
    return checked(await this.#client.readDebugKVCache(new this.#pb.ReadDebugKVCacheRequest({
      ...this.#ref, cacheId, slot, tokenOffset, tokenLimit,
    })));
  }

  /** Decode targets: one slot of one KV cache, read live, in token order. */
  async readKVCache(cacheId, slot = 0) {
    const chunks = [];
    let tokenOffset = 0, first;
    for (;;) {
      const chunk = await this.readKVCachePage(cacheId, slot, tokenOffset, 4096);
      first ??= chunk;
      chunks.push(chunk.data);
      const rows = Number(chunk.shape[0] ?? 0n);
      tokenOffset += rows;
      if (tokenOffset >= chunk.tokenCount || !rows) break;
    }
    const data = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.byteLength, 0));
    let offset = 0;
    for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.byteLength; }
    return {dtype: first.dtype, shape: [BigInt(tokenOffset), ...first.shape.slice(1)], data,
      quantization: first.quantization};
  }

  async readTensor(snapshotId, readOffset = 0n, readLimit = 65536) {
    return checked(await this.#client.readDebugTensor(new this.#pb.ReadDebugTensorRequest({
      ...this.#ref, snapshotId, readOffset, readLimit,
    })));
  }

  async tensorBlob(snapshot) {
    const chunks = [];
    let offset = 0n;
    for (;;) {
      const chunk = await this.readTensor(snapshot.snapshotId, offset);
      if (chunk.status !== this.#pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE)
        throw new Error('This snapshot has no retained values');
      chunks.push(chunk.data);
      offset += BigInt(chunk.data.byteLength);
      if (offset >= chunk.sizeBytes) break;
      if (!chunk.data.byteLength) throw new Error('Debug tensor read did not advance');
    }
    const blob = new Blob(chunks, {type: 'application/octet-stream'});
    if (BigInt(blob.size) !== snapshot.logicalBytes) throw new Error('Incomplete debug tensor download');
    return blob;
  }

  toJson() {
    return {
      // Keep diagnostic defaults explicit for tools that do not load the schema.
      session: {...this.info.toJson(), state: this.#pb.DebugState[this.info.state],
        stopReason: this.#pb.DebugStopReason[this.info.stopReason],
        nextStep: this.info.nextStep, captureComplete: this.info.captureComplete},
      plan: this.plan.toJson(),
      events: this.events.map(value => ({...value.toJson(), step: value.step,
        point: this.#pb.DebugPoint[value.point],
        snapshots: value.snapshots.map(snapshot => ({...snapshot.toJson(),
          dtype: this.#pb.DataType[snapshot.dtype], status: this.#pb.DebugTensorStatus[snapshot.status]}))})),
    };
  }

  release() {
    this.#releasePromise ??= this.#client.releaseDebugSession(this.#ref).then(checked);
    return this.#releasePromise;
  }
}
