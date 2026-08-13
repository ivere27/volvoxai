// Experimental WebNN (navigator.ml) backend. Maps the Volvox graph onto MLGraphBuilder
// ops and runs on the platform NPU/GPU. It covers feed-forward/vision graphs; ops it
// can't express throw during build so VolvoxAI cleanly falls back to a lower
// tier. Note: graph.tensors is a Map and node.inputs/outputs hold Tensor OBJECTS.
import { geluApproximation } from '../ops/gELU.js';
import { assertInferenceExecutionOptions, BackendEngine } from './BackendEngine.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { RuntimeDType, RuntimeTypedArray, TensorLike } from '../types.js';

// WebNN remains a moving browser API. Keep its platform objects structural at
// this boundary while preserving concrete graph/tensor contracts internally.
declare const MLGraphBuilder: new (context: any) => any;

function firstOutput(node: { outputs: Record<string, TensorLike> }): TensorLike {
  return Object.values(node.outputs)[0]!;
}

function storageDType(value: unknown): RuntimeDType | null {
  if (value instanceof Float32Array) return 'float32';
  if (value instanceof Int32Array) return 'int32';
  if (value instanceof Int8Array) return 'int8';
  if (value instanceof Uint8Array || value instanceof Uint8ClampedArray) return 'uint8';
  return null;
}

function storageFor(dtype: RuntimeDType, source: ArrayBuffer | ArrayBufferView): RuntimeTypedArray {
  const bytes = source instanceof ArrayBuffer
    ? new Uint8Array(source)
    : new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
  const owned = bytes.slice().buffer;
  if (dtype === 'float32') return new Float32Array(owned);
  if (dtype === 'int32') return new Int32Array(owned);
  if (dtype === 'int8') return new Int8Array(owned);
  return new Uint8Array(owned);
}

function webnnStorageView(
  dtype: RuntimeDType,
  value: RuntimeTypedArray | ArrayBuffer,
): RuntimeTypedArray {
  if (value instanceof ArrayBuffer) {
    if (dtype === 'float32') return new Float32Array(value);
    if (dtype === 'int32') return new Int32Array(value);
    if (dtype === 'int8') return new Int8Array(value);
    return new Uint8Array(value);
  }
  if (dtype === 'uint8' && value instanceof Uint8ClampedArray) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  return value;
}

function spatialPair(
  value: unknown,
  fallback: number,
  label: string,
  minimum = 0,
): [number, number] {
  const values = value === undefined ? [fallback, fallback] : Array.isArray(value) ? value : [value, value];
  if (values.length < 1 || values.length > 2 ||
      values.some((entry) => !Number.isSafeInteger(entry) || (entry as number) < minimum)) {
    throw new Error(
      `[WebNN] ${label} must contain one or two integers no smaller than ${minimum}.`,
    );
  }
  return [values[0] as number, (values[1] ?? values[0]) as number];
}

function webnnConvPadding(params: Record<string, unknown>): [number, number, number, number] {
  if (params.pads !== undefined) {
    if (!Array.isArray(params.pads) || params.pads.length !== 4 ||
        params.pads.some((entry) => !Number.isSafeInteger(entry) || entry < 0)) {
      throw new Error('[WebNN] Conv2D pads must contain top, left, bottom, and right.');
    }
    const [top, left, bottom, right] = params.pads as number[];
    return [top, bottom, left, right];
  }
  const [height, width] = spatialPair(params.padding, 0, 'Conv2D padding');
  return [height, height, width, width];
}

function fusedActivation(builder: any, operand: any, relu: unknown): any {
  if (relu === undefined || relu === 0) return operand;
  if (relu === 1) return builder.relu(operand);
  if (relu === 2) return builder.clamp(operand, { minValue: 0, maxValue: 6 });
  throw new Error('[WebNN] fused activation must be 0, 1, or 2.');
}

export class WebNNEngine extends BackendEngine {
  context: any;
  graph: RuntimeGraph | null;
  compiledGraph: any;
  operands: Record<string, any>;
  inputs: string[];
  outputs: string[];
  declare compiledWeightRevision: number;
  declare compiledTopologyRevision: number;

  constructor(context: any) {
    super('webnn', { outputLocation: 'host' });
    this.context = context;
    this.graph = null;
    this.compiledGraph = null;
    this.operands = {};
    this.inputs = [];
    this.outputs = [];
  }

  fork() {
    return new WebNNEngine(this.context);
  }

  dispose() {
    this.resetDecodeCache();
    try { this.compiledGraph?.destroy?.(); } catch { /* Disposal is best-effort. */ }
    this.graph = null;
    this.compiledGraph = null;
    this.operands = {};
    this.inputs = [];
    this.outputs = [];
  }

  /** Release the provider-owned MLContext after every fork has drained. */
  destroyContext() {
    this.dispose();
    try { this.context?.destroy?.(); } catch { /* Context destruction is terminal and best-effort. */ }
    this.context = null;
  }

  async allocateGraph(graph: RuntimeGraph) {
    this._assertPortableQuantizedGraph(graph);
    this.resetDecodeCache();
    try { this.compiledGraph?.destroy?.(); } catch { /* A new graph supersedes it. */ }
    this.compiledGraph = null;
    this.graph = graph;
    this.operands = {}; this.inputs = []; this.outputs = [];
    const builder = new MLGraphBuilder(this.context);
    const desc = (shape, dtype: RuntimeDType = 'float32') => ({
      dataType: dtype,
      shape,
    });

    // Weights become constants; anything else not produced by a node is a graph input.
    const generated = new Set();
    for (const node of graph.nodes)
      for (const t of Object.values(node.outputs) as TensorLike[]) if (t?.name) generated.add(t.name);

    for (const t of graph.tensors.values()) {
      if (t.isWeight && t.buffer) {
        this.operands[t.name] = builder.constant(
          desc(t.shape, t.dtype),
          webnnStorageView(t.dtype, t.buffer),
        );
        generated.add(t.name);
      } else if (!generated.has(t.name)) {
        this.inputs.push(t.name);
        this.operands[t.name] = builder.input(t.name, desc(t.shape, t.dtype));
        generated.add(t.name);
      }
    }

    const getOp = (name) => {
      if (!name || !this.operands[name]) {
        throw new Error(`[WebNN] missing graph operand '${String(name)}'.`);
      }
      return this.operands[name];
    };
    const nin = (node, key) => (node.inputs[key] ? node.inputs[key].name : null);

    for (const node of graph.nodes as any[]) {
      const op = node.opType;
      const outName = firstOutput(node).name;
      try {
        if (op === "MatMul" || op === "Linear" || op === "Gemm") {
          const a = getOp(nin(node, "input") || nin(node, "a"));
          const w = getOp(nin(node, "weight") || nin(node, "b"));
          // node.wLayout: 'dout' = [d_out, d_in] (needs Wᵀ), 'din' = [d_in, d_out] (x·W).
          const effectiveWeight = node.wLayout === "dout"
            ? builder.transpose(w, { permutation: [1, 0] })
            : w;
          let res = builder.matmul(a, effectiveWeight);
          if (nin(node, "bias")) res = builder.add(res, getOp(nin(node, "bias")));
          this.operands[outName] = res;
        } else if (op === "Add") {
          const result = builder.add(
            getOp(nin(node, "a") || nin(node, "input")),
            getOp(nin(node, "b")),
          );
          this.operands[outName] = fusedActivation(builder, result, node.params.relu);
        } else if (op === "Mul") {
          this.operands[outName] = builder.mul(getOp(nin(node, "a") || nin(node, "input")), getOp(nin(node, "b")));
        } else if (op === "ReLU") {
          this.operands[outName] = builder.relu(getOp(nin(node, "input")));
        } else if (op === "GELU") {
          if (geluApproximation(node) !== 'none') {
            throw new Error(`WebNN GELU node ${node.id} supports only approximate='none'.`);
          }
          this.operands[outName] = builder.gelu(getOp(nin(node, "input")));
        } else if (op === "SiLU") {
          const x = getOp(nin(node, "input"));
          this.operands[outName] = builder.mul(x, builder.sigmoid(x));
        } else if (op === "Sigmoid") {
          this.operands[outName] = builder.sigmoid(getOp(nin(node, "input")));
        } else if (op === "Softmax") {
          this.operands[outName] = builder.softmax(getOp(nin(node, "input")));
        } else if (op === "Reshape" || op === "Flatten") {
          const shape = firstOutput(node).shape;
          this.operands[outName] = builder.reshape(getOp(nin(node, "input")), shape);
        } else if (op === "LayerNorm") {
          const input = getOp(nin(node, "input"));
          const scale = nin(node, "weight") ? getOp(nin(node, "weight")) : undefined;
          const bias = nin(node, "bias") ? getOp(nin(node, "bias")) : undefined;
          const inShape = node.inputs.input.shape;
          this.operands[outName] = builder.layerNormalization(input, {
            axes: [inShape.length - 1], scale, bias, epsilon: node.params.eps ?? 1e-6,
          });
        } else if (op === "Conv2D") {
          if ((node.params.data_layout ?? 'NHWC') !== 'NHWC' ||
              (node.params.weight_layout ?? 'HWIO') !== 'HWIO') {
            throw new Error('WebNN Conv2D requires NHWC input and HWIO filter layouts.');
          }
          const result = builder.conv2d(
            getOp(nin(node, "input")),
            getOp(nin(node, "weight")),
            {
              bias: nin(node, "bias") ? getOp(nin(node, "bias")) : undefined,
              inputLayout: 'nhwc',
              filterLayout: 'hwio',
              strides: spatialPair(node.params.stride, 1, 'Conv2D stride', 1),
              padding: webnnConvPadding(node.params),
              dilations: spatialPair(node.params.dilation, 1, 'Conv2D dilation', 1),
              groups: node.params.groups ?? 1,
            },
          );
          this.operands[outName] = fusedActivation(builder, result, node.params.relu);
        } else if (op === "Embedding") {
          // Row gather: table[vocab, d_model] indexed by token ids.
          const input = node.inputs.input;
          const idx = input.dtype === 'int32'
            ? getOp(nin(node, "input"))
            : builder.cast(getOp(nin(node, "input")), "int32");
          this.operands[outName] = builder.gather(getOp(nin(node, "weight")), idx, { axis: 0 });
        } else if (op === "SDPA") {
          // SDPA is not a hardware/API primitive — decompose it into ops WebNN has:
          //   softmax( (Q·Kᵀ)·scale + causal_mask ) · V, batched over heads.
          const qkvT = node.inputs.qkv;
          const outputShape = firstOutput(node).shape;
          const rank = qkvT.shape.length;
          const batch = rank === 2 ? 1 : qkvT.shape[0];
          const seq = qkvT.shape[rank - 2];
          const d = outputShape[outputShape.length - 1];       // d_model
          const heads = node.params.heads || 8;
          const hd = Math.floor(d / heads);                    // head_dim
          if ((rank !== 2 && rank !== 3) || outputShape.length !== rank ||
              (rank === 3 && outputShape[0] !== batch) || qkvT.shape[rank - 1] !== 3 * d ||
              outputShape[rank - 2] !== seq || heads <= 0 || d % heads !== 0) {
            throw new Error("WebNN SDPA requires compatible rank-2 or rank-3 batched tensors");
          }
          if (nin(node, 'mask')) {
            throw new Error('WebNN SDPA does not support an explicit attention mask.');
          }
          const scale = node.params.scale !== undefined ? node.params.scale : 1 / Math.sqrt(hd);
          const qkv = builder.reshape(getOp(nin(node, "qkv")), [batch, seq, 3 * d]);
          const Q = builder.slice(qkv, [0, 0, 0], [batch, seq, d]);
          const K = builder.slice(qkv, [0, 0, d], [batch, seq, d]);
          const V = builder.slice(qkv, [0, 0, 2 * d], [batch, seq, d]);
          const toHeads = (x) => builder.transpose(builder.reshape(x, [batch, seq, heads, hd]),
            { permutation: [0, 2, 1, 3] });                  // [batch, heads, seq, hd]
          const Qh = toHeads(Q), Vh = toHeads(V);
          const Kh = builder.transpose(builder.reshape(K, [batch, seq, heads, hd]),
            { permutation: [0, 2, 3, 1] });                  // [batch, heads, hd, seq]
          let scores = builder.matmul(Qh, Kh);                // [batch, heads, seq, seq]
          scores = builder.mul(scores, builder.constant(desc([1]), new Float32Array([scale])));
          if (node.params.causal === true) {
            const mask = new Float32Array(seq * seq);          // causal: 0 for j<=i, -inf above
            for (let i = 0; i < seq; i++) {
              for (let j = 0; j < seq; j++) mask[i * seq + j] = j <= i ? 0 : -1e9;
            }
            scores = builder.add(scores, builder.constant(desc([seq, seq]), mask));
          }
          const attn = builder.softmax(scores, 3);             // over the last axis (keys)
          let out = builder.matmul(attn, Vh);                  // [batch, heads, seq, hd]
          out = builder.transpose(out, { permutation: [0, 2, 1, 3] });
          this.operands[outName] = builder.reshape(out, outputShape);
        } else {
          // Ops WebNN can't express here or that have not been wired yet.
          // Throw so VolvoxAI falls back to a tier that implements them.
          throw new Error(`Unsupported op in WebNNEngine: ${op}`);
        }
      } catch (e) {
        // Provider selection is complete before execution. Preserve the exact
        // mapping failure for the caller; do not log or retry another tier.
        throw e;
      }
    }

    const outputOperands: Record<string, any> = {};
    for (const name of graph.outputNames) {
      if (!this.operands[name]) throw new Error(`[WebNN] missing graph output operand '${name}'.`);
      outputOperands[name] = this.operands[name];
      this.outputs.push(name);
    }
    this.compiledGraph = await builder.build(outputOperands);
    this.compiledWeightRevision = graph.weightRevision || 0;
    this.compiledTopologyRevision = graph.topologyRevision || 0;
    return this;
  }

  async execute(inputsMap: Record<string, RuntimeTypedArray>, options: Record<string, any> = {}) {
    assertInferenceExecutionOptions(options, 'WebNN inference');
    if (!this.compiledGraph || !this.graph) throw new Error("WebNN graph not compiled.");
    const graph = this.graph;
    graph.assertTopologyRevision?.(this.compiledTopologyRevision, "WebNN");
    if ((graph.weightRevision || 0) !== this.compiledWeightRevision) {
      throw new Error("WebNN weights changed after compilation; recompile the graph before execution.");
    }
    const ctx = this.context;
    const numel = (name: string) => { const t = graph.tensors.get(name); return t ? t.shape.reduce((a, b) => a * b, 1) : 1; };
    const shapeOf = (name: string) => { const t = graph.tensors.get(name); return t && t.shape.length ? t.shape : [1]; };

    // Ordinary inputs are an atomic request contract.  In particular, never
    // turn an omitted input into an all-zero tensor: that changes model
    // semantics and can make a failed dynamic binding appear successful.
    for (const name of this.inputs) {
      if (!Object.prototype.hasOwnProperty.call(inputsMap, name)) {
        throw new Error(`[WebNN] missing ordinary input '${name}'.`);
      }
      const value = inputsMap[name];
      const expectedDType = graph.tensors.get(name)?.dtype;
      if (storageDType(value) !== expectedDType || value.length !== numel(name)) {
        throw new Error(
          `[WebNN] input '${name}' must contain exactly ${numel(name)} ${String(expectedDType)} elements.`,
        );
      }
    }

    if (typeof ctx.createTensor !== "function" || typeof ctx.writeTensor !== "function" ||
        typeof ctx.dispatch !== "function" || typeof ctx.readTensor !== "function") {
      throw new Error("WebNN context does not implement createTensor/writeTensor/dispatch/readTensor.");
    }
    const inT: Record<string, any> = {};
    const outT: Record<string, any> = {};
    try {
      for (const name of this.inputs) {
        const dtype = graph.tensors.get(name)!.dtype;
        const t = await ctx.createTensor({ dataType: dtype, shape: shapeOf(name), writable: true });
        inT[name] = t;
        ctx.writeTensor(t, webnnStorageView(dtype, inputsMap[name]));
      }
      for (const name of this.outputs) {
        const dtype = graph.tensors.get(name)?.dtype;
        outT[name] = await ctx.createTensor({
          dataType: dtype, shape: shapeOf(name), readable: true,
        });
      }

      ctx.dispatch(this.compiledGraph, inT, outT);

      const results: Record<string, RuntimeTypedArray> = {};
      for (const name of this.outputs) {
        const bytes = await ctx.readTensor(outT[name]);
        const dtype = graph.tensors.get(name)?.dtype;
        if (!dtype || (!(bytes instanceof ArrayBuffer) && !ArrayBuffer.isView(bytes))) {
          throw new Error(`[WebNN] output '${name}' returned invalid storage.`);
        }
        results[name] = storageFor(dtype, bytes);
      }
      return results;
    } finally {
      for (const tensor of [...Object.values(outT), ...Object.values(inT)]) {
        try { tensor?.destroy?.(); } catch { /* Execution failure remains authoritative. */ }
      }
    }
  }
}
