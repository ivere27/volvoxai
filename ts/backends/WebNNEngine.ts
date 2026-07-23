// Experimental WebNN (navigator.ml) backend. Maps the Volvox graph onto MLGraphBuilder
// ops and runs on the platform NPU/GPU. It covers feed-forward/vision graphs; ops it
// can't express throw during build so VolvoxAI cleanly falls back to a lower
// tier. Note: graph.tensors is a Map and node.inputs/outputs hold Tensor OBJECTS.
import { geluApproximation } from '../ops/gELU.js';
import { assertInferenceExecutionOptions, BackendEngine } from './BackendEngine.js';
import type { Graph } from '../core/Graph.js';
import type { TensorLike } from '../types.js';

// WebNN remains a moving browser API. Keep its platform objects structural at
// this boundary while preserving concrete graph/tensor contracts internally.
declare const MLGraphBuilder: new (context: any) => any;

function firstOutput(node: { outputs: Record<string, TensorLike> }): TensorLike {
  return Object.values(node.outputs)[0]!;
}

export class WebNNEngine extends BackendEngine {
  context: any;
  graph: Graph | null;
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
    this.graph = null;
    this.compiledGraph = null;
    this.operands = {};
    this.inputs = [];
    this.outputs = [];
  }

  async allocateGraph(graph: Graph) {
    this._assertPortableQuantizedGraph(graph);
    this.resetDecodeCache();
    if (graph.activeAdapter?.()) {
      throw new Error("WebNN adapter execution is unsupported; compile this graph with the CPU backend.");
    }
    this.graph = graph;
    this.operands = {}; this.inputs = []; this.outputs = [];
    const builder = new MLGraphBuilder(this.context);
    const desc = (shape) => ({ dataType: 'float32', shape: shape.length ? shape : [1] });

    // Weights become constants; anything else not produced by a node is a graph input.
    const generated = new Set();
    for (const node of graph.nodes)
      for (const t of Object.values(node.outputs) as TensorLike[]) if (t?.name) generated.add(t.name);

    for (const t of graph.tensors.values()) {
      if (t.isWeight && t.buffer) {
        this.operands[t.name] = builder.constant(desc(t.shape), t.buffer);
        generated.add(t.name);
      } else if (!generated.has(t.name)) {
        this.inputs.push(t.name);
        this.operands[t.name] = builder.input(t.name, desc(t.shape));
        generated.add(t.name);
      }
    }

    const getOp = (name) => {
      if (!name || !this.operands[name]) {
        console.warn(`[WebNN] missing operand: ${name}`);
        this.operands[name] = builder.constant(desc([1]), new Float32Array([0]));
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
          let res = node.wLayout === "dout" ? builder.gemm(a, w, { bTranspose: true }) : builder.matmul(a, w);
          if (nin(node, "bias")) res = builder.add(res, getOp(nin(node, "bias")));
          this.operands[outName] = res;
        } else if (op === "Add") {
          this.operands[outName] = builder.add(getOp(nin(node, "a") || nin(node, "input")), getOp(nin(node, "b")));
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
          this.operands[outName] = builder.reshape(getOp(nin(node, "input")), shape.length ? shape : [1]);
        } else if (op === "LayerNorm") {
          const input = getOp(nin(node, "input"));
          const scale = nin(node, "weight") ? getOp(nin(node, "weight")) : undefined;
          const bias = nin(node, "bias") ? getOp(nin(node, "bias")) : undefined;
          const inShape = node.inputs.input.shape;
          this.operands[outName] = builder.layerNormalization(input, {
            axes: [inShape.length - 1], scale, bias, epsilon: node.params.eps ?? 1e-6,
          });
        } else if (op === "Conv2D") {
          this.operands[outName] = builder.conv2d(getOp(nin(node, "input")), getOp(nin(node, "weight")),
            { bias: nin(node, "bias") ? getOp(nin(node, "bias")) : undefined,
              strides: node.params.stride, padding: node.params.padding, groups: node.params.groups || 1 });
        } else if (op === "Embedding") {
          // Row gather: table[vocab, d_model] indexed by token ids. WebNN gather needs
          // integer indices; the ids arrive as f32, so cast first.
          const idx = builder.cast(getOp(nin(node, "input")), "int32");
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
          const mask = new Float32Array(seq * seq);            // causal: 0 for j<=i, -inf above
          for (let i = 0; i < seq; i++) for (let j = 0; j < seq; j++) mask[i * seq + j] = j <= i ? 0 : -1e9;
          scores = builder.add(scores, builder.constant(desc([seq, seq]), mask));
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
        const message = e instanceof Error ? e.message : String(e);
        console.warn(`[WebNN] cannot map op ${op}: ${message}`);
        throw e;   // bubble up → VolvoxAI tries the next tier
      }
    }

    const outputOperands: Record<string, any> = {};
    for (const name of graph.outputNames) {
      if (this.operands[name]) { outputOperands[name] = this.operands[name]; this.outputs.push(name); }
    }
    console.log(`[WebNN] building graph, outputs:`, this.outputs);
    this.compiledGraph = await builder.build(outputOperands);
    this.compiledWeightRevision = graph.weightRevision || 0;
    this.compiledTopologyRevision = graph.topologyRevision || 0;
    graph.adapters?._markAcceleratedBackend("webnn");
    console.log(`[WebNN] graph compiled.`);
    return this;
  }

  async execute(inputsMap: Record<string, Float32Array>, options: Record<string, any> = {}) {
    assertInferenceExecutionOptions(options, 'WebNN inference');
    if (!this.compiledGraph || !this.graph) throw new Error("WebNN graph not compiled.");
    const graph = this.graph;
    graph.assertTopologyRevision?.(this.compiledTopologyRevision, "WebNN");
    const adapterPlan = graph.adapters?._pinExecution(options) || null;
    if (adapterPlan) {
      throw new Error("WebNN adapter execution is unsupported; use the CPU backend or merge before compilation.");
    }
    if ((graph.weightRevision || 0) !== this.compiledWeightRevision) {
      throw new Error("WebNN weights changed after compilation; recompile the graph before execution.");
    }
    const ctx = this.context;
    const numel = (name: string) => { const t = graph.tensors.get(name); return t ? t.shape.reduce((a, b) => a * b, 1) : 1; };
    const shapeOf = (name: string) => { const t = graph.tensors.get(name); return t && t.shape.length ? t.shape : [1]; };

    if (typeof ctx.createTensor !== "function" || typeof ctx.writeTensor !== "function" ||
        typeof ctx.dispatch !== "function" || typeof ctx.readTensor !== "function") {
      throw new Error("WebNN context does not implement createTensor/writeTensor/dispatch/readTensor.");
    }
    const inT: Record<string, any> = {};
    const outT: Record<string, any> = {};
    try {
      for (const name of this.inputs) {
        const t = await ctx.createTensor({ dataType: "float32", shape: shapeOf(name), writable: true });
        inT[name] = t;
        ctx.writeTensor(t, inputsMap[name] || new Float32Array(numel(name)));
      }
      for (const name of this.outputs) {
        outT[name] = await ctx.createTensor({
          dataType: "float32", shape: shapeOf(name), readable: true,
        });
      }

      ctx.dispatch(this.compiledGraph, inT, outT);

      const results: Record<string, Float32Array> = {};
      for (const name of this.outputs) {
        const bytes = await ctx.readTensor(outT[name]);
        results[name] = new Float32Array(bytes);
      }
      return results;
    } finally {
      for (const tensor of [...Object.values(outT), ...Object.values(inT)]) {
        try { tensor?.destroy?.(); } catch { /* Execution failure remains authoritative. */ }
      }
    }
  }
}
