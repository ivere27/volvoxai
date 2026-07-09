// Experimental WebNN (navigator.ml) backend. Maps the Volvox graph onto MLGraphBuilder
// ops and runs on the platform NPU/GPU. It covers feed-forward/vision graphs; ops it
// can't express throw during build so VolvoxAI cleanly falls back to a lower
// tier. Note: graph.tensors is a Map and node.inputs/outputs hold Tensor OBJECTS.
export class WebNNEngine {
  constructor(context) {
    this.context = context;
    this.graph = null;
    this.compiledGraph = null;
    this.operands = {};
    this.inputs = [];
    this.outputs = [];
  }

  async allocateGraph(graph) {
    this.graph = graph;
    this.operands = {}; this.inputs = []; this.outputs = [];
    const builder = new MLGraphBuilder(this.context);
    // MLOperandDescriptor changed across WebNN versions (dimensions→shape, added dataType).
    // Provide all spellings so constants/inputs build on old and new implementations.
    const desc = (shape) => { const d = shape.length ? shape : [1]; return { dataType: 'float32', type: 'float32', shape: d, dimensions: d }; };

    // Weights become constants; anything else not produced by a node is a graph input.
    const generated = new Set();
    for (const node of graph.nodes)
      for (const t of Object.values(node.outputs)) if (t && t.name) generated.add(t.name);

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

    for (const node of graph.nodes) {
      const op = node.opType;
      const outName = Object.values(node.outputs)[0].name;
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
          this.operands[outName] = builder.gelu(getOp(nin(node, "input")));
        } else if (op === "SiLU" || op === "Swish") {
          const x = getOp(nin(node, "input"));
          this.operands[outName] = builder.mul(x, builder.sigmoid(x));
        } else if (op === "Sigmoid") {
          this.operands[outName] = builder.sigmoid(getOp(nin(node, "input")));
        } else if (op === "Softmax") {
          this.operands[outName] = builder.softmax(getOp(nin(node, "input")));
        } else if (op === "Reshape" || op === "Flatten") {
          const shape = Object.values(node.outputs)[0].shape;
          this.operands[outName] = builder.reshape(getOp(nin(node, "input")), shape.length ? shape : [1]);
        } else if (op === "LayerNorm") {
          const input = getOp(nin(node, "input"));
          const scale = nin(node, "weight") ? getOp(nin(node, "weight")) : undefined;
          const bias = nin(node, "bias") ? getOp(nin(node, "bias")) : undefined;
          const inShape = node.inputs.input.shape;
          this.operands[outName] = builder.layerNormalization(input, { axes: [inShape.length - 1], scale, bias });
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
          const seq = qkvT.shape[1];
          const d = Math.floor(qkvT.shape[2] / 3);            // d_model
          const heads = node.params.heads || 8;
          const hd = Math.floor(d / heads);                    // head_dim
          const scale = node.params.scale !== undefined ? node.params.scale : 1 / Math.sqrt(hd);
          const qkv = builder.reshape(getOp(nin(node, "qkv")), [seq, 3 * d]);
          const Q = builder.slice(qkv, [0, 0], [seq, d]);
          const K = builder.slice(qkv, [0, d], [seq, d]);
          const V = builder.slice(qkv, [0, 2 * d], [seq, d]);
          const toHeads = (x) => builder.transpose(builder.reshape(x, [seq, heads, hd]), { permutation: [1, 0, 2] }); // [heads, seq, hd]
          const Qh = toHeads(Q), Vh = toHeads(V);
          const Kh = builder.transpose(builder.reshape(K, [seq, heads, hd]), { permutation: [1, 2, 0] });             // [heads, hd, seq]
          let scores = builder.matmul(Qh, Kh);                 // [heads, seq, seq]
          scores = builder.mul(scores, builder.constant(desc([1]), new Float32Array([scale])));
          const mask = new Float32Array(seq * seq);            // causal: 0 for j<=i, -inf above
          for (let i = 0; i < seq; i++) for (let j = 0; j < seq; j++) mask[i * seq + j] = j <= i ? 0 : -1e9;
          scores = builder.add(scores, builder.constant(desc([seq, seq]), mask));   // broadcast over heads
          const attn = builder.softmax(scores, 2);             // over the last axis (keys)
          let out = builder.matmul(attn, Vh);                  // [heads, seq, hd]
          out = builder.transpose(out, { permutation: [1, 0, 2] });   // [seq, heads, hd]
          this.operands[outName] = builder.reshape(out, [1, seq, d]);
        } else {
          // Ops WebNN can't express here or that have not been wired yet.
          // Throw so VolvoxAI falls back to a tier that implements them.
          throw new Error(`Unsupported op in WebNNEngine: ${op}`);
        }
      } catch (e) {
        console.warn(`[WebNN] cannot map op ${op}: ${e.message}`);
        throw e;   // bubble up → VolvoxAI tries the next tier
      }
    }

    const outputOperands = {};
    for (const name of graph.outputNames) {
      if (this.operands[name]) { outputOperands[name] = this.operands[name]; this.outputs.push(name); }
    }
    console.log(`[WebNN] building graph, outputs:`, this.outputs);
    this.compiledGraph = await builder.build(outputOperands);
    console.log(`[WebNN] graph compiled.`);
    return this;
  }

  async execute(inputsMap) {
    if (!this.compiledGraph) throw new Error("WebNN graph not compiled.");
    const ctx = this.context;
    const numel = (name) => { const t = this.graph.tensors.get(name); return t ? t.shape.reduce((a, b) => a * b, 1) : 1; };
    const shapeOf = (name) => { const t = this.graph.tensors.get(name); return t && t.shape.length ? t.shape : [1]; };

    // Legacy API: MLContext.compute(graph, inputs, outputs) with plain ArrayBufferViews.
    if (typeof ctx.compute === "function") {
      const ins = {}, outs = {};
      for (const name of this.inputs) ins[name] = inputsMap[name] || new Float32Array(numel(name));
      for (const name of this.outputs) outs[name] = new Float32Array(numel(name));
      return (await ctx.compute(this.compiledGraph, ins, outs)).outputs;
    }

    // Current API: MLTensor + dispatch + readTensor.
    const inT = {}, outT = {};
    for (const name of this.inputs) {
      const t = await ctx.createTensor({ dataType: "float32", shape: shapeOf(name), dimensions: shapeOf(name), writable: true });
      ctx.writeTensor(t, inputsMap[name] || new Float32Array(numel(name)));
      inT[name] = t;
    }
    for (const name of this.outputs)
      outT[name] = await ctx.createTensor({ dataType: "float32", shape: shapeOf(name), dimensions: shapeOf(name), readable: true });

    ctx.dispatch(this.compiledGraph, inT, outT);

    const results = {};
    for (const name of this.outputs) { const ab = await ctx.readTensor(outT[name]); results[name] = new Float32Array(ab); outT[name].destroy?.(); }
    for (const name of this.inputs) inT[name].destroy?.();
    return results;
  }
}
