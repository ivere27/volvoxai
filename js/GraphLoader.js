import { Graph } from './Graph.js';


export class GraphLoader {
  /**
   * Loads a graph from a single Safetensors file.
   * The Safetensors __metadata__ field must contain a 'volvox_nodes' JSON string.
   * @param {Object} graphBuilder - An empty Graph instance from VolvoxAI.createGraph()
   * @param {string} safetensorsUrl - URL to the .safetensors file
   * @returns {Promise<Graph>} Populated graph
   */
  static async load(graphBuilder, safetensorsUrl) {
    let configUrl;
    if (safetensorsUrl.endsWith('_weights.safetensors')) {
      configUrl = safetensorsUrl.replace('_weights.safetensors', '_config.json');
    } else {
      configUrl = safetensorsUrl.replace(/\/[^\/]+$/, '/config.json');
    }
    console.log(`[VolvoxAI] Loading config from ${configUrl}...`);
    const configResponse = await fetch(configUrl);
    if (!configResponse.ok) throw new Error(`Failed to load config.json: ${configResponse.statusText}`);
    const config = await configResponse.json();
    
    console.log(`[VolvoxAI] Loading Safetensors model from ${safetensorsUrl}...`);
    const response = await fetch(safetensorsUrl);
    if (!response.ok) throw new Error(`Failed to load safetensors: ${response.statusText}`);
    const buffer = await response.arrayBuffer();
    const dataView = new DataView(buffer);
    const headerLen = Number(dataView.getBigUint64(0, true));
    const headerBytes = new Uint8Array(buffer, 8, headerLen);
    const headerStr = new TextDecoder("utf-8").decode(headerBytes);
    const header = JSON.parse(headerStr);
    const metadata = header.__metadata__ || {};
    
    const binaryOffset = 8 + headerLen;
    const tensorsMap = /* @__PURE__ */ new Map();
    for (const [name, info] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const dtype = info.dtype === "I8" ? "int8" : info.dtype === "U8" ? "uint8" : "float32";
      const tensor = graphBuilder.addWeight(name, info.shape, dtype);
      const startByte = binaryOffset + info.data_offsets[0];
      const lengthBytes = info.data_offsets[1] - info.data_offsets[0];
      if (dtype === "int8") {
        tensor.buffer = new Int8Array(buffer, startByte, lengthBytes);
      } else if (dtype === "uint8") {
        tensor.buffer = new Uint8Array(buffer, startByte, lengthBytes);
      } else if (info.dtype === "F16") {
        tensor.buffer = GraphLoader._float16ToFloat32Array(buffer, startByte, lengthBytes);
      } else {
        tensor.buffer = new Float32Array(buffer, startByte, lengthBytes / 4);
      }
      tensorsMap.set(name, tensor);
    }
    if (config.inputs) {
      const inputsDef = config.inputs;
      for (const [name, info] of Object.entries(inputsDef)) {
        const tensor = graphBuilder.addInput(name, info.shape, info.dtype || "float32");
        tensorsMap.set(name, tensor);
      }
    }

    if (config.nodes) {
      GraphLoader._buildFromBlueprint(graphBuilder, config, tensorsMap);
    } else if (config.model_type) {
      if (typeof GraphLoader.ModelBuilders === 'undefined' || !GraphLoader.ModelBuilders[config.model_type]) {
        throw new Error(`[VolvoxAI] Unsupported Hugging Face model_type: '${config.model_type}'. No builder registered.`);
      }
      console.log(`[VolvoxAI] Building graph on the fly using model builder for '${config.model_type}'...`);
      GraphLoader.ModelBuilders[config.model_type](graphBuilder, config, tensorsMap);
    } else {
      throw new Error("[VolvoxAI] config.json must contain either 'nodes' (Volvox blueprint) or 'model_type' (Hugging Face).");
    }

    GraphLoader._dequantizeConvWeights(graphBuilder);
    GraphLoader._normalizeConvWeightsForImageLayout(graphBuilder);
    
    // Auto-detect output names: any tensor produced by a node that is NEVER used as an input
    const usedAsInput = new Set();
    for (const node of graphBuilder.nodes) {
        for (const t of Object.values(node.inputs)) {
            if (t && t.name) usedAsInput.add(t.name);
        }
    }
    const outputNames = [];
    for (const node of graphBuilder.nodes) {
        for (const t of Object.values(node.outputs)) {
            if (t && t.name && !usedAsInput.has(t.name)) {
                outputNames.push(t.name);
            }
        }
    }
    graphBuilder.outputNames = outputNames;
    GraphLoader._resolveMatMulLayouts(graphBuilder);
    console.log(`[VolvoxAI] Successfully assembled graph. Outputs:`, outputNames);
    return graphBuilder;
  }

  /**
   * MatMul weights come in two layouts: PyTorch Linear stores [d_out, d_in]
   * (out = x·Wᵀ); GPT-Neo/Conv1D stores [d_in, d_out] (out = x·W). Non-square weights
   * disambiguate by shape; square ones can't, so infer the model-wide convention from
   * the unambiguous weights and tag every MatMul node with `wLayout` ('dout' | 'din').
   */
  static _resolveMatMulLayouts(graph) {
    const isMM = (n) => n.opType === "MatMul" || n.opType === "Linear" || n.opType === "Gemm";
    const dims = (n) => {
      const K = n.inputs.input?.shape?.[n.inputs.input.shape.length - 1];
      const outT = Object.values(n.outputs)[0];
      const N = outT?.shape?.[outT.shape.length - 1];
      return [K, N];
    };
    let din = 0, dout = 0;
    for (const n of graph.nodes) {
      if (!isMM(n) || !n.inputs.weight || n.inputs.scale) continue;
      const w = n.inputs.weight.shape; if (!w || w.length < 2) continue;
      const [K, N] = dims(n); if (K === N) continue;
      if (w[0] === N && w[1] === K) dout++;
      else if (w[0] === K && w[1] === N) din++;
    }
    const model = din > dout ? "din" : "dout";
    for (const n of graph.nodes) {
      if (!isMM(n)) continue;
      const w = n.inputs.weight?.shape; const [K, N] = dims(n);
      if (n.inputs.scale) n.wLayout = "dout";                          // INT8 is [d_out, d_in]
      else if (w && w.length >= 2 && K !== N && w[0] === N && w[1] === K) n.wLayout = "dout";
      else if (w && w.length >= 2 && K !== N && w[0] === K && w[1] === N) n.wLayout = "din";
      else n.wLayout = model;                                          // square → model convention
    }
  }

  static _buildFromBlueprint(graphBuilder, config, tensorsMap) {
    const nodesDef = config.nodes;
    for (const nodeDef of nodesDef) {
      const inputs = {};
      for (const [key, tName] of Object.entries(nodeDef.inputs)) {
        let t = tensorsMap.get(tName) || graphBuilder.tensors.get(tName);
        if (!t) {
          console.warn(`[GraphLoader] Implicitly adding missing graph input '${tName}' with shape [1, 3, 224, 224]`);
          t = graphBuilder.addInput(tName, [1, 3, 224, 224], "float32");
          tensorsMap.set(tName, t);
        }
        inputs[key] = t;
      }
      const outputsShape = nodeDef.outputs_shape || {};
      const opName = nodeDef.opType || nodeDef.op;
      const outTensors = graphBuilder.addOp(opName, inputs, outputsShape, nodeDef.params || {});
      for (const [key, tName] of Object.entries(nodeDef.outputs)) {
        const t = outTensors[key];
        if (!t) continue;
        if (t.name !== tName) {
          graphBuilder.tensors.delete(t.name);
          t.name = tName;
          graphBuilder.tensors.set(tName, t);
        }
        tensorsMap.set(tName, t);
      }
    }
  }

  // Registry for Hugging Face model builders
  static ModelBuilders = {};

  static _float16ToFloat32Array(buffer, byteOffset, lengthBytes) {
    const n = lengthBytes / 2;
    const view = new DataView(buffer, byteOffset, lengthBytes);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = GraphLoader._float16BitsToFloat32(view.getUint16(i * 2, true));
    return out;
  }

  static _float16BitsToFloat32(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x03ff;
    if (exp === 0) {
      if (mant === 0) return sign < 0 ? -0 : 0;
      return sign * Math.pow(2, -14) * (mant / 1024);
    }
    if (exp === 31) return mant ? NaN : sign * Infinity;
    return sign * Math.pow(2, exp - 15) * (1 + mant / 1024);
  }

  /**
   * Conv kernels (JS and WASM) expect float32 weights and have no QConv path, so
   * fold per-output-channel int8 scales into the weights up front. MatMul keeps
   * its int8+scale fast path and is left untouched.
   */
  static _dequantizeConvWeights(graph) {
    for (const node of graph.nodes) {
      if (node.opType !== "Conv2D" && node.opType !== "Conv1D" && node.opType !== "QConv2D") continue;
      const w = node.inputs.weight;
      const s = node.inputs.scale || node.inputs.weight_scale;
      if (!s || !w || w.dtype !== "int8" || !w.buffer) continue;
      const zp = node.inputs.weight_zero_point;
      const outC = w.shape[0];
      const perOut = w.buffer.length / outC;
      const deq = new Float32Array(w.buffer.length);
      for (let oc = 0; oc < outC; oc++) {
        const sc = s.buffer[oc];
        const z = zp && zp.buffer ? zp.buffer[zp.buffer.length === 1 ? 0 : oc] : 0;
        const base = oc * perOut;
        for (let j = 0; j < perOut; j++) deq[base + j] = (w.buffer[base + j] - z) * sc;
      }
      if (node.opType === "QConv2D") {
        const deqWeight = graph.addWeight(`${w.name}__deq_${node.id}`, w.shape, "float32");
        deqWeight.buffer = deq;
        deqWeight.sizeBytes = deq.length * 4;
        node.inputs.weight = deqWeight;
      } else {
        w.buffer = deq;
        w.dtype = "float32";
        w.sizeBytes = deq.length * 4;
      }
      delete node.inputs.scale;
      delete node.inputs.weight_scale;
      delete node.inputs.weight_zero_point;
      if (node.opType === "QConv2D") node.opType = "Conv2D";
    }
  }

  static _normalizeConvWeightsForImageLayout(graph) {
    for (const node of graph.nodes) {
      if (node.opType !== "Conv2D") continue;
      const w = node.inputs.weight;
      const input = node.inputs.input;
      const output = node.outputs.out;
      if (!w || !w.buffer || !input || !output || w.shape.length !== 4) continue;
      const layout = node.params?.weight_layout || "HWIO";
      if (layout === "HWIO" || layout === "HWCM") continue;

      if (layout === "OHWI" || layout === "OIHW") {
        const [ocN, a, b, c] = w.shape;
        const kh = layout === "OHWI" ? a : b;
        const kw = layout === "OHWI" ? b : c;
        const icN = layout === "OHWI" ? c : a;
        const src = w.buffer;
        const dst = new Float32Array(src.length);
        for (let oc = 0; oc < ocN; oc++) {
          for (let y = 0; y < kh; y++) {
            for (let x = 0; x < kw; x++) {
              for (let ic = 0; ic < icN; ic++) {
                dst[(((y * kw + x) * icN + ic) * ocN) + oc] =
                  layout === "OHWI"
                    ? src[(((oc * kh + y) * kw + x) * icN) + ic]
                    : src[(((oc * icN + ic) * kh + y) * kw) + x];
              }
            }
          }
        }
        w.buffer = dst;
        w.shape = [kh, kw, icN, ocN];
        w.sizeBytes = dst.length * 4;
        node.params.weight_layout = "HWIO";
      } else if (layout === "1HWO" || layout === "1HWM") {
        const [, kh, kw, ocN] = w.shape;
        const icN = input.shape[3];
        const mult = output.shape[3] / icN;
        if (!Number.isInteger(mult) || mult <= 0 || icN * mult !== ocN) {
          throw new Error(`[GraphLoader] Invalid depthwise Conv2D shape for node ${node.id}`);
        }
        const src = w.buffer;
        const dst = new Float32Array(src.length);
        for (let y = 0; y < kh; y++) {
          for (let x = 0; x < kw; x++) {
            for (let ic = 0; ic < icN; ic++) {
              for (let m = 0; m < mult; m++) {
                dst[(((y * kw + x) * icN + ic) * mult) + m] =
                  src[((y * kw + x) * ocN) + ic * mult + m];
              }
            }
          }
        }
        w.buffer = dst;
        w.shape = [kh, kw, icN, mult];
        w.sizeBytes = dst.length * 4;
        node.params.weight_layout = "HWCM";
      } else {
        throw new Error(`[GraphLoader] Unsupported Conv2D weight_layout '${layout}'. VolvoxAI uses NHWC/HWIO only.`);
      }
    }
  }
};
