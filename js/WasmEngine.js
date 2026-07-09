import { _pair } from './Tensor.js';
import { Graph } from './Graph.js';
import { CPUEngine } from './CPUEngine.js';


export class WasmEngine extends CPUEngine {
  constructor(wasmModule) {
    super();
    this.wasmModule = wasmModule;
    this.api = wasmModule.instance.exports;
    this.mem = this.api.memory;
    this.pointers = new Map();
    console.log("[VolvoxAI] WASM Engine ready (Tier 2 Fallback).");
  }
  static async init(wasmUrl) {
    try {
      let buffer;
      if (typeof process !== "undefined" && process.versions && process.versions.node) {
          const fs = await import('fs');
          let url = wasmUrl;
          if (!url.startsWith('/')) url = './' + url;
          buffer = fs.readFileSync(url);
      } else {
          const response = await fetch(wasmUrl);
          if (!response.ok) throw new Error("WASM file not found.");
          buffer = await response.arrayBuffer();
      }
      const env = { expf: Math.exp, logf: Math.log, powf: Math.pow };
      const module = await WebAssembly.instantiate(buffer, { env, math: env });
      return new WasmEngine(module);
    } catch (e) {
      console.warn(`[VolvoxAI] Failed to load WASM from ${wasmUrl}:`, e);
      return null;
    }
  }
  createGraph() {
    return new Graph();
  }
  _alloc(tensor) {
    if (!this.pointers.has(tensor.name)) {
      const ptr = this.api.alloc_bytes(tensor.sizeBytes);
      // Grow WASM memory if the allocation exceeds current buffer
      const needed = ptr + tensor.sizeBytes;
      const currentBytes = this.mem.buffer.byteLength;
      if (needed > currentBytes) {
        const pagesNeeded = Math.ceil((needed - currentBytes) / 65536);
        this.mem.grow(pagesNeeded);
      }
      this.pointers.set(tensor.name, ptr);
      // Create a Float32Array view directly into WASM memory for CPU fallbacks
      const wasmView = new Float32Array(this.mem.buffer, ptr, tensor.sizeBytes / 4);
      if (tensor.isWeight && tensor.buffer) {
         let view = tensor.buffer;
         // The C matmul_f32 reads weights as [d_in, d_out]. PyTorch-style [d_out, d_in]
         // weights (node.wLayout==='dout') are transposed to [d_in, d_out] on the heap.
         const dw = this._doutWeights && this._doutWeights.get(tensor.name);
         if (dw) {
           const { din, dout } = dw;
           const t = new Float32Array(din * dout);
           for (let j = 0; j < dout; j++) for (let k = 0; k < din; k++) t[k * dout + j] = view[j * din + k];
           view = t;
         }
         const src = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
         const bytesToCopy = Math.min(view.byteLength, tensor.sizeBytes);
         new Uint8Array(this.mem.buffer, ptr, bytesToCopy).set(src.subarray(0, bytesToCopy));
      }
      tensor.buffer = wasmView;
    }
    return this.pointers.get(tensor.name);
  }
  allocateGraph(graph) {
    return this.compile(graph);
  }
  compile(graph) {
    console.log("[VolvoxAI WASM] Allocating graph tensors on WASM heap...");
    if (this.api.reset_heap) this.api.reset_heap();
    this.pointers.clear();
    // Weights consumed by a [d_out, d_in]-layout MatMul are transposed to the C
    // kernel's [d_in, d_out] layout during _alloc (keyed by weight name).
    this._doutWeights = new Map();
    for (const n of graph.nodes) {
      if ((n.opType === "MatMul" || n.opType === "Linear" || n.opType === "Gemm") && n.wLayout === "dout" && n.inputs.weight && !n.inputs.scale) {
        const din = n.inputs.input.shape[n.inputs.input.shape.length - 1];
        const dout = n.outputs.out.shape[n.outputs.out.shape.length - 1];
        this._doutWeights.set(n.inputs.weight.name, { din, dout });
      }
    }
    for (const [name, tensor] of graph.tensors.entries()) {
       this._alloc(tensor);
    }
    // Re-create views because mem.grow detaches old views
    for (const [name, tensor] of graph.tensors.entries()) {
       const ptr = this.pointers.get(name);
       tensor.buffer = new Float32Array(this.mem.buffer, ptr, tensor.sizeBytes / 4);
    }
    this.graph = graph;
    return this;
  }
  async execute(inputs) {
    for (const [name, data] of Object.entries(inputs)) {
      const tensor = this.graph.tensors.get(name);
      if (!tensor) continue;
      const ptr = this.pointers.get(name);
      new Float32Array(this.mem.buffer, ptr, data.length).set(data);
    }
    
    for (const node of this.graph.nodes) {
      try {
       const inPtr = this.pointers.get(node.inputs.input?.name);
       const outPtr = this.pointers.get(node.outputs.out?.name);
       const wPtr = this.pointers.get(node.inputs.weight?.name);
       const bPtr = this.pointers.get(node.inputs.bias?.name);

        if (node.opType === "Conv2D") {
          const wPtr = this.pointers.get(node.inputs.weight.name);
          const bPtr = node.inputs.bias ? this.pointers.get(node.inputs.bias.name) : 0;
          const inShape = node.inputs.input.shape;
          const outShape = node.outputs.out.shape;
          const wShape = node.inputs.weight.shape;
          const [sy, sx] = _pair(node.params.stride, 1);
          const [dy, dx] = _pair(node.params.dilation, 1);
          const padPair = _pair(node.params.padding, 0);
          const pads = node.params.pads || [padPair[0], padPair[1], padPair[0], padPair[1]];
          const groups = node.params.groups || 1;
          const relu = node.params.relu ? 1 : 0;
          this.api.conv2d_f32(
              inPtr, outPtr, wPtr, bPtr,
              inShape[0], inShape[1], inShape[2], inShape[3],
              wShape[0], wShape[1], wShape[2], wShape[3],
              outShape[1], outShape[2], sy, sx, pads[0], pads[1],
              groups, relu, dy, dx
          );
       } else if (node.opType === "ConvTranspose2D") {
          const wPtr = this.pointers.get(node.inputs.weight.name);
          const bPtr = node.inputs.bias ? this.pointers.get(node.inputs.bias.name) : 0;
          const [b, in_h, in_w, in_c] = node.inputs.input.shape;
          const [out_b, out_h, out_w, out_c] = node.outputs.out.shape;
          const kh = node.params.kernel[0], kw = node.params.kernel[1];
          const sh = node.params.stride ? node.params.stride[0] : 1;
          const sw = node.params.stride ? node.params.stride[1] : 1;
          const ph = node.params.padding ? node.params.padding[0] : 0;
          const pw = node.params.padding ? node.params.padding[1] : 0;
          this.api.conv_transpose2d_f32(
              inPtr, wPtr, bPtr, outPtr,
              b, in_h, in_w, in_c, out_h, out_w, out_c,
              kh, kw, sh, sw, ph, pw
          );
       } else if (node.opType === "ReduceSum") {
          const inShape = node.inputs.input ? node.inputs.input.shape : node.inputs.data.shape;
          const shape = inShape.length === 2 ? inShape : [1, node.inputs.input ? node.inputs.input.buffer.length : node.inputs.data.buffer.length];
          this.api.reduce_sum_f32(inPtr, outPtr, shape[0], shape[1]);
       } else if (node.opType === "ReduceMean") {
          const inShape = node.inputs.input ? node.inputs.input.shape : node.inputs.data.shape;
          const shape = inShape.length === 2 ? inShape : [1, node.inputs.input ? node.inputs.input.buffer.length : node.inputs.data.buffer.length];
          this.api.reduce_mean_f32(inPtr, outPtr, shape[0], shape[1]);
       } else if (node.opType === "MatMul") {
          const wPtr = this.pointers.get(node.inputs.weight.name);
          const bPtr = node.inputs.bias ? this.pointers.get(node.inputs.bias.name) : 0;
          const d_in = node.inputs.input.shape[node.inputs.input.shape.length-1];
          const d_out = node.outputs.out.shape[node.outputs.out.shape.length-1];
          const flatSeq = node.inputs.input.shape.slice(0, -1).reduce((a,b)=>a*b,1);
          if (node.inputs.scale) {
             const sPtr = this.pointers.get(node.inputs.scale.name);
             this.api.matmul_int8_f32(inPtr, wPtr, sPtr, bPtr, outPtr, flatSeq, d_in, d_out);
          } else {
             this.api.matmul_f32(inPtr, wPtr, bPtr, outPtr, flatSeq, d_in, d_out);
          }
       } else if (node.opType === "LayerNorm") {
          const flatSeq = node.inputs.input.shape.slice(0, -1).reduce((a,b)=>a*b,1);
          const d_model = node.params.d_model;
          this.api.layernorm_f32(inPtr, wPtr, bPtr, outPtr, flatSeq, d_model, 1e-5);
       } else if (node.opType === "SDPA") {
          const qkvPtr = this.pointers.get(node.inputs.qkv.name);
          const seqLen = node.inputs.qkv.shape[1]; // [batch, seq, d]
          const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
          const heads = node.params.heads || node.params.num_heads || 8;
          const head_dim = node.params.head_dim || d_model / heads;
          const scale = node.params.scale !== undefined ? node.params.scale : 1.0 / Math.sqrt(head_dim);
          this.api.sdpa_f32(qkvPtr, outPtr, seqLen, d_model, heads, head_dim, scale);
       } else if (node.opType === "CrossSDPA") {
          const qPtr = this.pointers.get(node.inputs.q.name);
          const kPtr = this.pointers.get(node.inputs.k.name);
          const vPtr = this.pointers.get(node.inputs.v.name);
          const seqQ = node.inputs.q.shape[1];
          const seqKV = node.inputs.k.shape[1];
          const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
          const heads = node.params.heads || node.params.num_heads || 8;
          const head_dim = node.params.head_dim || d_model / heads;
          this.api.cross_sdpa_f32(qPtr, kPtr, vPtr, outPtr, seqQ, seqKV, d_model, heads, head_dim, 1.0 / Math.sqrt(head_dim));
       } else if (node.opType === "Embedding") {
          const seqLen = node.inputs.input.shape.reduce((a, b) => a * b, 1);
          const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
          this.api.embedding_f32(inPtr, wPtr, outPtr, seqLen, d_model);
       } else if (node.opType === "ReLU") {
          const inShape = node.inputs.input.shape;
          const elements = inShape.reduce((a, b) => a * b, 1);
          this.api.relu_f32(inPtr, outPtr, elements);
       } else if (node.opType === "GELU") {
          const elements = node.inputs.input.sizeBytes / 4;
          this.api.gelu_f32(inPtr, outPtr, elements);
       } else if (node.opType === "Add") {
          this._cpuAdd(node);
       } else if (node.opType === "Mul") {
          this._cpuMul(node);
       } else if (node.opType === "Conv1D") {
          const inShape = node.inputs.input.shape;
          const wShape = node.inputs.weight.shape;
          const [st] = _pair(node.params.stride, 1);
          const [pd] = _pair(node.params.padding, 0);
          const groups = node.params.groups || 1;
          const relu = node.params.relu ? 1 : 0;
          this.api.conv1d_f32(
              inPtr, outPtr, wPtr, bPtr,
              inShape[1], inShape[2], wShape[0], wShape[1], wShape[2], st, pd, groups, relu
          );
       } else if (node.opType === "UpsampleNearest2D") {
          const inShape = node.inputs.input.shape;
          this.api.upsample_nearest2x_f32(inPtr, outPtr, inShape[3], inShape[1], inShape[2]);
       } else if (node.opType === "Concat" || node.opType === "Concat2") {
          // N-way channel concat via the heap-view buffers (like Add/Mul above);
          // the C concat2_f32 only handled two inputs.
          this._cpuConcat2(node);
       } else if (node.opType === "ProfileY") {
          const inShape = node.inputs.input.shape;
          this.api.profile_y_f32(inPtr, outPtr, inShape[3], inShape[1], inShape[2]);
       } else if (node.opType === "ProfileX") {
          const inShape = node.inputs.input.shape;
          this.api.profile_x_f32(inPtr, outPtr, inShape[3], inShape[1], inShape[2]);
       } else if (node.opType === "InterpLinear1D") {
          const inShape = node.inputs.input.shape; // [1, C, L]
          const outL = node.params.size;
          this.api.interp1d_f32(inPtr, outPtr, inShape[1], inShape[2], outL);
       } else if (node.opType === "SpatialSoftargmaxY") {
           const inShape = node.inputs.input.shape;
           this.api.spatial_softargmax_y_f32(inPtr, outPtr, inShape[3], inShape[1], inShape[2]);
        } else if (node.opType === "Sigmoid") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.sigmoid_f32(inPtr, outPtr, elements);
        } else if (node.opType === "Clip") {
           let minVal = node.params.min !== undefined ? node.params.min : -1e9;
           let maxVal = node.params.max !== undefined ? node.params.max : 1e9;
           if (node.inputs.min) {
              const p = this.pointers.get(node.inputs.min.name);
              minVal = new Float32Array(this.mem.buffer, p, 1)[0];
           }
           if (node.inputs.max) {
              const p = this.pointers.get(node.inputs.max.name);
              maxVal = new Float32Array(this.mem.buffer, p, 1)[0];
           }
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.clip_f32(inPtr, outPtr, elements, minVal, maxVal);
        } else if (node.opType === "HardSwish") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.hardswish_f32(inPtr, outPtr, elements);
       } else if (node.opType === "LeakyReLU") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           const alpha = node.params.alpha || 0.01;
           if (this.api.leakyrelu_f32) this.api.leakyrelu_f32(inPtr, outPtr, elements, alpha);
           else this._cpuLeakyReLU(node);
       } else if (node.opType === "PReLU") {
           if (this.api.prelu_f32) {
               const inS = node.inputs.input.shape;
               const wPtr = this.pointers.get(node.inputs.weight.name);
               this.api.prelu_f32(inPtr, wPtr, outPtr, inS[0], inS[1], inS[2], inS[3]);
           } else {
               this._cpuPReLU(node);
           }
        } else if (node.opType === "HardSigmoid") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.hardsigmoid_f32(inPtr, outPtr, elements);
        } else if (node.opType === "Reshape") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.copy_f32(inPtr, outPtr, elements);
        } else if (node.opType === "Transpose") {
           this._cpuTranspose(node);
        } else if (node.opType === "GlobalAveragePool") {
           const inShape = node.inputs.input.shape;
           this.api.global_average_pool_f32(inPtr, outPtr, inShape[0], inShape[1], inShape[2], inShape[3]);
        } else if (node.opType === "BatchNorm2D") {
           const wPtr = this.pointers.get(node.inputs.weight.name);
           const bPtr = this.pointers.get(node.inputs.bias.name);
           const rmPtr = this.pointers.get(node.inputs.running_mean.name);
           const rvPtr = this.pointers.get(node.inputs.running_var.name);
           const [b, h, w, c] = node.inputs.input.shape;
           const eps = node.params.eps || 1e-5;
           this.api.batch_norm2d_f32(inPtr, wPtr, bPtr, rmPtr, rvPtr, outPtr, b, h, w, c, eps);
        } else if (node.opType === "ResizeNearest2D") {
           this._cpuResize(node);
        } else if (node.opType === "Resize") {
           const [b, in_h, in_w, c] = node.inputs.input.shape;
           const [ob, out_h, out_w] = node.outputs.out.shape;
           this.api.resize_bilinear_f32(inPtr, outPtr, b, in_h, in_w, c, out_h, out_w);
       } else if (node.opType === "Cast") {
           this._cpuCast(node); // Keep cast in JS due to Float32Array aliasing
        } else if (node.opType === "Slice") {
           const in_s = [1,1,1,1].slice(0, 4-node.inputs.input.shape.length).concat(node.inputs.input.shape);
           const out_s = [1,1,1,1].slice(0, 4-node.outputs.out.shape.length).concat(node.outputs.out.shape);
           const starts = node.params.starts || [0,0,0,0]; const steps = node.params.steps || [1,1,1,1]; const axes = node.params.axes || [0,1,2,3];
           const st = [0,0,0,0]; const sp = [1,1,1,1];
           for(let i=0; i<axes.length; i++) {
               let ax = axes[i]; if(ax < 0) ax += node.inputs.input.shape.length; ax += (4 - node.inputs.input.shape.length);
               st[ax] = starts[i] < 0 ? starts[i] + in_s[ax] : starts[i]; sp[ax] = steps[i];
           }
           this.api.slice_4d_f32(inPtr, outPtr, in_s[0], in_s[1], in_s[2], in_s[3], out_s[0], out_s[1], out_s[2], out_s[3], st[0], st[1], st[2], st[3], sp[0], sp[1], sp[2], sp[3]);
        } else if (node.opType === "Split") {
           this._cpuSplit(node); // Delegate multi-output loop to CPU
        } else if (node.opType === "Gather") {
           const in_s = [1,1,1,1].slice(0, 4-node.inputs.input.shape.length).concat(node.inputs.input.shape);
           let axis = node.params.axis || 0; if (axis < 0) axis += node.inputs.input.shape.length; axis += (4 - node.inputs.input.shape.length);
           const idxPtr = this.pointers.get(node.inputs.indices.name);
           this.api.gather_4d_f32(inPtr, idxPtr, outPtr, in_s[0], in_s[1], in_s[2], in_s[3], node.inputs.indices.sizeBytes/4, axis);
        } else if (node.opType === "GatherElements") {
           this._cpuGatherElements(node); // Complex enough to fallback to CPU JS loop
        } else if (node.opType === "NonMaxSuppression") {
           this._cpuNonMaxSuppression(node);
         } else if (node.opType === "Where" || node.opType === "Mask") {
           const condPtr = this.pointers.get(node.inputs.cond ? node.inputs.cond.name : node.inputs.condition.name);
           const aPtr = this.pointers.get(node.inputs.x ? node.inputs.x.name : node.inputs.a.name);
           const bPtr = this.pointers.get(node.inputs.y ? node.inputs.y.name : node.inputs.b.name);
           this.api.where_f32(condPtr, aPtr, bPtr, outPtr, node.outputs.out.sizeBytes / 4);
         } else if (node.opType === "Pad") {
           const pads = node.params.pads;
           const pt = pads.length === 8 ? pads[1] : pads[0];
           const pl = pads.length === 8 ? pads[2] : pads[1];
           const pb = pads.length === 8 ? pads[5] : pads[2];
           const pr = pads.length === 8 ? pads[6] : pads[3];
           const val = node.params.value || 0.0;
           const inShape = node.inputs.input ? node.inputs.input.shape : node.inputs.data.shape;
           const s = inShape.length === 4 ? inShape : [1, inShape[0] || 1, inShape[1] || 1, 1];
           this.api.pad_2d_f32(inPtr, outPtr, val, s[0], s[1], s[2], s[3], pt, pb, pl, pr);
         } else if (node.opType === "AveragePool2D" || node.opType === "AveragePool") {
           const [b, in_h, in_w, c] = (node.inputs.input || node.inputs.x).shape;
           const [ob, out_h, out_w] = node.outputs.out.shape;
           const ky = node.params.kernel[0], kx = node.params.kernel[1];
           const sy = node.params.stride ? node.params.stride[0] : 1;
           const sx = node.params.stride ? node.params.stride[1] : 1;
           const py = node.params.padding ? node.params.padding[0] : 0;
           const px = node.params.padding ? node.params.padding[1] : 0;
           this.api.averagepool2d_f32(inPtr, outPtr, b, in_h, in_w, c, ky, kx, sy, sx, py, px, out_h, out_w);
         } else if (node.opType === "Div") {
           this._cpuDiv(node);
        } else if (node.opType === "MaxPool2D") {
           const inShape = node.inputs.input.shape;
           const outShape = node.outputs.out.shape;
           const ky = node.params.kernel[0], kx = node.params.kernel[1];
           const sy = node.params.stride[0], sx = node.params.stride[1];
           const py = node.params.padding ? node.params.padding[0] : 0;
           const px = node.params.padding ? node.params.padding[1] : 0;
           this.api.maxpool2d_f32(inPtr, outPtr, inShape[1], inShape[2], inShape[3], outShape[1], outShape[2], ky, kx, sy, sx, py, px);
        } else if (node.opType === "MeanHeight") {
           const inShape = node.inputs.input.shape;
           this.api.mean_height_f32(inPtr, outPtr, inShape[3], inShape[1], inShape[2]);
        } else if (node.opType === "Flatten" || node.opType === "Squeeze" || node.opType === "Unsqueeze" || node.opType === "Dropout" || node.opType === "Reshape" || node.opType === "Identity") {
            this._cpuReshape(node);
        } else {
             super._runNode(node);
        }
      } catch (e) {
          console.error("[WasmEngine] Execution failed at node:", node, e);
          throw e;
      }
    }

    const results = {};
    for (const name of this.graph.outputNames) {
        const tensor = this.graph.tensors.get(name);
        const ptr = this.pointers.get(name);
        results[name] = new Float32Array(this.mem.buffer, ptr, tensor.sizeBytes / 4).slice();
    }
    return results;
  }
};
