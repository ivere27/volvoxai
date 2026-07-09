import { _pair } from './Tensor.js';
import { Graph } from './Graph.js';

// ShaderLibrary statically imports every .wgsl file as text, which only the
// bundler (esbuild's .wgsl=text loader) can resolve. It is loaded lazily inside
// compile() so that merely importing the engine under plain Node (the WASM/CPU
// tiers, the CLI smoke test) never touches a .wgsl file. The WebGPU tier is the
// only consumer, and it always goes through compile() first.
let ShaderLibrary;


export class GraphExecutor {
  constructor(device, graph) {
    this.device = device;
    this.graph = graph;
    this.pipelines = [];
    this.gpuBuffers = /* @__PURE__ */ new Map();
    console.log("[VolvoxAI WebGPU] Starting Graph Compilation...");
  }
  /**
   * Allocate VRAM for all tensors and compile shaders.
   */
  async compile() {
    ({ ShaderLibrary } = await import('./ShaderLibrary.js'));
    // The linearF32/linearInt8 shaders read the weight as [d_out, d_in]. GPT-Neo-style
    // weights are stored [d_in, d_out] (node.wLayout==='din'), so transpose those once
    // before they are uploaded to VRAM (keyed by weight name).
    this._dinWeights = new Map();
    for (const n of this.graph.nodes) {
      if ((n.opType === "MatMul" || n.opType === "Linear" || n.opType === "Gemm") && n.wLayout === "din" && n.inputs.weight && !n.inputs.scale) {
        const din = n.inputs.input.shape[n.inputs.input.shape.length - 1];
        const dout = n.outputs.out.shape[n.outputs.out.shape.length - 1];
        this._dinWeights.set(n.inputs.weight.name, { din, dout });
      }
    }
    this._allocateBuffers();
    for (const node of this.graph.nodes) {
      await this._buildNodePipeline(node);
    }
    console.log(`[VolvoxAI WebGPU] Compilation complete. Allocated ${this.gpuBuffers.size} VRAM buffers.`);
  }
  _allocateBuffers() {
    for (const [name, tensor] of this.graph.tensors.entries()) {
      let usage = GPUBufferUsage.STORAGE;
      if (this.graph.nodes.some((n) => Object.values(n.inputs).some((t) => t.name === name))) {
        usage |= GPUBufferUsage.COPY_DST;
      }
      if (this.graph.nodes.some((n) => Object.values(n.outputs).some((t) => t.name === name))) {
        usage |= GPUBufferUsage.COPY_SRC;
      }
      if (!tensor.isWeight && !this.graph.nodes.some((n) => Object.values(n.outputs).some((t) => t.name === name))) {
        usage |= GPUBufferUsage.COPY_DST;
      }
      // Any weight we upload below needs COPY_DST, even if it is a dangling
      // constant not consumed as a node input (exporters emit these).
      if (tensor.isWeight && tensor.buffer) {
        usage |= GPUBufferUsage.COPY_DST;
      }
      const buffer = this.device.createBuffer({
        label: `Tensor_${name}`,
        size: Math.ceil(tensor.sizeBytes / 4) * 4,
        // Align to 4 bytes
        usage
      });
      tensor.gpuBuffer = buffer;
      this.gpuBuffers.set(name, buffer);
      // Upload learned-parameter data now. Weight buffers are consumed as node
      // inputs (so they carry COPY_DST); without this, Conv/MatMul/BatchNorm read
      // zeros on the GPU. Input tensors are written later at execute() time.
      if (tensor.isWeight && tensor.buffer) {
        let wbuf = tensor.buffer;
        const dw = this._dinWeights && this._dinWeights.get(name);
        if (dw) {   // transpose [d_in, d_out] -> [d_out, d_in] for the shader
          const { din, dout } = dw;
          const t = new Float32Array(din * dout);
          for (let k = 0; k < din; k++) for (let j = 0; j < dout; j++) t[j * din + k] = tensor.buffer[k * dout + j];
          wbuf = t;
        }
        const src = new Uint8Array(wbuf.buffer, wbuf.byteOffset, wbuf.byteLength);
        const padded = Math.ceil(src.byteLength / 4) * 4;
        if (padded === src.byteLength) {
          this.device.queue.writeBuffer(buffer, 0, src);
        } else {
          const tmp = new Uint8Array(padded);
          tmp.set(src);
          this.device.queue.writeBuffer(buffer, 0, tmp);
        }
      }
    }
  }
  async _buildNodePipeline(node) {
    let wgslCode = "";
    let fallbackWgslCode = "";
    let fallbackWorkgroupCount = null;
    let bindGroupEntries = [];
    let workgroupCount = [1, 1, 1];
    if (node.opType === "Conv2D") {
      wgslCode = ShaderLibrary.getConv2DShader();
      const inputBuf = node.inputs.input.gpuBuffer;
      const weightBuf = node.inputs.weight.gpuBuffer;
      const outputBuf = node.outputs.out.gpuBuffer;
      const [n, h, w, c] = node.inputs.input.shape;
      const [kh, kw] = node.inputs.weight.shape;
      const outC = node.outputs.out.shape[3];
      const outH = node.outputs.out.shape[1];
      const outW = node.outputs.out.shape[2];
      const [sy, sx] = _pair(node.params.stride, 1);
      const [pt, pl] = _pair(node.params.padding, 0);
      const [dy, dx] = _pair(node.params.dilation, 1);
      const groups = node.params.groups || 1;
      const pads = Array.isArray(node.params.pads) ? node.params.pads : [pt, pl, pt, pl];
      const noPad = pads.length >= 4 && pads[0] === 0 && pads[1] === 0 && pads[2] === 0 && pads[3] === 0;
      const weightLayout = node.params.weight_layout || (groups === c ? "HWCM" : "HWIO");
      fallbackWgslCode = wgslCode;
      fallbackWorkgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * outC];
      if (groups === 1 && weightLayout === "HWIO" && c === 3 && (outC & 15) === 0) {
        wgslCode = ShaderLibrary.getConv2DRegularC3Out16Shader();
        workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * (outC / 16)];
      } else if (groups === c && weightLayout === "HWCM" && outC === c && (outC & 7) === 0) {
        wgslCode = ShaderLibrary.getConv2DDepthwise8Shader();
        workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * Math.ceil(outC / 8)];
      } else if (groups === 1 && weightLayout === "HWIO" && kh === 1 && kw === 1 &&
                 sy === 1 && sx === 1 && noPad && dy === 1 && dx === 1 &&
                 outH === h && outW === w) {
        if ((outC & 15) === 0) {
          wgslCode = ShaderLibrary.getConv2DPointwise16TileShader();
          workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * (outC / 16)];
        } else if ((outC & 3) === 0) {
          wgslCode = ShaderLibrary.getConv2DPointwise8Vec4Shader();
          workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * Math.ceil(outC / 8)];
        } else if ((outC & 1) === 0) {
          wgslCode = ShaderLibrary.getConv2DPointwise8Vec2Shader();
          workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * Math.ceil(outC / 8)];
        }
      }
      const biasBuf = this.device.createBuffer({
        size: Math.ceil(outC * 4 / 4) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      if (node.inputs.bias) {
          this.device.queue.writeBuffer(biasBuf, 0, node.inputs.bias.buffer);
      }
      const p = new Uint32Array([
        n,
        h,
        w,
        c,
        outC,
        outH,
        outW,
        kh,
        kw,
        sy,
        sx,
        pt,
        pl,
        groups,
        node.params.relu || 0,
        dy,
        dx
      ]);
      const paramBuf = this.device.createBuffer({ size: Math.ceil(p.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: weightBuf } },
        { binding: 2, resource: { buffer: biasBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
        { binding: 4, resource: { buffer: paramBuf } }
      ];
      if (wgslCode === fallbackWgslCode) workgroupCount = fallbackWorkgroupCount;
    } else if (node.opType === "Conv1D") {
      wgslCode = ShaderLibrary.getConv1DShader();
      const inputBuf = node.inputs.input.gpuBuffer;
      const weightBuf = node.inputs.weight.gpuBuffer;
      const outputBuf = node.outputs.out.gpuBuffer;
      const biasBuf = this.device.createBuffer({
        size: Math.ceil((node.inputs.weight.shape[0] * 4) / 4) * 4 || 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      if (node.inputs.bias) {
          this.device.queue.writeBuffer(biasBuf, 0, node.inputs.bias.buffer);
      }
      const p = new Uint32Array([
        node.inputs.input.shape[1],
        node.inputs.input.shape[2],
        node.outputs.out.shape[1],
        node.inputs.weight.shape[2],
        _pair(node.params.stride, 1)[0],
        _pair(node.params.padding, 0)[0],
        node.params.relu ? 1 : 0
      ]);
      const paramBuf = this.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: weightBuf } },
        { binding: 2, resource: { buffer: biasBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
        { binding: 4, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [Math.ceil(node.outputs.out.shape[2] / 64), node.outputs.out.shape[1], 1];
    } else if (node.opType === "SpatialSoftargmaxY") {
      wgslCode = ShaderLibrary.getSpatialSoftargmaxYShader();
      const inputBuf = node.inputs.input.gpuBuffer;
      const outputBuf = node.outputs.out.gpuBuffer;
      const p = new Uint32Array([
        node.inputs.input.shape[1],
        node.inputs.input.shape[2],
        node.inputs.input.shape[3]
      ]);
      const paramBuf = this.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [Math.ceil(node.inputs.input.shape[2] / 64), node.inputs.input.shape[3], 1];
    } else if (node.opType === "UpsampleNearest2D") {
      wgslCode = ShaderLibrary.getUpsample2xShader();
      const inputBuf = node.inputs.input.gpuBuffer;
      const outputBuf = node.outputs.out.gpuBuffer;
      const p = new Uint32Array([
        node.inputs.input.shape[0],
        node.inputs.input.shape[1],
        node.inputs.input.shape[2],
        node.inputs.input.shape[3]
      ]);
      const paramBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [
        Math.ceil(node.outputs.out.shape[2] / 8),
        Math.ceil(node.outputs.out.shape[1] / 8),
        node.outputs.out.shape[0] * node.outputs.out.shape[3]
      ];
    } else if (node.opType === "Concat") {
      // N-way channel concat (batch=1): copy each input into the output at its
      // cumulative element offset via one strided-copy pipeline per input.
      const shaderModule = this.device.createShaderModule({ code: ShaderLibrary.getConcatCopyShader() });
      const pipeline = await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" }
      });
      let offset = 0;
      for (const k of ["input", "a", "b", "c", "d", "e", "f", "g", "h"]) {
        const t = node.inputs[k];
        if (!t) continue;
        const size = t.sizeBytes / 4;
        const p = new Uint32Array([size, offset]);
        const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(paramsBuf, 0, p);
        const bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: t.gpuBuffer } },
            { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
            { binding: 2, resource: { buffer: paramsBuf } }
          ]
        });
        this.pipelines.push({ pipeline, bindGroup, workgroupCount: [Math.ceil(size / 64), 1, 1], nodeName: `${node.id}_concat` });
        offset += size;
      }
      return;
    } else if (node.opType === "ProfileY") {
      wgslCode = ShaderLibrary.getProfileYShader();
      const inputBuf = node.inputs.input.gpuBuffer;
      const outputBuf = node.outputs.out.gpuBuffer;
      const p = new Uint32Array([node.inputs.input.shape[1], node.inputs.input.shape[2], node.inputs.input.shape[3]]);
      const paramBuf = this.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [Math.ceil(node.inputs.input.shape[1] / 64), node.inputs.input.shape[3], 1];
    } else if (node.opType === "ProfileX") {
      wgslCode = ShaderLibrary.getProfileXShader();
      const inputBuf = node.inputs.input.gpuBuffer;
      const outputBuf = node.outputs.out.gpuBuffer;
      const p = new Uint32Array([node.inputs.input.shape[1], node.inputs.input.shape[2], node.inputs.input.shape[3]]);
      const paramBuf = this.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [Math.ceil(node.inputs.input.shape[2] / 64), node.inputs.input.shape[3], 1];
    } else if (node.opType === "InterpLinear1D") {
      wgslCode = ShaderLibrary.getInterp1DShader();
      const inputBuf = node.inputs.input.gpuBuffer;
      const outputBuf = node.outputs.out.gpuBuffer;
      const p = new Uint32Array([node.inputs.input.shape[1], node.inputs.input.shape[2], node.params.size]);
      const paramBuf = this.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: outputBuf } },
        { binding: 2, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [Math.ceil(node.params.size / 64), node.inputs.input.shape[1], 1];
    } else if (node.opType === "MatMul") {
      if (node.inputs.scale) {
          wgslCode = ShaderLibrary.getLinearInt8Shader();
      } else {
          wgslCode = ShaderLibrary.getLinearF32Shader();
      }
      const dummyBias = this.device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
      const seq_len = node.inputs.input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
      const d_in = node.inputs.input.shape[node.inputs.input.shape.length - 1];
      const d_out = node.outputs.out.shape[node.outputs.out.shape.length - 1];
      const p = new Uint32Array([seq_len, d_in, d_out]);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      const biasBuf = node.inputs.bias ? node.inputs.bias.gpuBuffer : dummyBias;
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.weight.gpuBuffer } },
        // binding 2 (scale) exists only in the int8 shader; the f32 shader's
        // unused binding 2 is dropped by layout:"auto", so we must omit it.
        ...(node.inputs.scale ? [{ binding: 2, resource: { buffer: node.inputs.scale.gpuBuffer } }] : []),
        { binding: 3, resource: { buffer: biasBuf } },
        { binding: 4, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 5, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(d_out / 64), seq_len, 1];
    } else if (node.opType === "LayerNorm") {
      wgslCode = ShaderLibrary.getLayerNormShader();
      const d_model = node.params.d_model;
      const seq_len = node.inputs.input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
      const p = new Uint32Array([seq_len, d_model]);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.weight.gpuBuffer } },
        { binding: 2, resource: { buffer: node.inputs.bias.gpuBuffer } },
        { binding: 3, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 4, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(seq_len / 64), 1, 1];
    } else if (node.opType === "GELU") {
      wgslCode = ShaderLibrary.getGELUShader();
      const num_elements = node.outputs.out.shape.reduce((a, b) => a * b, 1);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([num_elements]));
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(num_elements / 64), 1, 1];
        } else if (node.opType === "Embedding") {
      wgslCode = ShaderLibrary.getEmbeddingShader();
      const seq_len = node.inputs.input.shape.reduce((a, b) => a * b, 1);
      const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
      const p = new Uint32Array([seq_len, d_model]);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.weight.gpuBuffer } },
        { binding: 2, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 3, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(seq_len / 64), 1, 1];
    } else if (node.opType === "SDPA") {
      wgslCode = ShaderLibrary.getSDPAShader();
      const seq_len = node.inputs.qkv.shape[1];
      const d_model = node.outputs.out.shape[2];
      const num_heads = node.params.heads || 8;
      const head_dim = d_model / num_heads;
      const scale = node.params.scale !== undefined ? node.params.scale : 1 / Math.sqrt(head_dim);
      const p = new ArrayBuffer(20);
      const p_u32 = new Uint32Array(p);
      const p_f32 = new Float32Array(p);
      p_u32[0] = seq_len;
      p_u32[1] = d_model;
      p_u32[2] = num_heads;
      p_u32[3] = head_dim;
      p_f32[4] = scale;
      const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.qkv.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(seq_len / 64), num_heads, 1];
    } else if (node.opType === "CrossSDPA") {
      wgslCode = ShaderLibrary.getCrossSDPAShader();
      const seq_len_q = node.inputs.q.shape[1];
      const seq_len_kv = node.inputs.k.shape[1];
      const d_model = node.inputs.q.shape[2];
      const num_heads = node.params.heads || 8;
      const head_dim = d_model / num_heads;
      const scale = 1 / Math.sqrt(head_dim);
      const p = new ArrayBuffer(24);
      const p_u32 = new Uint32Array(p);
      const p_f32 = new Float32Array(p);
      p_u32[0] = seq_len_q;
      p_u32[1] = seq_len_kv;
      p_u32[2] = d_model;
      p_u32[3] = num_heads;
      p_u32[4] = head_dim;
      p_f32[5] = scale;
      const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.q.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.k.gpuBuffer } },
        { binding: 2, resource: { buffer: node.inputs.v.gpuBuffer } },
        { binding: 3, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 4, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(seq_len_q / 64), num_heads, 1];
    } else if (node.opType === "CrossAttention") {
      wgslCode = ShaderLibrary.getCrossAttentionShader();
      const seq_len_q = node.inputs.q.shape[1];
      const seq_len_kv = node.inputs.kv.shape[1];
      const d_model = node.outputs.out.shape[2];
      const num_heads = node.params.heads || 8;
      const head_dim = d_model / num_heads;
      const scale_factor = 1 / Math.sqrt(head_dim);
      const p = new ArrayBuffer(32);
      const p_u32 = new Uint32Array(p);
      const p_f32 = new Float32Array(p);
      p_u32[0] = seq_len_q;
      p_u32[1] = seq_len_kv;
      p_u32[2] = d_model;
      p_u32[3] = num_heads;
      p_u32[4] = head_dim;
      p_f32[5] = scale_factor;
      p_u32[6] = node.inputs.scale ? 1 : 0;
      p_u32[7] = node.inputs.bias ? 1 : 0;
      const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      
      const dummyScale = this.device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
      const dummyBias = this.device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
      
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.q.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.kv.gpuBuffer } },
        { binding: 2, resource: { buffer: node.inputs.weight.gpuBuffer } },
        { binding: 3, resource: { buffer: node.inputs.scale ? node.inputs.scale.gpuBuffer : dummyScale } },
        { binding: 4, resource: { buffer: node.inputs.bias ? node.inputs.bias.gpuBuffer : dummyBias } },
        { binding: 5, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 6, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(seq_len_q / 64), num_heads, 1];
    } else if (node.opType === "MeanHeight") {
      wgslCode = ShaderLibrary.getMeanHeightShader();
      const p = new Uint32Array([node.inputs.input.shape[1], node.inputs.input.shape[2], node.inputs.input.shape[3]]);
      const paramBuf = this.device.createBuffer({ size: Math.ceil(p.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [Math.ceil(node.inputs.input.shape[2] / 64), node.inputs.input.shape[3], 1];
    } else if (node.opType === "ReLU" || node.opType === "Sigmoid" || node.opType === "HardSwish"
               || node.opType === "HardSigmoid" || node.opType === "SiLU" || node.opType === "Swish"
               || node.opType === "Tanh" || node.opType === "Reshape" || node.opType === "Squeeze"
               || node.opType === "Unsqueeze" || node.opType === "Flatten" || node.opType === "Dropout"
               || node.opType === "Identity") {
      // Elementwise unary ops, plus shape-only ops (Reshape/Squeeze/...) which
      // just copy the data through to the newly-allocated output buffer.
      if (node.opType === "ReLU") wgslCode = ShaderLibrary.getReLUShader();
      else if (node.opType === "Sigmoid") wgslCode = ShaderLibrary.getSigmoidShader();
      else if (node.opType === "HardSwish") wgslCode = ShaderLibrary.getHardSwishShader();
      else if (node.opType === "HardSigmoid") wgslCode = ShaderLibrary.getHardSigmoidShader();
      else if (node.opType === "SiLU" || node.opType === "Swish") wgslCode = ShaderLibrary.getSiLUShader();
      else if (node.opType === "Tanh") wgslCode = ShaderLibrary.getTanhShader();
      else wgslCode = ShaderLibrary.getCopyShader();
      const elements = node.outputs.out.shape.reduce((a, b) => a * b, 1);
      const p = new Uint32Array([elements]);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(elements / 64), 1, 1];
    } else if (node.opType === "Add" || node.opType === "Mul" || node.opType === "Sub" || node.opType === "Div") {
      const binOp = { Add: "out_val = av + bv;", Mul: "out_val = av * bv;",
                      Sub: "out_val = av - bv;", Div: "out_val = av / bv;" }[node.opType];
      wgslCode = ShaderLibrary.getBroadcastBinaryShader(binOp);
      const outShape = node.outputs.out.shape;
      const rank = outShape.length;
      const contigStrides = (shape) => {
        const st = new Array(shape.length);
        let s = 1;
        for (let i = shape.length - 1; i >= 0; i--) { st[i] = s; s *= shape[i]; }
        return st;
      };
      const outStrides = contigStrides(outShape);
      // numpy-style broadcast strides: left-pad the operand shape to the output
      // rank, then zero the stride of any dimension that is broadcast (size 1).
      const bcastStrides = (shape) => {
        const padded = new Array(rank).fill(1);
        for (let i = 0; i < shape.length; i++) padded[rank - shape.length + i] = shape[i];
        const st = contigStrides(padded);
        for (let i = 0; i < rank; i++) if (padded[i] === 1 && outShape[i] !== 1) st[i] = 0;
        return st;
      };
      const aStrides = bcastStrides(node.inputs.a.shape);
      const bStrides = bcastStrides(node.inputs.b.shape);
      const total = outShape.reduce((a, b) => a * b, 1);
      const meta = new Uint32Array(2 + rank * 3);
      meta[0] = total; meta[1] = rank;
      for (let d = 0; d < rank; d++) {
        meta[2 + d] = outStrides[d];
        meta[2 + rank + d] = aStrides[d];
        meta[2 + 2 * rank + d] = bStrides[d];
      }
      const metaBuf = this.device.createBuffer({ size: Math.ceil(meta.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(metaBuf, 0, meta);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.a.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.b.gpuBuffer } },
        { binding: 2, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 3, resource: { buffer: metaBuf } }
      ];
      workgroupCount = [Math.ceil(total / 64), 1, 1];
    } else if (node.opType === "Transpose") {
      wgslCode = ShaderLibrary.getGeneralTransposeShader();
      const inShape = node.inputs.input.shape;
      const rank = inShape.length;
      const perm = node.params.perm || [...Array(rank).keys()].reverse();
      const inStrides = new Array(rank);
      { let s = 1; for (let i = rank - 1; i >= 0; i--) { inStrides[i] = s; s *= inShape[i]; } }
      const outShape = perm.map((pp) => inShape[pp]);
      const outStrides = new Array(rank);
      { let s = 1; for (let i = rank - 1; i >= 0; i--) { outStrides[i] = s; s *= outShape[i]; } }
      const total = inShape.reduce((a, b) => a * b, 1);
      const meta = new Uint32Array(2 + rank * 2);
      meta[0] = total; meta[1] = rank;
      for (let d = 0; d < rank; d++) { meta[2 + d] = outStrides[d]; meta[2 + rank + d] = inStrides[perm[d]]; }
      const metaBuf = this.device.createBuffer({ size: Math.ceil(meta.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(metaBuf, 0, meta);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: metaBuf } }
      ];
      workgroupCount = [Math.ceil(total / 64), 1, 1];
    } else if (node.opType === "Softmax" || node.opType === "LogSoftmax") {
      // Normalizes over the last (innermost) axis: rows of `d`, `b` rows total.
      wgslCode = node.opType === "Softmax" ? ShaderLibrary.getSoftmaxShader() : ShaderLibrary.getLogSoftmaxShader();
      const shape = node.inputs.input.shape;
      const d = shape[shape.length - 1];
      const b = shape.reduce((a, x) => a * x, 1) / d;
      if (node.params.axis !== undefined) {
        let ax = node.params.axis; if (ax < 0) ax += shape.length;
        if (ax !== shape.length - 1) console.warn(`[VolvoxAI WebGPU] ${node.opType} axis ${node.params.axis} != last; using last-axis.`);
      }
      const p = new Uint32Array([b, d]);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(b / 64), 1, 1];
    } else if (node.opType === "LeakyReLU") {
      wgslCode = ShaderLibrary.getLeakyReLUShader();
      const elements = node.outputs.out.shape.reduce((a, b) => a * b, 1);
      const alpha = node.params.alpha !== undefined ? node.params.alpha : 0.01;
      const p = new ArrayBuffer(16);
      new Uint32Array(p, 0, 1)[0] = elements;
      new Float32Array(p, 4, 1)[0] = alpha;
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(elements / 64), 1, 1];
    } else if (node.opType === "PReLU") {
      wgslCode = ShaderLibrary.getPReLUShader();
      const shape = node.inputs.input.shape;
      const c = shape[shape.length - 1] || 1;
      const elements = shape.reduce((a, b) => a * b, 1);
      const p = new Uint32Array([elements, c]);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.weight.gpuBuffer } },
        { binding: 2, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 3, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(elements / 64), 1, 1];
    } else if (node.opType === "RMSNorm") {
      wgslCode = ShaderLibrary.getRMSNormShader();
      const shape = node.inputs.input.shape;
      const d_model = node.params.d_model || shape[shape.length - 1];
      const seq_len = shape.reduce((a, x) => a * x, 1) / d_model;
      const eps = node.params.eps !== undefined ? node.params.eps : 1e-6;
      const p = new ArrayBuffer(16);
      new Uint32Array(p, 0, 2).set([seq_len, d_model]);
      new Float32Array(p, 8, 1)[0] = eps;
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.weight.gpuBuffer } },
        { binding: 2, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 3, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(seq_len / 64), 1, 1];
    } else if (node.opType === "GlobalAveragePool") {
      wgslCode = ShaderLibrary.getGlobalAveragePoolShader();
      const inShape = node.inputs.input.shape;
      const B = inShape[0];
      const H = inShape[1] || 1;
      const W = inShape[2] || 1;
      const C = inShape[3] || 1;
      const p = new Uint32Array([B, H, W, C]);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(C / 64), B, 1];
    } else if (node.opType === "BatchNorm2D") {
      wgslCode = ShaderLibrary.getBatchNorm2DShader();
      const [bn, hn, wn, cn] = node.inputs.input.shape;
      const eps = node.params.eps !== undefined ? node.params.eps : 1e-5;
      const p = new ArrayBuffer(32);
      new Uint32Array(p, 0, 4).set([bn, cn, hn, wn]);
      new Float32Array(p, 16, 1)[0] = eps;
      const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.weight.gpuBuffer } },
        { binding: 2, resource: { buffer: node.inputs.bias.gpuBuffer } },
        { binding: 3, resource: { buffer: node.inputs.running_mean.gpuBuffer } },
        { binding: 4, resource: { buffer: node.inputs.running_var.gpuBuffer } },
        { binding: 5, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 6, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil((bn * cn * hn * wn) / 64), 1, 1];
    } else if (node.opType === "Resize" || node.opType === "ResizeNearest2D") {
      wgslCode = ShaderLibrary.getResizeShader();
      const [rb, inH, inW, rc] = node.inputs.input.shape;
      const [, outH, outW] = node.outputs.out.shape;
      const mode = (node.opType === "ResizeNearest2D" || node.params.mode === "nearest") ? 0 : 1;
      const p = new Uint32Array([rb, inH, inW, rc, outH, outW, mode, 0]);
      const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), rb * rc];
    } else if (node.opType === "Split") {
      // Split has multiple outputs; push one strided-copy pipeline per slice.
      const inShape = node.inputs.input.shape;
      let axis = node.params.axis || 0;
      if (axis < 0) axis += inShape.length;
      const outKeys = Object.keys(node.outputs).sort();
      const numOutputs = outKeys.length;
      const splitSize = inShape[axis] / numOutputs;
      let inner = 1;
      for (let i = axis + 1; i < inShape.length; i++) inner *= inShape[i];
      const shaderModule = this.device.createShaderModule({ code: ShaderLibrary.getSplitShader() });
      const pipeline = await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" }
      });
      for (let o = 0; o < numOutputs; o++) {
        const outT = node.outputs[outKeys[o]];
        const total = outT.shape.reduce((a, b) => a * b, 1);
        const p = new Uint32Array([total, inner, splitSize, inShape[axis], o * splitSize]);
        const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.device.queue.writeBuffer(paramsBuf, 0, p);
        const bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
            { binding: 1, resource: { buffer: outT.gpuBuffer } },
            { binding: 2, resource: { buffer: paramsBuf } }
          ]
        });
        this.pipelines.push({
          pipeline, bindGroup,
          workgroupCount: [Math.ceil(total / 64), 1, 1],
          nodeName: `${node.id}_split${o}`
        });
      }
      return;
    } else if (node.opType === "Clip") {
      wgslCode = ShaderLibrary.getClipShader();
      const elements = node.inputs.input.sizeBytes / 4;
      let minVal = node.params.min !== undefined ? node.params.min : -1e9;
      let maxVal = node.params.max !== undefined ? node.params.max : 1e9;
      if (node.inputs.min) minVal = new Float32Array(node.inputs.min.buffer)[0];
      if (node.inputs.max) maxVal = new Float32Array(node.inputs.max.buffer)[0];
      const p = new Float32Array([0, minVal, maxVal, 0]);
      new Uint32Array(p.buffer)[0] = elements;
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(elements / 64), 1, 1];
    } else if (node.opType === "MaxPool2D") {
      wgslCode = ShaderLibrary.getMaxPool2DShader();
      const [ky, kx] = _pair(node.params.kernel, 1);
      const [sy, sx] = _pair(node.params.stride, 1);
      const py = node.params.padding ? node.params.padding[0] : 0;
      const px = node.params.padding ? node.params.padding[1] : 0;
      const p = new Uint32Array([
        node.inputs.input.shape[1],
        node.inputs.input.shape[2],
        node.inputs.input.shape[3],
        node.outputs.out.shape[1],
        node.outputs.out.shape[2],
        ky, kx, sy, sx, py, px
      ]);
      const paramBuf = this.device.createBuffer({ size: Math.ceil(p.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: node.inputs.input.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramBuf } }
      ];
      workgroupCount = [
        Math.ceil(node.outputs.out.shape[2] / 8),
        Math.ceil(node.outputs.out.shape[1] / 8),
        node.outputs.out.shape[3]
      ];
    } else if (node.opType === "Cast") {
      // FP32 engine: a cast is a straight copy. Integer-target truncation is only
      // done on the CPU/WASM tier (see ops/cast.js).
      wgslCode = ShaderLibrary.getCopyShader();
      const inp = node.inputs.input || node.inputs.data;
      const elements = node.outputs.out.shape.reduce((a, b) => a * b, 1);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([elements]));
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(elements / 64), 1, 1];
    } else if (node.opType === "Where" || node.opType === "Mask") {
      wgslCode = ShaderLibrary.getWhereShader();
      const cond = node.inputs.cond || node.inputs.condition;
      const a = node.inputs.x || node.inputs.a;
      const b = node.inputs.y || node.inputs.b;
      const elements = node.outputs.out.shape.reduce((x, y) => x * y, 1);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([elements]));
      bindGroupEntries = [
        { binding: 0, resource: { buffer: cond.gpuBuffer } },
        { binding: 1, resource: { buffer: a.gpuBuffer } },
        { binding: 2, resource: { buffer: b.gpuBuffer } },
        { binding: 3, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 4, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(elements / 64), 1, 1];
    } else if (node.opType === "DequantizeLinear") {
      wgslCode = ShaderLibrary.getDequantizeLinearShader();
      const inp = node.inputs.input || node.inputs.x;
      const elements = node.outputs.out.shape.reduce((a, b) => a * b, 1);
      const hasZp = node.inputs.zero_point ? 1 : 0;
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([elements, hasZp]));
      const dummy = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.scale.gpuBuffer } },
        { binding: 2, resource: { buffer: hasZp ? node.inputs.zero_point.gpuBuffer : dummy } },
        { binding: 3, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 4, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(elements / 64), 1, 1];
    } else if (node.opType === "Expand" || node.opType === "Broadcast") {
      wgslCode = ShaderLibrary.getExpandShader();
      const inp = node.inputs.input || node.inputs.data;
      const pad4 = (sh) => [1, 1, 1, 1].slice(0, 4 - sh.length).concat(sh);
      const [ib, ih, iw, ic] = pad4(inp.shape);
      const [ob, oh, ow, oc] = pad4(node.outputs.out.shape);
      const p = new Uint32Array([ib, ih, iw, ic, ob, oh, ow, oc]);
      const paramsBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil((ob * oh * ow * oc) / 64), 1, 1];
    } else if (node.opType === "Pad") {
      wgslCode = ShaderLibrary.getPadShader();
      const inp = node.inputs.input || node.inputs.data;
      const pads = node.params.pads || [];
      const pt = pads.length === 8 ? pads[1] : (pads[0] || 0);
      const pl = pads.length === 8 ? pads[2] : (pads[1] || 0);
      const val = node.params.value || 0.0;
      const is = [1, 1, 1, 1].slice(0, 4 - inp.shape.length).concat(inp.shape);
      const os = [1, 1, 1, 1].slice(0, 4 - node.outputs.out.shape.length).concat(node.outputs.out.shape);
      const buf = new ArrayBuffer(48);
      const u = new Uint32Array(buf); const f = new Float32Array(buf);
      u[0] = is[0]; u[1] = is[1]; u[2] = is[2]; u[3] = is[3]; u[4] = os[1]; u[5] = os[2]; u[6] = pt; u[7] = pl; f[8] = val;
      const paramsBuf = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, buf);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil((os[0] * os[1] * os[2] * os[3]) / 64), 1, 1];
    } else if (node.opType === "Slice") {
      wgslCode = ShaderLibrary.getSliceShader();
      const inp = node.inputs.input || node.inputs.data;
      const starts = node.params.starts || [0, 0, 0, 0];
      const steps = node.params.steps || [1, 1, 1, 1];
      const axes = node.params.axes || [0, 1, 2, 3];
      const in_s = [1, 1, 1, 1].slice(0, 4 - inp.shape.length).concat(inp.shape);
      const out_s = [1, 1, 1, 1].slice(0, 4 - node.outputs.out.shape.length).concat(node.outputs.out.shape);
      const st = [0, 0, 0, 0], sp = [1, 1, 1, 1];
      for (let i = 0; i < axes.length; i++) {
        let ax = axes[i]; if (ax < 0) ax += inp.shape.length; ax += (4 - inp.shape.length);
        st[ax] = starts[i] < 0 ? starts[i] + in_s[ax] : starts[i];
        sp[ax] = steps[i];
      }
      const total = out_s[0] * out_s[1] * out_s[2] * out_s[3];
      const p = new Uint32Array([out_s[0], out_s[1], out_s[2], out_s[3], in_s[1], in_s[2], in_s[3],
        st[0], st[1], st[2], st[3], sp[0], sp[1], sp[2], sp[3], total]);
      const paramsBuf = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(total / 64), 1, 1];
    } else if (node.opType === "Gather") {
      let axis = node.params.axis || 0;
      if (axis < 0) axis += node.inputs.input.shape.length;
      if (axis !== 0) {
        console.warn(`[VolvoxAI WebGPU] Gather axis ${axis} not supported on GPU; node ${node.id} skipped (use WASM/CPU).`);
        return;
      }
      wgslCode = ShaderLibrary.getGatherShader();
      const inp = node.inputs.input;
      const rowSize = inp.shape.slice(1).reduce((a, b) => a * b, 1) || 1;
      const total = node.outputs.out.shape.reduce((a, b) => a * b, 1);
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([rowSize, total / rowSize, total]));
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.indices.gpuBuffer } },
        { binding: 2, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 3, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(total / 64), 1, 1];
    } else if (node.opType === "ReduceSum" || node.opType === "ReduceMean") {
      wgslCode = ShaderLibrary.getReduceShader();
      const inp = node.inputs.input || node.inputs.data;
      const in_shape = inp.shape.length === 2 ? inp.shape : [1, inp.shape.reduce((a, b) => a * b, 1)];
      const b = in_shape[0], d = in_shape[1];
      const inv = node.opType === "ReduceMean" ? 1.0 / d : 1.0;
      const buf = new ArrayBuffer(16);
      new Uint32Array(buf, 0, 2).set([b, d]); new Float32Array(buf, 8, 1)[0] = inv;
      const paramsBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, buf);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(b / 64), 1, 1];
    } else if (node.opType === "AveragePool" || node.opType === "AveragePool2D") {
      wgslCode = ShaderLibrary.getAveragePool2DShader();
      const inp = node.inputs.input || node.inputs.x;
      const [b, in_h, in_w, c] = inp.shape;
      const [, out_h, out_w] = node.outputs.out.shape;
      const [kh, kw] = _pair(node.params.kernel, 1);
      const [sh, sw] = _pair(node.params.stride, 1);
      const ph = node.params.padding ? node.params.padding[0] : 0;
      const pw = node.params.padding ? node.params.padding[1] : 0;
      const p = new Uint32Array([b, in_h, in_w, c, out_h, out_w, kh, kw, sh, sw, ph, pw]);
      const paramsBuf = this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 2, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(out_w / 8), Math.ceil(out_h / 8), b * c];
    } else if (node.opType === "ConvTranspose2D") {
      wgslCode = ShaderLibrary.getConvTranspose2DShader();
      const inp = node.inputs.input || node.inputs.x;
      const [b, in_h, in_w, in_c] = inp.shape;
      const [, out_h, out_w, out_c] = node.outputs.out.shape;
      const kh = node.params.kernel[0], kw = node.params.kernel[1];
      const sh = node.params.stride ? node.params.stride[0] : 1;
      const sw = node.params.stride ? node.params.stride[1] : 1;
      const ph = node.params.padding ? node.params.padding[0] : 0;
      const pw = node.params.padding ? node.params.padding[1] : 0;
      const hasBias = node.inputs.bias ? 1 : 0;
      const p = new Uint32Array([b, in_h, in_w, in_c, out_h, out_w, out_c, kh, kw, sh, sw, ph, pw, hasBias]);
      const paramsBuf = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(paramsBuf, 0, p);
      const dummyBias = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
      bindGroupEntries = [
        { binding: 0, resource: { buffer: inp.gpuBuffer } },
        { binding: 1, resource: { buffer: node.inputs.weight.gpuBuffer } },
        { binding: 2, resource: { buffer: hasBias ? node.inputs.bias.gpuBuffer : dummyBias } },
        { binding: 3, resource: { buffer: node.outputs.out.gpuBuffer } },
        { binding: 4, resource: { buffer: paramsBuf } }
      ];
      workgroupCount = [Math.ceil(out_w / 8), Math.ceil(out_h / 8), b * out_c];
    } else {
      console.warn(`[VolvoxAI WebGPU] Shader for ${node.opType} not implemented yet in Executor.`);
      return;
    }
    if (!wgslCode) {
      // A branch above delegated to a CPU helper (no GPU shader yet). Skip rather
      // than crash on createShaderModule(""); its output buffer stays unwritten.
      console.warn(`[VolvoxAI WebGPU] ${node.opType} has no native GPU shader; node ${node.id} skipped.`);
      return;
    }
    if (isNaN(workgroupCount[0]) || isNaN(workgroupCount[1]) || isNaN(workgroupCount[2]) || workgroupCount[0] <= 0 || workgroupCount[1] <= 0 || workgroupCount[2] <= 0) {
      console.error(`Invalid workgroupCount [${workgroupCount}] for node ${node.id} (${node.opType})`);
      workgroupCount = [1, 1, 1];
    }
    let shaderModule = this.device.createShaderModule({ code: wgslCode });
    let pipeline;
    try {
      pipeline = await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" }
      });
    } catch (err) {
      if (!fallbackWgslCode || fallbackWgslCode === wgslCode || !fallbackWorkgroupCount) throw err;
      console.warn(`[VolvoxAI WebGPU] Specialized shader for ${node.id} failed; falling back to generic Conv2D.`, err);
      wgslCode = fallbackWgslCode;
      workgroupCount = fallbackWorkgroupCount;
      shaderModule = this.device.createShaderModule({ code: wgslCode });
      pipeline = await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" }
      });
    }
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries
    });
    this.pipelines.push({ pipeline, bindGroup, workgroupCount, nodeName: node.id });
  }
  /**
   * Execute the compiled graph on the GPU.
   * @param {Object} inputs - Key-value pair of input tensor names to Float32Array/Int32Array
   */
  async execute(inputs) {
    for (const [name, data] of Object.entries(inputs)) {
      const buffer = this.gpuBuffers.get(name);
      if (buffer) {
        this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
      }
    }
    let commandEncoder = this.device.createCommandEncoder();
    let passEncoder = commandEncoder.beginComputePass();
    for (let i = 0; i < this.pipelines.length; i++) {
      const p = this.pipelines[i];
      passEncoder.setPipeline(p.pipeline);
      passEncoder.setBindGroup(0, p.bindGroup);
      passEncoder.dispatchWorkgroups(p.workgroupCount[0], p.workgroupCount[1], p.workgroupCount[2]);
      if ((i + 1) % 20 === 0) {
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
        commandEncoder = this.device.createCommandEncoder();
        passEncoder = commandEncoder.beginComputePass();
      }
    }
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
    const lastNode = this.graph.nodes[this.graph.nodes.length - 1];
    const outName = Object.keys(lastNode.outputs)[0];
    return this.gpuBuffers.get(lastNode.outputs[outName].name);
  }
  /**
   * Helper to read a GPUBuffer back to CPU (Float32Array) for validation.
   */
  async readBuffer(gpuBuffer, sizeBytes) {
    const stagingBuffer = this.device.createBuffer({
      size: sizeBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(gpuBuffer, 0, stagingBuffer, 0, sizeBytes);
    this.device.queue.submit([commandEncoder.finish()]);
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const copyArray = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    return copyArray;
  }
};
