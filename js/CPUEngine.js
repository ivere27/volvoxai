import { _cpuPReLU } from './ops/pReLU.js';
import { _cpuExpand } from './ops/expand.js';
import { _cpuDequantizeLinear } from './ops/dequantizeLinear.js';
import { _cpuTanh } from './ops/tanh.js';
import { _cpuRMSNorm } from './ops/rMSNorm.js';
import { _cpuSiLU } from './ops/siLU.js';
import { _cpuSub } from './ops/sub.js';
import { _cpuLogSoftmax } from './ops/logSoftmax.js';
import { _cpuSoftmax } from './ops/softmax.js';
import { _cpuWhere } from './ops/where.js';
import { _cpuPad } from './ops/pad.js';
import { _cpuAveragePool2D } from './ops/averagePool2D.js';
import { _cpuSlice } from './ops/slice.js';
import { _cpuConvTranspose2D } from './ops/convTranspose2D.js';
import { _cpuReduceSum } from './ops/reduceSum.js';
import { _cpuReduceMean } from './ops/reduceMean.js';
import { _cpuBatchNorm2D } from './ops/batchNorm2D.js';
import { _cpuDiv } from './ops/div.js';
import { _cpuNonMaxSuppression } from './ops/nonMaxSuppression.js';
import { _cpuGather } from './ops/gather.js';
import { _cpuGatherElements } from './ops/gatherElements.js';
import { _cpuCrossAttention } from './ops/crossAttention.js';
import { _cpuSpatialSoftargmaxY } from './ops/spatialSoftargmaxY.js';
import { _cpuTranspose } from './ops/transpose.js';
import { _cpuCrossSDPA } from './ops/crossSDPA.js';
import { _cpuMeanHeight } from './ops/meanHeight.js';
import { _cpuMaxPool2D } from './ops/maxPool2D.js';
import { _cpuInterp1D } from './ops/interp1D.js';
import { _cpuProfileX } from './ops/profileX.js';
import { _cpuProfileY } from './ops/profileY.js';
import { _cpuConcat2 } from './ops/concat2.js';
import { _cpuUpsample2x } from './ops/upsample2x.js';
import { _cpuSDPA } from './ops/sDPA.js';
import { _cpuEmbedding } from './ops/embedding.js';
import { _cpuMul } from './ops/mul.js';
import { _cpuAdd } from './ops/add.js';
import { _cpuGlobalAveragePool } from './ops/globalAveragePool.js';
import { _cpuClip } from './ops/clip.js';
import { _cpuReshape } from './ops/reshape.js';
import { _cpuSplit } from './ops/split.js';
import { _cpuResize } from './ops/resize.js';
import { _cpuSigmoid } from './ops/sigmoid.js';
import { _cpuHardSigmoid } from './ops/hardSigmoid.js';
import { _cpuHardSwish } from './ops/hardSwish.js';
import { _cpuReLU } from './ops/reLU.js';
import { _cpuLeakyReLU } from './ops/leakyReLU.js';
import { _cpuGELU } from './ops/gELU.js';
import { _cpuLayerNorm } from './ops/layerNorm.js';
import { _cpuConv2D } from './ops/conv2D.js';
import { _cpuConv1D } from './ops/conv1D.js';
import { _cpuMatMul } from './ops/matMul.js';
import { _cpuCast } from './ops/cast.js';
import { _cpuArgMax } from './ops/argMax.js';


export class CPUEngine {
  constructor() {
    this.tensors = /* @__PURE__ */ new Map();
    console.log("[VolvoxAI] CPU Fallback Engine ready.");
  }
  /**
   * Allocates CPU memory (ArrayBuffers) for the graph's tensors.
   */
  allocateGraph(graph) {
    this.graph = graph;
    for (const [name, tensor] of graph.tensors.entries()) {
      if (!tensor.buffer) {
        if (tensor.dtype === "int8") {
          tensor.buffer = new Int8Array(tensor.sizeBytes);
        } else {
          tensor.buffer = new Float32Array(tensor.sizeBytes / 4);
        }
      }
    }
  }
  /**
   * Executes the graph linearly on the CPU.
   */
  async execute(inputsOrGraph, maybeInputs) {
    const graph = maybeInputs ? inputsOrGraph : this.graph;
    const inputs = maybeInputs || inputsOrGraph;
    for (const [name, data] of Object.entries(inputs)) {
      const tensor = graph.tensors.get(name);
      if (tensor && tensor.buffer) {
        tensor.buffer.set(data);
      }
    }
    for (const node of graph.nodes) {
      this._runNode(node);
    }
    const result = {};
    if (graph.outputNames && graph.outputNames.length > 0) {
      for (const name of graph.outputNames) {
        result[name] = graph.tensors.get(name).buffer;
      }
    } else {
      const lastNode = graph.nodes[graph.nodes.length - 1];
      for (const [key, t] of Object.entries(lastNode.outputs)) {
        result[t.name] = graph.tensors.get(t.name).buffer;
      }
    }
    return result;
  }
  _runNode(node) {
    switch (node.opType) {
      // --- Linear / attention ---
      case "MatMul": return this._cpuMatMul(node);
      case "LayerNorm": return this._cpuLayerNorm(node);
      case "RMSNorm": return this._cpuRMSNorm(node);
      case "Embedding": return this._cpuEmbedding(node);
      case "SDPA": return this._cpuSDPA(node);
      case "CrossSDPA": return this._cpuCrossSDPA(node);
      case "CrossAttention": return this._cpuCrossAttention(node);

      // --- Convolution / pooling ---
      case "Conv2D": return this._cpuConv2D(node);
      case "Conv1D": return this._cpuConv1D(node);
      case "ConvTranspose2D": return this._cpuConvTranspose2D(node);
      case "MaxPool2D": return this._cpuMaxPool2D(node);
      case "AveragePool":
      case "AveragePool2D": return this._cpuAveragePool2D(node);
      case "GlobalAveragePool": return this._cpuGlobalAveragePool(node);
      case "BatchNorm2D": return this._cpuBatchNorm2D(node);
      case "ResizeNearest2D":
      case "Resize": return this._cpuResize(node);
      case "Upsample2x":
      case "UpsampleNearest2D": return this._cpuUpsample2x(node);
      case "Interp1D":
      case "InterpLinear1D": return this._cpuInterp1D(node);

      // --- Activations ---
      case "ReLU": return this._cpuReLU(node);
      case "LeakyReLU": return this._cpuLeakyReLU(node);
      case "PReLU": return this._cpuPReLU(node);
      case "GELU": return this._cpuGELU(node);
      case "SiLU":
      case "Swish": return this._cpuSiLU(node);
      case "Sigmoid": return this._cpuSigmoid(node);
      case "HardSwish": return this._cpuHardSwish(node);
      case "HardSigmoid": return this._cpuHardSigmoid(node);
      case "Tanh": return this._cpuTanh(node);
      case "Clip": return this._cpuClip(node);

      // --- Elementwise / reduction ---
      case "Add": return this._cpuAdd(node);
      case "Mul": return this._cpuMul(node);
      case "Sub": return this._cpuSub(node);
      case "Div": return this._cpuDiv(node);
      case "Softmax": return this._cpuSoftmax(node);
      case "LogSoftmax": return this._cpuLogSoftmax(node);
      case "ReduceSum": return this._cpuReduceSum(node);
      case "ReduceMean": return this._cpuReduceMean(node);
      case "ArgMax": return this._cpuArgMax(node);

      // --- Shape / gather / misc ---
      case "Transpose": return this._cpuTranspose(node);
      case "Concat":
      case "Concat2": return this._cpuConcat2(node);
      case "Split": return this._cpuSplit(node);
      case "Slice": return this._cpuSlice(node);
      case "Pad": return this._cpuPad(node);
      case "Expand":
      case "Broadcast": return this._cpuExpand(node);
      case "Gather": return this._cpuGather(node);
      case "GatherElements": return this._cpuGatherElements(node);
      case "Where":
      case "Mask": return this._cpuWhere(node);
      case "Cast": return this._cpuCast(node);
      case "DequantizeLinear": return this._cpuDequantizeLinear(node);
      case "NonMaxSuppression": return this._cpuNonMaxSuppression(node);
      case "SpatialSoftargmaxY": return this._cpuSpatialSoftargmaxY(node);
      case "ProfileX": return this._cpuProfileX(node);
      case "ProfileY": return this._cpuProfileY(node);
      case "MeanHeight": return this._cpuMeanHeight(node);

      // Shape-only ops just copy their data through to the output buffer.
      case "Reshape":
      case "Flatten":
      case "Squeeze":
      case "Unsqueeze":
      case "Dropout":
      case "Identity": return this._cpuReshape(node);

      default:
        console.warn(`[VolvoxAI CPU] Executing ${node.opType} is not implemented; node ${node.id} skipped.`);
    }
  }
};
CPUEngine.prototype._cpuWhere = _cpuWhere;
CPUEngine.prototype._cpuPad = _cpuPad;
CPUEngine.prototype._cpuAveragePool2D = _cpuAveragePool2D;
CPUEngine.prototype._cpuSlice = _cpuSlice;
CPUEngine.prototype._cpuConvTranspose2D = _cpuConvTranspose2D;
CPUEngine.prototype._cpuReduceSum = _cpuReduceSum;
CPUEngine.prototype._cpuReduceMean = _cpuReduceMean;
CPUEngine.prototype._cpuBatchNorm2D = _cpuBatchNorm2D;
CPUEngine.prototype._cpuSoftmax = _cpuSoftmax;
CPUEngine.prototype._cpuLogSoftmax = _cpuLogSoftmax;
CPUEngine.prototype._cpuSub = _cpuSub;
CPUEngine.prototype._cpuSiLU = _cpuSiLU;
CPUEngine.prototype._cpuRMSNorm = _cpuRMSNorm;
CPUEngine.prototype._cpuTanh = _cpuTanh;
CPUEngine.prototype._cpuDequantizeLinear = _cpuDequantizeLinear;
CPUEngine.prototype._cpuExpand = _cpuExpand;
CPUEngine.prototype._cpuPReLU = _cpuPReLU;
CPUEngine.prototype._cpuDiv = _cpuDiv;
CPUEngine.prototype._cpuNonMaxSuppression = _cpuNonMaxSuppression;
CPUEngine.prototype._cpuGather = _cpuGather;
CPUEngine.prototype._cpuGatherElements = _cpuGatherElements;
CPUEngine.prototype._cpuCrossAttention = _cpuCrossAttention;
CPUEngine.prototype._cpuSpatialSoftargmaxY = _cpuSpatialSoftargmaxY;
CPUEngine.prototype._cpuTranspose = _cpuTranspose;
CPUEngine.prototype._cpuCrossSDPA = _cpuCrossSDPA;
CPUEngine.prototype._cpuMeanHeight = _cpuMeanHeight;
CPUEngine.prototype._cpuMaxPool2D = _cpuMaxPool2D;
CPUEngine.prototype._cpuInterp1D = _cpuInterp1D;
CPUEngine.prototype._cpuProfileX = _cpuProfileX;
CPUEngine.prototype._cpuProfileY = _cpuProfileY;
CPUEngine.prototype._cpuConcat2 = _cpuConcat2;
CPUEngine.prototype._cpuUpsample2x = _cpuUpsample2x;
CPUEngine.prototype._cpuSDPA = _cpuSDPA;
CPUEngine.prototype._cpuEmbedding = _cpuEmbedding;
CPUEngine.prototype._cpuMul = _cpuMul;
CPUEngine.prototype._cpuAdd = _cpuAdd;
CPUEngine.prototype._cpuGlobalAveragePool = _cpuGlobalAveragePool;
CPUEngine.prototype._cpuClip = _cpuClip;
CPUEngine.prototype._cpuReshape = _cpuReshape;
CPUEngine.prototype._cpuSplit = _cpuSplit;
CPUEngine.prototype._cpuResize = _cpuResize;
CPUEngine.prototype._cpuSigmoid = _cpuSigmoid;
CPUEngine.prototype._cpuHardSigmoid = _cpuHardSigmoid;
CPUEngine.prototype._cpuHardSwish = _cpuHardSwish;
CPUEngine.prototype._cpuReLU = _cpuReLU;
CPUEngine.prototype._cpuLeakyReLU = _cpuLeakyReLU;
CPUEngine.prototype._cpuGELU = _cpuGELU;
CPUEngine.prototype._cpuLayerNorm = _cpuLayerNorm;
CPUEngine.prototype._cpuConv2D = _cpuConv2D;
CPUEngine.prototype._cpuConv1D = _cpuConv1D;
CPUEngine.prototype._cpuMatMul = _cpuMatMul;
CPUEngine.prototype._cpuCast = _cpuCast;
CPUEngine.prototype._cpuArgMax = _cpuArgMax;
