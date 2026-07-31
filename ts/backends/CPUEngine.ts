import { _cpuPReLU } from '../ops/pReLU.js';
import { _cpuExpand } from '../ops/expand.js';
import { _cpuDequantizeLinear } from '../ops/dequantizeLinear.js';
import { _cpuQuantizeLinear } from '../ops/quantizeLinear.js';
import { _cpuRequantizeLinear } from '../ops/requantizeLinear.js';
import { _cpuQAdd } from '../ops/qAdd.js';
import { _cpuQConv2D } from '../ops/qConv2D.js';
import { _cpuQEmbedding } from '../ops/qEmbedding.js';
import { _cpuQGELU } from '../ops/qGELU.js';
import { _cpuQGroupNorm } from '../ops/qGroupNorm.js';
import { _cpuQLayerNorm } from '../ops/qLayerNorm.js';
import { _cpuQLinear } from '../ops/qLinear.js';
import { _cpuQMaskedMean } from '../ops/qMaskedMean.js';
import { _cpuQSDPA } from '../ops/qSDPA.js';
import { _cpuQSiLU } from '../ops/qSiLU.js';
import { _cpuTanh } from '../ops/tanh.js';
import { _cpuSin } from '../ops/sin.js';
import { _cpuCos } from '../ops/cos.js';
import { _cpuRoPE } from '../ops/roPE.js';
import { _cpuSSMScan } from '../ops/ssmScan.js';
import { _cpuRMSNorm } from '../ops/rMSNorm.js';
import { _cpuSiLU } from '../ops/siLU.js';
import { _cpuSub } from '../ops/sub.js';
import { _cpuLogSoftmax } from '../ops/logSoftmax.js';
import { _cpuSoftmax } from '../ops/softmax.js';
import { _cpuWhere } from '../ops/where.js';
import { _cpuPad } from '../ops/pad.js';
import { _cpuAveragePool2D } from '../ops/averagePool2D.js';
import { _cpuSlice } from '../ops/slice.js';
import { _cpuConvTranspose2D } from '../ops/convTranspose2D.js';
import { _cpuReduceSum } from '../ops/reduceSum.js';
import { _cpuReduceMean } from '../ops/reduceMean.js';
import { _cpuBatchNorm2D } from '../ops/batchNorm2D.js';
import { _cpuDiv } from '../ops/div.js';
import { _cpuNonMaxSuppression } from '../ops/nonMaxSuppression.js';
import { _cpuGather } from '../ops/gather.js';
import { _cpuGatherElements } from '../ops/gatherElements.js';
import { _cpuCrossAttention } from '../ops/crossAttention.js';
import { _cpuSpatialSoftargmaxY } from '../ops/spatialSoftargmaxY.js';
import { _cpuTranspose } from '../ops/transpose.js';
import { _cpuCrossSDPA } from '../ops/crossSDPA.js';
import { _cpuMeanHeight } from '../ops/meanHeight.js';
import { _cpuMaxPool2D } from '../ops/maxPool2D.js';
import { _cpuInterp1D } from '../ops/interp1D.js';
import { _cpuProfileX } from '../ops/profileX.js';
import { _cpuProfileY } from '../ops/profileY.js';
import { _cpuConcat2 } from '../ops/concat2.js';
import { _cpuUpsample2x } from '../ops/upsample2x.js';
import { _cpuSDPA } from '../ops/sDPA.js';
import { _cpuEmbedding } from '../ops/embedding.js';
import { _cpuMul } from '../ops/mul.js';
import { _cpuAdd } from '../ops/add.js';
import { _cpuGlobalAveragePool } from '../ops/globalAveragePool.js';
import { _cpuClip } from '../ops/clip.js';
import { _cpuReshape } from '../ops/reshape.js';
import { _cpuSplit } from '../ops/split.js';
import { _cpuResize } from '../ops/resize.js';
import { _cpuSigmoid } from '../ops/sigmoid.js';
import { _cpuHardSigmoid } from '../ops/hardSigmoid.js';
import { _cpuHardSwish } from '../ops/hardSwish.js';
import { _cpuReLU } from '../ops/reLU.js';
import { _cpuLeakyReLU } from '../ops/leakyReLU.js';
import { _cpuGELU } from '../ops/gELU.js';
import { _cpuLayerNorm } from '../ops/layerNorm.js';
import { _cpuGroupNorm } from '../ops/groupNorm.js';
import { _cpuConv2D } from '../ops/conv2D.js';
import { _cpuConv1D } from '../ops/conv1D.js';
import { _cpuMatMul } from '../ops/matMul.js';
import { _cpuBatchMatMul } from '../ops/batchMatMul.js';
import { _cpuQBatchMatMul } from '../ops/qBatchMatMul.js';
import { _cpuComparison, _cpuNot } from '../ops/comparison.js';
import { _cpuLoRALinear } from '../ops/loraLinear.js';
import { _cpuCast } from '../ops/cast.js';
import { _cpuArgMax } from '../ops/argMax.js';
import { _cpuQArgMax } from '../ops/qArgMax.js';
import { _cpuMoERouter } from '../ops/moeRouter.js';
import { _cpuMoELinear } from '../ops/moeLinear.js';
import { Tensor } from '../core/Tensor.js';
import { assertInferenceExecutionOptions, BackendEngine } from './BackendEngine.js';
import { incrementalExecutionEnabled, incrementalNodeSelection } from './incrementalExecution.js';
import {
  incrementalRowPosition,
  prepareQuantizedRows,
} from './quantizedRowExecution.js';
import type { Graph } from '../core/Graph.js';
import type { GraphNode, TensorLike } from '../types.js';

interface CpuNodeExecution {
  graph?: Graph;
  adapterPlan?: any;
  nodeIndex?: number;
  rowPosition?: number | null;
}

function aliasInferenceDropout(node: GraphNode): void {
  const input = node.inputs.input || node.inputs.x;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  if (input?.dtype !== 'float32' || output?.dtype !== 'float32' ||
      !(input.buffer instanceof Float32Array) || !(output.buffer instanceof Float32Array) ||
      input.buffer.length !== output.buffer.length) {
    throw new Error(`Dropout node ${node.id ?? '<unnamed>'} requires equal-size F32 input/output tensors.`);
  }
  output.buffer = input.buffer;
}

export class CPUEngine extends BackendEngine {
  tensors: Map<string, TensorLike>;
  graph?: Graph;
  declare compiledTopologyRevision: number;

  declare _cpuAdd: typeof _cpuAdd;
  declare _cpuArgMax: typeof _cpuArgMax;
  declare _cpuAveragePool2D: typeof _cpuAveragePool2D;
  declare _cpuBatchNorm2D: typeof _cpuBatchNorm2D;
  declare _cpuCast: typeof _cpuCast;
  declare _cpuClip: typeof _cpuClip;
  declare _cpuConcat2: typeof _cpuConcat2;
  declare _cpuConv1D: typeof _cpuConv1D;
  declare _cpuConv2D: typeof _cpuConv2D;
  declare _cpuConvTranspose2D: typeof _cpuConvTranspose2D;
  declare _cpuCos: typeof _cpuCos;
  declare _cpuCrossAttention: typeof _cpuCrossAttention;
  declare _cpuCrossSDPA: typeof _cpuCrossSDPA;
  declare _cpuDequantizeLinear: typeof _cpuDequantizeLinear;
  declare _cpuDiv: typeof _cpuDiv;
  declare _cpuEmbedding: typeof _cpuEmbedding;
  declare _cpuExpand: typeof _cpuExpand;
  declare _cpuGELU: typeof _cpuGELU;
  declare _cpuGather: typeof _cpuGather;
  declare _cpuGatherElements: typeof _cpuGatherElements;
  declare _cpuGlobalAveragePool: typeof _cpuGlobalAveragePool;
  declare _cpuGroupNorm: typeof _cpuGroupNorm;
  declare _cpuHardSigmoid: typeof _cpuHardSigmoid;
  declare _cpuHardSwish: typeof _cpuHardSwish;
  declare _cpuInterp1D: typeof _cpuInterp1D;
  declare _cpuLayerNorm: typeof _cpuLayerNorm;
  declare _cpuLeakyReLU: typeof _cpuLeakyReLU;
  declare _cpuLoRALinear: typeof _cpuLoRALinear;
  declare _cpuLogSoftmax: typeof _cpuLogSoftmax;
  declare _cpuMatMul: typeof _cpuMatMul;
  declare _cpuBatchMatMul: typeof _cpuBatchMatMul;
  declare _cpuQBatchMatMul: typeof _cpuQBatchMatMul;
  declare _cpuComparison: typeof _cpuComparison;
  declare _cpuNot: typeof _cpuNot;
  declare _cpuMaxPool2D: typeof _cpuMaxPool2D;
  declare _cpuMeanHeight: typeof _cpuMeanHeight;
  declare _cpuMoELinear: typeof _cpuMoELinear;
  declare _cpuMoERouter: typeof _cpuMoERouter;
  declare _cpuMul: typeof _cpuMul;
  declare _cpuNonMaxSuppression: typeof _cpuNonMaxSuppression;
  declare _cpuPReLU: typeof _cpuPReLU;
  declare _cpuPad: typeof _cpuPad;
  declare _cpuProfileX: typeof _cpuProfileX;
  declare _cpuProfileY: typeof _cpuProfileY;
  declare _cpuQAdd: typeof _cpuQAdd;
  declare _cpuQArgMax: typeof _cpuQArgMax;
  declare _cpuQConv2D: typeof _cpuQConv2D;
  declare _cpuQEmbedding: typeof _cpuQEmbedding;
  declare _cpuQGELU: typeof _cpuQGELU;
  declare _cpuQGroupNorm: typeof _cpuQGroupNorm;
  declare _cpuQLayerNorm: typeof _cpuQLayerNorm;
  declare _cpuQLinear: typeof _cpuQLinear;
  declare _cpuQMaskedMean: typeof _cpuQMaskedMean;
  declare _cpuQSDPA: typeof _cpuQSDPA;
  declare _cpuQSiLU: typeof _cpuQSiLU;
  declare _cpuQuantizeLinear: typeof _cpuQuantizeLinear;
  declare _cpuRMSNorm: typeof _cpuRMSNorm;
  declare _cpuReLU: typeof _cpuReLU;
  declare _cpuReduceMean: typeof _cpuReduceMean;
  declare _cpuReduceSum: typeof _cpuReduceSum;
  declare _cpuRequantizeLinear: typeof _cpuRequantizeLinear;
  declare _cpuReshape: typeof _cpuReshape;
  declare _cpuResize: typeof _cpuResize;
  declare _cpuRoPE: typeof _cpuRoPE;
  declare _cpuSDPA: typeof _cpuSDPA;
  declare _cpuSSMScan: typeof _cpuSSMScan;
  declare _cpuSiLU: typeof _cpuSiLU;
  declare _cpuSigmoid: typeof _cpuSigmoid;
  declare _cpuSin: typeof _cpuSin;
  declare _cpuSlice: typeof _cpuSlice;
  declare _cpuSoftmax: typeof _cpuSoftmax;
  declare _cpuSpatialSoftargmaxY: typeof _cpuSpatialSoftargmaxY;
  declare _cpuSplit: typeof _cpuSplit;
  declare _cpuSub: typeof _cpuSub;
  declare _cpuTanh: typeof _cpuTanh;
  declare _cpuTranspose: typeof _cpuTranspose;
  declare _cpuUpsample2x: typeof _cpuUpsample2x;
  declare _cpuWhere: typeof _cpuWhere;

  constructor() {
    super('cpu', {
      incrementalExecution: true,
      incrementalRows: true,
      outputLocation: 'host',
    });
    this.tensors = /* @__PURE__ */ new Map();
    this._incrementalCacheValid = false;
    console.log("[VolvoxAI] CPU Fallback Engine ready.");
  }

  /** Create an independent graph owner for multi-graph inference sessions. */
  fork(): CPUEngine | Promise<CPUEngine> {
    return new CPUEngine();
  }

  /** Release graph-owned request storage when its execution context closes. */
  dispose(): void {
    this.resetDecodeCache();
    this.graph = undefined;
    this.tensors.clear();
  }

  /**
   * Allocates CPU memory (ArrayBuffers) for the graph's tensors.
   */
  allocateGraph(graph: Graph) {
    this._assertPortableQuantizedGraph(graph);
    this.graph = graph;
    this.resetDecodeCache();
    for (const [name, tensor] of graph.tensors.entries()) {
      if (!tensor.buffer) {
        if (tensor.dtype === "int8") {
          tensor.buffer = new Int8Array(tensor.sizeBytes);
        } else if (tensor.dtype === "uint8") {
          tensor.buffer = new Uint8Array(tensor.sizeBytes);
        } else if (tensor.dtype === "int32") {
          tensor.buffer = new Int32Array(tensor.sizeBytes / 4);
        } else {
          tensor.buffer = new Float32Array(tensor.sizeBytes / 4);
        }
      }
    }
    this.compiledTopologyRevision = graph.topologyRevision || 0;
    return this;
  }
  /**
   * Executes the graph linearly on the CPU.
   */
  async execute(inputsOrGraph: any, maybeInputs?: any, maybeOptions: Record<string, any> = {}) {
    const explicitGraph = inputsOrGraph?.tensors instanceof Map && Array.isArray(inputsOrGraph?.nodes);
    const graph = explicitGraph ? inputsOrGraph : this.graph;
    const inputs = explicitGraph ? (maybeInputs || {}) : (inputsOrGraph || {});
    const options = explicitGraph ? maybeOptions : (maybeInputs || {});
    if (!graph) throw new Error("CPUEngine.execute requires an allocated graph.");
    assertInferenceExecutionOptions(options, 'CPU inference');
    if (!explicitGraph) graph.assertTopologyRevision?.(this.compiledTopologyRevision, "CPU");
    this._beginDecodeExecution(options);
    const hasAdapterSelector = Object.prototype.hasOwnProperty.call(options, "adapter") ||
      Object.prototype.hasOwnProperty.call(options, "adapters");
    const adapterPlan = (hasAdapterSelector || graph.adapters?.hasActive())
      ? graph.adapters._pinExecution(options)
      : null;
    const incremental = incrementalExecutionEnabled(options, adapterPlan);
    const cacheWasValid = this._incrementalCacheValid;
    const selectedNodes = incremental
      ? incrementalNodeSelection(graph, inputs, options, cacheWasValid)
      : null;
    /* Publish validity only after every selected node completes. A reset or
     * partial failure must never expose intermediates from the prior image. */
    this._incrementalCacheValid = false;
    const rowPosition = incrementalRowPosition(options, selectedNodes, cacheWasValid);
    const incrementalRows = rowPosition == null
      ? null
      : prepareQuantizedRows(graph, selectedNodes, rowPosition, {
          changedInputs: options.changedInputs ?? Object.keys(inputs),
        });
    for (const [name, data] of Object.entries(inputs)) {
      const tensor = graph.tensors.get(name);
      if (!tensor?.isInput || !tensor.buffer) throw new Error(`Unknown graph input '${name}'.`);
      Tensor.assertCompatibleInput(tensor.dtype, data, tensor.sizeBytes, `Input '${name}'`);
      tensor.buffer.set(data);
    }
    for (let nodeIndex = 0; nodeIndex < graph.nodes.length; nodeIndex++) {
      if (selectedNodes && !selectedNodes.has(nodeIndex)) continue;
      this._runNode(incrementalRows?.get(nodeIndex) || graph.nodes[nodeIndex], {
        graph, adapterPlan, nodeIndex, rowPosition,
      });
    }
    const result: Record<string, any> = {};
    if (graph.outputNames && graph.outputNames.length > 0) {
      for (const name of graph.outputNames) {
        result[name] = graph.tensors.get(name).buffer;
      }
    } else {
      const lastNode = graph.nodes[graph.nodes.length - 1];
      for (const [, t] of Object.entries(lastNode.outputs) as Array<[string, TensorLike]>) {
        result[t.name] = graph.tensors.get(t.name).buffer;
      }
    }
    if (incremental) this._incrementalCacheValid = true;
    return result;
  }
  _runNode(node: GraphNode, execution: CpuNodeExecution = {}) {
    switch (node.opType) {
      // --- Linear / attention ---
      case "MatMul":
      case "Linear":
      case "Gemm": {
        const input = node.inputs.input || node.inputs.x || node.inputs.a;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        if (['int8', 'uint8'].includes(input?.dtype) && ['int8', 'uint8'].includes(output?.dtype)) {
          // The canonical W8A8 form uses the explicit QLinear operand names;
          // retaining these aliases lets exporters normalize QMatMul/QGemm
          // without accidentally selecting the older W8A32 implementation.
          return this._cpuQLinear({ ...node, inputs: { ...node.inputs, input }, outputs: { ...node.outputs, out: output } });
        }
        const result = this._cpuMatMul(node);
        if (execution.adapterPlan) this._cpuLoRALinear(node, execution.adapterPlan);
        return result;
      }
      case "BatchMatMul": return this._cpuBatchMatMul(node);
      case "QBatchMatMul": return this._cpuQBatchMatMul(node);
      case "QLinear":
      case "QMatMul":
      case "QGemm": {
        const input = node.inputs.input || node.inputs.x || node.inputs.a;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        return this._cpuQLinear({ ...node, inputs: { ...node.inputs, input }, outputs: { ...node.outputs, out: output } });
      }
      case "LayerNorm": return this._cpuLayerNorm(node);
      case "GroupNorm": return this._cpuGroupNorm(node);
      case "QGroupNorm": return this._cpuQGroupNorm(node);
      case "QLayerNorm": return this._cpuQLayerNorm(node);
      case "QMaskedMean": return this._cpuQMaskedMean(node);
      case "QSDPA": return this._cpuQSDPA(node);
      case "RMSNorm": return this._cpuRMSNorm(node);
      case "Embedding": return this._cpuEmbedding(node);
      case "QEmbedding": return this._cpuQEmbedding(node);
      case "MoERouter": return this._cpuMoERouter(node);
      case "MoELinear": return this._cpuMoELinear(node);
      case "SDPA": return this._cpuSDPA(node);
      case "CrossSDPA": return this._cpuCrossSDPA(node);
      case "CrossAttention": return this._cpuCrossAttention(node);

      // --- Convolution / pooling ---
      case "Conv2D": return this._cpuConv2D(node);
      case "QConv2D": return this._cpuQConv2D(node);
      case "Conv1D": return this._cpuConv1D(node);
      case "ConvTranspose2D": return this._cpuConvTranspose2D(node);
      case "MaxPool2D": return this._cpuMaxPool2D(node);
      case "AveragePool2D": return this._cpuAveragePool2D(node);
      case "GlobalAveragePool": return this._cpuGlobalAveragePool(node);
      case "BatchNorm2D": return this._cpuBatchNorm2D(node);
      case "ResizeNearest2D":
      case "Resize": return this._cpuResize(node);
      case "UpsampleNearest2D": return this._cpuUpsample2x(node);
      case "Interpolate1D": return this._cpuInterp1D(node);

      // --- Activations ---
      case "ReLU": return this._cpuReLU(node);
      case "LeakyReLU": return this._cpuLeakyReLU(node);
      case "PReLU": return this._cpuPReLU(node);
      case "GELU": return this._cpuGELU(node);
      case "SiLU": return this._cpuSiLU(node);
      case "QGELU": return this._cpuQGELU(node);
      case "QSiLU": return this._cpuQSiLU(node);
      case "Sigmoid": return this._cpuSigmoid(node);
      case "HardSwish": return this._cpuHardSwish(node);
      case "HardSigmoid": return this._cpuHardSigmoid(node);
      case "Tanh": return this._cpuTanh(node);
      case "Sin": return this._cpuSin(node);
      case "Cos": return this._cpuCos(node);
      case "RoPE": return this._cpuRoPE(node);
      case "SSMScan":
      case "SelectiveScan": return this._cpuSSMScan(node);
      case "Clip": return this._cpuClip(node);

      // --- Elementwise / reduction ---
      case "Add": return this._cpuAdd(node);
      case "QAdd": return this._cpuQAdd(node);
      case "Mul": return this._cpuMul(node);
      case "Sub": return this._cpuSub(node);
      case "Div": return this._cpuDiv(node);
      case "Equal":
      case "GreaterOrEqual": return this._cpuComparison(node);
      case "Not": return this._cpuNot(node);
      case "Softmax": return this._cpuSoftmax(node);
      case "LogSoftmax": return this._cpuLogSoftmax(node);
      case "ReduceSum": return this._cpuReduceSum(node);
      case "ReduceMean": return this._cpuReduceMean(node);
      case "ArgMax": return this._cpuArgMax(node);
      case "QArgMax": return this._cpuQArgMax(node);

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
      case "QuantizeLinear": return this._cpuQuantizeLinear(node);
      case "RequantizeLinear": return this._cpuRequantizeLinear(node);
      case "DequantizeLinear": return this._cpuDequantizeLinear(node);
      case "NonMaxSuppression": return this._cpuNonMaxSuppression(node);
      case "SpatialSoftargmaxY": return this._cpuSpatialSoftargmaxY(node);
      case "ProfileX": return this._cpuProfileX(node);
      case "ProfileY": return this._cpuProfileY(node);
      case "MeanHeight": return this._cpuMeanHeight(node);

      case "Dropout": return aliasInferenceDropout(node);

      // Shape-only ops just copy their data through to the output buffer.
      case "Reshape":
      case "Flatten":
      case "Squeeze":
      case "Unsqueeze":
      case "Identity": return this._cpuReshape(node);

      default:
        throw new Error(
          `[VolvoxAI CPU] Operator ${node.opType} at node ${node.id} is not implemented.`,
        );
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
CPUEngine.prototype._cpuSin = _cpuSin;
CPUEngine.prototype._cpuCos = _cpuCos;
CPUEngine.prototype._cpuRoPE = _cpuRoPE;
CPUEngine.prototype._cpuSSMScan = _cpuSSMScan;
CPUEngine.prototype._cpuDequantizeLinear = _cpuDequantizeLinear;
CPUEngine.prototype._cpuQuantizeLinear = _cpuQuantizeLinear;
CPUEngine.prototype._cpuRequantizeLinear = _cpuRequantizeLinear;
CPUEngine.prototype._cpuQAdd = _cpuQAdd;
CPUEngine.prototype._cpuQConv2D = _cpuQConv2D;
CPUEngine.prototype._cpuQEmbedding = _cpuQEmbedding;
CPUEngine.prototype._cpuQGELU = _cpuQGELU;
CPUEngine.prototype._cpuQGroupNorm = _cpuQGroupNorm;
CPUEngine.prototype._cpuQLayerNorm = _cpuQLayerNorm;
CPUEngine.prototype._cpuQLinear = _cpuQLinear;
CPUEngine.prototype._cpuQMaskedMean = _cpuQMaskedMean;
CPUEngine.prototype._cpuQSDPA = _cpuQSDPA;
CPUEngine.prototype._cpuQSiLU = _cpuQSiLU;
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
CPUEngine.prototype._cpuGroupNorm = _cpuGroupNorm;
CPUEngine.prototype._cpuConv2D = _cpuConv2D;
CPUEngine.prototype._cpuConv1D = _cpuConv1D;
CPUEngine.prototype._cpuMatMul = _cpuMatMul;
CPUEngine.prototype._cpuBatchMatMul = _cpuBatchMatMul;
CPUEngine.prototype._cpuQBatchMatMul = _cpuQBatchMatMul;
CPUEngine.prototype._cpuComparison = _cpuComparison;
CPUEngine.prototype._cpuNot = _cpuNot;
CPUEngine.prototype._cpuLoRALinear = _cpuLoRALinear;
CPUEngine.prototype._cpuCast = _cpuCast;
CPUEngine.prototype._cpuArgMax = _cpuArgMax;
CPUEngine.prototype._cpuQArgMax = _cpuQArgMax;
CPUEngine.prototype._cpuMoERouter = _cpuMoERouter;
CPUEngine.prototype._cpuMoELinear = _cpuMoELinear;
