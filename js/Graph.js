import { Tensor } from './Tensor.js';

export class Graph {
  constructor() {
    this.nodes = [];
    this.tensors = /* @__PURE__ */ new Map();
  }
  /**
   * Define an input tensor
   */
  addInput(name, shape, dtype = "float32") {
    const tensor = new Tensor(name, shape, dtype, false);
    this.tensors.set(name, tensor);
    return tensor;
  }
  /**
   * Define a weight tensor (learned parameter)
   */
  addWeight(name, shape, dtype = "float32") {
    const tensor = new Tensor(name, shape, dtype, true);
    this.tensors.set(name, tensor);
    return tensor;
  }
  /**
   * Add a computation operation to the graph
   * @param {string} opType - e.g., 'MatMul', 'Conv2D', 'LayerNorm'
   * @param {Object} inputs - Key-value pair of input names to Tensor objects
   * @param {Object} outputs - Key-value pair of output names to Tensor shapes
   * @param {Object} params - Uniform parameters for the shader (e.g., stride, kernel size)
   */
  addOp(opType, inputs, outputs, params = {}) {
    const outTensors = {};
    for (const [key, shape] of Object.entries(outputs)) {
      const outName = `${opType}_${this.nodes.length}_out_${key}`;
      const t = new Tensor(outName, shape, "float32", false);
      this.tensors.set(outName, t);
      outTensors[key] = t;
    }
    const node = {
      id: this.nodes.length,
      opType,
      inputs,
      outputs: outTensors,
      params
    };
    this.nodes.push(node);
    return outTensors;
  }
};