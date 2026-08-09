import { Tensor } from '../core/Tensor.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { RuntimeTypedArray } from '../types.js';
import { incrementalNodeSelection } from './incrementalExecution.js';
import { quantizedRowNode } from './quantizedRowExecution.js';
import type { GraphExecutor } from './GraphExecutor.js';
import type {
  DeviceFeedbackDecodeOptions,
  DeviceFeedbackDescriptor,
  DeviceFeedbackState,
  ExecutorNode,
  ExecutorTensor,
  IncrementalRowByteCopy,
  IncrementalRowCandidate,
  IncrementalRowPlan,
  WebGPUExecutionInputs,
} from './WebGPUContracts.js';

/** Context-local incremental and device-feedback decode owner. */
export class WebGPUDecodeState {
  readonly rowCandidates = new Map<number, IncrementalRowCandidate>();
  readonly rowPlans = new Map<number, IncrementalRowPlan>();
  readonly rowCopyTensorNames = new Set<string>();
  controlBuffer: GPUBuffer | null = null;
  sequenceLength = 0;
  feedback: Readonly<DeviceFeedbackState> | null = null;

  constructor(readonly host: GraphExecutor) {}

  resetExecution(): void {
    this.feedback = null;
  }

  /** Drop shape/seed-specific row pipelines and replay progress, not immutable candidates. */
  resetBinding(): void {
    this.rowPlans.clear();
    this.feedback = null;
  }

  resetCompilation(): void {
    this.rowCandidates.clear();
    this.rowPlans.clear();
    this.rowCopyTensorNames.clear();
    this.controlBuffer = null;
    this.sequenceLength = 0;
    this.feedback = null;
  }

    _rowTypedStorage(
      tensor: ExecutorTensor,
      sizeBytes: number = tensor.sizeBytes,
    ): RuntimeTypedArray {
      const elements = sizeBytes / Tensor.dtypeBytes(tensor.dtype);
      if (!Number.isSafeInteger(elements) || elements < 0) {
        throw new Error(`WebGPU incremental row tensor '${tensor?.name}' has invalid storage size.`);
      }
      if (tensor.dtype === 'float32') return new Float32Array(elements);
      if (tensor.dtype === 'int32') return new Int32Array(elements);
      if (tensor.dtype === 'int8') return new Int8Array(elements);
      if (tensor.dtype === 'uint8') return new Uint8Array(elements);
      throw new Error(`WebGPU incremental row tensor '${tensor?.name}' has unsupported dtype '${tensor?.dtype}'.`);
    }

    _prepareIncrementalRowNode(node: ExecutorNode, position: number): ExecutorNode {
      const shadows = new Map<ExecutorTensor, ExecutorTensor>();
      const shadow = (tensor: ExecutorTensor | undefined): ExecutorTensor | undefined => {
        if (!tensor) return tensor;
        let value = shadows.get(tensor);
        if (value) return value;
        if (ArrayBuffer.isView(tensor.buffer) && !(tensor.buffer instanceof DataView)) {
          value = tensor;
        } else {
          value = Object.assign(Object.create(Object.getPrototypeOf(tensor)), tensor, {
            buffer: this.host._rowTypedStorage(tensor),
          });
        }
        shadows.set(tensor, value as ExecutorTensor);
        return value as ExecutorTensor;
      };
      const storageNode = {
        ...node,
        inputs: Object.fromEntries(Object.entries(node.inputs || {}).map(([name, tensor]) =>
          [name, shadow(tensor)])),
        outputs: Object.fromEntries(Object.entries(node.outputs || {}).map(([name, tensor]) =>
          [name, shadow(tensor)])),
      };
      // Candidate construction proves storage/layout only. The changed-input
      // closure does not exist during global WebGPU candidate discovery;
      // requiring it here would incorrectly remove otherwise valid invariant
      // cross-attention and broadcast candidates before execution is requested.
      const prepared = quantizedRowNode(storageNode, position, {
        allowUnprovenInvariantInputs: true,
      }) as ExecutorNode;
      const normalize = (
        rowTensor: ExecutorTensor,
        originalTensor: ExecutorTensor | undefined,
        shadowTensor: ExecutorTensor | undefined,
        role: string,
        name: string,
      ): ExecutorTensor => {
        if (rowTensor === shadowTensor) return originalTensor as ExecutorTensor;
        const rowStorage = rowTensor?.buffer;
        const fullStorage = shadowTensor?.buffer;
        if (!ArrayBuffer.isView(rowStorage) || !ArrayBuffer.isView(fullStorage) ||
            rowStorage.buffer !== fullStorage.buffer) {
          throw new Error(`WebGPU incremental row node ${node.id} produced invalid ${role} '${name}' storage.`);
        }
        const offset = rowStorage.byteOffset - fullStorage.byteOffset;
        const normalized = Object.assign(Object.create(Object.getPrototypeOf(rowTensor)), rowTensor, {
          buffer: this.host._rowTypedStorage(rowTensor, rowTensor.sizeBytes),
        });
        normalized._webgpuRow = Object.freeze({
          offsetStride: position > 0 ? offset / position : 0,
          constantSize: rowTensor.sizeBytes,
          queryMaskPrefix: node.opType === 'QSDPA' && role === 'input' && name === 'mask' &&
            node.params?.causal === true && offset > 0,
        });
        return normalized;
      };
      return {
        ...prepared,
        inputs: Object.fromEntries(Object.entries(prepared.inputs).map(([name, rowTensor]) => [
          name,
          normalize(rowTensor, node.inputs?.[name], storageNode.inputs?.[name], 'input', name),
        ])),
        outputs: Object.fromEntries(Object.entries(prepared.outputs).map(([name, rowTensor]) => [
          name,
          normalize(rowTensor, node.outputs?.[name], storageNode.outputs?.[name], 'output', name),
        ])),
      } as ExecutorNode;
    }

    _incrementalRowCandidate(
      node: ExecutorNode,
      nodeIndex: number,
    ): IncrementalRowCandidate | null {
      // Float CrossSDPA row execution currently has no dynamic visible-prefix
      // uniform in the WebGPU pipeline contract. CPU/WASM execute the cloned
      // prefix directly; advertising the static position-one shader here would
      // silently reuse K/V length two at later positions.
      if (node?.opType === 'CrossSDPA') return null;
      const output = node?.outputs?.out || Object.values(node?.outputs || {})[0];
      if (output?.shape?.[0] !== 1 || !Number.isInteger(output.shape?.[1]) ||
          output.shape[1] <= 1) return null;
      let rowNode;
      try {
        rowNode = this.host._prepareIncrementalRowNode(node, 1);
      } catch {
        return null;
      }

      const scratchInputs = new Set<string>();
      const scratchOutputs = new Set<string>();
      const scratchCapacities = new Map<string, number>();
      const byteCopyInputs = new Set<string>();
      const byteCopyOutputs = new Set<string>();
      const invariantInputs = new Set<string>();
      const addScratch = (
        rowTensor: ExecutorTensor | undefined,
        originalTensor: ExecutorTensor | undefined,
        target: Set<string>,
        byteCopyTarget: Set<string>,
        capacity: number | undefined = rowTensor?.sizeBytes,
      ): boolean => {
        if (!rowTensor || !originalTensor || rowTensor === originalTensor || typeof capacity !== 'number' ||
            !Number.isSafeInteger(capacity) || capacity <= 0 ||
            !Number.isSafeInteger(rowTensor.sizeBytes) || rowTensor.sizeBytes <= 0) return false;
        const offset = rowTensor._webgpuRow?.offsetStride;
        if (typeof offset !== 'number' || !Number.isSafeInteger(offset) || offset < 0 ||
            offset + rowTensor.sizeBytes > originalTensor.sizeBytes) return false;
        target.add(rowTensor.name);
        if (offset % 4 !== 0 || rowTensor.sizeBytes % 4 !== 0) {
          byteCopyTarget.add(rowTensor.name);
        }
        scratchCapacities.set(
          rowTensor.name,
          Math.max(scratchCapacities.get(rowTensor.name) || 0, capacity),
        );
        return true;
      };

      for (const [name, rowTensor] of Object.entries(rowNode.inputs) as
        Array<[string, ExecutorTensor]>) {
        const originalTensor = node.inputs?.[name];
        if (rowTensor === originalTensor && originalTensor?.isWeight !== true &&
            originalTensor?.name) {
          invariantInputs.add(originalTensor.name);
        }
        if (node.opType === 'QSDPA' && node.params?.causal !== true &&
            (name === 'k' || name === 'v' || name === 'mask') &&
            originalTensor?.name) {
          invariantInputs.add(originalTensor.name);
        }
        if (!rowTensor || rowTensor === originalTensor) continue;
        if (node.opType === 'QSDPA' && (name === 'k' || name === 'v')) continue;
        if (node.opType === 'QSDPA' && name === 'mask') {
          const prefixStartsAtZero = rowTensor._webgpuRow?.offsetStride === 0;
          if (prefixStartsAtZero) continue;
          const keyWidth = originalTensor?.shape?.at(-1);
          const capacity = typeof keyWidth === 'number' && Number.isSafeInteger(keyWidth)
            ? keyWidth * Int32Array.BYTES_PER_ELEMENT
            : 0;
          if (!addScratch(rowTensor, originalTensor, scratchInputs, byteCopyInputs, capacity)) return null;
          continue;
        }
        if (!addScratch(rowTensor, originalTensor, scratchInputs, byteCopyInputs)) return null;
      }
      for (const [name, rowTensor] of Object.entries(rowNode.outputs) as
        Array<[string, ExecutorTensor]>) {
        const originalTensor = node.outputs?.[name];
        if (!rowTensor || rowTensor === originalTensor) continue;
        if (!addScratch(rowTensor, originalTensor, scratchOutputs, byteCopyOutputs)) return null;
      }
      if (scratchOutputs.size === 0) return null;
      return {
        nodeIndex,
        node,
        sampleNode: rowNode,
        scratchInputs,
        scratchOutputs,
        scratchCapacities,
        byteCopyInputs,
        byteCopyOutputs,
        invariantInputs,
      };
    }

    _assertIncrementalRowInvariants(
      selectedNodes: Iterable<number>,
      changedInputs: readonly string[],
    ): void {
      const nodeIndices = [...selectedNodes];
      const dirtyTensorNames = new Set(changedInputs);
      for (const nodeIndex of nodeIndices) {
        const node = this.host.graph.nodes[nodeIndex];
        for (const tensor of Object.values(node?.outputs || {})) {
          if (tensor?.name) dirtyTensorNames.add(tensor.name);
        }
      }
      for (const nodeIndex of nodeIndices) {
        const candidate = this.host.incrementalRowCandidates.get(nodeIndex);
        if (!candidate) continue;
        for (const name of candidate.invariantInputs) {
          if (dirtyTensorNames.has(name)) {
            throw new Error(
              `WebGPU incremental row node ${String(candidate.node?.id ?? nodeIndex)} `
              + `input '${name}' must remain invariant.`,
            );
          }
        }
      }
    }

    _collectIncrementalRows(): {
      candidates: Map<number, IncrementalRowCandidate>;
      copyTensorNames: Set<string>;
    } {
      const candidates = new Map<number, IncrementalRowCandidate>();
      const copyTensorNames = new Set<string>();
      if (!(this.host.graph?.tensors instanceof Map)) {
        return { candidates, copyTensorNames };
      }
      for (let nodeIndex = 0; nodeIndex < this.host.graph.nodes.length; nodeIndex++) {
        const candidate = this.host._incrementalRowCandidate(this.host.graph.nodes[nodeIndex], nodeIndex);
        if (!candidate) continue;
        candidates.set(nodeIndex, candidate);
        for (const name of candidate.scratchCapacities.keys()) {
          copyTensorNames.add(name);
        }
      }
      return { candidates, copyTensorNames };
    }

    _publishIncrementalRows(analysis: {
      candidates: ReadonlyMap<number, IncrementalRowCandidate>;
      copyTensorNames: ReadonlySet<string>;
    }): void {
      this.host.incrementalRowPlans.clear();
      this.host.incrementalRowCandidates.clear();
      this.host.incrementalRowCopyTensorNames.clear();
      for (const [nodeIndex, candidate] of analysis.candidates) {
        this.host.incrementalRowCandidates.set(nodeIndex, candidate);
      }
      for (const name of analysis.copyTensorNames) {
        this.host.incrementalRowCopyTensorNames.add(name);
      }
    }

    _analyzeIncrementalRows(): void {
      this._publishIncrementalRows(this._collectIncrementalRows());
    }

    _incrementalRowRange(
      rowTensor: ExecutorTensor | null,
      fullTensor: ExecutorTensor,
      position: number,
      label: string,
    ): { offset: number; size: number } {
      const descriptor = rowTensor?._webgpuRow;
      if (!descriptor || !Number.isInteger(position) || position < 1) {
        throw new Error(`WebGPU incremental row ${label} is missing its row-storage descriptor.`);
      }
      const offset = position * descriptor.offsetStride;
      const size = descriptor.queryMaskPrefix
        ? (position + 1) * Int32Array.BYTES_PER_ELEMENT
        : descriptor.constantSize;
      if (!Number.isSafeInteger(offset) || offset < 0 ||
          !Number.isSafeInteger(size) || size <= 0 ||
          offset + size > fullTensor.sizeBytes) {
        throw new Error(`WebGPU incremental row ${label} requires in-bounds storage.`);
      }
      return { offset, size };
    }

    _encodeIncrementalRowByteCopy(
      commandEncoder: GPUCommandEncoder,
      pipeline: GPUComputePipeline | null,
      copy: IncrementalRowByteCopy | undefined,
      sourceOffset: number,
      destinationOffset: number,
      size: number,
      label: string,
    ): void {
      if (!pipeline || !copy) {
        throw new Error(`WebGPU incremental row ${label} has no packed-byte copy pipeline.`);
      }
      const destinationWordCount = Math.ceil(((destinationOffset % 4) + size) / 4);
      if (![sourceOffset, destinationOffset, size, destinationWordCount].every((value) =>
        Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff) ||
          size === 0 || destinationWordCount === 0 ||
          sourceOffset + size > 0x100000000 || destinationOffset + size > 0x100000000) {
        throw new Error(`WebGPU incremental row ${label} has an invalid packed-byte range.`);
      }
      this.host.device.queue.writeBuffer(copy.paramsBuffer, 0, new Uint32Array([
        sourceOffset, destinationOffset, size, destinationWordCount,
      ]));
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, copy.bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(destinationWordCount / 64), 1, 1);
      passEncoder.end();
    }

    _rowTensorByName(
      tensors: Record<string, ExecutorTensor>,
      name: string,
    ): ExecutorTensor | null {
      return Object.values(tensors).find((tensor) => tensor.name === name) || null;
    }

    _encodeIncrementalRowsInto(
      commandEncoder: GPUCommandEncoder,
      selectedNodes: Iterable<number>,
      rowPosition: number,
      { qsdpaControlBuffer = null }: { qsdpaControlBuffer?: GPUBuffer | null } = {},
    ): void {
      for (const nodeIndex of selectedNodes) {
        const plan = this.host.incrementalRowPlans.get(nodeIndex)!;
        const rowNode = plan.sampleNode;
        if (plan.qsdpaParamsBuffer) {
          const seqKV = plan.node.params?.causal === true
            ? rowPosition + 1
            : plan.node.inputs?.k?.shape?.at(-2);
          if (typeof seqKV !== 'number' || !Number.isInteger(seqKV) ||
              seqKV <= 0 || seqKV > 0xffffffff) {
            throw new Error(`WebGPU QSDPA incremental row node ${rowNode.id} has an invalid visible K/V prefix.`);
          }
          if (qsdpaControlBuffer) {
            commandEncoder.copyBufferToBuffer(
              qsdpaControlBuffer,
              rowPosition * Uint32Array.BYTES_PER_ELEMENT,
              plan.qsdpaParamsBuffer,
              Uint32Array.BYTES_PER_ELEMENT,
              Uint32Array.BYTES_PER_ELEMENT,
            );
          } else {
            this.host.device.queue.writeBuffer(
              plan.qsdpaParamsBuffer, Uint32Array.BYTES_PER_ELEMENT, new Uint32Array([seqKV]),
            );
          }
        }
        for (const name of plan.scratchInputs) {
          const rowTensor = this.host._rowTensorByName(rowNode.inputs, name);
          const fullTensor = this.host.graph.tensors.get(name) as ExecutorTensor;
          const fullBuffer = this.host.gpuBuffers.get(name);
          const scratch = plan.scratchByName.get(name)!;
          const { offset, size } = this.host._incrementalRowRange(
            rowTensor, fullTensor, rowPosition, `input '${name}'`,
          );
          if (size > plan.scratchCapacities.get(name)!) {
            throw new Error(`WebGPU incremental row input '${name}' exceeds its compiled scratch capacity.`);
          }
          if (offset % 4 === 0 && size % 4 === 0) {
            commandEncoder.copyBufferToBuffer(fullBuffer!, offset, scratch, 0, size);
          } else {
            this.host._encodeIncrementalRowByteCopy(
              commandEncoder, plan.byteCopyPipeline, plan.inputByteCopies.get(name),
              offset, 0, size, `input '${name}'`,
            );
          }
        }
        const passEncoder = commandEncoder.beginComputePass();
        for (const pipeline of plan.pipelines) {
          passEncoder.setPipeline(pipeline.pipeline);
          passEncoder.setBindGroup(0, pipeline.bindGroup);
          passEncoder.dispatchWorkgroups(
            pipeline.workgroupCount[0], pipeline.workgroupCount[1], pipeline.workgroupCount[2],
          );
        }
        passEncoder.end();
        for (const name of plan.scratchOutputs) {
          const rowTensor = this.host._rowTensorByName(rowNode.outputs, name);
          const fullTensor = this.host.graph.tensors.get(name) as ExecutorTensor;
          const fullBuffer = this.host.gpuBuffers.get(name);
          const scratch = plan.scratchByName.get(name)!;
          const { offset, size } = this.host._incrementalRowRange(
            rowTensor, fullTensor, rowPosition, `output '${name}'`,
          );
          if (offset % 4 === 0 && size % 4 === 0) {
            commandEncoder.copyBufferToBuffer(scratch, 0, fullBuffer!, offset, size);
          } else {
            this.host._encodeIncrementalRowByteCopy(
              commandEncoder, plan.byteCopyPipeline, plan.outputByteCopies.get(name),
              0, offset, size, `output '${name}'`,
            );
          }
        }
      }
    }

    _encodeIncrementalRows(selectedNodes: Iterable<number>, rowPosition: number): void {
      const commandEncoder = this.host.device.createCommandEncoder();
      this.host._encodeIncrementalRowsInto(commandEncoder, selectedNodes, rowPosition);
      this.host.device.queue.submit([commandEncoder.finish()]);
    }

    _deviceFeedbackDescriptor(
      inputs: WebGPUExecutionInputs,
      options: DeviceFeedbackDecodeOptions = {},
    ): DeviceFeedbackDescriptor {
      if (!options || typeof options !== 'object' || ArrayBuffer.isView(options)) {
        throw new Error('WebGPU device-feedback decode options must be an object.');
      }
      const requireName = (value: unknown, label: string): string => {
        if (typeof value !== 'string' || value.length === 0) {
          throw new Error(`WebGPU device-feedback decode requires ${label}.`);
        }
        return value;
      };
      const tokenInputName = requireName(options.tokenInput, "a non-empty 'tokenInput' name");
      const keepInputName = requireName(options.keepInput, "a non-empty 'keepInput' name");
      const outputName = requireName(options.output, "a non-empty 'output' name");
      if (new Set([tokenInputName, keepInputName, outputName]).size !== 3) {
        throw new Error('WebGPU device-feedback token, keep, and output tensors must be distinct.');
      }
      const tokenInput = this.host.graph?.tensors?.get(tokenInputName);
      const keepInput = this.host.graph?.tensors?.get(keepInputName);
      const output = this.host.graph?.tensors?.get(outputName);
      const fixedI32Input = (tensor: Tensor | undefined): tensor is ExecutorTensor =>
        tensor?.isInput === true && tensor.dtype === 'int32' &&
        tensor.shape?.length === 2 && tensor.shape[0] === 1 &&
        Number.isInteger(tensor.shape[1]) && tensor.shape[1] > 1;
      if (!fixedI32Input(tokenInput) || !fixedI32Input(keepInput) ||
          tokenInput.shape[1] !== keepInput.shape[1]) {
        throw new Error('WebGPU device-feedback tokenInput and keepInput must be distinct I32 graph inputs with matching [1,S] shapes and S > 1.');
      }
      const sequenceLength = tokenInput.shape[1];
      if (!output || output.dtype !== 'int32' || output.shape?.length !== 2 ||
          output.shape[0] !== 1 || output.shape[1] !== sequenceLength ||
          !this.host.graph.outputNames?.includes(outputName)) {
        throw new Error("WebGPU device-feedback output must be a declared I32 graph output with shape [1,S] matching its inputs.");
      }
      const startPosition = options.startPosition == null ? 0 : options.startPosition;
      if (typeof startPosition !== 'number' || !Number.isInteger(startPosition) ||
          startPosition < 0 || startPosition >= sequenceLength) {
        throw new Error(`WebGPU device-feedback startPosition must be an integer in [0,${sequenceLength - 1}].`);
      }
      if (options.endPosition != null && options.tokenCount != null &&
          options.endPosition !== options.tokenCount) {
        throw new Error('WebGPU device-feedback endPosition and tokenCount aliases must agree.');
      }
      const endPosition = options.endPosition ?? options.tokenCount ?? sequenceLength;
      if (typeof endPosition !== 'number' || !Number.isInteger(endPosition) ||
          endPosition <= startPosition ||
          endPosition > sequenceLength) {
        throw new Error(`WebGPU device-feedback endPosition must be an integer in (${startPosition},${sequenceLength}].`);
      }
      const rowsPerSubmission = options.rowsPerSubmission == null ? 1 : options.rowsPerSubmission;
      if (rowsPerSubmission !== 1) {
        throw new Error('WebGPU device-feedback currently requires rowsPerSubmission = 1 for portable causal-QSDPA correctness.');
      }
      if (startPosition === 0) {
        if (!inputs || typeof inputs !== 'object' || ArrayBuffer.isView(inputs)) {
          throw new Error('WebGPU device-feedback position zero requires a named seed input object.');
        }
        Tensor.assertCompatibleInput(
          tokenInput.dtype, inputs[tokenInputName], tokenInput.sizeBytes,
          `Device-feedback token input '${tokenInputName}'`,
        );
        Tensor.assertCompatibleInput(
          keepInput.dtype, inputs[keepInputName], keepInput.sizeBytes,
          `Device-feedback keep input '${keepInputName}'`,
        );
        if (inputs[keepInputName][0] !== 1) {
          throw new Error('WebGPU device-feedback keepInput row zero must be visible (I32 value 1).');
        }
      }

      let producerIndex = -1;
      let producer: ExecutorNode | null = null;
      for (let nodeIndex = 0; nodeIndex < this.host.graph.nodes.length; nodeIndex++) {
        const node = this.host.graph.nodes[nodeIndex];
        if (!Object.values(node.outputs || {}).some((tensor) => tensor === output)) continue;
        if (producer) {
          throw new Error(`WebGPU device-feedback output '${outputName}' has multiple producers.`);
        }
        producer = node;
        producerIndex = nodeIndex;
      }
      if (producer?.opType !== 'QArgMax' || producer.inputs?.input == null) {
        throw new Error(`WebGPU device-feedback output '${outputName}' must be produced by terminal QArgMax.`);
      }
      if (this.host.graph.nodes.some((node) =>
        Object.values(node.inputs || {}).some((tensor) => tensor === output))) {
        throw new Error(`WebGPU device-feedback QArgMax output '${outputName}' must be terminal.`);
      }
      let outputAxis: unknown = producer.params?.axis;
      if (typeof outputAxis !== 'number' || !Number.isInteger(outputAxis)) {
        throw new Error(`WebGPU device-feedback QArgMax '${producer.id}' requires an explicit integer axis.`);
      }
      if (outputAxis < 0) outputAxis += producer.inputs.input.shape.length;
      const vocabularySize = producer.inputs.input.shape?.[outputAxis];
      if (!Number.isInteger(vocabularySize) || vocabularySize <= 0) {
        throw new Error(`WebGPU device-feedback QArgMax '${producer.id}' has an invalid vocabulary axis.`);
      }

      const changedInputs = [tokenInputName, keepInputName];
      const selectedNodesValue = incrementalNodeSelection(
        this.host.graph as RuntimeGraph, inputs, { incremental: true, changedInputs }, true,
      );
      if (!(selectedNodesValue instanceof Set) || selectedNodesValue.size === 0 ||
          !selectedNodesValue.has(producerIndex)) {
        throw new Error('WebGPU device-feedback changed inputs do not reach the terminal QArgMax output.');
      }
      const selectedNodes = selectedNodesValue as Set<number>;
      const embeddings: ExecutorNode[] = [];
      let causalKeepConsumer = false;
      for (const nodeIndex of selectedNodes) {
        const node = this.host.graph.nodes[nodeIndex];
        if (node.opType === 'QEmbedding' && node.inputs?.input === tokenInput) embeddings.push(node);
        if (node.opType === 'QSDPA' && node.params?.causal === true &&
            node.inputs?.mask === keepInput) causalKeepConsumer = true;
        const nodeOutput = node.outputs?.out || Object.values(node.outputs || {})[0];
        if (nodeOutput?.shape?.[0] !== 1 || nodeOutput.shape?.[1] !== sequenceLength ||
            !this.host.incrementalRowCandidates.has(nodeIndex)) {
          throw new Error(`WebGPU device-feedback selected node ${String(node?.id ?? nodeIndex)} has no compatible B=1 W8A8 row plan.`);
        }
      }
      if (embeddings.length === 0) {
        throw new Error(`WebGPU device-feedback tokenInput '${tokenInputName}' must feed a selected QEmbedding.`);
      }
      if (embeddings.some((node) => node.inputs?.weight?.shape?.[0] !== vocabularySize)) {
        throw new Error('WebGPU device-feedback QEmbedding vocabulary must match the terminal QArgMax axis.');
      }
      if (!causalKeepConsumer) {
        throw new Error(`WebGPU device-feedback keepInput '${keepInputName}' must mask a selected causal QSDPA.`);
      }
      this.host._assertIncrementalRowInvariants(selectedNodes, changedInputs);
      return {
        tokenInputName,
        keepInputName,
        outputName,
        tokenInput,
        keepInput,
        output,
        startPosition,
        endPosition,
        rowsPerSubmission,
        sequenceLength,
        selectedNodes,
      };
    }

    _deviceFeedbackControl(sequenceLength: number): GPUBuffer {
      if (this.host.deviceFeedbackControlBuffer) {
        if (this.host.deviceFeedbackSequenceLength !== sequenceLength) {
          throw new Error('WebGPU device-feedback control buffer does not match the compiled sequence.');
        }
        return this.host.deviceFeedbackControlBuffer;
      }
      const values = Uint32Array.from({ length: sequenceLength }, (_, index) => index + 1);
      const resources = this.host._activeSpecializationResources || this.host.auxiliaryBuffers;
      const previousBufferCreateCount = this.host.specializationBufferCreateCount;
      const buffer = this.host._createSpecializationBuffer({
        label: 'DeviceFeedback_control',
        size: values.byteLength,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      try {
        this.host.device.queue.writeBuffer(buffer, 0, values);
      } catch (error) {
        resources.delete(buffer);
        buffer.destroy?.();
        this.host.specializationBufferCreateCount = previousBufferCreateCount;
        throw error;
      }
      this.host.deviceFeedbackControlBuffer = buffer;
      this.host.deviceFeedbackSequenceLength = sequenceLength;
      return buffer;
    }

    /**
     * Seed at position zero or resume the exact next B=1 W8A8 decode chunk while
     * feeding terminal QArgMax IDs and keep-mask ones forward on the device.
     * The returned GPUBuffer can be mapped once after the requested chunk.
     */

    async executeDeviceFeedbackDecode(
      inputs: WebGPUExecutionInputs,
      options: DeviceFeedbackDecodeOptions = {},
    ): Promise<GPUBuffer> {
      this.host.graph.assertTopologyRevision?.(this.host.compiledTopologyRevision as number, 'WebGPU');
      if ((this.host.graph.weightRevision || 0) !== this.host.compiledWeightRevision) {
        throw new Error('WebGPU weights changed after compilation; recompile before device-feedback decode.');
      }
      const descriptor = this.host._deviceFeedbackDescriptor(inputs, options);
      const resume = descriptor.startPosition > 0;
      if (resume) {
        const state = this.host.deviceFeedbackState;
        const matchingState = state &&
          state.tokenInputName === descriptor.tokenInputName &&
          state.keepInputName === descriptor.keepInputName &&
          state.outputName === descriptor.outputName &&
          state.sequenceLength === descriptor.sequenceLength &&
          state.nextPosition === descriptor.startPosition &&
          state.cacheGeneration === this.host.decodeCacheGeneration;
        if (!matchingState || this.host._webGPUIncrementalCacheValid !== true) {
          throw new Error('WebGPU device-feedback resume requires the next position from a valid device-feedback seed/cache.');
        }
      }
      await this.host._compileIncrementalRowPipelines(descriptor.selectedNodes);
      for (const nodeIndex of descriptor.selectedNodes) {
        if (!this.host.incrementalRowPlans.has(nodeIndex)) {
          const node = this.host.graph.nodes[nodeIndex];
          throw new Error(`WebGPU device-feedback node ${String(node?.id ?? nodeIndex)} has no compiled row pipeline.`);
        }
      }
      if (resume) {
        // A resumed direct call is a new public cache mutation. Advance the
        // ownership generation after validating that it continues our state.
        this.host._beginWebGPUDecodeExecution({});
      }
      const firstRowPosition = Math.max(1, descriptor.startPosition);
      const controlBuffer = descriptor.endPosition > firstRowPosition
        ? this.host._deviceFeedbackControl(descriptor.sequenceLength)
        : null;
      if (!resume) {
        await this.host.execute(inputs, {
          incremental: true,
          incrementalReset: true,
          changedInputs: Object.keys(inputs),
        });
      }
      const outputBuffer = this.host.gpuBuffers.get(descriptor.outputName);

      const tokenBuffer = this.host.gpuBuffers.get(descriptor.tokenInputName);
      const keepBuffer = this.host.gpuBuffers.get(descriptor.keepInputName);
      if (!outputBuffer || !tokenBuffer || !keepBuffer ||
          new Set([outputBuffer, tokenBuffer, keepBuffer]).size !== 3) {
        this.host._webGPUIncrementalCacheValid = false;
        this.host.deviceFeedbackState = null;
        throw new Error('WebGPU device-feedback tensors require three distinct allocated device buffers.');
      }
      if (descriptor.endPosition === firstRowPosition) {
        this.host.deviceFeedbackState = Object.freeze({
          tokenInputName: descriptor.tokenInputName,
          keepInputName: descriptor.keepInputName,
          outputName: descriptor.outputName,
          sequenceLength: descriptor.sequenceLength,
          nextPosition: descriptor.endPosition,
          cacheGeneration: this.host.decodeCacheGeneration,
        });
        return outputBuffer;
      }
      const wordBytes = Int32Array.BYTES_PER_ELEMENT;
      try {
        for (let position = firstRowPosition; position < descriptor.endPosition; position++) {
          const commandEncoder = this.host.device.createCommandEncoder();
          commandEncoder.copyBufferToBuffer(
            outputBuffer, (position - 1) * wordBytes,
            tokenBuffer, position * wordBytes,
            wordBytes,
          );
          commandEncoder.copyBufferToBuffer(
            controlBuffer!, 0,
            keepBuffer, position * wordBytes,
            wordBytes,
          );
          this.host._encodeIncrementalRowsInto(
            commandEncoder, descriptor.selectedNodes, position,
          );
          // Submit every row in queue order. QSDPA prefix uniforms are updated
          // with queue.writeBuffer while the row is encoded; submitting before
          // encoding the next row guarantees each dispatch observes its own
          // prefix without an intermediate map or JavaScript await.
          this.host.device.queue.submit([commandEncoder.finish()]);
        }
      } catch (error) {
        this.host._webGPUIncrementalCacheValid = false;
        this.host.deviceFeedbackState = null;
        throw error;
      }
      this.host._webGPUIncrementalCacheValid = true;
      this.host.deviceFeedbackState = Object.freeze({
        tokenInputName: descriptor.tokenInputName,
        keepInputName: descriptor.keepInputName,
        outputName: descriptor.outputName,
        sequenceLength: descriptor.sequenceLength,
        nextPosition: descriptor.endPosition,
        cacheGeneration: this.host.decodeCacheGeneration,
      });
      return outputBuffer;
    }
    /**
     * Execute the compiled graph on the GPU.
     * @param {Object} inputs - Key-value pair of input tensor names to Float32Array/Int32Array
     */
}
