import { Tensor } from '../core/Tensor.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { RuntimeTypedArray } from '../types.js';
import { incrementalNodeSelection } from './incrementalExecution.js';
import { keepMaskReader, quantizedRowNode } from './quantizedRowExecution.js';
import {
  defaultScratchAllocator, kvPageCopyRanges, pagedSlot,
} from './kvPageAddressing.js';
import {
  decodeRowSet, rowSpanCopyRuns, rowSpanPaddingRuns, rowSpanSourceRows,
  singleLaneRowSet, stageKeepMask, writeRowIndices,
} from './decodeRowSet.js';
import type { DecodeRowSet } from './decodeRowSet.js';
import type { KVPagePlan } from './kvPageAddressing.js';
import type { GraphExecutor } from './GraphExecutor.js';
import type {
  DeviceFeedbackDecodeOptions,
  DeviceFeedbackDescriptor,
  DeviceFeedbackState,
  ExecutorNode,
  ExecutorTensor,
  IncrementalRowByteCopy,
  IncrementalRowCandidate,
  IncrementalRowKeepMask,
  IncrementalRowPlan,
  WebGPUExecutionInputs,
  WebGPURowStepOptions,
} from './WebGPUContracts.js';

/** One staged row's worth of device-to-device copy, in bytes. */
interface RowCopy {
  readonly full: number;
  readonly staged: number;
  readonly size: number;
}

const INT32_SAMPLE = new Int32Array(1);

/** Context-local incremental and device-feedback decode owner. */
export class WebGPUDecodeState {
  readonly rowCandidates = new Map<number, IncrementalRowCandidate>();
  readonly rowPlans = new Map<number, IncrementalRowPlan>();
  readonly rowCopyTensorNames = new Set<string>();
  /** Lane count the published candidates were analysed for. */
  rowCandidateLanes = 1;
  controlBuffer: GPUBuffer | null = null;
  sequenceLength = 0;
  feedback: Readonly<DeviceFeedbackState> | null = null;
  /**
   * Host values of the graph inputs a batched keep mask has to read.
   *
   * The mask's *values* live on the device, and the builder that folds each
   * lane's length into them runs on the host. The host wrote those values in
   * the first place, so mirroring the upload is what makes the shared builder
   * usable here instead of a WebGPU-only reimplementation of it. Kept per
   * context, like every other decode-owned resource.
   */
  readonly rowInputMirrors = new Map<string, RuntimeTypedArray>();
  /** Names the executor exposes so uploads know which inputs to mirror. */
  readonly rowMirrorNames = new Set<string>();
  /** Host staging for keep masks, sized geometrically like every other gather. */
  readonly rowHostScratch = defaultScratchAllocator();

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
    this.rowInputMirrors.clear();
    this.rowMirrorNames.clear();
    this.rowCandidateLanes = 1;
    this.controlBuffer = null;
    this.sequenceLength = 0;
    this.feedback = null;
  }

  /**
   * Retain a host copy of an input a batched keep mask reads.
   *
   * Copied rather than retained by reference: the caller owns the array it
   * passed and may reuse it for the next step, and a mask built from a mutated
   * array would publish the *next* step's visibility into this one.
   */
  mirrorRowInput(name: string, data: RuntimeTypedArray): void {
    if (!this.rowMirrorNames.has(name)) return;
    const existing = this.rowInputMirrors.get(name);
    if (existing && existing.length === data.length &&
        existing.constructor === data.constructor) {
      (existing as Int32Array).set(data as any);
      return;
    }
    this.rowInputMirrors.set(
      name, new (data.constructor as any)(data) as RuntimeTypedArray);
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

    /*
     * Build the sample row node a pipeline is compiled from.
     *
     * `rowSet` is a *sample*: it fixes the lane count and nothing else. Every
     * address the step actually uses is resolved again at encode time from the
     * step's own row set, so the sample's positions never reach a dispatch. It
     * exists because a pipeline needs concrete operand shapes to compile
     * against, not because the plan depends on where the sample happened to be.
     */
    _prepareIncrementalRowNode(node: ExecutorNode, rowSet: DecodeRowSet): ExecutorNode {
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
      const prepared = quantizedRowNode(storageNode, rowSet, {
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
        /* The span declares its own addressing. Deriving it here instead --
         * from where the sample's bytes happened to land -- is what tied this
         * path to "one row is a contiguous view of the whole tensor", which a
         * staged batch is not: `allocate()` hands it a separate buffer. */
        const geometry = (rowTensor as any)?._rowSpan;
        if (!geometry) {
          throw new Error(
            `WebGPU incremental row node ${node.id} ${role} '${name}' carries no row geometry.`);
        }
        const rowBytes = geometry.width * Tensor.dtypeBytes(rowTensor.dtype);
        if (!Number.isSafeInteger(rowBytes) || rowBytes <= 0) {
          throw new Error(
            `WebGPU incremental row node ${node.id} ${role} '${name}' has no uniform row width.`);
        }
        const normalized = Object.assign(Object.create(Object.getPrototypeOf(rowTensor)), rowTensor, {
          buffer: this.host._rowTypedStorage(rowTensor, rowTensor.sizeBytes),
        });
        normalized._webgpuRow = Object.freeze({
          geometry,
          rowBytes,
          lanes: rowSet.lanes,
          /* A key-only mask is a `window`: same address every step, visible
           * length published through `seq_kv`. A per-query one is a `row`, and
           * a causal step sees only its own prefix of that row. */
          queryMaskPrefix: (node.opType === 'QSDPA' || node.opType === 'CrossSDPA') &&
            role === 'input' && name === 'mask' && node.params?.causal === true &&
            geometry.kind === 'row',
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

    /*
     * The capacity a staged operand needs across *every* step, not this one.
     *
     * Sizing from the sample would be a correctness bug rather than a waste: a
     * `prefix` operand grows with the longest lane's active length and its keep
     * mask grows with it, so a buffer sized for the sample's positions would
     * overflow on the first longer step. Each bound is read from the declared
     * shape, which is the only thing that does not move.
     */
    _rowScratchCapacity(
      rowTensor: ExecutorTensor,
      originalTensor: ExecutorTensor | undefined,
      lanes: number,
      keyExtent: number,
    ): number {
      const descriptor = rowTensor._webgpuRow!;
      if (descriptor.geometry.kind === 'row') return lanes * descriptor.rowBytes;
      /* A staged prefix is `lanes * keyCapacity` rows and `keyCapacity` never
       * exceeds the retained sequence, so the pool's own size is a bound -- and
       * it is the bound `_compilePagedRowVariants` already uses. */
      return Math.max(
        lanes * keyExtent * descriptor.rowBytes, originalTensor?.sizeBytes ?? 0);
    }

    /** The retained key extent of an attention node: the bound every prefix shares. */
    _rowKeyExtent(node: ExecutorNode): number {
      const shape = node.inputs?.k?.shape;
      const extent = Array.isArray(shape) && shape.length >= 2
        ? shape[shape.length - 2] : 0;
      return Number.isSafeInteger(extent) && extent > 0 ? extent : 0;
    }

    _incrementalRowCandidate(
      node: ExecutorNode,
      nodeIndex: number,
      lanes = 1,
    ): IncrementalRowCandidate | null {
      // Float CrossSDPA row execution currently has no dynamic visible-prefix
      // uniform in the WebGPU pipeline contract. CPU/WASM execute the cloned
      // prefix directly; advertising the static position-one shader here would
      // silently reuse K/V length two at later positions.
      if (node?.opType === 'CrossSDPA') return null;
      const output = node?.outputs?.out || Object.values(node?.outputs || {})[0];
      if (output?.shape?.[0] !== lanes || !Number.isInteger(output.shape?.[1]) ||
          output.shape[1] <= 1) return null;
      const sequence = output.shape[1] as number;
      let rowNode;
      try {
        /* Position one for every lane. The shapes a lane presents do not depend
         * on where it is, so any legal position compiles the same pipeline and
         * this is the cheapest one in bounds for a sequence of two or more.
         * Unlike the stride this function used to derive, nothing downstream
         * reads it. */
        rowNode = this.host._prepareIncrementalRowNode(
          node, decodeRowSet(Array.from({ length: lanes }, () => ({ position: 1 }))));
      } catch {
        return null;
      }
      const keyExtent = this._rowKeyExtent(node) || sequence;

      const scratchInputs = new Set<string>();
      const scratchOutputs = new Set<string>();
      const scratchCapacities = new Map<string, number>();
      const byteCopyInputs = new Set<string>();
      const byteCopyOutputs = new Set<string>();
      const invariantInputs = new Set<string>();
      let keepMask: IncrementalRowKeepMask | null = null;
      const addScratch = (
        rowTensor: ExecutorTensor | undefined,
        originalTensor: ExecutorTensor | undefined,
        target: Set<string>,
        byteCopyTarget: Set<string>,
        capacity: number | undefined = undefined,
      ): boolean => {
        const descriptor = rowTensor?._webgpuRow;
        if (!rowTensor || !descriptor || rowTensor === originalTensor ||
            !Number.isSafeInteger(rowTensor.sizeBytes) || rowTensor.sizeBytes <= 0) return false;
        const bytes = capacity ?? this._rowScratchCapacity(
          rowTensor, originalTensor, lanes, keyExtent);
        if (!Number.isSafeInteger(bytes) || bytes <= 0) return false;
        /* A `prefix` is staged as runs of whole tokens whose destinations move
         * with the padded extent, and a run boundary is not a lane boundary --
         * so the packed-byte fallback, which describes exactly one range, cannot
         * express it. Naming the constraint beats surfacing an alignment error
         * from the driver at the first scattered lane. */
        if (descriptor.rowBytes % 4 !== 0 && descriptor.geometry.kind !== 'row') return false;
        target.add(rowTensor.name);
        if (descriptor.rowBytes % 4 !== 0) byteCopyTarget.add(rowTensor.name);
        scratchCapacities.set(
          rowTensor.name,
          Math.max(scratchCapacities.get(rowTensor.name) || 0, bytes),
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
        const kind = rowTensor._webgpuRow?.geometry?.kind;
        /* Same address every step: bind the operand whole and let `seq_kv` say
         * how much of it is visible. This is the one-lane causal K/V and the
         * key-only keep mask, and it is why B=1 decode stages nothing it did
         * not stage before this file learned about lanes. */
        if (kind === 'window') continue;
        if (kind === 'keepMask') {
          if (!addScratch(
            rowTensor, originalTensor, scratchInputs, byteCopyInputs,
            lanes * keyExtent * Int32Array.BYTES_PER_ELEMENT)) return null;
          keepMask = Object.freeze({
            name: rowTensor.name,
            sourceName: originalTensor?.isInput === true ? originalTensor.name : null,
            sourceTensor: originalTensor ?? null,
            queryLength: sequence,
            keyLength: keyExtent,
            causal: node.params?.causal === true,
          });
          continue;
        }
        if (!addScratch(rowTensor, originalTensor, scratchInputs, byteCopyInputs)) return null;
      }
      /* A graph mask with no host-visible source is the one case the batched
       * builder cannot serve: it folds each lane's length into the mask's
       * *values*, and the values of a computed activation live only on the
       * device. Refuse at compile time, where it is a capability answer, rather
       * than at the step, where it is a stall. */
      if (keepMask?.sourceTensor && !keepMask.sourceName) return null;
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
        lanes,
        scratchInputs,
        scratchOutputs,
        scratchCapacities,
        byteCopyInputs,
        byteCopyOutputs,
        invariantInputs,
        keepMask,
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

    /**
     * Re-analyse row candidates for a declared lane count.
     *
     * The lane count is *declared*, never inferred from a leading extent that
     * happens to match: a graph whose activations are spelled `[S,1,D]` has a
     * leading extent of S, and inferring from it would compile an S-lane
     * pipeline for a one-lane context. So compilation analyses one lane and a
     * step that declares more asks for its own analysis here, once.
     *
     * Every compiled plan is dropped, because the bind group, the scratch
     * capacities and the workgroup count all describe a fixed lane count -- a
     * plan reused across a change would dispatch against the wrong geometry.
     */
    _ensureIncrementalRowLanes(lanes: number): void {
      if (!Number.isSafeInteger(lanes) || lanes < 1) {
        throw new Error(`WebGPU incremental row lane count must be a positive integer.`);
      }
      if (this.rowCandidateLanes === lanes) return;
      this._publishIncrementalRows(this._collectIncrementalRows(lanes));
      this.rowCandidateLanes = lanes;
    }

    /**
     * Lane counts a context built on this graph could declare.
     *
     * Used only to decide which tensors need `COPY_SRC|COPY_DST`, never to
     * decide what a step executes -- which is why guessing wide is safe here
     * and would not be anywhere else. A leading extent that is a sequence
     * rather than a batch adds a lane count no context will ever declare; the
     * candidates it produces are discarded and the only trace is a usage flag
     * on a buffer that did not need one.
     *
     * Getting this wrong in the other direction is not benign. Buffers are
     * allocated once, at compile time, from the names some candidate stages. A
     * two-lane graph has no one-lane candidates at all, so analysing only one
     * lane leaves that set empty -- and the first batched step then asks the
     * driver to copy out of a buffer with no `COPY_SRC`, which is a validation
     * failure that drops the whole command buffer. Every row keeps its seed
     * value and the step looks like a decode bug.
     */
    _incrementalRowLaneCandidates(): number[] {
      const lanes = new Set<number>([1]);
      for (const node of this.host.graph.nodes) {
        const output = node?.outputs?.out || Object.values(node?.outputs || {})[0];
        const extent = output?.shape?.[0];
        if (Number.isSafeInteger(extent) && (extent as number) > 1) {
          lanes.add(extent as number);
        }
      }
      return [...lanes];
    }

    _collectIncrementalRows(lanes = 1): {
      candidates: Map<number, IncrementalRowCandidate>;
      copyTensorNames: Set<string>;
      mirrorNames: Set<string>;
    } {
      const candidates = new Map<number, IncrementalRowCandidate>();
      const copyTensorNames = new Set<string>();
      const mirrorNames = new Set<string>();
      if (!(this.host.graph?.tensors instanceof Map)) {
        return { candidates, copyTensorNames, mirrorNames };
      }
      /* Names for every lane count, candidates for the one asked about. */
      for (const laneCount of this._incrementalRowLaneCandidates()) {
        const publish = laneCount === lanes;
        for (let nodeIndex = 0; nodeIndex < this.host.graph.nodes.length; nodeIndex++) {
          const candidate = this.host._incrementalRowCandidate(
            this.host.graph.nodes[nodeIndex], nodeIndex, laneCount);
          if (!candidate) continue;
          for (const name of candidate.scratchCapacities.keys()) {
            copyTensorNames.add(name);
          }
          if (!publish) continue;
          candidates.set(nodeIndex, candidate);
          if (candidate.keepMask?.sourceName) mirrorNames.add(candidate.keepMask.sourceName);
        }
      }
      return { candidates, copyTensorNames, mirrorNames };
    }

    _publishIncrementalRows(analysis: {
      candidates: ReadonlyMap<number, IncrementalRowCandidate>;
      copyTensorNames: ReadonlySet<string>;
      mirrorNames?: ReadonlySet<string>;
    }): void {
      this.host.incrementalRowPlans.clear();
      this.host.incrementalRowCandidates.clear();
      this.host.incrementalRowCopyTensorNames.clear();
      this.rowMirrorNames.clear();
      this.rowInputMirrors.clear();
      for (const [nodeIndex, candidate] of analysis.candidates) {
        this.host.incrementalRowCandidates.set(nodeIndex, candidate);
      }
      for (const name of analysis.copyTensorNames) {
        this.host.incrementalRowCopyTensorNames.add(name);
      }
      for (const name of analysis.mirrorNames || []) {
        this.rowMirrorNames.add(name);
      }
    }

    _analyzeIncrementalRows(): void {
      this._publishIncrementalRows(this._collectIncrementalRows());
    }

    /*
     * The byte copies one operand needs this step, and the staged gaps no copy
     * fills.
     *
     * This replaced a function that returned a single `{offset, size}` computed
     * as `slot * stride`. Both halves of that were assumptions rather than
     * facts: that a step touches one span, and that its address is linear in
     * the position. A batched step touches one span *per lane* and a paged one
     * touches page-table slots, so the rows come from the shared row-set
     * resolver and the runs come from merging them -- which makes B=1 the case
     * where the merge yields exactly one copy, not a preserved special case.
     */
    _incrementalRowCopies(
      rowTensor: ExecutorTensor | null,
      fullTensor: ExecutorTensor,
      rowSet: DecodeRowSet,
      label: string,
      { skipParked = false }: { skipParked?: boolean } = {},
    ): { copies: RowCopy[]; padding: RowCopy[]; stagedBytes: number } {
      const descriptor = rowTensor?._webgpuRow;
      if (!descriptor) {
        throw new Error(`WebGPU incremental row ${label} is missing its row-storage descriptor.`);
      }
      /* Whether a tensor is paged is a property of the *step*, not of the
       * compiled plan: the staging buffers are demand-driven, so a plan is
       * built before any page table exists and the same plan serves a
       * contiguous context and a paged one. Width and lane stride come from the
       * shape and never move; this one is republished each step. */
      const paged = rowSet.pagedTensors.has(fullTensor.name);
      const geometry = paged === descriptor.geometry.paged
        ? descriptor.geometry
        : { ...descriptor.geometry, paged };
      const rows = rowSpanSourceRows(rowSet, geometry);
      if (!rows) {
        throw new Error(`WebGPU incremental row ${label} has no addressable source rows.`);
      }
      if (skipParked) {
        /* A parked lane occupies a dense row so the operand keeps its shape and
         * the kernel computes it; nothing writes it back. Blanking the row here
         * also stops the merge from joining a live lane to a dead one. */
        for (let lane = 0; lane < rowSet.lanes; lane++) {
          if (rowSet.parked[lane]) rows[lane] = -1;
        }
      }
      const { rowBytes } = descriptor;
      /* A causal per-query mask row is visible only up to the lane's own
       * length, and every lane shares the padded extent, so the copied prefix
       * is the padded one. The lanes' own lengths reach the kernel through the
       * mask's values, not through its size. */
      const size = descriptor.queryMaskPrefix
        ? rowSet.keyCapacity * Int32Array.BYTES_PER_ELEMENT
        : rowBytes;
      const stride = descriptor.queryMaskPrefix ? size : rowBytes;
      const copies: RowCopy[] = [];
      for (const run of rowSpanCopyRuns(rows)) {
        const full = run.source * rowBytes;
        const staged = run.destination * stride;
        const bytes = run.rows === 1 ? size : run.rows * rowBytes;
        if (!Number.isSafeInteger(full) || full < 0 ||
            !Number.isSafeInteger(bytes) || bytes <= 0 ||
            full + bytes > fullTensor.sizeBytes) {
          throw new Error(`WebGPU incremental row ${label} requires in-bounds storage.`);
        }
        copies.push({ full, staged, size: bytes });
      }
      const padding = rowSpanPaddingRuns(rows).map((run) => ({
        full: -1,
        staged: run.destination * stride,
        size: run.rows * stride,
      }));
      return { copies, padding, stagedBytes: rows.length * stride };
    }

    _encodeIncrementalRowByteCopy(
      commandEncoder: GPUCommandEncoder,
      pipeline: GPUComputePipeline | null,
      copy: IncrementalRowByteCopy | undefined,
      index: number,
      sourceOffset: number,
      destinationOffset: number,
      size: number,
      label: string,
    ): void {
      const paramsBuffer = copy?.paramsBuffers?.[index];
      const bindGroup = copy?.bindGroups?.[index];
      if (!pipeline || !paramsBuffer || !bindGroup) {
        throw new Error(`WebGPU incremental row ${label} has no packed-byte copy pipeline.`);
      }
      const destinationWordCount = Math.ceil(((destinationOffset % 4) + size) / 4);
      if (![sourceOffset, destinationOffset, size, destinationWordCount].every((value) =>
        Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff) ||
          size === 0 || destinationWordCount === 0 ||
          sourceOffset + size > 0x100000000 || destinationOffset + size > 0x100000000) {
        throw new Error(`WebGPU incremental row ${label} has an invalid packed-byte range.`);
      }
      this.host.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([
        sourceOffset, destinationOffset, size, destinationWordCount,
      ]));
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(destinationWordCount / 64), 1, 1);
      passEncoder.end();
    }

    /**
     * Move one operand's rows between the pool and its staging buffer.
     *
     * Direction decides which side is the source; everything else is the same
     * arithmetic, which is what keeps a gather and its scatter from disagreeing
     * about where a row lives.
     */
    _encodeRowStaging(
      commandEncoder: GPUCommandEncoder,
      plan: IncrementalRowPlan,
      name: string,
      rowNode: ExecutorNode,
      rowSet: DecodeRowSet,
      direction: 'read' | 'write',
    ): void {
      const role = direction === 'read' ? 'input' : 'output';
      const rowTensor = this.host._rowTensorByName(
        direction === 'read' ? rowNode.inputs : rowNode.outputs, name);
      const fullTensor = this.host.graph.tensors.get(name) as ExecutorTensor;
      const fullBuffer = this.host.gpuBuffers.get(name);
      const scratch = plan.scratchByName.get(name)!;
      const label = `${role} '${name}'`;
      const { copies, padding, stagedBytes } = this._incrementalRowCopies(
        rowTensor, fullTensor, rowSet, label, { skipParked: direction === 'write' });
      if (stagedBytes > plan.scratchCapacities.get(name)!) {
        throw new Error(
          `WebGPU incremental row ${label} exceeds its compiled scratch capacity.`);
      }
      const byteCopies = direction === 'read' ? plan.inputByteCopies : plan.outputByteCopies;
      let index = 0;
      for (const copy of copies) {
        const source = direction === 'read' ? copy.full : copy.staged;
        const destination = direction === 'read' ? copy.staged : copy.full;
        if (source % 4 === 0 && destination % 4 === 0 && copy.size % 4 === 0) {
          commandEncoder.copyBufferToBuffer(
            direction === 'read' ? fullBuffer! : scratch, source,
            direction === 'read' ? scratch : fullBuffer!, destination, copy.size);
        } else {
          this.host._encodeIncrementalRowByteCopy(
            commandEncoder, plan.byteCopyPipeline, byteCopies.get(name),
            index, source, destination, copy.size, label);
        }
        index++;
      }
      /* Padding a lane never wrote. The keep mask already stops the kernel from
       * reading it, so this is the second of two defences -- but padding still
       * holding another request's bytes is what turns a masking mistake from
       * wrong arithmetic into a cross-request leak. Reads only: a scatter's
       * gaps are rows the pool legitimately owns. */
      if (direction === 'read' && padding.length > 0) {
        if (typeof commandEncoder.clearBuffer !== 'function') {
          throw new Error(
            `WebGPU incremental row ${label} needs clearBuffer to zero staged padding.`);
        }
        for (const gap of padding) {
          commandEncoder.clearBuffer(scratch, gap.staged, gap.size);
        }
      }
    }

    _rowTensorByName(
      tensors: Record<string, ExecutorTensor>,
      name: string,
    ): ExecutorTensor | null {
      return Object.values(tensors).find((tensor) => tensor.name === name) || null;
    }

    /**
     * Bytes one token of an attention operand occupies.
     *
     * The page pool and the staging buffer are both token-slot arrays of this
     * stride, so it is the only conversion between the allocator's slot
     * indices and a GPU buffer offset.
     */
    _attentionRowBytes(tensor: ExecutorTensor | undefined, label: string): number {
      const shape = tensor?.shape;
      const sequence = Array.isArray(shape)
        ? (shape.length === 3 && shape[0] === 1 ? shape[1]
          : shape.length === 2 ? shape[0] : 0)
        : 0;
      const bytes = Number.isSafeInteger(sequence) && sequence > 0
        ? (tensor!.sizeBytes / sequence) : 0;
      if (!Number.isSafeInteger(bytes) || bytes <= 0) {
        throw new Error(`WebGPU paged K/V operand '${label}' has no uniform token stride.`);
      }
      /* copyBufferToBuffer requires every offset and size to be a multiple of
       * four, and each is a multiple of this stride. Refusing here names the
       * real constraint instead of surfacing a validation error from the
       * driver at the first scattered lane. */
      if (bytes % 4 !== 0) {
        throw new Error(
          `WebGPU paged K/V operand '${label}' has a ${bytes}-byte token stride; ` +
          'buffer-to-buffer staging requires a multiple of four.');
      }
      return bytes;
    }

    /** Whether this plan's step reads its K/V through the staging binding. */
    _pagedAttentionBinding(
      plan: IncrementalRowPlan,
      kvPages: KVPagePlan | null,
    ): boolean {
      if (!kvPages || !plan.pagedPipelines || !plan.pagedStaging) return false;
      const keyName = plan.node.inputs?.k?.name;
      return typeof keyName === 'string' && kvPages.pagedTensors.has(keyName);
    }

    /**
     * Copy the lane's active prefix out of the page pool into logical order.
     *
     * One copy per contiguous page run, so the cost tracks how fragmented the
     * lane actually is rather than how fragmented it could be.
     */
    _encodePagedKVStaging(
      commandEncoder: GPUCommandEncoder,
      plan: IncrementalRowPlan,
      kvPages: KVPagePlan,
      nodeIndex: number,
    ): void {
      const ranges = kvPageCopyRanges(kvPages);
      for (const port of ['k', 'v'] as const) {
        const name = plan.node.inputs?.[port]?.name;
        const pool = name ? this.host.gpuBuffers.get(name) : undefined;
        const staging = name ? plan.pagedStaging!.get(name) : undefined;
        if (!name || !pool || !staging) {
          throw new Error(
            `WebGPU paged row node ${String(plan.node?.id ?? nodeIndex)} has no staging for '${port}'.`);
        }
        const rowBytes = this.host._attentionRowBytes(
          this.host.graph.tensors.get(name) as ExecutorTensor, name);
        for (const range of ranges) {
          const size = range.tokens * rowBytes;
          const source = range.source * rowBytes;
          const destination = range.destination * rowBytes;
          if (source + size > pool.size || destination + size > staging.size) {
            throw new Error(
              `WebGPU paged K/V run for '${name}' is outside its pool or staging buffer.`);
          }
          commandEncoder.copyBufferToBuffer(pool, source, staging, destination, size);
        }
      }
    }

    /**
     * Build this step's `[lanes, keyCapacity]` keep mask and upload it.
     *
     * The one operand of a batched step that is uploaded rather than copied,
     * because it is not a region of any tensor: the current dense-row contract
     * publishes each lane's active length *through* the mask, so its contents
     * depend on the step's `kvLengths` and exist nowhere until they are
     * computed. Built by the same `stageKeepMask` CPU and WASM use — a second
     * implementation here would be exactly the divergence the shared corpus
     * exists to catch.
     */
    _uploadIncrementalKeepMask(
      plan: IncrementalRowPlan,
      rowSet: DecodeRowSet,
      nodeIndex: number,
    ): void {
      const spec = plan.keepMask!;
      const buffer = plan.scratchByName.get(spec.name);
      if (!buffer) {
        throw new Error(
          `WebGPU batched row node ${String(plan.node?.id ?? nodeIndex)} has no keep-mask buffer.`);
      }
      const mirror = spec.sourceName ? this.rowInputMirrors.get(spec.sourceName) : null;
      if (spec.sourceName && !mirror) {
        throw new Error(
          `WebGPU batched row node ${String(plan.node?.id ?? nodeIndex)} needs host values for ` +
          `mask input '${spec.sourceName}'; supply it with the step.`);
      }
      const source = mirror
        ? { ...(spec.sourceTensor as object), buffer: mirror }
        : null;
      const read = source
        ? keepMaskReader(
            source, { queryLength: spec.queryLength, keyLength: spec.keyLength },
            rowSet, plan.node)
        : null;
      const staged = stageKeepMask(
        rowSet.lanes, rowSet.keyCapacity, spec.causal ? rowSet.kvLengths : null,
        read, `${nodeIndex}:${spec.name}`, this.rowHostScratch, INT32_SAMPLE);
      staged.apply();
      const bytes = rowSet.lanes * rowSet.keyCapacity * Int32Array.BYTES_PER_ELEMENT;
      if (bytes > (plan.scratchCapacities.get(spec.name) ?? 0)) {
        throw new Error(
          `WebGPU batched row keep mask '${spec.name}' exceeds its compiled capacity.`);
      }
      this.host.device.queue.writeBuffer(
        buffer, 0, staged.storage.buffer, staged.storage.byteOffset, bytes);
    }

    _encodeIncrementalRowsInto(
      commandEncoder: GPUCommandEncoder,
      selectedNodes: Iterable<number>,
      rowSet: DecodeRowSet,
      { qsdpaControlBuffer = null }: WebGPURowStepOptions & {
        qsdpaControlBuffer?: GPUBuffer | null;
      } = {},
    ): void {
      const lanes = rowSet.lanes;
      /* The one-lane page plan drives the second *binding*; a batched step
       * carries one plan per lane and reads them through the row set, which is
       * why paging needs no separate binding once rows are staged anyway. */
      const kvPages = lanes === 1 ? rowSet.pages[0] : null;
      for (const nodeIndex of selectedNodes) {
        const plan = this.host.incrementalRowPlans.get(nodeIndex)!;
        const rowNode = plan.sampleNode;
        if (plan.lanes !== lanes) {
          throw new Error(
            `WebGPU incremental row node ${String(plan.node?.id ?? nodeIndex)} is compiled for ` +
            `${plan.lanes} lane(s) but the step declares ${lanes}.`);
        }
        /*
         * A row pipeline's bind group is compiled once and binds each attention
         * operand whole, from buffer offset zero. That expresses exactly one
         * mapping -- the lane's prefix is one page run based at physical slot
         * zero -- which is a property no lane but the first has.
         *
         * So a *one-lane* paged step uses the plan's second binding, whose K/V
         * operands point at staging buffers, and copies the lane's active prefix
         * into logical order first. One `copyBufferToBuffer` per contiguous page
         * run: an identity mapping is one copy, a fully scattered lane is one
         * per page, and everything in between is in between. A batched step
         * needs no second binding at all -- its K/V operands are staged spans in
         * the first one, so a page table is just a set of source rows. The
         * shader is untouched either way, so the 80-byte attention ABI five
         * backends share does not move.
         */
        const paged = this._pagedAttentionBinding(plan, kvPages);
        if (paged) {
          this._encodePagedKVStaging(commandEncoder, plan, kvPages!, nodeIndex);
        }
        const qsdpaParamsBuffer = paged
          ? plan.pagedQsdpaParamsBuffer
          : plan.qsdpaParamsBuffer;
        if (qsdpaParamsBuffer) {
          /* The padded key extent every lane shares, which for one lane is that
           * lane's own visible length -- so this is one expression, not a
           * batched case beside a scalar one. Lanes shorter than the maximum
           * are truncated by the keep mask, never by this number. */
          const seqKV = plan.node.params?.causal === true
            ? rowSet.keyCapacity
            : plan.node.inputs?.k?.shape?.at(-2);
          if (typeof seqKV !== 'number' || !Number.isInteger(seqKV) ||
              seqKV <= 0 || seqKV > 0xffffffff) {
            throw new Error(`WebGPU QSDPA incremental row node ${rowNode.id} has an invalid visible K/V prefix.`);
          }
          if (qsdpaControlBuffer) {
            commandEncoder.copyBufferToBuffer(
              qsdpaControlBuffer,
              rowSet.positions[0] * Uint32Array.BYTES_PER_ELEMENT,
              qsdpaParamsBuffer,
              Uint32Array.BYTES_PER_ELEMENT,
              Uint32Array.BYTES_PER_ELEMENT,
            );
          } else {
            this.host.device.queue.writeBuffer(
              qsdpaParamsBuffer, Uint32Array.BYTES_PER_ELEMENT, new Uint32Array([seqKV]),
            );
          }
        }
        if (plan.keepMask) this._uploadIncrementalKeepMask(plan, rowSet, nodeIndex);
        for (const name of plan.scratchInputs) {
          if (plan.keepMask?.name === name) continue;
          this._encodeRowStaging(commandEncoder, plan, name, rowNode, rowSet, 'read');
        }
        const passEncoder = commandEncoder.beginComputePass();
        for (const pipeline of (paged ? plan.pagedPipelines! : plan.pipelines)) {
          passEncoder.setPipeline(pipeline.pipeline);
          passEncoder.setBindGroup(0, pipeline.bindGroup);
          passEncoder.dispatchWorkgroups(
            pipeline.workgroupCount[0], pipeline.workgroupCount[1], pipeline.workgroupCount[2],
          );
        }
        passEncoder.end();
        for (const name of plan.scratchOutputs) {
          this._encodeRowStaging(commandEncoder, plan, name, rowNode, rowSet, 'write');
        }
      }
    }

    _encodeIncrementalRows(
      selectedNodes: Iterable<number>,
      rowSet: DecodeRowSet,
    ): void {
      const commandEncoder = this.host.device.createCommandEncoder();
      this.host._encodeIncrementalRowsInto(commandEncoder, selectedNodes, rowSet);
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
            commandEncoder, descriptor.selectedNodes, singleLaneRowSet(position),
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
