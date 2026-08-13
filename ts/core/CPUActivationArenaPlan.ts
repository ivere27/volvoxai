import type { RuntimeDType } from '../types.js';
import { runtimeDTypeBytes } from '../ops/shapeSystem.js';
import type {
  BoundActivationLivenessLayout,
  BoundActivationLivenessRegion,
} from './BoundExecutionGraph.js';
import type { Graph } from './Graph.js';

export const CPU_ACTIVATION_LIVENESS_PROTOCOL =
  'logical-topology-liveness/v1' as const;

export interface CPUActivationArenaTensor {
  readonly name: string;
  readonly kind: 'input' | 'value' | 'weight';
  readonly dtype: RuntimeDType;
  readonly sizeBytes: number;
}

interface MutableFreeRegion {
  offsetBytes: number;
  sizeBytes: number;
}

interface MutableArenaRegion {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly birth: number;
  readonly lastUse: number;
  readonly sizeBytes: number;
  offsetBytes: number;
}

const UTF8_ENCODER = new TextEncoder();

function compareCanonicalNames(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = UTF8_ENCODER.encode(left);
  const rightBytes = UTF8_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length;
}

export function maximumCPUActivationArenaTensors(
  graph: Graph,
): Readonly<Record<string, CPUActivationArenaTensor>> {
  const tensors: Record<string, CPUActivationArenaTensor> = {};
  for (const [name, descriptor] of Object.entries(graph.tensors)) {
    let elements = 1n;
    for (const dimension of descriptor.shape) {
      const extent = typeof dimension === 'number'
        ? dimension
        : (() => {
            const constraint = graph.dimensions[dimension];
            if (!constraint) return undefined;
            return constraint.max - (constraint.max % constraint.multiple_of);
          })();
      if (!Number.isSafeInteger(extent) || (extent as number) <= 0) {
        throw new Error(`CPU maximum activation '${name}' has an unbounded extent.`);
      }
      elements *= BigInt(extent as number);
    }
    const size = elements * BigInt(runtimeDTypeBytes(descriptor.dtype));
    if (size <= 0n || size > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`CPU maximum activation '${name}' exceeds safe byte limits.`);
    }
    tensors[name] = Object.freeze({
      name,
      kind: descriptor.kind,
      dtype: descriptor.dtype,
      sizeBytes: Number(size),
    });
  }
  return Object.freeze(tensors);
}

export function concreteCPUActivationArenaTensors(
  plan: { readonly tensors: Readonly<Record<string, CPUActivationArenaTensor>> },
): Readonly<Record<string, CPUActivationArenaTensor>> {
  return Object.freeze(Object.fromEntries(Object.entries(plan.tensors).map(([name, descriptor]) => [
    name,
    Object.freeze({
      name,
      kind: descriptor.kind,
      dtype: descriptor.dtype,
      sizeBytes: descriptor.sizeBytes,
    }),
  ])));
}

function checkedAdd(left: number, right: number, label: string): number {
  if (!Number.isSafeInteger(left) || left < 0 || !Number.isSafeInteger(right) || right < 0 ||
      left > Number.MAX_SAFE_INTEGER - right) {
    throw new Error(`${label} exceeds the safe integer range.`);
  }
  return left + right;
}

function topologyLifetimes(
  graph: Graph,
  tensors: Readonly<Record<string, CPUActivationArenaTensor>>,
): Readonly<Record<string, Readonly<{ birth: number; lastUse: number }>>> {
  const birth = new Map<string, number>();
  const lastUse = new Map<string, number>();
  for (const [name, descriptor] of Object.entries(tensors)) {
    if (descriptor.kind === 'weight') continue;
    if (descriptor.name !== name) {
      throw new Error(`CPU activation descriptor '${name}' has a mismatched name.`);
    }
    const initialBirth = descriptor.kind === 'input' ? -1 : null;
    if (initialBirth !== null) {
      birth.set(name, initialBirth);
      lastUse.set(name, initialBirth);
    }
  }
  for (const [nodeIndex, node] of graph.nodes.entries()) {
    for (const name of Object.values(node.inputs)) {
      const descriptor = tensors[name];
      if (descriptor === undefined) {
        throw new Error(`CPU topology node '${node.id}' references unknown tensor '${name}'.`);
      }
      if (descriptor.kind !== 'weight') {
        if (!birth.has(name)) {
          throw new Error(`CPU topology consumes '${name}' before it is produced.`);
        }
        lastUse.set(name, Math.max(lastUse.get(name)!, nodeIndex));
      }
    }
    for (const output of Object.values(node.outputs)) {
      const name = output.tensor;
      const descriptor = tensors[name];
      if (descriptor === undefined || descriptor.kind !== 'value') {
        throw new Error(`CPU topology node '${node.id}' has invalid value output '${name}'.`);
      }
      if (birth.has(name)) {
        throw new Error(`CPU topology produces '${name}' more than once.`);
      }
      birth.set(name, nodeIndex);
      lastUse.set(name, nodeIndex);
    }
  }
  for (const name of graph.outputs) {
    const descriptor = tensors[name];
    if (descriptor === undefined) {
      throw new Error(`CPU topology declares unknown output '${name}'.`);
    }
    if (descriptor.kind !== 'weight') {
      if (!birth.has(name)) throw new Error(`CPU topology output '${name}' is never produced.`);
      lastUse.set(name, graph.nodes.length);
    }
  }
  const activationNames = Object.entries(tensors)
    .filter(([, descriptor]) => descriptor.kind !== 'weight')
    .map(([name]) => name)
    .sort(compareCanonicalNames);
  if (activationNames.some((name) => !birth.has(name) || !lastUse.has(name))) {
    throw new Error('CPU topology contains an activation without a complete lifetime.');
  }
  return Object.freeze(Object.fromEntries(activationNames.map((name) => [
    name,
    Object.freeze({ birth: birth.get(name)!, lastUse: lastUse.get(name)! }),
  ])));
}

function coalesceFreeRegions(regions: MutableFreeRegion[]): void {
  regions.sort((left, right) => left.offsetBytes - right.offsetBytes);
  for (let index = 1; index < regions.length;) {
    const previous = regions[index - 1];
    const current = regions[index];
    if (previous.offsetBytes + previous.sizeBytes === current.offsetBytes) {
      previous.sizeBytes += current.sizeBytes;
      regions.splice(index, 1);
    } else {
      index++;
    }
  }
}

/**
 * Deterministic best-fit interval packing for one concrete CPU shape plan.
 * Node inputs and outputs overlap at the producing node, so a kernel can never
 * overwrite an operand while producing its result. Public outputs remain live
 * through the result-copy boundary at topology index `nodes.length`.
 */
export function planCPUActivationArena(
  graph: Graph,
  signature: string,
  tensors: Readonly<Record<string, CPUActivationArenaTensor>>,
): BoundActivationLivenessLayout {
  if (typeof signature !== 'string' || signature.length === 0) {
    throw new Error('CPU activation arena signature must be non-empty.');
  }
  const lifetimes = topologyLifetimes(graph, tensors);
  const byDType = new Map<RuntimeDType, MutableArenaRegion[]>();
  let logicalBytes = 0;
  for (const name of Object.keys(lifetimes).sort(compareCanonicalNames)) {
    const descriptor = tensors[name];
    if (!Number.isSafeInteger(descriptor.sizeBytes) || descriptor.sizeBytes <= 0) {
      throw new Error(`CPU activation '${name}' has an invalid byte size.`);
    }
    logicalBytes = checkedAdd(logicalBytes, descriptor.sizeBytes, 'CPU logical activation bytes');
    const lifetime = lifetimes[name];
    const values = byDType.get(descriptor.dtype) ?? [];
    values.push({
      name,
      dtype: descriptor.dtype,
      birth: lifetime.birth,
      lastUse: lifetime.lastUse,
      sizeBytes: descriptor.sizeBytes,
      offsetBytes: 0,
    });
    byDType.set(descriptor.dtype, values);
  }

  const mutableRegions: MutableArenaRegion[] = [];
  const capacityByArena: Record<string, number> = {};
  for (const [dtype, values] of [...byDType.entries()]
    .sort(([left], [right]) => compareCanonicalNames(left, right))) {
    values.sort((left, right) =>
      left.birth - right.birth || right.sizeBytes - left.sizeBytes ||
      compareCanonicalNames(left.name, right.name));
    const live: MutableArenaRegion[] = [];
    const free: MutableFreeRegion[] = [];
    let highWater = 0;
    for (const value of values) {
      for (let index = live.length - 1; index >= 0; index--) {
        if (live[index].lastUse < value.birth) {
          free.push({
            offsetBytes: live[index].offsetBytes,
            sizeBytes: live[index].sizeBytes,
          });
          live.splice(index, 1);
        }
      }
      coalesceFreeRegions(free);
      let best = -1;
      for (let index = 0; index < free.length; index++) {
        if (free[index].sizeBytes < value.sizeBytes) continue;
        if (best === -1 || free[index].sizeBytes < free[best].sizeBytes ||
            (free[index].sizeBytes === free[best].sizeBytes &&
             free[index].offsetBytes < free[best].offsetBytes)) {
          best = index;
        }
      }
      if (best === -1) {
        value.offsetBytes = highWater;
        highWater = checkedAdd(highWater, value.sizeBytes, `CPU ${dtype} arena capacity`);
      } else {
        const selected = free[best];
        value.offsetBytes = selected.offsetBytes;
        selected.offsetBytes += value.sizeBytes;
        selected.sizeBytes -= value.sizeBytes;
        if (selected.sizeBytes === 0) free.splice(best, 1);
      }
      live.push(value);
      mutableRegions.push(value);
    }
    capacityByArena[dtype] = highWater;
  }

  const regions: Record<string, BoundActivationLivenessRegion> = {};
  for (const value of mutableRegions.sort((left, right) =>
    compareCanonicalNames(left.name, right.name))) {
    regions[value.name] = Object.freeze({
      name: value.name,
      dtype: value.dtype,
      arena: value.dtype,
      offsetBytes: value.offsetBytes,
      sizeBytes: value.sizeBytes,
      birth: value.birth,
      lastUse: value.lastUse,
    });
  }
  const capacityBytes = Object.values(capacityByArena).reduce(
    (total, value) => checkedAdd(total, value, 'CPU activation arena capacity'),
    0,
  );
  return Object.freeze({
    protocol: CPU_ACTIVATION_LIVENESS_PROTOCOL,
    graphFingerprint: graph.fingerprint,
    signature,
    capacityByArena: Object.freeze(capacityByArena),
    capacityBytes,
    logicalBytes,
    regions: Object.freeze(regions),
  });
}

/**
 * Best-fit packing is not monotone under smaller interval sizes because its
 * placement decisions can fragment differently. Prefer the concrete packing
 * when every dtype arena fits its independently proved maximum-layout bound;
 * otherwise retain the maximum layout's offsets and shrink only exact views.
 */
export function planCPUActivationArenaWithinMaximum(
  graph: Graph,
  signature: string,
  tensors: Readonly<Record<string, CPUActivationArenaTensor>>,
  maximumLayout: BoundActivationLivenessLayout,
): BoundActivationLivenessLayout {
  const concrete = planCPUActivationArena(graph, signature, tensors);
  if (maximumLayout.graphFingerprint !== graph.fingerprint ||
      maximumLayout.protocol !== CPU_ACTIVATION_LIVENESS_PROTOCOL) {
    throw new Error('CPU maximum activation layout does not belong to this graph.');
  }
  const fits = Object.entries(concrete.capacityByArena).every(([arena, bytes]) =>
    Number.isSafeInteger(maximumLayout.capacityByArena[arena]) &&
    bytes <= maximumLayout.capacityByArena[arena]);
  if (fits) return concrete;

  const regions: Record<string, BoundActivationLivenessRegion> = {};
  const capacityByArena: Record<string, number> = {};
  for (const name of Object.keys(concrete.regions).sort(compareCanonicalNames)) {
    const current = concrete.regions[name];
    const maximum = maximumLayout.regions[name];
    if (!maximum || maximum.dtype !== current.dtype || maximum.arena !== current.arena ||
        maximum.birth !== current.birth || maximum.lastUse !== current.lastUse ||
        maximum.sizeBytes < current.sizeBytes) {
      throw new Error(`CPU maximum activation layout cannot project '${name}'.`);
    }
    regions[name] = Object.freeze({
      ...current,
      offsetBytes: maximum.offsetBytes,
    });
    const end = checkedAdd(maximum.offsetBytes, current.sizeBytes, `CPU projected '${name}' end`);
    capacityByArena[current.arena] = Math.max(capacityByArena[current.arena] ?? 0, end);
  }
  for (const [arena, bytes] of Object.entries(capacityByArena)) {
    if (bytes > maximumLayout.capacityByArena[arena]) {
      throw new Error(`CPU projected '${arena}' arena exceeds its proved maximum.`);
    }
  }
  const capacityBytes = Object.values(capacityByArena).reduce(
    (total, value) => checkedAdd(total, value, 'CPU projected activation capacity'),
    0,
  );
  return Object.freeze({
    protocol: CPU_ACTIVATION_LIVENESS_PROTOCOL,
    graphFingerprint: graph.fingerprint,
    signature,
    capacityByArena: Object.freeze(capacityByArena),
    capacityBytes,
    logicalBytes: concrete.logicalBytes,
    regions: Object.freeze(regions),
  });
}
