import {
  checkedShapeAdd,
  checkedShapeFloorDivide,
  checkedShapeMultiply,
  checkedShapeSubtract,
} from './shapeSystem.js';

function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function spatialKernelPorts(
  node: any,
  requiredInputs: readonly (readonly string[])[],
  optionalInputs: readonly string[],
  operation: string,
): Readonly<{ inputs: readonly any[]; optional: Readonly<Record<string, any>>; output: any }> {
  if (!isRecord(node?.inputs)) throw new Error(`${operation} inputs must be an object.`);
  if (!isRecord(node?.outputs)) throw new Error(`${operation} outputs must be an object.`);
  const allowed = new Set([...requiredInputs.flat(), ...optionalInputs]);
  const inputNames = Reflect.ownKeys(node.inputs);
  if (inputNames.some((name) => typeof name !== 'string' || !allowed.has(name))) {
    throw new Error(`${operation} has unsupported input ports.`);
  }
  const inputs = requiredInputs.map((aliases) => {
    const present = aliases.filter((name) => Object.hasOwn(node.inputs, name));
    if (present.length !== 1) {
      throw new Error(`${operation} requires exactly one '${aliases.join("' or '")}' input.`);
    }
    return node.inputs[present[0]];
  });
  const optional: Record<string, any> = {};
  for (const name of optionalInputs) {
    if (Object.hasOwn(node.inputs, name)) optional[name] = node.inputs[name];
  }
  const outputNames = Reflect.ownKeys(node.outputs);
  if (outputNames.length !== 1 || outputNames[0] !== 'out') {
    throw new Error(`${operation} requires exactly one 'out' output.`);
  }
  return Object.freeze({ inputs: Object.freeze(inputs), optional: Object.freeze(optional), output: node.outputs.out });
}

export function spatialPair(
  source: unknown,
  defaultValue: number,
  operation: string,
  name: string,
  allowZero: boolean,
  required = false,
): readonly [number, number] {
  if (source === undefined && required) throw new Error(`${operation} ${name} is required.`);
  const raw = source === undefined
    ? [defaultValue, defaultValue]
    : Array.isArray(source)
      ? source
      : [source, source];
  if (raw.length < 1 || raw.length > 2) {
    throw new Error(`${operation} ${name} must be a scalar or a one/two-element array.`);
  }
  const pair = [raw[0], raw[1] ?? raw[0]];
  const minimum = allowZero ? 0 : 1;
  if (pair.some((value) => !Number.isSafeInteger(value) || value < minimum)) {
    throw new Error(`${operation} ${name} must contain ${allowZero ? 'non-negative' : 'positive'} safe integers.`);
  }
  return Object.freeze(pair as [number, number]);
}

export function spatialScalar(
  source: unknown,
  defaultValue: number,
  operation: string,
  name: string,
  allowZero: boolean,
): number {
  const raw = source === undefined
    ? defaultValue
    : Array.isArray(source) && source.length === 1
      ? source[0]
      : source;
  const minimum = allowZero ? 0 : 1;
  if (!Number.isSafeInteger(raw) || (raw as number) < minimum) {
    throw new Error(
      `${operation} ${name} must be a ${allowZero ? 'non-negative' : 'positive'} safe integer or one-element array.`,
    );
  }
  return raw as number;
}

export function positiveInteger(
  source: unknown,
  defaultValue: number,
  operation: string,
  name: string,
): number {
  const value = source ?? defaultValue;
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`${operation} ${name} must be a positive safe integer.`);
  }
  return value as number;
}

export function fusedActivation(source: unknown, operation: string): number {
  const value = source ?? 0;
  if (!Number.isInteger(value) || (value as number) < 0 || (value as number) > 2) {
    throw new Error(`${operation} relu must be 0, 1, or 2.`);
  }
  return value as number;
}

export function assertFalseOrAbsent(source: unknown, operation: string, name: string): void {
  if (source !== undefined && source !== false && source !== 0) {
    throw new Error(`${operation} ${name} must be false, 0, or absent.`);
  }
}

export function assertCanonicalLayout(
  source: unknown,
  expected: string,
  operation: string,
  name: string,
): void {
  if (source !== undefined && source !== expected) {
    throw new Error(`${operation} ${name} must be '${expected}'.`);
  }
}

export function fullSpatialPads(
  params: Readonly<Record<string, any>>,
  operation: string,
  symmetricOnly = false,
): readonly [number, number, number, number] {
  const padding = spatialPair(params.padding, 0, operation, 'padding', true);
  if (params.pads === undefined) {
    return Object.freeze([padding[0], padding[1], padding[0], padding[1]]);
  }
  if (!Array.isArray(params.pads) || params.pads.length !== 4 ||
      params.pads.some((value: unknown) => !Number.isSafeInteger(value) || (value as number) < 0)) {
    throw new Error(`${operation} pads must contain non-negative top, left, bottom, and right safe integers.`);
  }
  const pads = params.pads as [number, number, number, number];
  if (params.padding !== undefined &&
      (pads[0] !== padding[0] || pads[1] !== padding[1] ||
       pads[2] !== padding[0] || pads[3] !== padding[1])) {
    throw new Error(`${operation} padding and pads must describe the same symmetric padding.`);
  }
  if (symmetricOnly && (pads[0] !== pads[2] || pads[1] !== pads[3])) {
    throw new Error(`${operation} pads must be symmetric.`);
  }
  return Object.freeze([...pads] as [number, number, number, number]);
}

export function checkedWindowOutput(
  input: number,
  kernel: number,
  stride: number,
  padBefore: number,
  padAfter: number,
  dilation: number,
  operation: string,
): number {
  const effectiveKernel = checkedShapeAdd(
    checkedShapeMultiply(dilation, checkedShapeSubtract(kernel, 1, operation), operation),
    1,
    operation,
  );
  const padded = checkedShapeAdd(checkedShapeAdd(input, padBefore, operation), padAfter, operation);
  const output = checkedShapeAdd(
    checkedShapeFloorDivide(checkedShapeSubtract(padded, effectiveKernel, operation), stride, operation),
    1,
    operation,
  );
  if (output <= 0) throw new Error(`${operation} produces a non-positive output extent.`);
  return output;
}

export function checkedTransposeOutput(
  input: number,
  kernel: number,
  stride: number,
  padding: number,
  operation: string,
): number {
  const output = checkedShapeSubtract(
    checkedShapeAdd(
      checkedShapeMultiply(checkedShapeSubtract(input, 1, operation), stride, operation),
      kernel,
      operation,
    ),
    checkedShapeMultiply(2, padding, operation),
    operation,
  );
  if (output <= 0) throw new Error(`${operation} produces a non-positive output extent.`);
  return output;
}

export function assertDistinctOutputStorage(
  output: any,
  inputs: readonly any[],
  operation: string,
): void {
  const outputView = output?.buffer;
  const overlaps = inputs.some((input) => {
    const inputView = input?.buffer;
    if (!ArrayBuffer.isView(outputView) || !ArrayBuffer.isView(inputView) ||
        outputView.buffer !== inputView.buffer) return false;
    const outputStart = outputView.byteOffset;
    const outputEnd = outputStart + outputView.byteLength;
    const inputStart = inputView.byteOffset;
    const inputEnd = inputStart + inputView.byteLength;
    return outputStart < inputEnd && inputStart < outputEnd;
  });
  if (overlaps) {
    throw new Error(`${operation} output storage must not alias an input.`);
  }
}
