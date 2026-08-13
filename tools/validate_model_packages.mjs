#!/usr/bin/env node

import { execFile } from 'node:child_process';
import { lstat, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { runtimeOperatorNames } from './generated/volvoxaiGraphOperators.mjs';

export const VOLVOX_GRAPH_FORMAT = 'volvox-graph/v1';
export const VOLVOX_AFFINE_FORMAT = 'volvox-affine-safetensors/v1';
const GRAPH_DTYPES = new Set(['float32', 'int32', 'int8', 'uint8']);
const GRAPH_DTYPE_BYTES = new Map([
  ['float32', 4n],
  ['int32', 4n],
  ['int8', 1n],
  ['uint8', 1n],
]);
const RUNTIME_OPERATOR_NAMES = new Set(runtimeOperatorNames);
const SHAPE_SYMBOL_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,63}$/u;
const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const ROOT_FIELDS = ['format', 'dimensions', 'inputs', 'nodes', 'outputs'];
const ROOT_OPTIONAL_FIELDS = ['banks', 'quantization'];
const INPUT_FIELDS = ['dtype', 'shape'];
const NODE_FIELDS = ['id', 'opType', 'inputs', 'outputs', 'params'];
const OUTPUT_FIELDS = ['tensor', 'dtype', 'shape'];
const runFile = promisify(execFile);
const REPOSITORY_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SAFETENSORS_DTYPES = new Map([
  ['F16', { graphDtype: 'float32', bytes: 2 }],
  ['F32', { graphDtype: 'float32', bytes: 4 }],
  ['I32', { graphDtype: 'int32', bytes: 4 }],
  ['I8', { graphDtype: 'int8', bytes: 1 }],
  ['U8', { graphDtype: 'uint8', bytes: 1 }],
]);
const LEGACY_QUANTIZATION_FIELDS = new Set([
  'weights_quantization', 'weights_quantization_storage',
]);
const INLINE_AFFINE_PARAM_FIELDS = new Set([
  'quantization', 'zero_point', 'scales', 'zero_points',
  'input_scale', 'input_zero_point', 'output_scale', 'output_zero_point',
  'weight_scale', 'weight_zero_point', 'scale_tensor', 'zero_point_tensor',
]);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isGraphFilename(filename) {
  return filename === 'graph.json' || filename.endsWith('.graph.json');
}

async function collectFiles(root) {
  const files = [];
  const visit = async (current) => {
    const metadata = await lstat(current);
    if (metadata.isSymbolicLink()) return;
    if (metadata.isFile()) {
      files.push(current);
      return;
    }
    if (!metadata.isDirectory()) return;
    const entries = await readdir(current, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      await visit(path.join(current, entry.name));
    }
  };
  await visit(root);
  return files;
}

async function parseJson(filename) {
  const text = await readFile(filename, 'utf8');
  try {
    return parseStrictJson(text, filename);
  } catch (error) {
    throw new Error(`${filename}: invalid JSON (${error.message})`);
  }
}

function parseStrictJson(source, label) {
  const parsed = JSON.parse(source);
  let cursor = 0;
  const skipWhitespace = () => {
    while (/\s/u.test(source[cursor] || '')) cursor++;
  };
  const readString = () => {
    const start = cursor++;
    while (cursor < source.length) {
      const character = source[cursor++];
      if (character === '\\') cursor++;
      else if (character === '"') return JSON.parse(source.slice(start, cursor));
    }
    return '';
  };
  const scanObject = () => {
    cursor++;
    skipWhitespace();
    const keys = new Set();
    if (source[cursor] === '}') {
      cursor++;
      return;
    }
    while (cursor < source.length) {
      skipWhitespace();
      const key = readString();
      if (keys.has(key)) {
        throw new SyntaxError(`${label} contains duplicate object key ${JSON.stringify(key)}`);
      }
      keys.add(key);
      skipWhitespace();
      cursor++;
      scanValue();
      skipWhitespace();
      if (source[cursor] === ',') {
        cursor++;
        continue;
      }
      cursor++;
      return;
    }
  };
  const scanArray = () => {
    cursor++;
    skipWhitespace();
    if (source[cursor] === ']') {
      cursor++;
      return;
    }
    while (cursor < source.length) {
      scanValue();
      skipWhitespace();
      if (source[cursor] === ',') {
        cursor++;
        continue;
      }
      cursor++;
      return;
    }
  };
  function scanValue() {
    skipWhitespace();
    if (source[cursor] === '{') scanObject();
    else if (source[cursor] === '[') scanArray();
    else if (source[cursor] === '"') readString();
    else {
      while (cursor < source.length && !/[\s,\]}]/u.test(source[cursor])) cursor++;
    }
  }
  scanValue();
  const seen = new WeakSet();
  const verifyValue = (value, pathLabel) => {
    if (value == null || typeof value === 'string' || typeof value === 'boolean') return;
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        throw new SyntaxError(`${pathLabel} contains a non-finite JSON number`);
      }
      if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
        throw new SyntaxError(`${pathLabel} contains an integer outside JSON's safe range`);
      }
      return;
    }
    if (typeof value !== 'object') {
      throw new SyntaxError(`${pathLabel} contains a non-JSON ${typeof value} value`);
    }
    if (seen.has(value)) throw new SyntaxError(`${pathLabel} contains a JSON cycle`);
    seen.add(value);
    if (Array.isArray(value)) {
      value.forEach((child, index) => verifyValue(child, `${pathLabel}[${index}]`));
    } else {
      for (const [key, child] of Object.entries(value)) {
        verifyValue(child, `${pathLabel}.${key}`);
      }
    }
    seen.delete(value);
  };
  verifyValue(parsed, label);
  return parsed;
}

async function readSafetensorsFile(filename) {
  const bytes = await readFile(filename);
  if (bytes.byteLength < 8) {
    throw new Error(`${filename}: safetensors header is truncated`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLengthBig = view.getBigUint64(0, true);
  if (headerLengthBig > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${filename}: safetensors header is too large`);
  }
  const headerLength = Number(headerLengthBig);
  if (headerLength <= 0 || headerLength > bytes.byteLength - 8) {
    throw new Error(`${filename}: safetensors header length is invalid`);
  }
  let header;
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(
      bytes.subarray(8, 8 + headerLength),
    );
    header = parseStrictJson(text, `${filename}: safetensors header`);
  } catch (error) {
    throw new Error(`${filename}: invalid safetensors header (${error.message})`);
  }
  if (!isObject(header)) throw new Error(`${filename}: safetensors header must be an object`);
  const metadata = header.__metadata__;
  if (Object.hasOwn(header, '__metadata__') && (!isObject(metadata) ||
      Object.values(metadata).some((value) => typeof value !== 'string'))) {
    throw new Error(`${filename}: safetensors metadata must contain only string values`);
  }
  for (const field of LEGACY_QUANTIZATION_FIELDS) {
    if (isObject(metadata) && Object.hasOwn(metadata, field)) {
      throw new Error(`${filename}: safetensors metadata forbids legacy field '${field}'`);
    }
  }

  const dataStart = 8 + headerLength;
  const dataLength = bytes.byteLength - dataStart;
  const tensors = new Map();
  const spans = [];
  for (const [name, entry] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (!name.trim() || !isObject(entry) || !sameFields(entry, ['dtype', 'shape', 'data_offsets'])) {
      throw new Error(`${filename}: tensor ${JSON.stringify(name)} has an invalid header record`);
    }
    const dtypeInfo = SAFETENSORS_DTYPES.get(entry.dtype);
    if (!dtypeInfo) {
      throw new Error(`${filename}: tensor ${JSON.stringify(name)} has unsupported dtype ${JSON.stringify(entry.dtype)}`);
    }
    if (!Array.isArray(entry.shape) || entry.shape.some((dimension) =>
      !Number.isSafeInteger(dimension) || dimension <= 0)) {
      throw new Error(`${filename}: tensor ${JSON.stringify(name)} has an invalid shape`);
    }
    let elementCount = 1n;
    for (const dimension of entry.shape) {
      elementCount *= BigInt(dimension);
    }
    const expectedBytes = elementCount * BigInt(dtypeInfo.bytes);
    if (elementCount > MAX_SAFE_INTEGER_BIGINT || expectedBytes > MAX_SAFE_INTEGER_BIGINT) {
      throw new Error(
        `${filename}: tensor ${JSON.stringify(name)} exceeds safe static allocation limits`,
      );
    }
    if (!Array.isArray(entry.data_offsets) ||
        entry.data_offsets.length !== 2 || entry.data_offsets.some((offset) =>
          !Number.isSafeInteger(offset))) {
      throw new Error(`${filename}: tensor ${JSON.stringify(name)} has invalid data offsets`);
    }
    const [start, end] = entry.data_offsets;
    if (start < 0 || end < start || end > dataLength ||
        BigInt(end - start) !== expectedBytes) {
      throw new Error(`${filename}: tensor ${JSON.stringify(name)} data span is invalid`);
    }
    const tensor = {
      dtype: dtypeInfo.graphDtype,
      safetensorsDtype: entry.dtype,
      shape: [...entry.shape],
      data: bytes.subarray(dataStart + start, dataStart + end),
    };
    tensors.set(name, tensor);
    spans.push({ name, start, end });
  }
  spans.sort((left, right) => left.start - right.start || left.name.localeCompare(right.name));
  let cursor = 0;
  for (const span of spans) {
    if (span.start !== cursor) {
      throw new Error(`${filename}: safetensors data has a gap, overlap, or aliased span near ${JSON.stringify(span.name)}`);
    }
    cursor = span.end;
  }
  if (cursor !== dataLength) {
    throw new Error(`${filename}: safetensors tensor spans do not account for the full data payload`);
  }
  return tensors;
}

function mergeSafetensorsIndexes(files, parsedFiles) {
  const tensors = new Map();
  for (const filename of files) {
    for (const [name, tensor] of parsedFiles.get(filename) || []) {
      if (tensors.has(name)) {
        throw new Error(`${filename}: tensor ${JSON.stringify(name)} occurs in multiple safetensors files`);
      }
      tensors.set(name, tensor);
    }
  }
  return tensors;
}

function nearestSafetensorsFiles(graphFilename, safetensorsFiles) {
  const graphDirectory = path.dirname(graphFilename);
  let nearestDirectory = null;
  for (const filename of safetensorsFiles) {
    const candidateDirectory = path.dirname(filename);
    const isAncestor = graphDirectory === candidateDirectory ||
      graphDirectory.startsWith(`${candidateDirectory}${path.sep}`);
    if (!isAncestor) continue;
    if (nearestDirectory == null || candidateDirectory.length > nearestDirectory.length) {
      nearestDirectory = candidateDirectory;
    }
  }
  if (nearestDirectory == null) return [];
  return safetensorsFiles.filter((filename) => path.dirname(filename) === nearestDirectory);
}

async function validateRuntimeSemantics(packages) {
  const args = ['-m', 'tools.exporter.validate_runtime_package'];
  for (const packageSpec of packages) {
    args.push('--package', JSON.stringify(packageSpec));
  }
  const python = process.env.PYTHON || process.env.PY || 'python3';
  try {
    await runFile(python, args, {
      cwd: REPOSITORY_ROOT,
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
    });
  } catch (error) {
    const detail = [error.stderr, error.stdout, error.message]
      .find((value) => typeof value === 'string' && value.trim())?.trim();
    throw new Error(
      `strict Python RuntimeIR validation failed${detail ? ` (${detail})` : ''}`,
    );
  }
}

function sameFields(value, expected) {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((field, index) => field === wanted[index]);
}

function exactFields(value, required, optional, label, failures) {
  if (!isObject(value)) {
    failures.push(`${label} must be an object`);
    return false;
  }
  const allowed = new Set([...required, ...optional]);
  const unsupported = Object.keys(value).filter((field) => !allowed.has(field)).sort();
  const missing = required.filter((field) => !Object.hasOwn(value, field));
  if (unsupported.length > 0) {
    failures.push(`${label} has unsupported field ${JSON.stringify(unsupported[0])}`);
  }
  if (missing.length > 0) {
    failures.push(`${label} requires field ${JSON.stringify(missing[0])}`);
  }
  return unsupported.length === 0 && missing.length === 0;
}

function validateDimensions(value, filename, failures) {
  if (!isObject(value)) {
    failures.push(`${filename}: graph dimensions must be an object`);
    return null;
  }
  const environment = new Map();
  for (const name of Object.keys(value).sort()) {
    const label = `${filename}: dimension ${JSON.stringify(name)}`;
    if (!SHAPE_SYMBOL_PATTERN.test(name)) {
      failures.push(`${label} name must match /^[A-Za-z][A-Za-z0-9_]{0,63}$/`);
      continue;
    }
    const descriptor = value[name];
    if (!exactFields(descriptor, ['min', 'max'], ['multiple_of'], label, failures)) {
      continue;
    }
    const minimum = descriptor.min;
    const maximum = descriptor.max;
    const multiple = Object.hasOwn(descriptor, 'multiple_of')
      ? descriptor.multiple_of
      : 1;
    if (![minimum, maximum, multiple].every((item) =>
      Number.isSafeInteger(item) && item > 0) || minimum > maximum) {
      failures.push(
        `${label} requires positive safe-integer min/max/multiple_of with min <= max`,
      );
      continue;
    }
    const minimumBig = BigInt(minimum);
    const maximumBig = BigInt(maximum);
    const multipleBig = BigInt(multiple);
    const firstLegal = ((minimumBig + multipleBig - 1n) / multipleBig) * multipleBig;
    if (firstLegal > maximumBig) {
      failures.push(`${label} has no legal multiple_of value inside its bounds`);
      continue;
    }
    environment.set(name, Object.freeze({ min: minimum, max: maximum, multiple_of: multiple }));
  }
  return environment;
}

function validateBoundedShape(value, dtype, environment, label, failures) {
  if (!Array.isArray(value)) {
    failures.push(`${label} must be a fixed-rank shape array`);
    return false;
  }
  let elementCount = 1n;
  let valid = true;
  for (const [axis, dimension] of value.entries()) {
    let maximum;
    if (Number.isSafeInteger(dimension) && dimension > 0) {
      maximum = dimension;
    } else if (typeof dimension === 'string' && environment?.has(dimension)) {
      maximum = environment.get(dimension).max;
    } else {
      failures.push(
        `${label}[${axis}] must be a positive safe integer or declared dimension symbol`,
      );
      valid = false;
      continue;
    }
    elementCount *= BigInt(maximum);
    if (elementCount > MAX_SAFE_INTEGER_BIGINT) {
      failures.push(`${label} maximum element count exceeds Number.MAX_SAFE_INTEGER`);
      valid = false;
      break;
    }
  }
  const dtypeBytes = GRAPH_DTYPE_BYTES.get(dtype);
  if (dtypeBytes !== undefined && elementCount * dtypeBytes > MAX_SAFE_INTEGER_BIGINT) {
    failures.push(`${label} maximum byte length exceeds Number.MAX_SAFE_INTEGER`);
    valid = false;
  }
  return valid;
}

function nodeOutputTensorNames(node) {
  if (!isObject(node?.outputs)) return [];
  const names = [];
  for (const descriptor of Object.values(node.outputs)) {
    if (isObject(descriptor) && typeof descriptor.tensor === 'string') {
      names.push(descriptor.tensor);
    }
  }
  return names;
}

function findInlineAffineParam(value, path = 'params') {
  if (Array.isArray(value)) {
    for (const [index, item] of value.entries()) {
      const found = findInlineAffineParam(item, `${path}[${index}]`);
      if (found) return found;
    }
    return null;
  }
  if (!isObject(value)) return null;
  for (const field of INLINE_AFFINE_PARAM_FIELDS) {
    if (Object.hasOwn(value, field)) return `${path}.${field}`;
  }
  for (const [field, item] of Object.entries(value)) {
    const found = findInlineAffineParam(item, `${path}.${field}`);
    if (found) return found;
  }
  return null;
}

function validateClosedGraphSchema(document, filename, failures) {
  if (document.format !== VOLVOX_GRAPH_FORMAT) {
    failures.push(
      `${filename}: format must be exactly '${VOLVOX_GRAPH_FORMAT}', got ${JSON.stringify(document.format)}`,
    );
  }
  exactFields(
    document,
    ROOT_FIELDS,
    ROOT_OPTIONAL_FIELDS,
    `${filename}: graph root`,
    failures,
  );
  const environment = validateDimensions(document.dimensions, filename, failures);

  if (!isObject(document.inputs)) {
    failures.push(`${filename}: graph requires an inputs object`);
  } else {
    for (const [name, descriptor] of Object.entries(document.inputs)) {
      const label = `${filename}: input ${JSON.stringify(name)}`;
      if (!name.trim()) failures.push(`${label} name must be non-empty`);
      if (!exactFields(descriptor, INPUT_FIELDS, [], label, failures)) continue;
      if (!GRAPH_DTYPES.has(descriptor.dtype)) {
        failures.push(`${label} requires a canonical lowercase runtime dtype`);
      }
      validateBoundedShape(
        descriptor.shape,
        descriptor.dtype,
        environment,
        `${label} shape`,
        failures,
      );
    }
  }

  const nodeIds = new Set();
  if (!Array.isArray(document.nodes)) {
    failures.push(`${filename}: graph requires a nodes array`);
  } else {
    document.nodes.forEach((node, index) => {
      const label = `${filename}: node ${index}`;
      if (!isObject(node)) {
        failures.push(`${label} must be an object`);
        return;
      }
      if (Object.hasOwn(node, 'outputs_shape') || Object.hasOwn(node, 'outputs_dtype')) {
        failures.push(
          `${label} uses legacy split output descriptors; re-export with unified outputs`,
        );
        return;
      }
      if (!exactFields(node, NODE_FIELDS, [], label, failures)) return;
      if (typeof node.id !== 'string' || !node.id.trim()) {
        failures.push(`${label} requires a non-empty id`);
      } else if (nodeIds.has(node.id)) {
        failures.push(`${label} duplicates node id ${JSON.stringify(node.id)}`);
      } else {
        nodeIds.add(node.id);
      }
      if (typeof node.opType !== 'string' || !node.opType.trim()) {
        failures.push(`${label} requires a non-empty opType`);
      } else if (!RUNTIME_OPERATOR_NAMES.has(node.opType)) {
        failures.push(
          `${label} uses unknown or offline-only current-v1 opType ${JSON.stringify(node.opType)}`,
        );
      }
      if (!isObject(node.inputs)) {
        failures.push(`${label} requires an inputs object`);
      } else {
        for (const [port, tensor] of Object.entries(node.inputs)) {
          if (!port.trim() || typeof tensor !== 'string' || !tensor.trim()) {
            failures.push(`${label} input ${JSON.stringify(port)} must name a tensor`);
          }
        }
      }
      if (!isObject(node.params)) {
        failures.push(`${label} params must be an object`);
      } else {
        const inlineAffinePath = findInlineAffineParam(node.params);
        if (inlineAffinePath) {
          failures.push(`${label} forbids inline affine field '${inlineAffinePath}'`);
        }
      }
      if (!isObject(node.outputs) || Object.keys(node.outputs).length === 0) {
        failures.push(`${label} requires named output descriptors`);
        return;
      }
      for (const [port, descriptor] of Object.entries(node.outputs)) {
        const outputLabel = `${label} output ${JSON.stringify(port)}`;
        if (!port.trim()) failures.push(`${outputLabel} port must be non-empty`);
        if (!exactFields(descriptor, OUTPUT_FIELDS, [], outputLabel, failures)) continue;
        if (typeof descriptor.tensor !== 'string' || !descriptor.tensor.trim()) {
          failures.push(`${outputLabel} tensor must be a non-empty name`);
        }
        if (!GRAPH_DTYPES.has(descriptor.dtype)) {
          failures.push(`${outputLabel} requires a canonical lowercase runtime dtype`);
        }
        validateBoundedShape(
          descriptor.shape,
          descriptor.dtype,
          environment,
          `${outputLabel} shape`,
          failures,
        );
      }
    });
  }

  if (!Array.isArray(document.outputs) || document.outputs.length === 0 ||
      document.outputs.some((name) => typeof name !== 'string' || !name.trim()) ||
      new Set(document.outputs).size !== document.outputs.length) {
    failures.push(
      `${filename}: graph outputs must be a non-empty array of unique non-empty tensor names`,
    );
  }
  return environment;
}

function tensorValues(tensor) {
  const view = new DataView(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
  if (tensor.safetensorsDtype === 'F32') {
    return Array.from({ length: tensor.data.byteLength / 4 }, (_, index) =>
      view.getFloat32(index * 4, true));
  }
  if (tensor.safetensorsDtype === 'I8') {
    return Array.from({ length: tensor.data.byteLength }, (_, index) => view.getInt8(index));
  }
  if (tensor.safetensorsDtype === 'U8') return Array.from(tensor.data);
  return [];
}

function validateTopology(document, safetensors, filename, failures) {
  const available = new Set(safetensors.keys());
  for (const name of Object.keys(isObject(document.inputs) ? document.inputs : {})) {
    if (!name.trim()) {
      failures.push(`${filename}: graph input names must be non-empty`);
    } else if (available.has(name)) {
      failures.push(`${filename}: graph input ${JSON.stringify(name)} collides with a safetensors tensor`);
    } else {
      available.add(name);
    }
  }
  for (const [index, node] of (Array.isArray(document.nodes) ? document.nodes : []).entries()) {
    if (!isObject(node)) continue;
    if (isObject(node.inputs)) {
      for (const [port, name] of Object.entries(node.inputs)) {
        if (!port.trim() || typeof name !== 'string' || !name.trim()) {
          failures.push(`${filename}: node ${index} input ${JSON.stringify(port)} must name a tensor`);
        } else if (!available.has(name)) {
          failures.push(`${filename}: node ${index} input ${JSON.stringify(name)} is unresolved or not topologically available`);
        }
      }
    }
    if (!isObject(node.outputs)) continue;
    for (const [port, descriptor] of Object.entries(node.outputs)) {
      const name = isObject(descriptor) ? descriptor.tensor : null;
      if (!port.trim() || typeof name !== 'string' || !name.trim()) continue;
      if (available.has(name)) {
        failures.push(`${filename}: node ${index} output ${JSON.stringify(name)} collides with an existing tensor`);
      } else {
        available.add(name);
      }
    }
  }
  if (!Array.isArray(document.outputs)) return;
  for (const name of document.outputs) {
    if (typeof name === 'string' && name.trim() && !available.has(name)) {
      failures.push(`${filename}: public output ${JSON.stringify(name)} is unresolved`);
    }
  }
}

function validateWeightBanks(document, safetensors, filename, failures) {
  if (!Object.hasOwn(document, 'banks')) return;
  if (!isObject(document.banks)) {
    failures.push(`${filename}: banks must be an object`);
    return;
  }
  const dimensions = isObject(document.dimensions) ? document.dimensions : {};
  for (const [name, dimension] of Object.entries(document.banks)) {
    const label = `${filename}: bank ${JSON.stringify(name)}`;
    if (typeof dimension !== 'string' || !dimension.trim()) {
      failures.push(`${label} must name a non-empty declared dimension`);
      continue;
    }
    const constraint = dimensions[dimension];
    if (!isObject(constraint)) {
      failures.push(`${label} references undeclared dimension ${JSON.stringify(dimension)}`);
      continue;
    }
    const multiple = Object.hasOwn(constraint, 'multiple_of') ? constraint.multiple_of : 1;
    if (![constraint.min, constraint.max, multiple].every((value) =>
      Number.isSafeInteger(value) && value > 0) || constraint.min > constraint.max) {
      // validateDimensions owns the malformed-constraint diagnostic.
      continue;
    }
    const weight = safetensors.get(name);
    if (!weight) {
      failures.push(`${label} does not name a supplied fixed weight`);
      continue;
    }
    if (weight.shape.length < 2) {
      failures.push(`${label} weight needs a slot axis and at least one payload axis`);
      continue;
    }
    const slots = weight.shape[0];
    if (slots < constraint.min || slots > constraint.max) {
      failures.push(
        `${label} supplies ${slots} slots outside dimension ${JSON.stringify(dimension)} ` +
        `bounds [${constraint.min}, ${constraint.max}]`,
      );
      continue;
    }
    if (slots % multiple !== 0) {
      failures.push(`${label} supplies ${slots} slots, which is not a multiple of ${multiple}`);
    }
  }
}

function validateAffineQuantization(document, safetensors, filename, failures) {
  for (const field of LEGACY_QUANTIZATION_FIELDS) {
    if (Object.hasOwn(document, field)) {
      failures.push(`${filename}: '${VOLVOX_GRAPH_FORMAT}' forbids legacy field '${field}'`);
    }
  }
  for (const [name, descriptor] of Object.entries(isObject(document.inputs) ? document.inputs : {})) {
    if (isObject(descriptor) && Object.hasOwn(descriptor, 'quantization')) {
      failures.push(`${filename}: input ${JSON.stringify(name)} forbids inline quantization`);
    }
  }
  for (const [index, node] of (Array.isArray(document.nodes) ? document.nodes : []).entries()) {
    if (isObject(node) && Object.hasOwn(node, 'outputs_quantization')) {
      failures.push(`${filename}: node ${index} forbids inline outputs_quantization`);
    }
  }
  if (document.quantization == null) return;
  const root = document.quantization;
  if (!isObject(root) || !sameFields(root, ['format', 'tensors']) ||
      root.format !== VOLVOX_AFFINE_FORMAT || !isObject(root.tensors) ||
      Object.keys(root.tensors).length === 0) {
    failures.push(`${filename}: quantization must be a non-empty exact '${VOLVOX_AFFINE_FORMAT}' table`);
    return;
  }

  const declared = new Map(safetensors);
  for (const [name, descriptor] of Object.entries(isObject(document.inputs) ? document.inputs : {})) {
    if (isObject(descriptor) && GRAPH_DTYPES.has(descriptor.dtype) && Array.isArray(descriptor.shape)) {
      declared.set(name, { dtype: descriptor.dtype, shape: descriptor.shape });
    }
  }
  for (const node of Array.isArray(document.nodes) ? document.nodes : []) {
    if (!isObject(node) || !isObject(node.outputs)) continue;
    for (const descriptor of Object.values(node.outputs)) {
      if (isObject(descriptor) && typeof descriptor.tensor === 'string' &&
          GRAPH_DTYPES.has(descriptor.dtype) && Array.isArray(descriptor.shape)) {
        declared.set(descriptor.tensor, {
          dtype: descriptor.dtype,
          shape: descriptor.shape,
        });
      }
    }
  }

  const parameterNames = new Set();
  for (const [name, descriptor] of Object.entries(root.tensors)) {
    const target = declared.get(name);
    const perAxis = isObject(descriptor) && descriptor.scheme === 'per_axis';
    const expectedFields = perAxis
      ? ['scheme', 'axis', 'scale_tensor', 'zero_point_tensor']
      : ['scheme', 'scale_tensor', 'zero_point_tensor'];
    if (!target || !['int8', 'uint8'].includes(target.dtype)) {
      failures.push(`${filename}: quantization target ${JSON.stringify(name)} must be a declared I8/U8 tensor`);
      continue;
    }
    if (!isObject(descriptor) || !sameFields(descriptor, expectedFields) ||
        !['per_tensor', 'per_axis'].includes(descriptor.scheme) ||
        typeof descriptor.scale_tensor !== 'string' || !descriptor.scale_tensor ||
        typeof descriptor.zero_point_tensor !== 'string' || !descriptor.zero_point_tensor ||
        descriptor.scale_tensor === descriptor.zero_point_tensor) {
      failures.push(`${filename}: quantization descriptor for ${JSON.stringify(name)} is invalid`);
      continue;
    }
    parameterNames.add(descriptor.scale_tensor);
    parameterNames.add(descriptor.zero_point_tensor);
    let count = 1;
    if (perAxis) {
      let axis = descriptor.axis;
      if (!Number.isInteger(axis)) {
        failures.push(`${filename}: per-axis quantization for ${JSON.stringify(name)} requires an integer axis`);
        continue;
      }
      if (axis < 0) axis += target.shape.length;
      if (axis < 0 || axis >= target.shape.length) {
        failures.push(`${filename}: per-axis quantization for ${JSON.stringify(name)} has an invalid axis`);
        continue;
      }
      count = target.shape[axis];
      if (!Number.isSafeInteger(count) || count <= 0) {
        failures.push(
          `${filename}: per-axis quantization for ${JSON.stringify(name)} requires a constant target axis`,
        );
        continue;
      }
    }
    const scale = safetensors.get(descriptor.scale_tensor);
    const zeroPoint = safetensors.get(descriptor.zero_point_tensor);
    const expectedZeroSafetensorsDtype = target.dtype === 'int8' ? 'I8' : 'U8';
    if (!scale || scale.safetensorsDtype !== 'F32' || scale.shape.length !== 1 || scale.shape[0] !== count ||
        !zeroPoint || zeroPoint.safetensorsDtype !== expectedZeroSafetensorsDtype || zeroPoint.shape.length !== 1 ||
        zeroPoint.shape[0] !== count) {
      failures.push(
        `${filename}: quantization parameters for ${JSON.stringify(name)} must be rank-1 safetensors ` +
        `F32 and ${expectedZeroSafetensorsDtype} arrays of length ${count}`,
      );
      continue;
    }
    if (tensorValues(scale).some((value) => !Number.isFinite(value) || value <= 0)) {
      failures.push(`${filename}: quantization scales for ${JSON.stringify(name)} must be finite and positive F32 values`);
    }
    const minimum = target.dtype === 'int8' ? -128 : 0;
    const maximum = target.dtype === 'int8' ? 127 : 255;
    if (tensorValues(zeroPoint).some((value) =>
      !Number.isInteger(value) || value < minimum || value > maximum)) {
      failures.push(`${filename}: quantization zero points for ${JSON.stringify(name)} must be exact ${expectedZeroSafetensorsDtype} values`);
    }
  }
  for (const name of parameterNames) {
    if (Object.hasOwn(root.tensors, name)) {
      failures.push(`${filename}: quantization parameter ${JSON.stringify(name)} cannot itself be quantized`);
    }
    if (Object.hasOwn(isObject(document.inputs) ? document.inputs : {}, name)) {
      failures.push(`${filename}: quantization parameter ${JSON.stringify(name)} cannot be a graph input`);
    }
    if ((Array.isArray(document.outputs) ? document.outputs : []).includes(name)) {
      failures.push(`${filename}: quantization parameter ${JSON.stringify(name)} cannot be a public output`);
    }
    for (const [index, node] of (Array.isArray(document.nodes) ? document.nodes : []).entries()) {
      if (isObject(node) && nodeOutputTensorNames(node).includes(name)) {
        failures.push(`${filename}: node ${index} cannot produce reserved quantization parameter ${JSON.stringify(name)}`);
      }
    }
  }
  for (const [index, node] of (Array.isArray(document.nodes) ? document.nodes : []).entries()) {
    if (!isObject(node) || !isObject(node.inputs) || !isObject(node.outputs)) continue;
    if (node.opType === 'QuantizeLinear') {
      for (const outputName of nodeOutputTensorNames(node)) {
        const descriptor = root.tensors[outputName];
        if (!isObject(descriptor) || descriptor.scale_tensor !== node.inputs.scale ||
            descriptor.zero_point_tensor !== node.inputs.zero_point) {
          failures.push(`${filename}: QuantizeLinear node ${index} operands must match its central output references`);
        }
      }
    } else if (node.opType === 'DequantizeLinear') {
      const descriptor = root.tensors[node.inputs.input];
      if (!isObject(descriptor) || descriptor.scale_tensor !== node.inputs.scale ||
          descriptor.zero_point_tensor !== node.inputs.zero_point) {
        failures.push(`${filename}: DequantizeLinear node ${index} operands must match its central input references`);
      }
    }
  }
}

export async function validateModelPackages(roots) {
  if (!Array.isArray(roots) || roots.length === 0) {
    throw new Error('provide at least one model package path');
  }
  const graphFiles = [];
  const failures = [];
  const semanticPackages = [];

  for (const requestedRoot of roots) {
    const root = path.resolve(requestedRoot);
    let primaryGraphCount = 0;
    let files;
    try {
      files = await collectFiles(root);
    } catch (error) {
      failures.push(`${root}: cannot inspect package path (${error.message})`);
      continue;
    }
    const rootGraphCount = graphFiles.length;
    const safetensorsFiles = files.filter((filename) =>
      path.basename(filename).endsWith('.safetensors'));
    const safetensorsCount = safetensorsFiles.length;
    const parsedSafetensors = new Map();
    for (const filename of safetensorsFiles) {
      try {
        parsedSafetensors.set(filename, await readSafetensorsFile(filename));
      } catch (error) {
        failures.push(error.message);
      }
    }
    const safetensorsIndexes = new Map();
    for (const filename of files) {
      const basename = path.basename(filename);
      if (isGraphFilename(basename)) {
        if (basename === 'graph.json') primaryGraphCount++;
        graphFiles.push(filename);
        let document;
        try {
          document = await parseJson(filename);
        } catch (error) {
          failures.push(error.message);
          continue;
        }
        if (!isObject(document)) {
          failures.push(`${filename}: graph document must be a JSON object`);
          continue;
        }
        validateClosedGraphSchema(document, filename, failures);
        const graphSafetensors = nearestSafetensorsFiles(filename, safetensorsFiles);
        semanticPackages.push(Object.freeze({
          graph: filename,
          weights: Object.freeze([...graphSafetensors]),
        }));
        const cacheKey = graphSafetensors.join('\0');
        if (!safetensorsIndexes.has(cacheKey)) {
          try {
            safetensorsIndexes.set(
              cacheKey,
              mergeSafetensorsIndexes(graphSafetensors, parsedSafetensors),
            );
          } catch (error) {
            failures.push(error.message);
            safetensorsIndexes.set(cacheKey, new Map());
          }
        }
        const safetensorsIndex = safetensorsIndexes.get(cacheKey);
        validateWeightBanks(document, safetensorsIndex, filename, failures);
        validateAffineQuantization(document, safetensorsIndex, filename, failures);
        validateTopology(document, safetensorsIndex, filename, failures);
        continue;
      }
      if (!basename.endsWith('.json')) continue;
      let document;
      try {
        document = await parseJson(filename);
      } catch (error) {
        if (error.message.includes('duplicate object key')) failures.push(error.message);
        continue;
      }
      if (!isObject(document)) continue;
      if (document.format === VOLVOX_GRAPH_FORMAT) {
        failures.push(`${filename}: '${VOLVOX_GRAPH_FORMAT}' documents must use graph.json or *.graph.json`);
      }
    }
    if (graphFiles.length === rootGraphCount || primaryGraphCount === 0) {
      failures.push(`${root}: no primary graph.json file found`);
    }
    if (safetensorsCount === 0) {
      failures.push(`${root}: no *.safetensors weight file found`);
    }
  }

  if (failures.length === 0) {
    try {
      await validateRuntimeSemantics(semanticPackages);
    } catch (error) {
      failures.push(error.message);
    }
  }
  if (failures.length > 0) {
    throw new Error(`model package validation failed:\n- ${failures.join('\n- ')}`);
  }
  return Object.freeze([...graphFiles]);
}

async function main() {
  const roots = process.argv.slice(2);
  const graphFiles = await validateModelPackages(roots);
  console.log(`Validated ${graphFiles.length} canonical Volvox graph file(s).`);
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
