import type { RuntimeDType, RuntimeTypedArray } from '../types.js';

export const SAFETENSORS_DTYPE_INFO = Object.freeze({
  BOOL: { bits: 8 },
  F4: { bits: 4 },
  F6_E2M3: { bits: 6 },
  F6_E3M2: { bits: 6 },
  U8: { bits: 8 },
  I8: { bits: 8 },
  F8_E5M2: { bits: 8 },
  F8_E4M3: { bits: 8 },
  F8_E8M0: { bits: 8 },
  F8_E4M3FNUZ: { bits: 8 },
  F8_E5M2FNUZ: { bits: 8 },
  I16: { bits: 16 },
  U16: { bits: 16 },
  F16: { bits: 16 },
  BF16: { bits: 16 },
  I32: { bits: 32 },
  U32: { bits: 32 },
  F32: { bits: 32 },
  C64: { bits: 64 },
  F64: { bits: 64 },
  I64: { bits: 64 },
  U64: { bits: 64 },
});

export type SafetensorsDType = keyof typeof SAFETENSORS_DTYPE_INFO;
export type SafetensorsMetadata = Record<string, string>;

export interface SafetensorsTensorInfo {
  dtype: SafetensorsDType;
  shape: number[];
  data_offsets: [number, number];
}

export interface SafetensorsFileOptions {
  metadata?: SafetensorsMetadata;
  hasMetadata?: boolean;
  tensors?: Map<string, SafetensorsTensor>;
  flags?: number;
}

export interface SafetensorsOpenOptions {
  flags?: number;
  writable?: boolean;
}

type ByteSource = ArrayBuffer | ArrayBufferView | ArrayLike<number>;
type RuntimeArray = Int8Array | Uint8Array | Int32Array | Float32Array;
type RuntimeArrayConstructor<T extends RuntimeArray> = {
  readonly BYTES_PER_ELEMENT: number;
  new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): T;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export const SAFETENSORS_TENSOR_READABLE = 1 << 0;
export const SAFETENSORS_TENSOR_WRITABLE = 1 << 1;
export const SAFETENSORS_OPEN_READ_ONLY = 0;
export const SAFETENSORS_OPEN_READ_WRITE = 1 << 0;

export function assertLosslessJSONValue(value: unknown, label = 'JSON'): void {
  const seen = new WeakSet<object>();
  const visit = (item: unknown, path: string): void => {
    if (item == null || typeof item === 'string' || typeof item === 'boolean') return;
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) {
        throw new SyntaxError(`${path} contains a non-finite JSON number.`);
      }
      if (Number.isInteger(item) && !Number.isSafeInteger(item)) {
        throw new SyntaxError(`${path} contains an integer outside JSON's safe range.`);
      }
      return;
    }
    if (typeof item !== 'object') {
      throw new SyntaxError(`${path} contains a non-JSON ${typeof item} value.`);
    }
    if (seen.has(item)) throw new SyntaxError(`${path} contains a JSON cycle.`);
    seen.add(item);
    if (Array.isArray(item)) {
      item.forEach((child, index) => visit(child, `${path}[${index}]`));
    } else {
      for (const [key, child] of Object.entries(item)) {
        visit(child, `${path}.${key}`);
      }
    }
    seen.delete(item);
  };
  visit(value, label);
}

export function parseStrictJSON(source: string, label = 'JSON'): unknown {
  const parsed = JSON.parse(source);
  let cursor = 0;
  const isWhitespace = (character: string): boolean =>
    character === ' ' || character === '\t' || character === '\n' || character === '\r';
  const skipWhitespace = (): void => {
    while (cursor < source.length && isWhitespace(source[cursor])) cursor++;
  };
  const readString = (): string => {
    const start = cursor++;
    while (cursor < source.length) {
      const character = source[cursor++];
      if (character === '\\') {
        cursor++;
      } else if (character === '"') {
        return JSON.parse(source.slice(start, cursor));
      }
    }
    return ''; // The first JSON.parse validated the source, so this is unreachable.
  };
  const scanObject = (): void => {
    cursor++;
    skipWhitespace();
    const keys = new Set<string>();
    if (source[cursor] === '}') {
      cursor++;
      return;
    }
    while (cursor < source.length) {
      skipWhitespace();
      const key = readString();
      if (keys.has(key)) {
        const error = new SyntaxError(`${label} contains duplicate object key ${JSON.stringify(key)}.`) as
          SyntaxError & { code: string };
        error.code = 'ERR_STRICT_JSON_DUPLICATE_KEY';
        throw error;
      }
      keys.add(key);
      skipWhitespace();
      cursor++; // ':'
      scanValue();
      skipWhitespace();
      if (source[cursor] === ',') {
        cursor++;
        continue;
      }
      cursor++; // '}'
      return;
    }
  };
  const scanArray = (): void => {
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
      cursor++; // ']'
      return;
    }
  };
  function scanValue(): void {
    skipWhitespace();
    if (source[cursor] === '{') {
      scanObject();
    } else if (source[cursor] === '[') {
      scanArray();
    } else if (source[cursor] === '"') {
      readString();
    } else {
      while (cursor < source.length &&
             !isWhitespace(source[cursor]) &&
             source[cursor] !== ',' && source[cursor] !== ']' && source[cursor] !== '}') {
        cursor++;
      }
    }
  }
  scanValue();
  assertLosslessJSONValue(parsed, label);
  return parsed;
}

export class SafetensorsTensor {
  file: SafetensorsFile;
  name: string;
  dtype: SafetensorsDType;
  shape: number[];
  dataOffsets: [number, number];
  sizeBytes: number;
  _data: Uint8Array;
  flags: number;

  constructor(file: SafetensorsFile, name: string, info: SafetensorsTensorInfo, data: Uint8Array) {
    this.file = file;
    this.name = name;
    this.dtype = info.dtype;
    this.shape = [...info.shape];
    this.dataOffsets = [...info.data_offsets] as [number, number];
    this.sizeBytes = this.dataOffsets[1] - this.dataOffsets[0];
    this._data = data;
    this.flags = SAFETENSORS_TENSOR_READABLE |
      ((file.flags & SAFETENSORS_OPEN_READ_WRITE) ? SAFETENSORS_TENSOR_WRITABLE : 0);
  }

  getBytes(): Uint8Array {
    return new Uint8Array(this._data);
  }

  getMutableBytes(): Uint8Array {
    if (!(this.flags & SAFETENSORS_TENSOR_WRITABLE)) {
      throw new Error(`Safetensors tensor '${this.name}' was not opened writable.`);
    }
    return this._data;
  }

  setBytes(bytes: ByteSource): void {
    let src: Uint8Array;
    if (bytes instanceof Uint8Array) {
      src = bytes;
    } else if (ArrayBuffer.isView(bytes)) {
      src = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    } else {
      src = new Uint8Array(bytes);
    }
    if (src.byteLength !== this.sizeBytes) {
      throw new Error(`Safetensors tensor '${this.name}' expected ${this.sizeBytes} bytes, got ${src.byteLength}.`);
    }
    this.getMutableBytes().set(src);
  }
}

export class SafetensorsFile {
  metadata: SafetensorsMetadata;
  hasMetadata: boolean;
  tensors: Map<string, SafetensorsTensor>;
  flags: number;

  constructor({
    metadata = {},
    hasMetadata = false,
    tensors = new Map<string, SafetensorsTensor>(),
    flags = SAFETENSORS_OPEN_READ_ONLY,
  }: SafetensorsFileOptions = {}) {
    this.metadata = metadata;
    this.hasMetadata = hasMetadata;
    this.tensors = tensors;
    this.flags = flags;
  }

  static empty(options: SafetensorsFileOptions = {}): SafetensorsFile {
    const flags = options.flags ?? SAFETENSORS_OPEN_READ_WRITE;
    const metadata = Object.prototype.hasOwnProperty.call(options, 'metadata') ? options.metadata : {};
    SafetensorsFile._validateMetadata(metadata);
    return new SafetensorsFile({
      metadata,
      hasMetadata: Object.keys(metadata).length > 0 || options.hasMetadata === true,
      tensors: new Map(),
      flags,
    });
  }

  static fromArrayBuffer(buffer: ArrayBuffer, options: SafetensorsOpenOptions = {}): SafetensorsFile {
    const flags = options.flags ?? (options.writable ? SAFETENSORS_OPEN_READ_WRITE : SAFETENSORS_OPEN_READ_ONLY);
    const dataView = new DataView(buffer);
    const headerLenBig = dataView.getBigUint64(0, true);
    if (headerLenBig > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error("Safetensors header is too large for this runtime.");
    }
    const headerLen = Number(headerLenBig);
    const dataOffset = 8 + headerLen;
    if (dataOffset > buffer.byteLength) throw new Error("Invalid safetensors header length.");

    const headerBytes = new Uint8Array(buffer, 8, headerLen);
    const headerStr = new TextDecoder("utf-8").decode(headerBytes);
    const parsedHeader = parseStrictJSON(headerStr, 'Safetensors header');
    if (!isRecord(parsedHeader)) throw new Error('Safetensors header must be an object.');
    const header = parsedHeader;
    const hasMetadata = Object.prototype.hasOwnProperty.call(header, "__metadata__");
    const metadata = hasMetadata ? header.__metadata__ : {};
    SafetensorsFile._validateMetadata(metadata);
    const file = new SafetensorsFile({
      metadata,
      hasMetadata,
      flags,
    });

    const ordered: SafetensorsTensor[] = [];
    for (const [name, info] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      SafetensorsFile._validateInfo(name, info, buffer.byteLength - dataOffset);
      const [start, end] = info.data_offsets;
      const view = new Uint8Array(buffer, dataOffset + start, end - start);
      const data = (flags & SAFETENSORS_OPEN_READ_WRITE) ? new Uint8Array(view) : view;
      const tensor = new SafetensorsTensor(file, name, info, data);
      file.tensors.set(name, tensor);
      ordered.push(tensor);
    }

    SafetensorsFile._validateCoverage(ordered, buffer.byteLength - dataOffset);
    return file;
  }

  /**
   * Create an independent read-only file view over the same immutable tensor
   * bytes. Metadata, the tensor map, tensor descriptors, shapes, and offsets
   * are copied so graph loaders can safely share one parsed model resource
   * without sharing mutable wrapper state between graphs.
   *
   * The underlying bytes remain shared. Safetensors mutation APIs stay
   * disabled on the clone; runtime typed-array views are therefore treated as
   * immutable model weights by inference graph loaders and executors.
   */
  cloneReadOnly(): SafetensorsFile {
    if (this.flags & SAFETENSORS_OPEN_READ_WRITE) {
      throw new Error("Writable safetensors files cannot be shared as read-only cache entries.");
    }
    SafetensorsFile._validateMetadata(this.metadata);
    const clone = new SafetensorsFile({
      metadata: { ...this.metadata },
      hasMetadata: this.hasMetadata,
      tensors: new Map(),
      flags: SAFETENSORS_OPEN_READ_ONLY,
    });
    for (const [name, tensor] of this.tensors) {
      clone.tensors.set(name, new SafetensorsTensor(clone, name, {
        dtype: tensor.dtype,
        shape: tensor.shape,
        data_offsets: tensor.dataOffsets,
      }, tensor._data));
    }
    return clone;
  }

  getTensor(name: string): SafetensorsTensor | undefined {
    return this.tensors.get(name);
  }

  tensorEntries(): MapIterator<[string, SafetensorsTensor]> {
    return this.tensors.entries();
  }

  listTensorNames(): string[] {
    return [...this.tensors.keys()];
  }

  addTensor(
    name: string,
    dtype: SafetensorsDType,
    shape: readonly number[],
    bytes: ByteSource | null = null,
  ): SafetensorsTensor {
    if (!(this.flags & SAFETENSORS_OPEN_READ_WRITE)) {
      throw new Error("Safetensors file was not opened writable.");
    }
    if (typeof name !== "string" || !name || name === "__metadata__") {
      throw new Error("Safetensors tensor name must be non-empty and cannot be '__metadata__'.");
    }
    if (this.tensors.has(name)) throw new Error(`Safetensors tensor '${name}' already exists.`);
    const sizeBytes = SafetensorsFile.tensorByteLength(dtype, shape, name);
    let data: Uint8Array;
    if (bytes == null) {
      data = new Uint8Array(sizeBytes);
    } else if (bytes instanceof Uint8Array) {
      data = new Uint8Array(bytes);
    } else if (ArrayBuffer.isView(bytes)) {
      data = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      data = new Uint8Array(data);
    } else {
      data = new Uint8Array(bytes);
    }
    if (data.byteLength !== sizeBytes) {
      throw new Error(`Safetensors tensor '${name}' expected ${sizeBytes} bytes, got ${data.byteLength}.`);
    }
    const info: SafetensorsTensorInfo = { dtype, shape: [...shape], data_offsets: [0, sizeBytes] };
    const tensor = new SafetensorsTensor(this, name, info, data);
    this.tensors.set(name, tensor);
    return tensor;
  }

  removeTensor(name: string): boolean {
    if (!(this.flags & SAFETENSORS_OPEN_READ_WRITE)) {
      throw new Error("Safetensors file was not opened writable.");
    }
    return this.tensors.delete(name);
  }

  getMutableTensor(name: string): SafetensorsTensor {
    const tensor = this.getTensor(name);
    if (!tensor) throw new Error(`Safetensors tensor '${name}' not found.`);
    tensor.getMutableBytes();
    return tensor;
  }

  setTensorBytes(name: string, bytes: ByteSource): void {
    this.getMutableTensor(name).setBytes(bytes);
  }

  inspect(): {
    metadata: SafetensorsMetadata;
    tensors: Array<{
      name: string;
      dtype: SafetensorsDType;
      shape: number[];
      dataOffsets: number[];
      sizeBytes: number;
      accessFlags: number;
    }>;
    sizeBytes: number;
    tensorCount: number;
  } {
    SafetensorsFile._validateMetadata(this.metadata);
    let dataBytes = 0;
    const tensors: Array<{
      name: string;
      dtype: SafetensorsDType;
      shape: number[];
      dataOffsets: number[];
      sizeBytes: number;
      accessFlags: number;
    }> = [];
    for (const [name, tensor] of this.tensors.entries()) {
      const dataOffsets = [dataBytes, dataBytes + tensor.sizeBytes];
      tensors.push({
        name,
        dtype: tensor.dtype,
        shape: [...tensor.shape],
        dataOffsets,
        sizeBytes: tensor.sizeBytes,
        accessFlags: tensor.flags,
      });
      dataBytes += tensor.sizeBytes;
    }
    const header: Record<string, SafetensorsMetadata | SafetensorsTensorInfo> = Object.create(null);
    if (this.hasMetadata || (this.metadata && Object.keys(this.metadata).length)) header.__metadata__ = this.metadata;
    let offset = 0;
    for (const [name, tensor] of this.tensors.entries()) {
      header[name] = {
        dtype: tensor.dtype,
        shape: tensor.shape,
        data_offsets: [offset, offset + tensor.sizeBytes],
      };
      offset += tensor.sizeBytes;
    }
    const headerBytes = new TextEncoder().encode(JSON.stringify(header));
    const sizeBytes = 8 + Math.ceil(headerBytes.byteLength / 8) * 8 + dataBytes;
    return {
      metadata: { ...this.metadata },
      tensors,
      sizeBytes,
      tensorCount: tensors.length,
    };
  }

  toArrayBuffer(): ArrayBuffer {
    SafetensorsFile._validateMetadata(this.metadata);
    const tensorEntries = [...this.tensors.entries()];
    const header: Record<string, SafetensorsMetadata | SafetensorsTensorInfo> = Object.create(null);
    if (this.hasMetadata || (this.metadata && Object.keys(this.metadata).length)) header.__metadata__ = this.metadata;

    let offset = 0;
    for (const [name, tensor] of tensorEntries) {
      const start = offset;
      const end = start + tensor.sizeBytes;
      tensor.dataOffsets = [start, end];
      header[name] = {
        dtype: tensor.dtype,
        shape: tensor.shape,
        data_offsets: [start, end],
      };
      offset = end;
    }

    const encoder = new TextEncoder();
    let headerBytes = encoder.encode(JSON.stringify(header));
    const alignedHeaderLength = Math.ceil(headerBytes.byteLength / 8) * 8;
    if (alignedHeaderLength !== headerBytes.byteLength) {
      const padded = new Uint8Array(alignedHeaderLength);
      padded.set(headerBytes);
      padded.fill(0x20, headerBytes.byteLength);
      headerBytes = padded;
    }

    const out = new ArrayBuffer(8 + headerBytes.byteLength + offset);
    const view = new DataView(out);
    view.setBigUint64(0, BigInt(headerBytes.byteLength), true);
    new Uint8Array(out, 8, headerBytes.byteLength).set(headerBytes);
    const data = new Uint8Array(out, 8 + headerBytes.byteLength);
    let cursor = 0;
    for (const [, tensor] of tensorEntries) {
      data.set(tensor.getBytes(), cursor);
      cursor += tensor.sizeBytes;
    }
    return out;
  }

  toBlob(): Blob {
    if (typeof Blob === "undefined") {
      throw new Error("Blob is not available in this runtime.");
    }
    return new Blob([this.toArrayBuffer()], { type: "application/octet-stream" });
  }

  toRuntimeTypedArray(tensor: SafetensorsTensor): RuntimeTypedArray {
    switch (tensor.dtype) {
      case "I8":
        return SafetensorsFile._typedArray(Int8Array, tensor._data, 0, tensor.sizeBytes);
      case "U8":
        return SafetensorsFile._typedArray(Uint8Array, tensor._data, 0, tensor.sizeBytes);
      case "I32":
        return SafetensorsFile._typedArray(Int32Array, tensor._data, 0, tensor.sizeBytes / 4);
      case "F16":
        return SafetensorsFile._float16ToFloat32Array(tensor._data);
      case "F32":
        return SafetensorsFile._typedArray(Float32Array, tensor._data, 0, tensor.sizeBytes / 4);
      default:
        throw new Error(`Unsupported safetensors dtype '${tensor.dtype}' for VolvoxAI runtime tensor '${tensor.name}'.`);
    }
  }

  static toGraphDType(dtype: SafetensorsDType): RuntimeDType {
    if (dtype === "I8") return "int8";
    if (dtype === "U8") return "uint8";
    if (dtype === "I32") return "int32";
    if (dtype === "F16" || dtype === "F32") return "float32";
    throw new Error(`Unsupported safetensors dtype '${dtype}' for VolvoxAI graph weights.`);
  }

  static tensorByteLength(dtype: SafetensorsDType, shape: readonly number[], name = "tensor"): number {
    const dtypeInfo = SAFETENSORS_DTYPE_INFO[dtype];
    if (!dtypeInfo) throw new Error(`Unknown safetensors dtype '${dtype}' for tensor '${name}'.`);
    if (!Array.isArray(shape)) throw new Error(`Invalid safetensors shape for tensor '${name}'.`);
    let elements = 1;
    for (const dim of shape) {
      if (!Number.isInteger(dim) || dim < 0) throw new Error(`Invalid safetensors shape for tensor '${name}'.`);
      elements *= dim;
      if (!Number.isSafeInteger(elements)) throw new Error(`Safetensors tensor '${name}' is too large.`);
    }
    const totalBits = elements * dtypeInfo.bits;
    if (!Number.isSafeInteger(totalBits) || totalBits % 8 !== 0) {
      throw new Error(`Safetensors tensor '${name}' has a non-byte-aligned dtype/shape.`);
    }
    return totalBits / 8;
  }

  static _validateCoverage(tensors: SafetensorsTensor[], dataLength: number): void {
    tensors.sort((a, b) => a.dataOffsets[0] - b.dataOffsets[0] || a.name.localeCompare(b.name));
    let cursor = 0;
    for (const tensor of tensors) {
      if (tensor.dataOffsets[0] !== cursor) {
        throw new Error(`Invalid safetensors tensor offsets near '${tensor.name}'.`);
      }
      cursor = tensor.dataOffsets[1];
    }
    if (cursor !== dataLength) throw new Error("Safetensors metadata does not cover the full file.");
  }

  static _validateMetadata(metadata: unknown): asserts metadata is SafetensorsMetadata {
    if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
      throw new Error("Safetensors __metadata__ must be an object mapping strings to strings.");
    }
    for (const key of Reflect.ownKeys(metadata)) {
      if (typeof key !== "string" || typeof (metadata as Record<PropertyKey, unknown>)[key] !== "string") {
        throw new Error(`Safetensors __metadata__ value for '${String(key)}' must be a string.`);
      }
    }
  }

  static _validateInfo(name: string, info: unknown, dataLength: number): asserts info is SafetensorsTensorInfo {
    const candidate = isRecord(info) ? info : {};
    const dtypeName = candidate.dtype;
    const dtype = typeof dtypeName === 'string' && dtypeName in SAFETENSORS_DTYPE_INFO
      ? SAFETENSORS_DTYPE_INFO[dtypeName as SafetensorsDType]
      : undefined;
    if (!dtype) throw new Error(`Unknown safetensors dtype '${String(dtypeName)}' for tensor '${name}'.`);
    if (!Array.isArray(candidate.shape) || !Array.isArray(candidate.data_offsets) || candidate.data_offsets.length < 2) {
      throw new Error(`Invalid safetensors metadata for tensor '${name}'.`);
    }

    let elements = 1;
    for (const dim of candidate.shape) {
      if (!Number.isInteger(dim) || dim < 0) throw new Error(`Invalid safetensors shape for tensor '${name}'.`);
      elements *= dim;
      if (!Number.isSafeInteger(elements)) throw new Error(`Safetensors tensor '${name}' is too large.`);
    }

    const totalBits = elements * dtype.bits;
    if (!Number.isSafeInteger(totalBits) || totalBits % 8 !== 0) {
      throw new Error(`Safetensors tensor '${name}' has a non-byte-aligned dtype/shape.`);
    }
    const expectedBytes = totalBits / 8;
    const [start, end] = candidate.data_offsets;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start || end > dataLength) {
      throw new Error(`Invalid safetensors data offsets for tensor '${name}'.`);
    }
    if (end - start !== expectedBytes) {
      throw new Error(`Safetensors tensor '${name}' byte length does not match dtype and shape.`);
    }
  }

  static _typedArray<T extends RuntimeArray>(
    ArrayType: RuntimeArrayConstructor<T>,
    bytes: Uint8Array,
    byteOffset: number,
    length: number,
  ): T {
    const sizeBytes = length * ArrayType.BYTES_PER_ELEMENT;
    const offset = bytes.byteOffset + byteOffset;
    if (offset % ArrayType.BYTES_PER_ELEMENT === 0) {
      return new ArrayType(bytes.buffer, offset, length);
    }
    const copy = new Uint8Array(sizeBytes);
    copy.set(new Uint8Array(bytes.buffer, offset, sizeBytes));
    return new ArrayType(copy.buffer);
  }

  static _float16ToFloat32Array(bytes: Uint8Array): Float32Array {
    const n = bytes.byteLength / 2;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = SafetensorsFile._float16BitsToFloat32(view.getUint16(i * 2, true));
    return out;
  }

  static _float16BitsToFloat32(h: number): number {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x03ff;
    if (exp === 0) {
      if (mant === 0) return sign < 0 ? -0 : 0;
      return sign * Math.pow(2, -14) * (mant / 1024);
    }
    if (exp === 31) return mant ? NaN : sign * Infinity;
    return sign * Math.pow(2, exp - 15) * (1 + mant / 1024);
  }
}
