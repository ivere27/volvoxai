import * as pb from '../generated/typescript/volvoxai_lite.js';
import type { ByteCall, CallOptions, Method, Transport } from '../generated/typescript/synurang_runtime.js';
import { PROTO_METHOD_RESPONSES } from '../../ts/generated/protoMethodsFull.js';
import { MemoryEvidenceValidationError, validateMemorySnapshot, validateMemoryBounds } from './MemoryEvidenceValidation.js';

export const DEFAULT_MAX_REPORT_RESPONSE_BYTES = 64 * 1024 * 1024;
const typedArrayPrototype = Object.getPrototypeOf(Uint8Array.prototype) as object;
const typedArrayByteLength = (() => {
  const getter = Object.getOwnPropertyDescriptor(
    typedArrayPrototype,
    'byteLength',
  )?.get;
  if (getter === undefined) {
    throw new Error('TypedArray byteLength intrinsic is unavailable');
  }
  return getter;
})();
const uint8ArraySet = Uint8Array.prototype.set;

function validateReports(value: unknown, visited: Set<object>): void {
  if (value === null || typeof value !== 'object') return;
  if (value instanceof pb.MemorySnapshot) { validateMemorySnapshot(value); return; }
  if (value instanceof pb.MemoryDomainAttestation) { validateMemoryBounds(value); return; }
  if (value instanceof Uint8Array || ArrayBuffer.isView(value)) return;
  if (visited.has(value)) return;
  visited.add(value);

  if (Array.isArray(value)) {
    for (const item of value) validateReports(item, visited);
    return;
  }

  for (const nested of Object.values(value)) validateReports(nested, visited);
}


/** Development transport that validates every schema-declared report. */
export class MemoryEvidenceValidatingTransport implements Transport {
  constructor(private readonly transport: Transport,
      private readonly maxResponseBytes = DEFAULT_MAX_REPORT_RESPONSE_BYTES) {
    if (!Number.isSafeInteger(maxResponseBytes) || maxResponseBytes < 0) {
      throw new RangeError('maxResponseBytes must be a non-negative safe integer');
    }
  }

  async open(method: Method, options?: CallOptions): Promise<ByteCall> {
    const call = await this.transport.open(method, options);
    return {
      send: data => call.send(data),
      halfClose: () => call.halfClose(),
      cancel: code => call.cancel(code),
      close: () => call.close(),
      recv: async () => {
        const response = await call.recv();
        if (response === null) return null;
        const typeName = (PROTO_METHOD_RESPONSES as Readonly<Record<string, string>>)[method.path];
        const codec = (pb as unknown as Readonly<Record<string, {
          readonly fields: readonly { messageType?: string }[];
          fromBinary(data: Uint8Array): object;
        }>>)[typeName];
        if (!codec) return response;
        let size: number;
        try { size = Reflect.apply(typedArrayByteLength, response, []); } catch {
          throw new MemoryEvidenceValidationError('INVALID_RESPONSE', method.path, 'response is not a Uint8Array');
        }
        if (size > this.maxResponseBytes) {
          throw new MemoryEvidenceValidationError('RESPONSE_TOO_LARGE', method.path,
            `encoded response is ${size} bytes; limit is ${this.maxResponseBytes}`);
        }
        const stable = new Uint8Array(size);
        Reflect.apply(uint8ArraySet, stable, [response]);
        let decoded: object;
        try { decoded = codec.fromBinary(stable); } catch {
          throw new MemoryEvidenceValidationError('INVALID_RESPONSE', method.path, 'cannot decode response');
        }
        validateReports(decoded, new Set<object>());
        return stable;
      },
    };
  }
}
