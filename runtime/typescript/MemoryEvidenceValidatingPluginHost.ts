import * as pb from '../generated/typescript/volvoxai_lite.js';
import {
  RuntimeServiceMethods,
  type PluginHost,
  type PluginStream,
} from '../generated/typescript/volvoxai_ffi.js';
import {
  MemoryEvidenceValidationError,
  validateOperationReportMemoryEvidence,
} from './MemoryEvidenceValidation.js';

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

/**
 * Unary RuntimeService methods whose response can contain an OperationReport,
 * either directly or through a nested response message.
 */
export const MEMORY_EVIDENCE_REPORT_RESPONSE_METHODS = Object.freeze([
  RuntimeServiceMethods.CreateRuntime,
  RuntimeServiceMethods.CloseRuntime,
  RuntimeServiceMethods.LoadModel,
  RuntimeServiceMethods.GetModelRevision,
  RuntimeServiceMethods.PublishAdapter,
  RuntimeServiceMethods.CreateTrainer,
  RuntimeServiceMethods.CloseTrainer,
  RuntimeServiceMethods.TrainStep,
  RuntimeServiceMethods.CommitTrainer,
  RuntimeServiceMethods.RollbackTrainer,
  RuntimeServiceMethods.ExportTrainerWeights,
  RuntimeServiceMethods.CreatePtqPlan,
  RuntimeServiceMethods.ClosePtqPlan,
  RuntimeServiceMethods.CalibratePtqPlan,
  RuntimeServiceMethods.InspectPtqPlan,
  RuntimeServiceMethods.WritePtqPackage,
  RuntimeServiceMethods.CompileModel,
  RuntimeServiceMethods.GetCompiledModelReport,
  RuntimeServiceMethods.CreateExecutionContext,
  RuntimeServiceMethods.CloseExecutionContext,
  RuntimeServiceMethods.Execute,
  RuntimeServiceMethods.DecodeSeed,
  RuntimeServiceMethods.DecodeStep,
  RuntimeServiceMethods.ResetDecode,
  RuntimeServiceMethods.SelectAdapter,
  RuntimeServiceMethods.RebindAdapter,
] as const);

/** Unary RuntimeService methods whose responses cannot contain OperationReport. */
export const MEMORY_EVIDENCE_NON_REPORT_RESPONSE_METHODS = Object.freeze([
  RuntimeServiceMethods.ReleaseRuntime,
  RuntimeServiceMethods.ReleaseModel,
  RuntimeServiceMethods.ReleaseTrainer,
  RuntimeServiceMethods.ReleasePtqPlan,
  RuntimeServiceMethods.ReleaseCompiledModel,
  RuntimeServiceMethods.ReleaseExecutionContext,
  RuntimeServiceMethods.GetResult,
  RuntimeServiceMethods.ReadOutput,
  RuntimeServiceMethods.ReleaseResult,
] as const);

export type MemoryEvidenceReportResponseMethod =
  (typeof MEMORY_EVIDENCE_REPORT_RESPONSE_METHODS)[number];

interface BinaryMessageDecoder {
  fromBinary(data: Uint8Array): object;
}

export const MEMORY_EVIDENCE_REPORT_RESPONSE_DECODERS: Readonly<
  Record<MemoryEvidenceReportResponseMethod, BinaryMessageDecoder>
> = Object.freeze({
  [RuntimeServiceMethods.CreateRuntime]: pb.RuntimeHandle,
  [RuntimeServiceMethods.CloseRuntime]: pb.OperationReport,
  [RuntimeServiceMethods.LoadModel]: pb.ModelHandle,
  [RuntimeServiceMethods.GetModelRevision]: pb.RevisionInfo,
  [RuntimeServiceMethods.PublishAdapter]: pb.AdapterRevision,
  [RuntimeServiceMethods.CreateTrainer]: pb.TrainerHandle,
  [RuntimeServiceMethods.CloseTrainer]: pb.OperationReport,
  [RuntimeServiceMethods.TrainStep]: pb.TrainStepResult,
  [RuntimeServiceMethods.CommitTrainer]: pb.RevisionInfo,
  [RuntimeServiceMethods.RollbackTrainer]: pb.OperationReport,
  [RuntimeServiceMethods.ExportTrainerWeights]: pb.OperationReport,
  [RuntimeServiceMethods.CreatePtqPlan]: pb.PtqPlanHandle,
  [RuntimeServiceMethods.ClosePtqPlan]: pb.OperationReport,
  [RuntimeServiceMethods.CalibratePtqPlan]: pb.PtqCalibrationInfo,
  [RuntimeServiceMethods.InspectPtqPlan]: pb.PtqPlanInfo,
  [RuntimeServiceMethods.WritePtqPackage]: pb.PtqPackageInfo,
  [RuntimeServiceMethods.CompileModel]: pb.CompiledModelHandle,
  [RuntimeServiceMethods.GetCompiledModelReport]: pb.OperationReport,
  [RuntimeServiceMethods.CreateExecutionContext]: pb.ExecutionContextHandle,
  [RuntimeServiceMethods.CloseExecutionContext]: pb.OperationReport,
  [RuntimeServiceMethods.Execute]: pb.ExecutionResultHandle,
  [RuntimeServiceMethods.DecodeSeed]: pb.ExecutionResultHandle,
  [RuntimeServiceMethods.DecodeStep]: pb.ExecutionResultHandle,
  [RuntimeServiceMethods.ResetDecode]: pb.OperationReport,
  [RuntimeServiceMethods.SelectAdapter]: pb.OperationReport,
  [RuntimeServiceMethods.RebindAdapter]: pb.OperationReport,
});

export interface MemoryEvidenceValidatingPluginHostOptions {
  /** Maximum encoded size of a report-bearing unary response. */
  readonly maxReportResponseBytes?: number;
}

function decoderFor(methodName: string): BinaryMessageDecoder | undefined {
  return (
    MEMORY_EVIDENCE_REPORT_RESPONSE_DECODERS as Readonly<
      Record<string, BinaryMessageDecoder>
    >
  )[methodName];
}

function validateReports(value: unknown, visited: Set<object>): void {
  if (value === null || typeof value !== 'object') return;
  if (value instanceof pb.OperationReport) {
    validateOperationReportMemoryEvidence(value);
    return;
  }
  if (value instanceof Uint8Array || ArrayBuffer.isView(value)) return;
  if (visited.has(value)) return;
  visited.add(value);

  if (Array.isArray(value)) {
    for (const item of value) validateReports(item, visited);
    return;
  }

  for (const nested of Object.values(value)) validateReports(nested, visited);
}

/**
 * Validates memory evidence before RuntimeServiceFfi performs its own decode.
 * A byte-for-byte snapshot of the response is returned so the validation
 * decode and RuntimeServiceFfi consume the same bytes. This also retains
 * unrecognized protobuf fields for downstream forwarding or storage.
 */
export class MemoryEvidenceValidatingPluginHost implements PluginHost {
  private readonly maxReportResponseBytes: number;

  constructor(
    private readonly host: PluginHost,
    options: MemoryEvidenceValidatingPluginHostOptions = {},
  ) {
    const maxReportResponseBytes =
      options.maxReportResponseBytes ?? DEFAULT_MAX_REPORT_RESPONSE_BYTES;
    if (!Number.isSafeInteger(maxReportResponseBytes) || maxReportResponseBytes < 0) {
      throw new RangeError(
        'maxReportResponseBytes must be a non-negative safe integer',
      );
    }
    this.maxReportResponseBytes = maxReportResponseBytes;
  }

  invoke(serviceName: string, methodName: string, data: Uint8Array): Uint8Array {
    const response = this.host.invoke(serviceName, methodName, data);
    const decoder = serviceName === 'RuntimeService' ? decoderFor(methodName) : undefined;
    if (decoder === undefined) return response;

    if (!(response instanceof Uint8Array)) {
      throw new MemoryEvidenceValidationError(
        'INVALID_RESPONSE',
        `RuntimeService response ${methodName}`,
        'host returned a value that is not a Uint8Array',
      );
    }
    let responseByteLength: number;
    try {
      responseByteLength = Reflect.apply(typedArrayByteLength, response, []);
    } catch {
      throw new MemoryEvidenceValidationError(
        'INVALID_RESPONSE',
        `RuntimeService response ${methodName}`,
        'host returned a value without Uint8Array internal storage',
      );
    }
    if (responseByteLength > this.maxReportResponseBytes) {
      throw new MemoryEvidenceValidationError(
        'RESPONSE_TOO_LARGE',
        `RuntimeService response ${methodName}`,
        `encoded response is ${responseByteLength} bytes; limit is ${this.maxReportResponseBytes}`,
      );
    }

    // A host may return a SharedArrayBuffer-backed view. Decode and return one
    // private snapshot so concurrent mutation cannot create a validation/use
    // time-of-check/time-of-use gap.
    const stableResponse = new Uint8Array(responseByteLength);
    Reflect.apply(uint8ArraySet, stableResponse, [response]);

    let decoded: object;
    try {
      decoded = decoder.fromBinary(stableResponse);
    } catch {
      throw new MemoryEvidenceValidationError(
        'INVALID_RESPONSE',
        `RuntimeService response ${methodName}`,
        'failed to decode the report-bearing response',
      );
    }
    validateReports(decoded, new Set<object>());
    return stableResponse;
  }

  openStream(serviceName: string, methodName: string): PluginStream {
    return this.host.openStream(serviceName, methodName);
  }
}

export function createMemoryEvidenceValidatingPluginHost(
  host: PluginHost,
  options: MemoryEvidenceValidatingPluginHostOptions = {},
): PluginHost {
  return new MemoryEvidenceValidatingPluginHost(host, options);
}
