import type {
  NativeStatus,
  OperationCodeNumber,
  OperationStageNumber,
} from '../generated/volvoxaiEnums.js';

export type VolvoxAIErrorCode = OperationCodeNumber;

export interface VolvoxAIErrorOptions<Report = unknown, Response = unknown> {
  status: NativeStatus;
  stage: OperationStageNumber;
  backend?: string | null;
  node?: string | number | null;
  report?: Report | null;
  response?: Response | null;
  operation?: string | null;
  cause?: unknown;
}

/** Error adapter for the proto report; no separate error or phase vocabulary. */
export class VolvoxAIError<Report = unknown, Response = unknown> extends Error {
  readonly code: OperationCodeNumber;
  readonly status: NativeStatus;
  readonly stage: OperationStageNumber;
  readonly backend: string | null;
  readonly node: string | number | null;
  readonly report: Report | null;
  readonly response: Response | null;
  readonly operation: string | null;

  constructor(code: OperationCodeNumber, message: string, {
    status,
    stage,
    backend = null,
    node = null,
    report = null,
    response = null,
    operation = null,
    cause,
  }: VolvoxAIErrorOptions<Report, Response>) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'VolvoxAIError';
    this.code = code;
    this.status = status;
    this.stage = stage;
    this.backend = backend;
    this.node = node;
    this.report = report;
    this.response = response;
    this.operation = operation;
  }
}
