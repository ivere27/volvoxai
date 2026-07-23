import type { NativeFailureCode } from '../generated/volvoxaiEnums.js';

export type VolvoxAIErrorCode = NativeFailureCode;

export type RuntimeFailurePhase =
  | 'initialization'
  | 'selection'
  | 'compilation'
  | 'execution'
  | 'readback'
  | 'lifecycle';

export interface VolvoxAIErrorOptions {
  phase: RuntimeFailurePhase;
  backend?: string | null;
  node?: string | number | null;
  report?: unknown;
  cause?: unknown;
}

/** Stable-code failure used by the public runtime lifecycle. */
export class VolvoxAIError extends Error {
  readonly code: VolvoxAIErrorCode;
  readonly phase: RuntimeFailurePhase;
  readonly backend: string | null;
  readonly node: string | number | null;
  readonly report: unknown;

  constructor(code: VolvoxAIErrorCode, message: string, {
    phase,
    backend = null,
    node = null,
    report = null,
    cause,
  }: VolvoxAIErrorOptions) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'VolvoxAIError';
    this.code = code;
    this.phase = phase;
    this.backend = backend;
    this.node = node;
    this.report = report;
  }
}

export function runtimeError(
  error: unknown,
  code: VolvoxAIErrorCode,
  message: string,
  options: VolvoxAIErrorOptions,
): VolvoxAIError {
  if (error instanceof VolvoxAIError) return error;
  const detail = error instanceof Error && error.message ? ` ${error.message}` : '';
  return new VolvoxAIError(code, `${message}${detail}`, { ...options, cause: error });
}
