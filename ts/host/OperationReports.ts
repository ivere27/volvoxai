import {
  NativeStatus,
  OperationCode,
  OperationStage,
  type OperationCodeNumber,
  type OperationStageNumber,
  nativeStatusCodes,
} from '../generated/volvoxaiEnums.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';

/** Structural subset shared by every generated OperationReport projection. */
export interface OperationReportLike {
  readonly status: NativeStatus;
  readonly stage?: OperationStageNumber;
  readonly code?: OperationCodeNumber;
  readonly message?: string;
  readonly backend?: string;
  readonly offendingNode?: { readonly index?: number };
}

/**
 * Adapt an explicitly selected report, retaining the original response.
 * The call transport selects reports by schema, including report-only RPCs.
 */
export function reportError<Response, Report extends OperationReportLike>(
  response: Response,
  report: Report | undefined,
  operation: string,
): VolvoxAIError<Report, Response> | null {
  if (!report) {
    return new VolvoxAIError<Report, Response>(
      OperationCode.Internal,
      `${operation} returned no OperationReport.`,
      { status: NativeStatus.Internal, stage: OperationStage.None, response, operation },
    );
  }
  if (report.status === NativeStatus.Ok) return null;
  const detail = report.message || nativeStatusCodes[-report.status] ||
    `status ${report.status}`;
  return new VolvoxAIError<Report, Response>(
    report.code ?? OperationCode.None,
    `${operation} failed: ${detail}`,
    {
      status: report.status,
      stage: report.stage ?? OperationStage.None,
      backend: report.backend || null,
      node: report.offendingNode?.index ?? null,
      report,
      response,
      operation,
    },
  );
}

/** Internal transport adapter. Generated response metadata selects the report. */
export function assertOperationResponse(
  codecs: Readonly<Record<string, unknown>> & { readonly OperationReport: { readonly typeName: string } },
  responses: Readonly<Record<string, string>>,
  path: string,
  data: Uint8Array,
): void {
  const typeName = responses[path];
  if (!typeName) return;
  const codec = codecs[typeName] as {
    fromBinary(data: Uint8Array): unknown;
    readonly fields: readonly { readonly messageType?: string }[];
  };
  const direct = typeName === 'OperationReport';
  if (!direct && !codec.fields.some(field => field.messageType === codecs.OperationReport.typeName)) return;
  const response = codec.fromBinary(data);
  const report = direct ? response : (response as { report?: OperationReportLike }).report;
  const error = reportError(response, report as OperationReportLike | undefined, path);
  if (error) throw error;
}
