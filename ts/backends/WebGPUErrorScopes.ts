import { VolvoxAIError, type RuntimeFailurePhase } from '../core/RuntimeErrors.js';

interface WebGPUErrorScopeDevice {
  pushErrorScope?: GPUDevice['pushErrorScope'];
  popErrorScope?: GPUDevice['popErrorScope'];
}

export interface CapturedWebGPUErrorScopes<T> {
  /** The synchronous API result. It must not be published before `check` settles. */
  readonly value: T;
  /** Rejects when WebGPU reported a scoped validation/OOM or scope-lifecycle failure. */
  readonly check: Promise<void>;
}

export interface WebGPUErrorScopeOptions {
  readonly label: string;
  readonly phase?: RuntimeFailurePhase;
}

function errorDetail(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'object' && error !== null &&
      typeof (error as { message?: unknown }).message === 'string') {
    return (error as { message: string }).message;
  }
  return String(error);
}

function isDeviceLoss(error: unknown): boolean {
  const name = typeof error === 'object' && error !== null
    ? String((error as { name?: unknown }).name || '')
    : '';
  const detail = errorDetail(error);
  return /device.*lost|lost.*device/i.test(`${name} ${detail}`);
}

function scopeLifecycleError(
  error: unknown,
  label: string,
  phase: RuntimeFailurePhase,
): VolvoxAIError {
  const lost = isDeviceLoss(error);
  return new VolvoxAIError(
    lost ? 'DEVICE_LOST' : 'EXECUTION_FAILED',
    `${label} could not resolve its WebGPU error scopes. ${errorDetail(error)}`,
    { phase, backend: 'webgpu', cause: error },
  );
}

function scopedGPUError(
  code: 'OUT_OF_MEMORY' | 'EXECUTION_FAILED',
  error: GPUError,
  label: string,
  phase: RuntimeFailurePhase,
): VolvoxAIError {
  const kind = code === 'OUT_OF_MEMORY' ? 'out of memory' : 'validation failure';
  return new VolvoxAIError(
    code,
    `${label} reported a WebGPU ${kind}. ${errorDetail(error)}`,
    { phase, backend: 'webgpu', cause: error },
  );
}

function callPopErrorScope(device: WebGPUErrorScopeDevice): Promise<GPUError | null> {
  try {
    return Promise.resolve(device.popErrorScope!.call(device));
  } catch (error) {
    return Promise.reject(error);
  }
}

async function inspectScopeResults(
  outOfMemory: Promise<GPUError | null>,
  validation: Promise<GPUError | null>,
  { label, phase = 'execution' }: WebGPUErrorScopeOptions,
): Promise<void> {
  // Attach to both promises at once so a second rejection can never become an
  // unhandled rejection while the first failure is being normalized.
  const [oomResult, validationResult] = await Promise.allSettled([
    outOfMemory,
    validation,
  ]);
  for (const result of [oomResult, validationResult]) {
    if (result.status === 'rejected' && isDeviceLoss(result.reason)) {
      throw scopeLifecycleError(result.reason, label, phase);
    }
  }
  for (const result of [oomResult, validationResult]) {
    if (result.status === 'rejected') {
      throw scopeLifecycleError(result.reason, label, phase);
    }
  }
  if (oomResult.status === 'fulfilled' && oomResult.value !== null) {
    throw scopedGPUError('OUT_OF_MEMORY', oomResult.value, label, phase);
  }
  if (validationResult.status === 'fulfilled' && validationResult.value !== null) {
    throw scopedGPUError('EXECUTION_FAILED', validationResult.value, label, phase);
  }
}

/**
 * Capture errors from one synchronous sequence of WebGPU API calls.
 *
 * Both scopes are pushed and popped before this function returns. Only the
 * returned promises remain pending, so independent contexts cannot interleave
 * a device-global error-scope stack across an `await`. Callers may use `value`
 * internally while staging, but must await `check` before publishing it.
 * Devices in unit tests may omit both scope methods; a partial implementation
 * is rejected because it cannot preserve stack balance.
 */
export function captureWebGPUErrorScopesSync<T>(
  device: GPUDevice,
  options: WebGPUErrorScopeOptions,
  operation: () => T,
): CapturedWebGPUErrorScopes<T> {
  const scopedDevice = device as GPUDevice & WebGPUErrorScopeDevice;
  const hasPush = typeof scopedDevice.pushErrorScope === 'function';
  const hasPop = typeof scopedDevice.popErrorScope === 'function';
  if (!hasPush && !hasPop) {
    return Object.freeze({ value: operation(), check: Promise.resolve() });
  }
  if (!hasPush || !hasPop) {
    throw new VolvoxAIError(
      'EXECUTION_FAILED',
      `${options.label} requires paired WebGPU pushErrorScope/popErrorScope methods.`,
      { phase: options.phase ?? 'execution', backend: 'webgpu' },
    );
  }

  let pushed = 0;
  try {
    // Validation is outer and OOM is inner. Each error reaches its nearest
    // matching filter, and the inner OOM scope is popped first below.
    scopedDevice.pushErrorScope.call(scopedDevice, 'validation');
    pushed++;
    scopedDevice.pushErrorScope.call(scopedDevice, 'out-of-memory');
    pushed++;
  } catch (error) {
    const cleanup: Promise<unknown>[] = [];
    while (pushed-- > 0) cleanup.push(callPopErrorScope(scopedDevice));
    void Promise.allSettled(cleanup);
    throw scopeLifecycleError(error, options.label, options.phase ?? 'execution');
  }

  let value: T;
  try {
    value = operation();
  } catch (error) {
    // Pop calls themselves, not merely their eventual settlement, belong to
    // this synchronous turn. Drain their promises without replacing the
    // operation's direct exception.
    const outOfMemory = callPopErrorScope(scopedDevice);
    const validation = callPopErrorScope(scopedDevice);
    void Promise.allSettled([outOfMemory, validation]);
    throw error;
  }

  const outOfMemory = callPopErrorScope(scopedDevice);
  const validation = callPopErrorScope(scopedDevice);
  return Object.freeze({
    value,
    check: inspectScopeResults(outOfMemory, validation, options),
  });
}

/** Run synchronous GPU API calls and wait for their already-popped scopes. */
export async function runWebGPUErrorScopedSync<T>(
  device: GPUDevice,
  options: WebGPUErrorScopeOptions,
  operation: () => T,
): Promise<T> {
  const captured = captureWebGPUErrorScopesSync(device, options, operation);
  await captured.check;
  return captured.value;
}
