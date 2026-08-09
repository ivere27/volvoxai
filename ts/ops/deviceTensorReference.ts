import type { RuntimeDType } from '../types.js';

/**
 * Nominal marker for a result-owned device tensor accepted as shaped input.
 * The symbol is intentionally not re-exported by a package entry point.
 * Runtime acceptance additionally requires the private registry below, so a
 * structurally similar object or a bare GPUBuffer is never a device input.
 * @internal
 */
export const DEVICE_TENSOR_REFERENCE_BRAND: unique symbol =
  Symbol('volvoxai.device-tensor-reference');

/** Opaque public input storage implemented by a live device TensorResult. */
export interface DeviceTensorReference {
  readonly [DEVICE_TENSOR_REFERENCE_BRAND]: true;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly logicalSizeBytes: number;
}

export interface DeviceTensorReferenceDescriptor {
  readonly kind: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly logicalSizeBytes: number;
}

/** Opaque lease passed through core to a provider without exposing a buffer. */
export interface DeviceTensorInputLease extends DeviceTensorReferenceDescriptor {
  readonly lease: 'volvoxai-device-tensor-input/v1';
}

interface DeviceTensorRegistration extends DeviceTensorReferenceDescriptor {
  readonly owner: object;
  readonly resource: object;
  readonly acquire: () => () => void;
}

interface DeviceTensorLeaseState {
  readonly registration: DeviceTensorRegistration;
  readonly reference: object;
  readonly releaseReference: () => void;
  retirement: Promise<void> | null;
  released: boolean;
}

const REFERENCES = new WeakMap<object, DeviceTensorRegistration>();
const LEASES = new WeakMap<object, DeviceTensorLeaseState>();

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length &&
    left.every((dimension, axis) => dimension === right[axis]);
}

/** Register one provider-created TensorResult as a device-input source. @internal */
export function registerDeviceTensorReference(
  reference: DeviceTensorReference,
  registration: DeviceTensorRegistration,
): void {
  if (!reference || typeof reference !== 'object' || REFERENCES.has(reference) ||
      typeof registration.kind !== 'string' || registration.kind.length === 0 ||
      !registration.owner || typeof registration.owner !== 'object' ||
      !registration.resource || typeof registration.resource !== 'object' ||
      typeof registration.acquire !== 'function' ||
      registration.dtype !== reference.dtype ||
      registration.logicalSizeBytes !== reference.logicalSizeBytes ||
      !sameShape(registration.shape, reference.shape)) {
    throw new TypeError('Invalid VolvoxAI device tensor registration.');
  }
  REFERENCES.set(reference, Object.freeze({
    ...registration,
    shape: Object.freeze([...registration.shape]),
  }));
}

/** Return immutable logical metadata only for an issued device reference. */
export function inspectDeviceTensorReference(
  value: unknown,
): Readonly<DeviceTensorReferenceDescriptor> | null {
  if (!value || typeof value !== 'object') return null;
  const registration = REFERENCES.get(value);
  if (!registration) return null;
  return Object.freeze({
    kind: registration.kind,
    dtype: registration.dtype,
    shape: registration.shape,
    logicalSizeBytes: registration.logicalSizeBytes,
  });
}

/** Identify an opaque acquired lease without resolving its provider resource. */
export function isDeviceTensorInputLease(value: unknown): value is DeviceTensorInputLease {
  return !!value && typeof value === 'object' && LEASES.has(value);
}

/** Acquire result ownership synchronously when an execute call is accepted. @internal */
export function acquireDeviceTensorInputLease(value: unknown): DeviceTensorInputLease {
  if (!value || typeof value !== 'object') {
    throw new TypeError('Device tensor input must be a live VolvoxAI TensorResult.');
  }
  const registration = REFERENCES.get(value);
  if (!registration) {
    throw new TypeError('Device tensor input must be a live VolvoxAI TensorResult.');
  }
  const releaseReference = registration.acquire();
  const lease: DeviceTensorInputLease = Object.freeze({
    lease: 'volvoxai-device-tensor-input/v1',
    kind: registration.kind,
    dtype: registration.dtype,
    shape: registration.shape,
    logicalSizeBytes: registration.logicalSizeBytes,
  });
  LEASES.set(lease, {
    registration,
    reference: value,
    releaseReference,
    retirement: null,
    released: false,
  });
  return lease;
}

/** Verify that an accepted lease still names the caller's current input value. @internal */
export function deviceTensorInputLeaseMatches(
  lease: DeviceTensorInputLease,
  value: unknown,
): boolean {
  const state = LEASES.get(lease);
  return !!state && !state.released && state.reference === value;
}

/**
 * Keep the source result alive until submitted provider work no longer
 * references it. Rejection means the device queue became terminal, so it is
 * also a valid retirement boundary. This must be registered before provider
 * execution settles and core requests lease release.
 * @internal
 */
export function deferDeviceTensorInputLeaseReleaseUntil(
  lease: DeviceTensorInputLease,
  completion: PromiseLike<unknown>,
): void {
  const state = LEASES.get(lease);
  if (!state || state.released) {
    throw new TypeError('Device tensor input lease cannot register a late retirement fence.');
  }
  const settled = Promise.resolve(completion).then(
    () => undefined,
    () => undefined,
  );
  state.retirement = state.retirement === null
    ? settled
    : Promise.all([state.retirement, settled]).then(() => undefined);
}

/** Resolve a lease only for the exact provider kind and physical owner. @internal */
export function resolveDeviceTensorInputResource(
  lease: DeviceTensorInputLease,
  kind: string,
  owner: object,
): object {
  const state = LEASES.get(lease);
  if (!state || state.released) {
    throw new TypeError('Device tensor input lease is invalid or already released.');
  }
  if (state.registration.kind !== kind) {
    throw new TypeError(
      `Device tensor input belongs to '${state.registration.kind}', not '${kind}'.`,
    );
  }
  if (state.registration.owner !== owner) {
    throw new TypeError('Device tensor input belongs to a different physical device.');
  }
  return state.registration.resource;
}

/** Release one accepted execute's source-result ownership. @internal */
export function releaseDeviceTensorInputLease(lease: DeviceTensorInputLease): void {
  const state = LEASES.get(lease);
  if (!state || state.released) return;
  state.released = true;
  const release = () => {
    state.releaseReference();
    LEASES.delete(lease);
  };
  if (state.retirement === null) {
    release();
    return;
  }
  void state.retirement.then(release, release);
}
