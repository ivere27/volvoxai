const UPDATE_MODES = Object.freeze(["assign", "add", "sgd", "adamw"] as const);

export type TrainingUpdateMode = 'sgd' | 'adamw';

export interface TrainingOptimizerOptions {
  learningRate?: number;
  lr?: number;
  weightDecay?: number;
  maxGradNorm?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
  step?: number;
}

export interface TrainingOptimizerValues {
  learningRate: number;
  weightDecay: number;
  maxGradNorm: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
}

export interface TrainingOptimizerDescriptor {
  updateMode: TrainingUpdateMode;
  optimizer: TrainingOptimizerValues;
}

export function normalizeTrainingUpdateMode(value: unknown = "adamw"): TrainingUpdateMode {
  const normalized = typeof value === 'number' && Number.isInteger(value)
    ? UPDATE_MODES[value]
    : typeof value === "string" ? value.toLowerCase() : null;
  if (normalized !== "sgd" && normalized !== "adamw") {
    throw new Error("Training updateMode must be SGD/AdamW (2/3).");
  }
  return normalized;
}

function finite(name: string, value: unknown, predicate: (value: number) => boolean): number {
  if (typeof value !== "number" || !Number.isFinite(value) || !predicate(value)) {
    throw new Error(`Optimizer ${name} is invalid.`);
  }
  return value;
}

/** Return a complete, serializable optimizer descriptor (the step is stored separately). */
export function canonicalOptimizerDescriptor(
  updateMode: unknown,
  optimizer: TrainingOptimizerOptions = {},
): TrainingOptimizerDescriptor {
  const mode = normalizeTrainingUpdateMode(updateMode);
  if (!optimizer || typeof optimizer !== "object" || Array.isArray(optimizer)) {
    throw new Error("Optimizer options must be an object.");
  }
  const learningRate = finite(
    "learningRate",
    optimizer.learningRate ?? optimizer.lr ?? 1e-3,
    (value) => value >= 0,
  );
  const weightDecay = finite("weightDecay", optimizer.weightDecay ?? 0, (value) => value >= 0);
  const maxGradNorm = finite("maxGradNorm", optimizer.maxGradNorm ?? 0, (value) => value >= 0);
  const values: TrainingOptimizerValues = { learningRate, weightDecay, maxGradNorm };
  if (mode === "adamw") {
    values.beta1 = finite("beta1", optimizer.beta1 ?? 0.9, (value) => value >= 0 && value < 1);
    values.beta2 = finite("beta2", optimizer.beta2 ?? 0.999, (value) => value >= 0 && value < 1);
    values.epsilon = finite("epsilon", optimizer.epsilon ?? 1e-8, (value) => value > 0);
  }
  return { updateMode: mode, optimizer: values };
}

/**
 * Validate a caller-supplied optimizer options object where every field is
 * optional (unset fields keep their checkpointed/default value). This is the
 * pre-flight counterpart to `canonicalOptimizerDescriptor`, which fills
 * defaults; here we only reject values that are present and invalid. `mode` is
 * a normalized update mode ("sgd" or "adamw").
 */
export function validateOptimizerOptions(
  optimizer: TrainingOptimizerOptions,
  mode: TrainingUpdateMode,
): void {
  if (!optimizer || typeof optimizer !== "object" || Array.isArray(optimizer)) {
    throw new Error("Optimizer options must be an object.");
  }
  const check = (
    name: string,
    value: unknown,
    predicate: (value: number) => boolean = () => true,
  ) => {
    if (value != null && (typeof value !== "number" || !Number.isFinite(value) || !predicate(value))) {
      throw new Error(`Optimizer ${name} is invalid.`);
    }
  };
  check("learningRate", optimizer.learningRate ?? optimizer.lr, (value) => value >= 0);
  check("weightDecay", optimizer.weightDecay, (value) => value >= 0);
  check("maxGradNorm", optimizer.maxGradNorm, (value) => value >= 0);
  check("step", optimizer.step, (value) => Number.isSafeInteger(value) && value > 0);
  if (mode === "adamw") {
    check("beta1", optimizer.beta1, (value) => value >= 0 && value < 1);
    check("beta2", optimizer.beta2, (value) => value >= 0 && value < 1);
    check("epsilon", optimizer.epsilon, (value) => value > 0);
  }
}

/** Resolve optional per-call overrides against the checkpointed optimizer configuration. */
export function resolveTrainingOptimizer(
  graph: { optimizerDescriptor?: TrainingOptimizerDescriptor | null } | null | undefined,
  updateMode?: unknown,
  optimizer?: TrainingOptimizerOptions | null,
) {
  const saved = graph?.optimizerDescriptor || null;
  const mode = normalizeTrainingUpdateMode(updateMode ?? saved?.updateMode ?? "adamw");
  if (optimizer != null && (typeof optimizer !== "object" || Array.isArray(optimizer))) {
    throw new Error("Optimizer options must be an object.");
  }
  const overrides = optimizer == null ? {} : { ...optimizer };
  if (overrides.learningRate == null && overrides.lr != null) overrides.learningRate = overrides.lr;
  delete overrides.lr;
  const base = saved?.updateMode === mode ? saved.optimizer : {};
  const step = overrides.step;
  delete overrides.step;
  const descriptor = canonicalOptimizerDescriptor(mode, { ...base, ...overrides });
  return {
    ...descriptor,
    optimizer: { ...descriptor.optimizer, ...(step == null ? {} : { step }) },
    descriptor,
  };
}
