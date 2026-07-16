/** Normalize a scalar, pair, or omitted spatial parameter into a [y, x] pair. */
export function normalizeSpatialPair(value, defaultValue) {
  if (Array.isArray(value)) return [value[0], value[1] ?? value[0]];
  const scalar = value ?? defaultValue;
  return [scalar, scalar];
}
