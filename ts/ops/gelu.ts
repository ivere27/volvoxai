const SQRT_1_2 = 0.7071067811865476;
const SQRT_2_OVER_PI = 0.7978845608028654;
const INV_SQRT_2PI = 0.3989422804014327;

/** Portable erf used by the exact (non-tanh) GELU path. Maximum error is about 1.5e-7. */
export function erfApprox(value) {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
    0.284496736) * t + 0.254829592) * t;
  return sign * (1 - polynomial * Math.exp(-x * x));
}

export function geluApproximation(node) {
  const value = node?.params?.approximate ?? "none";
  if (value === "none") return "none";
  if (value === "tanh") return "tanh";
  throw new Error(`GELU node ${node?.id ?? "<unnamed>"} approximate must be 'none' or 'tanh'.`);
}

export function geluValue(x, approximation = "none") {
  if (approximation === "tanh") {
    return 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
  }
  return 0.5 * x * (1 + erfApprox(x * SQRT_1_2));
}

export function geluDerivative(x, approximation = "none") {
  if (approximation === "tanh") {
    const x2 = x * x;
    const u = SQRT_2_OVER_PI * (x + 0.044715 * x * x2);
    const t = Math.tanh(u);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * SQRT_2_OVER_PI *
      (1 + 3 * 0.044715 * x2);
  }
  return 0.5 * (1 + erfApprox(x * SQRT_1_2)) + x * Math.exp(-0.5 * x * x) * INV_SQRT_2PI;
}

export function _cpuGELU(node) {

    const inBuf = node.inputs.input.buffer;
    const outBuf = node.outputs.out.buffer;
    const approximation = geluApproximation(node);
    for (let i = 0; i < inBuf.length; i++) {
      outBuf[i] = geluValue(inBuf[i], approximation);
    }
  }
