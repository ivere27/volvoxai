export class Tensor {
  constructor(name, shape, dtype = "float32", isWeight = false) {
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
    this.isWeight = isWeight;
    this.gpuBuffer = null;
    this.sizeBytes = this._calculateByteSize();
  }
  _calculateByteSize() {
    const elements = this.shape.reduce((a, b) => a * b, 1);
    if (this.dtype === "float32" || this.dtype === "int32") return elements * 4;
    if (this.dtype === "int8" || this.dtype === "uint8") return elements;
    return elements * 4;
  }
};

// Normalize a stride/padding param that may be a scalar (e.g. 2), an array
// (e.g. [2, 2]), or missing into a [y, x] pair. KIE emits scalars; the CTC/DET
// exporters emit arrays.
export function _pair(v, def) {
  if (Array.isArray(v)) return [v[0], v[1] ?? v[0]];
  const s = v ?? def;
  return [s, s];
}
