import { assertRawQuantizedShapeTensors } from './quantizedShape.js';

function pairParameter(value, fallback) {
  const values = Array.isArray(value) ? value : [value ?? fallback];
  return [values[0], values[1] ?? values[0]];
}

function falseOrAbsent(value) {
  return value == null || value === false || value === 0;
}

// MaxPool preserves integer ordering when its I8/U8 descriptor is unchanged.
// Keep that typed path deliberately narrow so CPU, WASM, and WebGPU all use
// the same floor-window, unit-dilation NHWC semantics.
export function _cpuMaxPool2D(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const [n, h, w, c] = input.shape;
  const params = node.params || {};
  const [ky, kx] = pairParameter(params.kernel, 1);
  const [sy, sx] = pairParameter(params.stride, 1);
  const [defaultPadY, defaultPadX] = pairParameter(params.padding, 0);
  let padY = defaultPadY;
  let padX = defaultPadX;
  const outH = output.shape[1];
  const outW = output.shape[2];
  const inBuf = input.buffer;
  const outBuf = output.buffer;
  const quantized = assertRawQuantizedShapeTensors(node, [input, output], 'MaxPool2D');

  if (quantized) {
    if (!falseOrAbsent(params.ceil_mode)) {
      throw new Error(`MaxPool2D node ${node.id || '<unnamed>'} does not support ceil_mode for raw I8/U8 storage.`);
    }
    if (params.kernel == null || ![ky, kx, sy, sx].every((value) => Number.isInteger(value) && value > 0) ||
        (params.padding != null && ![defaultPadY, defaultPadX].every((value) => Number.isInteger(value) && value >= 0))) {
      throw new Error(`MaxPool2D node ${node.id || '<unnamed>'} requires positive kernel/stride and non-negative padding parameters for raw I8/U8 storage.`);
    }
    const [dilationY, dilationX] = pairParameter(params.dilation, 1);
    if (![dilationY, dilationX].every((value) => Number.isInteger(value) && value === 1)) {
      throw new Error(`MaxPool2D node ${node.id || '<unnamed>'} supports raw I8/U8 storage only with unit dilation.`);
    }
    const pads = params.pads ?? [defaultPadY, defaultPadX, defaultPadY, defaultPadX];
    if (!Array.isArray(pads) || pads.length !== 4 ||
        !pads.every((value) => Number.isInteger(value) && value >= 0) ||
        input.shape.length !== 4 || output.shape.length !== 4 ||
        input.shape[0] !== output.shape[0] || input.shape[3] !== output.shape[3]) {
      throw new Error(`MaxPool2D node ${node.id || '<unnamed>'} requires canonical rank-4 NHWC I8/U8 tensors and non-negative top/left/bottom/right pads.`);
    }
    const expectedHeight = Math.floor((h + pads[0] + pads[2] - ky) / sy) + 1;
    const expectedWidth = Math.floor((w + pads[1] + pads[3] - kx) / sx) + 1;
    if (outH !== expectedHeight || outW !== expectedWidth) {
      throw new Error(`MaxPool2D node ${node.id || '<unnamed>'} output shape is incompatible with canonical raw I8/U8 parameters.`);
    }
    [padY, padX] = pads;
  }

  for (let b = 0; b < n; b++) {
    for (let oy = 0; oy < outH; oy++) {
      for (let ox = 0; ox < outW; ox++) {
        for (let ch = 0; ch < c; ch++) {
          let best = quantized ? (input.dtype === 'int8' ? -128 : 0) : -Infinity;
          for (let dy = 0; dy < ky; dy++) {
            for (let dx = 0; dx < kx; dx++) {
              const ih = oy * sy + dy - padY;
              const iw = ox * sx + dx - padX;
              if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                const value = inBuf[((b * h + ih) * w + iw) * c + ch];
                if (value > best) best = value;
              }
            }
          }
          outBuf[((b * outH + oy) * outW + ox) * c + ch] = best;
        }
      }
    }
  }
}
