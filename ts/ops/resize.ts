import { assertRawQuantizedShapeTensors } from './quantizedShape.js';

function falseOrAbsent(value) {
  return value == null || value === false || value === 0;
}

function assertCanonicalTypedNearest(node) {
  const params = node.params || {};
  if (node.opType === 'Resize') {
    if (params.mode !== 'nearest') {
      throw new Error(`Resize node ${node.id || '<unnamed>'} cannot bilinearly interpolate quantized byte storage; use nearest resize or an explicit F32 boundary.`);
    }
  } else if (params.mode != null && params.mode !== 'nearest') {
    throw new Error(`ResizeNearest2D node ${node.id || '<unnamed>'} only supports mode "nearest" for raw I8/U8 storage.`);
  }
  if (params.coordinate_transform_mode != null) {
    throw new Error(`Resize node ${node.id || '<unnamed>'} does not define coordinate_transform_mode; use coordinate_transformation_mode.`);
  }
  if (params.coordinate_transformation_mode != null &&
      params.coordinate_transformation_mode !== 'asymmetric') {
    throw new Error(`Resize node ${node.id || '<unnamed>'} supports raw I8/U8 nearest resize only with coordinate_transformation_mode "asymmetric".`);
  }
  if (params.nearest_mode != null && params.nearest_mode !== 'floor') {
    throw new Error(`Resize node ${node.id || '<unnamed>'} supports raw I8/U8 nearest resize only with nearest_mode "floor".`);
  }
  if (!falseOrAbsent(params.align_corners) || !falseOrAbsent(params.antialias)) {
    throw new Error(`Resize node ${node.id || '<unnamed>'} does not support align_corners or antialias for raw I8/U8 nearest resize.`);
  }
}

export function _cpuResize(node) {
  const inp = node.inputs.input || node.inputs.x || node.inputs.data;
  const out = node.outputs.out || Object.values(node.outputs || {})[0];
  const [b, inH, inW, c] = inp.shape;
  const [, outH, outW] = out.shape;
  const src = inp.buffer;
  const dst = out.buffer;
  const quantized = assertRawQuantizedShapeTensors(node, [inp, out], node.opType);
  if (quantized) {
    if (inp.shape.length !== 4 || out.shape.length !== 4 || inp.shape[0] !== out.shape[0] ||
        inp.shape[3] !== out.shape[3]) {
      throw new Error(`Resize node ${node.id || '<unnamed>'} requires canonical rank-4 NHWC I8/U8 input/output tensors.`);
    }
    assertCanonicalTypedNearest(node);
  }
  if (node.opType === 'ResizeNearest2D' || node.params?.mode === 'nearest') {
    for (let n = 0; n < b; n++) {
      for (let y = 0; y < outH; y++) {
        let iy = Math.floor(y * inH / outH);
        if (iy >= inH) iy = inH - 1;
        for (let x = 0; x < outW; x++) {
          let ix = Math.floor(x * inW / outW);
          if (ix >= inW) ix = inW - 1;
          for (let ch = 0; ch < c; ch++) {
            dst[((n * outH + y) * outW + x) * c + ch] = src[((n * inH + iy) * inW + ix) * c + ch];
          }
        }
      }
    }
    return;
  }

  // Bilinear, half-pixel (align_corners=False) to match PyTorch F.interpolate.
  const sy = inH / outH;
  const sx = inW / outW;
  for (let n = 0; n < b; n++) {
    for (let y = 0; y < outH; y++) {
      let iy = (y + 0.5) * sy - 0.5;
      if (iy < 0) iy = 0;
      const y0 = Math.min(inH - 1, Math.floor(iy));
      const y1 = Math.min(inH - 1, y0 + 1);
      const dy = iy - y0;
      for (let x = 0; x < outW; x++) {
        let ix = (x + 0.5) * sx - 0.5;
        if (ix < 0) ix = 0;
        const x0 = Math.min(inW - 1, Math.floor(ix));
        const x1 = Math.min(inW - 1, x0 + 1);
        const dx = ix - x0;
        for (let ch = 0; ch < c; ch++) {
          const v00 = src[((n * inH + y0) * inW + x0) * c + ch];
          const v01 = src[((n * inH + y0) * inW + x1) * c + ch];
          const v10 = src[((n * inH + y1) * inW + x0) * c + ch];
          const v11 = src[((n * inH + y1) * inW + x1) * c + ch];
          dst[((n * outH + y) * outW + x) * c + ch] =
            v00 * (1 - dy) * (1 - dx) + v01 * (1 - dy) * dx +
            v10 * dy * (1 - dx) + v11 * dy * dx;
        }
      }
    }
  }
}
