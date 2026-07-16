// Quantized NHWC MaxPool2D. With an unchanged quantization descriptor, raw
// integer order is real-value order, so pooling can retain packed I8/U8 bytes
// without dequantizing. One invocation owns a complete output word.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  n : u32,
  h : u32,
  w : u32,
  c : u32,
  out_h : u32,
  out_w : u32,
  ky : u32,
  kx : u32,
  sy : u32,
  sx : u32,
  py : u32,
  px : u32,
  dtype : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn typed_value(value : u32) -> i32 {
  if (params.dtype == 2u && value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn pool_one(index : u32) -> u32 {
  let elements = params.n * params.out_h * params.out_w * params.c;
  if (index >= elements) { return 0u; }
  let ch = index % params.c;
  let output_spatial = index / params.c;
  let ox = output_spatial % params.out_w;
  let output_row = output_spatial / params.out_w;
  let oy = output_row % params.out_h;
  let batch = output_row / params.out_h;
  var best = select(0u, 128u, params.dtype == 2u);
  for (var yy = 0u; yy < params.ky; yy = yy + 1u) {
    let iy = i32(oy * params.sy + yy) - i32(params.py);
    if (iy < 0 || iy >= i32(params.h)) { continue; }
    for (var xx = 0u; xx < params.kx; xx = xx + 1u) {
      let ix = i32(ox * params.sx + xx) - i32(params.px);
      if (ix < 0 || ix >= i32(params.w)) { continue; }
      let input_index = ((batch * params.h + u32(iy)) * params.w + u32(ix)) * params.c + ch;
      let value = word_byte(input_words[input_index / 4u], input_index);
      if (typed_value(value) > typed_value(best)) { best = value; }
    }
  }
  return best;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_element = output_word * 4u;
  let elements = params.n * params.out_h * params.out_w * params.c;
  if (first_element >= elements) { return; }
  var packed = pool_one(first_element);
  packed = packed | (pool_one(first_element + 1u) << 8u);
  packed = packed | (pool_one(first_element + 2u) << 16u);
  packed = packed | (pool_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
