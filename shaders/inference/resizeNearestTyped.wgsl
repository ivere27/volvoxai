// Quantized NHWC nearest-neighbor resize. It forwards raw bytes rather than
// converting to F32, and each invocation writes one packed destination word.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  n : u32,
  h : u32,
  w : u32,
  c : u32,
  out_h : u32,
  out_w : u32,
  _pad0 : u32,
  _pad1 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn resize_one(index : u32) -> u32 {
  let elements = params.n * params.out_h * params.out_w * params.c;
  if (index >= elements) { return 0u; }
  let ch = index % params.c;
  let output_spatial = index / params.c;
  let ox = output_spatial % params.out_w;
  let output_row = output_spatial / params.out_w;
  let oy = output_row % params.out_h;
  let batch = output_row / params.out_h;
  let iy = (oy * params.h) / params.out_h;
  let ix = (ox * params.w) / params.out_w;
  let input_index = ((batch * params.h + iy) * params.w + ix) * params.c + ch;
  return word_byte(input_words[input_index / 4u], input_index);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_element = output_word * 4u;
  let elements = params.n * params.out_h * params.out_w * params.c;
  if (first_element >= elements) { return; }
  var packed = resize_one(first_element);
  packed = packed | (resize_one(first_element + 1u) << 8u);
  packed = packed | (resize_one(first_element + 2u) << 16u);
  packed = packed | (resize_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
