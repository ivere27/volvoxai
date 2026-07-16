// Raw I8/U8 concat copy. Concatenation boundaries are not generally aligned
// to u32 words, so each invocation updates exactly one byte using a CAS loop.
// This preserves all neighboring bytes even when different concat inputs share
// a packed destination word.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<atomic<u32>>;

struct Params {
  size : u32,
  axis_offset : u32,
  input_axis : u32,
  output_axis : u32,
  inner : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn output_index(input_index : u32) -> u32 {
  let inner_index = input_index % params.inner;
  let outer_axis = input_index / params.inner;
  let axis_index = outer_axis % params.input_axis;
  let outer_index = outer_axis / params.input_axis;
  return ((outer_index * params.output_axis + params.axis_offset + axis_index) * params.inner) + inner_index;
}

fn store_byte(index : u32, value : u32) {
  let word_index = index / 4u;
  let shift = (index % 4u) * 8u;
  let mask = 255u << shift;
  var old_value = atomicLoad(&output_words[word_index]);
  loop {
    let replacement = (old_value & ~mask) | ((value & 255u) << shift);
    let result = atomicCompareExchangeWeak(&output_words[word_index], old_value, replacement);
    if (result.exchanged) { break; }
    old_value = result.old_value;
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let input_index = gid.x;
  if (input_index >= params.size) { return; }
  let value = word_byte(input_words[input_index / 4u], input_index);
  store_byte(output_index(input_index), value);
}
