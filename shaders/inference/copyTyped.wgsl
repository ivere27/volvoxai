// Physical byte copy for I8/U8 shape-only graph nodes. One invocation owns a
// complete packed word, including a harmless padded tail word, so no byte-lane
// write races occur.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  elements : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let word = gid.x;
  if (word * 4u >= params.elements) { return; }
  output_words[word] = input_words[word];
}
