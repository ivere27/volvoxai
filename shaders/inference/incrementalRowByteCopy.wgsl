// @volvoxai-browser-only
// Copy an arbitrary byte range between packed storage buffers. One invocation
// owns each touched destination word, so edge-byte read/modify/write preserves
// neighboring rows without atomics or overlapping writes.
@group(0) @binding(0) var<storage, read> source_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> destination_words : array<u32>;

struct Params {
  source_offset : u32,
  destination_offset : u32,
  size : u32,
  destination_word_count : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn load_byte(index : u32) -> u32 {
  let word = source_words[index / 4u];
  return (word >> ((index % 4u) * 8u)) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.destination_word_count) { return; }
  let destination_word = params.destination_offset / 4u + gid.x;
  let destination_start = params.destination_offset;
  var value = destination_words[destination_word];
  for (var lane = 0u; lane < 4u; lane++) {
    let destination_byte = destination_word * 4u + lane;
    if (destination_byte >= destination_start) {
      let relative_byte = destination_byte - destination_start;
      if (relative_byte < params.size) {
        let source_byte = params.source_offset + relative_byte;
        let shift = lane * 8u;
        let mask = 255u << shift;
        value = (value & ~mask) | (load_byte(source_byte) << shift);
      }
    }
  }
  destination_words[destination_word] = value;
}
