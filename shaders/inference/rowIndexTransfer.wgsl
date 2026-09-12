// Move whole rows between two packed buffers, choosing them by index.
//
// This is the device half of `vx_decode_row_gather` / `vx_decode_row_scatter`.
// A batched decode step names its rows as a *set* -- lane b writes row
// `b * S + position[b]`, and a paged K/V lane's prefix is whatever physical
// slots its page table holds -- so there is no start-and-count a binding could
// express. On the host those sets are resolved by copying; a decode that keeps
// its activations on the device needs the same copy to happen there, or the
// step reads a stale host mirror of a buffer the device already owns.
//
// Rows, not elements, and rows measured in bytes. That makes the shader
// indifferent to what a row contains: an int8 row of 64 channels and an int4
// row of 128 are both 16 words, and this moves either without knowing which.
// The caller is the one that converts a width and a dtype into `row_bytes`,
// which is where the sub-byte arithmetic belongs and where it is checked.
//
// A negative index means "no source": in a gather it zeroes the destination
// row, which is a lane's padding beyond its own key length; in a scatter it
// skips the row, which is a parked lane holding no request. Zeroing rather
// than leaving the previous contents is deliberate -- padding that still held
// another request's bytes would turn a masking mistake into a cross-request
// leak.
@group(0) @binding(0) var<storage, read> source_words : array<u32>;
@group(0) @binding(1) var<storage, read> row_indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> destination_words : array<atomic<u32>>;

struct Params {
  // Indexed rows. One invocation per (row, word) pair.
  rows : u32,
  row_bytes : u32,
  // Bound on an index, so a table that outlived its tensor cannot address past
  // the end of one. Adjacent byte rows may share a packed word.
  indexed_rows : u32,
  // 0: gather -- `row_indices[i]` selects the source row for destination row i.
  // 1: scatter -- `row_indices[i]` selects the destination row for source row i.
  mode : u32,
}
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = params.rows * params.row_bytes;
  let first = gid.x * 4u;
  if (first >= total) { return; }
  if ((params.row_bytes & 3u) == 0u) {
    let row = first / params.row_bytes;
    let offset = first % params.row_bytes;
    let indexed = row_indices[row];
    let valid = indexed >= 0 && u32(indexed) < params.indexed_rows;
    if (params.mode == 0u) {
      var value = 0u;
      if (valid) { value = source_words[(u32(indexed) * params.row_bytes + offset) / 4u]; }
      atomicStore(&destination_words[gid.x], value);
    } else if (valid) {
      atomicStore(&destination_words[(u32(indexed) * params.row_bytes + offset) / 4u], source_words[gid.x]);
    }
    return;
  }
  // A packed-byte row may start or end inside a word retained by another lane.
  for (var byte = first; byte < min(first + 4u, total); byte++) {
    let row = byte / params.row_bytes;
    let offset = byte % params.row_bytes;
    let indexed = row_indices[row];
    let valid = indexed >= 0 && u32(indexed) < params.indexed_rows;
    if (params.mode != 0u && !valid) { continue; }
    var source = byte;
    var destination = byte;
    if (params.mode == 0u) { source = u32(max(indexed, 0)) * params.row_bytes + offset; }
    else { destination = u32(indexed) * params.row_bytes + offset; }
    var value = 0u;
    if (valid) { value = (source_words[source / 4u] >> ((source & 3u) * 8u)) & 255u; }
    let shift = (destination & 3u) * 8u;
    let mask = 255u << shift;
    loop {
      let old = atomicLoad(&destination_words[destination / 4u]);
      let next = (old & ~mask) | (value << shift);
      if (atomicCompareExchangeWeak(&destination_words[destination / 4u], old, next).exchanged) { break; }
    }
  }
}
