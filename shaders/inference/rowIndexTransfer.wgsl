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
// Rows, not elements, and rows measured in words. That makes the shader
// indifferent to what a row contains: an int8 row of 64 channels and an int4
// row of 128 are both 16 words, and this moves either without knowing which.
// The caller is the one that converts a width and a dtype into `row_words`,
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
@group(0) @binding(2) var<storage, read_write> destination_words : array<u32>;

struct Params {
  // Indexed rows. One invocation per (row, word) pair.
  rows : u32,
  row_words : u32,
  // Bound on an index, so a table that outlived its tensor cannot address past
  // the end of one. Rows are whole and disjoint, so no two invocations write
  // the same word and no atomics are needed.
  indexed_rows : u32,
  // 0: gather -- `row_indices[i]` selects the source row for destination row i.
  // 1: scatter -- `row_indices[i]` selects the destination row for source row i.
  mode : u32,
}
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = params.rows * params.row_words;
  if (gid.x >= total) { return; }
  let row = gid.x / params.row_words;
  let word = gid.x % params.row_words;
  let indexed = row_indices[row];

  if (params.mode == 0u) {
    if (indexed < 0) {
      destination_words[row * params.row_words + word] = 0u;
      return;
    }
    let source_row = u32(indexed);
    if (source_row >= params.indexed_rows) {
      destination_words[row * params.row_words + word] = 0u;
      return;
    }
    destination_words[row * params.row_words + word] =
      source_words[source_row * params.row_words + word];
    return;
  }

  // Scatter. A row with no destination is skipped rather than written
  // anywhere, which is what makes a parked lane cost nothing.
  if (indexed < 0) { return; }
  let destination_row = u32(indexed);
  if (destination_row >= params.indexed_rows) { return; }
  destination_words[destination_row * params.row_words + word] =
    source_words[row * params.row_words + word];
}
