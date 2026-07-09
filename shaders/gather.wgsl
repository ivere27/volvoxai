@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
// Gather along axis 0: output[i, ...] = input[indices[i], ...]. `row_size` is the
// number of contiguous elements per gathered row (product of input dims after
// axis 0); `num_idx` is the number of indices. total = num_idx * row_size.
struct Params { row_size : u32, num_idx : u32, total : u32 }
@group(0) @binding(3) var<uniform> p : Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.total) { return; }
  let k = idx % p.row_size;
  let i = idx / p.row_size;
  let row = u32(indices[i]);
  output[idx] = input[row * p.row_size + k];
}
