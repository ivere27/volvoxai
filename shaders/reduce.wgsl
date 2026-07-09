@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
// Reduce over the last (innermost) axis: `b` rows of length `d` -> `b` outputs.
// `inv` scales the sum: 1.0 for ReduceSum, 1.0/d for ReduceMean.
struct Params { b : u32, d : u32, inv : f32 }
@group(0) @binding(2) var<uniform> p : Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.b) { return; }
  let offset = i * p.d;
  var sum = 0.0;
  for (var j = 0u; j < p.d; j = j + 1u) {
    sum = sum + input[offset + j];
  }
  output[i] = sum * p.inv;
}
