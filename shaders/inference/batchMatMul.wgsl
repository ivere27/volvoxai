// Canonical rank-2..8 ONNX matrix multiplication with broadcast batch axes.
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<storage, read> metadata : array<u32>;

// metadata:
// [batch_rank, M, K, N, output_batch_count,
//  output_batch_strides[batch_rank],
//  a_batch_strides[batch_rank],
//  b_batch_strides[batch_rank]]
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let batch_rank = metadata[0];
  let m = metadata[1];
  let k_width = metadata[2];
  let n = metadata[3];
  let output_batches = metadata[4];
  let row = gid.y;
  let column = gid.x;
  let batch = gid.z;
  if (row >= m || column >= n || batch >= output_batches) { return; }

  var remaining = batch;
  var a_base = 0u;
  var b_base = 0u;
  for (var axis = 0u; axis < batch_rank; axis = axis + 1u) {
    let output_stride = metadata[5u + axis];
    let coordinate = remaining / output_stride;
    remaining = remaining % output_stride;
    a_base = a_base + coordinate * metadata[5u + batch_rank + axis];
    b_base = b_base + coordinate * metadata[5u + 2u * batch_rank + axis];
  }

  var sum = 0.0;
  for (var inner = 0u; inner < k_width; inner = inner + 1u) {
    sum = sum + a[a_base + row * k_width + inner] *
      b[b_base + inner * n + column];
  }
  output[(batch * m + row) * n + column] = sum;
}
