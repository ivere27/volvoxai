// @volvoxai-browser-only
@group(0) @binding(0) var<storage, read> a : array<i32>;
@group(0) @binding(1) var<storage, read> b : array<i32>;
@group(0) @binding(2) var<storage, read_write> output : array<i32>;
@group(0) @binding(3) var<storage, read> metadata : array<u32>;

// metadata:
// [elements, rank, output_strides[rank], a_strides[rank],
//  b_strides[rank], operation], operation: 0=Equal, 1=GreaterOrEqual.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  let elements = metadata[0];
  let rank = metadata[1];
  if (index >= elements) { return; }
  var remaining = index;
  var a_index = 0u;
  var b_index = 0u;
  for (var axis = 0u; axis < rank; axis = axis + 1u) {
    let stride = metadata[2u + axis];
    let coordinate = remaining / stride;
    remaining = remaining % stride;
    a_index = a_index + coordinate * metadata[2u + rank + axis];
    b_index = b_index + coordinate * metadata[2u + 2u * rank + axis];
  }
  let operation = metadata[2u + 3u * rank];
  let selected = select(a[a_index] >= b[b_index], a[a_index] == b[b_index],
    operation == 0u);
  output[index] = select(0, 1, selected);
}
