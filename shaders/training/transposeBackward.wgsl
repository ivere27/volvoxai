@group(0) @binding(0) var<storage, read> grad_output : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input : array<f32>;
// [rank, total, input_shape[8], perm[8], input_strides[8], output_strides[8]]
@group(0) @binding(2) var<storage, read> params : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let output_index = gid.x;
    let rank = params[0];
    let total = params[1];
    if (output_index >= total || rank > 8u) { return; }
    var remainder = output_index;
    var input_index = 0u;
    for (var d = 0u; d < rank; d = d + 1u) {
        let output_stride = params[26u + d];
        let coordinate = remainder / output_stride;
        remainder = remainder % output_stride;
        let input_dimension = params[10u + d];
        input_index = input_index + coordinate * params[18u + input_dimension];
    }
    grad_input[input_index] = grad_input[input_index] + grad_output[output_index];
}
