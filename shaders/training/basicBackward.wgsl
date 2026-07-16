@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_a : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_b : array<f32>;

struct Params {
    // Raw layout (all u32):
    // [rank, output_length, a_length, b_length, kind, pad x3,
    //  output_strides[8], a_broadcast_strides[8], b_broadcast_strides[8]]
    values : array<u32>,
}
@group(0) @binding(5) var<storage, read> params : Params;

fn operand_index(output_index : u32, stride_base : u32) -> u32 {
    var remainder = output_index;
    var index = 0u;
    for (var dimension = 0u; dimension < params.values[0]; dimension = dimension + 1u) {
        let output_stride = params.values[8u + dimension];
        let coordinate = remainder / output_stride;
        remainder = remainder % output_stride;
        index = index + coordinate * params.values[stride_base + dimension];
    }
    return index;
}

fn a_derivative(output_index : u32) -> f32 {
    let kind = params.values[4];
    if (kind == 1u) { return b[operand_index(output_index, 24u)]; } // Mul
    if (kind == 3u) { return 1.0 / b[operand_index(output_index, 24u)]; } // Div
    return 1.0; // Add / Sub
}

fn b_derivative(output_index : u32) -> f32 {
    let kind = params.values[4];
    if (kind == 1u) { return a[operand_index(output_index, 16u)]; } // Mul
    if (kind == 2u) { return -1.0; } // Sub
    if (kind == 3u) { // Div
        let av = a[operand_index(output_index, 16u)];
        let bv = b[operand_index(output_index, 24u)];
        return -av / (bv * bv);
    }
    return 1.0; // Add
}

@compute @workgroup_size(64)
fn a_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.values[2]) { return; }
    var sum = 0.0;
    // One invocation owns each destination element. This avoids float atomics
    // while correctly reducing arbitrary NumPy-style broadcast dimensions.
    for (var out_index = 0u; out_index < params.values[1]; out_index = out_index + 1u) {
        if (operand_index(out_index, 16u) == index) {
            sum = sum + grad_output[out_index] * a_derivative(out_index);
        }
    }
    grad_a[index] = grad_a[index] + sum;
}

@compute @workgroup_size(64)
fn b_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.values[3]) { return; }
    var sum = 0.0;
    for (var out_index = 0u; out_index < params.values[1]; out_index = out_index + 1u) {
        if (operand_index(out_index, 24u) == index) {
            sum = sum + grad_output[out_index] * b_derivative(out_index);
        }
    }
    grad_b[index] = grad_b[index] + sum;
}
