@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

struct Metadata {
    // [length, rank, op, relu, output_strides[8], a_strides[8], b_strides[8]]
    values : array<u32>,
}
@group(0) @binding(3) var<storage, read> metadata : Metadata;

fn operand_index(output_index : u32, stride_base : u32) -> u32 {
    var remainder = output_index;
    var index = 0u;
    for (var dimension = 0u; dimension < metadata.values[1]; dimension = dimension + 1u) {
        let output_stride = metadata.values[4u + dimension];
        let coordinate = remainder / output_stride;
        remainder = remainder % output_stride;
        index = index + coordinate * metadata.values[stride_base + dimension];
    }
    return index;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) groups : vec3<u32>) {
    let output_index = gid.x + gid.y * groups.x * 64u;
    if (output_index >= metadata.values[0]) { return; }
    let av = a[operand_index(output_index, 12u)];
    let bv = b[operand_index(output_index, 20u)];
    let op = metadata.values[2];
    var value = av * bv;
    if (op == 1u) {
        value = av - bv;
    } else if (op == 2u) {
        value = av / bv;
    } else if (op == 3u) {
        value = av + bv;
    }
    if (metadata.values[3] == 1u) { value = max(value, 0.0); }
    else if (metadata.values[3] == 2u) { value = clamp(value, 0.0, 6.0); }
    output[output_index] = value;
}
