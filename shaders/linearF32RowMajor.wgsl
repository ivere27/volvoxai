@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

struct Params {
    seq_len : u32,
    d_in : u32,
    d_out : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.seq_len || col >= params.d_out) { return; }

    var sum = bias[col];
    for (var k = 0u; k < params.d_in; k = k + 1u) {
        sum = sum + input[row * params.d_in + k] * weight[k * params.d_out + col];
    }
    output[row * params.d_out + col] = sum;
}
