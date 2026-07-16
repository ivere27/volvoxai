@group(0) @binding(0) var<storage, read> tokens : array<i32>;
@group(0) @binding(1) var<storage, read> grad_output : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_weight : array<f32>;

struct Params {
    token_count : u32,
    d_model : u32,
    vocab_size : u32,
    _pad : u32,
}
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let total = params.vocab_size * params.d_model;
    if (flat >= total) { return; }
    let token_id = flat / params.d_model;
    let feature = flat - token_id * params.d_model;
    var sum = 0.0;
    for (var row = 0u; row < params.token_count; row = row + 1u) {
        if (tokens[row] >= 0 && u32(tokens[row]) == token_id) {
            sum = sum + grad_output[row * params.d_model + feature];
        }
    }
    grad_weight[flat] = grad_weight[flat] + sum;
}
