@group(0) @binding(0) var<storage, read> output : array<f32>;
@group(0) @binding(1) var<storage, read> grad_output : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input : array<f32>;

struct Params {
    rows : u32,
    width : u32,
    log_softmax : u32,
    _pad : u32,
}
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.rows * params.width) { return; }
    let row = flat / params.width;
    let offset = row * params.width;
    var reduction = 0.0;
    if (params.log_softmax == 0u) {
        for (var col = 0u; col < params.width; col = col + 1u) {
            reduction = reduction + grad_output[offset + col] * output[offset + col];
        }
        grad_input[flat] = grad_input[flat] + output[flat] * (grad_output[flat] - reduction);
    } else {
        for (var col = 0u; col < params.width; col = col + 1u) {
            reduction = reduction + grad_output[offset + col];
        }
        grad_input[flat] = grad_input[flat] + grad_output[flat] - exp(output[flat]) * reduction;
    }
}
