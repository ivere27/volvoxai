@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weight : array<f32>;

struct Params {
    length : u32,
    channels : u32,
    weight_length : u32,
    _pad : u32,
}
@group(0) @binding(5) var<uniform> params : Params;

fn weight_index(element : u32) -> u32 {
    if (params.weight_length == 1u) { return 0u; }
    return element % params.channels;
}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.length) { return; }
    let derivative = select(weight[weight_index(index)], 1.0, input[index] > 0.0);
    grad_input[index] = grad_input[index] + grad_output[index] * derivative;
}

@compute @workgroup_size(64)
fn weight_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.weight_length) { return; }
    var sum = 0.0;
    for (var element = index; element < params.length; element = element + params.weight_length) {
        if (input[element] <= 0.0) {
            sum = sum + grad_output[element] * input[element];
        }
    }
    grad_weight[index] = grad_weight[index] + sum;
}
