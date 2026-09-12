@group(0) @binding(0) var<storage, read_write> weights : array<f32>;
@group(0) @binding(1) var<storage, read> gradients : array<f32>;
@group(0) @binding(2) var<storage, read_write> first_moment : array<f32>;
@group(0) @binding(3) var<storage, read_write> second_moment : array<f32>;
@group(0) @binding(4) var<storage, read> statistics : array<f32>;
struct Params {
    length : u32, mode : u32, pad0 : u32, pad1 : u32,
    learning_rate : f32, weight_decay : f32, beta1 : f32, beta2 : f32,
    epsilon : f32, correction1 : f32, correction2 : f32, pad2 : f32,
}
@group(0) @binding(5) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.length || statistics[0] != 1.0) { return; }
    let gradient = gradients[i] * statistics[1];
    let weight = weights[i];
    if (params.mode == 0u) {
        weights[i] = weight - params.learning_rate * (gradient + params.weight_decay * weight);
        return;
    }
    let m = params.beta1 * first_moment[i] + (1.0 - params.beta1) * gradient;
    let v = params.beta2 * second_moment[i] + (1.0 - params.beta2) * gradient * gradient;
    first_moment[i] = m;
    second_moment[i] = v;
    weights[i] = weight - params.learning_rate * params.weight_decay * weight -
        params.learning_rate * (m / params.correction1) /
        (sqrt(v / params.correction2) + params.epsilon);
}
