struct OptimizerParams {
    length : u32,
    mode : u32,
    workgroups_x : u32,
    _padding : u32,
    learning_rate : f32,
    weight_decay : f32,
    beta1 : f32,
    beta2 : f32,
    epsilon : f32,
    correction1 : f32,
    correction2 : f32,
    gradient_scale : f32,
}

@group(0) @binding(0) var<storage, read_write> weights : array<f32>;
@group(0) @binding(1) var<storage, read> gradients : array<f32>;
@group(0) @binding(2) var<storage, read_write> first_moment : array<f32>;
@group(0) @binding(3) var<storage, read_write> second_moment : array<f32>;
@group(0) @binding(4) var<uniform> params : OptimizerParams;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup : vec3<u32>,
    @builtin(local_invocation_id) local : vec3<u32>,
) {
    let linear_workgroup = workgroup.y * params.workgroups_x + workgroup.x;
    let index = linear_workgroup * 64u + local.x;
    if (index >= params.length) { return; }

    let gradient = gradients[index] * params.gradient_scale;
    let weight = weights[index];
    if (params.mode == 0u) {
        weights[index] = weight - params.learning_rate *
            (gradient + params.weight_decay * weight);
        return;
    }

    let m = params.beta1 * first_moment[index] +
        (1.0 - params.beta1) * gradient;
    let v = params.beta2 * second_moment[index] +
        (1.0 - params.beta2) * gradient * gradient;
    first_moment[index] = m;
    second_moment[index] = v;
    let decayed = weight - params.learning_rate * params.weight_decay * weight;
    weights[index] = decayed - params.learning_rate *
        (m / params.correction1) /
        (sqrt(v / params.correction2) + params.epsilon);
}
