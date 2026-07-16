@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
struct Params {
    size : u32,
    approximate_tanh : u32,
    _pad0 : u32,
    _pad1 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn erf_approx(value : f32) -> f32 {
    let sign = select(-1.0, 1.0, value >= 0.0);
    let x = abs(value);
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t);
    return sign * (1.0 - polynomial * exp(-x * x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) { return; }
    let x = input[idx];
    var cdf = 0.5 * (1.0 + erf_approx(x * 0.7071067811865476));
    if (params.approximate_tanh != 0u) {
        cdf = 0.5 * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    }
    output[idx] = x * cdf;
}
