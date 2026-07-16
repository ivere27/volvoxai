@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> output : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input : array<f32>;

struct Params {
    length : u32,
    kind : u32,
    alpha : f32,
    beta : f32,
}
@group(0) @binding(4) var<uniform> params : Params;

fn erf_approx(value : f32) -> f32 {
    let sign = select(-1.0, 1.0, value >= 0.0);
    let x = abs(value);
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t);
    return sign * (1.0 - polynomial * exp(-x * x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.length) { return; }
    let x = input[i];
    let y = output[i];
    var derivative = 1.0;
    if (params.kind == 0u) { // ReLU
        derivative = select(0.0, 1.0, x > 0.0);
    } else if (params.kind == 1u) { // erf-based GELU (PyTorch approximate='none')
        derivative = 0.5 * (1.0 + erf_approx(x * 0.7071067811865476))
            + x * exp(-0.5 * x * x) * 0.3989422804014327;
    } else if (params.kind == 9u) { // tanh-approximate GELU
        let c = 0.7978845608;
        let u = c * (x + 0.044715 * x * x * x);
        let t = tanh(u);
        let du = c * (1.0 + 3.0 * 0.044715 * x * x);
        derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
    } else if (params.kind == 2u) { // SiLU / Swish
        let s = 1.0 / (1.0 + exp(-x));
        derivative = s + x * s * (1.0 - s);
    } else if (params.kind == 3u) { // Sigmoid
        derivative = y * (1.0 - y);
    } else if (params.kind == 4u) { // Tanh
        derivative = 1.0 - y * y;
    } else if (params.kind == 5u) { // LeakyReLU
        derivative = select(params.alpha, 1.0, x > 0.0);
    } else if (params.kind == 6u) { // HardSigmoid
        derivative = select(0.0, 1.0 / 6.0, x > -3.0 && x < 3.0);
    } else if (params.kind == 7u) { // HardSwish
        if (x <= -3.0) {
            derivative = 0.0;
        } else if (x >= 3.0) {
            derivative = 1.0;
        } else {
            derivative = x / 3.0 + 0.5;
        }
    } else if (params.kind == 8u) { // Clip (alpha=min, beta=max)
        derivative = select(0.0, 1.0, x > params.alpha && x < params.beta);
    }
    grad_input[i] = grad_input[i] + grad_output[i] * derivative;
}
