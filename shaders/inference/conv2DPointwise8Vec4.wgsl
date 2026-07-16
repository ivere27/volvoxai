@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output : array<vec4<f32>>;

struct Params {
    n : u32,
    in_h : u32,
    in_w : u32,
    in_c : u32,
    out_c : u32,
    out_h : u32,
    out_w : u32,
    kh : u32,
    kw : u32,
    sy : u32,
    sx : u32,
    pt : u32,
    pl : u32,
    groups : u32,
    relu : u32,
    dy : u32,
    dx : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

fn apply_relu4(v_in : vec4<f32>) -> vec4<f32> {
    var v = v_in;
    if (params.relu == 1u) {
        v = max(v, vec4<f32>(0.0));
    } else if (params.relu >= 2u) {
        v = min(max(v, vec4<f32>(0.0)), vec4<f32>(6.0));
    }
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc8_count = (params.out_c + 7u) / 8u;
    let nb = gid.z / oc8_count;
    let oc0 = (gid.z - nb * oc8_count) * 8u;
    if (nb >= params.n || ox >= params.out_w || oy >= params.out_h || oc0 >= params.out_c) { return; }

    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    let input_base = ((nb * params.in_h + oy) * params.in_w + ox) * params.in_c;
    let has_second = oc0 + 4u < params.out_c;
    for (var ic = 0u; ic < params.in_c; ic = ic + 1u) {
        let x = input[input_base + ic];
        let wbase = (ic * params.out_c + oc0) / 4u;
        sum0 = sum0 + x * weight[wbase];
        if (has_second) {
            sum1 = sum1 + x * weight[wbase + 1u];
        }
    }

    let output_base = (((nb * params.out_h + oy) * params.out_w + ox) * params.out_c + oc0) / 4u;
    let bias_base = oc0 / 4u;
    output[output_base] = apply_relu4(sum0 + bias[bias_base]);
    if (has_second) {
        output[output_base + 1u] = apply_relu4(sum1 + bias[bias_base + 1u]);
    }
}
