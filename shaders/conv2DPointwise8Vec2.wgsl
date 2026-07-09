@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> bias : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output : array<vec2<f32>>;

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

fn apply_relu2(v_in : vec2<f32>) -> vec2<f32> {
    var v = v_in;
    if (params.relu == 1u) {
        v = max(v, vec2<f32>(0.0));
    } else if (params.relu >= 2u) {
        v = min(max(v, vec2<f32>(0.0)), vec2<f32>(6.0));
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

    let has1 = oc0 + 2u < params.out_c;
    let has2 = oc0 + 4u < params.out_c;
    let has3 = oc0 + 6u < params.out_c;
    var sum0 = vec2<f32>(0.0);
    var sum1 = vec2<f32>(0.0);
    var sum2 = vec2<f32>(0.0);
    var sum3 = vec2<f32>(0.0);

    let input_base = ((nb * params.in_h + oy) * params.in_w + ox) * params.in_c;
    for (var ic = 0u; ic < params.in_c; ic = ic + 1u) {
        let x = input[input_base + ic];
        let wbase = (ic * params.out_c + oc0) / 2u;
        sum0 = sum0 + x * weight[wbase];
        if (has1) { sum1 = sum1 + x * weight[wbase + 1u]; }
        if (has2) { sum2 = sum2 + x * weight[wbase + 2u]; }
        if (has3) { sum3 = sum3 + x * weight[wbase + 3u]; }
    }

    let output_base = (((nb * params.out_h + oy) * params.out_w + ox) * params.out_c + oc0) / 2u;
    let bias_base = oc0 / 2u;
    output[output_base] = apply_relu2(sum0 + bias[bias_base]);
    if (has1) { output[output_base + 1u] = apply_relu2(sum1 + bias[bias_base + 1u]); }
    if (has2) { output[output_base + 2u] = apply_relu2(sum2 + bias[bias_base + 2u]); }
    if (has3) { output[output_base + 3u] = apply_relu2(sum3 + bias[bias_base + 3u]); }
}
