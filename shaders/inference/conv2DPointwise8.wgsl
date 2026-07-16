@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

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

fn apply_relu(v_in : f32) -> f32 {
    var v = v_in;
    if (params.relu == 1u) {
        v = max(v, 0.0);
    } else if (params.relu >= 2u) {
        v = min(max(v, 0.0), 6.0);
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

    let has1 = oc0 + 1u < params.out_c;
    let has2 = oc0 + 2u < params.out_c;
    let has3 = oc0 + 3u < params.out_c;
    let has4 = oc0 + 4u < params.out_c;
    let has5 = oc0 + 5u < params.out_c;
    let has6 = oc0 + 6u < params.out_c;
    let has7 = oc0 + 7u < params.out_c;
    var sum0 = 0.0;
    var sum1 = 0.0;
    var sum2 = 0.0;
    var sum3 = 0.0;
    var sum4 = 0.0;
    var sum5 = 0.0;
    var sum6 = 0.0;
    var sum7 = 0.0;

    let input_base = ((nb * params.in_h + oy) * params.in_w + ox) * params.in_c;
    for (var ic = 0u; ic < params.in_c; ic = ic + 1u) {
        let x = input[input_base + ic];
        let wbase = ic * params.out_c + oc0;
        sum0 = sum0 + x * weight[wbase];
        if (has1) { sum1 = sum1 + x * weight[wbase + 1u]; }
        if (has2) { sum2 = sum2 + x * weight[wbase + 2u]; }
        if (has3) { sum3 = sum3 + x * weight[wbase + 3u]; }
        if (has4) { sum4 = sum4 + x * weight[wbase + 4u]; }
        if (has5) { sum5 = sum5 + x * weight[wbase + 5u]; }
        if (has6) { sum6 = sum6 + x * weight[wbase + 6u]; }
        if (has7) { sum7 = sum7 + x * weight[wbase + 7u]; }
    }

    let output_base = ((nb * params.out_h + oy) * params.out_w + ox) * params.out_c + oc0;
    output[output_base] = apply_relu(sum0 + bias[oc0]);
    if (has1) { output[output_base + 1u] = apply_relu(sum1 + bias[oc0 + 1u]); }
    if (has2) { output[output_base + 2u] = apply_relu(sum2 + bias[oc0 + 2u]); }
    if (has3) { output[output_base + 3u] = apply_relu(sum3 + bias[oc0 + 3u]); }
    if (has4) { output[output_base + 4u] = apply_relu(sum4 + bias[oc0 + 4u]); }
    if (has5) { output[output_base + 5u] = apply_relu(sum5 + bias[oc0 + 5u]); }
    if (has6) { output[output_base + 6u] = apply_relu(sum6 + bias[oc0 + 6u]); }
    if (has7) { output[output_base + 7u] = apply_relu(sum7 + bias[oc0 + 7u]); }
}
