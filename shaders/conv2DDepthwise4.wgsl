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
    let c4_count = (params.out_c + 3u) / 4u;
    let nb = gid.z / c4_count;
    let ch0 = (gid.z - nb * c4_count) * 4u;
    if (nb >= params.n || ox >= params.out_w || oy >= params.out_h || ch0 >= params.out_c) { return; }

    let has1 = ch0 + 1u < params.out_c;
    let has2 = ch0 + 2u < params.out_c;
    let has3 = ch0 + 3u < params.out_c;
    var sum0 = 0.0;
    var sum1 = 0.0;
    var sum2 = 0.0;
    var sum3 = 0.0;

    for (var yy = 0u; yy < params.kh; yy = yy + 1u) {
        let iy = i32(oy * params.sy + yy * params.dy) - i32(params.pt);
        if (iy < 0 || iy >= i32(params.in_h)) { continue; }
        for (var xx = 0u; xx < params.kw; xx = xx + 1u) {
            let ix = i32(ox * params.sx + xx * params.dx) - i32(params.pl);
            if (ix < 0 || ix >= i32(params.in_w)) { continue; }
            let input_base = ((nb * params.in_h + u32(iy)) * params.in_w + u32(ix)) * params.in_c + ch0;
            let weight_base = ((yy * params.kw + xx) * params.in_c) + ch0;
            sum0 = sum0 + input[input_base] * weight[weight_base];
            if (has1) { sum1 = sum1 + input[input_base + 1u] * weight[weight_base + 1u]; }
            if (has2) { sum2 = sum2 + input[input_base + 2u] * weight[weight_base + 2u]; }
            if (has3) { sum3 = sum3 + input[input_base + 3u] * weight[weight_base + 3u]; }
        }
    }

    let output_base = ((nb * params.out_h + oy) * params.out_w + ox) * params.out_c + ch0;
    output[output_base] = apply_relu(sum0 + bias[ch0]);
    if (has1) { output[output_base + 1u] = apply_relu(sum1 + bias[ch0 + 1u]); }
    if (has2) { output[output_base + 2u] = apply_relu(sum2 + bias[ch0 + 2u]); }
    if (has3) { output[output_base + 3u] = apply_relu(sum3 + bias[ch0 + 3u]); }
}
