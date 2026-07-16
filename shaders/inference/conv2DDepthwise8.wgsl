@group(0) @binding(0) var<storage, read> input : array<vec4<f32>>;
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
    let c8_count = (params.out_c + 7u) / 8u;
    let nb = gid.z / c8_count;
    let ch0 = (gid.z - nb * c8_count) * 8u;
    if (nb >= params.n || ox >= params.out_w || oy >= params.out_h || ch0 >= params.out_c) { return; }

    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);

    for (var yy = 0u; yy < params.kh; yy = yy + 1u) {
        let iy = i32(oy * params.sy + yy * params.dy) - i32(params.pt);
        if (iy < 0 || iy >= i32(params.in_h)) { continue; }
        for (var xx = 0u; xx < params.kw; xx = xx + 1u) {
            let ix = i32(ox * params.sx + xx * params.dx) - i32(params.pl);
            if (ix < 0 || ix >= i32(params.in_w)) { continue; }
            let input_base = (((nb * params.in_h + u32(iy)) * params.in_w + u32(ix)) * params.in_c + ch0) / 4u;
            let weight_base = (((yy * params.kw + xx) * params.in_c) + ch0) / 4u;
            sum0 = sum0 + input[input_base] * weight[weight_base];
            sum1 = sum1 + input[input_base + 1u] * weight[weight_base + 1u];
        }
    }

    let output_base = (((nb * params.out_h + oy) * params.out_w + ox) * params.out_c + ch0) / 4u;
    let bias_base = ch0 / 4u;
    output[output_base] = apply_relu4(sum0 + bias[bias_base]);
    output[output_base + 1u] = apply_relu4(sum1 + bias[bias_base + 1u]);
}
