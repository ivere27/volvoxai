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

const TILE_C : u32 = 64u;
var<workgroup> tile_weight : array<vec4<f32>, 256>;

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
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(local_invocation_id) lid3 : vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc16_count = params.out_c / 16u;
    let nb = gid.z / oc16_count;
    let oc0 = (gid.z - nb * oc16_count) * 16u;
    let lid = lid3.y * 8u + lid3.x;
    let in_bounds = nb < params.n && ox < params.out_w && oy < params.out_h;

    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);
    let input_base = ((nb * params.in_h + oy) * params.in_w + ox) * params.in_c;

    for (var tile_start = 0u; tile_start < params.in_c; tile_start = tile_start + TILE_C) {
        let tile_count = min(TILE_C, params.in_c - tile_start);
        for (var wi = lid; wi < tile_count * 4u; wi = wi + 64u) {
            let ic = tile_start + wi / 4u;
            let oc_vec = wi - (wi / 4u) * 4u;
            tile_weight[wi] = weight[(ic * params.out_c + oc0) / 4u + oc_vec];
        }
        workgroupBarrier();

        if (in_bounds) {
            for (var ti = 0u; ti < tile_count; ti = ti + 1u) {
                let x = input[input_base + tile_start + ti];
                let wbase = ti * 4u;
                sum0 = sum0 + x * tile_weight[wbase];
                sum1 = sum1 + x * tile_weight[wbase + 1u];
                sum2 = sum2 + x * tile_weight[wbase + 2u];
                sum3 = sum3 + x * tile_weight[wbase + 3u];
            }
        }
        workgroupBarrier();
    }

    if (!in_bounds) { return; }
    let output_base = (((nb * params.out_h + oy) * params.out_w + ox) * params.out_c + oc0) / 4u;
    let bias_base = oc0 / 4u;
    output[output_base] = apply_relu4(sum0 + bias[bias_base]);
    output[output_base + 1u] = apply_relu4(sum1 + bias[bias_base + 1u]);
    output[output_base + 2u] = apply_relu4(sum2 + bias[bias_base + 2u]);
    output[output_base + 3u] = apply_relu4(sum3 + bias[bias_base + 3u]);
}
