@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> output : array<f32>;
@group(0) @binding(3) var<storage, read> grad_output : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_weight : array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_bias : array<f32>;

struct Params {
    n : u32,
    in_h : u32,
    in_w : u32,
    in_c : u32,
    out_h : u32,
    out_w : u32,
    out_c : u32,
    kh : u32,
    kw : u32,
    sy : u32,
    sx : u32,
    pt : u32,
    pl : u32,
    groups : u32,
    dy : u32,
    dx : u32,
    relu : u32,
    has_bias : u32,
    weight_count : u32,
    _pad : u32,
}
@group(0) @binding(7) var<uniform> params : Params;

fn activated_gradient(b : u32, oh : u32, ow : u32, oc : u32) -> f32 {
    let offset = ((b * params.out_h + oh) * params.out_w + ow) * params.out_c + oc;
    let y = output[offset];
    if (params.relu == 1u && y <= 0.0) { return 0.0; }
    if (params.relu >= 2u && (y <= 0.0 || y >= 6.0)) { return 0.0; }
    return grad_output[offset];
}

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let count = params.n * params.in_h * params.in_w * params.in_c;
    if (flat >= count) { return; }
    var tmp = flat;
    let ic = tmp % params.in_c; tmp = tmp / params.in_c;
    let iw = tmp % params.in_w; tmp = tmp / params.in_w;
    let ih = tmp % params.in_h; let b = tmp / params.in_h;
    let in_per_group = params.in_c / params.groups;
    let out_per_group = params.out_c / params.groups;
    let group = ic / in_per_group;
    let oc_begin = group * out_per_group;
    let oc_end = oc_begin + out_per_group;
    var sum = 0.0;
    for (var oc = oc_begin; oc < oc_end; oc = oc + 1u) {
        for (var ky = 0u; ky < params.kh; ky = ky + 1u) {
            let oh_num = i32(ih) + i32(params.pt) - i32(ky * params.dy);
            if (oh_num < 0 || oh_num % i32(params.sy) != 0) { continue; }
            let oh = u32(oh_num / i32(params.sy));
            if (oh >= params.out_h) { continue; }
            for (var kx = 0u; kx < params.kw; kx = kx + 1u) {
                let ow_num = i32(iw) + i32(params.pl) - i32(kx * params.dx);
                if (ow_num < 0 || ow_num % i32(params.sx) != 0) { continue; }
                let ow = u32(ow_num / i32(params.sx));
                if (ow >= params.out_w) { continue; }
                var wi = 0u;
                if (params.groups == params.in_c) {
                    let multiplier = params.out_c / params.in_c;
                    let m = oc - ic * multiplier;
                    wi = (((ky * params.kw + kx) * params.in_c + ic) * multiplier) + m;
                } else {
                    let local_ic = ic - group * in_per_group;
                    wi = (((ky * params.kw + kx) * in_per_group + local_ic) * params.out_c) + oc;
                }
                sum = sum + activated_gradient(b, oh, ow, oc) * weight[wi];
            }
        }
    }
    grad_input[flat] = grad_input[flat] + sum;
}

@compute @workgroup_size(64)
fn weight_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    if (flat >= params.weight_count) { return; }
    var ky = 0u; var kx = 0u; var ic = 0u; var oc = 0u;
    if (params.groups == params.in_c) {
        let multiplier = params.out_c / params.in_c;
        var tmp = flat;
        let m = tmp % multiplier; tmp = tmp / multiplier;
        ic = tmp % params.in_c; tmp = tmp / params.in_c;
        kx = tmp % params.kw; ky = tmp / params.kw;
        oc = ic * multiplier + m;
    } else {
        let in_per_group = params.in_c / params.groups;
        let out_per_group = params.out_c / params.groups;
        var tmp = flat;
        oc = tmp % params.out_c; tmp = tmp / params.out_c;
        let local_ic = tmp % in_per_group; tmp = tmp / in_per_group;
        kx = tmp % params.kw; ky = tmp / params.kw;
        ic = (oc / out_per_group) * in_per_group + local_ic;
    }
    var sum = 0.0;
    for (var b = 0u; b < params.n; b = b + 1u) {
        for (var oh = 0u; oh < params.out_h; oh = oh + 1u) {
            let ih = i32(oh * params.sy + ky * params.dy) - i32(params.pt);
            if (ih < 0 || ih >= i32(params.in_h)) { continue; }
            for (var ow = 0u; ow < params.out_w; ow = ow + 1u) {
                let iw = i32(ow * params.sx + kx * params.dx) - i32(params.pl);
                if (iw < 0 || iw >= i32(params.in_w)) { continue; }
                let ii = ((b * params.in_h + u32(ih)) * params.in_w + u32(iw)) * params.in_c + ic;
                sum = sum + input[ii] * activated_gradient(b, oh, ow, oc);
            }
        }
    }
    grad_weight[flat] = grad_weight[flat] + sum;
}

@compute @workgroup_size(64)
fn bias_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let oc = gid.x;
    if (oc >= params.out_c || params.has_bias == 0u) { return; }
    var sum = 0.0;
    for (var b = 0u; b < params.n; b = b + 1u) {
        for (var oh = 0u; oh < params.out_h; oh = oh + 1u) {
            for (var ow = 0u; ow < params.out_w; ow = ow + 1u) {
                sum = sum + activated_gradient(b, oh, ow, oc);
            }
        }
    }
    grad_bias[oc] = grad_bias[oc] + sum;
}
