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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let z = gid.z;
    let nb = z / params.out_c;
    let oc = z - nb * params.out_c;
    if (nb >= params.n || ox >= params.out_w || oy >= params.out_h) { return; }

    var sum = 0.0;
    if (params.groups == params.in_c) {
        let mult = params.out_c / params.in_c;
        let ic = oc / mult;
        let m = oc - ic * mult;
        for (var yy = 0u; yy < params.kh; yy = yy + 1u) {
            let iy = i32(oy * params.sy + yy * params.dy) - i32(params.pt);
            if (iy < 0 || iy >= i32(params.in_h)) { continue; }
            for (var xx = 0u; xx < params.kw; xx = xx + 1u) {
                let ix = i32(ox * params.sx + xx * params.dx) - i32(params.pl);
                if (ix < 0 || ix >= i32(params.in_w)) { continue; }
                let ii = ((nb * params.in_h + u32(iy)) * params.in_w + u32(ix)) * params.in_c + ic;
                let wi = (((yy * params.kw + xx) * params.in_c + ic) * mult) + m;
                sum = sum + input[ii] * weight[wi];
            }
        }
    } else {
        let out_per_g = params.out_c / params.groups;
        let in_per_g = params.in_c / params.groups;
        let g = oc / out_per_g;
        let ic0 = g * in_per_g;
        for (var icl = 0u; icl < in_per_g; icl = icl + 1u) {
            let ic = ic0 + icl;
            for (var yy = 0u; yy < params.kh; yy = yy + 1u) {
                let iy = i32(oy * params.sy + yy * params.dy) - i32(params.pt);
                if (iy < 0 || iy >= i32(params.in_h)) { continue; }
                for (var xx = 0u; xx < params.kw; xx = xx + 1u) {
                    let ix = i32(ox * params.sx + xx * params.dx) - i32(params.pl);
                    if (ix < 0 || ix >= i32(params.in_w)) { continue; }
                    let ii = ((nb * params.in_h + u32(iy)) * params.in_w + u32(ix)) * params.in_c + ic;
                    let wi = (((yy * params.kw + xx) * in_per_g + icl) * params.out_c) + oc;
                    sum = sum + input[ii] * weight[wi];
                }
            }
        }
    }

    var v = sum + bias[oc];
    if (params.relu == 1u) {
        v = max(v, 0.0);
    } else if (params.relu >= 2u) {
        v = min(max(v, 0.0), 6.0);
    }
    output[((nb * params.out_h + oy) * params.out_w + ox) * params.out_c + oc] = v;
}
