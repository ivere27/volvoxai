@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Params {
    n : u32,
    h : u32,
    w : u32,
    c : u32,
    out_h : u32,
    out_w : u32,
    ky : u32,
    kx : u32,
    sy : u32,
    sx : u32,
    py : u32,
    px : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let batch = gid.z / params.c;
    let ch = gid.z % params.c;
    if (ox >= params.out_w || oy >= params.out_h || batch >= params.n) { return; }
    let input_base = batch * params.h * params.w * params.c;
    let output_base = batch * params.out_h * params.out_w * params.c;

    var best = -3.402823466e38;
    for (var yy = 0u; yy < params.ky; yy = yy + 1u) {
        let iy = i32(oy * params.sy + yy) - i32(params.py);
        if (iy < 0 || iy >= i32(params.h)) { continue; }
        for (var xx = 0u; xx < params.kx; xx = xx + 1u) {
            let ix = i32(ox * params.sx + xx) - i32(params.px);
            if (ix < 0 || ix >= i32(params.w)) { continue; }
            best = max(best, input[input_base + (u32(iy) * params.w + u32(ix)) * params.c + ch]);
        }
    }
    output[output_base + (oy * params.out_w + ox) * params.c + ch] = best;
}
