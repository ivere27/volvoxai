@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Params {
    n : u32,
    h : u32,
    w : u32,
    c : u32,
    out_h : u32,
    out_w : u32,
    mode : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let z = gid.z;
    let nb = z / params.c;
    let ch = z - nb * params.c;
    if (nb >= params.n || ox >= params.out_w || oy >= params.out_h) { return; }

    if (params.mode == 0u) {
        let iy = (oy * params.h) / params.out_h;
        let ix = (ox * params.w) / params.out_w;
        output[((nb * params.out_h + oy) * params.out_w + ox) * params.c + ch] =
            input[((nb * params.h + iy) * params.w + ix) * params.c + ch];
        return;
    }

    let scale_y = f32(params.h) / f32(params.out_h);
    let scale_x = f32(params.w) / f32(params.out_w);
    var fy = (f32(oy) + 0.5) * scale_y - 0.5;
    var fx = (f32(ox) + 0.5) * scale_x - 0.5;
    if (fy < 0.0) { fy = 0.0; }
    if (fx < 0.0) { fx = 0.0; }
    let y0 = u32(fy);
    let x0 = u32(fx);
    let y1 = min(y0 + 1u, params.h - 1u);
    let x1 = min(x0 + 1u, params.w - 1u);
    let dy = fy - f32(y0);
    let dx = fx - f32(x0);
    let base = nb * params.h * params.w * params.c;
    let v00 = input[base + (y0 * params.w + x0) * params.c + ch];
    let v01 = input[base + (y0 * params.w + x1) * params.c + ch];
    let v10 = input[base + (y1 * params.w + x0) * params.c + ch];
    let v11 = input[base + (y1 * params.w + x1) * params.c + ch];
    let val = v00 * (1.0 - dy) * (1.0 - dx) + v01 * (1.0 - dy) * dx +
              v10 * dy * (1.0 - dx) + v11 * dy * dx;
    output[((nb * params.out_h + oy) * params.out_w + ox) * params.c + ch] = val;
}
