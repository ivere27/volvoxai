@group(0) @binding(0) var<storage, read> grad_output : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input : array<f32>;

struct Params {
    n : u32,
    h : u32,
    w : u32,
    c : u32,
    out_h : u32,
    out_w : u32,
    mode : u32, // 0 nearest, 1 half-pixel bilinear
    _pad : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let count = params.n * params.h * params.w * params.c;
    if (flat >= count) { return; }
    var tmp = flat;
    let channel = tmp % params.c; tmp = tmp / params.c;
    let ix = tmp % params.w; tmp = tmp / params.w;
    let iy = tmp % params.h; let batch = tmp / params.h;
    var sum = 0.0;
    for (var oy = 0u; oy < params.out_h; oy = oy + 1u) {
        for (var ox = 0u; ox < params.out_w; ox = ox + 1u) {
            var coefficient = 0.0;
            if (params.mode == 0u) {
                let source_y = (oy * params.h) / params.out_h;
                let source_x = (ox * params.w) / params.out_w;
                if (source_y == iy && source_x == ix) { coefficient = 1.0; }
            } else {
                let scale_y = f32(params.h) / f32(params.out_h);
                let scale_x = f32(params.w) / f32(params.out_w);
                let fy = max(0.0, (f32(oy) + 0.5) * scale_y - 0.5);
                let fx = max(0.0, (f32(ox) + 0.5) * scale_x - 0.5);
                let y0 = u32(fy);
                let x0 = u32(fx);
                let y1 = min(y0 + 1u, params.h - 1u);
                let x1 = min(x0 + 1u, params.w - 1u);
                let dy = fy - f32(y0);
                let dx = fx - f32(x0);
                if (iy == y0 && ix == x0) { coefficient = coefficient + (1.0 - dy) * (1.0 - dx); }
                if (iy == y0 && ix == x1) { coefficient = coefficient + (1.0 - dy) * dx; }
                if (iy == y1 && ix == x0) { coefficient = coefficient + dy * (1.0 - dx); }
                if (iy == y1 && ix == x1) { coefficient = coefficient + dy * dx; }
            }
            let output_index = ((batch * params.out_h + oy) * params.out_w + ox) * params.c + channel;
            sum = sum + grad_output[output_index] * coefficient;
        }
    }
    grad_input[flat] = grad_input[flat] + sum;
}
