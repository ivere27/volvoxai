@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> grad_output : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input : array<f32>;

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
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn global_average_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let count = params.n * params.h * params.w * params.c;
    if (flat >= count) { return; }
    let channel = flat % params.c;
    let batch = flat / (params.h * params.w * params.c);
    grad_input[flat] = grad_input[flat] +
        grad_output[batch * params.c + channel] / f32(params.h * params.w);
}

// Each invocation owns one input gradient and recomputes the first strict
// maximum of every window, avoiding unavailable portable f32 atomics.
@compute @workgroup_size(64)
fn max_pool_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let flat = gid.x;
    let count = params.n * params.h * params.w * params.c;
    if (flat >= count) { return; }
    let batch = flat / (params.h * params.w * params.c);
    let local = flat % (params.h * params.w * params.c);
    let input_base = batch * params.h * params.w * params.c;
    let output_base = batch * params.out_h * params.out_w * params.c;
    var sum = 0.0;
    for (var oy = 0u; oy < params.out_h; oy = oy + 1u) {
        for (var ox = 0u; ox < params.out_w; ox = ox + 1u) {
            var best = -3.402823466e38;
            var best_index = 0xffffffffu;
            for (var yy = 0u; yy < params.ky; yy = yy + 1u) {
                let iy = i32(oy * params.sy + yy) - i32(params.py);
                if (iy < 0 || iy >= i32(params.h)) { continue; }
                for (var xx = 0u; xx < params.kx; xx = xx + 1u) {
                    let ix = i32(ox * params.sx + xx) - i32(params.px);
                    if (ix < 0 || ix >= i32(params.w)) { continue; }
                    let candidate = input_base +
                        (u32(iy) * params.w + u32(ix)) * params.c + local % params.c;
                    let value = input[candidate];
                    if (best_index == 0xffffffffu || value > best) {
                        best = value;
                        best_index = candidate;
                    }
                }
            }
            if (best_index == flat) {
                sum = sum + grad_output[output_base +
                    (oy * params.out_w + ox) * params.c + local % params.c];
            }
        }
    }
    grad_input[flat] = grad_input[flat] + sum;
}
