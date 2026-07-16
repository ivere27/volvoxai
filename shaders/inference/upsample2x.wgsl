@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Params {
    n : u32,
    h : u32,
    w : u32,
    c : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let z = gid.z;
    let nb = z / params.c;
    let ch = z - nb * params.c;
    let out_h = params.h * 2u;
    let out_w = params.w * 2u;
    if (nb >= params.n || ox >= out_w || oy >= out_h) { return; }
    output[((nb * out_h + oy) * out_w + ox) * params.c + ch] =
        input[((nb * params.h + (oy / 2u)) * params.w + (ox / 2u)) * params.c + ch];
}
