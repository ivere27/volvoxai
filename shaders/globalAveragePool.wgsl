@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Params { n : u32, h : u32, w : u32, c : u32 }
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ch = gid.x;
    let nb = gid.y;
    if (nb >= params.n || ch >= params.c) { return; }
    var sum = 0.0;
    for (var y = 0u; y < params.h; y = y + 1u) {
        for (var x = 0u; x < params.w; x = x + 1u) {
            sum = sum + input[((nb * params.h + y) * params.w + x) * params.c + ch];
        }
    }
    output[nb * params.c + ch] = sum / f32(params.h * params.w);
}
