@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read> c : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

struct Params {
    size : u32,
    relu : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.size) { return; }
    var v = a[i] + b[i] + c[i];
    if (params.relu == 1u) {
        v = max(v, 0.0);
    } else if (params.relu >= 2u) {
        v = min(max(v, 0.0), 6.0);
    }
    output[i] = v;
}
