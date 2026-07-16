@group(0) @binding(0) var<storage, read> grad_output : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input : array<f32>;
struct Params { length : u32, _p0 : u32, _p1 : u32, _p2 : u32 }
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i < params.length) { grad_input[i] = grad_input[i] + grad_output[i]; }
}
