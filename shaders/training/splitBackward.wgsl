@group(0) @binding(0) var<storage, read> grad_output : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input : array<f32>;
struct Params {
    total : u32,
    inner : u32,
    split_size : u32,
    input_axis : u32,
    axis_offset : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.total) { return; }
    let block = params.split_size * params.inner;
    let outer = index / block;
    let within = index - outer * block;
    let input_index = (outer * params.input_axis + params.axis_offset) * params.inner + within;
    grad_input[input_index] = grad_input[input_index] + grad_output[index];
}
