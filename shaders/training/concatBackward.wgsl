@group(0) @binding(0) var<storage, read> grad_output : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input : array<f32>;
struct Params {
    length : u32,
    axis_offset : u32,
    input_axis : u32,
    output_axis : u32,
    inner : u32,
    _p0 : u32,
    _p1 : u32,
    _p2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index < params.length) {
        let inner_index = index % params.inner;
        let outer_axis = index / params.inner;
        let axis_index = outer_axis % params.input_axis;
        let outer_index = outer_axis / params.input_axis;
        let output_index = ((outer_index * params.output_axis + params.axis_offset + axis_index) * params.inner) + inner_index;
        grad_input[index] = grad_input[index] + grad_output[output_index];
    }
}
