// One invocation owns one scan lane, so all axis elements are visited once.
// Accumulation follows source order (or reverse order) with O(N) work and no
// temporary tensor. Raw U32 addition defines I32 overflow modulo 2^32.
@group(0) @binding(0) var<storage, read> input : array<u32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;
struct Params {
    scans : u32, length : u32, inner : u32, exclusive : u32,
    reverse : u32, integer : u32, pad0 : u32, pad1 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) grid : vec3<u32>) {
    let row_start = gid.y * (grid.x * 64u);
    if (row_start >= params.scans || gid.x >= params.scans - row_start) { return; }
    let lane = row_start + gid.x;
    let base = (lane / params.inner) * params.length * params.inner + lane % params.inner;
    var float_sum = 0.0;
    var integer_sum = 0u;
    for (var t = 0u; t < params.length; t = t + 1u) {
        let position = select(t, params.length - 1u - t, params.reverse != 0u);
        let index = base + position * params.inner;
        let value = input[index];
        if (params.integer != 0u) {
            let previous = integer_sum;
            integer_sum = integer_sum + value;
            output[index] = select(integer_sum, previous, params.exclusive != 0u);
        } else {
            let previous = float_sum;
            float_sum = float_sum + bitcast<f32>(value);
            output[index] = bitcast<u32>(select(float_sum, previous, params.exclusive != 0u));
        }
    }
}
