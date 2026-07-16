@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Params {
    length : u32,
    threshold : u32,
    seed : u32,
    counter : u32,
    scale : f32,
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn random_bits(index : u32) -> u32 {
    var value = params.seed ^ index ^ (params.counter * 0x9e3779b9u);
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let index = gid.x;
    if (index >= params.length) { return; }
    output[index] = select(0.0, input[index] * params.scale,
        random_bits(index) >= params.threshold);
}
