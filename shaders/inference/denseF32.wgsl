@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;
struct Params {
    rows : u32, input_width : u32, output_width : u32, transposed : u32,
    has_bias : u32, pad0 : u32, pad1 : u32, pad2 : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let column = gid.x;
    let row = gid.y;
    if (column >= params.output_width || row >= params.rows) { return; }
    var sum = 0.0;
    if (params.has_bias != 0u) { sum = bias[column]; }
    for (var k = 0u; k < params.input_width; k++) {
        let at = select(k * params.output_width + column,
                        column * params.input_width + k, params.transposed != 0u);
        sum += input[row * params.input_width + k] * weight[at];
    }
    output[row * params.output_width + column] = sum;
}
