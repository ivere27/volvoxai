@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> adapter_a : array<f32>;
@group(0) @binding(2) var<storage, read> adapter_b : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

struct Params {
    rows : u32,
    d_in : u32,
    d_out : u32,
    rank : u32,
    row_start : u32,
    row_count : u32,
    scale : f32,
}
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let col = gid.x;
    let local_row = gid.y;
    if (col >= params.d_out || local_row >= params.row_count) { return; }
    let row = params.row_start + local_row;
    if (row >= params.rows) { return; }

    var delta = 0.0;
    for (var r = 0u; r < params.rank; r = r + 1u) {
        var projected = 0.0;
        for (var d = 0u; d < params.d_in; d = d + 1u) {
            projected = projected + input[row * params.d_in + d] *
                adapter_a[d * params.rank + r];
        }
        delta = delta + projected * adapter_b[r * params.d_out + col];
    }
    let offset = row * params.d_out + col;
    output[offset] = output[offset] + params.scale * delta;
}
