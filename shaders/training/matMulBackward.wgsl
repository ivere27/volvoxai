@group(0) @binding(0) var<storage, read> input : array<f32>;
// WebGPU forward normalizes every F32 linear weight to [d_out, d_in]. Native
// training may retain either layout and selects it through weight_din_layout.
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> grad_output : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weight : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_bias : array<f32>;

struct Params {
    rows : u32,
    d_in : u32,
    d_out : u32,
    has_bias : u32,
    canonical_din_layout : u32,
    weight_din_layout : u32,
    _pad1 : u32,
    _pad2 : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

// input_main uses these as [row,reduction] and [reduction,d_in].
// weight_main reuses them as [d_in,reduction] and [reduction,d_out].
var<workgroup> left_tile : array<f32, 256>;
var<workgroup> right_tile : array<f32, 256>;

// dX = dY * W. Each 8x8 workgroup computes a 16x16 [row,d_in]
// tile, with four accumulators per invocation and d_out tiled by 16.
@compute @workgroup_size(8, 8, 1)
fn input_main(@builtin(workgroup_id) group_id : vec3<u32>,
              @builtin(local_invocation_id) local_id : vec3<u32>) {
    let d_base = group_id.x * 16u;
    let row_base = group_id.y * 16u;
    let lane = local_id.y * 8u + local_id.x;
    var sum00 = 0.0;
    var sum01 = 0.0;
    var sum10 = 0.0;
    var sum11 = 0.0;

    for (var column_base = 0u; column_base < params.d_out; column_base = column_base + 16u) {
        for (var offset = lane; offset < 256u; offset = offset + 64u) {
            let tile_row = offset / 16u;
            let tile_column = offset - tile_row * 16u;

            let row = row_base + tile_row;
            let column = column_base + tile_column;
            var grad_value = 0.0;
            if (row < params.rows && column < params.d_out) {
                grad_value = grad_output[row * params.d_out + column];
            }
            left_tile[offset] = grad_value;

            let weight_column = column_base + tile_row;
            let d = d_base + tile_column;
            var weight_value = 0.0;
            if (weight_column < params.d_out && d < params.d_in) {
                let weight_index = select(weight_column * params.d_in + d,
                    d * params.d_out + weight_column, params.weight_din_layout != 0u);
                weight_value = weight[weight_index];
            }
            right_tile[offset] = weight_value;
        }
        workgroupBarrier();

        for (var column = 0u; column < 16u; column = column + 1u) {
            let left0 = left_tile[local_id.y * 16u + column];
            let left1 = left_tile[(local_id.y + 8u) * 16u + column];
            let right0 = right_tile[column * 16u + local_id.x];
            let right1 = right_tile[column * 16u + local_id.x + 8u];
            sum00 = sum00 + left0 * right0;
            sum01 = sum01 + left0 * right1;
            sum10 = sum10 + left1 * right0;
            sum11 = sum11 + left1 * right1;
        }
        workgroupBarrier();
    }

    let row0 = row_base + local_id.y;
    let row1 = row0 + 8u;
    let d0 = d_base + local_id.x;
    let d1 = d0 + 8u;
    if (row0 < params.rows && d0 < params.d_in) {
        let index = row0 * params.d_in + d0;
        grad_input[index] = grad_input[index] + sum00;
    }
    if (row0 < params.rows && d1 < params.d_in) {
        let index = row0 * params.d_in + d1;
        grad_input[index] = grad_input[index] + sum01;
    }
    if (row1 < params.rows && d0 < params.d_in) {
        let index = row1 * params.d_in + d0;
        grad_input[index] = grad_input[index] + sum10;
    }
    if (row1 < params.rows && d1 < params.d_in) {
        let index = row1 * params.d_in + d1;
        grad_input[index] = grad_input[index] + sum11;
    }
}

// dW = X^T * dY. Each workgroup computes a 16x16 [d_in,d_out]
// tile and walks the row reduction through cooperative 16-row tiles.
@compute @workgroup_size(8, 8, 1)
fn weight_main(@builtin(workgroup_id) group_id : vec3<u32>,
               @builtin(local_invocation_id) local_id : vec3<u32>) {
    let column_base = group_id.x * 16u;
    let d_base = group_id.y * 16u;
    let lane = local_id.y * 8u + local_id.x;
    var sum00 = 0.0;
    var sum01 = 0.0;
    var sum10 = 0.0;
    var sum11 = 0.0;

    for (var row_base = 0u; row_base < params.rows; row_base = row_base + 16u) {
        for (var offset = lane; offset < 256u; offset = offset + 64u) {
            let tile_row = offset / 16u;
            let tile_column = offset - tile_row * 16u;

            let d = d_base + tile_row;
            let input_row = row_base + tile_column;
            var input_value = 0.0;
            if (d < params.d_in && input_row < params.rows) {
                input_value = input[input_row * params.d_in + d];
            }
            left_tile[offset] = input_value;

            let grad_row = row_base + tile_row;
            let column = column_base + tile_column;
            var grad_value = 0.0;
            if (grad_row < params.rows && column < params.d_out) {
                grad_value = grad_output[grad_row * params.d_out + column];
            }
            right_tile[offset] = grad_value;
        }
        workgroupBarrier();

        for (var row = 0u; row < 16u; row = row + 1u) {
            let left0 = left_tile[local_id.y * 16u + row];
            let left1 = left_tile[(local_id.y + 8u) * 16u + row];
            let right0 = right_tile[row * 16u + local_id.x];
            let right1 = right_tile[row * 16u + local_id.x + 8u];
            sum00 = sum00 + left0 * right0;
            sum01 = sum01 + left0 * right1;
            sum10 = sum10 + left1 * right0;
            sum11 = sum11 + left1 * right1;
        }
        workgroupBarrier();
    }

    let d0 = d_base + local_id.y;
    let d1 = d0 + 8u;
    let column0 = column_base + local_id.x;
    let column1 = column0 + 8u;
    if (d0 < params.d_in && column0 < params.d_out) {
        let canonical = column0 * params.d_in + d0;
        let destination = select(canonical, d0 * params.d_out + column0,
            params.canonical_din_layout != 0u);
        grad_weight[destination] = grad_weight[destination] + sum00;
    }
    if (d0 < params.d_in && column1 < params.d_out) {
        let canonical = column1 * params.d_in + d0;
        let destination = select(canonical, d0 * params.d_out + column1,
            params.canonical_din_layout != 0u);
        grad_weight[destination] = grad_weight[destination] + sum01;
    }
    if (d1 < params.d_in && column0 < params.d_out) {
        let canonical = column0 * params.d_in + d1;
        let destination = select(canonical, d1 * params.d_out + column0,
            params.canonical_din_layout != 0u);
        grad_weight[destination] = grad_weight[destination] + sum10;
    }
    if (d1 < params.d_in && column1 < params.d_out) {
        let canonical = column1 * params.d_in + d1;
        let destination = select(canonical, d1 * params.d_out + column1,
            params.canonical_din_layout != 0u);
        grad_weight[destination] = grad_weight[destination] + sum11;
    }
}

// Bias remains one independent row reduction per output channel. Mapping the
// 8x8 local IDs to a 64-wide logical lane keeps Metal's per-module workgroup
// contract uniform across all three entry points.
@compute @workgroup_size(8, 8, 1)
fn bias_main(@builtin(workgroup_id) group_id : vec3<u32>,
             @builtin(local_invocation_id) local_id : vec3<u32>) {
    let lane = local_id.y * 8u + local_id.x;
    let column = group_id.x * 64u + lane;
    if (column >= params.d_out || params.has_bias == 0u) { return; }
    var sum = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        sum = sum + grad_output[row * params.d_out + column];
    }
    grad_bias[column] = grad_bias[column] + sum;
}
