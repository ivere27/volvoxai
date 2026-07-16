@group(0) @binding(0) var<storage, read> boxes : array<f32>;
@group(0) @binding(1) var<storage, read> scores : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

struct Params {
    batches : u32,
    spatial : u32,
    classes : u32,
    max_output : u32,
    output_rows : u32,
    iou_threshold : f32,
    score_threshold : f32,
}
@group(0) @binding(3) var<uniform> params : Params;

fn box_iou(b : u32, a_idx : u32, b_idx : u32) -> f32 {
    let base_a = (b * params.spatial + a_idx) * 4u;
    let base_b = (b * params.spatial + b_idx) * 4u;
    let ay1 = boxes[base_a + 0u];
    let ax1 = boxes[base_a + 1u];
    let ay2 = boxes[base_a + 2u];
    let ax2 = boxes[base_a + 3u];
    let by1 = boxes[base_b + 0u];
    let bx1 = boxes[base_b + 1u];
    let by2 = boxes[base_b + 2u];
    let bx2 = boxes[base_b + 3u];
    let xx1 = max(ax1, bx1);
    let yy1 = max(ay1, by1);
    let xx2 = min(ax2, bx2);
    let yy2 = min(ay2, by2);
    let w = max(0.0, xx2 - xx1);
    let h = max(0.0, yy2 - yy1);
    let inter = w * h;
    let area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1);
    let area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1);
    let denom = area_a + area_b - inter;
    if (denom <= 0.0) { return 0.0; }
    return inter / denom;
}

fn selected_suppresses(b : u32, c : u32, candidate : u32, out_start : u32, selected_count : u32) -> bool {
    for (var j = 0u; j < selected_count; j = j + 1u) {
        let row = out_start + j;
        if (row >= params.output_rows) { return false; }
        let prev_b = u32(output[row * 3u + 0u]);
        let prev_c = u32(output[row * 3u + 1u]);
        let prev_s = u32(output[row * 3u + 2u]);
        if (prev_b == b && prev_c == c && box_iou(b, candidate, prev_s) > params.iou_threshold) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if (global_id.x != 0u) { return; }

    for (var i = 0u; i < params.output_rows; i = i + 1u) {
        output[i * 3u + 0u] = -1.0;
        output[i * 3u + 1u] = -1.0;
        output[i * 3u + 2u] = -1.0;
    }

    var out_idx = 0u;
    for (var b = 0u; b < params.batches; b = b + 1u) {
        for (var c = 0u; c < params.classes; c = c + 1u) {
            let class_start = out_idx;
            var selected = 0u;
            loop {
                if (selected >= params.max_output || out_idx >= params.output_rows) { break; }
                var best_score = params.score_threshold;
                var best_s = params.spatial;
                for (var s = 0u; s < params.spatial; s = s + 1u) {
                    let score = scores[b * (params.classes * params.spatial) + c * params.spatial + s];
                    if (score < best_score) { continue; }
                    if (selected_suppresses(b, c, s, class_start, selected)) { continue; }
                    if (best_s == params.spatial || score > best_score) {
                        best_score = score;
                        best_s = s;
                    }
                }
                if (best_s == params.spatial) { break; }
                output[out_idx * 3u + 0u] = f32(b);
                output[out_idx * 3u + 1u] = f32(c);
                output[out_idx * 3u + 2u] = f32(best_s);
                out_idx = out_idx + 1u;
                selected = selected + 1u;
            }
        }
    }
}
