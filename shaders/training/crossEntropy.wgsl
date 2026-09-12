// Each invocation owns one label row. Separate dispatches accumulate losses
// sharing logits, so gradient addition needs no floating-point atomics.
@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read> targets : array<i32>;
@group(0) @binding(2) var<storage, read_write> gradient : array<f32>;
@group(0) @binding(3) var<storage, read_write> metrics : array<f32>;
struct Params {
    classes : u32, count : u32, first_row : u32, row_stride : u32,
    ignore_index : i32, metric_offset : u32, weight : f32, normalizer : f32,
}
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let r = gid.x;
    if (r >= params.count) { return; }
    let metric = params.metric_offset + r * 4u;
    metrics[metric] = 0.0;
    metrics[metric + 1u] = 0.0;
    metrics[metric + 2u] = 0.0;
    metrics[metric + 3u] = params.normalizer;
    let label = targets[r];
    if (label == params.ignore_index) { return; }
    metrics[metric + 2u] = 1.0;
    if (params.weight == 0.0) { return; }
    let base = (params.first_row + r * params.row_stride) * params.classes;
    var maximum = logits[base];
    var predicted = 0u;
    for (var c = 0u; c < params.classes; c++) {
        let value = logits[base + c];
        if (!(abs(value) <= 3.402823e38)) {
            metrics[metric + 2u] = -1.0;
            return;
        }
        if (value > maximum) { maximum = value; predicted = c; }
    }
    var denominator = 0.0;
    for (var c = 0u; c < params.classes; c++) {
        denominator += exp(logits[base + c] - maximum);
    }
    metrics[metric] = (maximum + log(denominator) - logits[base + u32(label)]) *
        params.weight / params.normalizer;
    metrics[metric + 1u] = select(0.0, 1.0, predicted == u32(label));
    for (var c = 0u; c < params.classes; c++) {
        let probability = exp(logits[base + c] - maximum) / denominator;
        gradient[base + c] += (probability - select(0.0, 1.0, c == u32(label))) *
            params.weight / params.normalizer;
    }
}
