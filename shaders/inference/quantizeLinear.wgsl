@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;

struct Params {
    size : u32,
    input_zp : i32,
    output_zp : i32,
    has_input_scale : u32,
    input_scale : f32,
    output_scale : f32,
    pad0 : u32,
    pad1 : u32,
}

@group(0) @binding(2) var<uniform> params : Params;

fn clamp_i8(v : i32) -> i32 {
    return min(max(v, -128), 127);
}

fn i8_byte(v : i32) -> u32 {
    if (v < 0) {
        return u32(v + 256) & 255u;
    }
    return u32(v) & 255u;
}

fn round_even(v : f32) -> f32 {
    let lo = floor(v);
    let frac = v - lo;
    if (frac < 0.5) {
        return lo;
    }
    if (frac > 0.5) {
        return lo + 1.0;
    }
    if (floor(lo * 0.5) == lo * 0.5) {
        return lo;
    }
    return lo + 1.0;
}

fn quantize_one(idx : u32) -> u32 {
    if (idx >= params.size) {
        return 0u;
    }
    var v = input[idx];
    if (params.has_input_scale == 1u) {
        v = (v - f32(params.input_zp)) * params.input_scale;
    }
    let qf = round_even(v / params.output_scale + f32(params.output_zp));
    let q = clamp_i8(i32(min(max(qf, -128.0), 127.0)));
    return i8_byte(q);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let word_idx = global_id.x;
    let base = word_idx * 4u;
    if (base >= params.size) {
        return;
    }

    var packed = quantize_one(base);
    packed = packed | (quantize_one(base + 1u) << 8u);
    packed = packed | (quantize_one(base + 2u) << 16u);
    packed = packed | (quantize_one(base + 3u) << 24u);
    output[word_idx] = packed;
}
