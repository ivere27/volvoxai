@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> positions : array<i32>;
@group(0) @binding(2) var<storage, read> metadata : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

struct Params {
    elements : u32,
    sequence : u32,
    width : u32,
    heads : u32,
    rotary_width : u32,
    position_offset : u32,
    interleaved : u32,
    position_mode : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

fn add_split(a : vec2<f32>, b : vec2<f32>) -> vec2<f32> {
    let sum = a.x + b.x;
    let virtual_value = sum - a.x;
    let residual = (a.x - (sum - virtual_value)) + (b.x - virtual_value) + a.y + b.y;
    let high = sum + residual;
    return vec2<f32>(high, residual - (high - sum));
}

// Input angles use the same F32 position/frequency division as native RoPE.
// Break the exact F32 value into at most 24 powers of two and reduce with the
// immutable split remainders; sin/cos receive an argument inside [-pi,pi].
fn reduced_angle(angle : f32) -> f32 {
    if (angle <= 3.1415927410125732421875) { return angle; }
    let bits = bitcast<u32>(angle);
    let exponent = i32((bits >> 23u) & 255u) - 127;
    let mantissa = (bits & 8388607u) | 8388608u;
    let tau = vec2<f32>(6.283185482025146484375, -0.0000001748455531469517);
    var remainder = vec2<f32>(0.0, 0.0);
    for (var bit = 0u; bit < 24u; bit = bit + 1u) {
        if ((mantissa & (1u << bit)) == 0u) { continue; }
        let power = exponent - 23 + i32(bit);
        var part : vec2<f32>;
        if (power >= 0) {
            let index = params.rotary_width / 2u + u32(power) * 2u;
            part = vec2<f32>(metadata[index], metadata[index + 1u]);
        } else {
            part = vec2<f32>(bitcast<f32>(u32(127 + power) << 23u), 0.0);
        }
        remainder = add_split(remainder, part);
        if (remainder.x > tau.x || (remainder.x == tau.x && remainder.y >= tau.y)) {
            remainder = add_split(remainder, -tau);
        }
    }
    if (remainder.x + remainder.y > 3.1415927410125732421875) {
        remainder = add_split(remainder, -tau);
    }
    return remainder.x + remainder.y;
}

// WGSL built-in trigonometry permits comparatively loose approximation error.
// On the reduced interval these polynomials keep rotation error near F32
// rounding on both hardware and software implementations.
fn rotation(angle : f32) -> vec2<f32> {
    var x = angle;
    var cosine_sign = 1.0;
    if (x > 1.57079637050628662109375) {
        x = 3.1415927410125732421875 - x;
        cosine_sign = -1.0;
    } else if (x < -1.57079637050628662109375) {
        x = -3.1415927410125732421875 - x;
        cosine_sign = -1.0;
    }
    let square = x * x;
    let sine = x * (1.0 + square * (-0.1666666666666667 + square * (0.008333333333333333 +
        square * (-0.0001984126984126984 + square * (0.0000027557319223985893 +
        square * (-0.00000002505210838544172 + square * 0.00000000016059043836821615))))));
    let cosine = 1.0 + square * (-0.5 + square * (0.041666666666666664 +
        square * (-0.001388888888888889 + square * (0.0000248015873015873 +
        square * (-0.0000002755731922398589 + square * 0.00000000208767569878681)))));
    return vec2<f32>(sine, cosine * cosine_sign);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.elements) { return; }
    let head_width = params.width / params.heads;
    let dimension = gid % head_width;
    if (dimension >= params.rotary_width) {
        output[gid] = input[gid];
        return;
    }
    let row = gid / params.width;
    let sequence_index = row % params.sequence;
    var position = i32(params.position_offset + sequence_index);
    if (params.position_mode == 1u) { position = positions[sequence_index]; }
    if (params.position_mode == 2u) { position = positions[row]; }
    let half = params.rotary_width / 2u;
    var pair = dimension % half;
    var left = pair;
    var right = pair + half;
    if (params.interleaved != 0u) {
        pair = dimension / 2u;
        left = pair * 2u;
        right = left + 1u;
    }
    let angle = reduced_angle(f32(position) / metadata[pair]);
    let phase = rotation(angle);
    let sine = phase.x;
    let cosine = phase.y;
    let base = gid - dimension;
    let a = input[base + left];
    let b = input[base + right];
    output[gid] = select(a * sine + b * cosine, a * cosine - b * sine, dimension == left);
}
