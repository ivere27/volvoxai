// Cooperative regular NHWC/HWIO Conv2D for groups=1 and output channels
// divisible by 16. One invocation accumulates sixteen adjacent channels so
// every scalar input load is shared across four vec4 weight/output lanes.
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output : array<vec4<f32>>;

struct Params {
    n : u32,
    in_h : u32,
    in_w : u32,
    in_c : u32,
    out_c : u32,
    out_h : u32,
    out_w : u32,
    kh : u32,
    kw : u32,
    sy : u32,
    sx : u32,
    pt : u32,
    pl : u32,
    groups : u32,
    relu : u32,
    dy : u32,
    dx : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

fn apply_relu4(value : vec4<f32>) -> vec4<f32> {
    if (params.relu == 1u) {
        return max(value, vec4<f32>(0.0));
    }
    if (params.relu >= 2u) {
        return min(max(value, vec4<f32>(0.0)), vec4<f32>(6.0));
    }
    return value;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let output_blocks = params.out_c / 16u;
    let batch_index = gid.z / output_blocks;
    let output_channel = (gid.z % output_blocks) * 16u;
    if (batch_index >= params.n || ox >= params.out_w || oy >= params.out_h) {
        return;
    }

    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);
    // Keep each channel's reduction order identical to the scalar kernel.
    for (var input_channel = 0u; input_channel < params.in_c;
         input_channel = input_channel + 1u) {
        for (var kernel_y = 0u; kernel_y < params.kh;
             kernel_y = kernel_y + 1u) {
            let input_y = i32(oy * params.sy + kernel_y * params.dy) -
                i32(params.pt);
            if (input_y < 0 || input_y >= i32(params.in_h)) { continue; }
            for (var kernel_x = 0u; kernel_x < params.kw;
                 kernel_x = kernel_x + 1u) {
                let input_x = i32(ox * params.sx + kernel_x * params.dx) -
                    i32(params.pl);
                if (input_x < 0 || input_x >= i32(params.in_w)) { continue; }
                let input_index = ((batch_index * params.in_h + u32(input_y)) *
                    params.in_w + u32(input_x)) * params.in_c + input_channel;
                let weight_base = ((((kernel_y * params.kw + kernel_x) *
                    params.in_c + input_channel) * params.out_c) +
                    output_channel) / 4u;
                let value = input[input_index];
                sum0 = sum0 + value * weight[weight_base];
                sum1 = sum1 + value * weight[weight_base + 1u];
                sum2 = sum2 + value * weight[weight_base + 2u];
                sum3 = sum3 + value * weight[weight_base + 3u];
            }
        }
    }

    let output_base = (((batch_index * params.out_h + oy) * params.out_w +
        ox) * params.out_c + output_channel) / 4u;
    let bias_base = output_channel / 4u;
    output[output_base] = apply_relu4(sum0 + bias[bias_base]);
    output[output_base + 1u] = apply_relu4(sum1 + bias[bias_base + 1u]);
    output[output_base + 2u] = apply_relu4(sum2 + bias[bias_base + 2u]);
    output[output_base + 3u] = apply_relu4(sum3 + bias[bias_base + 3u]);
}
