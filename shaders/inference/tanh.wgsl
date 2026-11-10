// F32 tanh rounds to exactly +/-1 in these tails. Some device builtins
// form exp(2*x), whose overflow would otherwise produce Inf/Inf and NaN.
fn stable_tanh(x : f32) -> f32 {
    if (x > 10.0) { return 1.0; }
    if (x < -10.0) { return -1.0; }
    if (x == 0.0) { return x; }
    return tanh(x);
}
@group(0) @binding(0) var<storage, read> input : array<f32>;
        @group(0) @binding(1) var<storage, read_write> output : array<f32>;
        struct Params { size : u32 }
        @group(0) @binding(2) var<uniform> params : Params;
        @compute @workgroup_size(64)
        fn main(
            @builtin(global_invocation_id) global_id : vec3<u32>,
            @builtin(num_workgroups) grid : vec3<u32>,
        ) {
            let idx = global_id.x + global_id.y * (grid.x * 64u);
            if (idx >= params.size) { return; }
            let x = input[idx];
            var out_val = x;
            out_val = stable_tanh(x);
            output[idx] = out_val;
        }
