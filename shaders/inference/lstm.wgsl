// One persistent workgroup per batch/direction; every time step observes the
// complete preceding hidden state. ONNX gate ordering is input/output/forget/cell.
fn stable_tanh(x : f32) -> f32 {
    if (x > 10.0) { return 1.0; }
    if (x < -10.0) { return -1.0; }
    if (x == 0.0) { return x; }
    return tanh(x);
}
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weight : array<f32>;
@group(0) @binding(2) var<storage, read> recurrent : array<f32>;
@group(0) @binding(3) var<storage, read> bias : array<f32>;
@group(0) @binding(4) var<storage, read> initial_h : array<f32>;
@group(0) @binding(5) var<storage, read> initial_c : array<f32>;
@group(0) @binding(6) var<storage, read_write> output : array<f32>;
struct Params {
    time : u32, batch : u32, width : u32, hidden : u32,
    direction : u32, flags : u32, _pad0 : u32, _pad1 : u32,
}
@group(0) @binding(7) var<uniform> params : Params;
var<workgroup> h : array<f32, 1024>;
var<workgroup> c : array<f32, 1024>;
var<workgroup> next_h : array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(workgroup_id) gid : vec3<u32>) {
    let directions = select(1u, 2u, params.direction == 2u);
    let d = gid.x / params.batch;
    let b = gid.x % params.batch;
    let state_base = (d * params.batch + b) * params.hidden;
    for (var j = lid.x; j < params.hidden; j += 256u) {
        h[j] = 0.0; c[j] = 0.0;
        if ((params.flags & 2u) != 0u) { h[j] = initial_h[state_base + j]; }
        if ((params.flags & 4u) != 0u) { c[j] = initial_c[state_base + j]; }
    }
    workgroupBarrier();
    for (var step = 0u; step < params.time; step += 1u) {
        let t = select(step, params.time - 1u - step, params.direction == 1u || d == 1u);
        for (var j = lid.x; j < params.hidden; j += 256u) {
            var gates = vec4<f32>(0.0);
            var correction = vec4<f32>(0.0);
            for (var k = 0u; k < params.width; k += 1u) {
                let x = input[(t * params.batch + b) * params.width + k];
                let base = d * 4u * params.hidden * params.width + j * params.width + k;
                let stride = params.hidden * params.width;
                let product = x * vec4<f32>(weight[base], weight[base + stride], weight[base + 2u * stride], weight[base + 3u * stride]);
                let adjusted = product - correction;
                let total = gates + adjusted;
                correction = (total - gates) - adjusted;
                gates = total;
            }
            for (var k = 0u; k < params.hidden; k += 1u) {
                let base = d * 4u * params.hidden * params.hidden + j * params.hidden + k;
                let stride = params.hidden * params.hidden;
                let product = h[k] * vec4<f32>(recurrent[base], recurrent[base + stride], recurrent[base + 2u * stride], recurrent[base + 3u * stride]);
                let adjusted = product - correction;
                let total = gates + adjusted;
                correction = (total - gates) - adjusted;
                gates = total;
            }
            if ((params.flags & 1u) != 0u) {
                let base = d * 8u * params.hidden + j;
                let stride = params.hidden;
                gates += vec4<f32>(bias[base] + bias[base + 4u * stride],
                    bias[base + stride] + bias[base + 5u * stride],
                    bias[base + 2u * stride] + bias[base + 6u * stride],
                    bias[base + 3u * stride] + bias[base + 7u * stride]);
            }
            let activated = vec3<f32>(1.0) / (vec3<f32>(1.0) + exp(-gates.xyz));
            c[j] = activated.z * c[j] + activated.x * stable_tanh(gates.w);
            next_h[j] = activated.y * stable_tanh(c[j]);
        }
        workgroupBarrier();
        for (var j = lid.x; j < params.hidden; j += 256u) {
            h[j] = next_h[j];
            output[((t * directions + d) * params.batch + b) * params.hidden + j] = next_h[j];
        }
        workgroupBarrier();
    }
}
