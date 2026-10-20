// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> update:array<f32>;
@group(0) @binding(2) var<storage, read> post:array<f32>;
@group(0) @binding(3) var<storage, read> comb:array<f32>;
@group(0) @binding(4) var<storage, read_write> output:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, pad0:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, pad2:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(5) var<uniform> p:Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let c=unit%p.key_dim; let stream=(unit/p.key_dim)%p.streams; let row=unit/(p.key_dim*p.streams);
    var value=post[row*p.streams+stream]*update[row*p.key_dim+c];
    for(var i=0u;i<p.streams;i++) {value+=comb[(row*p.streams+i)*p.streams+stream]*input[(row*p.streams+i)*p.key_dim+c];}
    output[unit]=value;
}
