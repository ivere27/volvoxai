// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> ids:array<i32>;
@group(0) @binding(1) var<storage, read> weight:array<f32>;
@group(0) @binding(2) var<storage, read_write> output:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, pad0:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, pad2:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(3) var<uniform> p:Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let id=ids[unit/p.key_dim]; var value=0.0;
    if(id>=0 && u32(id)>=p.rotary_offset) {let row=u32(id)-p.rotary_offset; if(row<p.value_dim) {value=weight[row*p.key_dim+unit%p.key_dim];}}
    output[unit]=value;
}
