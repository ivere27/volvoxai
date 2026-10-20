// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> cosine:array<f32>;
@group(0) @binding(2) var<storage, read> sine:array<f32>;
@group(0) @binding(3) var<storage, read_write> output:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, pad0:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, pad2:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(4) var<uniform> p:Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let d=unit%p.key_dim; let row=unit/p.key_dim; let token=row/p.heads;
    if(d<p.rotary_offset || d-p.rotary_offset>=p.rotary_dim) {output[unit]=input[unit]; return;}
    let local=d-p.rotary_offset; var frequency=local; var partner=d; var sign=1.0; var table_width=p.rotary_dim;
    if(p.interleaved!=0u) {frequency=local/2u; table_width/=2u; if(local%2u==0u) {partner=d+1u; sign=-1.0;} else {partner=d-1u;}}
    else {if(local<p.rotary_dim/2u) {partner=d+p.rotary_dim/2u; sign=-1.0;} else {partner=d-p.rotary_dim/2u;}}
    let table=token*table_width+frequency;
    output[unit]=input[unit]*cosine[table]+sign*input[row*p.key_dim+partner]*sine[table];
}
