// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> q:array<f32>;
@group(0) @binding(1) var<storage, read> k:array<f32>;
@group(0) @binding(2) var<storage, read> weight:array<f32>;
@group(0) @binding(3) var<storage, read> cosine:array<f32>;
@group(0) @binding(4) var<storage, read> sine:array<f32>;
@group(0) @binding(5) var<storage, read> mask:array<i32>;
@group(0) @binding(6) var<storage, read_write> scores:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(7) var<uniform> p:Params;
fn keep(batch:u32,query:u32,key:u32)->bool {
    if(p.causal!=0u && key>query+p.query_offset) {return false;}
    if(p.mask_mode==0u) {return true;}
    var offset=key;
    if(p.mask_mode==2u) {offset+=batch*p.keys;}
    if(p.mask_mode==3u) {offset+=query*p.keys;}
    if(p.mask_mode==4u) {offset+=(batch*p.sequence+query)*p.keys;}
    return mask[offset]!=0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let capacity=(p.keys+p.rate-1u)/p.rate; let block=unit%capacity; let row=unit/capacity; let query=row%p.sequence; let batch=row/p.sequence;
    var key:array<f32,512>; for(var d=0u;d<p.key_dim;d++) {key[d]=0.0;}
    var visible=0u; var first=p.keys; var used=0u;
    for(var j=0u;j<p.keys;j++) {if(!keep(batch,query,j)) {continue;}
        if(visible/p.rate==block) {if(first==p.keys) {first=j;} for(var d=0u;d<p.key_dim;d++) {key[d]+=k[(batch*p.keys+j)*p.key_dim+d]/f32(p.rate);} used++;}
        visible++; if(visible>=(block+1u)*p.rate) {break;}}
    if(used!=p.rate) {scores[unit]=0.0; return;}
    var squared=0.0; for(var d=0u;d<p.key_dim;d++) {squared+=key[d]*key[d];}
    let inverse=inverseSqrt(squared/f32(p.key_dim)+p.eps);
    for(var d=0u;d<p.key_dim;d++) {key[d]*=inverse*weight[d];}
    let half=p.rotary_dim/2u; let table=(batch*p.keys+first)*p.rotary_dim;
    for(var d=0u;d<half;d++) {let left=key[d]; let right=key[d+half]; key[d]=left*cosine[table+d]-right*sine[table+d]; key[d+half]=right*cosine[table+d+half]+left*sine[table+d+half];}
    var score=0.0;
    for(var h=0u;h<p.heads;h++) {var dot=0.0; for(var d=0u;d<p.key_dim;d++) {dot+=q[(row*p.heads+h)*p.key_dim+d]*key[d];} score+=max(dot,0.0);}
    scores[unit]=score/sqrt(f32(p.key_dim));
}
