// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> scores:array<f32>;
@group(0) @binding(1) var<storage, read> mask:array<i32>;
@group(0) @binding(2) var<storage, read_write> output:array<i32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(3) var<uniform> p:Params;
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
    let row=gid.x; if(row>=p.work) {return;}
    let query=row%p.sequence; let batch=row/p.sequence; let capacity=(p.keys+p.rate-1u)/p.rate;
    var visible=0u; for(var j=0u;j<p.keys;j++) {visible+=select(0u,1u,keep(batch,query,j));}
    let blocks=visible/p.rate; var rank=0u; var last_block=0xffffffffu; var selected=false;
    for(var j=0u;j<p.keys;j++) {
        output[row*p.keys+j]=0; if(!keep(batch,query,j)) {continue;}
        let block=rank/p.rate;
        if(block!=last_block) {
            selected=block>=blocks || p.top_k>=blocks; last_block=block;
            if(!selected) {let score=scores[row*capacity+block]; var higher=0u;
                for(var other=0u;other<blocks;other++) {let candidate=scores[row*capacity+other]; if(candidate>score || (candidate==score && other<block)) {higher++;} if(higher>=p.top_k) {break;}}
                selected=higher<p.top_k;}
        }
        output[row*p.keys+j]=select(0,1,selected); rank++;
    }
}
