// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> gate:array<f32>;
@group(0) @binding(2) var<storage, read> bias:array<f32>;
@group(0) @binding(3) var<storage, read> weight:array<f32>;
@group(0) @binding(4) var<storage, read> cosine:array<f32>;
@group(0) @binding(5) var<storage, read> sine:array<f32>;
@group(0) @binding(6) var<storage, read_write> output:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(7) var<uniform> p:Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    if(gid.x>=p.work) {return;} let unit=gid.x+p.first_unit;
    let batch=unit/p.sequence; let window=unit%p.sequence; let out=unit*p.key_dim;
    if(window>=p.sequence/p.rate) {for(var d=0u;d<p.key_dim;d++) {output[out+d]=0.0;} return;}
    var first=window*p.rate; if(p.groups==2u && window>0u) {first=(window-1u)*p.rate;}
    let end=(window+1u)*p.rate; let width=p.groups*p.key_dim; var values:array<f32,512>; var squared=0.0;
    for(var d=0u;d<p.key_dim;d++) {
        var maximum=-3.402823466e38;
        for(var t=first;t<end;t++) {var offset=0u; if(p.groups==2u && t>=window*p.rate) {offset=p.key_dim;}
            let index=(batch*p.sequence+t)*width+offset+d; maximum=max(maximum,gate[index]+bias[(t%p.rate)*width+offset+d]);}
        var total=0.0; var denominator=0.0;
        for(var t=first;t<end;t++) {var offset=0u; if(p.groups==2u && t>=window*p.rate) {offset=p.key_dim;}
            let index=(batch*p.sequence+t)*width+offset+d; let probability=exp(gate[index]+bias[(t%p.rate)*width+offset+d]-maximum);
            total+=probability*input[index]; denominator+=probability;}
        values[d]=total/denominator; squared+=values[d]*values[d];
    }
    let inverse=inverseSqrt(squared/f32(p.key_dim)+p.eps);
    for(var d=0u;d<p.key_dim;d++) {values[d]*=inverse*weight[d];}
    if((p.flags&64u)!=0u) {
        let half=p.rotary_dim/2u; var table_width=p.rotary_dim; if(p.interleaved!=0u) {table_width=half;}
        for(var pair=0u;pair<half;pair++) {
            var left=p.rotary_offset+pair; var right=left+half; var second=pair+half;
            if(p.interleaved!=0u) {left=p.rotary_offset+pair*2u; right=left+1u; second=pair;}
            let x=values[left]; let y=values[right]; let table=window*table_width;
            values[left]=x*cosine[table+pair]-y*sine[table+pair]; values[right]=y*cosine[table+second]+x*sine[table+second];
        }
    }
    for(var d=0u;d<p.key_dim;d++) {output[out+d]=values[d];}
}
