// Exact score order; equal scores select the lowest index. The source
// candidates mask is independent of its own final selected mask.
@group(0) @binding(0) var<storage,read> scores:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<i32>;
@group(0) @binding(2) var<storage,read_write> candidates:array<i32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(3) var<uniform> p:Params;
fn block_score(row:u32,block:u32,visible:u32)->f32 {
    if(block==(visible-1u)/p.window){return bitcast<f32>(0x7f800000u);}
    var best=bitcast<f32>(0xff800000u);
    let begin=block*p.window;let end=begin+min(visible-begin,p.window);
    for(var j=begin;j<end;j++){best=max(best,scores[row*p.keys+j]);}
    return best;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let id=gid.x;if(id>=p.work){return;}
    let row=id/p.keys;let j=id%p.keys;let visible=(row%p.sequence+1u)/p.rate;
    output[id]=0;if((p.flags&2u)!=0u){candidates[id]=0;}
    if(j<visible){
        let score=scores[id];var higher=0u;
        for(var other=0u;other<visible;other++) {
            let value=scores[row*p.keys+other];
            if(value>score||(value==score&&other<j)){higher++;}
            if(higher>=p.top_k){break;}
        }
        output[id]=select(0,1,higher<p.top_k);
    }
    if((p.flags&2u)!=0u && visible>0u && j<p.keys/p.rate) {
        if(j/p.window>(visible-1u)/p.window){return;}
        let block=j/p.window;let blocks=(visible-1u)/p.window+1u;
        let value=block_score(row,block,visible);var better=0u;
        for(var other=0u;other<blocks;other++) {
            let candidate=block_score(row,other,visible);
            if(candidate>value||(candidate==value&&other<block)){better++;}
            if(better>=p.groups){break;}
        }
        candidates[id]=select(0,1,value>bitcast<f32>(0xff800000u)&&better<p.groups);
    }
}
