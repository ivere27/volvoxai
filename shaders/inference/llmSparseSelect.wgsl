// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> index_q:array<f32>;
@group(0) @binding(1) var<storage, read> index_k:array<f32>;
@group(0) @binding(2) var<storage, read> index_weights:array<f32>;
@group(0) @binding(3) var<storage, read> positions:array<i32>;
@group(0) @binding(4) var<storage, read_write> selection:array<i32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(5) var<uniform> p:Params;

var<workgroup> scores:array<f32,512>;
var<workgroup> chosen:array<u32,512>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=wid.x; if(row>=p.work) {return;}
    let query=row%p.sequence; let batch=row/p.sequence;
    var position=query+p.query_offset; if((p.flags&32u)!=0u) {position=u32(positions[query]);}
    let count=min((position+1u)/p.rate,p.keys/p.rate); let base=row*(p.top_k+1u);
    if((p.flags&128u)!=0u) {
        if(lane==0u) {
            var selected=0u;
            for(var j=0u;j<count;j++) {if(bitcast<i32>(index_q[row*p.keys+j])!=0) {
                selection[base+1u+selected]=i32(j); selected++;
            }}
            selection[base]=-i32(selected);
        }
        return;
    }
    let indexed=(p.flags&16u)!=0u && p.top_k>0u && p.top_k<count;
    // Non-indexed compression only needs a count. Return on uniform controls
    // before touching bounded workgroup arrays, allowing banks above 512 rows.
    if((p.flags&16u)==0u || p.top_k==0u) {
        if(lane==0u) {selection[base]=i32(count);}
        return;
    }
    for(var j=lane;j<count;j+=64u) {
        chosen[j]=0u; scores[j]=0.0;
        if(indexed) {
            var total=0.0;
            for(var h=0u;h<p.index_heads;h++) {
                var dot=0.0;
                for(var d=0u;d<p.index_dim;d++) {dot+=index_q[(row*p.index_heads+h)*p.index_dim+d]*index_k[(batch*p.keys+j)*p.index_dim+d];}
                total+=max(dot,0.0)*index_weights[row*p.index_heads+h];
            }
            scores[j]=total/sqrt(f32(p.index_heads*p.index_dim));
        }
    }
    workgroupBarrier();
    if(lane!=0u) {return;}
    selection[base]=i32(count);
    if(indexed) {
        selection[base]=-i32(p.top_k);
        for(var pick=0u;pick<p.top_k;pick++) {
            var best=count;
            for(var j=0u;j<count;j++) {if(chosen[j]==0u && (best==count || scores[j]>scores[best])) {best=j;}}
            chosen[best]=1u; selection[base+1u+pick]=i32(best);
        }
    }
}
