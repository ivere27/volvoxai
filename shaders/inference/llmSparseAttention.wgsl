// Portable F32 LLM inference. ABI: native/src/backends/llm_shader_abi.md.
@group(0) @binding(0) var<storage, read> q:array<f32>;
@group(0) @binding(1) var<storage, read> k:array<f32>;
@group(0) @binding(2) var<storage, read> v:array<f32>;
@group(0) @binding(3) var<storage, read> compressed:array<f32>;
@group(0) @binding(4) var<storage, read> sinks:array<f32>;
@group(0) @binding(5) var<storage, read> mask:array<i32>;
@group(0) @binding(6) var<storage, read> selection:array<i32>;
@group(0) @binding(7) var<storage, read_write> output:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first_unit:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, query_offset:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(8) var<uniform> p:Params;
fn keep(batch:u32,query:u32,key:u32)->bool {
    if(p.causal!=0u && key>query+p.query_offset) {return false;}
    if(p.window>0u && query+p.query_offset>=p.window && key<=query+p.query_offset-p.window) {return false;}
    if(p.mask_mode==0u) {return true;}
    var offset=key;
    if(p.mask_mode==2u) {offset+=batch*p.keys;}
    if(p.mask_mode==3u) {offset+=query*p.keys;}
    if(p.mask_mode==4u) {offset+=(batch*p.sequence+query)*p.keys;}
    return mask[offset]!=0;
}

// Long ordinary attention fixes the maximum before reducing probabilities.
// A small key tile bounds the AV summation length without a scores matrix or a
// second value-width accumulator. Compressed attention retains its online path.
fn sum_correction(next:f32,previous:f32,addend:f32)->f32 {
    // Do not manufacture NaN from Inf-Inf while compensating an ordinary sum.
    // Non-finite data still participates in the sum; only compensation resets.
    if((bitcast<u32>(next)&0x7f800000u)==0x7f800000u) {return 0.0;}
    return (next-previous)-addend;
}
fn long_score(unit:u32,key:u32)->f32 {
    var score=0.0; var correction=0.0;
    for(var d=0u;d<p.key_dim;d++) {
        let product=q[unit*p.key_dim+d]*k[key*p.key_dim+d];
        let addend=product-correction; let next=score+addend;
        correction=sum_correction(next,score,addend); score=next;
    }
    return score*p.scale;
}
fn long_dense(unit:u32,h:u32,query:u32,batch:u32,kh:u32,
              values:ptr<function,array<f32,512>>) {
    var maximum=0.0; var started=false;
    if((p.flags&8u)!=0u) {maximum=sinks[h]; started=true;}
    for(var j=0u;j<p.keys;j++) {
        if(!keep(batch,query,j)) {continue;}
        let key=(batch*p.keys+j)*p.kv_heads+kh; let score=long_score(unit,key);
        if(started) {maximum=max(maximum,score);} else {maximum=score; started=true;}
    }
    for(var c=0u;c<p.value_dim;c++) {(*values)[c]=0.0;}
    var denominator=0.0; var correction=0.0;
    if((p.flags&8u)!=0u) {denominator=exp(sinks[h]-maximum);}
    var probabilities:array<f32,32>;
    if(started) {
        for(var first=0u;first<p.keys;first+=32u) {
            let count=min(32u,p.keys-first);
            for(var j=0u;j<count;j++) {
                var probability=0.0;
                if(keep(batch,query,first+j)) {
                    let key=(batch*p.keys+first+j)*p.kv_heads+kh;
                    probability=exp(long_score(unit,key)-maximum);
                }
                probabilities[j]=probability;
                let addend=probability-correction; let next=denominator+addend;
                correction=sum_correction(next,denominator,addend); denominator=next;
            }
            for(var c=0u;c<p.value_dim;c++) {
                var partial=0.0; var partial_correction=0.0;
                for(var j=0u;j<count;j++) {
                    if(!keep(batch,query,first+j)) {continue;}
                    let key=(batch*p.keys+first+j)*p.kv_heads+kh;
                    let product=probabilities[j]*v[key*p.value_dim+c];
                    let addend=product-partial_correction; let next=partial+addend;
                    partial_correction=sum_correction(next,partial,addend); partial=next;
                }
                (*values)[c]+=partial;
            }
        }
    }
    for(var c=0u;c<p.value_dim;c++) {
        var value=0.0; if(denominator>0.0) {value=(*values)[c]/denominator;}
        output[unit*p.value_dim+c]=value;
    }
}

// One query/head uses online softmax; storage never holds a dense score matrix.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let unit=gid.x; if(unit>=p.work) {return;}
    let h=unit%p.heads; let row=unit/p.heads; let query=row%p.sequence; let batch=row/p.sequence;
    let kh=h/(p.heads/p.kv_heads); var values:array<f32,512>;
    if(p.keys>=512u && (p.flags&4u)==0u) {long_dense(unit,h,query,batch,kh,&values); return;}
    var maximum=0.0; var denominator=0.0; var started=false;
    if((p.flags&8u)!=0u) {maximum=sinks[h]; denominator=1.0; started=true;}
    for(var c=0u;c<p.value_dim;c++) {values[c]=0.0;}
    for(var j=0u;j<p.keys;j++) {
        if(!keep(batch,query,j)) {continue;}
        let key=(batch*p.keys+j)*p.kv_heads+kh; var score=0.0;
        for(var d=0u;d<p.key_dim;d++) {score+=q[unit*p.key_dim+d]*k[key*p.key_dim+d];}
        score*=p.scale; var old=0.0; var probability=1.0;
        if(started) {let next=max(maximum,score); old=exp(maximum-next); probability=exp(score-next); maximum=next;} else {maximum=score; started=true;}
        denominator=denominator*old+probability;
        for(var c=0u;c<p.value_dim;c++) {values[c]=values[c]*old+probability*v[key*p.value_dim+c];}
    }
    if((p.flags&4u)!=0u) {
        let base=row*(p.top_k+1u); let count=u32(selection[base]);
        // A negative header distinguishes a selected top-k list from all keys.
        let is_list=selection[base]<0; var total=count;
        if(is_list) {total=u32(-selection[base]);}
        for(var pick=0u;pick<total;pick++) {
            var j=pick; if(is_list) {j=u32(selection[base+1u+pick]);}
            let key=(batch*p.keys+j)*p.key_dim; var score=0.0;
            for(var d=0u;d<p.key_dim;d++) {score+=q[unit*p.key_dim+d]*compressed[key+d];}
            score*=p.scale; var old=0.0; var probability=1.0;
            if(started) {let next=max(maximum,score); old=exp(maximum-next); probability=exp(score-next); maximum=next;} else {maximum=score; started=true;}
            denominator=denominator*old+probability;
            for(var c=0u;c<p.value_dim;c++) {values[c]=values[c]*old+probability*compressed[key+c];}
        }
    }
    for(var c=0u;c<p.value_dim;c++) {var value=0.0; if(denominator>0.0) {value=values[c]/denominator;} output[unit*p.value_dim+c]=value;}
}
