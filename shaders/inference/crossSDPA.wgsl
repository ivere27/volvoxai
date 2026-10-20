@group(0) @binding(0) var<storage, read> q_in : array<f32>;
            @group(0) @binding(1) var<storage, read> k_in : array<f32>;
            @group(0) @binding(2) var<storage, read> v_in : array<f32>;
            @group(0) @binding(3) var<storage, read> mask : array<i32>;
            @group(0) @binding(4) var<storage, read_write> output : array<f32>;
            
            struct Params {
                seq_len_q : u32,
                seq_len_kv : u32,
                d_model : u32,
                num_heads : u32,
                head_dim : u32,
                batch : u32,
                scale : f32,
                causal : u32,
                mask_mode : u32,
                kv_heads : u32,
                _pad1 : u32,
                _pad2 : u32,
            }
            @group(0) @binding(5) var<uniform> params : Params;

            fn key_allowed(batch_idx : u32, query : u32, key : u32) -> bool {
                if (params.causal != 0u && key > query) { return false; }
                if (params.mask_mode == 0u) { return true; }
                var index = key;
                if (params.mask_mode == 2u) {
                    index = batch_idx * params.seq_len_kv + key;
                } else if (params.mask_mode == 3u) {
                    index = query * params.seq_len_kv + key;
                } else if (params.mask_mode == 4u) {
                    index = (batch_idx * params.seq_len_q + query) * params.seq_len_kv + key;
                }
                return mask[index] != 0;
            }
            

// Long attention shares each QK score across every value component. Key tiles
// bound probability and value reductions without a sequence-sized score buffer.
fn sum_error(next:f32, previous:f32, addend:f32)->f32 {
    if ((bitcast<u32>(next)&0x7f800000u)==0x7f800000u) {return 0.0;}
    return (next-previous)-addend;
}
fn long_score(q_cache:ptr<function,array<f32,128>>,key_base:u32)->f32 {
    var score=0.0; var error=0.0;
    for(var d=0u;d<params.head_dim;d++) {
        let product=(*q_cache)[d]*k_in[key_base+d];
        let addend=product-error; let next=score+addend;
        error=sum_error(next,score,addend); score=next;
    }
    return score*params.scale;
}
fn long_attention(q_cache:ptr<function,array<f32,128>>,q_base:u32,
                  kv_base:u32,kv_width:u32,kv_head:u32,
                  q_idx:u32,h_idx:u32,batch_idx:u32) {
    var maximum=0.0; var started=false;
    for(var key=0u;key<params.seq_len_kv;key++) {
        if(!key_allowed(batch_idx,q_idx,key)) {continue;}
        let score=long_score(q_cache,kv_base+key*kv_width+kv_head*params.head_dim);
        if(started) {maximum=max(maximum,score);} else {maximum=score;started=true;}
    }
    var values:array<f32,128>;
    for(var d=0u;d<params.head_dim;d++) {values[d]=0.0;}
    var denominator=0.0; var probabilities:array<f32,32>;
    if(started) {
        for(var first=0u;first<params.seq_len_kv;first+=32u) {
            let count=min(32u,params.seq_len_kv-first); var tile_sum=0.0;
            for(var j=0u;j<count;j++) {
                var probability=0.0;
                if(key_allowed(batch_idx,q_idx,first+j)) {
                    let key_base=kv_base+(first+j)*kv_width+kv_head*params.head_dim;
                    probability=exp(long_score(q_cache,key_base)-maximum);
                }
                probabilities[j]=probability;tile_sum+=probability;
            }
            denominator+=tile_sum;
            for(var d=0u;d<params.head_dim;d++) {
                var partial=0.0;
                for(var j=0u;j<count;j++) {
                    if(!key_allowed(batch_idx,q_idx,first+j)) {continue;}
                    let key_base=kv_base+(first+j)*kv_width+kv_head*params.head_dim;
                    partial+=probabilities[j]*v_in[key_base+d];
                }
                values[d]+=partial;
            }
        }
    }

    for(var d=0u;d<params.head_dim;d++) {
        var value=0.0;if(started) {value=values[d]/denominator;}
        output[q_base+q_idx*params.d_model+h_idx*params.head_dim+d]=value;
    }
}

            @compute @workgroup_size(64, 1, 1)
            fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
                let q_idx = global_id.x; 
                let h_idx = global_id.y; 
                let batch_idx = global_id.z;
                
                if (q_idx >= params.seq_len_q || h_idx >= params.num_heads || batch_idx >= params.batch) { return; }
                
                let head_dim = params.head_dim;
                let d_model = params.d_model;
                let q_base = batch_idx * params.seq_len_q * d_model;
                let kv_width = params.kv_heads * head_dim;
                let kv_head = h_idx / (params.num_heads / params.kv_heads);
                let kv_base = batch_idx * params.seq_len_kv * kv_width;
                
                // Cache Q for this head and this q_idx
                var q_cache : array<f32, 128>;
                for (var d = 0u; d < head_dim; d = d + 1u) {
                    q_cache[d] = q_in[q_base + q_idx * d_model + (h_idx * head_dim) + d];
                }
                
                if(params.seq_len_kv>=256u) {
                    long_attention(&q_cache,q_base,kv_base,kv_width,kv_head,q_idx,h_idx,batch_idx);
                    return;
                }
                var max_logit : f32 = -1e38;
                
                // Pass 1: find max
                var valid_keys = 0u;
                for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
                    if (!key_allowed(batch_idx, q_idx, k_idx)) { continue; }
                    var score : f32 = 0.0;
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        let k_val = k_in[kv_base + k_idx * kv_width + (kv_head * head_dim) + d];
                        score = score + (q_cache[d] * k_val);
                    }
                    score = score * params.scale;
                    if (score > max_logit) { max_logit = score; }
                    valid_keys = valid_keys + 1u;
                }

                if (valid_keys == 0u) {
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        output[q_base + q_idx * d_model + (h_idx * head_dim) + d] = 0.0;
                    }
                    return;
                }
                
                // Pass 2: sum exp
                var sum_exp : f32 = 0.0;
                for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
                    if (!key_allowed(batch_idx, q_idx, k_idx)) { continue; }
                    var score : f32 = 0.0;
                    for (var d = 0u; d < head_dim; d = d + 1u) {
                        let k_val = k_in[kv_base + k_idx * kv_width + (kv_head * head_dim) + d];
                        score = score + (q_cache[d] * k_val);
                    }
                    score = score * params.scale;
                    sum_exp = sum_exp + exp(score - max_logit);
                }
                
                // Pass 3: output
                for (var d = 0u; d < head_dim; d = d + 1u) {
                    var out_val : f32 = 0.0;
                    for (var k_idx = 0u; k_idx < params.seq_len_kv; k_idx = k_idx + 1u) {
                        if (!key_allowed(batch_idx, q_idx, k_idx)) { continue; }
                        var score : f32 = 0.0;
                        for (var kd = 0u; kd < head_dim; kd = kd + 1u) {
                            let k_val = k_in[kv_base + k_idx * kv_width + (kv_head * head_dim) + kd];
                            score = score + (q_cache[kd] * k_val);
                        }
                        score = score * params.scale;
                        
                        let w = exp(score - max_logit) / sum_exp;
                        let v_val = v_in[kv_base + k_idx * kv_width + (kv_head * head_dim) + d];
                        out_val = out_val + (w * v_val);
                    }
                    output[q_base + q_idx * d_model + (h_idx * head_dim) + d] = out_val;
                }
            }
