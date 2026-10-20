#ifndef VOLVOXAI_LLM_MATH_H
#define VOLVOXAI_LLM_MATH_H

#if !defined(VX_LLM_CUDA_MATH)
#include "mathcompat.h"
#endif
#include "llm_ops.h"
#ifndef VX_LLM_INLINE
#define VX_LLM_INLINE static inline
#endif

VX_LLM_INLINE float vx_llm_max(float a, float b) { return a > b ? a : b; }

VX_LLM_INLINE float vx_llm_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}
VX_LLM_INLINE float vx_llm_softplus(float x) {
    return vx_llm_max(x, 0.0f) + logf(1.0f + expf(-fabsf(x)));
}
VX_LLM_INLINE float vx_llm_dot(const float* a, const float* b, int width) {
    float result = 0.0f;
    for (int i = 0; i < width; ++i) result += a[i] * b[i];
    return result;
}

VX_LLM_INLINE void vx_llm_delta_unit(const VxLLMOpArgs* a, uint64_t unit,
                                    float* state) {
    int h = (int)(unit % a->heads), batch = (int)(unit / a->heads);
    int kh = h / (a->heads / a->kv_heads);
    uint64_t state_width = (uint64_t)a->key_dim * a->value_dim;
    for (uint64_t i = 0; i < state_width; ++i)
        state[i] = a->x[7] ? a->x[7][unit * state_width + i] : 0.0f;
    for (int t = 0; t < a->sequence; ++t) {
        uint64_t row = (uint64_t)batch * a->sequence + t;
        const float* q = a->x[0] + (row * a->kv_heads + kh) * a->key_dim;
        const float* k = a->x[1] + (row * a->kv_heads + kh) * a->key_dim;
        const float* v = a->x[2] + (row * a->heads + h) * a->value_dim;
        float qn = a->normalize ? 1.0f / sqrtf(vx_llm_dot(q, q, a->key_dim) + a->eps) : 1.0f;
        float kn = a->normalize ? 1.0f / sqrtf(vx_llm_dot(k, k, a->key_dim) + a->eps) : 1.0f;
        float beta = vx_llm_sigmoid(a->x[4][row * a->heads + h]);
        float decay = expf(-expf(a->x[5][h]) *
            vx_llm_softplus(a->x[3][row * a->heads + h] + a->x[6][h]));
        float* out = a->y[0] + (row * a->heads + h) * a->value_dim;
        for (int c = 0; c < a->value_dim; ++c) {
            float prediction = 0.0f;
            for (int d = 0; d < a->key_dim; ++d) {
                uint64_t index = (uint64_t)d * a->value_dim + c;
                state[index] *= decay;
                prediction += state[index] * k[d] * kn;
            }
            float correction = (v[c] - prediction) * beta;
            float value = 0.0f;
            for (int d = 0; d < a->key_dim; ++d) {
                uint64_t index = (uint64_t)d * a->value_dim + c;
                state[index] += k[d] * kn * correction;
                value += state[index] * q[d] * qn * a->scale;
            }
            out[c] = value;
        }
    }
    if (a->y[1]) for (uint64_t i = 0; i < state_width; ++i)
        a->y[1][unit * state_width + i] = state[i];
}

VX_LLM_INLINE int vx_llm_attention_keep(const VxLLMOpArgs* a,
                                       int batch, int query, int key) {
    if (a->causal && key > query + a->query_offset) return 0;
    if (a->window > 0 && key <= query + a->query_offset - a->window) return 0;
    if (!a->mask) return 1;
    uint64_t index = (uint64_t)key;
    if (a->mask_mode == 2) index += (uint64_t)batch * a->keys;
    if (a->mask_mode == 3) index += (uint64_t)query * a->keys;
    if (a->mask_mode == 4)
        index += ((uint64_t)batch * a->sequence + query) * a->keys;
    return a->mask[index] != 0;
}

VX_LLM_INLINE void vx_llm_attention_unit(const VxLLMOpArgs* a,
                                        uint64_t unit, float* scratch) {
    int h = (int)(unit % a->heads);
    uint64_t row = unit / a->heads;
    int query = (int)(row % a->sequence), batch = (int)(row / a->sequence);
    int kh = h / (a->heads / a->kv_heads);
    const float* q = a->x[0] + unit * a->key_dim;
    float* output = a->y[0] + unit * a->value_dim;
    float* scores = scratch;
    float* selected = scratch + a->keys;
    int count = 0;
    if (a->x[3]) {
        int position = a->positions ? a->positions[query] : query + a->query_offset;
        count = (position + 1) / a->rate;
        int complete = a->keys / a->rate;
        if (count > complete) count = complete;
        for (int j = 0; j < count; ++j) selected[j] = a->x[5] && !a->index_heads ?
            (float)(((const int32_t*)a->x[5])[row*a->keys+j]!=0) : 1.0f;
        if (a->index_heads > 0 && a->top_k > 0 && a->top_k < count) {
            for (int j = 0; j < count; ++j) {
                float total = 0.0f;
                const float* key = a->x[6] + ((uint64_t)batch * a->keys + j) * a->index_dim;
                for (int ih = 0; ih < a->index_heads; ++ih) {
                    const float* iq = a->x[5] + (row * a->index_heads + ih) * a->index_dim;
                    total += vx_llm_max(vx_llm_dot(iq, key, a->index_dim), 0.0f) *
                        a->x[7][row * a->index_heads + ih];
                }
                scores[j] = total / sqrtf((float)a->index_heads * a->index_dim);
                selected[j] = 0.0f;
            }
            for (int pick = 0; pick < a->top_k; ++pick) {
                int best = -1;
                for (int j = 0; j < count; ++j)
                    if (selected[j] == 0.0f &&
                        (best < 0 || scores[j] > scores[best])) best = j;
                if (best >= 0) selected[best] = 1.0f;
            }
        }
    }
    float maximum = a->x[4] ? a->x[4][h] : -__builtin_inff();
    for (int j = 0; j < a->keys; ++j) {
        scores[j] = -__builtin_inff();
        if (vx_llm_attention_keep(a, batch, query, j)) {
            const float* key = a->x[1] +
                (((uint64_t)batch * a->keys + j) * a->kv_heads + kh) * a->key_dim;
            scores[j] = vx_llm_dot(q, key, a->key_dim) * a->scale;
            maximum = vx_llm_max(maximum, scores[j]);
        }
    }
    for (int j = 0; j < count; ++j) if (selected[j] != 0.0f) {
        const float* key = a->x[3] + ((uint64_t)batch * a->keys + j) * a->key_dim;
        maximum = vx_llm_max(maximum, vx_llm_dot(q, key, a->key_dim) * a->scale);
    }
    for (int c = 0; c < a->value_dim; ++c) output[c] = 0.0f;
    if (maximum == -__builtin_inff()) return;
    float denominator = a->x[4] ? expf(a->x[4][h] - maximum) : 0.0f;
    for (int j = 0; j < a->keys; ++j) if (scores[j] != -__builtin_inff()) {
        float probability = expf(scores[j] - maximum);
        const float* value = a->x[2] +
            (((uint64_t)batch * a->keys + j) * a->kv_heads + kh) * a->value_dim;
        denominator += probability;
        for (int c = 0; c < a->value_dim; ++c) output[c] += probability * value[c];
    }
    for (int j = 0; j < count; ++j) if (selected[j] != 0.0f) {
        const float* value = a->x[3] + ((uint64_t)batch * a->keys + j) * a->key_dim;
        float probability = expf(vx_llm_dot(q, value, a->key_dim) * a->scale - maximum);
        denominator += probability;
        for (int c = 0; c < a->value_dim; ++c) output[c] += probability * value[c];
    }
    if (denominator > 0.0f)
        for (int c = 0; c < a->value_dim; ++c) output[c] /= denominator;
}

VX_LLM_INLINE void vx_llm_compress_unit(const VxLLMOpArgs* a, uint64_t unit) {
    int batch = (int)(unit / a->sequence), w = (int)(unit % a->sequence);
    float* output = a->y[0] + unit * a->key_dim;
    for (int d = 0; d < a->key_dim; ++d) output[d] = 0.0f;
    if (w >= a->sequence / a->rate) return;
    int first = a->groups == 2 && w > 0 ? (w - 1) * a->rate : w * a->rate;
    int end = (w + 1) * a->rate;
    int width = a->groups * a->key_dim;
    float squared = 0.0f;
    for (int d = 0; d < a->key_dim; ++d) {
        float maximum = -__builtin_inff();
        for (int t = first; t < end; ++t) {
            int offset = a->groups == 2 && t >= w * a->rate ? a->key_dim : 0;
            uint64_t index = ((uint64_t)batch * a->sequence + t) * width + offset + d;
            float score = a->x[1][index] + a->x[2][(t % a->rate) * width + offset + d];
            maximum = vx_llm_max(maximum, score);
        }
        float total = 0.0f, denominator = 0.0f;
        for (int t = first; t < end; ++t) {
            int offset = a->groups == 2 && t >= w * a->rate ? a->key_dim : 0;
            uint64_t index = ((uint64_t)batch * a->sequence + t) * width + offset + d;
            float score = a->x[1][index] + a->x[2][(t % a->rate) * width + offset + d];
            float probability = expf(score - maximum);
            total += probability * a->x[0][index]; denominator += probability;
        }
        output[d] = total / denominator;
        squared += output[d] * output[d];
    }
    float inverse = 1.0f / sqrtf(squared / a->key_dim + a->eps);
    for (int d = 0; d < a->key_dim; ++d) output[d] *= inverse * a->x[3][d];
    if (a->x[4]) {
        int half=a->rotary_dim/2;
        const float* cosine=a->x[4]+(uint64_t)w*(a->interleaved?half:a->rotary_dim);
        const float* sine=a->x[5]+(uint64_t)w*(a->interleaved?half:a->rotary_dim);
        for (int pair=0;pair<half;pair++) {
            int left=a->rotary_offset+(a->interleaved?pair*2:pair);
            int right=a->rotary_offset+(a->interleaved?pair*2+1:pair+half);
            float x=output[left],y=output[right];
            output[left]=x*cosine[pair]-y*sine[pair];
            int second=a->interleaved?pair:pair+half;
            output[right]=y*cosine[second]+x*sine[second];
        }
    }
}

VX_LLM_INLINE void vx_llm_hyper_unit(const VxLLMOpArgs* a, uint64_t row,
                                    int head_only) {
    int m = a->streams, d = a->key_dim, width = m * d;
    const float* input = a->x[0] + row * width;
    float inverse = 1.0f / sqrtf(vx_llm_dot(input, input, width) / width + a->eps);
    float* collapsed = a->y[0] + row * (head_only ? d : m);
    if (head_only) for (int c = 0; c < d; ++c) collapsed[c] = 0.0f;
    for (int i = 0; i < m; ++i) {
        float raw = vx_llm_dot(input, a->x[1] + (uint64_t)i * width, width) * inverse;
        float pre = vx_llm_sigmoid(raw * a->x[3][0] + a->x[2][i]) + a->hc_eps;
        if (head_only) for (int c = 0; c < d; ++c) collapsed[c] += pre * input[i * d + c];
        else collapsed[i] = pre;
    }
    if (head_only) return;
    float* post = a->y[1] + row * m;
    float* comb = a->y[2] + row * m * m;
    for (int i = 0; i < m; ++i) {
        float raw = vx_llm_dot(input, a->x[1] + (uint64_t)(m + i) * width, width) * inverse;
        post[i] = 2.0f * vx_llm_sigmoid(raw * a->x[3][1] + a->x[2][m + i]);
        float maximum = -__builtin_inff();
        for (int j = 0; j < m; ++j) {
            int index = 2 * m + i * m + j;
            comb[i * m + j] = vx_llm_dot(input, a->x[1] + (uint64_t)index * width, width) *
                inverse * a->x[3][2] + a->x[2][index];
            maximum = vx_llm_max(maximum, comb[i * m + j]);
        }
        float denominator = 0.0f;
        for (int j = 0; j < m; ++j) {
            comb[i * m + j] = expf(comb[i * m + j] - maximum);
            denominator += comb[i * m + j];
        }
        for (int j = 0; j < m; ++j) comb[i * m + j] =
            comb[i * m + j] / denominator + a->hc_eps;
    }
    for (int iteration = 0; iteration < a->iterations; ++iteration) {
        if (iteration) for (int i = 0; i < m; ++i) {
            float total = a->hc_eps;
            for (int j = 0; j < m; ++j) total += comb[i * m + j];
            for (int j = 0; j < m; ++j) comb[i * m + j] /= total;
        }
        for (int j = 0; j < m; ++j) {
            float total = a->hc_eps;
            for (int i = 0; i < m; ++i) total += comb[i * m + j];
            for (int i = 0; i < m; ++i) comb[i * m + j] /= total;
        }
    }
}

VX_LLM_INLINE void vx_llm_hyper_residual_unit(const VxLLMOpArgs* a, uint64_t unit) {
    int c = (int)(unit % a->key_dim), stream = (int)((unit / a->key_dim) % a->streams);
    uint64_t row = unit / ((uint64_t)a->key_dim * a->streams);
    float result = a->x[2][row * a->streams + stream] * a->x[1][row * a->key_dim + c];
    for (int i = 0; i < a->streams; ++i)
        result += a->x[3][(row * a->streams + i) * a->streams + stream] *
            a->x[0][(row * a->streams + i) * a->key_dim + c];
    a->y[0][unit] = result;
}

VX_LLM_INLINE void vx_llm_rotary_unit(const VxLLMOpArgs* a, uint64_t unit) {
    int d = (int)(unit % a->key_dim);
    uint64_t row = unit / a->key_dim, token = row / a->heads;
    int local = d - a->rotary_offset, width = a->rotary_dim;
    if (local < 0 || local >= width) { a->y[0][unit] = a->x[0][unit]; return; }
    int frequency, partner; float sign;
    if (a->interleaved) {
        frequency = local / 2; partner = d + (local % 2 ? -1 : 1);
        sign = local % 2 ? 1.0f : -1.0f;
    } else {
        frequency = local; partner = d + (local < width / 2 ? width / 2 : -width / 2);
        sign = local < width / 2 ? -1.0f : 1.0f;
    }
    int table_width = a->interleaved ? width / 2 : width;
    uint64_t table = token * table_width + frequency;
    a->y[0][unit] = a->x[0][unit] * a->x[1][table] +
        sign * a->x[0][row * a->key_dim + partner] * a->x[2][table];
}

/* The Qwen indexer groups visible tokens separately for every query. */
VX_LLM_INLINE float vx_llm_qsa_block_score(const VxLLMOpArgs* a,int batch,int query,int block) {
    float key[512];
    for(int d=0;d<a->key_dim;d++) key[d]=0.0f;
    int visible=0,first=-1;
    for(int j=0;j<a->keys;j++) if(vx_llm_attention_keep(a,batch,query,j)) {
        if(visible/a->rate==block) {
            if(first<0) first=j;
            const float* raw=a->x[1]+((uint64_t)batch*a->keys+j)*a->key_dim;
            for(int d=0;d<a->key_dim;d++) key[d]+=raw[d]/a->rate;
        }
        visible++;
        if(visible>=(block+1)*a->rate) break;
    }
    float inverse=1.0f/sqrtf(vx_llm_dot(key,key,a->key_dim)/a->key_dim+a->eps);
    for(int d=0;d<a->key_dim;d++) key[d]*=inverse*a->x[2][d];
    const float* cosine=a->x[3]+((uint64_t)batch*a->keys+first)*a->rotary_dim;
    const float* sine=a->x[4]+((uint64_t)batch*a->keys+first)*a->rotary_dim;
    int half=a->rotary_dim/2;
    for(int d=0;d<half;d++) {
        float left=key[d],right=key[d+half];
        key[d]=left*cosine[d]-right*sine[d];
        key[d+half]=right*cosine[d+half]+left*sine[d+half];
    }
    float score=0.0f;
    for(int h=0;h<a->heads;h++) {
        const float* q=a->x[0]+(((uint64_t)batch*a->sequence+query)*a->heads+h)*a->key_dim;
        score+=vx_llm_max(vx_llm_dot(q,key,a->key_dim),0.0f);
    }
    return score/sqrtf((float)a->key_dim);
}
VX_LLM_INLINE void vx_llm_qsa_unit(const VxLLMOpArgs* a,uint64_t unit) {
    int query=(int)(unit%a->sequence),batch=(int)(unit/a->sequence);
    int32_t* output=(int32_t*)a->y[0]+unit*a->keys;
    int visible=0;
    for(int j=0;j<a->keys;j++) {output[j]=0;visible+=vx_llm_attention_keep(a,batch,query,j);}
    int blocks=visible/a->rate,rank=0;
    for(int j=0;j<a->keys;j++) if(vx_llm_attention_keep(a,batch,query,j)) {
        int block=rank/a->rate;
        int selected=block>=blocks||a->top_k>=blocks;
        if(!selected) {
            float score=vx_llm_qsa_block_score(a,batch,query,block);
            int higher=0;
            for(int other=0;other<blocks;other++) {
                float candidate=vx_llm_qsa_block_score(a,batch,query,other);
                higher+=(candidate>score||(candidate==score&&other<block));
                if(higher>=a->top_k) break;
            }
            selected=higher<a->top_k;
        }
        output[j]=selected;rank++;
    }
}

/* SparseIndexer is a whole-prefix compressed-bank selector. CPU caches one
 * row of scores; vx_llm_forward_f32 checks its allocation before dispatch.
 * CUDA recomputes them without unplanned device workspace. */
VX_LLM_INLINE float vx_llm_sparse_index_score(const VxLLMOpArgs* a,uint64_t row,int j) {
    if(a->x[3] && !((const int32_t*)a->x[3])[row*a->keys+j]) return -__builtin_inff();
    uint64_t batch=row/a->sequence;
    const float* k=a->x[1]+(batch*a->keys+j)*a->key_dim;
    float score=0.0f;
    for(int h=0;h<a->heads;h++) {
        const float* q=a->x[0]+(row*a->heads+h)*a->key_dim;
        score+=vx_llm_max(vx_llm_dot(q,k,a->key_dim),0.0f)*a->x[2][row*a->heads+h];
    }
    return score/sqrtf((float)a->key_dim*(float)a->heads);
}
VX_LLM_INLINE float vx_llm_sparse_block_score(const VxLLMOpArgs* a,uint64_t row,
        int block,int visible,const float* scores) {
    if(block==(visible-1)/a->window) return __builtin_inff();
    float best=-__builtin_inff();
    int begin=block*a->window,end=begin+(visible-begin<a->window?visible-begin:a->window);
#if defined(VX_LLM_CUDA_MATH)
    for(int j=begin;j<end;j++) best=vx_llm_max(best,scores?scores[j]:vx_llm_sparse_index_score(a,row,j));
#else
    (void)row;
    for(int j=begin;j<end;j++) best=vx_llm_max(best,scores[j]);
#endif
    return best;
}
VX_LLM_INLINE void vx_llm_sparse_index_unit(const VxLLMOpArgs* a,uint64_t row,float* scores) {
    int visible=((int)(row%a->sequence)+1)/a->rate;
    int32_t* out=(int32_t*)a->y[0]+row*a->keys;
    int32_t* candidates=a->y[1]?(int32_t*)a->y[1]+row*a->keys:0;
    if(scores) for(int j=0;j<visible;j++) scores[j]=vx_llm_sparse_index_score(a,row,j);
    for(int j=0;j<a->keys;j++) {
        out[j]=0;if(candidates)candidates[j]=0;
        if(j<visible) {
#if defined(VX_LLM_CUDA_MATH)
            float score=scores?scores[j]:vx_llm_sparse_index_score(a,row,j);
#else
            float score=scores[j];
#endif
            int higher=0;
            for(int other=0;other<visible;other++) {
#if defined(VX_LLM_CUDA_MATH)
                float value=scores?scores[other]:vx_llm_sparse_index_score(a,row,other);
#else
                float value=scores[other];
#endif
                higher+=value>score||(value==score&&other<j);
                if(higher>=a->top_k)break;
            }
            out[j]=higher<a->top_k;
        }
        if(candidates && visible>0 && j<a->keys/a->rate && j/a->window<=(visible-1)/a->window) {
            int block=j/a->window,blocks=(visible-1)/a->window+1;
            float value=vx_llm_sparse_block_score(a,row,block,visible,scores);
            int better=0;
            for(int other=0;other<blocks;other++) {
                float candidate=vx_llm_sparse_block_score(a,row,other,visible,scores);
                better+=candidate>value||(candidate==value&&other<block);
                if(better>=a->groups)break;
            }
            candidates[j]=value>-__builtin_inff()&&better<a->groups;
        }
    }
}

VX_LLM_INLINE void vx_llm_run_unit(const VxLLMOpArgs* a, uint64_t unit, float* scratch) {
    switch (a->kind) {
        case VX_OP_EMBEDDING_SHARD: {
            int32_t id=((const int32_t*)a->x[0])[unit/a->key_dim];
            int64_t row=(int64_t)id-a->rotary_offset;
            a->y[0][unit]=row>=0&&row<a->value_dim?a->x[1][(uint64_t)row*a->key_dim+unit%a->key_dim]:0.0f;
            break;
        }
        case VX_OP_SIGNED_SQRT: {
            float value=a->x[0][unit];
            a->y[0][unit]=value==0.0f?0.0f:(value>0.0f?1.0f:-1.0f)*sqrtf(vx_llm_max(fabsf(value),a->eps));
            break;
        }
        case VX_OP_QSA_INDEXER: vx_llm_qsa_unit(a,unit); break;
        case VX_OP_SPARSE_INDEXER: vx_llm_sparse_index_unit(a,unit,scratch); break;
        case VX_OP_GATED_DELTA_NET: vx_llm_delta_unit(a, unit, scratch); break;
        case VX_OP_SPARSE_ATTENTION: vx_llm_attention_unit(a, unit, scratch); break;
        case VX_OP_KV_COMPRESS: vx_llm_compress_unit(a, unit); break;
        case VX_OP_HYPER_CONNECTION: vx_llm_hyper_unit(a, unit, 0); break;
        case VX_OP_HYPER_HEAD: vx_llm_hyper_unit(a, unit, 1); break;
        case VX_OP_HYPER_RESIDUAL: vx_llm_hyper_residual_unit(a, unit); break;
        case VX_OP_ROTARY_EMBEDDING: vx_llm_rotary_unit(a, unit); break;
        default: break;
    }
}
#undef VX_LLM_INLINE
#endif
