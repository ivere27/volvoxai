#ifndef VOLVOXAI_LLM_OPS_H
#define VOLVOXAI_LLM_OPS_H

#include <stdint.h>
#include "../generated/operator_vocabulary.h"

/* Private numerical dispatch contract. All pointers name non-overlapping F32
 * tensors, except keep masks, position IDs and QSAIndexer output, which are I32. Shape
 * admission validates every extent and byte span before constructing this job.
 * The same descriptor is copied with device pointers for CUDA execution.
 *
 * GatedDeltaNet x: q,k,v,a,b,a_log,dt_bias,initial_state; y: out,state.
 * SparseAttention x: q,k,v,compressed_kv,sinks,index_q,index_k,index_weights.
 * SparseAttention x[5] alternatively holds I32 compressed_mask when
 * index_heads is zero; explicit masks and integrated indexer inputs are exclusive.
 * SparseIndexer x: q,k,index_weights,candidates(I32); y: out,candidates(I32).
 * Its window/groups fields hold candidate_block_size/candidate_top_k.
 * KVCompress x: projected_kv,projected_gate,position_bias,norm_weight.
 * HyperConnection x: streams,weight,bias,scale; y: pre,post,comb.
 * HyperHead x: streams,weight,bias,scale; y: collapsed streams.
 * HyperResidual x: streams,update,post,comb. RotaryEmbedding x: input,cos,sin.
 * PackedLinear x: input,packed_weight(I32),scale,bias(optional); y: out.
 * key_dim/value_dim are K/N, rate=storage group size (32, 64, 128 or 256),
 * normalize=operand bits (8, 16 or 32),
 * interleaved permits compensated TF32 only for explicit F32_FAST.
 * groups=weight_bits (2, 3, 4 or 8; zero defaults to 4 for private kernel callers).
 * batch*sequence is its active row count; x/y element spans retain physical
 * capacity during prefill so device ownership survives later whole-span reads.
 * The F32 graph boundary rounds activation operands inside the linear op.
 * CUDA lowering uses x[4]/x[5] for bounded prepared activation words / per-row
 * metadata. Those pointers are context workspace, not graph inputs. CUDA tile
 * and route choices live in its private launch plan, separate from heads/window.
 * Attention/GDN use [B,S,H,D]; compression uses [B,S,groups*D]; hyper
 * connections use [B,S,streams,D]. Compressed banks retain S capacity, with
 * only floor(S/rate) entries valid; no zero-sized or value-shaped tensor is
 * published. Their causal valid count is computed separately for each query.
 */
typedef struct VxLLMOpArgs {
    int32_t kind, batch, sequence, keys, heads, kv_heads, key_dim, value_dim;
    int32_t streams, groups, rate, window, top_k, iterations;
    int32_t index_heads, index_dim, rotary_dim, rotary_offset;
    int32_t normalize, causal, interleaved, mask_mode;
    float eps, hc_eps, scale;
    const float* x[8];
    float* y[3];
    const int32_t* mask;
    const int32_t* positions;
    uint64_t x_elements[8], y_elements[3], mask_elements, position_elements;
    /* Private incremental projection: Q/out may contain one token while KV
     * retains its capacity. A zero unit_count preserves whole-graph dispatch. */
    int32_t query_offset;
    uint64_t first_unit, unit_count;
} VxLLMOpArgs;

/* Returns zero only after all work has completed. No partial result is
 * published by the owning runtime when allocation or execution fails. */
int vx_llm_forward_f32(const VxLLMOpArgs* args);
uint64_t vx_llm_work_items(const VxLLMOpArgs* args);
uint64_t vx_llm_scratch_elements(const VxLLMOpArgs* args);

#endif
