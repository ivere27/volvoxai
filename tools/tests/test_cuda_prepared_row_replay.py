"""Execute the production typed row recipe with an independent CPU graph driver."""
from pathlib import Path
import re
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]


def record_source(source, name):
    pattern = r"typedef struct \{(?:(?!typedef struct \{).)*?\} " + name + r";"
    match = re.search(pattern, source, re.S)
    if not match:
        raise AssertionError("production record missing: " + name)
    return match.group() + "\n"


# Candidate002 source archive ef6891a678e174e2e50403bbceb36c60b3832d9d0698d2a684b849b23e0f5267.
# Preserve its actual helper as the equivalence oracle, independently of edits.
CANDIDATE002_ROW_POINTER = r'''
static int cuda_replay_row_pointer(const VxCudaReplayRow* row,
    CUdeviceptr pointer, int prefix, CudaReplayRowPointer* patch) {
    VxEngineState* state = vx_engine_state_current();
    int found = 0;
    for (int idx = 0; idx < row->node_count; idx++) {
        if (!row->nodes[idx]) continue;
        const VxIncrementalCudaCommand* command = &state->incremental_scratch.cuda_commands[idx];
        if (!command->valid) continue;
        for (int port = -1; port < command->input_count; port++) {
            const VxIncrementalCudaOperand* operand = port < 0 ? &command->output : &command->inputs[port];
            if (operand->prefix != prefix || operand->tensor_index < 0 ||
                operand->tensor_index >= state->tensor_count || !operand->width ||
                operand->width > SIZE_MAX / sizeof(uint32_t)) continue;
            const T* tensor = &state->tensors[operand->tensor_index];
            size_t stride = operand->width * sizeof(uint32_t);
            if (!tensor->data || (!prefix && (size_t)row->position > SIZE_MAX / stride)) continue;
            size_t offset = prefix ? 0 : (size_t)row->position * stride;
            if (tensor->numel <= 0 || (uint64_t)tensor->numel > SIZE_MAX / sizeof(uint32_t) ||
                offset >= (size_t)tensor->numel * sizeof(uint32_t)) continue;
            const void* selected = (const unsigned char*)tensor->data + offset;
            int slot_index = graph_find_containing_slot(selected, stride);
            if (slot_index < 0) continue;
            const CudaTensorSlot* slot = &graph_slots[slot_index];
            size_t displacement = (uintptr_t)selected - (uintptr_t)slot->host;
            if (!slot->device || slot->is_weight || displacement > slot->bytes ||
                stride > slot->bytes - displacement ||
                slot->device > UINT64_MAX - displacement || slot->device + displacement != pointer) continue;
            if (pointer < offset) continue;
            CUdeviceptr base = pointer - offset;
            if (found && (patch->base != base || patch->stride != (prefix ? 0 : stride))) return 0;
            patch->base = base; patch->stride = prefix ? 0 : stride; found = 1;
        }
    }
    return found;
}
'''
# Exact source92 row binder/replay oracle; full source SHA 849795cb05a1ccf13664f6c3fb30eaff876681e5411958e8f737dba271d7105d.
SOURCE92_ROW_BINDING = r'''
static int source92_command_bind(const VxIncrementalCudaCommand* command,
        int row, VxDecodeRowProjection* p) {
    if (row < 0 || row == INT32_MAX) return 0;
    memset(p, 0, sizeof(*p));
    if (g_decode_slots > 0) p->rows = g_decode_rows;
    else {
        p->scalar_position = row; p->scalar_length = row + 1;
        p->rows.slots = p->rows.live = 1; p->rows.key_capacity = row + 1;
        p->rows.positions = &p->scalar_position; p->rows.empty = &p->scalar_empty;
        p->rows.kv_lengths = &p->scalar_length; p->rows.pages = &p->scalar_pages;
        p->rows.unpaged = !vx_paged_bound_locked();
        if (!p->rows.unpaged && vx_paged_slot_pages_locked(0, &p->scalar_pages) != 1) return 0;
    }
    if (p->rows.slots <= 0 || !p->rows.positions || !p->rows.empty || !p->rows.kv_lengths)
        return 0;
    p->output = vx_decode_cuda_command_operand_bind(&command->output);
    /* Layout geometry includes the batch declaration. Changing it retains the
     * ordinary projection instead of silently reusing a different token axis. */
    if ((uint64_t)p->rows.slots * p->output.sequence * p->output.width !=
        (uint64_t)p->output.tensor->numel) return 0;
    for (int slot = 0; slot < p->rows.slots; slot++)
        if (!p->rows.empty[slot] && (p->rows.positions[slot] < 0 ||
            p->rows.positions[slot] >= p->output.sequence)) return 0;
    p->input_count = command->input_count;
    for (int i = 0; i < p->input_count; i++)
        p->inputs[i] = vx_decode_cuda_command_operand_bind(&command->inputs[i]);
    p->view = command->view; p->rope = command->rope;
    p->attention = command->attention; p->causal = command->causal;
    p->mask = command->mask_index >= 0 ? &g_t[command->mask_index] : NULL;
    p->mask_layout = command->mask_layout;
    p->keys = p->causal ? p->rows.key_capacity : command->fixed_keys;
    return 1;
}
int source92_prepared_row_replay(int row, int seal) {
    VxEngineState* state = vx_engine_state_current();
    const VxDecodeRowSet* rows = &g_decode_rows;
    if (!g_incremental_plan.cuda_commands_active || !g_incremental_scratch.cuda_commands ||
        !g_incremental_scratch.hybrid_nodes || rows->slots != 1 || rows->live != 1 ||
        !rows->positions || !rows->empty || !rows->kv_lengths || !rows->pages ||
        rows->empty[0] || rows->positions[0] != row || row < 0 || row == INT32_MAX ||
        rows->kv_lengths[0] != row + 1 || rows->key_capacity != row + 1 ||
        !state->decode_metadata_stage ||
        (size_t)row + 2u > state->decode_metadata_stage_bytes / sizeof(int32_t)) return 0;
    const VxDecodeSlotPages* pages = &rows->pages[0];
    if (pages->page_table) {
        if (pages->page_tokens <= 0 || pages->pages_per_slot <= 0 ||
            row / pages->page_tokens >= pages->pages_per_slot) return 0;
        for (int page = 0; page <= row / pages->page_tokens; page++)
            if (pages->page_table[page] != page) return 0;
    }
    int32_t position = 0;
    int have_position = 0, have_mask = 0;
    const T* mask = NULL;
    int mask_layout = 0;
    for (int idx = 0; idx < g_nn; idx++) {
        if (!g_incremental_scratch.hybrid_nodes[idx] || g_n[idx].skip || g_n[idx].disabled) continue;
        const VxIncrementalCudaCommand* command = &g_incremental_scratch.cuda_commands[idx];
        if (!command->valid || command->operator_kind != g_n[idx].operator_kind ||
            g_node_fusion[idx].id || g_n[idx].fuse_relu6) return 0;
        VxDecodeRowProjection p;
        if (!source92_command_bind(command, row, &p)) return 0;
        if (p.attention) {
            if (!p.causal || p.keys != row + 1) return 0;
            if (have_mask && (mask != p.mask || mask_layout != p.mask_layout)) return 0;
            if (!have_mask) {
                int source_keys = p.mask ? p.mask->shape[p.mask->ndim - 1] : p.keys;
                for (int key = 0; key < p.keys; key++)
                    if (!attention_mask_allowed(p.mask, p.mask_layout, 0, row, key,
                            p.output.sequence, source_keys, 0)) return 0;
                mask = p.mask; mask_layout = p.mask_layout; have_mask = 1;
            }
        }
        if (p.rope) {
            VxSequenceRoPE rope;
            if (!vx_sequence_rope_parameters(&g_n[idx], p.inputs[0].tensor, p.output.tensor, &rope)) return 0;
            int32_t value;
            if (rope.positions)
                memcpy(&value, (const unsigned char*)rope.positions->data + (size_t)row * sizeof(value), sizeof(value));
            else {
                if (rope.position_offset > INT32_MAX - row) return 0;
                value = rope.position_offset + row;
            }
            if (value < 0 || (have_position && value != position)) return 0;
            position = value; have_position = 1;
        }
    }
    VxCudaReplayRow recipe = {row, row + 1, position,
        g_incremental_scratch.hybrid_nodes, g_nn};
    if (seal) { cuda_graph_replay_row_seal(&recipe); return 0; }
    int result = cuda_graph_replay_row(&recipe);
    if (result > 0) {
        state->decode_cuda_mask_valid = state->decode_cuda_positions_valid = 0;
    }
    return result;
}

'''

PREAMBLE = r'''
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "sequence_runtime.h"
#include "cuda_engine.h"
#include "attention_mask.h"
typedef uintptr_t CUdeviceptr, CUfunction;
typedef void* CUgraphNode;
typedef void* CUgraph;
typedef void* CUgraphExec;
#define CUDA_GRAPH_MAX_TENSORS 32
#define CUDA_GRAPH_MAX_LAUNCHES 32
enum {CUDA_REPLAY_PASS_INELIGIBLE, CUDA_REPLAY_PASS_OBSERVE,
      CUDA_REPLAY_PASS_CAPTURE, CUDA_REPLAY_PASS_VALIDATE};
enum {CUDA_REPLAY_COMMAND_KERNEL, CUDA_REPLAY_COMMAND_UPLOAD, CUDA_REPLAY_COMMAND_COPY};
typedef struct {const char* symbol;const uint16_t* sizes;uint16_t count,bytes,storage;} VxCudaKernelArgumentABI;
'''

STATE = r'''
enum {CAP=8, WIDTH=4, VOCAB=11, TENSORS=11, NODES=7};
static VxEngineState engine;
VxEngineState* vx_engine_state_current(void) {return &engine;}
static CudaReplayState plan;
static CudaReplayState* cuda_replay_active=&plan;
#define cuda_replay (*cuda_replay_active)
static CudaTensorSlot graph_slots[CUDA_GRAPH_MAX_TENSORS];
static int graph_slot_count;
static uint64_t graph_slot_epoch=19;
static int cuda_forward_scope_active=1,cuda_forward_context_ready=1,cuda_forward_context_failed;
static int cuda_domain_enforced;
static int graph_find_containing_slot(const void* host,size_t bytes) {
    for(int i=0;i<graph_slot_count;i++) {
        uintptr_t a=(uintptr_t)host,b=(uintptr_t)graph_slots[i].host;
        if(a>=b && a-b<=graph_slots[i].bytes && bytes<=graph_slots[i].bytes-(a-b))return i;
    }
    return -1;
}
static int vx_paged_bound_locked(void) {return 0;}
static int vx_paged_slot_pages_locked(int slot,VxDecodeSlotPages* pages) {(void)slot;(void)pages;return 0;}
static T tensors[TENSORS],position_tensor;
static Node nodes[NODES];
static GraphNodeFusion fusion[NODES];
static VxIncrementalCudaCommand commands[NODES];
static unsigned char closure[NODES];
static int32_t token_host[CAP],token_device[CAP],selected_host[CAP],selected_device[CAP];
static int32_t position_host[CAP],metadata_host[128],metadata_device[128];
static float host_rows[6][CAP*VOCAB],device_rows[6][CAP*VOCAB];
static float embedding[VOCAB*WIDTH],value_weight[WIDTH*WIDTH],logit_weight[VOCAB*WIDTH];
static unsigned char argument_host[2048],argument_device[2048],upload_host[2048];
static int row_position,row_empty,row_length,page_table[CAP];
static VxDecodeSlotPages pages;
static int rope_calls,bind_refusals,rope_offsets[NODES],rope_uses_positions=1;
int vx_sequence_rope_parameters(Node* node,const T* in,const T* out,VxSequenceRoPE* value) {
    (void)in;(void)out;rope_calls++;memset(value,0,sizeof(*value));
    if(bind_refusals)return 0;
    value->positions=rope_uses_positions?&position_tensor:NULL;
    value->position_mode=1;value->sequence=CAP;
    value->position_offset=rope_offsets[node-nodes];return 1;
}
static const uint16_t embedding_sizes[]={8,8,8,4,4,8};
static const uint16_t gemv_sizes[]={8,8,8,8,4,4,4,8};
static const uint16_t rope_sizes[]={8,8,8,4,4,4,4,4,8,4,4,4,8};
static const uint16_t attention_sizes[]={8,8,8,8,8,4,4,4,4,4,4,4,4,4,4,4,8};
static const uint16_t argmax_sizes[]={8,8,4,4,4,8};
static const VxCudaKernelArgumentABI embedding_abi={"vx_cuda_embedding_f32",embedding_sizes,6,48,1};
static const VxCudaKernelArgumentABI gemv_abi={"vx_cuda_gemv_f32",gemv_sizes,8,64,1};
static const VxCudaKernelArgumentABI rope_abi={"vx_cuda_rope_heads_f32",rope_sizes,13,104,1};
static const VxCudaKernelArgumentABI attention_abi={"vx_cuda_cross_sdpa_gqa_f32",attention_sizes,17,136,1};
static const VxCudaKernelArgumentABI argmax_abi={"vx_cuda_argmax_f32_i32",argmax_sizes,6,48,1};
'''

FIXTURE = r'''
static void ptr(unsigned char* a,unsigned i,CUdeviceptr value) {memcpy(a+i*8u,&value,8);}
static void u32(unsigned char* a,unsigned i,uint32_t value) {memcpy(a+i*8u,&value,4);}
static CUdeviceptr getptr(const unsigned char* a,unsigned i) {CUdeviceptr v;memcpy(&v,a+i*8u,8);return v;}
static uint32_t get32(const unsigned char* a,unsigned i) {uint32_t v;memcpy(&v,a+i*8u,4);return v;}
static void tensor(int i,void* host,void* device,int width,int input,int weight) {
    tensors[i].data=host;tensors[i].numel=CAP*width;tensors[i].elem_size=4;
    tensors[i].dtype=i==0 || i==7?T_I32:T_F32;
    tensors[i].ndim=3;tensors[i].shape[0]=1;tensors[i].shape[1]=CAP;tensors[i].shape[2]=width;
    tensors[i].is_graph_input=input;
    graph_slots[i]=(CudaTensorSlot){.host=host,.device=(CUdeviceptr)device,.bytes=CAP*(size_t)width*4,.is_weight=weight};
}
static VxIncrementalCudaOperand operand(int tensor_index,int width,int prefix) {
    return (VxIncrementalCudaOperand){.tensor_index=tensor_index,.axis=1,.sequence=CAP,.prefix=prefix,.width=width};
}
static void command(int i,VxOperatorKind op,int output,int width,int input,int input_width) {
    commands[i]=(VxIncrementalCudaCommand){.valid=1,.operator_kind=op,.input_count=1,.mask_index=-1,
        .output=operand(output,width,0),.inputs={operand(input,input_width,0)}};
    nodes[i].operator_kind=op;closure[i]=1;plan.pass_writes[output]=1;
}
static unsigned char* kernel(const VxCudaKernelArgumentABI* abi) {
    CudaLaunchSignature* c=&plan.launches[plan.launch_count++];
    c->argument_abi=abi;c->argument_offset=plan.argument_bytes;
    c->storage_device=plan.argument_device+c->argument_offset;
    unsigned char* a=plan.argument_data+c->argument_offset;
    plan.argument_bytes+=abi->bytes;ptr(a,abi->count-1,c->storage_device);return a;
}
static void upload(int slot) {
    CudaLaunchSignature* c=&plan.launches[plan.launch_count++];c->command_kind=CUDA_REPLAY_COMMAND_UPLOAD;
    c->upload_host=graph_slots[slot].host;c->copy_destination=graph_slots[slot].device;
    c->copy_bytes=graph_slots[slot].bytes;c->upload_offset=plan.upload_bytes;
    memcpy((unsigned char*)plan.upload_data+c->upload_offset,c->upload_host,c->copy_bytes);
    plan.upload_bytes+=(uint32_t)c->copy_bytes;
}
static void initialize(void) {
    memset(&engine,0,sizeof(engine));memset(&plan,0,sizeof(plan));
    memset(tensors,0,sizeof(tensors));memset(nodes,0,sizeof(nodes));memset(commands,0,sizeof(commands));
    memset(closure,0,sizeof(closure));memset(argument_host,0,sizeof(argument_host));
    for(int i=0;i<6;i++)for(int j=0;j<CAP*VOCAB;j++)host_rows[i][j]=device_rows[i][j]=-31.25f;
    for(int i=0;i<CAP;i++){token_host[i]=i%VOCAB;selected_device[i]=selected_host[i]=-97;position_host[i]=3+i*7;page_table[i]=i;}
    for(int i=0;i<VOCAB*WIDTH;i++)embedding[i]=(float)((i*7)%19-9)*.063f;
    for(int i=0;i<WIDTH*WIDTH;i++)value_weight[i]=i/WIDTH==i%WIDTH?1.f:0.f;
    for(int i=0;i<VOCAB*WIDTH;i++)logit_weight[i]=(float)((i*11)%23-11)*.051f;
    tensor(0,token_host,token_device,1,1,0);
    for(int i=1;i<=6;i++)tensor(i,host_rows[i-1],device_rows[i-1],i==6?VOCAB:WIDTH,0,0);
    tensor(7,selected_host,selected_device,1,0,0);
    tensor(8,embedding,embedding,WIDTH,0,1);tensor(9,value_weight,value_weight,WIDTH,0,1);
    tensor(10,logit_weight,logit_weight,WIDTH,0,1);
    for(int i=8;i<TENSORS;i++){tensors[i].ndim=2;tensors[i].shape[0]=i==9?WIDTH:VOCAB;tensors[i].shape[1]=WIDTH;
        tensors[i].numel=tensors[i].shape[0]*WIDTH;graph_slots[i].bytes=(size_t)tensors[i].numel*4;}
    graph_slot_count=12;graph_slots[11]=(CudaTensorSlot){.host=metadata_host,.device=(CUdeviceptr)metadata_device,.bytes=sizeof(metadata_host)};
    engine.tensors=tensors;engine.tensor_count=TENSORS;engine.nodes=nodes;engine.node_count=NODES;engine.node_fusion=fusion;
    engine.incremental_scratch.cuda_commands=commands;engine.incremental_scratch.hybrid_nodes=closure;
    engine.incremental_plan.cuda_commands_active=1;engine.decode_slots=1;
    engine.decode_metadata_stage=(unsigned char*)metadata_host;engine.decode_metadata_stage_bytes=sizeof(metadata_host);
    pages=(VxDecodeSlotPages){.page_tokens=1,.pages_per_slot=CAP,.page_table=page_table};
    engine.decode_rows=(VxDecodeRowSet){.slots=1,.live=1,.key_capacity=1,.positions=&row_position,.empty=&row_empty,.kv_lengths=&row_length,.pages=&pages};
    position_tensor=(T){.data=(float*)position_host,.dtype=T_I32,.elem_size=4,.numel=CAP};
    plan.pass=CUDA_REPLAY_PASS_OBSERVE;plan.prepared=plan.dynamic_parameters=1;
    plan.argument_data=argument_host;plan.argument_device=(CUdeviceptr)argument_device;plan.argument_pinned=1;
    plan.upload_data=upload_host;plan.executable=&plan;
    row_position=row_empty=0;row_length=1;metadata_host[1]=position_host[0];
    command(0,VX_OP_EMBEDDING,1,WIDTH,0,1);
    command(1,VX_OP_ROPE,2,WIDTH,1,WIDTH);commands[1].rope=1;
    command(2,VX_OP_ROPE,3,WIDTH,1,WIDTH);commands[2].rope=1;
    command(3,VX_OP_LINEAR,4,WIDTH,1,WIDTH);
    command(4,VX_OP_CROSS_ATTENTION,5,WIDTH,2,WIDTH);commands[4].attention=commands[4].causal=1;
    commands[4].input_count=3;commands[4].inputs[1]=operand(3,WIDTH,1);commands[4].inputs[2]=operand(4,WIDTH,1);
    command(5,VX_OP_LINEAR,6,VOCAB,5,WIDTH);command(6,VX_OP_ARG_MAX,7,1,6,VOCAB);
    upload(0);upload(11);
    unsigned char* a=kernel(&embedding_abi);ptr(a,0,(CUdeviceptr)token_device);ptr(a,1,(CUdeviceptr)embedding);ptr(a,2,(CUdeviceptr)device_rows[0]);
    u32(a,3,1);u32(a,4,WIDTH);
    for(int i=1;i<=2;i++){a=kernel(&rope_abi);ptr(a,0,(CUdeviceptr)device_rows[0]);ptr(a,1,(CUdeviceptr)&metadata_device[1]);ptr(a,2,(CUdeviceptr)device_rows[i]);
        u32(a,3,WIDTH);u32(a,4,1);u32(a,5,WIDTH);u32(a,6,1);u32(a,7,WIDTH);u32(a,11,2);}
    a=kernel(&gemv_abi);ptr(a,0,(CUdeviceptr)device_rows[0]);ptr(a,1,(CUdeviceptr)value_weight);ptr(a,3,(CUdeviceptr)device_rows[3]);u32(a,4,WIDTH);u32(a,5,WIDTH);
    a=kernel(&attention_abi);ptr(a,0,(CUdeviceptr)device_rows[1]);ptr(a,1,(CUdeviceptr)device_rows[2]);ptr(a,2,(CUdeviceptr)device_rows[3]);ptr(a,4,(CUdeviceptr)device_rows[4]);
    u32(a,5,1);u32(a,6,1);u32(a,7,WIDTH);u32(a,8,WIDTH);u32(a,9,1);u32(a,10,1);u32(a,11,WIDTH);u32(a,12,1);
    a=kernel(&gemv_abi);ptr(a,0,(CUdeviceptr)device_rows[4]);ptr(a,1,(CUdeviceptr)logit_weight);ptr(a,3,(CUdeviceptr)device_rows[5]);u32(a,4,WIDTH);u32(a,5,VOCAB);
    a=kernel(&argmax_abi);ptr(a,0,(CUdeviceptr)device_rows[5]);ptr(a,1,(CUdeviceptr)selected_device);u32(a,2,1);u32(a,3,VOCAB);
    rope_calls=bind_refusals=0;rope_uses_positions=1;memset(rope_offsets,0,sizeof(rope_offsets));
}
/* Independent graph executor: uploads and kernels consume the sealed images,
 * never current runtime operands. Device argument storage is uploaded once. */
static void execute_graph(void) {
    memcpy(argument_device,argument_host,plan.argument_bytes);
    for(unsigned i=0;i<plan.launch_count;i++) {
        const CudaLaunchSignature* c=&plan.launches[i];
        if(c->command_kind==CUDA_REPLAY_COMMAND_UPLOAD){memcpy((void*)c->copy_destination,upload_host+c->upload_offset,c->copy_bytes);continue;}
        const unsigned char* a=(const unsigned char*)c->storage_device;
        if(c->argument_abi==&embedding_abi) {
            int token=*(const int32_t*)getptr(a,0);float* o=(float*)getptr(a,2);
            memcpy(o,embedding+token*WIDTH,WIDTH*4);
        } else if(c->argument_abi==&rope_abi) {
            const float* in=(const float*)getptr(a,0);float* out=(float*)getptr(a,2);
            int position=*(const int32_t*)getptr(a,1);
            for(int j=0;j<WIDTH;j+=2){double angle=position*(j?.1:1.);double co=cos(angle),si=sin(angle);
                out[j]=(float)(in[j]*co-in[j+1]*si);out[j+1]=(float)(in[j]*si+in[j+1]*co);}
        } else if(c->argument_abi==&gemv_abi) {
            const float* in=(const float*)getptr(a,0),*w=(const float*)getptr(a,1);float* out=(float*)getptr(a,3);
            for(unsigned j=0;j<get32(a,5);j++){double sum=0;for(unsigned k=0;k<get32(a,4);k++)sum+=in[k]*w[j*get32(a,4)+k];out[j]=(float)sum;}
        } else if(c->argument_abi==&attention_abi) {
            const float* q=(const float*)getptr(a,0),*k=(const float*)getptr(a,1),*v=(const float*)getptr(a,2);float* out=(float*)getptr(a,4);
            double scores[CAP],total=0;unsigned keys=get32(a,6);assert(keys>0 && keys<=CAP);
            for(unsigned key=0;key<keys;key++){double sum=0;for(int j=0;j<WIDTH;j++)sum+=q[j]*k[key*WIDTH+j];scores[key]=exp(sum*.5);total+=scores[key];}
            for(int j=0;j<WIDTH;j++){double sum=0;for(unsigned key=0;key<keys;key++)sum+=scores[key]/total*v[key*WIDTH+j];out[j]=(float)sum;}
        } else {assert(c->argument_abi==&argmax_abi);const float* in=(const float*)getptr(a,0);int32_t* out=(int32_t*)getptr(a,1);
            int best=0;for(unsigned j=1;j<get32(a,3);j++)if(in[j]>in[best])best=(int)j;*out=best;}
    }
}
static double ref_key[CAP][WIDTH],ref_value[CAP][WIDTH];
static void reference(int row,int token,int position) {
    double q[WIDTH],attention[WIDTH],logits[VOCAB],scores[CAP],total=0;
    for(int j=0;j<WIDTH;j+=2){double a=embedding[token*WIDTH+j],b=embedding[token*WIDTH+j+1],angle=position*(j?.1:1.);
        q[j]=a*cos(angle)-b*sin(angle);q[j+1]=a*sin(angle)+b*cos(angle);}
    for(int j=0;j<WIDTH;j++){ref_key[row][j]=q[j];ref_value[row][j]=embedding[token*WIDTH+j];}
    for(int key=0;key<=row;key++){double dot=0;for(int j=0;j<WIDTH;j++)dot+=q[j]*ref_key[key][j];scores[key]=exp(dot*.5);total+=scores[key];}
    for(int j=0;j<WIDTH;j++){attention[j]=0;for(int key=0;key<=row;key++)attention[j]+=scores[key]/total*ref_value[key][j];}
    int best=0;
    for(int j=0;j<VOCAB;j++){logits[j]=0;for(int k=0;k<WIDTH;k++)logits[j]+=attention[k]*logit_weight[j*WIDTH+k];if(logits[j]>logits[best])best=j;
        assert(fabs(logits[j]-device_rows[5][row*VOCAB+j])<2e-6);}
    assert(selected_device[row]==best);
    for(int key=0;key<=row;key++)for(int j=0;j<WIDTH;j++) {
        assert(fabs(ref_key[key][j]-device_rows[2][key*WIDTH+j])<1e-6);
        assert(ref_value[key][j]==device_rows[3][key*WIDTH+j]);
    }
    for(int future=row+1;future<CAP;future++)assert(selected_device[future]==-97);
}
static VxCudaReplayRow recipe(int row) {return (VxCudaReplayRow){row,row+1,position_host[row],closure,NODES};}
static void assert_miss(VxCudaReplayRow* row) {
    unsigned char saved[sizeof(argument_host)],uploads[sizeof(upload_host)];int32_t metadata[128];CudaTensorSlot slots[CUDA_GRAPH_MAX_TENSORS];
    memcpy(saved,argument_host,sizeof(saved));memcpy(uploads,upload_host,sizeof(uploads));memcpy(metadata,metadata_host,sizeof(metadata));memcpy(slots,graph_slots,sizeof(slots));
    assert(cuda_graph_replay_row(row)==0);
    assert(!memcmp(saved,argument_host,sizeof(saved)) && !memcmp(uploads,upload_host,sizeof(uploads)) && !memcmp(metadata,metadata_host,sizeof(metadata)) && !memcmp(slots,graph_slots,sizeof(slots)));
}
int main(void) {
    /* A complete admitted cold row still submits ordinary commands while its
     * graph is observed, before graph construction and the final stream wait. */
    initialize();cuda_domain_enforced=1;
    plan.launch_count=plan.argument_bytes=plan.upload_bytes=0;plan.executable=NULL;
    assert(!vx_runtime_cuda_prepared_row_replay_locked(0,0));
    assert(plan.pass==CUDA_REPLAY_PASS_OBSERVE && !plan.launch_count && !plan.launch_index);
    cuda_domain_enforced=0;
    initialize();assert(!vx_runtime_cuda_prepared_row_replay_locked(0,1));assert(plan.row_ready);
    execute_graph();reference(0,token_host[0],position_host[0]);
    plan.pass=CUDA_REPLAY_PASS_VALIDATE;
    for(int row=1;row<CAP;row++) {
        row_position=row;row_length=engine.decode_rows.key_capacity=row+1;plan.launch_index=0;memset(plan.pass_writes,0,sizeof(plan.pass_writes));
        token_host[row]=(row*3+1)%VOCAB;int token=token_host[row];graph_slots[0].host_dirty=1;
        float past[2][CAP*WIDTH];memcpy(past[0],device_rows[2],sizeof(past[0]));memcpy(past[1],device_rows[3],sizeof(past[1]));
        assert(vx_runtime_cuda_prepared_row_replay_locked(row,0)==1);
        assert(plan.launch_index==plan.launch_count && !graph_slots[0].host_dirty && !graph_slots[0].device_dirty);
        assert(!engine.decode_cuda_mask_valid && !engine.decode_cuda_positions_valid);
        token_host[row]=10;metadata_host[row+1]=-991; /* delayed copies must keep the completed snapshot */
        execute_graph();reference(row,token,position_host[row]);
        assert(!memcmp(past[0],device_rows[2],row*WIDTH*4) && !memcmp(past[1],device_rows[3],row*WIDTH*4));
        for(int i=1;i<=7;i++)assert(graph_slots[i].device_dirty && plan.pass_writes[i]);
    }
    /* Every rejected guard leaves all owned values, uploads and coherence alone. */
    initialize();VxCudaReplayRow first=recipe(0);cuda_graph_replay_row_seal(&first);assert(plan.row_ready);plan.pass=CUDA_REPLAY_PASS_VALIDATE;
    VxCudaReplayRow next=recipe(1);
    graph_slot_epoch++;assert_miss(&next);graph_slot_epoch--;
    closure[2]=0;assert_miss(&next);closure[2]=1;
    next.node_count--;assert_miss(&next);next.node_count++;
    next.keys++;assert_miss(&next);next.keys--;
    plan.upload_inflight=1;assert_miss(&next);plan.upload_inflight=0;
    plan.launch_index=1;assert_miss(&next);plan.launch_index=0;
    plan.failed=1;assert_miss(&next);plan.failed=0;
    plan.argument_device++;assert_miss(&next);plan.argument_device--;
    plan.launches[0].upload_slot=0;assert_miss(&next);plan.launches[0].upload_slot=1;
    plan.launches[0].upload_offset=UINT32_MAX;assert_miss(&next);plan.launches[0].upload_offset=0;
    engine.decode_metadata_stage_bytes=4;assert_miss(&next);engine.decode_metadata_stage_bytes=sizeof(metadata_host);
    next.position=INT32_MAX-1;next.keys=INT32_MAX;assert_miss(&next);
    /* Unsupported semantics stay on the ordinary path; no backend call occurs. */
    row_position=1;row_length=engine.decode_rows.key_capacity=2;
    page_table[1]=3;assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));page_table[1]=1;
    row_empty=1;assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));row_empty=0;
    engine.decode_rows.live=0;assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));engine.decode_rows.live=1;
    commands[4].causal=0;assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));commands[4].causal=1;
    int32_t mask[CAP];for(int i=0;i<CAP;i++)mask[i]=1;mask[0]=0;
    T original=tensors[8];tensors[8]=(T){.data=(float*)mask,.dtype=T_I32,.ndim=1,.numel=CAP,.shape={CAP}};
    commands[4].mask_index=8;commands[4].mask_layout=1;
    assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));commands[4].mask_index=-1;tensors[8]=original;
    rope_uses_positions=0;rope_offsets[1]=0;rope_offsets[2]=2;assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));
    rope_uses_positions=1;position_host[1]=-1;assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));position_host[1]=10;
    commands[0].operator_kind=VX_OP_ADD;assert(!vx_runtime_cuda_prepared_row_replay_locked(1,0));commands[0].operator_kind=VX_OP_EMBEDDING;
    /* A changed/unknown ABI or a staging output cannot be sealed accidentally. */
    initialize();first=recipe(0);VxCudaKernelArgumentABI unknown=embedding_abi;unknown.symbol="new_kernel";plan.launches[2].argument_abi=&unknown;
    cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    initialize();first=recipe(0);unsigned char* a=argument_host+plan.launches[2].argument_offset;ptr(a,2,(CUdeviceptr)argument_device);
    cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    initialize();first=recipe(0);a=argument_host+plan.launches[6].argument_offset;ptr(a,3,(CUdeviceptr)metadata_device);
    cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    initialize();first=recipe(0);plan.launches[0].upload_host=host_rows[0];
    cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    /* A positive row proof cannot authorize a different prefix view. The
     * embedding output is already proved before this attention argument. */
    initialize();first=recipe(0);a=argument_host+plan.launches[6].argument_offset;
    ptr(a,1,(CUdeviceptr)device_rows[0]);
    cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    /* Completed seals do not retain proofs across storage/owner changes. */
    for(int change=0;change<3;change++) {
        initialize();first=recipe(0);cuda_graph_replay_row_seal(&first);assert(plan.row_ready);
        graph_slot_epoch++;
        if(change==0)graph_slots[1].device=0;
        if(change==1)graph_slots[1].bytes=WIDTH*4-1;
        if(change==2)graph_slots[1].is_weight=1;
        cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    }
    /* A colliding unowned address must miss rather than borrow a prior proof.
     * This address is never dereferenced by the owner-resolution path. */
    initialize();first=recipe(0);
    CUdeviceptr earlier=(CUdeviceptr)device_rows[0],collision=UINT64_C(0x8000000000000000);
    unsigned bucket=(unsigned)((earlier*UINT64_C(0x9e3779b97f4a7c15))>>59);
    while((unsigned)((collision*UINT64_C(0x9e3779b97f4a7c15))>>59)!=bucket)collision++;
    a=argument_host+plan.launches[5].argument_offset;ptr(a,0,collision);
    cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    /* Every matching alias remains part of the first complete proof. */
    initialize();first=recipe(0);tensors[2].data=tensors[1].data;
    graph_slots[2].host=graph_slots[1].host;graph_slots[2].device=graph_slots[1].device;
    commands[1].output.width=WIDTH/2;
    cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);
    puts("PASS prepared CUDA rows: production typed bindings, 8-row F64 full-vocabulary/KV oracle, delayed input snapshots, closure/storage/page/mask/RoPE/ABI no-mutation guards");
}
'''


ROW_GEOMETRY_CHECKS = r'''

static int comparisons;
static void canonicalize_scalar(VxDecodeRowProjection* p) {
    if(p->rows.positions==&p->scalar_position)p->rows.positions=NULL;
    if(p->rows.empty==&p->scalar_empty)p->rows.empty=NULL;
    if(p->rows.kv_lengths==&p->scalar_length)p->rows.kv_lengths=NULL;
    if(p->rows.pages==&p->scalar_pages)p->rows.pages=NULL;
}
static void compare_bind(int row,int expected) {
    VxDecodeRowProjection a,b;memset(&a,0xa5,sizeof(a));memset(&b,0xa5,sizeof(b));
    paged_calls=0;int av=source92_command_bind(&commands[0],row,&a),calls=paged_calls;
    paged_calls=0;int bv=vx_decode_cuda_command_bind(&commands[0],row,&b);
    assert(av==expected && bv==av && calls==paged_calls);
    canonicalize_scalar(&a);canonicalize_scalar(&b);
    assert(!memcmp(&a,&b,sizeof(a)));comparisons++;
}
static void binding_cases(void) {
    int positions[2],empty[2]={0,0},lengths[2]={1,1};VxDecodeSlotPages slot_pages[2]={{0}};
    for(int slots=1;slots<=2;slots++)for(int sequence=1;sequence<=8;sequence*=8)
    for(int width=1;width<=4;width*=4)for(int change=0;change<8;change++) {
        initialize();engine.decode_slots=slots;
        engine.decode_rows=(VxDecodeRowSet){.slots=slots,.live=slots,.key_capacity=sequence,
            .positions=positions,.empty=empty,.kv_lengths=lengths,.pages=slot_pages};
        positions[0]=positions[1]=sequence-1;empty[0]=empty[1]=0;
        commands[0].output.sequence=sequence;commands[0].output.width=(size_t)width;
        tensors[1].numel=slots*sequence*width;
        int expected=change==0 || change==5;
        if(change==1)tensors[1].numel--;
        if(change==2)tensors[1].numel++;
        if(change==3)positions[slots-1]=-1;
        if(change==4)positions[slots-1]=sequence;
        if(change==5){positions[slots-1]=INT32_MAX;empty[slots-1]=1;}
        if(change==6)engine.decode_rows.positions=NULL;
        if(change==7)engine.decode_rows.kv_lengths=NULL;
        compare_bind(0,expected);
    }
    initialize();compare_bind(-1,0);compare_bind(INT32_MAX,0);
    for(int change=0;change<8;change++) {
        initialize();
        if(change==0)engine.decode_rows.slots=0;
        if(change==1)tensors[1].numel=0;
        if(change==2)tensors[1].numel=-1;
        if(change==3)commands[0].output.sequence=0;
        if(change==4)commands[0].output.width=SIZE_MAX;
        if(change==5)engine.decode_rows.empty=NULL;
        if(change==6)engine.decode_rows.live=0; /* Binder retains geometry-only behavior. */
        if(change==7)engine.decode_rows.pages=NULL; /* Replay, not binder, owns this guard. */
        compare_bind(0,change>=6);
    }
    for(int bound=0;bound<2;bound++)for(int result=-1;result<=1;result++) {
        initialize();engine.decode_slots=0;paged_bound=bound;paged_result=result;
        compare_bind(0,!bound || result==1);
    }
}
typedef struct {
    int status,rope_calls,paged_calls,mask_valid,positions_valid;
    CudaReplayState plan;CudaTensorSlot slots[CUDA_GRAPH_MAX_TENSORS];
    unsigned char arguments[sizeof(argument_host)],uploads[sizeof(upload_host)];
    int32_t metadata[128];
} Outcome;
static void snapshot(Outcome* out,int status) {
    memset(out,0,sizeof(*out));out->status=status;out->rope_calls=rope_calls;out->paged_calls=paged_calls;
    out->mask_valid=engine.decode_cuda_mask_valid;out->positions_valid=engine.decode_cuda_positions_valid;
    out->plan=plan;memcpy(out->slots,graph_slots,sizeof(out->slots));
    memcpy(out->arguments,argument_host,sizeof(out->arguments));memcpy(out->uploads,upload_host,sizeof(out->uploads));
    memcpy(out->metadata,metadata_host,sizeof(out->metadata));
}
static int invoke(int candidate,int row,int seal) {
    return candidate?vx_runtime_cuda_prepared_row_replay_locked(row,seal):source92_prepared_row_replay(row,seal);
}
static void replay_case(int candidate,int change,int seal,Outcome* out) {
    initialize();memset(fusion,0,sizeof(fusion));graph_slot_epoch=19;cuda_forward_context_failed=0;cuda_domain_enforced=0;
    paged_calls=paged_bound=0;paged_result=1;engine.decode_cuda_mask_valid=engine.decode_cuda_positions_valid=1;
    memset(metadata_host,0,sizeof(metadata_host));memset(upload_host,0,sizeof(upload_host));
    VxCudaReplayRow first=recipe(0);cuda_graph_replay_row_seal(&first);assert(plan.row_ready);
    plan.pass=CUDA_REPLAY_PASS_VALIDATE;plan.launch_index=0;
    int row=1;row_position=row;row_length=engine.decode_rows.key_capacity=row+1;
    switch(change) {
    case 0:break;
    case 1:tensors[1].numel--;break;
    case 2:tensors[7].numel=0;break;
    case 3:engine.decode_rows.live=0;break;
    case 4:row_empty=1;break;
    case 5:row_position=2;break;
    case 6:row=-1;break;
    case 7:row=INT32_MAX;break;
    case 8:engine.decode_rows.slots=2;break;
    case 9:engine.decode_rows.pages=NULL;break;
    case 10:page_table[1]=3;break;
    case 11:row_length=1;break;
    case 12:engine.decode_metadata_stage_bytes=4;break;
    case 13:commands[0].output.sequence=1;break;
    case 14:commands[0].output.width=SIZE_MAX;break;
    case 15:commands[4].causal=0;break;
    case 16:commands[4].fixed_keys=1;commands[4].causal=0;break;
    case 17:position_host[1]=-1;break;
    case 18:rope_uses_positions=0;rope_offsets[1]=INT32_MAX;break;
    case 19:rope_uses_positions=0;rope_offsets[2]=2;break;
    case 20:bind_refusals=1;break;
    case 21:commands[1].output.sequence=1;break;
    case 22:commands[4].output.sequence=1;break;
    case 23:commands[0].valid=0;break;
    case 24:commands[0].operator_kind=VX_OP_ADD;break;
    case 25:nodes[0].fuse_relu6=1;break;
    case 26:fusion[0].id=1;break;
    case 27:engine.decode_slots=0;break;
    case 28:engine.decode_slots=0;paged_bound=1;paged_result=0;break;
    case 29:engine.decode_slots=0;paged_bound=1;paged_result=1;break;
    case 30:plan.failed=1;break;
    case 31:graph_slot_epoch++;break;
    case 32:graph_slots[1].bytes=1;break;
    case 33:plan.upload_inflight=1;break;
    case 34:commands[0].view=1;break;
    case 35:{
        static int32_t mask[CAP];for(int i=0;i<CAP;i++)mask[i]=1;mask[0]=0;
        tensors[8]=(T){.data=(float*)mask,.dtype=T_I32,.ndim=1,.numel=CAP,.shape={CAP}};
        commands[4].mask_index=8;commands[4].mask_layout=1;break;}
    case 36:assert(invoke(candidate,row,0)==1);plan.launch_index=0;tensors[1].numel--;break;
    case 37:assert(invoke(candidate,row,0)==1);plan.launch_index=0;row=2;break;
    case 38:assert(invoke(candidate,row,0)==1);plan.launch_index=0;
        row=row_position=2;row_length=engine.decode_rows.key_capacity=3;break;
    default:assert(0);
    }
    snapshot(out,invoke(candidate,row,seal));
}
int main(void) {
    binding_cases();
    Outcome* a=malloc(sizeof(*a)),*b=malloc(sizeof(*b));assert(a && b);
    for(int seal=0;seal<2;seal++)for(int change=0;change<39;change++) {
        replay_case(0,change,seal,a);replay_case(1,change,seal,b);
        if(memcmp(a,b,sizeof(*a)))fprintf(stderr,"geometry differential seal=%d change=%d\n",seal,change);
        assert(!memcmp(a,b,sizeof(*a)));comparisons++;
        if(!seal && change==0)assert(a->status==1);
        if(!seal && change==36)assert(a->status==0); /* Fresh numel, no saved verdict. */
        if(!seal && change==37)assert(a->status==0); /* Cursor must match the committed rows. */
        if(!seal && change==38)assert(a->status==1);
        if(seal && change==0)assert(a->plan.row_ready && !a->status);
    }
    free(a);free(b);
    printf("PASS current row geometry: %d source92 full-binding/replay comparisons; batch/live/scalar paging, attention/RoPE, second-call mutation and complete owned failure state\n",comparisons);
}

'''

# Authentic source004 seal preserves the pre-index fallback oracle independently.
SOURCE004_ROW_SEAL = r'''
void source004_row_seal(const VxCudaReplayRow* row) {
    CudaReplayState* replay = cuda_replay_active;
    if (!cuda_replay_row_scope(row) || !replay->argument_data || !replay->launch_count ||
        (replay->pass == CUDA_REPLAY_PASS_VALIDATE && replay->launch_index != replay->launch_count)) return;
    replay->row_ready = 0;
    /* Positive proofs live only within this owned seal. Collisions miss;
     * prefix views retain their complete, distinct owner proof. */
    struct { CUdeviceptr pointer, base; size_t stride; } proved[32] = {{0}};
    VxEngineState* state = vx_engine_state_current();
    int metadata_slot = graph_find_containing_slot(state->decode_metadata_stage, state->decode_metadata_stage_bytes);
    for (uint32_t index = 0; index < replay->launch_count; index++) {
        CudaLaunchSignature* command = &replay->launches[index];
        command->row_pointer_count = 0;
        if (command->command_kind == CUDA_REPLAY_COMMAND_UPLOAD) {
            if (!command->upload_host || !command->copy_bytes) return;
            int slot_index = graph_find_containing_slot(command->upload_host, command->copy_bytes);
            if (slot_index < 0 || graph_slots[slot_index].device != command->copy_destination ||
                graph_slots[slot_index].host != command->upload_host ||
                graph_slots[slot_index].bytes != command->copy_bytes) return;
            int authoritative = slot_index == metadata_slot || graph_slots[slot_index].is_weight;
            for (int tensor = 0; !authoritative && tensor < state->tensor_count; tensor++)
                authoritative = state->tensors[tensor].is_graph_input &&
                    state->tensors[tensor].data == command->upload_host;
            if (!authoritative) return;
            command->upload_slot = (uint16_t)(slot_index + 1);
            continue;
        }
        if (command->command_kind != CUDA_REPLAY_COMMAND_KERNEL) return;
        unsigned rows, prefixes;
        if (!cuda_replay_row_layout(command, &rows, &prefixes)) return;
        if (command->argument_offset > replay->argument_bytes ||
            command->argument_abi->bytes > replay->argument_bytes - command->argument_offset) return;
        unsigned char* arguments = replay->argument_data + command->argument_offset;
        for (unsigned argument = 0; argument + 1 < command->argument_abi->count; argument++) {
            if (!((rows | prefixes) & (1u << argument))) continue;
            if (command->argument_abi->sizes[argument] != sizeof(CUdeviceptr)) return;
            CUdeviceptr pointer; memcpy(&pointer, arguments + argument * 8u, sizeof(pointer));
            CudaReplayRowPointer patch = {0};
            int prefix = !!(prefixes & (1u << argument));
            unsigned bucket = (unsigned)((pointer * UINT64_C(0x9e3779b97f4a7c15)) >> 59);
            if (!prefix && proved[bucket].stride && proved[bucket].pointer == pointer) {
                patch.base = proved[bucket].base; patch.stride = proved[bucket].stride;
            } else {
                if (!cuda_replay_row_pointer(row, pointer, prefix, &patch)) return;
                if (!prefix) {
                    proved[bucket].pointer = pointer; proved[bucket].base = patch.base;
                    proved[bucket].stride = patch.stride;
                }
            }
            if (rows & (1u << argument)) {
                if (command->row_pointer_count >= 4) return;
                patch.argument = (uint16_t)argument;
                command->row_pointers[command->row_pointer_count++] = patch;
            }
        }
        if (command->row_keys_argument != UINT16_MAX) {
            uint32_t keys, mask_mode; CUdeviceptr mask;
            memcpy(&keys, arguments + 6u * 8u, sizeof(keys));
            memcpy(&mask, arguments + 3u * 8u, sizeof(mask));
            memcpy(&mask_mode, arguments + 15u * 8u, sizeof(mask_mode));
            if (keys != (uint32_t)row->keys || mask || mask_mode) return;
        }
        if (command->row_position_argument != UINT16_MAX) {
            CUdeviceptr positions; uint32_t position_mode;
            memcpy(&positions, arguments + 8u, sizeof(positions));
            memcpy(&position_mode, arguments + 11u * 8u, sizeof(position_mode));
            if (metadata_slot < 0 || position_mode != 2u ||
                graph_slots[metadata_slot].device > UINT64_MAX - (size_t)row->keys * sizeof(int32_t) ||
                positions != graph_slots[metadata_slot].device + (size_t)row->keys * sizeof(int32_t)) return;
            command->row_position_base = graph_slots[metadata_slot].device;
        }
    }
    replay->row_node_count = row->node_count;
    memcpy(replay->row_nodes, row->nodes, (size_t)row->node_count);
    memcpy(replay->row_writes, replay->pass_writes, sizeof(replay->row_writes));
    replay->row_slot_epoch = graph_slot_epoch;
    replay->row_ready = 1;
}

'''

PROOF_INDEX_STATE = r'''
enum {N=512,CAP=64,WIDTH=8,MAX_TENSORS=N*2+2};
#define DEVICE_BASE UINT64_C(0x400000000000)
#define DEVICE_ROW(i) (DEVICE_BASE+(uint64_t)(i)*CAP*WIDTH*sizeof(float))
static VxEngineState engine;
VxEngineState* vx_engine_state_current(void) {return &engine;}
static CudaReplayState plan,*cuda_replay_active=&plan;
#define cuda_replay (*cuda_replay_active)
static CudaTensorSlot graph_slots[CUDA_GRAPH_MAX_TENSORS];static int graph_slot_count;
static uint64_t graph_slot_epoch=19;
static int cuda_forward_scope_active=1,cuda_forward_context_ready=1,cuda_forward_context_failed;
#define CUDA_GRAPH_SLOT_HASH_CAPACITY (CUDA_GRAPH_MAX_TENSORS*2)
static uint16_t graph_slot_hash[CUDA_GRAPH_SLOT_HASH_CAPACITY];static int graph_slot_hash_valid;
typedef struct {int domain_enforced;struct {uint64_t epoch;uint16_t order[CUDA_GRAPH_MAX_TENSORS];
    uint16_t count;int disjoint;} owner_index;} CudaContextState;
static CudaContextState context;
static CudaContextState* cuda_context_state_current(void) {return &context;}
static T tensors[MAX_TENSORS];static VxIncrementalCudaCommand commands[N];static unsigned char closure[N];
static float host_rows[MAX_TENSORS][CAP*WIDTH];
static unsigned char argument_host[N*56];static VxCudaReplayRow row;
static const uint16_t unary_sizes[]={8,8,4,4,4,4,8};
static const VxCudaKernelArgumentABI unary_abi={"vx_cuda_unary_f32",unary_sizes,7,56,1};

'''

PROOF_INDEX_FIXTURE = r'''
static void ptr(unsigned char* a,unsigned i,CUdeviceptr value) {memcpy(a+i*8u,&value,8);}
static void setup(int kind) {
    memset(&engine,0,sizeof(engine));memset(&plan,0,sizeof(plan));memset(&context,0,sizeof(context));
    memset(graph_slots,0,sizeof(graph_slots));memset(commands,0,sizeof(commands));memset(tensors,0,sizeof(tensors));
    memset(argument_host,0,sizeof(argument_host));memset(closure,1,sizeof(closure));
    engine.tensors=tensors;engine.tensor_count=(kind==0 || kind>=3)?N+1:kind==1?N*2:1;
    engine.node_count=N;engine.incremental_scratch.cuda_commands=commands;graph_slot_count=engine.tensor_count;
    for(int i=0;i<graph_slot_count;i++) {
        tensors[i].data=host_rows[i];tensors[i].numel=CAP*WIDTH;
        graph_slots[i]=(CudaTensorSlot){.host=host_rows[i],.device=DEVICE_ROW(i),
            .bytes=sizeof(host_rows[i]),.capacity=sizeof(host_rows[i]),.domain_span=1};
    }
    for(int i=0;i<N;i++) {
        int input=kind==0?i:kind==1?i*2:kind==3?(i<16?0:i-15):kind==4?(i<4?0:i-3):0,output=(kind==0 || kind>=3)?i+1:kind==1?i*2+1:0;
        commands[i]=(VxIncrementalCudaCommand){.valid=1,.input_count=1,
            .output={.tensor_index=output,.width=WIDTH},.inputs={{.tensor_index=input,.width=WIDTH}}};
        CudaLaunchSignature* c=&plan.launches[i];c->argument_abi=&unary_abi;c->argument_offset=i*56;
        c->storage_device=UINT64_C(0x90000000)+c->argument_offset;
        ptr(argument_host+i*56,0,DEVICE_ROW(input)+31*WIDTH*sizeof(float));
        ptr(argument_host+i*56,1,DEVICE_ROW(output)+31*WIDTH*sizeof(float));
    }
    plan.pass=CUDA_REPLAY_PASS_OBSERVE;plan.prepared=plan.dynamic_parameters=1;
    plan.argument_data=argument_host;plan.argument_device=UINT64_C(0x90000000);
    plan.argument_bytes=sizeof(argument_host);plan.launch_count=N;
    row=(VxCudaReplayRow){.position=31,.keys=32,.nodes=closure,.node_count=N};
    context.domain_enforced=1;graph_rebuild_slot_hash();
}

static int comparisons;
static void check_seal(int expected) {
    CudaReplayState* initial=malloc(sizeof(plan));
    CudaReplayState* oracle=malloc(sizeof(plan));assert(initial && oracle);
    *initial=plan;source004_row_seal(&row);*oracle=plan;plan=*initial;
    cuda_graph_replay_row_seal(&row);
    assert(!memcmp(oracle,&plan,sizeof(plan)) && plan.row_ready==expected);
    free(initial);free(oracle);comparisons++;
}
static void compare_all_entries(void) {
    CudaReplayRowProof proofs[1024]={{0}};
    assert(cuda_replay_row_proofs_build(&row,proofs));
    for(unsigned i=0;i<1024;i++)if(proofs[i].pointer) {
        CudaReplayRowPointer full={0};
        int accepted=cuda_replay_row_pointer(&row,proofs[i].pointer,proofs[i].prefix,&full);
        assert(accepted==!proofs[i].ambiguous);
        if(accepted)assert(full.base==proofs[i].base && full.stride==proofs[i].stride);
    }
}
static void refresh_arguments(void) {
    for(int i=0;i<N;i++) {
        VxIncrementalCudaCommand* c=&commands[i];
        int ids[]={c->inputs[0].tensor_index,c->output.tensor_index};
        size_t widths[]={c->inputs[0].width,c->output.width};
        for(int p=0;p<2;p++)ptr(argument_host+i*56,p,
            graph_slots[ids[p]].device+(size_t)row.position*widths[p]*sizeof(uint32_t));
    }
}
int main(void) {
    assert(sizeof(CudaReplayRowProof[1024])==32768);
    for(int route=0;route<5;route++){setup(route);compare_all_entries();check_seal(1);}
    /* Every one of1024 distinct keys fits. Missing keys in a completely full
     * table are bounded misses, not a reused unrelated proof. */
    setup(1);CudaReplayRowProof proofs[1024]={{0}};
    assert(cuda_replay_row_proofs_build(&row,proofs));int count=0;
    for(unsigned i=0;i<1024;i++)count+=!!proofs[i].pointer;
    assert(count==1024 && !cuda_replay_row_proof_find(proofs,UINT64_C(0xffffffffff000000),0));
    check_seal(1);
    /*1026 proven keys disable the entire index. The full ordinary local32
     * fallback retains the exact recipe, including all prior partial writes. */
    for(int i=N*2;i<N*2+2;i++) {
        tensors[i].data=host_rows[i];tensors[i].numel=CAP*WIDTH;
        graph_slots[i]=(CudaTensorSlot){.host=host_rows[i],.device=DEVICE_ROW(i),.bytes=sizeof(host_rows[i])};
    }
    engine.tensor_count=graph_slot_count=N*2+2;commands[0].input_count=3;
    commands[0].inputs[1]=(VxIncrementalCudaOperand){.tensor_index=N*2,.width=WIDTH};
    commands[0].inputs[2]=(VxIncrementalCudaOperand){.tensor_index=N*2+1,.width=WIDTH};
    graph_slot_epoch++;graph_rebuild_slot_hash();memset(proofs,0,sizeof(proofs));
    assert(!cuda_replay_row_proofs_build(&row,proofs));check_seal(1);
    /* Real chosen owners with64 same hash buckets retain complete address
     * identity through probing. These are live slot proofs, not seeded hits. */
    setup(1);CUdeviceptr key=UINT64_C(0x600000000000);
    for(int i=0;i<64;i++) {
        key+=4096;
        while((unsigned)((key*UINT64_C(0x9e3779b97f4a7c15))>>54)!=0)key+=256;
        graph_slots[i].device=key-(size_t)row.position*WIDTH*sizeof(uint32_t);
    }
    refresh_arguments();compare_all_entries();check_seal(1);
    /* Prefix0/1 is part of the key; prefixes with unlike widths still use
     * actual width-sensitive owners and may merge the same base/zero stride. */
    setup(2);commands[0].inputs[0].prefix=1;commands[1].inputs[0].prefix=1;
    commands[1].inputs[0].width=WIDTH/2;row.position=0;row.keys=1;refresh_arguments();
    compare_all_entries();memset(proofs,0,sizeof(proofs));
    assert(cuda_replay_row_proofs_build(&row,proofs));
    CudaReplayRowProof* prefix=cuda_replay_row_proof_find(proofs,DEVICE_ROW(0),1);
    assert(prefix && prefix->pointer && !prefix->ambiguous && !prefix->stride);
    CudaReplayRowProof* ordinary=cuda_replay_row_proof_find(proofs,DEVICE_ROW(0),0);
    assert(ordinary && ordinary->pointer && !ordinary->ambiguous && ordinary->stride==WIDTH*4);
    /* Exact nested owners remain width-sensitive. A too-small exact view
     * leaves the enclosing owner authoritative, while a fitting narrow view
     * uses its own device identity. Both are authentic selected host proofs. */
    for(int width=WIDTH/2;width<=WIDTH;width+=WIDTH/2) {
        setup(1);row.position=0;row.keys=1;tensors[0].data=&host_rows[0][1];
        commands[0].inputs[0].width=(size_t)width;
        graph_slots[N*2]=(CudaTensorSlot){.host=&host_rows[0][1],.device=DEVICE_ROW(N*2),
            .bytes=WIDTH/2*sizeof(float),.capacity=WIDTH/2*sizeof(float),.domain_span=1};
        graph_slot_count++;graph_slot_epoch++;graph_rebuild_slot_hash();refresh_arguments();
        ptr(argument_host,0,width==WIDTH/2?DEVICE_ROW(N*2):DEVICE_ROW(0)+sizeof(float));
        compare_all_entries();check_seal(1);
    }
    /* A late row ambiguity cannot erase earlier successfully written patches. */
    setup(1);row.position=0;row.keys=1;commands[N-1].output.width=WIDTH/2;
    graph_slots[N*2-1].device=graph_slots[N*2-2].device;
    refresh_arguments();compare_all_entries();check_seal(0);
    /* Independent live bytes, ownership and stale future patches remain guards. */
    for(int mode=0;mode<8;mode++) {
        setup(1);int last=N*2-1;
        if(mode==0)graph_slots[last].bytes=1;
        if(mode==1)graph_slots[last].device=0;
        if(mode==2)graph_slots[last].is_weight=1;
        if(mode==3)tensors[last].data=NULL;
        if(mode==4)tensors[last].numel=0;
        if(mode==5)commands[N-1].output.width=0;
        if(mode==6)commands[N-1].output.tensor_index=-1;
        if(mode==7)commands[N-1].valid=0;
        plan.launches[N-1].row_pointer_count=4;
        for(int p=0;p<4;p++)plan.launches[N-1].row_pointers[p]=(CudaReplayRowPointer){DEVICE_ROW(last),WIDTH*4,(uint16_t)p};
        compare_all_entries();check_seal(0);
    }
    /* Checked host reconstruction aborts the index and leaves the original
     * resolver as fallback. An unrepresentable private host is not dereferenced. */
    setup(0);tensors[0].data=(float*)(UINTPTR_MAX-3);memset(proofs,0,sizeof(proofs));
    assert(!cuda_replay_row_proofs_build(&row,proofs));
    printf("PASS bounded operand proof index: %d complete004fallback/partial recipes; actual owner/fullproof agreement,1024full/1026overflow,64livecollisions,prefix/ambiguity/liveguards\n",comparisons);
    return 0;
}

'''

class PreparedRowReplayTests(unittest.TestCase):
    def test_pointer_resolution_matches_candidate002_with_bounded_owner_lookups(self):
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        old = CANDIDATE002_ROW_POINTER.lstrip("\n")
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),
                         "29500301a0a2e9a674b60e35f3b09b544e51cb486f3340c88bc993be5c87e6b9")
        old = old.replace("cuda_replay_row_pointer(", "candidate002_row_pointer(", 1)
        engine = (ROOT / "native/src/backends/cuda_engine.c").read_text()
        source = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        records = "".join(record_source(engine, name) for name in
                          ("CudaTensorSlot", "CudaReplayRowPointer"))
        state = r'''
enum {N=512, CAP=64, WIDTH=8};
static VxEngineState engine;
VxEngineState* vx_engine_state_current(void) {return &engine;}
static T tensors[N];static VxIncrementalCudaCommand commands[N];
static unsigned char closure[N];
static float host_rows[N][CAP*WIDTH],device_rows[N][CAP*WIDTH];
static CudaTensorSlot graph_slots[N];static int graph_slot_count,disjoint;
static uint64_t owner_lookups,old_lookups,new_lookups;static int comparisons;
/* This driver preserves the product lookup's exact-owner priority and
 * smallest containing live extent. Nominal disjoint owners use binary search;
 * overlap cases use the scan, independently of the pointer prefilter. */
static int graph_find_containing_slot(const void* host,size_t bytes) {
    owner_lookups++;
    uintptr_t a=(uintptr_t)host;
    if(!host || !bytes || a>UINTPTR_MAX-bytes)return -1;
    if(disjoint) {
        size_t lo=0,hi=(size_t)graph_slot_count;
        while(lo<hi){size_t mid=lo+(hi-lo)/2;if((uintptr_t)graph_slots[mid].host<=a)lo=mid+1;else hi=mid;}
        if(!lo)return -1;
        int i=(int)lo-1;const CudaTensorSlot* s=&graph_slots[i];uintptr_t delta=a-(uintptr_t)s->host;
        return s->device && delta<=s->bytes && bytes<=s->bytes-delta?i:-1;
    }
    int best=-1;
    for(int i=0;i<graph_slot_count;i++) {
        const CudaTensorSlot* s=&graph_slots[i];uintptr_t b=(uintptr_t)s->host;
        if(!s->host || !s->device || !s->bytes || b>UINTPTR_MAX-s->bytes || a<b || a-b>s->bytes || bytes>s->bytes-(a-b))continue;
        if(a==b)return i;
        if(best<0 || s->bytes<graph_slots[best].bytes)best=i;
    }
    return best;
}
static VxCudaReplayRow row;
static int candidate002_row_pointer(const VxCudaReplayRow*,CUdeviceptr,int,CudaReplayRowPointer*);
static int cuda_replay_row_pointer(const VxCudaReplayRow*,CUdeviceptr,int,CudaReplayRowPointer*);
static void initialize(void) {
    memset(&engine,0,sizeof(engine));memset(tensors,0,sizeof(tensors));memset(commands,0,sizeof(commands));
    memset(graph_slots,0,sizeof(graph_slots));memset(closure,1,sizeof(closure));
    engine.tensors=tensors;engine.tensor_count=N;engine.node_count=N;engine.incremental_scratch.cuda_commands=commands;
    graph_slot_count=N;disjoint=1;row=(VxCudaReplayRow){.position=31,.keys=32,.nodes=closure,.node_count=N};
    for(int i=0;i<N;i++) {
        tensors[i].data=host_rows[i];tensors[i].numel=CAP*WIDTH;
        commands[i].valid=1;commands[i].input_count=1;
        commands[i].inputs[0]=commands[i].output=(VxIncrementalCudaOperand){.tensor_index=i,.width=WIDTH};
        graph_slots[i]=(CudaTensorSlot){.host=host_rows[i],.device=(CUdeviceptr)device_rows[i],
            .bytes=sizeof(host_rows[i]),.capacity=sizeof(host_rows[i])};
    }
}
static void compare(CUdeviceptr pointer,int prefix,int expected) {
    CudaReplayRowPointer before={0},after={0};
    owner_lookups=0;int a=candidate002_row_pointer(&row,pointer,prefix,&before);old_lookups+=owner_lookups;
    owner_lookups=0;int b=cuda_replay_row_pointer(&row,pointer,prefix,&after);new_lookups+=owner_lookups;
    assert(a==b && b==expected);
    if(a)assert(before.base==after.base && before.stride==after.stride);
    comparisons++;
}
static void small(void) {
    initialize();disjoint=0;graph_slot_count=1;row.node_count=1;row.position=1;row.keys=2;
}
'''
        checks = r'''
int main(void) {
    initialize();old_lookups=new_lookups=0;
    for(int i=0;i<N;i++)compare((CUdeviceptr)&device_rows[i][31*WIDTH],0,1);
    assert(old_lookups==(uint64_t)N*N*2 && new_lookups==(uint64_t)N*2);
    printf("PASS 512-node production pointer seal: %llu candidate002 owner lookups -> %llu; authentic helper agreement\n",
        (unsigned long long)old_lookups,(unsigned long long)new_lookups);
    /* Prefix and current logical extent, not reserved capacity. */
    small();commands[0].output.prefix=commands[0].inputs[0].prefix=1;compare((CUdeviceptr)device_rows[0],1,1);
    small();graph_slots[0].bytes=WIDTH*4-1;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();compare(graph_slots[0].device+graph_slots[0].bytes,0,0);compare(graph_slots[0].device-1,0,0);
    small();graph_slots[0].host=NULL;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();graph_slots[0].device=0;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();graph_slots[0].is_weight=1;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();closure[0]=0;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();commands[0].valid=0;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();commands[0].output.tensor_index=commands[0].inputs[0].tensor_index=N;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();commands[0].output.width=commands[0].inputs[0].width=SIZE_MAX;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();commands[0].output.width=commands[0].inputs[0].width=0;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();tensors[0].data=NULL;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();tensors[0].numel=0;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();tensors[0].numel=-1;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    small();tensors[0].numel=WIDTH;compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    /* Interior tensor views keep the actual owner's displacement. */
    small();tensors[0].data=&host_rows[0][4];compare((CUdeviceptr)&device_rows[0][WIDTH+4],0,1);
    /* A nested host owner changes ownership even when the outer device also
     * contains the pointer. The original host resolver remains authoritative. */
    small();graph_slot_count=2;tensors[0].data=&host_rows[0][4];
    graph_slots[1]=(CudaTensorSlot){.host=&host_rows[0][4],.device=(CUdeviceptr)device_rows[1],.bytes=128};
    compare((CUdeviceptr)&device_rows[1][WIDTH],0,1);compare((CUdeviceptr)&device_rows[0][WIDTH+4],0,0);
    graph_slots[1].device=(CUdeviceptr)&device_rows[0][4];compare((CUdeviceptr)&device_rows[0][WIDTH+4],0,1);
    /* Distinct host aliases of the same device span are all considered.
     * Their equal selected address but different row stride is ambiguous. */
    small();row.node_count=2;graph_slot_count=2;
    graph_slots[1].device=graph_slots[0].device;
    commands[1].output.width=commands[1].inputs[0].width=WIDTH/2;
    tensors[1].data=&host_rows[1][WIDTH/2];compare((CUdeviceptr)&device_rows[0][WIDTH],0,0);
    commands[1].output.width=commands[1].inputs[0].width=WIDTH;tensors[1].data=host_rows[1];
    compare((CUdeviceptr)&device_rows[0][WIDTH],0,1);
    /* The same selected host address can come from incompatible views of
     * one owner. Prefix views have no cursor stride and are unambiguous. */
    small();row.node_count=2;graph_slot_count=1;
    tensors[0].data=&host_rows[0][4];tensors[1].data=host_rows[0];
    commands[0].output.width=commands[0].inputs[0].width=4;
    compare((CUdeviceptr)&device_rows[0][8],0,0);
    commands[0].output.prefix=commands[0].inputs[0].prefix=commands[1].output.prefix=commands[1].inputs[0].prefix=1;
    tensors[0].data=tensors[1].data=host_rows[0];compare((CUdeviceptr)device_rows[0],1,1);
    /* Difference bounds cannot manufacture a wrapped host address. A device
     * extent near UINTPTR_MAX preserves the original start-address proof. */
    small();graph_slots[0].host=(const void*)(UINTPTR_MAX-3);graph_slots[0].device=16;graph_slots[0].bytes=32;
    compare(24,0,0);
    small();commands[0].output.width=commands[0].inputs[0].width=1;row.position=3;
    graph_slots[0].device=UINTPTR_MAX-16;compare(UINTPTR_MAX-4,0,1);
    printf("PASS %d candidate002 equivalence cases: live extents, interior/nested/overlapping owners, aliases, stride ambiguity, prefix, invalid views and overflow\n",comparisons);
}
'''
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-row-pointer-") as temp:
            path, binary = Path(temp) / "test.c", Path(temp) / "test"
            path.write_text(PREAMBLE + records + state + old +
                            function_source(source, "cuda_replay_row_pointer") + checks)
            command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=1",
                       "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0",
                       "-DVOLVOXAI_ENABLE_METAL=0"]
            command += [f"-I{ROOT / path}" for path in includes]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(path), "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS 512-node production pointer seal", result.stdout)
            print(result.stdout.strip())

    def test_current_row_geometry_matches_full_binding(self):
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        engine_source = (ROOT / "native/src/backends/cuda_engine.c").read_text()
        backend = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        row_source = (ROOT / "native/src/runtime/engine_runtime_row_cuda.inc").read_text()
        projection = (ROOT / "native/src/runtime/decode_projection.h").read_text()
        records = "".join(record_source(engine_source, name) for name in
                          ("CudaTensorSlot", "CudaReplayRowPointer", "CudaLaunchSignature", "CudaReplayState"))
        records += record_source(backend, "CudaReplayRowProof")
        records += "".join(record_source(projection, name) for name in
                           ("VxDecodeRowOperand", "VxDecodeRowProjection"))
        functions = "".join(function_source(backend, name) for name in
                            ("cuda_argument_abi_valid", "cuda_replay_row_layout", "cuda_replay_row_pointer",
                             "cuda_replay_row_scope", "cuda_replay_row_proof_find", "cuda_replay_row_proofs_build",
                             "cuda_graph_replay_row_seal", "cuda_graph_replay_row"))
        functions += "".join(function_source(row_source, name) for name in
                             ("vx_decode_cuda_command_operand_bind", "vx_decode_cuda_row_geometry_current",
                              "vx_decode_cuda_command_bind", "vx_runtime_cuda_prepared_row_replay_locked"))
        state = STATE.replace("static int vx_paged_bound_locked(void) {return 0;}",
                              "static int paged_bound,paged_result=1,paged_calls;\n"
                              "static int vx_paged_bound_locked(void) {return paged_bound;}")
        state = state.replace("static int vx_paged_slot_pages_locked(int slot,VxDecodeSlotPages* pages) {(void)slot;(void)pages;return 0;}",
                              "static int vx_paged_slot_pages_locked(int slot,VxDecodeSlotPages* pages) {(void)slot;(void)pages;paged_calls++;return paged_result;}")
        checks = ROW_GEOMETRY_CHECKS
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-current-row-geometry-") as temp:
            unit, binary = Path(temp) / "test.c", Path(temp) / "test"
            unit.write_text(PREAMBLE + records + state + functions + SOURCE92_ROW_BINDING +
                            FIXTURE[:FIXTURE.index("int main(void) {")] + checks)
            command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                       "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                       "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
            command += [f"-I{ROOT / path}" for path in includes]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), str(ROOT / "native/src/runtime/attention_mask.c"), "-lm", "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS current row geometry", result.stdout)
            print(result.stdout.strip())

    def test_production_recipe_numerics_and_guards(self):
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        engine_source = (ROOT / "native/src/backends/cuda_engine.c").read_text()
        backend = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        row_source = (ROOT / "native/src/runtime/engine_runtime_row_cuda.inc").read_text()
        projection = (ROOT / "native/src/runtime/decode_projection.h").read_text()
        records = "".join(record_source(engine_source, name) for name in
                          ("CudaTensorSlot", "CudaReplayRowPointer", "CudaLaunchSignature", "CudaReplayState"))
        records += record_source(backend, "CudaReplayRowProof")
        records += "".join(record_source(projection, name) for name in
                           ("VxDecodeRowOperand", "VxDecodeRowProjection"))
        functions = "".join(function_source(backend, name) for name in
                            ("cuda_argument_abi_valid", "cuda_replay_row_layout", "cuda_replay_row_pointer",
                             "cuda_replay_row_scope", "cuda_replay_row_proof_find", "cuda_replay_row_proofs_build",
                             "cuda_graph_replay_row_seal", "cuda_graph_replay_row"))
        functions += "".join(function_source(row_source, name) for name in
                             ("vx_decode_cuda_command_operand_bind", "vx_decode_cuda_row_geometry_current",
                              "vx_decode_cuda_command_bind", "vx_runtime_cuda_prepared_row_replay_locked"))
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-row-reuse-") as temp:
            unit, binary = Path(temp) / "test.c", Path(temp) / "test"
            unit.write_text(PREAMBLE + records + STATE + functions + FIXTURE)
            command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                       "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                       "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
            command += [f"-I{ROOT / path}" for path in includes]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), str(ROOT / "native/src/runtime/attention_mask.c"), "-lm", "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS prepared CUDA rows", result.stdout)
            print(result.stdout.strip())

    def test_packed_decode_records_patch_only_row_operands(self):
        """Control-plane oracle; CUDA arithmetic is covered by the GPU suite."""
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        engine_source = (ROOT / "native/src/backends/cuda_engine.c").read_text()
        backend = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        row_source = (ROOT / "native/src/runtime/engine_runtime_row_cuda.inc").read_text()
        projection = (ROOT / "native/src/runtime/decode_projection.h").read_text()
        records = "".join(record_source(engine_source, name) for name in
                          ("CudaTensorSlot", "CudaReplayRowPointer", "CudaLaunchSignature", "CudaReplayState"))
        records += record_source(backend, "CudaReplayRowProof")
        records += "".join(record_source(projection, name) for name in
                           ("VxDecodeRowOperand", "VxDecodeRowProjection"))
        functions = "".join(function_source(backend, name) for name in
                            ("cuda_argument_abi_valid", "cuda_replay_row_layout", "cuda_replay_row_pointer",
                             "cuda_replay_row_scope", "cuda_replay_row_proof_find", "cuda_replay_row_proofs_build",
                             "cuda_graph_replay_row_seal", "cuda_graph_replay_row"))
        functions += "".join(function_source(row_source, name) for name in
                             ("vx_decode_cuda_command_operand_bind", "vx_decode_cuda_row_geometry_current",
                              "vx_decode_cuda_command_bind", "vx_runtime_cuda_prepared_row_replay_locked"))
        state = STATE.replace("VOCAB=11", "VOCAB=8192")
        checks = r'''
static const uint16_t packed_sizes[]={8,8,8,8,8,4,4,4,4,4,8,4,8};
static const VxCudaKernelArgumentABI packed_abi={"vx_cuda_packed_linear_decode_f32",packed_sizes,13,104,1};
static float packed_workspace[33];
static int executed_graphs,prepared_rows,computed_rows;
static void packed_setup(int bits,int activation,int prepared) {
    initialize();memset(fusion,0,sizeof(fusion));
    cuda_forward_context_failed=cuda_domain_enforced=0;graph_slot_epoch=19;
    memset(commands,0,sizeof(commands));memset(closure,0,sizeof(closure));
    memset(plan.pass_writes,0,sizeof(plan.pass_writes));
    memset(argument_host,0xa5,sizeof(argument_host));
    memset(packed_workspace,0,sizeof(packed_workspace));
    memset(plan.launches,0,sizeof(plan.launches));
    plan.launch_count=plan.argument_bytes=plan.upload_bytes=0;
    command(5,VX_OP_PACKED_LINEAR,6,VOCAB,1,WIDTH);
    for(int row=0;row<CAP;row++)for(int k=0;k<WIDTH;k++)
        device_rows[0][row*WIDTH+k]=(float)(1000*row+k);
    for(int pass=0;pass<(prepared?2:1);pass++) {
        unsigned char* a=kernel(&packed_abi);
        ptr(a,0,(CUdeviceptr)device_rows[0]);ptr(a,1,(CUdeviceptr)value_weight);
        ptr(a,2,(CUdeviceptr)embedding);ptr(a,3,(CUdeviceptr)logit_weight);
        ptr(a,4,(CUdeviceptr)device_rows[5]);u32(a,5,WIDTH);u32(a,6,VOCAB);
        u32(a,7,(uint32_t)bits);u32(a,8,(uint32_t)activation);u32(a,9,activation==8?4u:0u);
        ptr(a,10,prepared?(CUdeviceptr)packed_workspace:0);u32(a,11,prepared && !pass?1u:0u);
        unsigned rows=0,prefixes=0;
        CudaLaunchSignature* c=&plan.launches[pass];
        assert(cuda_replay_row_layout(c,&rows,&prefixes));
        assert(rows==((1u<<0)|(1u<<4)) && !prefixes);
        assert(c->row_keys_argument==UINT16_MAX && c->row_position_argument==UINT16_MAX);
    }
    executed_graphs=prepared_rows=computed_rows=0;
}
/* Consume only copied device argument records. This emulator checks row
 * addressing and phase order; it deliberately does not emulate CUDA math. */
static void packed_execute(void) {
    memcpy(argument_device,argument_host,plan.argument_bytes);executed_graphs++;
    for(unsigned i=0;i<plan.launch_count;i++) {
        const CudaLaunchSignature* c=&plan.launches[i];
        const unsigned char* a=(const unsigned char*)c->storage_device;
        assert(c->argument_abi==&packed_abi && get32(a,5)==WIDTH && get32(a,6)==VOCAB);
        const float* input=(const float*)getptr(a,0);float* output=(float*)getptr(a,4);
        float* workspace=(float*)getptr(a,10);
        if(get32(a,11)==1) {
            assert(workspace && get32(a,8)==8);memcpy(workspace,input,WIDTH*sizeof(float));prepared_rows++;
        } else {
            assert(!get32(a,11));const float* values=workspace?workspace:input;
            for(int j=0;j<VOCAB;j++)output[j]=values[j%WIDTH]+(float)(get32(a,7)+get32(a,8)+7);
            computed_rows++;
        }
    }
}
static void packed_check_records(const unsigned char* sealed,int row) {
    for(unsigned i=0;i<plan.launch_count;i++) {
        CudaLaunchSignature* c=&plan.launches[i];const unsigned char* a=argument_host+c->argument_offset;
        const unsigned char* original=sealed+c->argument_offset;
        assert(c->row_pointer_count==2 && c->row_pointers[0].argument==0 && c->row_pointers[1].argument==4);
        assert(getptr(a,0)==(CUdeviceptr)&device_rows[0][row*WIDTH]);
        assert(getptr(a,4)==(CUdeviceptr)&device_rows[5][row*VOCAB]);
        for(unsigned slot=0;slot<13;slot++)if(slot!=0 && slot!=4)
            assert(!memcmp(a+slot*8,original+slot*8,8));
    }
}
static void packed_miss(const VxCudaReplayRow* row) {
    CudaReplayState saved=plan;CudaTensorSlot slots[CUDA_GRAPH_MAX_TENSORS];
    unsigned char arguments[sizeof(argument_host)],uploads[sizeof(upload_host)];int32_t metadata[128];
    memcpy(arguments,argument_host,sizeof(arguments));memcpy(uploads,upload_host,sizeof(uploads));
    memcpy(metadata,metadata_host,sizeof(metadata));memcpy(slots,graph_slots,sizeof(slots));
    int before=executed_graphs;
    assert(cuda_graph_replay_row(row)==0 && executed_graphs==before);
    assert(!memcmp(&saved,&plan,sizeof(plan)) && !memcmp(arguments,argument_host,sizeof(arguments)));
    assert(!memcmp(uploads,upload_host,sizeof(uploads)) && !memcmp(metadata,metadata_host,sizeof(metadata)));
    assert(!memcmp(slots,graph_slots,sizeof(slots)));
}
int main(void) {
    int routes=0,replay_misses=0,seal_misses=0;
    for(int bits=4;bits<=8;bits+=4)for(int activation=8;activation<=16;activation+=8)
    for(int prepared=0;prepared<=(activation==8);prepared++) {
        packed_setup(bits,activation,prepared);
        VxCudaReplayRow first=recipe(0);cuda_graph_replay_row_seal(&first);assert(plan.row_ready);
        unsigned char sealed[sizeof(argument_host)];memcpy(sealed,argument_host,sizeof(sealed));
        packed_check_records(sealed,0);packed_execute();
        plan.pass=CUDA_REPLAY_PASS_VALIDATE;
        for(int row=1;row<CAP;row++) {
            row_position=row;row_length=engine.decode_rows.key_capacity=row+1;plan.launch_index=0;
            graph_slots[6].host_dirty=1;graph_slots[6].device_dirty=0;
            assert(vx_runtime_cuda_prepared_row_replay_locked(row,0)==1);
            assert(plan.launch_index==plan.launch_count && graph_slots[6].device_dirty && !graph_slots[6].host_dirty);
            packed_check_records(sealed,row);packed_execute();
            for(int earlier=0;earlier<=row;earlier++)for(int j=0;j<VOCAB;j++)
                assert(device_rows[5][earlier*VOCAB+j]==(float)(earlier*1000+j%WIDTH+bits+activation+7));
            for(int future=row+1;future<CAP;future++)for(int j=0;j<VOCAB;j++)
                assert(device_rows[5][future*VOCAB+j]==-31.25f);
        }
        assert(executed_graphs==CAP && computed_rows==CAP && prepared_rows==(prepared?CAP:0));routes++;
    }
    for(int change=0;change<12;change++) {
        packed_setup(4,8,1);VxCudaReplayRow first=recipe(0),next=recipe(1);
        cuda_graph_replay_row_seal(&first);assert(plan.row_ready);plan.pass=CUDA_REPLAY_PASS_VALIDATE;
        switch(change) {
        case 0:plan.failed=1;break;
        case 1:graph_slot_epoch++;break;
        case 2:closure[5]=0;break;
        case 3:plan.launch_index=1;break;
        case 4:plan.upload_inflight=1;break;
        case 5:plan.argument_device++;break;
        case 6:plan.launches[1].storage_device++;break;
        case 7:plan.launches[1].argument_offset=UINT32_MAX;break;
        case 8:plan.launches[1].row_pointers[0].stride=SIZE_MAX;break;
        case 9:engine.decode_metadata_stage_bytes=4;break;
        case 10:plan.argument_pinned=0;break;
        case 11:cuda_forward_context_failed=1;break;
        }
        packed_miss(&next);replay_misses++;
    }
    for(int change=0;change<6;change++) {
        packed_setup(4,8,1);VxCudaReplayRow first=recipe(0);VxCudaKernelArgumentABI changed=packed_abi;
        switch(change) {
        case 0:changed.count=12;break;
        case 1:changed.bytes=96;break;
        case 2:changed.storage=0;break;
        case 3:changed.symbol="vx_cuda_llm_f32";break;
        case 4:ptr(argument_host,0,(CUdeviceptr)packed_workspace);break;
        case 5:ptr(argument_host,4,(CUdeviceptr)value_weight);break;
        }
        plan.launches[0].argument_abi=&changed;cuda_graph_replay_row_seal(&first);assert(!plan.row_ready);seal_misses++;
    }
    printf("PASS Packed decode row records: %d routes, 8 rows, 13 slots, fixed scratch/phases, full output guards, %d atomic replay refusals and %d seal refusals\n",routes,replay_misses,seal_misses);
}
'''
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-packed-row-records-") as temp:
            unit, binary = Path(temp) / "test.c", Path(temp) / "test"
            unit.write_text(PREAMBLE + records + state + functions +
                            FIXTURE[:FIXTURE.index("int main(void) {")] + checks)
            command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                       "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                       "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
            command += [f"-I{ROOT / path}" for path in includes]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), str(ROOT / "native/src/runtime/attention_mask.c"), "-lm", "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS Packed decode row records", result.stdout)
            print(result.stdout.strip())

    def test_vector_attention_record_domain_and_alignment(self):
        """Control-plane oracle; CUDA arithmetic is covered by the GPU suite."""
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        engine_source = (ROOT / "native/src/backends/cuda_engine.c").read_text()
        backend = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        row_source = (ROOT / "native/src/runtime/engine_runtime_row_cuda.inc").read_text()
        projection = (ROOT / "native/src/runtime/decode_projection.h").read_text()
        records = "".join(record_source(engine_source, name) for name in
                          ("CudaTensorSlot", "CudaReplayRowPointer", "CudaLaunchSignature", "CudaReplayState"))
        records += record_source(backend, "CudaReplayRowProof")
        records += "".join(record_source(projection, name) for name in
                           ("VxDecodeRowOperand", "VxDecodeRowProjection"))
        functions = "".join(function_source(backend, name) for name in
                            ("cuda_argument_abi_valid", "cuda_replay_row_layout", "cuda_replay_row_pointer",
                             "cuda_replay_row_scope", "cuda_replay_row_proof_find", "cuda_replay_row_proofs_build",
                             "cuda_graph_replay_row_seal", "cuda_graph_replay_row"))
        functions += "".join(function_source(row_source, name) for name in
                             ("vx_decode_cuda_command_operand_bind", "vx_decode_cuda_row_geometry_current",
                              "vx_decode_cuda_command_bind", "vx_runtime_cuda_prepared_row_replay_locked"))
        state = STATE.replace("CAP=8, WIDTH=4, VOCAB=11", "CAP=2052, WIDTH=64, VOCAB=64")
        state = state.replace("metadata_host[128],metadata_device[128]", "metadata_host[4097],metadata_device[4097]")
        checks = r'''
static const VxCudaKernelArgumentABI vector_abi={"vx_cuda_cross_sdpa_gqa_decode_vector_f32",attention_sizes,17,136,1};
static VxCudaReplayRow vector_setup(int keys,int unaligned_argument) {
    initialize();memset(fusion,0,sizeof(fusion));graph_slot_epoch=19;
    cuda_forward_context_failed=cuda_domain_enforced=0;
    memset(commands,0,sizeof(commands));memset(closure,0,sizeof(closure));
    memset(plan.pass_writes,0,sizeof(plan.pass_writes));memset(plan.launches,0,sizeof(plan.launches));
    memset(argument_host,0,sizeof(argument_host));memset(metadata_host,0x5a,sizeof(metadata_host));
    plan.launch_count=plan.argument_bytes=plan.upload_bytes=0;
    command(4,VX_OP_CROSS_ATTENTION,5,WIDTH,2,WIDTH);commands[4].attention=commands[4].causal=1;
    commands[4].input_count=3;commands[4].inputs[1]=operand(3,WIDTH,1);commands[4].inputs[2]=operand(4,WIDTH,1);
    /* Offset a real selected owner's device identity, not an unowned pointer,
     * so the alignment rejection is independent of ordinary ownership proof. */
    if(unaligned_argument>=0) {
        int slot=unaligned_argument==0?2:unaligned_argument==1?3:unaligned_argument==2?4:5;
        graph_slots[slot].device+=4;
    }
    unsigned char* a=kernel(&vector_abi);size_t offset=(size_t)(keys-1)*WIDTH*sizeof(float);
    ptr(a,0,graph_slots[2].device+offset);ptr(a,1,graph_slots[3].device);
    ptr(a,2,graph_slots[4].device);ptr(a,3,0);ptr(a,4,graph_slots[5].device+offset);
    u32(a,5,1);u32(a,6,(uint32_t)keys);u32(a,7,WIDTH);u32(a,8,WIDTH);
    u32(a,9,1);u32(a,10,1);u32(a,11,WIDTH);u32(a,12,1);u32(a,13,0);u32(a,14,0);u32(a,15,0);
    row_position=keys-1;row_length=engine.decode_rows.key_capacity=keys;
    unsigned rows=0,prefixes=0;
    assert(cuda_replay_row_layout(&plan.launches[0],&rows,&prefixes));
    assert(rows==((1u<<0)|(1u<<4)) && prefixes==((1u<<1)|(1u<<2)));
    assert(plan.launches[0].row_keys_argument==6 && plan.launches[0].row_position_argument==UINT16_MAX);
    return recipe(keys-1);
}
static void vector_seal(const VxCudaReplayRow* row,int expected) {
    unsigned char arguments[sizeof(argument_host)];int32_t metadata[4097];CudaTensorSlot slots[CUDA_GRAPH_MAX_TENSORS];
    memcpy(arguments,argument_host,sizeof(arguments));memcpy(metadata,metadata_host,sizeof(metadata));
    memcpy(slots,graph_slots,sizeof(slots));
    cuda_graph_replay_row_seal(row);assert(plan.row_ready==expected);
    assert(!memcmp(arguments,argument_host,sizeof(arguments)) && !memcmp(metadata,metadata_host,sizeof(metadata)));
    assert(!memcmp(slots,graph_slots,sizeof(slots)));
}
static void vector_miss(const VxCudaReplayRow* row) {
    CudaReplayState saved=plan;CudaTensorSlot slots[CUDA_GRAPH_MAX_TENSORS];
    unsigned char arguments[sizeof(argument_host)],uploads[sizeof(upload_host)];int32_t metadata[4097];
    memcpy(arguments,argument_host,sizeof(arguments));memcpy(uploads,upload_host,sizeof(uploads));
    memcpy(metadata,metadata_host,sizeof(metadata));memcpy(slots,graph_slots,sizeof(slots));
    assert(cuda_graph_replay_row(row)==0);
    assert(!memcmp(&saved,&plan,sizeof(plan)) && !memcmp(arguments,argument_host,sizeof(arguments)));
    assert(!memcmp(uploads,upload_host,sizeof(uploads)) && !memcmp(metadata,metadata_host,sizeof(metadata)));
    assert(!memcmp(slots,graph_slots,sizeof(slots)));
}
int main(void) {
    int keys[]={511,512,2048,2049};
    for(unsigned i=0;i<sizeof(keys)/sizeof(keys[0]);i++) {
        VxCudaReplayRow row=vector_setup(keys[i],-1);vector_seal(&row,keys[i]>=512 && keys[i]<=2048);
    }
    for(int alignment=-1;alignment<=4;alignment++) {
        if(alignment==3)continue; /* mask is required null by the row recipe. */
        VxCudaReplayRow row=vector_setup(512,alignment);
        vector_seal(&row,alignment!=0 && alignment!=1);
    }
    for(unsigned i=0;i<sizeof(keys)/sizeof(keys[0]);i++) {
        VxCudaReplayRow first=vector_setup(512,-1);vector_seal(&first,1);plan.pass=CUDA_REPLAY_PASS_VALIDATE;
        VxCudaReplayRow next=recipe(keys[i]-1);unsigned char before[136];memcpy(before,argument_host,sizeof(before));
        if(keys[i]<512 || keys[i]>2048)vector_miss(&next);
        else {
            assert(cuda_graph_replay_row(&next)==1 && plan.launch_index==1);
            assert(getptr(argument_host,0)==graph_slots[2].device+(size_t)next.position*WIDTH*4);
            assert(getptr(argument_host,4)==graph_slots[5].device+(size_t)next.position*WIDTH*4);
            assert(get32(argument_host,6)==(uint32_t)next.keys && metadata_host[next.keys]==next.rope_position);
            assert(graph_slots[5].device_dirty && plan.pass_writes[5]);
            for(unsigned slot=0;slot<17;slot++)if(slot!=0 && slot!=4 && slot!=6)
                assert(!memcmp(before+slot*8,argument_host+slot*8,8));
        }
    }
    /* A capture-aligned Q may acquire an unaligned next row. Check the actual
     * candidate address before any argument/metadata/coherence mutation. */
    VxCudaReplayRow first=vector_setup(512,-1);vector_seal(&first,1);plan.pass=CUDA_REPLAY_PASS_VALIDATE;
    CudaReplayRowPointer* q=&plan.launches[0].row_pointers[0];assert(q->argument==0);
    q->stride+=4;q->base-=(size_t)first.position*4;
    assert(!((q->base+(size_t)first.position*q->stride)&15u));
    VxCudaReplayRow next=recipe(512);assert((q->base+(size_t)next.position*q->stride)&15u);vector_miss(&next);
    for(int change=0;change<3;change++) {
        first=vector_setup(512,-1);VxCudaKernelArgumentABI bad=vector_abi;
        if(change==0)bad.count=19;
        if(change==1)bad.bytes=128;
        if(change==2)bad.storage=0;
        plan.launches[0].argument_abi=&bad;vector_seal(&first,0);
    }
    puts("PASS vector attention row records: 17 slots, keys511/512/2048/2049, owned Q/K alignment, fixed K/V/scalars, atomic future-Q alignment refusal");
}
'''
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-vector-row-records-") as temp:
            unit, binary = Path(temp) / "test.c", Path(temp) / "test"
            unit.write_text(PREAMBLE + records + state + functions +
                            FIXTURE[:FIXTURE.index("int main(void) {")] + checks)
            command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                       "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                       "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
            command += [f"-I{ROOT / path}" for path in includes]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), str(ROOT / "native/src/runtime/attention_mask.c"), "-lm", "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS vector attention row records", result.stdout)
            print(result.stdout.strip())

    def test_scheduler_reuses_only_an_admitted_complete_pass(self):
        from tools.tests.test_incremental_prepared_bindings import FIXTURE as scheduler
        from tools.tests.test_decode_shape_plan_transfer import CHECKS
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        runtime = ROOT / "native/src/runtime"
        source = (runtime / "decode_session.c").read_text()
        unit = scheduler[:scheduler.index("int main(void) {")]
        unit = unit.replace("void vx_decode_session_invalidate_cache_locked(void) {}", "")
        session = source[source.index("#define VOLVOXAI_DECODE_SESSION_MAGIC"):
                         source.index("int vx_decode_session_active_locked")]
        run = source[source.index("typedef struct {\n    VolvoxAIDecodeSession* session;"):
                     source.index("static void decode_run_finish_locked")]
        production = '#include "engine_core.h"\n' + session
        for name in ("vx_decode_session_active_locked", "vx_decode_session_invalidate_cache_locked",
                     "decode_session_valid", "decode_session_attached"):
            production += function_source(source, name)
        production += run
        for name in ("decode_run_finish_locked", "decode_run_begin_locked", "decode_run_end_locked"):
            production += function_source(source, name)
        unit = unit.replace('#include "../src/runtime/incremental_runtime.c"',
                            f'#include "{runtime / "incremental_runtime.c"}"\n' + production)
        unit = unit.replace("static int lookups, classifications, calls, graph_mode, proof_calls, refuse;",
                            "static int lookups, classifications, calls, graph_mode, proof_calls, refuse;\nstatic int device_rows,finish_fails;")
        unit = unit.replace("int vx_runtime_backend_has_device_rows(void) {return graph_mode;}",
                            "int vx_runtime_backend_has_device_rows(void) {return graph_mode && device_rows;}")
        unit = unit.replace("assert(!replay);", "(void)replay;")
        unit = unit.replace("int vx_runtime_backend_end_forward(int ok) {return ok?0:-1;}",
                            "int vx_runtime_backend_end_forward(int ok) {return ok && !finish_fails?0:-1;}")
        unit = unit.replace(function_source(unit, "vx_runtime_node_incremental_row_compatible"),
                            "static int live_compatible(Node* node,int index,int row);\n"
                            "int vx_runtime_node_incremental_row_compatible(Node* node,int index,int row) {return live_compatible(node,index,row);}\n")
        unit = unit.replace("for(int i=0;i<12;i++) {\n        if(node->operator_kind",
                            "for(int i=g_active_row<0?0:g_active_row*3;i<(g_active_row<0?12:(g_active_row+1)*3);i++) {\n        if(node->operator_kind")
        unit = unit.replace(function_source(unit, "vx_runtime_cuda_prepared_row_replay_locked"), r'''
static int reuse,reuse_calls,seal_calls;
int vx_runtime_cuda_prepared_row_replay_locked(int row,int seal) {
    if(seal){seal_calls++;return 0;}
    reuse_calls++;
    if(!reuse)return 0;
    assert(g_decode_slots==1 && g_decode_rows.positions[0]==row);
    /* This mock backend computes the same admitted row from its closure,
     * without calling a node lowerer. The other fixture exercises the actual
     * typed CUDA recipe and immutable input snapshots separately. */
    for(int k=row*3;k<(row+1)*3;k++) {
        if(g_incremental_scratch.hybrid_nodes[0])g_t[2].data[k]=g_t[0].data[k]+g_t[5].data[k%3];
        if(g_incremental_scratch.hybrid_nodes[1])g_t[3].data[k]=g_t[1].data[k]*2;
        if(g_incremental_scratch.hybrid_nodes[2])g_t[4].data[k]=g_t[2].data[k]+g_t[3].data[k];
    }
    return 1;
}
''')
        checks = CHECKS[:CHECKS.index("static void session_cases(void)")]
        checks = checks.replace("if(!failure && row<4)assert(!status && !memcmp(numerical,values[0],sizeof(numerical)));",
                                "if(finish_fails)assert(status==-1 && !session.prefilled && !g_cache_valid);\n"
                                "else if(!failure && row<4)assert(!status && !memcmp(numerical,values[0],sizeof(numerical)));")
        unit += (ROOT / "tools/tests/fixtures/decode_session_admission_reference.inc").read_text() + checks + r'''
int main(void) {
    setup(0);
#if VOLVOXAI_ENABLE_CUDA
    int comparisons=0;
    for(int device=0;device<2;device++)for(int automatic=0;automatic<2;automatic++)
    for(int row=0;row<4;row++)for(int roots=1;roots<4;roots++)for(int stepwise=0;stepwise<2;stepwise++) {
        Outcome ordinary={0},prepared={0};
        reuse=0;run_case(0,device,automatic,row,roots,0,stepwise,&ordinary);
        reuse=1;reuse_calls=seal_calls=0;run_case(0,device,automatic,row,roots,0,stepwise,&prepared);
        int eligible=device && !stepwise;
        assert(ordinary.status==prepared.status && ordinary.mode==prepared.mode && ordinary.last_mode==prepared.last_mode && ordinary.cache==prepared.cache);
        assert(!memcmp(ordinary.values,prepared.values,sizeof(prepared.values)) && prepared.proof_count==ordinary.proof_count);
        assert(reuse_calls==eligible && !seal_calls && (eligible?prepared.dispatch_count==0:prepared.dispatch_count==ordinary.dispatch_count));
        comparisons++;
    }
    /* Admission refusal reaches neither the prepared backend nor a lowerer. */
    for(int failure=1;failure<=6;failure++) {
        Outcome denied={0};reuse=1;reuse_calls=seal_calls=0;run_case(0,1,0,1,3,failure,0,&denied);
        assert(denied.status==-1 && !reuse_calls && !seal_calls && !denied.dispatch_count && !denied.cache);
    }
    Outcome failed={0};reuse=1;finish_fails=1;reuse_calls=seal_calls=0;
    run_case(0,1,0,1,3,0,0,&failed);
    assert(failed.status==-1 && reuse_calls==1 && !seal_calls && !failed.dispatch_count && !failed.cache && !failed.prefilled);
    finish_fails=0;
    printf("PASS prepared scheduler: %d row/closure/CPU/AUTO/stepwise numerical comparisons, unchanged admission proofs, zero lowerer calls for reuse, six terminal pre-dispatch refusals, failed graph completion invalidates retained state\n",comparisons);
#endif
    vx_engine_state_release_graph_metadata(&contexts[0]);
}
'''
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-row-scheduler-") as temp:
            path = Path(temp) / "test.c"
            path.write_text(unit)
            for training in (0, 1):
                binary = Path(temp) / f"test-{training}"
                command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                           "-Wno-unused-function", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                           f"-DVOLVOXAI_ENABLE_TRAINING={training}", "-DVOLVOXAI_ENABLE_CUDA=1",
                           "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0",
                           "-DVOLVOXAI_ENABLE_METAL=0"]
                command += [f"-I{ROOT / include}" for include in includes]
                command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
                command += [str(path), str(runtime / "decode_row_set.c"), "-lm", "-o", str(binary)]
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([str(binary)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS prepared scheduler", result.stdout)
                print(result.stdout.strip())

    def test_bounded_operand_index_and_ordinary_fallback(self):
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")
        engine = (ROOT / "native/src/backends/cuda_engine.c").read_text()
        backend = (ROOT / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        records = "".join(record_source(engine, name) for name in
                          ("CudaTensorSlot", "CudaReplayRowPointer", "CudaLaunchSignature", "CudaReplayState"))
        records += record_source(backend, "CudaReplayRowProof")
        names = ("graph_slot_hash_bucket", "graph_find_slot_linear", "graph_find_slot", "graph_insert_slot_hash",
                 "graph_slot_is_resident_weight", "graph_rebuild_slot_hash", "graph_owner_index_less",
                 "graph_owner_index_sift", "graph_owner_index_build", "graph_owner_index_find",
                 "graph_find_containing_slot", "cuda_argument_abi_valid", "cuda_replay_row_layout",
                 "cuda_replay_row_pointer", "cuda_replay_row_scope", "cuda_replay_row_proof_find",
                 "cuda_replay_row_proofs_build", "cuda_graph_replay_row_seal")
        functions = "".join(function_source(backend, name) for name in names)
        preamble = PREAMBLE.replace("CUDA_GRAPH_MAX_TENSORS 32", "CUDA_GRAPH_MAX_TENSORS 2048")
        preamble = preamble.replace("CUDA_GRAPH_MAX_LAUNCHES 32", "CUDA_GRAPH_MAX_LAUNCHES 2048")
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-cuda-row-proof-index-") as temp:
            unit, binary = Path(temp) / "test.c", Path(temp) / "test"
            unit.write_text(preamble + records + PROOF_INDEX_STATE + functions + SOURCE004_ROW_SEAL + PROOF_INDEX_FIXTURE)
            command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                       "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                       "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
            command += [f"-I{ROOT / path}" for path in includes]
            command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), "-lm", "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS bounded operand proof index", result.stdout)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
