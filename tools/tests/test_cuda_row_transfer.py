"""Execute the production CUDA row transfer with a deferred driver recorder."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.tests import function_source




PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
typedef uintptr_t CUdeviceptr;
typedef struct {const void* host; unsigned char* device;size_t bytes;int host_dirty,device_dirty;} CudaTensorSlot;
typedef struct {CudaTensorSlot* slot;CUdeviceptr device;} CudaTensorView;
typedef struct {int owned;} CudaContextGuard;
enum {CUDA_REPLAY_PASS_INELIGIBLE, CUDA_REPLAY_PASS_CAPTURE};
static struct {int pass,dynamic_parameters;} cuda_replay;
enum {CUDA_REPLAY_COMMAND_UPLOAD=1,CUDA_REPLAY_COMMAND_COPY=2};
static int cuda_stream;
static CudaTensorSlot slots[8];static int nslots;
static int uploads,waits,copies,kernels,marks,fail_copy,fail_enter,fail_leave,fail_upload;
typedef struct {CUdeviceptr to,from;size_t bytes;} Pending;
static Pending pending[64];static unsigned npending;
static int record_mode,recorded_copies;
static int cuda_replay_copy_command(int kind,CUdeviceptr to,CUdeviceptr from,const void* host,size_t bytes) {
    (void)kind;(void)host;
    if(record_mode<0)return -1;
    if(!record_mode)return 0;
    assert(npending<64);pending[npending++]=(Pending){to,from,bytes};recorded_copies++;return 1;
}
static void drain(void) {
    for(unsigned i=0;i<npending;i++)memcpy((void*)pending[i].to,(void*)pending[i].from,pending[i].bytes);
    npending=0;
}
static int cuda_context_enter(CudaContextGuard* guard) {guard->owned=!fail_enter;return guard->owned;}
static int cuda_context_leave(CudaContextGuard* guard) {assert(guard->owned);return !fail_leave;}
static int cuda_ok(int status,const char* name) {(void)name;return status==0;}
static int cuda_trace_copy_device(CUdeviceptr to,CUdeviceptr from,size_t bytes,int stream) {
    (void)stream;copies++;
    if(fail_copy)return 1;
    assert(npending<64);pending[npending++]=(Pending){to,from,bytes};return 0;
}
static void graph_mark_device(CudaTensorSlot* slot) {slot->device_dirty=1;slot->host_dirty=0;marks++;}
static int graph_get_view(const void* host,size_t bytes,int weight,CudaTensorView* view) {
    (void)weight;
    for(int i=0;i<nslots;i++) {
        uintptr_t offset=(uintptr_t)host-(uintptr_t)slots[i].host;
        if((uintptr_t)host>=(uintptr_t)slots[i].host && offset<=slots[i].bytes && bytes<=slots[i].bytes-offset) {
            view->slot=&slots[i];view->device=(uintptr_t)slots[i].device+offset;return 1;
        }
    }
    return 0;
}
static int graph_ensure_input_view(const void* host,size_t bytes,int weight,CudaTensorView* view) {
    if(!graph_get_view(host,bytes,weight,view))return 0;
    if(view->slot->host_dirty) {
        /* The existing pinned H2D slab waits before caller storage can change. */
        drain();waits++;uploads++;memcpy(view->slot->device,view->slot->host,view->slot->bytes);
        if(fail_upload)return 0;
        view->slot->host_dirty=view->slot->device_dirty=0;
    }
    return 1;
}
static int graph_output_view(const void* host,size_t bytes,CudaTensorView* view) {return graph_get_view(host,bytes,0,view);}
#define CUDA_FORWARD_FUNCTION(name) 1
static int cuda_launch_1d(int function,uint32_t count,void** args) {
    (void)function;drain();kernels++;
    const uint32_t* source=(const uint32_t*)*(CUdeviceptr*)args[0];
    const int32_t* indices=(const int32_t*)*(CUdeviceptr*)args[1];
    uint32_t* destination=(uint32_t*)*(CUdeviceptr*)args[2];
    uint32_t rows=*(uint32_t*)args[3],width=*(uint32_t*)args[4],full=*(uint32_t*)args[5],scatter=*(uint32_t*)args[6];
    assert(count==rows*width);
    for(uint32_t row=0;row<rows;row++)for(uint32_t word=0;word<width;word++) {
        int32_t index=indices[row];
        if(scatter) {if(index>=0 && (uint32_t)index<full)destination[(uint32_t)index*width+word]=source[row*width+word];}
        else destination[row*width+word]=(index>=0 && (uint32_t)index<full)?source[(uint32_t)index*width+word]:0;
    }
    return 1;
}
'''

FIXTURE = r'''
int main(void) {
    enum {FULL=9,WIDTH=13};
    uint32_t full[FULL*WIDTH],packed[4*WIDTH],retained[FULL*WIDTH];
    uint32_t full_device[FULL*WIDTH],packed_device[4*WIDTH],retained_device[FULL*WIDTH];
    int32_t indices[4],indices_device[4];
    for(unsigned i=0;i<FULL*WIDTH;i++)full[i]=0x7fa00000u+i;
    full[3*WIDTH]=0x80000000u;full[3*WIDTH+1]=0x7fcabcdeu;
    for(unsigned i=0;i<FULL*WIDTH;i++)retained[i]=0xdeadbeefu;
    slots[0]=(CudaTensorSlot){full,(unsigned char*)full_device,sizeof(full),1,0};
    slots[1]=(CudaTensorSlot){packed,(unsigned char*)packed_device,sizeof(packed),0,0};
    slots[2]=(CudaTensorSlot){retained,(unsigned char*)retained_device,sizeof(retained),1,0};
    slots[3]=(CudaTensorSlot){indices,(unsigned char*)indices_device,sizeof(indices),1,0};nslots=4;
    indices[0]=3;indices[1]=4;indices[2]=5;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,3,packed,sizeof(packed),WIDTH,FULL,0));
    assert(copies==1 && kernels==0 && uploads==1 && waits==1 && npending==1);
    /* The driver borrowed only device spans and copied the row offset. The
     * caller can change/reuse indices before stream completion. */
    indices[0]=6;indices[1]=7;indices[2]=8;drain();
    assert(!memcmp(packed_device,full+3*WIDTH,3*WIDTH*4));
    /* Successive queued scalar gathers use their own captured offsets. */
    indices[0]=1;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,1,packed,sizeof(packed),WIDTH,FULL,0));
    indices[0]=8;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,1,packed+WIDTH,sizeof(packed)-WIDTH*4,WIDTH,FULL,0));
    indices[0]=0;drain();
    assert(!memcmp(packed_device,full+WIDTH,WIDTH*4));
    assert(!memcmp(packed_device+WIDTH,full+8*WIDTH,WIDTH*4));
    assert(uploads==1 && waits==1 && kernels==0);
    /* Scatter preserves stale-host sentinel rows outside the selection. */
    indices[0]=4;indices[1]=5;
    assert(cuda_graph_row_index_transfer(packed,sizeof(packed),indices,2,retained,sizeof(retained),WIDTH,FULL,1));
    assert(npending==1 && uploads==2 && waits==2);drain();
    for(unsigned i=0;i<FULL*WIDTH;i++)assert(retained_device[i]==(i>=4*WIDTH && i<6*WIDTH?packed_device[i-4*WIDTH]:0xdeadbeefu));
    /* Permutation, page discontinuity and empty rows retain indexed semantics. */
    indices[0]=8;indices[1]=2;indices[2]=5;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,3,packed,sizeof(packed),WIDTH,FULL,0));
    assert(kernels==1 && !memcmp(packed_device,full+8*WIDTH,WIDTH*4));
    assert(!memcmp(packed_device+WIDTH,full+2*WIDTH,WIDTH*4));
    indices[0]=-1;indices[1]=3;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,2,packed,sizeof(packed),WIDTH,FULL,0));
    for(unsigned i=0;i<WIDTH;i++)assert(packed_device[i]==0);
    assert(!memcmp(packed_device+WIDTH,full+3*WIDTH,WIDTH*4));
    unsigned before=uploads;indices[0]=0;indices[1]=7;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,2,packed,sizeof(packed),WIDTH,FULL,0));
    assert(uploads==(int)before+1 && !memcmp(packed_device+WIDTH,full+7*WIDTH,WIDTH*4));
    /* A replay plan still sees the existing launch signature path. */
    cuda_replay.pass=CUDA_REPLAY_PASS_CAPTURE;indices[0]=4;before=copies;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,1,packed,sizeof(packed),WIDTH,FULL,0));
    assert(copies==(int)before && !memcmp(packed_device,full+4*WIDTH,WIDTH*4));cuda_replay.pass=0;
    /* Interior same-address views perform coherence without a copy; a
     * genuinely overlapping range must retain the indexed kernel. */
    before=copies;indices[0]=3;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,1,full+3*WIDTH,WIDTH*4,WIDTH,FULL,0));
    assert(copies==(int)before);
    unsigned old_kernels=kernels;
    assert(cuda_graph_row_index_transfer(full,sizeof(full),indices,1,full+3*WIDTH+1,WIDTH*4,WIDTH,FULL,0));
    assert(copies==(int)before && kernels==(int)old_kernels+1);
    /* Failed submission must not publish destination coherence or pending work. */
    fail_copy=1;before=marks;
    assert(!cuda_graph_row_index_transfer(full,sizeof(full),indices,1,packed,sizeof(packed),WIDTH,FULL,0));
    assert(marks==(int)before && npending==0);fail_copy=0;
    fail_enter=1;before=copies;
    assert(!cuda_graph_row_index_transfer(full,sizeof(full),indices,1,packed,sizeof(packed),WIDTH,FULL,0));
    assert(copies==(int)before && npending==0);fail_enter=0;
    fail_leave=1;
    assert(!cuda_graph_row_index_transfer(full,sizeof(full),indices,1,packed,sizeof(packed),WIDTH,FULL,0));
    drain();fail_leave=0;
    /* The private span route never borrows indices. Dynamic replay records
     * the exact D2D addresses in order; a failed recorder is terminal. */
    cuda_replay.pass=CUDA_REPLAY_PASS_CAPTURE;cuda_replay.dynamic_parameters=1;record_mode=1;
    before=copies;
    assert(cuda_graph_row_span_transfer(full,sizeof(full),3,2,packed,sizeof(packed),WIDTH,FULL,0)==1);
    assert(recorded_copies==1 && copies==(int)before && npending==1);drain();
    assert(!memcmp(packed_device,full_device+3*WIDTH,2*WIDTH*4));
    record_mode=-1;
    assert(!cuda_graph_row_span_transfer(full,sizeof(full),1,1,packed,sizeof(packed),WIDTH,FULL,0));
    assert(npending==0 && copies==(int)before);record_mode=0;
    cuda_replay.dynamic_parameters=0;before=uploads;
    assert(cuda_graph_row_span_transfer(full,sizeof(full),1,1,packed,sizeof(packed),WIDTH,FULL,0)==-1);
    assert(uploads==(int)before && !npending);
    cuda_replay.pass=CUDA_REPLAY_PASS_INELIGIBLE;
    assert(cuda_graph_row_span_transfer(full,sizeof(full),3,1,full+3*WIDTH+1,WIDTH*4,WIDTH,FULL,0)==-1);
    assert(!cuda_graph_row_span_transfer(full,sizeof(full),FULL,1,packed,sizeof(packed),WIDTH,FULL,0));
    assert(!cuda_graph_row_span_transfer(full,sizeof(full),FULL-1,2,packed,sizeof(packed),WIDTH,FULL,0));
    assert(uploads==(int)before && !npending);
    puts("PASS CUDA resident row copies, exact bits, changing indices, staging fallback and failure cleanup");
}
'''

ROW_PREAMBLE = r'''
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "sequence_runtime.h"
#include "decode_projection.h"
static VxEngineState owner;
static T tensors[4];static Node nodes[1];static const char* routes[1];
static int fail_numerical,expected_direct_input,expected_direct_output;
static int rope_enabled,rope_offset,last_position,last_mask_sum;
static int mask_reads;
static int counted_attention_mask_allowed(const T* mask,int mode,int batch,int query,int key,int queries,int keys,int causal) {
    mask_reads++;return attention_mask_allowed(mask,mode,batch,query,key,queries,keys,causal);
}
static T* rope_positions;
static const char* paged_tensor_name="";
VxEngineState* vx_engine_state_current(void) {return &owner;}
T* t_find(const char* name) {
    for(unsigned i=0;i<4;i++)if(!strcmp(name,tensors[i].name))return &tensors[i];
    return NULL;
}
static T* nin(Node* node,VxPortKind port) {
    for(int i=0;i<node->nin;i++)if(node->ins[i].port==port)return t_find(node->ins[i].name);
    return NULL;
}
int volvoxai_engine_tensor_is_model_weight_locked(const char* name) {return !strcmp(name,"weight");}
int vx_incremental_tensor_dirty_before_node_locked(const T* tensor,int index) {(void)index;return strcmp(tensor->name,"weight")!=0;}
int vx_paged_bound_locked(void) {return !g_decode_rows.unpaged;}
int vx_paged_slot_pages_locked(int slot,VxDecodeSlotPages* out) {assert(slot==0);*out=g_decode_rows.pages[0];return 1;}
int vx_paged_tensor_locked(const T* tensor) {return !strcmp(tensor->name,paged_tensor_name);}
int vx_sequence_rope_parameters(Node* node,const T* input,const T* output,VxSequenceRoPE* p) {
    (void)node;(void)output;
    if(!rope_enabled)return 0;
    *p=(VxSequenceRoPE){.batch=1,.sequence=9,.width=input->shape[2],.heads=1,.rotary_width=2,
        .theta=10000.f,.position_offset=rope_offset,.position_mode=1,.positions=rope_positions};return 1;
}
static void cuda_graph_mark_host(const void* host,size_t bytes,int weight) {
    CudaTensorView view;assert(graph_get_view(host,bytes,weight,&view));view.slot->host_dirty=1;view.slot->device_dirty=0;
}
static int cuda_graph_rope_heads_f32(const float* input,const int32_t* positions,long position_count,
    float* output,int batch,int sequence,int width,int heads,int rotary_width,float theta,
    int32_t offset,int interleaved,int mode) {
    (void)heads;(void)rotary_width;(void)theta;(void)offset;(void)interleaved;
    assert(batch==1 && sequence==1 && position_count==1 && mode==2);
    CudaTensorView from,to,control;
    assert(graph_ensure_input_view(input,(size_t)width*4,0,&from));
    assert(graph_ensure_input_view(positions,4,0,&control));
    assert(graph_output_view(output,(size_t)width*4,&to));drain();
    last_position=*(const int32_t*)control.device;
    for(int i=0;i<width;i++)((float*)to.device)[i]=((const float*)from.device)[i]+last_position;
    graph_mark_device(to.slot);return 1;
}
static int vx_decode_cuda_attention(Node* node,const VxDecodeRowProjection* p,T* output,const int32_t* mask) {
    (void)node;CudaTensorView from,to,control;
    assert(graph_ensure_input_view(p->inputs[0].tensor->data,(size_t)output->numel*4,0,&from));
    assert(graph_ensure_input_view(mask,(size_t)p->keys*4,0,&control));
    assert(graph_output_view(output->data,(size_t)output->numel*4,&to));drain();last_mask_sum=0;
    for(int i=0;i<p->keys;i++)last_mask_sum+=((const int32_t*)control.device)[i];
    for(long i=0;i<output->numel;i++)((float*)to.device)[i]=((const float*)from.device)[i]+last_mask_sum;
    graph_mark_device(to.slot);return 0;
}
static int vx_decode_cuda_project_compatible(Node* node,int idx,VxDecodeRowProjection* p) {
    (void)node;(void)idx;return p->input_count==1;
}
static int vx_decode_cuda_command_run(const VxIncrementalCudaCommand* command,
    const T* narrow,T* output) {(void)command;(void)narrow;(void)output;assert(0);return 0;}
static void cuda_graph_note_node_route(int builtin) {(void)builtin;assert(0);}
static int cuda_graph_copy_32(const void* source,void* destination,long elements) {
    CudaTensorView from,to;
    assert(graph_ensure_input_view(source,(size_t)elements*4,0,&from));
    assert(graph_output_view(destination,(size_t)elements*4,&to));drain();
    memcpy((void*)to.device,(const void*)from.device,(size_t)elements*4);graph_mark_device(to.slot);return 1;
}
int run_node(Node* node,int idx,int last) {
    (void)last;assert(g_active_row==-1 && g_execution_row==-1 && !g_decode_slots && !owner.paged.bound);
    T* input=nin(node,VX_PORT_INPUT);T* output=t_find(node->out);
    assert(input && output && input->shape[1]==1 && output->shape[1]==1);
    assert((input->data!=(float*)owner.decode_numeric_stage[0])==expected_direct_input);
    assert((output->data!=(float*)owner.decode_numeric_stage[3])==expected_direct_output);
    CudaTensorView from,to;
    assert(graph_ensure_input_view(input->data,(size_t)input->numel*4,0,&from));
    assert(graph_output_view(output->data,(size_t)output->numel*4,&to));drain();
    if(fail_numerical)return -1;
    for(long i=0;i<output->numel;i++)((float*)to.device)[i]=((const float*)from.device)[i]*1.5f+2.0f;
    graph_mark_device(to.slot);routes[idx]="cuda-recorder";return 0;
}
'''

ROW_FIXTURE = r'''
static T tensor(const char* name,void* host,int width) {
    T result={.dtype=T_F32,.elem_size=4,.ndim=3,.shape={1,9,width},.numel=9*width,.data=host};
    strcpy(result.name,name);return result;
}
static void register_span(void* host,void* device,size_t bytes,int host_dirty) {
    assert(nslots<8);slots[nslots++]=(CudaTensorSlot){host,device,bytes,host_dirty,!host_dirty};
}
static VxDecodeRowProjection projection(int position,const VxDecodeSlotPages* pages) {
    assert(vx_decode_row_set_init(&owner.decode_rows,1,&position,pages)==VX_DECODE_ROW_SET_OK);
    VxDecodeRowProjection p={.rows=owner.decode_rows,.input_count=1};
    /* vx_decode_row_set_init borrows the position; the owner supplies stable storage. */
    static int stable_position;stable_position=position;p.rows.positions=&stable_position;
    p.inputs[0]=(VxDecodeRowOperand){&tensors[0],1,9,(size_t)tensors[0].shape[2],0};
    p.output=(VxDecodeRowOperand){&tensors[1],1,9,(size_t)tensors[1].shape[2],0};return p;
}
static void restored(const T* input,const T* output) {
    assert(!memcmp(&tensors[0],input,sizeof(T)) && !memcmp(&tensors[1],output,sizeof(T)));
    assert(g_active_row==4 && g_execution_row==5 && g_decode_slots==1 && g_prefix_rows==2 && owner.paged.bound==1);
}
int main(void) {
    enum {WIDTH=13,FULL=9};
    float host[2*FULL*WIDTH],resident[2*FULL*WIDTH],staging[4][FULL*WIDTH],stage_device[4][FULL*WIDTH];
    int32_t metadata[64],metadata_device[64];Ref input_port={.port=VX_PORT_INPUT};strcpy(input_port.name,"input");
    memset(host,0xab,sizeof(host));memset(staging,0,sizeof(staging));memset(metadata,0,sizeof(metadata));
    for(unsigned i=0;i<2*FULL*WIDTH;i++)resident[i]=(float)i*.125f;
    owner.nodes=nodes;owner.node_count=1;owner.tensors=tensors;owner.tensor_count=4;
    owner.runtime_route_backend=routes;owner.node_capacity=1;owner.decode_row_stage_reserved=1;
    owner.decode_metadata_stage=(unsigned char*)metadata;owner.decode_metadata_stage_bytes=sizeof(metadata);
    for(unsigned i=0;i<4;i++) {
        owner.decode_numeric_stage[i]=(unsigned char*)staging[i];owner.decode_numeric_stage_bytes[i]=sizeof(staging[i]);
        register_span(staging[i],stage_device[i],sizeof(staging[i]),0);
    }
    register_span(metadata,metadata_device,sizeof(metadata),0);
    /* Both tensors are offset views of one owning CUDA span. The stale host
     * activation mirror must never overwrite the resident input or old rows. */
    register_span(host,resident,sizeof(host),0);
    tensors[0]=tensor("input",host,WIDTH);tensors[1]=tensor("output",host+FULL*WIDTH,WIDTH);
    nodes[0]=(Node){.operator_kind=VX_OP_LINEAR,.ins=&input_port,.nin=1};strcpy(nodes[0].out,"output");
    g_active_row=4;g_execution_row=5;g_decode_slots=1;g_prefix_rows=2;owner.paged.bound=1;
    T input=tensors[0],output=tensors[1];float before[2*FULL*WIDTH];memcpy(before,resident,sizeof(before));
    VxDecodeRowProjection p=projection(6,NULL);expected_direct_input=expected_direct_output=1;
    assert(!vx_decode_cuda_project(&nodes[0],0,1,&p));drain();restored(&input,&output);
    for(unsigned i=0;i<2*FULL*WIDTH;i++)assert(resident[i]==(i>=15*WIDTH && i<16*WIDTH?before[i-9*WIDTH]*1.5f+2.0f:before[i]));
    assert(!copies && !uploads && !waits && !kernels);
    /* A later row uses new offsets even though every owning address is reused. */
    p=projection(2,NULL);assert(!vx_decode_cuda_project(&nodes[0],0,1,&p));restored(&input,&output);
    for(unsigned i=0;i<WIDTH;i++)assert(resident[11*WIDTH+i]==before[2*WIDTH+i]*1.5f+2.0f);
    /* Original output coherence is prepared before a partial numerical write. */
    slots[5].host_dirty=1;slots[5].device_dirty=0;
    for(unsigned i=0;i<2*FULL*WIDTH;i++)host[i]=(float)i*.25f;
    p=projection(4,NULL);assert(!vx_decode_cuda_project(&nodes[0],0,1,&p));restored(&input,&output);
    assert(uploads==1 && waits==1);
    for(unsigned i=0;i<2*FULL*WIDTH;i++)assert(resident[i]==(i>=13*WIDTH && i<14*WIDTH?host[i-9*WIDTH]*1.5f+2.0f:host[i]));
    /* Distinct tensors may alias one selected range: retain the separate
     * output stage so the numerical call never becomes accidentally in-place. */
    tensors[1]=tensor("output",host,WIDTH);output=tensors[1];p=projection(1,NULL);expected_direct_output=0;
    unsigned copies_before=copies;assert(!vx_decode_cuda_project(&nodes[0],0,1,&p));drain();restored(&input,&output);
    assert(copies==(int)copies_before+1);
    for(unsigned i=0;i<WIDTH;i++)assert(resident[WIDTH+i]==host[WIDTH+i]*1.5f+2.0f);
    /* Failure restores descriptors and execution flags before terminal cleanup. */
    fail_numerical=1;p=projection(3,NULL);copies_before=copies;
    assert(vx_decode_cuda_project(&nodes[0],0,1,&p)==-1);drain();restored(&input,&output);
    assert(copies==(int)copies_before);fail_numerical=0;
    /* Existing synchronous upload failure during full-span preparation is
     * propagated before numerical mutation or tensor-descriptor rebinding. */
    tensors[1]=tensor("output",host+FULL*WIDTH,WIDTH);output=tensors[1];
    expected_direct_output=1;slots[5].host_dirty=1;fail_upload=1;p=projection(0,NULL);copies_before=copies;
    assert(vx_decode_cuda_project(&nodes[0],0,1,&p)==-1);restored(&input,&output);
    assert(copies==(int)copies_before && npending==0);fail_upload=0;slots[5].host_dirty=0;
    /* Slot-only operators retain staged inputs/output, even for scalar rows. */
    nodes[0].operator_kind=VX_OP_MUL;expected_direct_input=expected_direct_output=0;p=projection(7,NULL);
    assert(!vx_decode_cuda_project(&nodes[0],0,1,&p));drain();restored(&input,&output);
    /* Noncontiguous prefix pages cannot become a view; the scalar physical
     * write still can, with page-table row addressing rather than logical rows. */
    nodes[0].operator_kind=VX_OP_LINEAR;int table[]={2,0,3,1,4};
    VxDecodeSlotPages pages={.page_table=table,.page_tokens=2,.pages_per_slot=5};
    p=projection(5,&pages);paged_tensor_name="input";
    T narrow;assert(vx_decode_cuda_storage(&p.inputs[0],&p.rows,0,&narrow));
    int32_t* indices=metadata;
    assert(vx_decode_cuda_contiguous_view(&p.inputs[0],&p.rows,indices,&narrow));
    assert(narrow.data==tensors[0].data+7*WIDTH);
    p.inputs[0].prefix=1;
    assert(vx_decode_cuda_storage(&p.inputs[0],&p.rows,0,&narrow));
    assert(!vx_decode_cuda_contiguous_view(&p.inputs[0],&p.rows,indices,&narrow));
    puts("PASS CUDA production row projection, direct resident views, coherence, aliases, paged fallback and error restoration");
}
'''

METADATA_FIXTURE = r'''
int main(void) {
    enum {WIDTH=3,FULL=9};
    float host[2*FULL*WIDTH],resident[2*FULL*WIDTH],stage[4][FULL*WIDTH],stage_device[4][FULL*WIDTH];
    int32_t metadata[64],metadata_device[64],explicit_positions[9],source_mask[9];
    for(int i=0;i<2*FULL*WIDTH;i++){host[i]=-999.f;resident[i]=(float)i*.125f;}
    for(int i=0;i<64;i++)metadata[i]=metadata_device[i]=0x123456;
    for(int i=0;i<9;i++){explicit_positions[i]=i+10;source_mask[i]=1;}
    owner.nodes=nodes;owner.node_count=owner.node_capacity=1;owner.tensors=tensors;owner.tensor_count=4;
    owner.runtime_route_backend=routes;owner.decode_row_stage_reserved=1;
    owner.decode_metadata_stage=(unsigned char*)metadata;owner.decode_metadata_stage_bytes=sizeof(metadata);
    for(int i=0;i<4;i++) {
        owner.decode_numeric_stage[i]=(unsigned char*)stage[i];owner.decode_numeric_stage_bytes[i]=sizeof(stage[i]);
        register_span(stage[i],stage_device[i],sizeof(stage[i]),0);
    }
    register_span(metadata,metadata_device,sizeof(metadata),0);register_span(host,resident,sizeof(host),0);
    tensors[0]=tensor("input",host,WIDTH);tensors[1]=tensor("output",host+FULL*WIDTH,WIDTH);
    Ref input={.port=VX_PORT_INPUT};strcpy(input.name,"input");
    nodes[0]=(Node){.operator_kind=VX_OP_ROPE,.ins=&input,.nin=1};strcpy(nodes[0].out,"output");
    g_active_row=4;g_execution_row=5;g_decode_slots=1;g_prefix_rows=2;owner.paged.bound=1;
    T saved_input=tensors[0],saved_output=tensors[1];
    rope_enabled=1;VxDecodeRowProjection p=projection(2,NULL);p.rope=1;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));restored(&saved_input,&saved_output);
    assert(uploads==1 && last_position==2 && metadata[63]==0x123456);
    unsigned prior=uploads;
    for(int layer=0;layer<7;layer++)assert(!vx_decode_cuda_project(nodes,0,1,&p));
    assert(uploads==(int)prior && !copies && !kernels);
    rope_offset=5;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(uploads==(int)prior+1 && last_position==7);
    tensors[2]=(T){.dtype=T_I32,.elem_size=4,.ndim=1,.shape={9},.numel=9,.data=(float*)explicit_positions};
    rope_positions=&tensors[2];assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_position==12);
    explicit_positions[2]=23;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_position==23);
    prior=uploads;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(uploads==(int)prior);
    for(int i=0;i<WIDTH;i++)assert(resident[(FULL+2)*WIDTH+i]==resident[2*WIDTH+i]+23);
    /* A smaller node's scratch indices cover the cached position. The host
     * value can already equal the next desired value while GPU storage is
     * stale; the overlap guard must force its upload anyway. */
    VxDecodeRowProjection smaller=projection(1,NULL);int32_t* indices=metadata+3;
    uint32_t count,rows;size_t bytes;
    assert(vx_decode_cuda_indices(&smaller.output,&smaller.rows,indices,&count,&rows,&bytes));
    assert(metadata[3]==1 && !owner.decode_cuda_positions_valid);
    explicit_positions[2]=1;prior=uploads;
    p=projection(2,NULL);p.rope=1;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_position==1 && uploads==(int)prior+1);
    /* Attention changes only its mask section. Exact reuse and a mutable
     * authoritative mask both preserve the value consumed on the device. */
    nodes[0].operator_kind=VX_OP_CROSS_SDPA;p.rope=0;p.attention=p.causal=1;p.keys=3;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_mask_sum==3);
    prior=uploads;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(uploads==(int)prior);
    tensors[3]=(T){.dtype=T_I32,.elem_size=4,.ndim=1,.shape={9},.numel=9,.data=(float*)source_mask};
    p.mask=&tensors[3];p.mask_layout=ATTN_MASK_KEY;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_mask_sum==3);
    prior=uploads;source_mask[1]=0;assert(!vx_decode_cuda_project(nodes,0,1,&p));
    assert(last_mask_sum==2 && uploads==(int)prior+1);
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(uploads==(int)prior+1);
    /* Changing key geometry cannot borrow a prior mask or positions slice. */
    p.keys=2;prior=uploads;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_mask_sum==1 && uploads==(int)prior+1);
    nodes[0].operator_kind=VX_OP_ROPE;p=projection(2,NULL);p.rope=1;
    prior=uploads;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_position==1 && uploads==(int)prior);
    /* A new pass/context must not inherit validity from earlier storage. */
    owner.decode_cuda_mask_valid=owner.decode_cuda_positions_valid=0;
    prior=uploads;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(uploads==(int)prior+1);
    restored(&saved_input,&saved_output);
    puts("PASS CUDA metadata exact reuse, mutable controls, key geometry, overlapping scratch and pass isolation");
}
'''

SPAN_FIXTURE = r'''
int main(void) {
    enum {FULL=96,WIDTH=3};
    uint32_t host[FULL*WIDTH],resident[FULL*WIDTH],packed[FULL*WIDTH],packed_device[FULL*WIDTH];
    int32_t metadata[512],metadata_device[512],expected[FULL];
    for(int i=0;i<FULL*WIDTH;i++){host[i]=0xfeed0000u+i;resident[i]=0x7f000000u+i;}
    for(int i=0;i<512;i++)metadata[i]=metadata_device[i]=0x123456;
    owner.nodes=nodes;owner.tensors=tensors;owner.tensor_count=4;owner.decode_metadata_stage=(unsigned char*)metadata;
    owner.decode_metadata_stage_bytes=sizeof(metadata);owner.incremental_plan.cuda_commands_active=1;
    register_span(host,resident,sizeof(host),0);register_span(packed,packed_device,sizeof(packed),0);
    register_span(metadata,metadata_device,sizeof(metadata),0);
    tensors[0]=tensor("input",host,WIDTH);tensors[0].numel=FULL*WIDTH;
    unsigned proofs=0,indexed=0,rejected=0;
    for(int slots_count=1;slots_count<=3;slots_count++)for(int mode=0;mode<7;mode++)
        for(int position=0;position<12;position++)for(int prefix=0;prefix<=1;prefix++) {
        int positions[3]={position,position/2,VX_DECODE_ROW_EMPTY};
        if(mode==6 && slots_count>1)positions[0]=VX_DECODE_ROW_EMPTY;
        int table[3][16];VxDecodeSlotPages pages[3];
        for(int slot=0;slot<3;slot++) {
            for(int page=0;page<16;page++)table[slot][page]=mode==2?15-page:page+slot*16;
            if(mode==3)table[slot][2]=VX_PAGED_KV_UNMAPPED;
            if(mode==4)table[slot][1]=INT_MAX;
            pages[slot]=(VxDecodeSlotPages){table[slot],2,16};
        }
        VxDecodeRowSet rows;
        assert(vx_decode_row_set_init(&rows,slots_count,positions,mode?pages:NULL)==VX_DECODE_ROW_SET_OK);
        VxDecodeRowOperand operand={&tensors[0],1,16,WIDTH,prefix};
        uint32_t first,count,indexed_rows;size_t bytes;
        memset(owner.decode_cuda_spans,0,sizeof(owner.decode_cuda_spans));
        int reference=prefix?vx_decode_row_set_prefix_rows(&rows,16,mode!=0,expected)
                            :vx_decode_row_set_write_rows(&rows,16,mode!=0,expected);
        count=(uint32_t)(slots_count*(prefix?rows.key_capacity:1));
        if(!prefix)for(int slot=0;slot<slots_count;slot++)if(rows.empty[slot])expected[slot]=-1;
        int valid=reference==VX_DECODE_ROW_SET_OK;
        if(valid)for(uint32_t row=0;row<count;row++)if(expected[row]<-1 || expected[row]>=FULL)valid=0;
        paged_tensor_name=mode?"input":"";
        int proven=vx_decode_cuda_span(&operand,&rows,&first,&count,&indexed_rows,&bytes);
        if(proven) {
            proofs++;assert(valid && indexed_rows==FULL && bytes==sizeof(host));
            for(uint32_t row=0;row<count;row++)assert(expected[row]==(int)(first+row));
            /* The second proof and the transfer must not touch index scratch. */
            int32_t saved[512];memcpy(saved,metadata,sizeof(saved));
            assert(vx_decode_cuda_span(&operand,&rows,&first,&count,&indexed_rows,&bytes));
            T narrow=tensors[0];narrow.data=(float*)packed;narrow.numel=(long)count*WIDTH;
            assert(vx_decode_cuda_transfer(&operand,&rows,&narrow,metadata,0));drain();
            assert(!memcmp(saved,metadata,sizeof(saved)));
        } else if(valid) {
            indexed++;T narrow=tensors[0];narrow.data=(float*)packed;narrow.numel=(long)count*WIDTH;
            assert(vx_decode_cuda_transfer(&operand,&rows,&narrow,metadata,0));drain();
        } else {
            rejected++;uint32_t selected,extent;size_t storage;
            assert(!vx_decode_cuda_indices(&operand,&rows,metadata,&selected,&extent,&storage));
        }
        if(valid)for(uint32_t row=0;row<count;row++)for(int word=0;word<WIDTH;word++)
            assert(packed_device[row*WIDTH+word]==(expected[row]<0?0:resident[expected[row]*WIDTH+word]));
        vx_decode_row_set_dispose(&rows);
    }
    assert(proofs>100 && indexed>100 && rejected>0);
    /* A cached row proof still checks each operand's current extent and row
     * values. A new pass clears the proof before a page table may be reused. */
    int position=3,table[]={2,3,4,5};VxDecodeSlotPages pages={table,2,4};VxDecodeRowSet rows;
    assert(vx_decode_row_set_init(&rows,1,&position,&pages)==VX_DECODE_ROW_SET_OK);
    paged_tensor_name="input";VxDecodeRowOperand operand={&tensors[0],1,8,WIDTH,1};
    uint32_t first,count,extent;size_t bytes;memset(owner.decode_cuda_spans,0,sizeof(owner.decode_cuda_spans));
    assert(vx_decode_cuda_span(&operand,&rows,&first,&count,&extent,&bytes) && first==4 && count==4);
    tensors[0].numel=6*WIDTH;assert(!vx_decode_cuda_span(&operand,&rows,&first,&count,&extent,&bytes));
    tensors[0].numel=FULL*WIDTH;table[0]=7;table[1]=8;
    memset(owner.decode_cuda_spans,0,sizeof(owner.decode_cuda_spans));
    assert(vx_decode_cuda_span(&operand,&rows,&first,&count,&extent,&bytes) && first==14);
    rows.positions[0]=4;rows.kv_lengths[0]=rows.key_capacity=5;
    assert(!vx_decode_cuda_span(&operand,&rows,&first,&count,&extent,&bytes));
    operand.prefix=0;assert(vx_decode_cuda_span(&operand,&rows,&first,&count,&extent,&bytes) && first==8);
    unsigned before_copies=copies,before_uploads=uploads;
    rows.pages[0].page_tokens=0;
    assert(!vx_decode_cuda_span(&operand,&rows,&first,&count,&extent,&bytes));
    assert(!vx_decode_cuda_indices(&operand,&rows,metadata,&count,&extent,&bytes));
    rows.pages[0].page_tokens=2;operand.prefix=1;
    assert(!vx_decode_cuda_indices(&operand,&rows,metadata+511,&count,&extent,&bytes));
    assert(copies==(int)before_copies && uploads==(int)before_uploads);
    vx_decode_row_set_dispose(&rows);
    puts("PASS CUDA span proofs: checked row corpus, exact numerical copies, page bounds, padding, permutations and pass invalidation");
}
'''

MASK_CACHE_CHECKS = r'''
    owner.incremental_plan.cuda_commands_active=1;owner.decode_cuda_mask_valid=0;
    nodes[0].operator_kind=VX_OP_CROSS_SDPA;p=projection(2,NULL);p.attention=p.causal=1;p.keys=3;
    p.mask=&tensors[3];p.mask_layout=ATTN_MASK_KEY;tensors[3].is_graph_input=1;
    mask_reads=0;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_mask_sum==2 && mask_reads==3);
    prior=uploads;float selected[WIDTH];memcpy(selected,resident+(FULL+2)*WIDTH,sizeof(selected));
    for(int layer=0;layer<8;layer++)assert(!vx_decode_cuda_project(nodes,0,1,&p));
    assert(mask_reads==3 && uploads==(int)prior && !memcmp(selected,resident+(FULL+2)*WIDTH,sizeof(selected)));
    /* Causal lengths, source data/descriptor/layout, and overwritten scratch
     * each invalidate the exact key before the next numerical invocation. */
    p.rows.kv_lengths[0]=2;assert(!vx_decode_cuda_project(nodes,0,1,&p));
    assert(last_mask_sum==1 && mask_reads==5);
    p.causal=0;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(last_mask_sum==2 && mask_reads==8);
    tensors[3].shape[0]=8;tensors[3].numel=8;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(mask_reads==11 && last_mask_sum==2);
    int32_t replacement[9]={0,0,1,1,1,1,1,1,1};tensors[3].data=(float*)replacement;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(mask_reads==14 && last_mask_sum==1);
    vx_decode_cuda_metadata_write(metadata,1);metadata[0]=123;
    assert(!owner.decode_cuda_mask_valid && !vx_decode_cuda_project(nodes,0,1,&p));
    assert(mask_reads==17 && last_mask_sum==1);
    /* Controls may change between passes. The existing run-begin validity
     * reset is mandatory even when storage and all descriptors stay equal. */
    replacement[0]=1;owner.decode_cuda_mask_valid=0;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(mask_reads==20 && last_mask_sum==2);
    owner.incremental_plan.cuda_commands_active=0;replacement[1]=1;
    assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(mask_reads==23 && last_mask_sum==3);
    replacement[2]=0;assert(!vx_decode_cuda_project(nodes,0,1,&p));assert(mask_reads==26 && last_mask_sum==2);
'''


class CudaRowTransferTests(unittest.TestCase):
    def test_deferred_contiguous_and_indexed_rows(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        source = (root / "native/src/backends/cuda/host/cuda_graph_quantized_boundary_shape_host.inc").read_text()
        translation = PREAMBLE + function_source(source, "cuda_graph_row_span_transfer")
        translation += function_source(source, "cuda_graph_row_index_transfer") + FIXTURE
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-row-transfer-") as temporary:
            unit, binary = Path(temporary) / "test.c", Path(temporary) / "test"
            unit.write_text(translation)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(unit), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("PASS CUDA resident row copies", result.stdout)

    def test_production_row_projection_views(self):
        self._row_fixture(ROW_FIXTURE, "PASS CUDA production row projection")

    def test_metadata_reuse_preserves_control_values(self):
        self._row_fixture(ROW_FIXTURE[:ROW_FIXTURE.index("int main(void)")] + METADATA_FIXTURE,
                          "PASS CUDA metadata exact reuse")

    def test_proven_span_numerical_and_adversarial_rows(self):
        self._row_fixture(ROW_FIXTURE[:ROW_FIXTURE.index("int main(void)")] + SPAN_FIXTURE,
                          "PASS CUDA span proofs")

    def test_pass_mask_cache_reuses_and_invalidates(self):
        fixture = METADATA_FIXTURE.replace('    restored(&saved_input,&saved_output);\n    puts(',
                                           MASK_CACHE_CHECKS + '    restored(&saved_input,&saved_output);\n    puts(')
        self._row_fixture(ROW_FIXTURE[:ROW_FIXTURE.index("int main(void)")] + fixture,
                          "PASS CUDA metadata exact reuse")

    def _row_fixture(self, fixture, expected):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        runtime = root / "native/src/runtime"
        backend = (root / "native/src/backends/cuda/host/cuda_graph_quantized_boundary_shape_host.inc").read_text()
        source = (runtime / "engine_runtime_row_cuda.inc").read_text()
        translation = PREAMBLE + ROW_PREAMBLE
        translation += function_source(backend, "cuda_graph_row_span_transfer")
        translation += function_source(backend, "cuda_graph_row_index_transfer")
        for name in ("vx_decode_cuda_metadata_write", "vx_decode_cuda_storage", "vx_decode_cuda_selection_geometry",
                     "vx_decode_cuda_span", "vx_decode_cuda_indices", "vx_decode_cuda_view_safe",
                     "vx_decode_cuda_contiguous_view", "vx_decode_cuda_transfer", "vx_decode_cuda_mask_key",
                     "vx_decode_cuda_mask_key_equal", "vx_decode_cuda_project"):
            function = function_source(source, name)
            if name == "vx_decode_cuda_project":
                function = function.replace("attention_mask_allowed(", "counted_attention_mask_allowed(")
            translation += function
        translation += fixture
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-row-projection-") as temporary:
            unit, binary = Path(temporary) / "test.c", Path(temporary) / "test"
            unit.write_text(translation)
            includes = ["-I" + str(root / path) for path in ("native/include", "native/third_party", "native/src",
                        "native/src/runtime", "native/src/kernels", "native/src/backends")]
            command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=1",
                       "-DVOLVOXAI_ENABLE_WEBGPU=0", *includes, str(unit), str(runtime / "decode_row_set.c"),
                       str(runtime / "attention_mask.c"),
                       "-o", str(binary)]
            built = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
