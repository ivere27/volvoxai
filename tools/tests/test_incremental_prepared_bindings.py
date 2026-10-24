"""Prepared row references/provenance and dispatch dirtiness share real C state."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

from tools.tests.test_cuda_row_transfer import PREAMBLE, ROW_PREAMBLE, ROW_FIXTURE
from tools.tests import function_source


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "native/src/runtime"


def packed_descriptor_validation_source():
    """Use the real packed descriptor validator in extracted row fixtures."""
    source = (RUNTIME / "engine_runtime_llm.inc").read_text()
    result = '#include "llm_ops.h"\n#include "packed_linear.h"\n#include <limits.h>\n#include <math.h>\n'
    result += '''static T* nout(Node* node, VxPortKind port) {
        return port == VX_PORT_OUT ? t_find(node->out) : NULL;
    }\n'''
    result += function_source((ROOT / "native/src/kernels/llm_ops.inc").read_text(),
                              "vx_llm_work_items")
    for name in ("vx_llm_operator", "vx_llm_tensor_valid", "vx_engine_llm_node_args"):
        result += function_source(source, name)
    return result


FIXTURE = r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "../src/runtime/runtime_state.c"
#include "../src/runtime/incremental_runtime.c"

static int lookups, classifications, calls, graph_mode, proof_calls, refuse;
#if VOLVOXAI_ENABLE_CUDA
static int pair_result,pair_calls;
#endif
static unsigned char expected[6];
static float values[2][6][12];
static VxEngineState contexts[2];
void cJSON_Delete(cJSON* item) {(void)item;assert(!item);}
T* t_find(const char* name) {
    VxEngineState* s=vx_engine_state_current();lookups++;
    for(int i=0;i<s->tensor_count;i++) if(!strcmp(s->tensors[i].name,name)) return &s->tensors[i];
    return NULL;
}
uint64_t volvoxai_engine_model_generation_locked(void) {return vx_engine_state_current()->model_generation;}
const char* vx_operator_kind_name(VxOperatorKind op) {(void)op;return "fixture";}
int volvoxai_engine_tensor_is_model_weight_locked(const char* name) {
    classifications++;T* t=t_find(name);
    return t && !strcmp(name,"weight") && t->dtype==T_F32 && t->ndim==1 &&
        t->shape[0]==3 && t->numel==3 && t->elem_size==4 && !t->is_graph_input &&
        t->data==values[vx_engine_state_current()==&contexts[1]][5];
}
int vx_runtime_backend_has_graph(void) {return graph_mode;}
int vx_runtime_backend_has_device_rows(void) {return graph_mode;}
void vx_runtime_backend_begin_forward(int replay,uint64_t generation) {(void)generation;assert(!replay);}
int vx_runtime_backend_prepare_forward(void) {return 0;}
int vx_runtime_backend_end_forward(int ok) {return ok?0:-1;}
void vx_adapter_run_end(void) {}
int vx_adapter_run_begin(void) {return 0;}
void vk_mark_owned_tensors_host_dirty(void) {}
int vx_trace_nodes(const VxTraceScope* trace) {(void)trace;return 0;}
int vx_runtime_node_incremental_row_compatible(Node* node,int index,int row) {
    (void)node;(void)index;(void)row;proof_calls++;return !refuse;
}
void vx_decode_session_invalidate_cache_locked(void) {}
double volvoxai_engine_now_ms(void) {return 0;}
int run_node(Node* node,int index,int last) {
    (void)last;calls++;
    /* An independent pre-node closure is the oracle, including the current
     * output if it aliases an unchanged input. Every tensor must agree. */
    for(int i=0;i<6;i++) assert(vx_incremental_tensor_dirty_before_node_locked(&g_t[i],index)==expected[i]);
    T* a=vx_incremental_node_input_locked(node,0);
    T* b=node->nin>1?vx_incremental_node_input_locked(node,1):NULL;
    T* out=vx_incremental_node_output_locked(node,-1);
    for(int i=0;i<12;i++) {
        if(node->operator_kind==VX_OP_SILU)out->data[i]=a->data[i]/(1.f+expf(-a->data[i]));
        else if(node->operator_kind==VX_OP_MUL)out->data[i]=a->data[i]*b->data[i];
        else out->data[i]=index==0?a->data[i]+b->data[i%3]:index==1?a->data[i]*2:a->data[i]+b->data[i];
    }
    return 0;
}
int run_node_cpu_direct(Node* node,int index,int last) {return run_node(node,index,last);}
int vx_run_node_profiled(Node* node,int index,int last,int cpu) {(void)cpu;return run_node(node,index,last);}
#if VOLVOXAI_ENABLE_CUDA
int vx_runtime_cuda_prepared_row_replay_locked(int row,int seal) {
    (void)row;(void)seal;return 0;
}
int vx_runtime_cuda_prepared_row_node_locked(int node,int row,int last) {
    (void)node;(void)row;(void)last;return 0;
}
int vx_runtime_cuda_gemv_pair_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_packed_gemv_pair_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_packed_gemv_silu_mul_quad_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_packed_gemv_triple_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_rmsnorm_packed_gemv_triple_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_rmsnorm_packed_gemv_silu_mul_quad_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_rope_pair_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_packed_gemv_add_pair_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_gemv_add_pair_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_head_rms_rope_triple_locked(int node,int row) {
    (void)node;(void)row;return 0;
}
int vx_runtime_cuda_silu_mul_pair_locked(int node,int row) {
    (void)row;pair_calls++;assert(node==0 && g_n[0].operator_kind==VX_OP_SILU && g_n[1].operator_kind==VX_OP_MUL);
    if(pair_result!=1)return pair_result;
    for(int i=0;i<12;i++) {
        g_t[2].data[i]=g_t[0].data[i]/(1.f+expf(-g_t[0].data[i]));
        g_t[3].data[i]=g_t[2].data[i]*g_t[1].data[i];
    }
    return 1;
}
#endif

static void setup(int context) {
    VxEngineState* s=&contexts[context];g_current_engine_state=s;
    s->model_generation=7;s->loaded=1;s->active_row=-1;s->execution_row=-1;
    assert(vx_engine_state_reserve_graph_metadata(s,6,3)==VX_ENGINE_RESULT_OK);
    s->tensor_count=6;s->node_count=3;
    const char* names[]={"x","y","a","b","out","weight"};
    for(int i=0;i<6;i++) {
        T* t=&s->tensors[i];strcpy(t->name,names[i]);t->data=values[context][i];
        t->dtype=T_F32;t->elem_size=4;t->ndim=i==5?1:3;t->numel=i==5?3:12;
        if(i==5)t->shape[0]=3;else{t->shape[0]=1;t->shape[1]=4;t->shape[2]=3;}
        t->is_graph_input=i<2;
        for(int k=0;k<12;k++) values[context][i][k]=(float)(context*31+i*3+k)*.125f;
    }
    const int input_indices[3][2]={{0,5},{1,-1},{2,3}};
    for(int i=0;i<3;i++) {
        Node* n=&s->nodes[i];n->nin=i==1?1:2;n->nout=1;
        assert(!vx_node_reserve_refs(n,n->nin,1));
        strcpy(n->out,names[i+2]);strcpy(n->outs[0].name,n->out);
        for(int p=0;p<n->nin;p++){strcpy(n->ins[p].name,names[input_indices[i][p]]);n->ins[p].port=p?VX_PORT_WEIGHT:VX_PORT_INPUT;}
    }
    assert(!incremental_plan_build_locked());
}

static void bindings(void) {
    setup(0);setup(1);g_current_engine_state=&contexts[0];
    classifications=lookups=0;
    assert(vx_incremental_node_input_locked(&g_n[0],0)==&g_t[0]);
    assert(vx_incremental_node_output_locked(&g_n[2],-1)==&g_t[4]);
    assert(vx_incremental_tensor_is_model_weight_locked(&g_t[5]));
    assert(!vx_incremental_tensor_is_model_weight_locked(&g_t[0]));
    assert(!lookups && !classifications);
    T saved=g_t[5];
    for(int mutation=0;mutation<7;mutation++) {
        g_t[5]=saved;
        if(mutation==0)g_t[5].data=values[0][4];
        if(mutation==1)g_t[5].shape[0]=4;
        if(mutation==2)g_t[5].ndim=2;
        if(mutation==3)g_t[5].numel=4;
        if(mutation==4)g_t[5].dtype=T_I32;
        if(mutation==5)g_t[5].elem_size=2;
        if(mutation==6)g_t[5].is_graph_input=1;
        int before=classifications;assert(!vx_incremental_tensor_is_model_weight_locked(&g_t[5]));
        assert(classifications==before+1);
    }
    g_t[5]=saved;
    Node temporary=g_n[0];int before=lookups;
    assert(vx_incremental_node_input_locked(&temporary,0)==&g_t[0] && lookups==before+1);
    g_current_engine_state=&contexts[1];before=lookups;
    assert(vx_incremental_node_input_locked(&contexts[0].nodes[0],0)==&g_t[0] && lookups==before+1);
    assert(vx_incremental_tensor_is_model_weight_locked(&g_t[5]));
    g_current_engine_state=&contexts[0];g_n[0].ins[0].tensor_index=1;
    contexts[0].model_generation++;before=lookups;
    assert(vx_incremental_node_input_locked(&g_n[0],0)==&g_t[0] && lookups==before+1);
    assert(!incremental_plan_build_locked());
    size_t bytes;assert(!vx_engine_state_graph_metadata_bytes_for(12,6,&bytes));
    assert(vx_engine_state_reserve_graph_metadata(&contexts[0],12,6)==VX_ENGINE_RESULT_OK);
    assert(!g_incremental_plan.valid && contexts[0].graph_metadata_bytes==bytes);
    uintptr_t start=(uintptr_t)contexts[0].graph_metadata_storage;
    uintptr_t records=(uintptr_t)contexts[0].incremental_scratch.tensor_bindings;
    assert(records>=start && records+12*sizeof(VxIncrementalTensorBinding)<=start+bytes);
    before=lookups;assert(vx_incremental_node_input_locked(&g_n[0],0)==&g_t[0] && lookups==before+1);
    assert(!incremental_plan_build_locked());
    /* Producer ordering retains the first private redirected write. */
    assert(g_incremental_scratch.tensor_bindings[2].producer==0);
    assert(g_incremental_scratch.tensor_bindings[5].producer==-1);
}

static void dependency_numerics(void) {
    g_current_engine_state=&contexts[0];
    /* The current output aliases an input in the last case. This verifies
     * that its early dirty mark cannot change the node's invariant proof. */
    for(int alias=0;alias<2;alias++) {
        strcpy(g_n[2].out,alias?"a":"out");strcpy(g_n[2].outs[0].name,g_n[2].out);
        contexts[0].model_generation++;assert(!incremental_plan_build_locked());
        for(int mode=0;mode<2;mode++) for(int selected=0;selected<8;selected++) {
            g_portable_selection_active=1;
            for(int i=0;i<3;i++)g_portable_selected_nodes[i]=(selected>>i)&1;
            memset(g_dirty,0,6);memset(expected,0,sizeof(expected));
            g_dirty[0]=expected[0]=1;g_dirty[1]=expected[1]=1;
            for(int i=0;i<6;i++)for(int k=0;k<12;k++)values[0][i][k]=(float)(i*3+k)*.125f;
            float reference[6][12];memcpy(reference,values[0],sizeof(reference));
            VxIncrementalRun run={.row=1,.direct_cpu=!mode};calls=0;
            for(int i=0;i<3;i++) {
                int actual=vx_incremental_run_next_locked(&run);
                if(!(selected&(1<<i)))continue;
                assert(actual==i);
                int output=alias&&i==2?2:i+2;
                for(int k=0;k<12;k++)reference[output][k]=i==0?reference[0][k]+reference[5][k%3]:i==1?reference[1][k]*2:reference[2][k]+reference[3][k];
                assert(!vx_incremental_run_node_locked(&run,i));
                expected[output]=1;
                assert(!g_incremental_plan.row_probe_tensors);
            }
            assert(vx_incremental_run_next_locked(&run)==3);
            assert(!memcmp(reference,values[0],sizeof(reference)));
            /* A planner call after dispatch must still propagate the roots
             * rather than accidentally borrowing an expired node snapshot. */
            for(int i=0;i<6;i++)assert(vx_incremental_tensor_dirty_before_node_locked(&g_t[i],3)==expected[i]);
        }
    }
    g_portable_selection_active=0;
    /* Ordinary dirty-root scheduling, including cached branches. */
    for(int roots=0;roots<4;roots++) {
        memset(g_dirty,0,6);memset(expected,0,6);g_dirty[0]=expected[0]=roots&1;g_dirty[1]=expected[1]=(roots>>1)&1;
        VxIncrementalRun run={.row=1};
        int node;
        while((node=vx_incremental_run_next_locked(&run))<3) {
            assert(!vx_incremental_run_node_locked(&run,node));expected[g_n[node].primary_output_index]=1;
        }
        assert(node==3);
    }
    /* A full refresh selects invariant branches as well. Its pre-node proof
     * must use actual predecessor execution, not only the original roots. */
    memset(g_dirty,0,6);memset(expected,0,6);VxIncrementalRun full={.force_full=1};
    for(int node=0;node<3;node++) {
        assert(vx_incremental_run_next_locked(&full)==node);
        assert(!vx_incremental_run_node_locked(&full,node));expected[g_n[node].primary_output_index]=1;
    }
}

static void command_domain_guards(void) {
#if VOLVOXAI_ENABLE_CUDA
    g_current_engine_state=&contexts[0];
    assert(!incremental_plan_build_locked());
    VxIncrementalCudaCommand* commands=g_incremental_scratch.cuda_commands;
    assert(commands && sizeof(*commands)<=256);
    uintptr_t begin=(uintptr_t)contexts[0].graph_metadata_storage;
    assert((uintptr_t)commands>=begin && (uintptr_t)(commands+g_node_capacity)<=begin+contexts[0].graph_metadata_bytes);
    commands[0].valid=1;
    assert(!incremental_cuda_commands_prepare_locked(0) && !g_incremental_plan.cuda_commands_active && commands[0].valid);
    assert(!incremental_cuda_commands_prepare_locked(1) && g_incremental_plan.cuda_commands_active && commands[0].valid);
    T original=g_t[0];
    for(int mutation=0;mutation<8;mutation++) {
        g_t[0]=original;commands[0].valid=1;
        if(mutation==0)g_t[0].data=values[0][1];
        if(mutation==1)g_t[0].numel++;
        if(mutation==2)g_t[0].elem_size=2;
        if(mutation==3)g_t[0].dtype=T_I32;
        if(mutation==4)g_t[0].ndim=2;
        if(mutation==5)g_t[0].shape[1]++;
        if(mutation==6)g_t[0].is_graph_input=0;
        if(mutation==7)g_t[0].shape[7]++;
        assert(!incremental_cuda_commands_prepare_locked(1) && !commands[0].valid && g_incremental_plan.cuda_commands_active);
        g_t[0]=original;assert(!incremental_cuda_commands_prepare_locked(1));
    }
    /* New value bytes do not replace storage/shape or require interpretation. */
    commands[0].valid=1;g_t[0].data[0]+=1.f;
    assert(!incremental_cuda_commands_prepare_locked(1) && commands[0].valid);
    classifications=lookups=0;assert(!incremental_cuda_commands_prepare_locked(1));
    assert(!classifications && !lookups);
    /* A generation change clears the private commands even for equal-sized
     * descriptors; the immutable typed parameters belong to that revision. */
    contexts[0].model_generation++;assert(!incremental_plan_build_locked());
    assert(!commands[0].valid && !g_incremental_plan.cuda_commands_active);
    g_current_engine_state=&contexts[1];assert(!incremental_cuda_commands_prepare_locked(1));
    assert(g_incremental_scratch.cuda_commands!=commands);
    g_incremental_scratch.cuda_commands[0].valid=1;
    g_current_engine_state=&contexts[0];assert(!commands[0].valid);
    commands[0].valid=1;vx_incremental_invalidate_locked();
    assert(!g_incremental_plan.cuda_commands_active && !g_cache_valid);
#endif
}

static void closure_proof(void) {
    VxHybridRowPlan plan;
    assert(!hybrid_row_plan_init(&plan));memset(g_dirty,0,6);g_dirty[0]=1;
    unsigned char prior[6];memcpy(prior,g_dirty,6);refuse=0;proof_calls=0;
    assert(hybrid_row_plan_build_locked(1,&plan)==1 && proof_calls==2);
    assert(!memcmp(prior,g_dirty,6) && !g_incremental_plan.row_probe_tensors);
    refuse=1;assert(hybrid_row_plan_build_locked(1,&plan)==0);
    assert(!memcmp(prior,g_dirty,6) && !g_incremental_plan.row_probe_tensors);refuse=0;
    assert(hybrid_row_plan_build_locked(1,&plan)==1);
    /* Once the complete B>1 device closure has passed, the scheduler consumes
     * that proof; an unprepared CPU/device caller still checks compatibility. */
    VxIncrementalRun prepared={.row_prepared=1};g_decode_slots=2;proof_calls=0;
    assert(vx_incremental_run_next_locked(&prepared)==0 && !proof_calls);
    prepared.row_prepared=0;assert(vx_incremental_run_next_locked(&prepared)==0 && proof_calls==1);
    g_decode_slots=0;
}

static void pair_selection_and_failure(void) {
#if VOLVOXAI_ENABLE_CUDA
    g_n[0].operator_kind=VX_OP_SILU;g_n[0].nin=1;
    g_n[1].operator_kind=VX_OP_MUL;g_n[1].nin=2;
    assert(!vx_node_reserve_refs(&g_n[1],2,1));
    strcpy(g_n[1].ins[0].name,"a");strcpy(g_n[1].ins[1].name,"y");
    contexts[0].model_generation++;assert(!incremental_plan_build_locked());
    g_portable_selection_active=1;g_portable_selected_nodes[0]=g_portable_selected_nodes[1]=1;
    g_portable_selected_nodes[2]=0;
    for(int result=0;result<2;result++) {
        memset(g_dirty,0,6);memset(expected,0,6);g_dirty[0]=expected[0]=1;
        for(int i=0;i<12;i++){g_t[0].data[i]=(float)(i-5)*.125f;g_t[1].data[i]=(float)(i+1)*.25f;}
        VxIncrementalRun run={.row=1,.cuda_pair_eligible=1};pair_result=result;pair_calls=calls=0;
        assert(!vx_incremental_run_node_locked(&run,0) && pair_calls==1);
        if(!result){expected[2]=1;assert(run.cursor==1);assert(!vx_incremental_run_node_locked(&run,1));}
        assert(run.cursor==2 && run.executed==2 && g_dirty[2] && g_dirty[3]);
        for(int i=0;i<12;i++) {
            double x=(double)(i-5)*.125, y=(double)(i+1)*.25;
            double silu=x/(1.+exp(-x));assert(fabs(g_t[2].data[i]-silu)<1e-6);
            assert(fabs(g_t[3].data[i]-silu*y)<1e-6);
        }
        assert(!g_incremental_plan.row_probe_tensors && calls==(!result?2:0));
    }
    /* Selecting only SiLU cannot submit/write the unselected Mul. */
    memset(g_dirty,0,6);memset(expected,0,6);g_dirty[0]=expected[0]=1;
    g_portable_selected_nodes[1]=0;VxIncrementalRun only={.cuda_pair_eligible=1};pair_calls=0;pair_result=1;
    assert(!vx_incremental_run_node_locked(&only,0) && !pair_calls && only.cursor==1 && !g_dirty[3]);
    /* Preserved logical boundaries disable lowering (trace/debug setting). */
    memset(g_dirty,0,6);memset(expected,0,6);g_dirty[0]=expected[0]=1;
    g_portable_selected_nodes[1]=1;VxIncrementalRun boundaries={0};pair_calls=0;
    assert(!vx_incremental_run_node_locked(&boundaries,0) && !pair_calls && boundaries.cursor==1 && !g_dirty[3]);
    /* Submitted failure is terminal; it never falls back or advances nodes. */
    memset(g_dirty,0,6);memset(expected,0,6);g_dirty[0]=expected[0]=1;
    VxIncrementalRun fail={.cuda_pair_eligible=1,.begun=1};pair_result=-1;calls=pair_calls=0;
    assert(vx_incremental_run_node_locked(&fail,0)==-1 && pair_calls==1 && !calls && !fail.cursor && !fail.executed);
    assert(!g_incremental_plan.row_probe_tensors);
    assert(vx_incremental_run_end_locked(&fail,0)==-1 && !g_cache_valid && !g_portable_selection_active);
#endif
}
int main(void) {
    bindings();dependency_numerics();command_domain_guards();closure_proof();pair_selection_and_failure();
    for(int i=0;i<2;i++)vx_engine_state_release_graph_metadata(&contexts[i]);
    puts("PASS prepared bindings: context/revision/storage/shape guards, charged growth, exact dependency CPU/device numerics");
}
'''


GEOMETRY_REFRESH_CHECKS = r'''
#if VOLVOXAI_ENABLE_CUDA
static void reset_geometry_refresh(void) {
    for(int c=0;c<2;c++)vx_engine_state_release_graph_metadata(&contexts[c]);
    memset(contexts,0,sizeof(contexts));setup(0);g_current_engine_state=&contexts[0];
    g_cache_valid=1;g_weight_caches_dirty=0;memset(g_dirty,0,6);g_dirty[0]=1;
    memset(g_incremental_scratch.cuda_commands,0x5a,3*sizeof(VxIncrementalCudaCommand));
    classifications=lookups=0;
}
static void geometry_refresh_checks(void) {
    for(int mutation=0;mutation<20;mutation++) {
        reset_geometry_refresh();int result=0,expected_classes=1,expected_lookups=1;
        int weight=1;T original=g_t[5];
        if(mutation==0){g_t[0].data[0]+=1.f;expected_classes=expected_lookups=0;}
        if(mutation==1)g_t[0].data=values[0][1];
        if(mutation==2){g_t[5].data=values[0][4];weight=0;}
        if(mutation==3){g_t[5].numel=4;weight=0;}
        if(mutation==4){g_t[5].elem_size=2;weight=0;}
        if(mutation==5){g_t[5].dtype=T_I32;weight=0;}
        if(mutation==6){g_t[5].ndim=2;weight=0;}
        if(mutation==7){g_t[5].shape[0]=4;weight=0;}
        if(mutation==8){g_t[5].is_graph_input=1;weight=0;}
        if(mutation==9)g_t[0].shape[7]=9;
        if(mutation>=10) {
            g_t[0].data=values[0][1];expected_classes=6;expected_lookups=17;
            if(mutation==10)g_incremental_plan.valid=0;
            if(mutation==11)contexts[0].model_generation++;
            if(mutation==12){contexts[0].compiled_weight_conversions=(VxCompiledWeightConversions*)(uintptr_t)1;g_t[5].owns=1;weight=0;}
            if(mutation==13)g_incremental_scratch.tensor_capacity=5;
            if(mutation==14)g_incremental_plan.node_capacity=2;
            if(mutation==15)g_incremental_scratch.node_capacity=2;
            if(mutation==16)g_tensor_capacity=5;
            if(mutation==17){strcpy(g_n[0].ins[0].name,"y");g_incremental_plan.valid=0;}
            if(mutation==18){strcpy(g_n[0].ins[0].name,"missing");g_incremental_plan.valid=0;}
            if(mutation==19)g_incremental_plan.node_count--;
            if(mutation==13 || mutation==14 || mutation==16){result=-1;expected_classes=expected_lookups=0;}
            if(mutation==18){result=-1;expected_lookups=-1;}
        }
        assert(incremental_cuda_commands_prepare_locked(1)==result);
        assert(classifications==expected_classes && (expected_lookups<0 || lookups==expected_lookups));
        assert(g_cache_valid && !g_weight_caches_dirty && g_dirty[0]);
        for(int i=1;i<6;i++)assert(!g_dirty[i]);
        assert(g_incremental_plan.cuda_commands_active==!result);
        if(!result) {
            const VxIncrementalTensorBinding* b=&g_incremental_scratch.tensor_bindings[5];
            assert(b->is_model_weight==weight && b->producer==-1);
            assert(b->data==g_t[5].data && b->numel==g_t[5].numel && b->dtype==g_t[5].dtype &&
                b->elem_size==g_t[5].elem_size && b->ndim==g_t[5].ndim && b->is_graph_input==g_t[5].is_graph_input &&
                !memcmp(b->shape,g_t[5].shape,sizeof(b->shape)));
            assert(g_incremental_scratch.tensor_bindings[2].producer==0 &&
                g_incremental_scratch.tensor_bindings[3].producer==1 && g_incremental_scratch.tensor_bindings[4].producer==2);
            assert(g_n[0].ins[0].tensor_index==(mutation==17?1:0) && g_n[0].ins[1].tensor_index==5);
            assert(g_n[2].ins[0].tensor_index==2 && g_n[2].ins[1].tensor_index==3 && g_n[2].primary_output_index==4);
            const unsigned char* p=(const unsigned char*)g_incremental_scratch.cuda_commands;
            size_t cleared=(mutation==0?0:g_incremental_scratch.node_capacity)*sizeof(VxIncrementalCudaCommand);
            for(size_t i=0;i<3*sizeof(VxIncrementalCudaCommand);i++)assert(p[i]==(i<cleared?0:0x5a));
        }
        /* Restoring a weight's live descriptor proves negative->positive;
         * one changed descriptor refreshes without re-resolving graph edges. */
        if(mutation==2) {
            g_t[5]=original;classifications=lookups=0;
            assert(!incremental_cuda_commands_prepare_locked(1) && classifications==1 && lookups==1);
            assert(g_incremental_scratch.tensor_bindings[5].is_model_weight);
        }
        /* Mixed compiled CPU ownership changes use the complete original
         * path when another live descriptor changes. No owns-only new cache
         * invariant is introduced by this refresh optimization. */
        if(mutation==12) {
            g_t[5].owns=0;g_t[0].data=values[0][2];classifications=lookups=0;
            assert(!incremental_cuda_commands_prepare_locked(1) && classifications==6 && lookups==17);
            assert(g_incremental_scratch.tensor_bindings[5].is_model_weight);
        }
    }
    puts("PASS geometry refresh:20same/replaced descriptors, weight flips, structural/ownership/capacity fallbacks, scratch bounds, preserved dirty closure");
}
#endif
'''



BINDING_SCAN_CHECKS = r'''

#if VOLVOXAI_ENABLE_CUDA
/* Freeze the original acceptance predicates independently of the production
 * local-state scan. No tensor payload or descriptor is simplified here. */
static int original_tensor_binding_current(int index) {
    const T* tensor = &g_t[index];
    const VxIncrementalTensorBinding* binding = &g_incremental_scratch.tensor_bindings[index];
    return tensor->ndim >= 0 && tensor->ndim <= 8 &&
        tensor->data == binding->data && tensor->numel == binding->numel &&
        tensor->elem_size == binding->elem_size && tensor->dtype == binding->dtype &&
        tensor->ndim == binding->ndim && tensor->is_graph_input == binding->is_graph_input &&
        !memcmp(tensor->shape, binding->shape, sizeof(binding->shape));
}

static int original_bindings_current(void) {
    if (!g_incremental_plan.valid ||
        g_incremental_plan.generation != volvoxai_engine_model_generation_locked() ||
        g_incremental_plan.tensor_count != g_nt || g_incremental_plan.node_count != g_nn ||
        g_nt < 0 || g_nn < 0 || !g_t || !g_n ||
        g_incremental_scratch.tensor_capacity < (size_t)g_nt ||
        g_incremental_scratch.node_capacity < (size_t)g_nn ||
        !g_incremental_scratch.tensor_bindings || !g_incremental_scratch.cuda_commands)
        return 0;
    for (int index = 0; index < g_nt; index++)
        if (!original_tensor_binding_current(index)) return 0;
    return 1;
}

static VxEngineState binding_states[2];
static T binding_tensors[2][256];
static Node binding_nodes[2];
static VxIncrementalTensorBinding binding_records[2][256];
static VxIncrementalCudaCommand binding_commands[2];
static int binding_comparisons;

static void binding_scan_setup(int count) {
    assert(count >= 0 && count <= 256);
    for (int owner = 0; owner < 2; owner++) {
        VxEngineState* state = &binding_states[owner];
        memset(state, 0, sizeof(*state));
        memset(binding_tensors[owner], 0, sizeof(binding_tensors[owner]));
        memset(binding_records[owner], 0, sizeof(binding_records[owner]));
        state->tensors = binding_tensors[owner];
        state->nodes = &binding_nodes[owner];
        state->tensor_count = count;
        state->node_count = 1;
        state->model_generation = 7;
        state->incremental_plan.valid = 1;
        state->incremental_plan.generation = 7;
        state->incremental_plan.tensor_count = count;
        state->incremental_plan.node_count = 1;
        state->incremental_scratch.tensor_capacity = (size_t)count;
        state->incremental_scratch.node_capacity = 1;
        state->incremental_scratch.tensor_bindings = binding_records[owner];
        state->incremental_scratch.cuda_commands = &binding_commands[owner];
        for (int index = 0; index < count; index++) {
            T* tensor = &state->tensors[index];
            VxIncrementalTensorBinding* binding = &binding_records[owner][index];
            tensor->data = values[owner][index % 6];
            binding->data = tensor->data;
            tensor->ndim = binding->ndim = 2;
            tensor->dtype = binding->dtype = T_F32;
            tensor->numel = binding->numel = 12;
            tensor->elem_size = binding->elem_size = 4;
            tensor->is_graph_input = binding->is_graph_input = index == 0;
            for (int dim = 0; dim < 8; dim++)
                tensor->shape[dim] = binding->shape[dim] = dim < 2 ? dim + 3 : 0;
        }
    }
    g_current_engine_state = &binding_states[0];
}

static void binding_scan_compare(int expected_result) {
    assert(original_bindings_current() == expected_result);
    assert(incremental_cuda_bindings_current_locked() == expected_result);
    binding_comparisons++;
}

static void binding_scan_predicate_checks(void) {
    VxEngineState* previous = g_current_engine_state;
    for (int owner = 0; owner < 2; owner++) {
        for (int mutation = 0; mutation < 14; mutation++) {
            binding_scan_setup(6);
            g_current_engine_state = &binding_states[owner];
            VxEngineState* state = g_current_engine_state;
            VxEngineState saved = *state;
            switch (mutation) {
                case 0: state->incremental_plan.valid = 0; break;
                case 1: state->incremental_plan.generation++; break;
                case 2: state->model_generation++; break;
                case 3: state->incremental_plan.tensor_count--; break;
                case 4: state->incremental_plan.node_count--; break;
                case 5: state->tensor_count = state->incremental_plan.tensor_count = -1; break;
                case 6: state->node_count = state->incremental_plan.node_count = -1; break;
                case 7: state->tensors = NULL; break;
                case 8: state->nodes = NULL; break;
                case 9: state->incremental_scratch.tensor_capacity = 5; break;
                case 10: state->incremental_scratch.node_capacity = 0; break;
                case 11: state->incremental_scratch.tensor_bindings = NULL; break;
                case 12: state->incremental_scratch.cuda_commands = NULL; break;
                case 13: state->tensor_count = state->incremental_plan.tensor_count = 0; break;
            }
            binding_scan_compare(mutation == 13);
            *state = saved;
            binding_scan_compare(1);
            g_current_engine_state = &binding_states[1 - owner];
            binding_scan_compare(1);
        }
        for (int index = 0; index < 6; index++) {
            for (int mutation = 0; mutation < 17; mutation++) {
                binding_scan_setup(6);
                g_current_engine_state = &binding_states[owner];
                T* tensor = &binding_tensors[owner][index];
                T saved = *tensor;
                switch (mutation) {
                    case 0: tensor->ndim = -1; break;
                    case 1: tensor->ndim = 9; break;
                    case 2: tensor->data = NULL; break;
                    case 3: tensor->numel++; break;
                    case 4: tensor->elem_size++; break;
                    case 5: tensor->dtype = T_I32; break;
                    case 6: tensor->ndim = 3; break;
                    case 7: tensor->is_graph_input ^= 1; break;
                    case 16: tensor->owns ^= 1; break;
                    default: tensor->shape[mutation - 8]++; break;
                }
                /* Storage ownership is intentionally absent from the original
                 * predicate; its independent provenance guards remain active. */
                binding_scan_compare(mutation == 16);
                *tensor = saved;
                binding_scan_compare(1);
                g_current_engine_state = &binding_states[1 - owner];
                binding_scan_compare(1);
            }
        }
    }
    const int counts[] = {0, 1, 256};
    for (size_t index = 0; index < sizeof(counts) / sizeof(counts[0]); index++) {
        binding_scan_setup(counts[index]);
        binding_scan_compare(1);
    }
    assert(binding_comparisons == 699);
    g_current_engine_state = previous;
    puts("PASS binding scan:699 original predicates, fields/shapes/capacities/generation and two-owner comparisons");
}
#endif
'''

HEAD_SCHEDULER_MOCK = r"""
static int head_active,head_result,head_calls;
static const char* head_routes[3];
int vx_runtime_cuda_head_rms_rope_triple_locked(int node,int row) {
    if(!head_active)return 0;
    head_calls++;assert(node==0 && row>=0 && row<4);
    if(head_result!=1)return head_result;
    /* Kernel arithmetic/alias safety has its own authentic reduction test.
     * This callee controls only the scheduler's admitted/miss/terminal paths. */
    for(int k=row*3;k<(row+1)*3;k++) {
        g_t[2].data[k]=g_t[0].data[k]+g_t[5].data[k%3];
        g_t[3].data[k]=g_t[2].data[k];
        g_t[4].data[k]=g_t[3].data[k]*2.f;
    }
    for(int i=0;i<3;i++)head_routes[i]="cuda-graph-row+head-rms-rope";
    return 1;
}
"""

HEAD_SCHEDULER_CHECKS = r"""
#if VOLVOXAI_ENABLE_CUDA
static VxIncrementalRun head_scheduler_setup(int row,int alias) {
    for(int c=0;c<2;c++)vx_engine_state_release_graph_metadata(&contexts[c]);
    memset(contexts,0,sizeof(contexts));setup(0);g_current_engine_state=&contexts[0];
    g_n[0].operator_kind=VX_OP_RMS_NORM;
    g_n[1].operator_kind=VX_OP_RESHAPE;g_n[1].skip=alias;
    strcpy(g_n[1].ins[0].name,"a");g_n[1].ins[0].port=VX_PORT_INPUT;
    g_n[2].operator_kind=VX_OP_ROPE;g_n[2].nin=1;
    strcpy(g_n[2].ins[0].name,"b");g_n[2].ins[0].port=VX_PORT_INPUT;
    if(alias)g_t[3].data=g_t[2].data;
    contexts[0].model_generation++;assert(!incremental_plan_build_locked());
    memset(g_dirty,0,6);memset(expected,0,6);g_dirty[0]=expected[0]=1;
    g_portable_selection_active=1;
    for(int i=0;i<3;i++) {
        g_portable_selected_nodes[i]=g_incremental_scratch.hybrid_nodes[i]=i!=1 || !alias;
        head_routes[i]=NULL;
    }
    contexts[0].runtime_route_backend=head_routes;
    head_active=head_result=1;head_calls=calls=0;
    return (VxIncrementalRun){.row=row,.row_prepared=1,.cuda_pair_eligible=1};
}
static void head_scheduler_check(int row,int alias,int mode) {
    VxIncrementalRun run=head_scheduler_setup(row,alias);
    if(mode==0)g_portable_selected_nodes[2]=0;
    if(mode==1)g_incremental_scratch.hybrid_nodes[2]=0;
    if(mode==2)g_incremental_scratch.hybrid_nodes[0]=0;
    if(mode==3)run.row_prepared=0;
    if(mode==4)run.direct_cpu=1;
    if(mode==5)run.cuda_pair_eligible=0;
    if(mode==6)run.stepwise=1;
    if(mode>=7 && mode<=9) {
        g_n[1].skip=0;
        g_portable_selected_nodes[1]=mode==8;
        g_incremental_scratch.hybrid_nodes[1]=mode==9;
    }
    if(mode==10)head_result=0;
    if(mode==11)head_result=-1;
    if(mode==12){run.force_full=1;g_portable_selected_nodes[2]=0;}
    if(mode==13){run.force_full=1;g_incremental_scratch.hybrid_nodes[2]=0;}
    const int accepted=mode==-1 || mode==12,terminal=mode==11;
    float before[6][12];memcpy(before,values[0],sizeof(before));
    T descriptors[6];Node nodes[3];memcpy(descriptors,g_t,sizeof(descriptors));memcpy(nodes,g_n,sizeof(nodes));
    unsigned char selection[3],hybrid[3];memcpy(selection,g_portable_selected_nodes,3);memcpy(hybrid,g_incremental_scratch.hybrid_nodes,3);
    assert(vx_incremental_run_node_locked(&run,0)==(terminal?-1:0));
    assert(head_calls==(accepted || mode==10 || terminal));
    assert(calls==(!accepted && !terminal));
    assert(run.cursor==(terminal?0:accepted?3:1) && run.executed==run.cursor && !run.skipped);
    assert(!g_incremental_plan.row_probe_tensors);
    assert(!memcmp(descriptors,g_t,sizeof(descriptors))&&!memcmp(nodes,g_n,sizeof(nodes)));
    assert(!memcmp(selection,g_portable_selected_nodes,3)&&!memcmp(hybrid,g_incremental_scratch.hybrid_nodes,3));
    for(int tensor=0;tensor<6;tensor++) {
        assert(g_dirty[tensor]==(tensor==0 || (!terminal && tensor==2) || (accepted && (tensor==3 || tensor==4))));
        for(int k=0;k<12;k++) {
            float value=before[tensor][k];
            /* The existing ordinary run_node mock writes its full tensor;
             * the admitted triple mock writes only the selected row. */
            if(!terminal && (!accepted || (k>=row*3 && k<(row+1)*3))) {
                float norm=before[0][k]+before[5][k%3];
                if(tensor==2 || (accepted && !alias && tensor==3))value=norm;
                if(accepted && tensor==4)value=norm*2.f;
            }
            assert(!memcmp(&value,&values[0][tensor][k],sizeof(value)));
        }
    }
    for(int i=0;i<3;i++)assert(accepted?head_routes[i] && !strcmp(head_routes[i],"cuda-graph-row+head-rms-rope"):!head_routes[i]);
}
static void head_scheduler_checks(void) {
    for(int row=0;row<4;row++)head_scheduler_check(row,1,-1);
    head_scheduler_check(2,0,-1); /* Existing disjoint three-owner route. */
    for(int mode=0;mode<14;mode++)head_scheduler_check(1,1,mode);
    head_active=0;
    puts("PASS head scheduler:19 skipped-view/disjoint, selection/closure, cursor/dirty/routes, fallback and terminal states");
}
#endif
"""


class PreparedBindingsTests(unittest.TestCase):
    def test_real_private_state_and_scheduler(self):
        cc = shutil.which("cc")
        if not cc:
            self.skipTest("C compiler unavailable")

        # Keep the exported base fixture unchanged for other scheduler tests.
        # This unit adds ownership-aware provenance and refresh guards while
        # retaining all original CPU/device and F64 numerical cases.
        fixture = FIXTURE.replace(
            "t->data==values[vx_engine_state_current()==&contexts[1]][5];",
            "t->data==values[vx_engine_state_current()==&contexts[1]][5] && "
            "(!vx_engine_state_current()->compiled_weight_conversions || !t->owns);")
        # Keep the shared fixture's ordinary Head fallback unchanged for its
        # importers; this method alone exercises the real triple scheduler.
        fixture = fixture.replace(
            "int vx_runtime_cuda_head_rms_rope_triple_locked(int node,int row) {\n"
            "    (void)node;(void)row;return 0;\n}", HEAD_SCHEDULER_MOCK)
        fixture = fixture.replace("int main(void) {", GEOMETRY_REFRESH_CHECKS + BINDING_SCAN_CHECKS + HEAD_SCHEDULER_CHECKS + "\nint main(void) {")
        fixture = fixture.replace(
            "    for(int i=0;i<2;i++)vx_engine_state_release_graph_metadata(&contexts[i]);",
            "#if VOLVOXAI_ENABLE_CUDA\n    binding_scan_predicate_checks();\n    geometry_refresh_checks();\n    head_scheduler_checks();\n#endif\n"
            "    for(int i=0;i<2;i++)vx_engine_state_release_graph_metadata(&contexts[i]);")

        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-prepared-bindings-") as temp:
            unit = Path(temp) / "test.c"
            unit.write_text(fixture)
            for profile, cuda in ((0, 0), (1, 0), (0, 1), (1, 1)):
                binary = Path(temp) / f"test-{profile}-{cuda}"
                command = [cc, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                           "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                           f"-DVOLVOXAI_ENABLE_TRAINING={profile}", f"-DVOLVOXAI_ENABLE_CUDA={cuda}",
                           "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_VULKAN=0",
                           "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
                command += [f"-I{ROOT / path}" for path in includes]
                command += shlex.split(os.environ.get("VOLVOXAI_TEST_CFLAGS", ""))
                command += [str(unit), "-lm", "-o", str(binary)]
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([str(binary)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS prepared bindings", result.stdout)
                if cuda:
                    self.assertIn("PASS binding scan:699", result.stdout)
                    self.assertIn("PASS geometry refresh", result.stdout)
                    self.assertIn("PASS head scheduler:19", result.stdout)


class IncrementalHistoryDomainTests(unittest.TestCase):
    def test_hand_authored_modern_history(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = ROOT
        runtime = RUNTIME
        source = r'''#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include "engine_internal.h"
#include "volvoxai_types.h"
typedef struct { const char* name; } TestDeclaredTensor;
struct VxModel { cJSON* logical_graph; TestDeclaredTensor* inputs; size_t input_count; };
struct VxCompiledModel { VxModel* model; };
struct VxExecutionContext { VxCompiledModel* compiled; VxEngineState* engine_state; };
static VxEngineState test_state;
static VxEngineState* current = &test_state;
static T tensors[4];
static int scope_depth;
VxEngineState* vx_engine_state_current(void) { return current; }
VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* state) {
    VxEngineStateScope result = {.previous=current, .bound=state};
    ++scope_depth; current=state; return result;
}
void vx_engine_state_scope_leave(VxEngineStateScope scope) {
    assert(scope_depth>0 && current==scope.bound); --scope_depth; current=scope.previous;
}
T* t_find(const char* name) {
    for (unsigned i=0; i<4; ++i) if (!strcmp(tensors[i].name,name)) return &tensors[i];
    return NULL;
}
'''
        contract = (runtime / "public_api_execution_contract.inc").read_text()
        for name in ("vx_execution_document", "vx_execution_string", "vx_execution_names_contains"):
            source += function_source(contract, name)
        source += function_source(contract, "vx_incremental_control_kind")
        guards = (runtime / "public_api_decode_feedback.inc").read_text()
        for name in ("vx_incremental_modern_history", "vx_incremental_previous_rows_immutable",
                     "vx_incremental_controls_validate"):
            source += function_source(guards, name)
        source += r'''
static VxTensorBinding binding(T* tensor,const void* data) {
    VxTensorBinding result=VX_TENSOR_BINDING_INIT;
    result.name=tensor->name; result.dtype=VX_DTYPE_I32; result.rank=(uint32_t)tensor->ndim;
    for(int d=0;d<tensor->ndim;d++) result.shape[d]=tensor->shape[d];
    result.data=data; result.byte_size=(size_t)tensor->numel*4u; result.location=VX_MEMORY_HOST;
    return result;
}
int main(void) {
    TestDeclaredTensor declared[]={{"tokens"},{"positions"},{"mask"},{"mrope"}};
    VxModel model={.logical_graph=cJSON_Parse("{\"execution\":{\"incremental\":{\"active_lengths_input\":\"active\",\"position_input\":\"positions\",\"sequence_axis\":1,\"update_inputs\":[\"tokens\",\"positions\",\"mask\",\"mrope\"]}}}"),
        .inputs=declared,.input_count=4}; assert(model.logical_graph);
    VxCompiledModel compiled={.model=&model};
    VxExecutionContext context={.compiled=&compiled,.engine_state=&test_state};
    int32_t token_before[4]={11,12,0,0}, token_after[4]={11,12,13,0};
    int32_t position_before[4]={0,1,2,3}, position_after[4]={0,1,2,3};
    int32_t mask_before[16]={1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0};
    int32_t mask_after[16]; memcpy(mask_after,mask_before,sizeof(mask_after));
    int32_t mrope_before[12]={0,1,0,0,0,1,0,0,0,1,0,0};
    int32_t mrope_after[12]; memcpy(mrope_after,mrope_before,sizeof(mrope_after));
    strcpy(tensors[0].name,"tokens");tensors[0].data=(float*)token_before;
    tensors[0].dtype=T_I32;tensors[0].elem_size=4;tensors[0].ndim=2;
    tensors[0].shape[0]=1;tensors[0].shape[1]=4;tensors[0].numel=4;
    tensors[1]=tensors[0];strcpy(tensors[1].name,"positions");tensors[1].data=(float*)position_before;
    tensors[1].ndim=1;tensors[1].shape[0]=4;
    tensors[2]=tensors[0];strcpy(tensors[2].name,"mask");tensors[2].data=(float*)mask_before;
    tensors[2].ndim=3;tensors[2].shape[1]=tensors[2].shape[2]=4;tensors[2].numel=16;
    tensors[3]=tensors[0];strcpy(tensors[3].name,"mrope");tensors[3].data=(float*)mrope_before;
    tensors[3].shape[0]=3;tensors[3].numel=12;
    Ref mask_ref={.port=VX_PORT_MASK};strcpy(mask_ref.name,"mask");
    Node nodes[3]={{.operator_kind=VX_OP_GATED_DELTA_NET},
        {.operator_kind=VX_OP_KV_COMPRESS,.ins=&mask_ref,.nin=1}, {0}};
    test_state.nodes=nodes;test_state.node_count=2;
    VxTensorBinding inputs[4]={binding(&tensors[0],token_after),binding(&tensors[1],position_after),
        binding(&tensors[2],mask_after),binding(&tensors[3],mrope_after)};
    assert(vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    token_after[0]=99;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));token_after[0]=11;
    position_after[0]=4;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));position_after[0]=0;
    mask_after[4]=0;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));mask_after[4]=1;
    mask_after[8]=1;mask_after[9]=1;mask_after[10]=1;mask_after[2]=1;
    assert(vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    mrope_after[4]=9;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));mrope_after[4]=0;
    mrope_after[2]=mrope_after[6]=mrope_after[10]=2;
    assert(vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    inputs[0].location=VX_MEMORY_DEVICE;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    inputs[0].location=VX_MEMORY_HOST;
    /* Query-shaped masks have the same semantics without a batch axis. */
    tensors[2].ndim=2;tensors[2].shape[0]=tensors[2].shape[1]=4;
    inputs[2]=binding(&tensors[2],mask_after);
    assert(vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    mask_after[5]=0;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));mask_after[5]=1;
    /* S=3 must be treated as a query mask, not as three MRope planes. */
    int32_t mask3_before[9]={1,0,0,1,1,0,0,0,0};
    int32_t mask3_after[9]={1,0,1,1,1,1,1,1,1};
    tensors[2].shape[0]=tensors[2].shape[1]=3;tensors[2].numel=9;
    tensors[2].data=(float*)mask3_before;inputs[2]=binding(&tensors[2],mask3_after);
    assert(vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    mask3_after[3]=0;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));mask3_after[3]=1;
    assert(vx_incremental_controls_validate(&context,inputs,4,3,3)==VX_STATUS_OK);
    position_after[1]=5;assert(vx_incremental_controls_validate(&context,inputs,4,3,3)==VX_STATUS_INVALID_ARGUMENT);
    position_after[1]=1;
    inputs[1].location=VX_MEMORY_DEVICE;assert(vx_incremental_controls_validate(&context,inputs,4,3,3)==VX_STATUS_INVALID_ARGUMENT);
    inputs[1].location=VX_MEMORY_HOST;
    /* Graph-only callers cannot omit required logical positions or rely on
     * stale future positions becoming valid after mutation. */
    cJSON* incremental=cJSON_GetObjectItemCaseSensitive(vx_execution_document(&model),"incremental");
    cJSON_DeleteItemFromObjectCaseSensitive(incremental,"position_input");
    assert(vx_incremental_controls_validate(&context,inputs,4,3,3)==VX_STATUS_INVALID_ARGUMENT);
    cJSON_AddStringToObject(incremental,"position_input","positions");
    assert(vx_incremental_controls_validate(&context,NULL,0,3,3)==VX_STATUS_OK);
    position_before[2]=7;assert(vx_incremental_controls_validate(&context,NULL,0,3,3)==VX_STATUS_INVALID_ARGUMENT);
    position_before[2]=2;
    /* A caller can directly update a causal-convolution activation stream.
     * Only actual Conv2D INPUT topology selects axis2 in [1,1,S,C]. */
    float conv_before[8]={1,2,3,4,0,0,0,0},conv_after[8]={1,2,3,4,5,6,0,0};
    Ref conv_ref={.port=VX_PORT_INPUT};strcpy(conv_ref.name,"tokens");
    nodes[2]=(Node){.operator_kind=VX_OP_CONV_2D,.ins=&conv_ref,.nin=1};
    test_state.node_count=3;
    tensors[0].ndim=4;tensors[0].shape[0]=tensors[0].shape[1]=1;
    tensors[0].shape[2]=4;tensors[0].shape[3]=2;tensors[0].numel=8;
    tensors[0].dtype=T_F32;tensors[0].data=conv_before;
    inputs[0]=binding(&tensors[0],conv_after);inputs[0].dtype=VX_DTYPE_F32;
    assert(vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    conv_after[3]=9;assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));conv_after[3]=4;
    nodes[2].operator_kind=VX_OP_CROSS_SDPA;
    assert(!vx_incremental_previous_rows_immutable(&context,inputs,4,2));
    nodes[2].operator_kind=VX_OP_CONV_2D;
    test_state.node_count=1;
    const VxOperatorKind modern_ops[]={VX_OP_SIGNED_SQRT,VX_OP_MOE_ROUTER,VX_OP_MOE_LINEAR};
    for(unsigned i=0;i<3;i++) {
        nodes[0].operator_kind=modern_ops[i];assert(vx_incremental_modern_history(&context));
    }
    nodes[0].operator_kind=VX_OP_MATMUL;assert(!vx_incremental_modern_history(&context));
    assert(!scope_depth && current==&test_state);
    cJSON_Delete(model.logical_graph);
    puts("PASS production hand-authored modern immutable history and V4 logical positions");
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-incremental-history-") as directory:
            c_source, binary = Path(directory) / "history.c", Path(directory) / "history"
            c_source.write_text(source)
            built = subprocess.run([compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2",
                "-Wall", "-Wextra", "-Werror", "-DVOLVOXAI_ENABLE_TRAINING=0",
                *["-I" + str(root / path) for path in ("native/include", "native/third_party", "native/src",
                    "native/src/runtime", "native/src/kernels", "native/src/backends")],
                str(c_source), str(root / "native/third_party/cJSON.c"), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS production hand-authored modern", checked.stdout)


PREFLIGHT_PRELUDE = r'''
#include "public_api_internal.h"
/* Only context fields consumed by the extracted production preflight. */
typedef struct {const char* name;} TestInput;
typedef struct {size_t input_count;TestInput* inputs;} TestModel;
typedef struct {void* provider;VxBackendKind builtin_backend;TestModel* model;} TestCompiled;
struct VxExecutionContext {
 TestCompiled* compiled;VxDecodeRowMode decode_row_mode;VxEngineState* engine_state;
 uint32_t decode_slots;unsigned char* decode_default_inputs;
 int prepared_engine_forbidden;void* debug_session;char adapter_route_version[96];
};
VxKernelThreadPoolScope vx_kernel_thread_pool_scope_enter(VxKernelThreadPool* pool) {(void)pool;return (VxKernelThreadPoolScope){0};}
void vx_kernel_thread_pool_scope_leave(VxKernelThreadPoolScope previous) {(void)previous;}
static int fail_dirty_allocation;
static void* preflight_malloc(size_t bytes) {return fail_dirty_allocation?NULL:malloc(bytes);}
static int lock_depth, row_fail, probe_enabled, cached_probes, uncached_probes, policy_trace;
static int oracle_mode,preflight_comparisons;
static int bad_control, bad_page, fail_after_node=-1;
static unsigned probe_nodes;
static int probe_position;
int volvoxai_engine_model_route_lease_active(void) {return 0;}
void volvoxai_engine_model_lock(void) {assert(!lock_depth);lock_depth++;}
void volvoxai_engine_model_unlock(void) {assert(lock_depth==1);lock_depth--;}
static int vx_native_gpu_backend(VxBackendKind kind) {return kind==VX_BACKEND_KIND_CUDA;}
VxDecodeRowSetStatus vx_paged_row_set_init_locked(VxDecodeRowSet* set,int slots,const int32_t* positions) {
 if(row_fail)return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
 return vx_decode_row_set_init(set,slots,positions,NULL);
}
static int control_compatible(Node* node,int index,int row) {
 (void)node;proof_calls++;if(!probe_enabled)return !refuse;
 assert(row==probe_position && g_decode_rows.slots==1 && g_decode_rows.positions[0]==row);
 if(!oracle_mode && !probe_nodes){
  assert(!vx_engine_state_current()->decode_cuda_mask_valid);
  assert(!vx_engine_state_current()->decode_cuda_positions_valid);
#if VOLVOXAI_ENABLE_CUDA
  VxDecodeCudaSpanCache zero[2]={0};assert(!memcmp(zero,vx_engine_state_current()->decode_cuda_spans,sizeof(zero)));
#endif
 }
 if(!oracle_mode)probe_nodes|=1u<<index;
#if VOLVOXAI_ENABLE_CUDA
 if(!oracle_mode){
  if(g_incremental_plan.cuda_commands_active)cached_probes++;else uncached_probes++;
  memset(vx_engine_state_current()->decode_cuda_spans,0x5a,sizeof(vx_engine_state_current()->decode_cuda_spans));
 }
#endif
 if(!oracle_mode){
  vx_engine_state_current()->decode_cuda_mask_valid=1;
  vx_engine_state_current()->decode_cuda_positions_valid=1;
 }
 return !refuse && !bad_control && !bad_page && row>=0 && row<4 && index!=fail_after_node;
}
'''

PREFLIGHT_CHECKS = r'''
static void proof_poison(void) {
 vx_engine_state_current()->decode_cuda_mask_valid=1;
 vx_engine_state_current()->decode_cuda_positions_valid=1;
#if VOLVOXAI_ENABLE_CUDA
 memset(vx_engine_state_current()->decode_cuda_spans,0xa5,sizeof(vx_engine_state_current()->decode_cuda_spans));
#endif
}
static void proofs_clear(void) {
 assert(!vx_engine_state_current()->decode_cuda_mask_valid && !vx_engine_state_current()->decode_cuda_positions_valid);
#if VOLVOXAI_ENABLE_CUDA
 VxDecodeCudaSpanCache zero[2]={0};assert(!memcmp(zero,vx_engine_state_current()->decode_cuda_spans,sizeof(zero)));
#endif
}
static void preflight_cases(void) {
 g_current_engine_state=&contexts[0];
 /* Restore a simple unaliased graph after the baseline numerical cases. */
 g_n[0].operator_kind=VX_OP_ADD;g_n[0].nin=2;strcpy(g_n[0].ins[0].name,"x");strcpy(g_n[0].ins[1].name,"weight");
 g_n[1].operator_kind=VX_OP_ADD;g_n[1].nin=1;strcpy(g_n[1].ins[0].name,"y");
 strcpy(g_n[2].out,"out");strcpy(g_n[2].outs[0].name,"out");
 contexts[0].model_generation++;assert(!incremental_plan_build_locked());
 graph_mode=1;g_cache_valid=1;g_weight_caches_dirty=0;g_decode_slots=7;g_hybrid_prepared_row=19;
#if VOLVOXAI_ENABLE_CUDA
 g_use_cuda=1;
#endif
 TestInput names[]={{"x"},{"y"}};TestModel model={2,names};TestCompiled compiled={NULL,VX_BACKEND_KIND_CUDA,&model};
 VxExecutionContext context={.compiled=&compiled,.decode_row_mode=VX_DECODE_ROW_REQUIRED,.engine_state=&contexts[0],.decode_slots=1};
 contexts[0].decode_row_stage_reserved=1;
 int32_t position=1;VxTensorBinding input={.name="x"};
 unsigned char prior[6]={0,1,0,1,0,1};memcpy(g_dirty,prior,6);
 VxDecodeRowSet outer={0};int32_t outerposition=2;assert(!vx_decode_row_set_init(&outer,1,&outerposition,NULL));g_decode_rows=outer;
 probe_enabled=1;probe_position=1;
 for(int policy=0;policy<12;policy++) {
  probe_nodes=0;cached_probes=uncached_probes=0;proof_poison();
  context.prepared_engine_forbidden=policy==1;context.debug_session=policy==2?(void*)1:NULL;
  context.adapter_route_version[0]=policy==3?'a':0;g_debug=policy==4;
  contexts[0].preserve_node_boundaries=policy==5;policy_trace=policy==6;
#if VOLVOXAI_ENABLE_TRAINING
  g_native_training_mode=policy==7;
#endif
  graph_mode=policy!=8;
#if VOLVOXAI_ENABLE_CUDA
  g_use_cuda=policy!=9;g_incremental_plan.cuda_commands_active=policy==11;
#endif
  int stepwise=policy==10;
  assert(vx_context_preflight_decode_rows(&context,&position,&input,1,NULL,stepwise)==VX_STATUS_OK);
  assert(probe_nodes==5 && !memcmp(g_dirty,prior,6));
  assert(g_decode_rows.positions==outer.positions && g_decode_slots==7 && g_hybrid_prepared_row==19 && !lock_depth);
  proofs_clear();
#if VOLVOXAI_ENABLE_CUDA
  int eligible=policy==0 || policy==11;
#if !VOLVOXAI_ENABLE_TRAINING
  eligible|=policy==7;
#endif
  assert(cached_probes==(eligible?2:0) && uncached_probes==(eligible?0:2));
  assert(g_incremental_plan.cuda_commands_active==(policy==11));
#endif
 }
 context.prepared_engine_forbidden=0;context.debug_session=NULL;context.adapter_route_version[0]=0;
 g_debug=0;contexts[0].preserve_node_boundaries=0;policy_trace=0;graph_mode=1;
#if VOLVOXAI_ENABLE_TRAINING
 g_native_training_mode=0;
#endif
#if VOLVOXAI_ENABLE_CUDA
 g_use_cuda=1;g_incremental_plan.cuda_commands_active=0;
 /* Every descriptor dimension changes storage/geometry admission, while
  * value-only changes retain the prepared route and remain numerical inputs. */
 T original=g_t[0];
 for(int mutation=0;mutation<8;mutation++) {
  g_t[0]=original;
  if(mutation==0)g_t[0].data=values[0][1];if(mutation==1)g_t[0].numel++;
  if(mutation==2)g_t[0].elem_size=2;if(mutation==3)g_t[0].dtype=T_I32;
  if(mutation==4)g_t[0].ndim=2;if(mutation==5)g_t[0].shape[1]++;
  if(mutation==6)g_t[0].is_graph_input=0;if(mutation==7)g_t[0].shape[7]++;
  cached_probes=uncached_probes=0;probe_nodes=0;proof_poison();
  assert(vx_context_preflight_decode_rows(&context,&position,&input,1,NULL,0)==VX_STATUS_OK);
  assert(!cached_probes && uncached_probes==2 && !g_incremental_plan.cuda_commands_active);proofs_clear();
 }
 g_t[0]=original;
 /* A changed value in the same public input allocation keeps descriptor reuse;
  * admission is read-only and dispatch later consumes the current bytes. */
 values[0][0][0]+=.25f;float changed_input=values[0][0][0];
 cached_probes=uncached_probes=0;probe_nodes=0;proof_poison();
 assert(vx_context_preflight_decode_rows(&context,&position,&input,1,NULL,0)==VX_STATUS_OK);
 assert(cached_probes==2 && !uncached_probes && values[0][0][0]==changed_input);
 assert(!memcmp(g_dirty,prior,6));proofs_clear();
#endif
 /* Dirty closure/FIXED selection is rebuilt every probe, never borrowed. */
 for(int roots=0;roots<4;roots++)for(int selected=-1;selected<8;selected++) {
  VxTensorBinding roots_input[2]={{.name="x"},{.name="y"}};size_t count=0;
  if(roots&1)roots_input[count++]=(VxTensorBinding){.name="x"};
  if(roots&2)roots_input[count++]=(VxTensorBinding){.name="y"};
  g_portable_selection_active=selected>=0;
  for(int i=0;i<3;i++)g_portable_selected_nodes[i]=selected>=0?(selected>>i)&1:0;
  probe_nodes=0;proof_poison();
  VxStatus expected_status=(selected>=0 || roots)?VX_STATUS_OK:VX_STATUS_BACKEND_UNSUPPORTED;
  assert(vx_context_preflight_decode_rows(&context,&position,roots_input,count,NULL,0)==expected_status);
  unsigned wanted=selected>=0?(unsigned)selected:roots==1?5u:roots==2?6u:roots==3?7u:0u;
  assert(probe_nodes==wanted && !memcmp(g_dirty,prior,6));proofs_clear();
 }
 g_portable_selection_active=0;
 /* Controls/page contents mutate without pointer changes: they are still
  * checked on each probe and a refusal leaves the caller's state unchanged. */
 for(int failure=0;failure<5;failure++) {
  row_fail=failure==0;refuse=failure==1;bad_control=failure==2;bad_page=failure==3;fail_after_node=failure==4?2:-1;
  float before[6][12];memcpy(before,values[0],sizeof(before));proof_poison();probe_nodes=0;
  assert(vx_context_preflight_decode_rows(&context,&position,&input,1,NULL,0)==VX_STATUS_BACKEND_UNSUPPORTED);
  assert(!memcmp(g_dirty,prior,6) && !memcmp(before,values[0],sizeof(before)) && g_decode_rows.positions==outer.positions && !lock_depth);
  proofs_clear();
 }
 row_fail=refuse=bad_control=bad_page=0;fail_after_node=-1;
 /* Failure before the temporary scope never changes the owning row state or
  * an activation already held by another private no-command caller. */
 fail_dirty_allocation=1;probe_nodes=0;
#if VOLVOXAI_ENABLE_CUDA
 g_incremental_plan.cuda_commands_active=1;
#endif
 assert(vx_context_preflight_decode_rows(&context,&position,&input,1,NULL,0)==VX_STATUS_OUT_OF_MEMORY);
 assert(!probe_nodes && !memcmp(g_dirty,prior,6) && g_decode_rows.positions==outer.positions && !lock_depth);
#if VOLVOXAI_ENABLE_CUDA
 assert(g_incremental_plan.cuda_commands_active==1);g_incremental_plan.cuda_commands_active=0;
#endif
 fail_dirty_allocation=0;
 /* Public omitted bindings use only the configured default changed roots. */
 unsigned char defaults[2]={1,0};context.decode_default_inputs=defaults;probe_nodes=0;
 assert(vx_context_preflight_decode_rows(&context,&position,NULL,0,&g_t[4],0)==VX_STATUS_OK && probe_nodes==5);
 context.decode_default_inputs=NULL;

 /* A required output outside the changed closure cannot be admitted. */
 probe_nodes=0;proof_poison();assert(vx_context_preflight_decode_rows(&context,&position,&input,1,&g_t[3],0)==VX_STATUS_BACKEND_UNSUPPORTED);
 assert(!probe_nodes && !memcmp(g_dirty,prior,6));proofs_clear();
 /* Cold valid-generation miss executes uncached planning. Failed rebuilding
  * cannot accidentally reactivate the saved command state. */
#if VOLVOXAI_ENABLE_CUDA
 for(int failed=0;failed<2;failed++) {
  size_t cap=g_incremental_scratch.tensor_capacity;g_incremental_plan.valid=0;g_incremental_plan.cuda_commands_active=1;
  if(failed)g_incremental_scratch.tensor_capacity=0;
  cached_probes=uncached_probes=0;probe_nodes=0;proof_poison();
  VxStatus status=vx_context_preflight_decode_rows(&context,&position,&input,1,NULL,0);
  assert(status==(failed?VX_STATUS_EXECUTION_FAILED:VX_STATUS_OK));
  assert(!cached_probes && uncached_probes==(failed?0:2) && !g_incremental_plan.cuda_commands_active);
  assert(!memcmp(g_dirty,prior,6) && !lock_depth);proofs_clear();
  g_incremental_scratch.tensor_capacity=cap;assert(!incremental_plan_build_locked());
 }
 /* Scope restoration refuses a model revision changed during the probe. */
 g_incremental_plan.cuda_commands_active=1;VxIncrementalRowProbeScope scope;
 vx_incremental_row_probe_begin_locked(&scope,0,1);assert(g_incremental_plan.cuda_commands_active);
 contexts[0].model_generation++;assert(!incremental_plan_build_locked());
 vx_incremental_row_probe_end_locked(&scope);assert(!g_incremental_plan.cuda_commands_active);proofs_clear();
#endif
 g_decode_rows=(VxDecodeRowSet){0};vx_decode_row_set_dispose(&outer);g_decode_slots=0;
 probe_enabled=0;graph_mode=0;g_portable_selection_active=0;
 /* The real scheduler's independent numerical branch/alias/FIXED oracle still
  * produces the same results after all temporary admission scopes. */
 dependency_numerics();
}
'''


class WarmPreflightTests(unittest.TestCase):
    def test_real_state_precommit_transaction_and_numerics(self):
        cc = shutil.which('cc')
        if not cc:
            self.skipTest('C compiler unavailable')
        f = FIXTURE
        for name in ('runtime_state.c', 'incremental_runtime.c'):
            f = f.replace(f'#include "../src/runtime/{name}"', f'#include "{RUNTIME}/{name}"')
        f = f.replace('static int lookups, classifications, calls, graph_mode, proof_calls, refuse;',
                      'static int lookups, classifications, calls, graph_mode, proof_calls, refuse;\nstatic int control_compatible(Node*,int,int);\nstatic int policy_trace;')
        f = f.replace('int vx_trace_nodes(const VxTraceScope* trace) {(void)trace;return 0;}',
                      'int vx_trace_nodes(const VxTraceScope* trace) {(void)trace;return policy_trace;}')
        old = function_source(f, 'vx_runtime_node_incremental_row_compatible')
        f = f.replace(old, 'int vx_runtime_node_incremental_row_compatible(Node* node,int index,int row) {return control_compatible(node,index,row);}\n')
        f = f[:f.index('int main(void)')]
        prelude = PREFLIGHT_PRELUDE.replace(', policy_trace;', ';')
        prod = function_source((RUNTIME / 'public_api_execute_result.inc').read_text(), 'vx_context_preflight_decode_rows').replace('malloc((size_t)g_nt)', 'preflight_malloc((size_t)g_nt)')
        reference = (ROOT / 'tools/tests/fixtures/incremental_preflight_reference.inc').read_text().replace('malloc((size_t)g_nt)', 'preflight_malloc((size_t)g_nt)')
        checked = r'''
static VxStatus vx_context_preflight_decode_rows_checked(VxExecutionContext* context,
 const int32_t* positions,const VxTensorBinding* inputs,size_t count,const T* required,int stepwise) {
 VxStatus actual=vx_context_preflight_decode_rows(context,positions,inputs,count,required,stepwise);
 oracle_mode=1;
 VxStatus expected_status=vx_context_preflight_decode_rows_reference(context,positions,inputs,count,required);
 oracle_mode=0;assert(actual==expected_status);preflight_comparisons++;return actual;
}
'''
        translation = f + prelude + prod + reference + checked + PREFLIGHT_CHECKS.replace('vx_context_preflight_decode_rows(', 'vx_context_preflight_decode_rows_checked(') + r'''
int main(void) {
 bindings();dependency_numerics();command_domain_guards();closure_proof();pair_selection_and_failure();preflight_cases();
 for(int i=0;i<2;i++)vx_engine_state_release_graph_metadata(&contexts[i]);
 printf("reference-vs-production preflight comparisons=%d\n",preflight_comparisons);
 puts("PASS warm preflight: unchanged real closure/numerics, exact bindings, explicit debug policy, failure atomicity");
}
'''
        includes = ['native/tests', 'native/include', 'native/src', 'native/src/runtime', 'native/src/kernels',
                    'native/src/backends', 'native/third_party', 'native/third_party/synurang/include', 'runtime/generated/c']
        with tempfile.TemporaryDirectory(prefix='volvox-warm-preflight-') as temp:
            unit = Path(temp) / 'test.c'
            unit.write_text(translation)
            for training, cuda in ((0, 0), (1, 0), (0, 1), (1, 1)):
                binary = Path(temp) / f'test-{training}-{cuda}'
                cmd = [cc, '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O2', '-Wall', '-Wextra', '-Werror',
                       '-Wno-misleading-indentation', '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections',
                       f'-DVOLVOXAI_ENABLE_TRAINING={training}', f'-DVOLVOXAI_ENABLE_CUDA={cuda}',
                       '-DVOLVOXAI_ENABLE_WEBGPU=0', '-DVOLVOXAI_ENABLE_VULKAN=0', '-DVOLVOXAI_ENABLE_OPENGL=0',
                       '-DVOLVOXAI_ENABLE_METAL=0', f'-I{RUNTIME}'] + [f'-I{ROOT / p}' for p in includes] + \
                      [str(unit), str(ROOT / 'native/src/runtime/decode_row_set.c'), str(ROOT / 'native/src/runtime/paged_kv.c'), '-lm', '-o', str(binary)]
                if os.getenv('VOLVOXAI_TEST_SANITIZE'):
                    cmd[1:1] = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
                r = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                r = subprocess.run([str(binary)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn('PASS warm preflight', r.stdout)


COMMANDS_MOCKS = r'''
#include <math.h>
static int prepared_calls, route_notes;
static int tensor_same_shape(const T* a,const T* b) {
    return a && b && a->ndim==b->ndim && !memcmp(a->shape,b->shape,(size_t)a->ndim*sizeof(int));
}
static int producer_index_any(const char* name) {(void)name;return -1;}
static const float* readable(const float* host,size_t count) {
    CudaTensorView view;assert(graph_ensure_input_view(host,count*4,0,&view));drain();
    return (const float*)view.device;
}
static float* writable(float* host,size_t count,CudaTensorView* view) {
    assert(graph_output_view(host,count*4,view));drain();return (float*)view->device;
}
static int cuda_graph_matmul_layout_f32(const float* input,const float* weight,const float* bias,
    float* output,int rows,int din,int dout,int out_in) {
    prepared_calls++;if(fail_numerical)return 0;
    const float* x=readable(input,(size_t)rows*din),*w=readable(weight,(size_t)din*dout);
    const float* b=bias?readable(bias,dout):NULL;CudaTensorView view;float* y=writable(output,(size_t)rows*dout,&view);
    for(int r=0;r<rows;r++)for(int o=0;o<dout;o++) {
        double sum=b?b[o]:0.;for(int k=0;k<din;k++)sum+=(double)x[r*din+k]*w[out_in?o*din+k:k*dout+o];
        y[r*dout+o]=(float)sum;
    }
    graph_mark_device(view.slot);return 1;
}
static int cuda_graph_rmsnorm_f32(const float* input,const float* weight,float* output,
    int rows,int width,float epsilon) {
    prepared_calls++;if(fail_numerical)return 0;
    const float* x=readable(input,(size_t)rows*width),*w=readable(weight,width);
    CudaTensorView view;float* y=writable(output,(size_t)rows*width,&view);
    for(int r=0;r<rows;r++) {
        double square=0.;for(int k=0;k<width;k++)square+=(double)x[r*width+k]*x[r*width+k];
        double scale=1./sqrt(square/width+epsilon);
        for(int k=0;k<width;k++)y[r*width+k]=(float)(x[r*width+k]*w[k]*scale);
    }
    graph_mark_device(view.slot);return 1;
}
static int cuda_graph_silu_f32(const float* input,float* output,long count) {
    prepared_calls++;if(fail_numerical)return 0;
    const float* x=readable(input,count);CudaTensorView view;float* y=writable(output,count,&view);
    for(long i=0;i<count;i++)y[i]=(float)((double)x[i]/(1.+exp(-(double)x[i])));
    graph_mark_device(view.slot);return 1;
}
static int cuda_graph_add_relu_f32(const float* a,const float* b,float* output,long count,int relu) {
    prepared_calls++;if(fail_numerical)return 0;
    const float* x=readable(a,count),*z=readable(b,count);CudaTensorView view;float* y=writable(output,count,&view);
    for(long i=0;i<count;i++){y[i]=x[i]+z[i];if(relu && y[i]<0)y[i]=0;if(relu==2 && y[i]>6)y[i]=6;}
    graph_mark_device(view.slot);return 1;
}
static int cuda_graph_binary_f32(const float* a,long ac,const float* b,long bc,float* output,long count,
    const uint32_t* os,const uint32_t* as,const uint32_t* bs,int rank,int op) {
    assert(ac==count && bc==count && rank==1 && op==0 && *os==1 && *as==1 && *bs==1);
    prepared_calls++;if(fail_numerical)return 0;
    const float* x=readable(a,count),*z=readable(b,count);CudaTensorView view;float* y=writable(output,count,&view);
    for(long i=0;i<count;i++)y[i]=x[i]*z[i];
    graph_mark_device(view.slot);return 1;
}
static int cuda_graph_embedding_f32(const int32_t* input,const float* weight,float* output,
    int count,int width,int vocabulary) {
    prepared_calls++;if(fail_numerical)return 0;
    const int32_t* ids=(const int32_t*)readable((const float*)input,count);
    const float* w=readable(weight,(size_t)vocabulary*width);CudaTensorView view;float* y=writable(output,(size_t)count*width,&view);
    for(int r=0;r<count;r++){assert(ids[r]>=0 && ids[r]<vocabulary);memcpy(y+r*width,w+ids[r]*width,width*4);}
    graph_mark_device(view.slot);return 1;
}
static int cuda_graph_argmax_f32(const float* input,int32_t* output,uint32_t outer,uint32_t axis,uint32_t inner) {
    assert(inner==1);prepared_calls++;if(fail_numerical)return 0;
    const float* x=readable(input,(size_t)outer*axis);CudaTensorView view;
    int32_t* y=(int32_t*)writable((float*)output,outer,&view);
    for(uint32_t r=0;r<outer;r++){uint32_t best=0;for(uint32_t k=1;k<axis;k++)if(x[r*axis+k]>x[r*axis+best])best=k;y[r]=(int32_t)best;}
    graph_mark_device(view.slot);return 1;
}
static void cuda_graph_note_node_route(int builtin) {assert(builtin);route_notes++;}
static int cuda_graph_packed_linear_decode_f32(const float* x,const int32_t* w,
    const float* s,const float* b,float* y,uint32_t k,uint32_t n,uint32_t wb,uint32_t ab) {
    (void)x;(void)w;(void)s;(void)b;(void)y;(void)k;(void)n;(void)wb;(void)ab;
    assert(0);return 0; /* These existing fixtures exercise non-packed routes. */
}
'''

COMMANDS_FIXTURE = r'''
static float host[4][36],device[4][36],stage_host[4][36],stage_device[4][36];
static int32_t metadata[64],metadata_device[64];
static Ref ins[3],outs[1];static GraphNodeFusion fusion[1];
static VxIncrementalTensorBinding bindings[4];static VxIncrementalCudaCommand commands[1];
static void setup(VxOperatorKind op,int layout) {
    memset(&owner,0,sizeof(owner));memset(tensors,0,sizeof(tensors));memset(nodes,0,sizeof(nodes));
    memset(ins,0,sizeof(ins));memset(outs,0,sizeof(outs));memset(commands,0,sizeof(commands));
    nslots=0;npending=0;prepared_calls=route_notes=0;
    owner.tensors=tensors;owner.tensor_count=4;owner.tensor_capacity=4;
    owner.nodes=nodes;owner.node_count=1;owner.node_capacity=1;owner.node_fusion=fusion;
    owner.model_generation=7;owner.use_cuda=1;owner.runtime_route_backend=routes;
    owner.incremental_plan=(VxIncrementalPlan){.valid=1,.generation=7,.tensor_count=4,.node_count=1,.cuda_commands_active=1};
    owner.incremental_scratch=(VxIncrementalScratch){.tensor_bindings=bindings,.tensor_capacity=4,.node_capacity=1,.cuda_commands=commands};
    owner.decode_row_stage_reserved=1;owner.decode_metadata_stage=(unsigned char*)metadata;owner.decode_metadata_stage_bytes=sizeof(metadata);
    owner.decode_slots=1;owner.active_row=owner.execution_row=3;
    for(int i=0;i<4;i++) {
        owner.decode_numeric_stage[i]=(unsigned char*)stage_host[i];owner.decode_numeric_stage_bytes[i]=sizeof(stage_host[i]);
        for(int k=0;k<36;k++){host[i][k]=-900.f;device[i][k]=(float)(k%9-4)*(i+1)*.25f;}
        register_span(i<3?(void*)host[i]:(void*)metadata,i<3?(void*)device[i]:(void*)metadata_device,i<3?sizeof(host[i]):sizeof(metadata),0);
    }
    for(int i=0;i<4;i++)register_span(stage_host[i],stage_device[i],sizeof(stage_host[i]),0);
    tensors[0]=tensor("input",host[0],3);tensors[1]=tensor("output",host[1],3);
    tensors[2]=(T){.dtype=T_F32,.elem_size=4,.ndim=2,.shape={3,3},.numel=9,.data=host[2]};strcpy(tensors[2].name,"weight");
    nodes[0]=(Node){.operator_kind=op,.nin=1,.nout=1,.ins=ins,.outs=outs,.primary_output_index=1};
    nodes[0].parsed_params.parsed=1;strcpy(nodes[0].out,"output");strcpy(outs[0].name,"output");outs[0].tensor_index=1;
    strcpy(ins[0].name,"input");ins[0].port=VX_PORT_INPUT;ins[0].tensor_index=0;
    if(op==VX_OP_LINEAR || op==VX_OP_MATMUL || op==VX_OP_GEMM || op==VX_OP_RMS_NORM || op==VX_OP_EMBEDDING) {
        nodes[0].nin=2;strcpy(ins[1].name,"weight");ins[1].port=VX_PORT_WEIGHT;ins[1].tensor_index=2;
    }
    if(op==VX_OP_LINEAR || op==VX_OP_MATMUL || op==VX_OP_GEMM) {
        nodes[0].parsed_params.cache[VX_NODE_PARAM_TRANS_B]=(VxCachedNodeParam){.kind=VX_NODE_PARAM_I32,.value.i32=layout};
    }
    if(op==VX_OP_RMS_NORM){tensors[2].ndim=1;tensors[2].shape[0]=3;tensors[2].shape[1]=0;tensors[2].numel=3;}
    if(op==VX_OP_ADD || op==VX_OP_MUL) {
        nodes[0].nin=2;ins[0].port=VX_PORT_A;strcpy(ins[1].name,"weight");ins[1].port=VX_PORT_B;ins[1].tensor_index=2;
        tensors[2]=tensor("weight",host[2],3);
    }
    if(op==VX_OP_EMBEDDING){tensors[0].dtype=T_I32;tensors[0].ndim=2;tensors[0].shape[1]=9;tensors[0].numel=9;
        for(int k=0;k<9;k++)((int32_t*)device[0])[k]=k%3;}
    if(op==VX_OP_ARG_MAX){tensors[1].dtype=T_I32;tensors[1].ndim=2;tensors[1].shape[1]=9;tensors[1].numel=9;}
    for(int i=0;i<4;i++){bindings[i].is_model_weight=i==2;bindings[i].producer=-1;}
    static int position=3;assert(vx_decode_row_set_init(&owner.decode_rows,1,&position,NULL)==VX_DECODE_ROW_SET_OK);
    owner.runtime_cuda_route_tracking=1;
}
int main(void) {
    const VxOperatorKind ops[]={VX_OP_LINEAR,VX_OP_MATMUL,VX_OP_GEMM,VX_OP_RMS_NORM,VX_OP_SILU,VX_OP_ADD,VX_OP_MUL,VX_OP_EMBEDDING,VX_OP_ARG_MAX};
    for(unsigned test=0;test<sizeof(ops)/sizeof(ops[0]);test++)for(int layout=0;layout<2;layout++) {
        setup(ops[test],layout);VxDecodeRowProjection p;
        assert(vx_decode_cuda_projection_describe(nodes,0,3,&p) && commands[0].valid && commands[0].numerical);
        float before[36];memcpy(before,device[1],sizeof(before));
        T saved_input=tensors[0],saved_output=tensors[1];
        unsigned copies_before=copies;
        assert(vx_runtime_cuda_prepared_row_node_locked(0,3,1)==1 && prepared_calls==1 && route_notes==1);
        drain();
        assert(!memcmp(&tensors[0],&saved_input,sizeof(T)) && !memcmp(&tensors[1],&saved_output,sizeof(T)));
        int out_width=ops[test]==VX_OP_ARG_MAX?1:3;
        if(ops[test]==VX_OP_ARG_MAX)assert(copies==(int)copies_before);
        for(int i=0;i<36;i++)if(i<3*out_width || i>=4*out_width)assert(device[1][i]==before[i]);
        for(int k=0;k<out_width;k++) {
            double expected=0.,x=device[0][9+k];
            if(test<3){for(int j=0;j<3;j++)expected+=(double)device[0][9+j]*device[2][layout?k*3+j:j*3+k];}
            else if(ops[test]==VX_OP_RMS_NORM){double square=0.;for(int j=0;j<3;j++)square+=(double)device[0][9+j]*device[0][9+j];expected=x*device[2][k]/sqrt(square/3.+1e-6);}
            else if(ops[test]==VX_OP_SILU)expected=x/(1.+exp(-x));
            else if(ops[test]==VX_OP_ADD)expected=x+device[2][9+k];
            else if(ops[test]==VX_OP_MUL)expected=x*device[2][9+k];
            else if(ops[test]==VX_OP_EMBEDDING)expected=device[2][((int32_t*)device[0])[3]*3+k];
            else {int best=0;for(int j=1;j<3;j++)if(device[0][9+j]>device[0][9+best])best=j;assert(((int32_t*)device[1])[3]==best);continue;}
            if(fabs(device[1][9+k]-expected)>=1e-5)fprintf(stderr,"op=%u layout=%d k=%d actual=%g expected=%g\n",test,layout,k,device[1][9+k],expected);
            assert(fabs(device[1][9+k]-expected)<1e-5);
        }
        /* The same private command rebinds a different logical row and never
         * retains the previous row-set pointer or device index. */
        owner.decode_rows.positions[0]=6;owner.decode_rows.kv_lengths[0]=7;owner.decode_rows.key_capacity=7;
        assert(vx_runtime_cuda_prepared_row_node_locked(0,6,1)==1 && prepared_calls==2);
        drain();
        assert(!memcmp(host[0],saved_input.data,sizeof(host[0])));
        fail_numerical=1;int calls=prepared_calls;
        assert(vx_runtime_cuda_prepared_row_node_locked(0,6,1)==-1 && prepared_calls==calls+1);fail_numerical=0;
        nodes[0].disabled=1;calls=prepared_calls;assert(!vx_runtime_cuda_prepared_row_node_locked(0,6,1) && prepared_calls==calls);
        vx_decode_row_set_dispose(&owner.decode_rows);
    }
    /* ArgMax views preserve first-tie semantics, prepare the entire input
     * owner, and retain every output row outside the selected I32 slice. */
    setup(VX_OP_ARG_MAX,0);
    for(int k=0;k<36;k++)host[0][k]=(float)k*.125f;
    host[0][9]=host[0][10]=host[0][11]=2.f;slots[0].host_dirty=1;slots[0].device_dirty=0;
    float retained[36];memcpy(retained,device[1],sizeof(retained));
    VxDecodeRowProjection tied;assert(vx_decode_cuda_projection_describe(nodes,0,3,&tied));
    unsigned prior_copies=copies,prior_uploads=uploads;
    assert(vx_runtime_cuda_prepared_row_node_locked(0,3,1)==1);drain();
    assert(((int32_t*)device[1])[3]==0 && copies==(int)prior_copies && uploads==(int)prior_uploads+1);
    for(int k=0;k<36;k++)if(k!=3)assert(device[1][k]==retained[k]);
    vx_decode_row_set_dispose(&owner.decode_rows);
    /* The ordinary ArgMax contract rejects selecting the last tied index.
     * Private row admission and prepared lowering must refuse before writing
     * device outputs, even when called outside public ShapeDomain validation. */
    for(int select_last=1;select_last<=2;select_last++) {
        setup(VX_OP_ARG_MAX,0);nodes[0].parsed_params.select_last_index=select_last;
        VxDecodeRowProjection unsupported;
        assert(vx_decode_cuda_projection_describe(nodes,0,3,&unsupported));
        assert(!commands[0].valid && !commands[0].numerical);
        float before[36];memcpy(before,device[1],sizeof(before));
        assert(!vx_decode_cuda_project_compatible(nodes,0,&unsupported));
        assert(!vx_runtime_cuda_prepared_row_node_locked(0,3,1) && !prepared_calls);
        drain();assert(!memcmp(before,device[1],sizeof(before)));
        vx_decode_row_set_dispose(&owner.decode_rows);
    }
    setup(VX_OP_SILU,0);VxDecodeRowProjection p;assert(vx_decode_cuda_projection_describe(nodes,0,3,&p));
    owner.incremental_plan.cuda_commands_active=0;assert(!vx_runtime_cuda_prepared_row_node_locked(0,3,1));
    owner.incremental_plan.cuda_commands_active=1;nodes[0].operator_kind=VX_OP_GELU;
    assert(!vx_runtime_cuda_prepared_row_node_locked(0,3,1));
    assert(vx_decode_cuda_projection_describe(nodes,0,3,&p) && !commands[0].valid);
    /* Unknown/fused/graph-mutable invariants remain an unsubmitted miss. */
    nodes[0].operator_kind=VX_OP_SILU;nodes[0].fuse_relu6=1;assert(!vx_decode_cuda_command_compile(nodes,0,&p,commands));
    nodes[0].fuse_relu6=0;nodes[0].nin=2;ins[1].port=VX_PORT_WEIGHT;strcpy(ins[1].name,"weight");ins[1].tensor_index=2;
    tensors[2].is_graph_input=1;bindings[2].is_model_weight=0;
    assert(!vx_decode_cuda_command_compile(nodes,0,&p,commands));
    vx_decode_row_set_dispose(&owner.decode_rows);
    puts("PASS production compiled row commands: numerical routes, immutable layouts, dynamic rows, preservation, unsupported fallback and terminal failure");
}
'''


class PreparedCommandsTests(unittest.TestCase):
    def test_production_row_commands(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        runtime = ROOT / "native/src/runtime"
        row = (runtime / "engine_runtime_row_cuda.inc").read_text()
        prelude = ROW_PREAMBLE
        prelude = prelude.replace('return !strcmp(name,"weight");',
                                  'T* tensor=t_find(name);return tensor && !tensor->is_graph_input && !strcmp(name,"weight");')
        for name in ("vx_decode_cuda_command_run", "cuda_graph_note_node_route", "vx_decode_cuda_project_compatible"):
            prelude = prelude.replace(function_source(prelude, name), "")
        translation = PREAMBLE + prelude + COMMANDS_MOCKS
        translation += packed_descriptor_validation_source()
        translation += function_source((runtime / "engine_runtime_dispatch.inc").read_text(), "linear_node_weight_layout")
        translation += function_source((runtime / "engine_runtime_f32_cpu.inc").read_text(), "normalization_dimensions")
        backend = (ROOT / "native/src/backends/cuda/host/cuda_graph_quantized_boundary_shape_host.inc").read_text()
        translation += function_source(backend, "cuda_graph_row_span_transfer")
        translation += function_source(backend, "cuda_graph_row_index_transfer")
        names = ("vx_decode_cuda_metadata_write", "vx_decode_cuda_projection_describe_uncached",
                 "vx_decode_cuda_command_operand", "vx_decode_cuda_command_operand_bind", "vx_decode_cuda_command_compile",
                 "vx_decode_cuda_row_geometry_current", "vx_decode_cuda_command_bind", "vx_decode_cuda_projection_describe", "vx_decode_cuda_command_run",
                 "vx_decode_cuda_host_control", "vx_decode_cuda_storage", "vx_decode_cuda_selection_geometry",
                 "vx_decode_cuda_span", "vx_decode_cuda_indices", "vx_decode_cuda_view_safe",
                 "vx_decode_cuda_contiguous_view", "vx_decode_cuda_transfer", "vx_decode_cuda_project_compatible",
                 "vx_decode_cuda_mask_key", "vx_decode_cuda_mask_key_equal", "vx_decode_cuda_project",
                 "vx_runtime_cuda_prepared_row_node_locked")
        for name in names:
            translation += function_source(row, name)
        translation += ROW_FIXTURE[:ROW_FIXTURE.index("int main(void)")] + COMMANDS_FIXTURE
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime", "native/src/kernels",
                    "native/src/backends", "native/third_party", "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-prepared-commands-") as temporary:
            unit, binary = Path(temporary) / "test.c", Path(temporary) / "test"
            unit.write_text(translation)
            command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-DVOLVOXAI_ENABLE_CUDA=1", "-DVOLVOXAI_ENABLE_TRAINING=0",
                       "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0",
                       "-DVOLVOXAI_ENABLE_METAL=0", *[f"-I{ROOT / path}" for path in includes], str(unit),
                       str(runtime / "decode_row_set.c"), str(runtime / "attention_mask.c"), "-lm", "-o", str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("PASS production compiled row commands", result.stdout)


GUARD_PRELUDE = r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "decode_projection.h"
static VxEngineState owner;
static T tensors[5];static Node nodes[1];static Ref inputs[4],outputs[1];
static GraphNodeFusion fusions[1];static VxIncrementalTensorBinding bindings[5];
static VxIncrementalCudaCommand commands[1];static float storage[5][16];
static unsigned dirty;
VxEngineState* vx_engine_state_current(void){return &owner;}
uint64_t volvoxai_engine_model_generation_locked(void){return owner.model_generation;}
static int device_row_backend_selected(void){return 1;}
int vx_trace_nodes(const VxTraceScope* scope){(void)scope;return 0;}
T* t_find(const char* name){for(int i=0;i<5;i++)if(!strcmp(tensors[i].name,name))return &tensors[i];return NULL;}
int vx_incremental_tensor_dirty_before_node_locked(const T* tensor,int node){assert(node==0);return (dirty>>(tensor-tensors))&1u;}
int volvoxai_engine_tensor_is_model_weight_locked(const char* name){return !strcmp(name,"key_weight")||!strcmp(name,"value_weight");}
int vx_paged_bound_locked(void){return 0;}
int vx_paged_slot_pages_locked(int slot,VxDecodeSlotPages* pages){(void)slot;(void)pages;return 0;}
static T* nin(Node* node,VxPortKind port){return vx_decode_projection_port(node,port);}
static int tensor_same_shape(const T* a,const T* b){return a->ndim==b->ndim&&!memcmp(a->shape,b->shape,sizeof(a->shape));}
static int producer_index_any(const char* name){(void)name;return -1;}
'''

GUARD_CHECKS = r'''
static void setup(int causal,int memory_is_weight) {
 memset(&owner,0,sizeof(owner));memset(tensors,0,sizeof(tensors));memset(nodes,0,sizeof(nodes));memset(bindings,0,sizeof(bindings));memset(commands,0,sizeof(commands));
 owner.tensors=tensors;owner.tensor_count=5;owner.tensor_capacity=5;owner.nodes=nodes;owner.node_count=1;owner.node_capacity=1;owner.node_fusion=fusions;owner.model_generation=7;owner.use_cuda=1;
 owner.incremental_plan=(VxIncrementalPlan){.valid=1,.generation=7,.tensor_count=5,.node_count=1,.cuda_commands_active=0};
 owner.incremental_scratch=(VxIncrementalScratch){.tensor_bindings=bindings,.tensor_capacity=5,.node_capacity=1,.cuda_commands=commands};
 const char* names[]={"query","output","key_weight","value_weight","mask"};
 for(int i=0;i<5;i++) {
  T* t=tensors+i;strcpy(t->name,names[i]);t->dtype=i==4?T_I32:T_F32;t->elem_size=4;t->ndim=i==4?1:3;
  int sequence=i<2?4:causal?4:3;
  if(i==4){t->shape[0]=sequence;t->numel=sequence;}else{t->shape[0]=1;t->shape[1]=sequence;t->shape[2]=2;t->numel=sequence*2;}
  t->data=storage[i];t->is_graph_input=i==0||i==4;
  for(int k=0;k<16;k++)storage[i][k]=(float)(i*16+k);
  bindings[i]=(VxIncrementalTensorBinding){.data=t->data,.numel=t->numel,.elem_size=4,.dtype=t->dtype,.ndim=t->ndim,.is_graph_input=t->is_graph_input,.is_model_weight=memory_is_weight&&(i==2||i==3)};
  memcpy(bindings[i].shape,t->shape,sizeof(t->shape));
 }
 if(!memory_is_weight){strcpy(tensors[2].name,"key_input");strcpy(tensors[3].name,"value_input");}
 nodes[0]=(Node){.operator_kind=VX_OP_CROSS_SDPA,.nin=4,.nout=1,.ins=inputs,.outs=outputs,.input_capacity=4,.output_capacity=1,.primary_output_index=1};
 strcpy(nodes[0].out,"output");outputs[0].tensor_index=1;strcpy(outputs[0].name,"output");
 VxPortKind ports[]={VX_PORT_Q,VX_PORT_K,VX_PORT_V,VX_PORT_MASK};int indices[]={0,2,3,4};
 for(int i=0;i<4;i++){inputs[i].port=ports[i];inputs[i].tensor_index=indices[i];strcpy(inputs[i].name,tensors[indices[i]].name);}
 nodes[0].parsed_params.parsed=1;nodes[0].parsed_params.cache[VX_NODE_PARAM_CAUSAL].kind=VX_NODE_PARAM_BOOL;nodes[0].parsed_params.cache[VX_NODE_PARAM_CAUSAL].value.i32=causal;
 dirty=1;
}
int main(void) {
 VxDecodeRowProjection p,candidate;unsigned checks=0;
 for(int memory_is_weight=0;memory_is_weight<2;memory_is_weight++)for(int causal=0;causal<2;causal++) {
  setup(causal,memory_is_weight);
  VxIncrementalRowProbeScope scope;vx_incremental_row_probe_begin_locked(&scope,0,1);
  assert(owner.incremental_plan.cuda_commands_active);
  assert(vx_decode_cuda_projection_describe(nodes,0,1,&p));
  vx_incremental_row_probe_end_locked(&scope);assert(!owner.incremental_plan.cuda_commands_active);
  assert(commands[0].valid==(causal||memory_is_weight));
  float before[16];memcpy(before,storage[1],sizeof(before));
  for(int mask_dirty=0;mask_dirty<2;mask_dirty++) {
   dirty=1u|(mask_dirty?16u:0u);memcpy(storage[4],&mask_dirty,sizeof(mask_dirty));
   vx_incremental_row_probe_begin_locked(&scope,0,1);assert(owner.incremental_plan.cuda_commands_active);
   int ordinary=vx_decode_cuda_projection_describe_uncached(nodes,0,1,&p);
   int current=vx_decode_cuda_projection_describe(nodes,0,1,&candidate);
   assert(ordinary==(causal||!mask_dirty));assert(current==ordinary);
   if(!causal && memory_is_weight && mask_dirty)assert(!current && commands[0].valid);
   assert(!memcmp(before,storage[1],sizeof(before)));checks++;
   vx_incremental_row_probe_end_locked(&scope);assert(!owner.incremental_plan.cuda_commands_active);
  }
  if(causal) {
   dirty=17u;vx_incremental_row_probe_begin_locked(&scope,0,1);assert(vx_decode_cuda_projection_describe(nodes,0,2,&p) && p.keys==3 && p.rows.positions[0]==2);
   assert(commands[0].valid);checks++;vx_incremental_row_probe_end_locked(&scope);
  }
 }
 printf("PASS real CUDA projection invariants: %u checks; noncausal dirty mask refuses before writes, causal dynamic masks retained\n",checks);
}
'''


class ProjectionInvariantTests(unittest.TestCase):
    def test_current_dirty_mask_matches_uncached_admission(self):
        cc = shutil.which('cc')
        if not cc:
            self.skipTest('C compiler unavailable')
        runtime = ROOT / 'native/src/runtime'
        src = (runtime / 'engine_runtime_row_cuda.inc').read_text()
        text = GUARD_PRELUDE
        text += packed_descriptor_validation_source()
        incremental = (runtime / 'incremental_runtime.c').read_text()
        for name in ['incremental_row_controls_clear_locked', 'incremental_cuda_commands_eligible_locked',
                     'incremental_cuda_tensor_binding_current_locked', 'incremental_cuda_bindings_current_locked',
                     'vx_incremental_row_probe_begin_locked', 'vx_incremental_row_probe_end_locked']:
            text += function_source(incremental, name)
        text += function_source((runtime / 'engine_runtime_dispatch.inc').read_text(), 'linear_node_weight_layout') + \
                function_source((runtime / 'engine_runtime_f32_cpu.inc').read_text(), 'normalization_dimensions')
        for name in ['vx_decode_cuda_projection_describe_uncached', 'vx_decode_cuda_command_operand',
                     'vx_decode_cuda_command_operand_bind', 'vx_decode_cuda_command_compile',
                     'vx_decode_cuda_row_geometry_current', 'vx_decode_cuda_command_bind',
                     'vx_decode_cuda_projection_describe']:
            text += function_source(src, name)
        text += GUARD_CHECKS
        includes = ['native/include', 'native/src', 'native/src/runtime', 'native/src/kernels', 'native/src/backends', 'native/third_party']
        with tempfile.TemporaryDirectory(prefix='volvox-projection-invariant-') as temp:
            unit = Path(temp) / 'test.c'
            binary = Path(temp) / 'test'
            unit.write_text(text)
            cmd = [cc, '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                   '-DVOLVOXAI_ENABLE_TRAINING=0', '-DVOLVOXAI_ENABLE_CUDA=1', '-DVOLVOXAI_ENABLE_WEBGPU=0',
                   '-DVOLVOXAI_ENABLE_VULKAN=0', '-DVOLVOXAI_ENABLE_OPENGL=0', '-DVOLVOXAI_ENABLE_METAL=0',
                   f'-I{runtime}'] + [f'-I{ROOT / p}' for p in includes] + [str(unit), str(runtime / 'attention_mask.c'), '-lm', '-o', str(binary)]
            if os.getenv('VOLVOXAI_TEST_SANITIZE'):
                cmd[1:1] = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('PASS real CUDA projection invariants', r.stdout)





PACKED_COMMAND_CHECKS = r"""
static int backend_status, backend_calls;
static const float* expected_input;
static float* expected_output;
static int cuda_graph_packed_linear_decode_f32(const float* input,const int32_t* weight,
 const float* scale,const float* bias,float* output,uint32_t k,uint32_t n,uint32_t wb,uint32_t ab) {
 backend_calls++;
 assert(input==expected_input && output==expected_output && weight==(const int32_t*)storage[2]);
 assert(scale==storage[3] && bias==(nodes[0].nin==4?storage[4]:NULL));
 assert(k==9 && n==3 && wb==(uint32_t)vx_node_param_i32(nodes,VX_NODE_PARAM_WEIGHT_BITS,0));
 assert(ab==(uint32_t)vx_engine_packed_activation_bits());
 return backend_status;
}
static void packed_setup(int wb,int ab,int bias) {
 memset(&owner,0,sizeof(owner));memset(tensors,0,sizeof(tensors));memset(nodes,0,sizeof(nodes));
 memset(fusions,0,sizeof(fusions));memset(commands,0,sizeof(commands));
 owner.tensors=tensors;owner.tensor_count=owner.tensor_capacity=5;
 owner.nodes=nodes;owner.node_count=owner.node_capacity=1;owner.node_fusion=fusions;
 owner.model_generation=7;owner.use_cuda=1;owner.active_row=owner.execution_row=-1;
 const char* names[]={"input","output","weight","scale","bias"};
 for(int i=0;i<5;i++) {
  T* t=tensors+i;strcpy(t->name,names[i]);t->data=storage[i];t->elem_size=4;
  t->dtype=i==2?T_I32:T_F32;t->ndim=i<2?3:i<4?2:1;t->is_graph_input=i==0;
  if(i<2){t->shape[0]=1;t->shape[1]=4;t->shape[2]=i?3:9;}
  else {t->shape[0]=3;if(i<4)t->shape[1]=i==2?(9+32/wb-1)/(32/wb):1;}
  t->numel=1;for(int d=0;d<t->ndim;d++)t->numel*=t->shape[d];
 }
 nodes[0]=(Node){.operator_kind=VX_OP_PACKED_LINEAR,.nin=bias?4:3,.nout=1,.ins=inputs,.outs=outputs,
  .input_capacity=4,.output_capacity=1,.primary_output_index=1};
 strcpy(nodes[0].out,"output");strcpy(outputs[0].name,"output");outputs[0].tensor_index=1;
 VxPortKind ports[]={VX_PORT_INPUT,VX_PORT_WEIGHT,VX_PORT_SCALE,VX_PORT_BIAS};int indices[]={0,2,3,4};
 for(int i=0;i<4;i++){inputs[i].port=ports[i];inputs[i].tensor_index=indices[i];strcpy(inputs[i].name,names[indices[i]]);}
 nodes[0].parsed_params.parsed=1;
 nodes[0].parsed_params.cache[VX_NODE_PARAM_GROUP_SIZE]=(VxCachedNodeParam){.kind=VX_NODE_PARAM_I32,.value.i32=128};
 nodes[0].parsed_params.cache[VX_NODE_PARAM_WEIGHT_BITS]=(VxCachedNodeParam){.kind=VX_NODE_PARAM_I32,.value.i32=wb};
 owner.packed_arithmetic=ab==8?VX_PACKED_ARITHMETIC_DYNAMIC_I8:ab==16?VX_PACKED_ARITHMETIC_F16:VX_PACKED_ARITHMETIC_F32;
 dirty=1;backend_calls=0;
}
static void packed_checks(void) {
 VxDecodeRowProjection p,bound;unsigned checks=0;
 for(int wb=4;wb<=8;wb+=4)for(int ab=8;ab<=32;ab*=2)for(int bias=0;bias<2;bias++) {
  packed_setup(wb,ab,bias);
  assert(vx_decode_projection_describe(nodes,1,&p));
  assert(p.input_count==1 && p.inputs[0].width==9 && p.output.width==3);
  assert(vx_decode_cuda_command_compile(nodes,0,&p,commands));
  assert(commands[0].numerical && commands[0].valid && commands[0].weight_index==2 && commands[0].scale_index==3);
  assert(commands[0].bias_index==(bias?4:-1) && commands[0].weight_bits==wb && commands[0].activation_bits==ab);
  assert(vx_decode_cuda_view_safe(nodes,&p));
  assert(vx_decode_cuda_command_bind(commands,2,&bound) && bound.rows.positions[0]==2);
  assert(bound.inputs[0].tensor==tensors && bound.output.tensor==tensors+1);
  assert(!vx_decode_cuda_command_bind(commands,4,&bound));
  T narrow=tensors[0],out=tensors[1];narrow.numel=9;out.numel=3;
  narrow.data+=18;out.data+=6;expected_input=narrow.data;expected_output=out.data;
  for(backend_status=-1;backend_status<=1;backend_status++) {
   assert(vx_decode_cuda_command_run(commands,&narrow,&out)==(backend_status==1));checks++;
  }
  assert(backend_calls==3);narrow.numel=18;assert(!vx_decode_cuda_command_run(commands,&narrow,&out));
  assert(backend_calls==3);checks++;
 }
 for(int mutation=0;mutation<19;mutation++) {
  packed_setup(4,8,1);assert(vx_decode_projection_describe(nodes,1,&p));
  switch(mutation) {
   case 0:tensors[2].dtype=T_F32;break;
   case 1:tensors[2].shape[1]++;tensors[2].numel+=3;break;
   case 2:tensors[3].shape[1]++;tensors[3].numel+=3;break;
   case 3:tensors[4].shape[0]++;tensors[4].numel++;break;
   case 4:tensors[1].data=tensors[0].data;break;
   case 5:nodes[0].parsed_params.cache[VX_NODE_PARAM_WEIGHT_BITS].value.i32=3;break;
   case 6:tensors[3].dtype=T_I32;break;
   case 7:nodes[0].parsed_params.cache[VX_NODE_PARAM_GROUP_SIZE].value.i32=64;break;
   case 8:tensors[3].is_graph_input=1;break;
   case 9:p.rows.slots=2;break;
   case 10:p.inputs[0].width++;break;
   case 11:p.output.width++;break;
   case 12:tensors[2].elem_size=2;break;
   case 13:tensors[3].elem_size=2;break;
   case 14:tensors[4].elem_size=2;break;
   case 15:tensors[0].elem_size=2;break;
   case 16:tensors[1].elem_size=2;break;
   case 17:tensors[3].data=NULL;break;
   case 18:tensors[2].is_graph_input=1;break;
  }
  assert(!vx_decode_cuda_command_compile(nodes,0,&p,commands));
  assert(!commands[0].valid && !backend_calls);checks++;
 }
 printf("PASS PackedLinear typed row commands: %u checks\n",checks);
}
int main(void){packed_checks();return 0;}
"""


class PackedCommandsTests(unittest.TestCase):
    def test_packed_descriptor_binding_and_submission_status(self):
        cc = shutil.which('cc')
        if not cc:
            self.skipTest('C compiler unavailable')
        row = (RUNTIME / 'engine_runtime_row_cuda.inc').read_text()
        text = GUARD_PRELUDE.replace('storage[5][16]', 'storage[5][64]')
        text = text.replace('return !strcmp(name,"key_weight")||!strcmp(name,"value_weight");',
            'T* t=t_find(name);return t && !t->is_graph_input && (!strcmp(name,"weight")||!strcmp(name,"scale")||!strcmp(name,"bias"));')
        text += packed_descriptor_validation_source()
        text += function_source((RUNTIME / 'engine_runtime_dispatch.inc').read_text(), 'linear_node_weight_layout')
        text += function_source((RUNTIME / 'engine_runtime_f32_cpu.inc').read_text(), 'normalization_dimensions')
        # Unexpected backend calls fail; only the new private Packed entry may run.
        for name in ('cuda_graph_matmul_layout_f32', 'cuda_graph_rmsnorm_f32', 'cuda_graph_silu_f32',
                     'cuda_graph_add_relu_f32', 'cuda_graph_binary_f32', 'cuda_graph_embedding_f32',
                     'cuda_graph_argmax_f32'):
            signature = function_source(COMMANDS_MOCKS, name).split('{', 1)[0]
            text += signature + '{assert(0);return 0;}\n'
        cut = PACKED_COMMAND_CHECKS.index('static void packed_checks')
        text += PACKED_COMMAND_CHECKS[:cut]
        for name in ('vx_decode_cuda_command_operand', 'vx_decode_cuda_command_operand_bind',
                     'vx_decode_cuda_command_compile', 'vx_decode_cuda_row_geometry_current',
                     'vx_decode_cuda_command_bind', 'vx_decode_cuda_command_run', 'vx_decode_cuda_view_safe'):
            text += function_source(row, name)
        text += PACKED_COMMAND_CHECKS[cut:]
        includes = ['native/include', 'native/src', 'native/src/runtime', 'native/src/kernels',
                    'native/src/backends', 'native/third_party']
        with tempfile.TemporaryDirectory(prefix='volvox-packed-row-') as temp:
            unit, binary = Path(temp) / 'test.c', Path(temp) / 'test'
            unit.write_text(text)
            cmd = [cc, '-std=c11', '-D_POSIX_C_SOURCE=200809L', '-O2', '-Wall', '-Wextra', '-Werror',
                   '-Wno-unused-function', '-Wno-unused-parameter', '-Wno-unused-variable',
                   '-DVOLVOXAI_ENABLE_TRAINING=0', '-DVOLVOXAI_ENABLE_CUDA=1', '-DVOLVOXAI_ENABLE_WEBGPU=0',
                   '-DVOLVOXAI_ENABLE_VULKAN=0', '-DVOLVOXAI_ENABLE_OPENGL=0', '-DVOLVOXAI_ENABLE_METAL=0',
                   *[f'-I{ROOT / p}' for p in includes], str(unit), str(RUNTIME / 'attention_mask.c'), '-lm', '-o', str(binary)]
            if os.getenv('VOLVOXAI_TEST_SANITIZE'):
                cmd[1:1] = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            r = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('PASS PackedLinear typed row commands', r.stdout)


if __name__ == "__main__":
    unittest.main()
