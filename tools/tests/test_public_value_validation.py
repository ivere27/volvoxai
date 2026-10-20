"""Compare production compiled predicates with the preceding preflight.

The reference fixture is the previous production implementation, isolated to
tests. Both paths read the same real Node/cJSON/binding representations.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

PREAMBLE = r'''
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "engine_internal.h"
#include "vx_lifecycle.h"
'''

MODEL = r'''
static size_t cold_lookup_calls;
struct VxModel { cJSON* logical_graph; };
struct VxCompiledModel { VxModel* model; VxBackendKind builtin_backend; };
struct VxExecutionContext {
 VxEngineState* engine_state; VxCompiledModel* compiled;
 const VxDeclaredTensor* logical_tensors; size_t logical_tensor_count;
 uint64_t allocated_bytes; VxPublicValuePlan public_value_plan;
};
typedef struct VxStagedBindingBatch { VxTensorBinding* bindings; size_t count; } VxStagedBindingBatch;
static int fail_allocation;
static void* predicate_realloc(void* pointer,size_t bytes) {
 return fail_allocation==1 ? NULL : realloc(pointer,bytes);
}
static void* predicate_calloc(size_t count,size_t bytes) {
 return fail_allocation==2 ? NULL : calloc(count,bytes);
}
'''

FIXTURE = r'''
typedef struct Fixture {
 VxExecutionContext context; VxCompiledModel compiled; VxModel model;
 VxEngineState state; T tensors[24]; Node nodes[8]; Ref inputs[8][4];
 VxDeclaredTensor logical[8]; VxTensorBinding bindings[6];
 int32_t integers[6][4]; float floats[6][4]; size_t node_count,tensor_count;
} Fixture;
static uint32_t tensor(Fixture* f,const char* name,int dtype,int public,int rank,int size) {
 uint32_t index=(uint32_t)f->tensor_count++; T* t=&f->tensors[index];
 snprintf(t->name,sizeof(t->name),"%s",name); t->dtype=dtype;
 t->is_graph_input=public; t->ndim=rank; t->shape[0]=size;
 for(int a=1;a<rank;a++)t->shape[a]=1;
 t->numel=size; t->elem_size=4; t->data=(float*)f->integers;
 f->state.tensor_count=(int)f->tensor_count; return index;
}
static Node* node(Fixture* f,const char* kind,const char* out,const char* params) {
 size_t index=f->node_count++; Node* n=&f->nodes[index];
 n->operator_kind=vx_operator_kind_from_name(kind); assert(n->operator_kind);
 snprintf(n->out,sizeof(n->out),"%s",out); n->ins=f->inputs[index];
 n->params=cJSON_Parse(params?params:"{}");
 cJSON* description=cJSON_CreateObject(); cJSON_AddStringToObject(description,"opType",kind);
 cJSON_AddItemToObject(description,"inputs",cJSON_CreateObject());
 cJSON* outputs=cJSON_AddObjectToObject(description,"outputs");
 cJSON* port=cJSON_AddObjectToObject(outputs,"out"); cJSON_AddStringToObject(port,"tensor",out);
 cJSON_AddItemToArray(cJSON_GetObjectItemCaseSensitive(f->model.logical_graph,"nodes"),description);
 f->logical[index].name=(char*)n->out;
 f->context.logical_tensor_count=f->node_count;
 f->state.node_count=(int)f->node_count;
 tensor(f,out,T_I32,0,1,1); return n;
}
static void input(Fixture* f,Node* n,VxPortKind role,const char* key,const char* name) {
 Ref* r=&n->ins[n->nin++]; r->port=role; snprintf(r->name,sizeof(r->name),"%s",name);
 int index=(int)(n-f->nodes); cJSON* document=cJSON_GetArrayItem(
  cJSON_GetObjectItemCaseSensitive(f->model.logical_graph,"nodes"),index);
 cJSON_AddStringToObject(cJSON_GetObjectItemCaseSensitive(document,"inputs"),key,name);
}
static void initialize(Fixture* f) {
 memset(f,0,sizeof(*f)); f->model.logical_graph=cJSON_Parse("{\"nodes\":[]}");
 f->compiled.model=&f->model; f->compiled.builtin_backend=VX_BACKEND_KIND_CUDA;
 f->state.nodes=f->nodes; f->state.tensors=f->tensors; f->state.model_generation=7;
 f->context.engine_state=&f->state; f->context.compiled=&f->compiled;
 f->context.logical_tensors=f->logical; f->context.allocated_bytes=100;
}
static void dispose(Fixture* f) {
 vx_public_value_plan_clear(&f->context); assert(f->context.allocated_bytes==100);
 for(size_t i=0;i<f->node_count;i++)cJSON_Delete(f->nodes[i].params);
 cJSON_Delete(f->model.logical_graph);
}
static VxTensorBinding* bind(Fixture* f,size_t i,const char* name,int dtype) {
 VxTensorBinding* b=&f->bindings[i]; memset(b,0,sizeof(*b));
 b->name=name; b->dtype=dtype; b->location=VX_MEMORY_HOST; b->byte_size=4;
 b->data=dtype==VX_DTYPE_I32?(void*)f->integers[i]:(void*)f->floats[i]; return b;
}
static void equivalent(Fixture* f,size_t count,const VolvoxAIEngineResolvedTensor* resolved,
 size_t resolved_count,VxStatus expected) {
 VxStagedBindingBatch staged={.bindings=f->bindings,.count=count};
 VxStatus before=reference_vx_native_gpu_preflight_public_values(&f->context,&staged,resolved,resolved_count);
 int current=f->context.public_value_plan.valid &&
  f->context.public_value_plan.model_generation==f->state.model_generation;
 size_t lookups=cold_lookup_calls;
 VxStatus after=vx_native_gpu_preflight_public_values(&f->context,&staged,resolved,resolved_count);
 if(current)assert(cold_lookup_calls==lookups);
 if(before!=expected || after!=before)fprintf(stderr,"expected %d old %d compiled %d\n",expected,before,after);
 assert(before==expected && after==before);
}
static void embedding_tests(void) {
 Fixture f; initialize(&f); tensor(&f,"ids",T_I32,1,1,1);
 uint32_t weight=tensor(&f,"weight",T_F32,0,2,4);
 Node* n=node(&f,"Embedding","out",NULL); input(&f,n,VX_PORT_INPUT,"input","ids");
 input(&f,n,VX_PORT_WEIGHT,"weight","weight"); bind(&f,0,"ids",VX_DTYPE_I32);
 f.integers[0][0]=3; equivalent(&f,1,NULL,0,VX_STATUS_OK);
 assert(f.context.public_value_plan.count==2); void* allocation=f.context.public_value_plan.predicates;
 f.integers[0][0]=4; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.integers[0][0]=-1; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 equivalent(&f,0,NULL,0,VX_STATUS_OK);
 f.integers[0][0]=4; f.tensors[weight].shape[0]=5; equivalent(&f,1,NULL,0,VX_STATUS_OK);
 assert(allocation==f.context.public_value_plan.predicates);
 f.bindings[0].data=NULL; f.bindings[0].byte_size=0; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.bindings[0].data=f.integers[0]; equivalent(&f,1,NULL,0,VX_STATUS_OK);
 f.bindings[0].byte_size=3; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.bindings[0].byte_size=4; f.bindings[0].location=VX_MEMORY_DEVICE;
 equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.bindings[0].location=VX_MEMORY_HOST; f.bindings[0].dtype=VX_DTYPE_F32;
 equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 dispose(&f);
}
static void where_gather_tests(void) {
 Fixture f; initialize(&f); tensor(&f,"left",T_I32,1,1,1); tensor(&f,"right",T_I32,1,1,1);
 uint32_t table=tensor(&f,"table",T_F32,0,2,4);
 Node* where=node(&f,"Where","chosen",NULL);
 input(&f,where,VX_PORT_X,"x","left"); input(&f,where,VX_PORT_Y,"y","right");
 Node* gather=node(&f,"Gather","gathered","{\"axis\":0}");
 input(&f,gather,VX_PORT_INPUT,"input","table"); input(&f,gather,VX_PORT_INDICES,"indices","chosen");
 bind(&f,0,"left",VX_DTYPE_I32); bind(&f,1,"right",VX_DTYPE_I32);
 f.integers[0][0]=-4; f.integers[1][0]=3; equivalent(&f,2,NULL,0,VX_STATUS_OK);
 assert(f.context.public_value_plan.count==3);
 f.integers[1][0]=4; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.integers[1][0]=0; f.integers[0][0]=-5; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.integers[0][0]=-5; VolvoxAIEngineResolvedTensor resolved={.name="table",.rank=2,.shape={5,1}};
 equivalent(&f,2,&resolved,1,VX_STATUS_OK);
 resolved.shape[0]=3; f.integers[0][0]=-3; f.integers[1][0]=3;
 equivalent(&f,2,&resolved,1,VX_STATUS_INVALID_ARGUMENT);
 uint32_t resident[6]={0,UINT32_MAX,1,UINT32_MAX,2,UINT32_MAX};
 gather->resident_slot_domain=6; gather->resident_slot_rows=resident;
 f.integers[0][0]=-2; f.integers[1][0]=0; equivalent(&f,2,NULL,0,VX_STATUS_OK);
 f.integers[0][0]=-1; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.tensors[table].shape[0]=0; gather->resident_slot_domain=0; gather->resident_slot_rows=NULL;
 equivalent(&f,0,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.tensors[table].shape[0]=4; ++f.state.model_generation;
 where->operator_kind=VX_OP_LINEAR;
 equivalent(&f,0,NULL,0,VX_STATUS_INVALID_GRAPH);
 where->operator_kind=VX_OP_WHERE; ++f.state.model_generation;
 equivalent(&f,0,NULL,0,VX_STATUS_OK); dispose(&f);
}
static void position_tests(void) {
 Fixture f; initialize(&f); tensor(&f,"positions",T_I32,1,1,1);
 Node* n=node(&f,"RoPE","rope0",NULL); input(&f,n,VX_PORT_POSITION_IDS,"position_ids","positions");
 n=node(&f,"RoPE","rope1",NULL); input(&f,n,VX_PORT_POSITION_IDS,"position_ids","positions");
 bind(&f,0,"positions",VX_DTYPE_I32); f.integers[0][0]=INT32_MAX;
 equivalent(&f,1,NULL,0,VX_STATUS_OK); assert(f.context.public_value_plan.count==1);
 f.integers[0][0]=-1; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 dispose(&f);
}
static void moe_tests(void) {
 Fixture f; initialize(&f); tensor(&f,"routes",T_F32,1,1,1); tensor(&f,"weights",T_F32,1,1,1);
 tensor(&f,"expert_bank",T_F32,0,3,4);
 Node* n=node(&f,"MoELinear","moe",NULL);
 input(&f,n,VX_PORT_EXPERT_WEIGHT,"expert_weight","expert_bank");
 input(&f,n,VX_PORT_ROUTE_INDICES,"route_indices","routes");
 input(&f,n,VX_PORT_ROUTE_WEIGHTS,"route_weights","weights");
 bind(&f,0,"routes",VX_DTYPE_F32); bind(&f,1,"weights",VX_DTYPE_F32);
 f.floats[0][0]=3; f.floats[1][0]=-.5f; equivalent(&f,2,NULL,0,VX_STATUS_OK);
 for(int i=0;i<4;i++) { f.floats[0][0]=(float[]){4,-1,1.25f,NAN}[i]; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT); }
 f.floats[0][0]=1; f.floats[1][0]=INFINITY; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 uint32_t rows[5]={0,UINT32_MAX,1,2,UINT32_MAX}; n->resident_slot_domain=5; n->resident_slot_rows=rows;
 f.floats[1][0]=1; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.floats[0][0]=3; equivalent(&f,2,NULL,0,VX_STATUS_OK);
 f.floats[0][0]=4; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 dispose(&f);
 initialize(&f); tensor(&f,"bias",T_F32,1,1,1); tensor(&f,"correction",T_F32,1,1,1);
 n=node(&f,"MoERouter","router",NULL); input(&f,n,VX_PORT_BIAS,"bias","bias");
 input(&f,n,VX_PORT_CORRECTION_BIAS,"correction_bias","correction");
 bind(&f,0,"bias",VX_DTYPE_F32); bind(&f,1,"correction",VX_DTYPE_F32);
 f.floats[0][0]=-.5f; f.floats[1][0]=2; equivalent(&f,2,NULL,0,VX_STATUS_OK);
 f.floats[1][0]=NAN; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT); dispose(&f);
}
static void hash_and_finite_tests(void) {
 Fixture f; initialize(&f); tensor(&f,"ids",T_I32,1,1,1);
 uint32_t hash=tensor(&f,"hash",T_I32,0,2,3);
 Node* n=node(&f,"MoERouter","router",NULL);
 input(&f,n,VX_PORT_TOKEN_IDS,"token_ids","ids"); input(&f,n,VX_PORT_HASH_INDICES,"hash_indices","hash");
 bind(&f,0,"ids",VX_DTYPE_I32); f.integers[0][0]=2; equivalent(&f,1,NULL,0,VX_STATUS_OK);
 f.integers[0][0]=3; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.tensors[hash].shape[0]=4; equivalent(&f,1,NULL,0,VX_STATUS_OK);
 f.tensors[hash].ndim=1; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_GRAPH); dispose(&f);
 initialize(&f); tensor(&f,"scale",T_F32,1,1,1); tensor(&f,"bias",T_F32,1,1,1);
 n=node(&f,"QLayerNorm","normalized",NULL);
 input(&f,n,VX_PORT_WEIGHT,"scale","scale"); input(&f,n,VX_PORT_BIAS,"bias","bias");
 bind(&f,0,"scale",VX_DTYPE_F32); bind(&f,1,"bias",VX_DTYPE_F32);
 f.floats[0][0]=1; f.floats[1][0]=-2; equivalent(&f,2,NULL,0,VX_STATUS_OK);
 f.floats[0][0]=NAN; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT);
 f.floats[0][0]=1; f.floats[1][0]=INFINITY; equivalent(&f,2,NULL,0,VX_STATUS_INVALID_ARGUMENT); dispose(&f);
 initialize(&f); tensor(&f,"ids",T_I32,1,1,1);
 n=node(&f,"NGramIds","ngrams",NULL); input(&f,n,VX_PORT_TOKEN_IDS,"token_ids","ids");
 n->parsed_params.cache[VX_NODE_PARAM_VOCAB_SIZE].kind=VX_NODE_PARAM_I32;
 n->parsed_params.cache[VX_NODE_PARAM_VOCAB_SIZE].value.i32=4;
 bind(&f,0,"ids",VX_DTYPE_I32); f.integers[0][0]=3; equivalent(&f,1,NULL,0,VX_STATUS_OK);
 f.integers[0][0]=4; equivalent(&f,1,NULL,0,VX_STATUS_INVALID_ARGUMENT); dispose(&f);
}
static void allocation_tests(void) {
 Fixture f; initialize(&f); tensor(&f,"ids",T_I32,1,1,1); tensor(&f,"weight",T_F32,0,2,4);
 Node* n=node(&f,"Embedding","out",NULL); input(&f,n,VX_PORT_INPUT,"input","ids"); input(&f,n,VX_PORT_WEIGHT,"weight","weight");
 fail_allocation=1; assert(vx_native_gpu_value_plan_prepare(&f.context)==VX_STATUS_OUT_OF_MEMORY);
 assert(!f.context.public_value_plan.valid && f.context.allocated_bytes==100);
 fail_allocation=0; assert(vx_native_gpu_value_plan_prepare(&f.context)==VX_STATUS_OK);
 void* previous=f.context.public_value_plan.predicates; uint64_t charged=f.context.allocated_bytes;
 ++f.state.model_generation; fail_allocation=1;
 assert(vx_native_gpu_value_plan_prepare(&f.context)==VX_STATUS_OUT_OF_MEMORY);
 assert(previous==f.context.public_value_plan.predicates && charged==f.context.allocated_bytes);
 fail_allocation=2; assert(vx_native_gpu_value_plan_prepare(&f.context)==VX_STATUS_OUT_OF_MEMORY);
 assert(previous==f.context.public_value_plan.predicates && charged==f.context.allocated_bytes);
 fail_allocation=0; dispose(&f);
 VxPublicValuePlan plan={0}; for(uint32_t i=0;i<VX_PUBLIC_VALUE_PREDICATE_LIMIT;i++) {
  VxPublicValuePredicate p=vx_public_value_predicate(VX_PUBLIC_VALUE_F32_FINITE,0,0,UINT32_MAX); p.tensor=i;
  assert(vx_public_value_add(&plan,p)==VX_STATUS_OK);
 }
 VxPublicValuePredicate p=vx_public_value_predicate(VX_PUBLIC_VALUE_F32_FINITE,0,0,UINT32_MAX); p.tensor=VX_PUBLIC_VALUE_PREDICATE_LIMIT;
 assert(vx_public_value_add(&plan,p)==VX_STATUS_OUT_OF_MEMORY);
 assert(plan.capacity*sizeof(p)<=256u*1024u); free(plan.predicates);
 uint64_t peak=0; assert(vx_public_value_plan_peak_bytes(123,&peak));
 assert(peak==10240u*sizeof(VxPublicValuePredicate)+123u);
 assert(!vx_public_value_plan_peak_bytes(UINT64_MAX,&peak));
 assert(!vx_public_value_plan_peak_bytes(0,NULL));
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+(double)t.tv_nsec*1e-9; }
static void benchmark(int count) {
 VxEngineState state={0}; VxModel model={0}; VxCompiledModel compiled={.model=&model,.builtin_backend=VX_BACKEND_KIND_CUDA};
 Node* nodes=calloc((size_t)count,sizeof(*nodes)); T* tensors=calloc((size_t)count+3,sizeof(*tensors));
 Ref* refs=calloc((size_t)count*2,sizeof(*refs));
 VxDeclaredTensor* logical=calloc((size_t)count,sizeof(*logical));
 model.logical_graph=cJSON_Parse("{\"nodes\":[]}");
 const char* names[]={"ids","positions","embedding_weight"};
 for(int i=0;i<3;i++) { snprintf(tensors[i].name,sizeof(tensors[i].name),"%s",names[i]);
  tensors[i].dtype=i==2?T_F32:T_I32; tensors[i].ndim=i==2?2:1;
  tensors[i].shape[0]=i==2?131072:256; tensors[i].shape[1]=i==2?1024:0;
  tensors[i].is_graph_input=i<2; }
 for(int i=0;i<count;i++) {
  snprintf(nodes[i].out,sizeof(nodes[i].out),"output%d",i); snprintf(tensors[i+3].name,sizeof(tensors[i+3].name),"output%d",i);
  const char* kind=i==0?"Embedding":i<=60?"RoPE":"Linear";
  nodes[i].operator_kind=vx_operator_kind_from_name(kind); logical[i].name=nodes[i].out;
  nodes[i].ins=refs+i*2;
  cJSON* n=cJSON_CreateObject(); cJSON_AddStringToObject(n,"opType",kind); cJSON* input=cJSON_AddObjectToObject(n,"inputs");
  if(i==0) { nodes[i].nin=2; nodes[i].ins[0].port=VX_PORT_INPUT; strcpy(nodes[i].ins[0].name,"ids");
   nodes[i].ins[1].port=VX_PORT_WEIGHT; strcpy(nodes[i].ins[1].name,"embedding_weight");
   cJSON_AddStringToObject(input,"input","ids"); cJSON_AddStringToObject(input,"weight","embedding_weight"); }
  else if(i<=60) { nodes[i].nin=1; nodes[i].ins[0].port=VX_PORT_POSITION_IDS; strcpy(nodes[i].ins[0].name,"positions");
   cJSON_AddStringToObject(input,"position_ids","positions"); }
  cJSON* out=cJSON_AddObjectToObject(cJSON_AddObjectToObject(n,"outputs"),"out"); cJSON_AddStringToObject(out,"tensor",nodes[i].out);
  cJSON_AddItemToArray(cJSON_GetObjectItemCaseSensitive(model.logical_graph,"nodes"),n);
 }
 state.nodes=nodes; state.node_count=count; state.tensors=tensors; state.tensor_count=count+3; state.model_generation=1;
 VxExecutionContext context={.compiled=&compiled,.engine_state=&state,.logical_tensors=logical,.logical_tensor_count=(size_t)count};
 int32_t values[256]={0}; VxTensorBinding bindings[2]={
  {.name="ids",.dtype=VX_DTYPE_I32,.location=VX_MEMORY_HOST,.data=values,.byte_size=sizeof(values)},
  {.name="positions",.dtype=VX_DTYPE_I32,.location=VX_MEMORY_HOST,.data=values,.byte_size=sizeof(values)}};
 VxStagedBindingBatch staged={.bindings=bindings,.count=2}; assert(vx_native_gpu_value_plan_prepare(&context)==VX_STATUS_OK);
 assert(context.public_value_plan.count==3);
 const int passes=200; double begin=now();
 for(int i=0;i<passes;i++)assert(reference_vx_native_gpu_preflight_public_values(&context,&staged,NULL,0)==VX_STATUS_OK);
 double previous=(now()-begin)*1e6/passes; begin=now();
 for(int i=0;i<passes;i++)assert(vx_native_gpu_preflight_public_values(&context,&staged,NULL,0)==VX_STATUS_OK);
 double prepared=(now()-begin)*1e6/passes;
 printf("{\"synthetic_nodes\":%d,\"public_values\":512,\"predicates\":3,\"passes\":%d,\"old_us\":%.6f,\"compiled_us\":%.6f,\"scope\":\"local CPU synthetic binding microbenchmark; no model/GPU latency attribution\"}\n",count,passes,previous,prepared);
 vx_public_value_plan_clear(&context); cJSON_Delete(model.logical_graph); free(nodes); free(tensors); free(logical); free(refs);
}
int main(void) { embedding_tests(); where_gather_tests(); position_tests(); moe_tests(); hash_and_finite_tests(); allocation_tests();
 benchmark(484); benchmark(620); puts("compiled-public-value-validation PASS"); return 0; }
'''


class PublicValueValidation(unittest.TestCase):
    def test_production_predicates_match_reference_and_fail_before_mutation(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        runtime = root / "native/src/runtime"
        api = (runtime / "public_api.c").read_text()
        domain = (runtime / "public_api_native_gpu_domain.inc").read_text()
        execution = (runtime / "public_api_execution_context.inc").read_text()
        declared_start = api.index("typedef struct VxDeclaredTensor {")
        declared_end = api.index("} VxDeclaredTensor;", declared_start) + len("} VxDeclaredTensor;")
        plan_start = api.index("#define VX_PUBLIC_VALUE_PREDICATE_LIMIT")
        plan_end = api.index("} VxPublicValuePlan;", plan_start) + len("} VxPublicValuePlan;")
        unit = PREAMBLE + api[declared_start:declared_end] + "\n" + api[plan_start:plan_end] + MODEL
        unit += function_source(api, "vx_public_value_plan_clear")
        unit += function_source(api, "vx_public_value_plan_peak_bytes")
        unit += "static int vx_builtin_gpu_backend(VxBackendKind backend) { return backend == VX_BACKEND_KIND_CUDA; }\n"
        for name in ("vx_declared_tensor_find",):
            body = function_source((runtime / "public_api_symbolic_dimensions.inc").read_text(), name)
            unit += body.replace("{", "{ ++cold_lookup_calls;", 1)
        for name in ("vx_native_gpu_node_output", "vx_native_gpu_integer_parameter",
                     "vx_native_gpu_validation_tensor_find", "vx_native_gpu_validation_node_input_name",
                     "vx_native_gpu_validation_node_data_name", "vx_native_gpu_validation_node_producer",
                     "vx_native_gpu_validation_node_index", "vx_native_gpu_value_preserving_node"):
            body = function_source(domain, name)
            if name in ("vx_native_gpu_node_output", "vx_native_gpu_validation_node_producer", "vx_native_gpu_validation_tensor_find"):
                body = body.replace("{", "{ ++cold_lookup_calls;", 1)
            unit += body
        for name in ("vx_staged_binding_find", "vx_native_gpu_public_f32_source", "vx_resolved_tensor_find",
                     "vx_native_gpu_preflight_i32_binding", "vx_native_gpu_preflight_finite_binding",
                     "vx_moe_preflight_route_binding"):
            unit += function_source(execution, name)
        unit += "\n#define realloc predicate_realloc\n#define calloc predicate_calloc\n" + (runtime / "public_api_value_validation.inc").read_text()
        unit += "\n#undef realloc\n#undef calloc\n" + (root / "tools/tests/fixtures/public_value_preflight_reference.inc").read_text() + FIXTURE
        with tempfile.TemporaryDirectory(prefix="volvox-public-value-validation-") as directory:
            source, binary = Path(directory) / "test.c", Path(directory) / "test"
            source.write_text(unit)
            command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=0", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                       *["-I" + str(root / p) for p in ("native/include", "native/third_party", "native/src", "native/src/runtime", "native/src/kernels", "native/src/backends")],
                       str(source), str(root / "native/third_party/cJSON.c"),
                       str(root / "native/src/generated/operator_vocabulary.c"), "-o", str(binary)]
            built = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("compiled-public-value-validation PASS", checked.stdout)
            print(checked.stdout, end="")
