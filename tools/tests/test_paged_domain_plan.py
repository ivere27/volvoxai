"""Compare the compiled paged predicate with its retained production predecessor.

The C fixtures use real engine metadata and page allocation. They exercise
selection semantics, revision guards, transactional OOM and numerical gathers.
"""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "native/src/runtime/engine_internal.h").is_file())
RUNTIME = SOURCE_ROOT / "native/src/runtime"
INCLUDES = ["native/tests", "native/include", "native/src", "native/src/runtime",
            "native/src/kernels", "native/src/backends", "native/third_party",
            "native/third_party/synurang/include", "runtime/generated/c"]

PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "runtime_state.h"
#include "engine_internal.h"
static size_t malloc_calls,calloc_calls,lookup_calls;
static int fail_malloc,fail_calloc;
static void* checked_malloc(size_t n) {malloc_calls++;return fail_malloc?NULL:malloc(n);}
static void* checked_calloc(size_t n,size_t s) {calloc_calls++;return fail_calloc?NULL:calloc(n,s);}
#define malloc checked_malloc
#define calloc checked_calloc
#include "runtime_state.c"
#include "paged_binding.c"
#undef malloc
#undef calloc
#include "paged_domain_reference.inc"
void cJSON_Delete(cJSON* item) {assert(!item);}
const char* vx_operator_kind_name(VxOperatorKind op) {(void)op;return "fixture";}
T* t_find(const char* name) {
 lookup_calls++;if(!name || !name[0])return NULL;VxEngineState* s=vx_engine_state_current();
 for(int i=0;i<s->tensor_count;i++)if(!strcmp(s->tensors[i].name,name))return &s->tensors[i];
 return NULL;
}
static VxEngineState states[2];
static VxPagedKVCache* caches[2];
static unsigned long equivalence_checks,warm_checks;
static uint32_t rng=0xa3173b29;
static uint32_t random_u32(void) {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void context(int index,int nt,int nn) {
 VxEngineState* s=&states[index];g_current_engine_state=s;
 vx_paged_unbind_locked();vx_engine_state_release_graph_metadata(s);
 assert(vx_engine_state_reserve_graph_metadata(s,(size_t)nt,(size_t)nn)==VX_ENGINE_RESULT_OK);
 s->tensor_count=nt;s->node_count=nn;s->model_generation++;
 for(int i=0;i<nt;i++)snprintf(s->tensors[i].name,128,"tensor%d",i);
 for(int i=0;i<nn;i++)assert(!vx_node_reserve_refs(&s->nodes[i],4,4));
}
static void prepared(void) {
 VxEngineState* s=vx_engine_state_current();
 s->incremental_plan.valid=1;s->incremental_plan.generation=s->model_generation;
 s->incremental_plan.tensor_count=s->tensor_count;s->incremental_plan.node_count=s->node_count;
 for(int i=0;i<s->node_count;i++) {
  Node* n=&s->nodes[i];
  for(int j=0;j<n->nin;j++){T* t=t_find(n->ins[j].name);n->ins[j].tensor_index=t?(int)(t-s->tensors):-1;}
  for(int j=0;j<n->nout;j++){T* t=t_find(n->outs[j].name);n->outs[j].tensor_index=t?(int)(t-s->tensors):-1;}
 }
}
static void ref(Ref* r,int tensor,VxPortKind port) {
 snprintf(r->name,sizeof(r->name),"tensor%d",tensor);r->port=port;
 snprintf(r->key,sizeof(r->key),"port%d",(int)port);
}
static void bind_mask(int mask,int cache,int slot) {
 const char* names[24];int count=0;
 for(int i=0;i<g_nt;i++)if(mask&(1<<i))names[count++]=g_t[i].name;
 assert(!vx_paged_bind_locked(caches[cache],slot,names,count));
}
static int equivalent(const unsigned char* selected) {
 int expected=reference_paged_domain_supported_locked(selected);
 int actual=vx_paged_domain_supported_locked(selected);assert(actual==expected);equivalence_checks++;
 int compiled=paged_domain_key_matches(vx_engine_state_current());
 size_t before=lookup_calls,allocs=malloc_calls+calloc_calls;
 assert(vx_paged_domain_supported_locked(selected)==expected);
 assert(malloc_calls+calloc_calls==allocs);
 if(compiled && !getenv("VOLVOXAI_ROW_DEBUG")){assert(lookup_calls==before);warm_checks++;}
 for(int i=0;i<g_nt;i++)assert(vx_paged_tensor_locked(&g_t[i])==reference_paged_tensor_locked(&g_t[i]));
 return actual;
}
static void selection_edges(void) {
 context(0,4,4);Node* n=g_n;
 /* Two outputs of one selected node may name the same paged tensor. */
 n[0].nout=2;ref(&n[0].outs[0],0,VX_PORT_OUT);ref(&n[0].outs[1],0,VX_PORT_OUT);
 n[1].nout=1;ref(&n[1].outs[0],0,VX_PORT_OUT);
 n[2].operator_kind=VX_OP_Q_SDPA;n[2].nin=2;
 ref(&n[2].ins[0],0,VX_PORT_K);ref(&n[2].ins[1],1,VX_PORT_V);
 n[3].operator_kind=VX_OP_MUL;n[3].nin=1;ref(&n[3].ins[0],0,VX_PORT_K);
 bind_mask(3,0,0);prepared();
 for(int bits=0;bits<16;bits++) {
  unsigned char selected[4];for(int i=0;i<4;i++)selected[i]=(bits&(1<<i))?255:0;
  int expected=!(bits&8)&&((bits&3)!=3);assert(equivalent(selected)==expected);
 }
 unsigned char only[4]={7,0,0,0};assert(equivalent(only));
 assert(!equivalent(NULL));n[3].operator_kind=VX_OP_CROSS_SDPA;
 states[0].model_generation++;prepared();assert(equivalent(only));
 only[0]=0;only[3]=1;assert(equivalent(only));
 n[3].ins[0].port=VX_PORT_Q;states[0].model_generation++;prepared();assert(!equivalent(only));
 /* A selected output count is independent of the primary output spelling. */
 strcpy(n[0].out,"tensor3");only[0]=1;only[3]=0;assert(equivalent(only));
 assert(!setenv("VOLVOXAI_ROW_DEBUG","1",1));only[0]=0;only[3]=1;
 assert(!equivalent(only));assert(!unsetenv("VOLVOXAI_ROW_DEBUG"));
}
static void random_closures(void) {
 for(int trial=0;trial<256;trial++) {
  int nt=1+(int)(random_u32()%12),nn=(int)(random_u32()%9);context(0,nt,nn);
  for(int i=0;i<nn;i++) {
   Node* n=&g_n[i];VxOperatorKind kinds[]={VX_OP_MUL,VX_OP_Q_SDPA,VX_OP_CROSS_SDPA};
   n->operator_kind=kinds[random_u32()%3];n->nin=(int)(random_u32()%5);n->nout=(int)(random_u32()%5);
   n->disabled=random_u32()&1;
   for(int j=0;j<n->nin;j++) {
    VxPortKind ports[]={VX_PORT_K,VX_PORT_V,VX_PORT_Q,VX_PORT_INPUT};
    ref(&n->ins[j],(int)(random_u32()%(unsigned)(nt+((trial%5)==0))),ports[random_u32()%4]);
   }
   for(int j=0;j<n->nout;j++)ref(&n->outs[j],(int)(random_u32()%(unsigned)(nt+((trial%7)==0))),VX_PORT_OUT);
  }
  bind_mask((int)(random_u32()&((1u<<nt)-1)),trial&1,(trial>>1)&1);prepared();
  if(trial%11==0)states[0].incremental_plan.valid=0;
  equivalent(NULL);
  int selections=nn<=6?(1<<nn):64;
  for(int selection=0;selection<selections;selection++) {
   unsigned bits=nn<=6?(unsigned)selection:random_u32();unsigned char selected[12]={0};
   for(int i=0;i<nn;i++)selected[i]=(bits&(1u<<i))?(unsigned char)(1+(random_u32()%255)):0;
   equivalent(selected);
  }
 }
}
static void guard_edges(void) {
 context(0,4,2);g_n[0].nout=1;ref(&g_n[0].outs[0],0,VX_PORT_OUT);
 g_n[1].operator_kind=VX_OP_Q_SDPA;g_n[1].nin=1;ref(&g_n[1].ins[0],0,VX_PORT_K);
 prepared();bind_mask(1,0,0);assert(equivalent(NULL));
 VxEngineState* s=&states[0];VxPagedDomainPlan saved=s->paged_domain;
 uint64_t revision=s->paged.names_revision;size_t allocs=malloc_calls+calloc_calls;
 bind_mask(1,0,0);assert(s->paged.names_revision==revision && s->paged_domain.valid);
 assert(malloc_calls+calloc_calls==allocs);assert(equivalent(NULL));
 const char* duplicate[]={"tensor0","tensor0"};const char* missing[]={NULL};
 const char* changed[]={"tensor1"};char long_name[129];memset(long_name,'x',128);long_name[128]=0;
 const char* long_names[]={long_name};
 assert(vx_paged_bind_locked(caches[0],0,duplicate,2)==-1);
 assert(vx_paged_bind_locked(caches[0],0,missing,1)==-1);
 assert(vx_paged_bind_locked(caches[0],0,long_names,1)==-1);
 assert(vx_paged_bind_locked(caches[0],2,changed,1)==-1);
 assert(vx_paged_bind_locked(caches[0],0,changed,-1)==-1);
 fail_malloc=1;assert(vx_paged_bind_locked(caches[0],0,changed,1)==-1);fail_malloc=0;
 assert(s->paged.names_revision==revision && !memcmp(&saved,&s->paged_domain,sizeof(saved)));
 assert(equivalent(NULL));
 /* Cached membership applies only to exact table descriptors. */
 T outside=g_t[0];assert(vx_paged_tensor_locked(&outside));strcpy(outside.name,"tensor2");
 assert(!vx_paged_tensor_locked(&outside));assert(!vx_paged_tensor_locked(NULL));
 allocs=lookup_calls;g_t[0].shape[0]=256;g_t[0].numel=256;g_t[0].data=(float*)&outside;
 assert(equivalent(NULL));assert(lookup_calls>allocs); /* The independent oracle looked up names. */
 allocs=lookup_calls;assert(vx_paged_domain_supported_locked(NULL));assert(lookup_calls==allocs);
 bind_mask(2,0,0);assert(!s->paged_domain.valid);assert(equivalent(NULL));
 bind_mask(1,0,1);assert(!s->paged_domain.valid);assert(equivalent(NULL));
 bind_mask(1,1,1);assert(!s->paged_domain.valid);assert(equivalent(NULL));
 s->incremental_plan.valid=0;assert(equivalent(NULL));assert(!s->paged_domain.valid);prepared();
 s->model_generation++;assert(equivalent(NULL));assert(!s->paged_domain.valid);prepared();assert(equivalent(NULL));
 s->incremental_plan.tensor_count--;assert(equivalent(NULL));prepared();
 s->incremental_plan.node_count--;assert(equivalent(NULL));prepared();
 g_n[1].ins[0].tensor_index=3;s->paged_domain.valid=0;assert(equivalent(NULL));assert(!s->paged_domain.valid);prepared();
 /* A malformed/unresolved unselected reference must not reject the closure. */
 strcpy(g_n[1].ins[0].name,"absent");s->model_generation++;prepared();
 unsigned char selected[]={1,0};assert(equivalent(selected));assert(equivalent(NULL));assert(!s->paged_domain.valid);
 ref(&g_n[1].ins[0],0,VX_PORT_K);prepared();assert(equivalent(NULL));
 int old=g_n[1].nin;g_n[1].nin=-1;s->model_generation++;prepared();assert(equivalent(NULL));assert(!s->paged_domain.valid);
 g_n[1].nin=old;prepared();assert(equivalent(NULL));
 int cap=g_n[1].input_capacity;g_n[1].input_capacity=0;s->paged_domain.valid=0;
 assert(equivalent(NULL));assert(!s->paged_domain.valid);g_n[1].input_capacity=cap;
 prepared();assert(equivalent(NULL));
 /* Duplicate tensor names retain the authoritative first-entry rule. */
 strcpy(g_t[3].name,"tensor0");s->model_generation++;prepared();assert(equivalent(NULL));
 g_n[0].outs[0].tensor_index=3;s->paged_domain.valid=0;
 assert(equivalent(NULL));assert(!s->paged_domain.valid);prepared();
 strcpy(g_t[3].name,"tensor3");s->model_generation++;prepared();assert(equivalent(NULL));
 /* Empty references are ignored by the legacy domain guard, including when
  * a private caller has an empty tensor name in the table. */
 g_n[1].ins[0].name[0]=0;s->model_generation++;prepared();assert(equivalent(NULL));
 assert(!s->paged_domain.valid);ref(&g_n[1].ins[0],0,VX_PORT_K);prepared();assert(equivalent(NULL));
 /* Table identity guards work even if private callers retain a generation. */
 T other[4];memcpy(other,g_t,sizeof(other));T* old_t=s->tensors;s->tensors=other;
 assert(equivalent(NULL));assert(s->paged_domain.tensors==other);s->tensors=old_t;
 Node other_nodes[2];memcpy(other_nodes,g_n,sizeof(other_nodes));Node* old_n=s->nodes;s->nodes=other_nodes;
 assert(equivalent(NULL));assert(s->paged_domain.nodes==other_nodes);s->nodes=old_n;assert(equivalent(NULL));
 /* A distinct context never borrows its predecessor's cached certificate. */
 context(1,2,1);g_n[0].operator_kind=VX_OP_MUL;g_n[0].nin=1;ref(&g_n[0].ins[0],0,VX_PORT_K);
 bind_mask(1,0,0);prepared();assert(!equivalent(NULL));g_current_engine_state=&states[0];assert(equivalent(NULL));
 assert(!vx_paged_bind_locked(NULL,0,NULL,0));assert(!s->paged_domain.valid);assert(equivalent(NULL));
 bind_mask(0,0,0);assert(equivalent(NULL));assert(!vx_paged_tensor_locked(&g_t[0]));
 context(0,0,0);assert(!vx_paged_bind_locked(caches[0],0,NULL,0));prepared();assert(equivalent(NULL));
 /* No reserved metadata/private unprepared graph keeps the legacy route. */
 context(0,2,1);g_n[0].nout=1;ref(&g_n[0].outs[0],0,VX_PORT_OUT);prepared();bind_mask(1,0,0);
 unsigned char* membership=states[0].paged_domain.tensor_membership;states[0].paged_domain.tensor_membership=NULL;
 assert(equivalent(NULL));assert(!states[0].paged_domain.valid);states[0].paged_domain.tensor_membership=membership;
}
static void memory_lifecycle(void) {
 context(0,7,3);g_n[0].nout=2;ref(&g_n[0].outs[0],0,VX_PORT_OUT);ref(&g_n[0].outs[1],0,VX_PORT_OUT);
 prepared();bind_mask(1,0,0);assert(equivalent(NULL));VxEngineState* s=&states[0];
 uintptr_t begin=(uintptr_t)s->graph_metadata_storage,end=begin+s->graph_metadata_bytes;
 assert((uintptr_t)s->paged_domain.tensor_membership>=begin);
 assert((uintptr_t)(s->paged_domain.tensor_membership+s->tensor_capacity)<=end);
 assert((uintptr_t)s->paged_domain.selected_writers%_Alignof(int)==0);
 assert((uintptr_t)s->paged_domain.selected_writers>=begin);
 assert((uintptr_t)(s->paged_domain.selected_writers+s->tensor_capacity)<=end);
 assert((uintptr_t)s->paged_domain.bad_reads>=begin);
 assert((uintptr_t)(s->paged_domain.bad_reads+s->node_capacity)<=end);
 size_t next_bytes;assert(!vx_engine_state_graph_metadata_bytes_for(19,11,&next_bytes));
 size_t old_bytes=s->graph_metadata_bytes,old_peak=s->graph_metadata_transaction_peak_bytes;
 VxEngineState saved=*s;fail_calloc=1;
 assert(vx_engine_state_reserve_graph_metadata(s,19,11)==VX_ENGINE_RESULT_OUT_OF_MEMORY);fail_calloc=0;
 assert(!memcmp(&saved,s,sizeof(saved)));assert(equivalent(NULL));
 assert(vx_engine_state_reserve_graph_metadata(s,19,11)==VX_ENGINE_RESULT_OK);
 assert(s->graph_metadata_bytes==next_bytes && !s->paged_domain.valid && !s->incremental_plan.valid);
 assert(s->graph_metadata_transaction_peak_bytes==(old_peak>old_bytes+next_bytes?old_peak:old_bytes+next_bytes));
 uint64_t resident,peak,ports=0;for(size_t i=0;i<s->node_capacity;i++)ports+=(uint64_t)(s->nodes[i].input_capacity+s->nodes[i].output_capacity)*sizeof(Ref);
 assert(!vx_engine_state_graph_metadata_owned_bytes(s,&resident,&peak));
 assert(resident==next_bytes+ports && peak==s->graph_metadata_transaction_peak_bytes+2*ports);
 prepared();assert(equivalent(NULL));
 assert(vx_engine_state_reserve_graph_metadata(s,SIZE_MAX,0)==VX_ENGINE_RESULT_ERROR);
 vx_engine_state_release_graph_metadata(s);assert(!s->paged_domain.valid);
 assert(!s->paged_domain.tensor_membership && !s->paged_domain.selected_writers && !s->paged_domain.bad_reads);
 assert(!s->graph_metadata_bytes && !s->graph_metadata_transaction_peak_bytes);
 assert(equivalent(NULL));vx_paged_unbind_locked();
}
static void numerical_mapping(void) {
 context(0,3,2);g_n[0].nout=1;ref(&g_n[0].outs[0],0,VX_PORT_OUT);
 g_n[1].operator_kind=VX_OP_Q_SDPA;g_n[1].nin=1;ref(&g_n[1].ins[0],0,VX_PORT_K);
 prepared();bind_mask(1,0,0);assert(equivalent(NULL));
 vx_paged_kv_reset(caches[0]);assert(vx_paged_kv_append(caches[0],1,4)==VX_PAGED_KV_OK);
 assert(vx_paged_kv_append(caches[0],0,3)==VX_PAGED_KV_OK);
 float pool[32][4];for(int row=0;row<32;row++)for(int col=0;col<4;col++)pool[row][col]=(float)(row*13+col)*.125f;
 float saved[32][4];memcpy(saved,pool,sizeof(pool));
 for(int tokens=3;tokens<=9;tokens+=2) {
  int existing=vx_paged_kv_lengths(caches[0])[0];
  assert(vx_paged_kv_append(caches[0],0,tokens-existing)==VX_PAGED_KV_OK);
  assert(paged_domain_key_matches(&states[0]));assert(equivalent(NULL));
  const float* gathered=vx_paged_gather_f32_locked(&pool[0][0],32,4,tokens,0);assert(gathered);
  for(int logical=0;logical<tokens;logical++) {
   int expected=vx_paged_kv_physical_page(caches[0],0,logical/2)*2+logical%2;
   assert(vx_paged_row_locked(&g_t[0],logical)==expected);
   assert(vx_paged_row_locked(&g_t[1],logical)==logical);
   assert(!memcmp(gathered+logical*4,pool[expected],4*sizeof(float)));
  }
 }
 assert(vx_paged_row_locked(&g_t[0],15)==-1);
 assert(!memcmp(saved,pool,sizeof(pool)));
 /* Rejection is a pure metadata predicate: no cache/activation side effects. */
 VxPagedKVTelemetry before,after;vx_paged_kv_telemetry(caches[0],&before);
 g_n[1].operator_kind=VX_OP_MUL;states[0].model_generation++;prepared();assert(!equivalent(NULL));
 vx_paged_kv_telemetry(caches[0],&after);assert(!memcmp(&before,&after,sizeof(before)));
 assert(!memcmp(saved,pool,sizeof(pool)));
}
int main(void) {
 unsetenv("VOLVOXAI_ROW_DEBUG");
 VxPagedKVOptions options={.slots=2,.page_tokens=2,.slot_token_capacity=16,.max_pages=16,.policy=VX_PAGED_KV_POLICY_PAGED};
 for(int i=0;i<2;i++){caches[i]=vx_paged_kv_create(&options);assert(caches[i]);}
 selection_edges();random_closures();guard_edges();memory_lifecycle();numerical_mapping();
 for(int i=0;i<2;i++){g_current_engine_state=&states[i];vx_paged_unbind_locked();vx_engine_state_release_graph_metadata(&states[i]);vx_paged_kv_destroy(caches[i]);}
 printf("PASS paged domain %lu oracle comparisons %lu warm zero-lookup checks sizeof_plan=%zu sizeof_state=%zu\n",equivalence_checks,warm_checks,sizeof(VxPagedDomainPlan),sizeof(VxEngineState));
}
'''

MEMORY_PROBE = r'''
#include <stdio.h>
#include "runtime_state.c"
int main(void) {
 const size_t cases[][2]={{0,0},{1,0},{3,5},{8,8},{511,484},{650,620},{4096,2048}};
 for(size_t i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
  size_t bytes; if(vx_engine_state_graph_metadata_bytes_for(cases[i][0],cases[i][1],&bytes))return 1;
  printf("%zu %zu %zu %zu\n",cases[i][0],cases[i][1],bytes,sizeof(VxEngineState));
 }
}
'''


def compile_fixture(unit, binary, profile, cuda, source_root=SOURCE_ROOT, sanitize=False, paged=False):
    compiler = shutil.which("cc")
    flags = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
             "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
             f"-DVOLVOXAI_ENABLE_TRAINING={profile}", f"-DVOLVOXAI_ENABLE_CUDA={cuda}",
             "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_VULKAN=0",
             "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-no-pie", "-O1"]
    flags += [f"-I{source_root / 'native/src/runtime'}", f"-I{SOURCE_ROOT / 'tools/tests/fixtures'}"]
    flags += [f"-I{ROOT / path}" for path in INCLUDES]
    flags += [str(unit)]
    if paged:
        flags += [str(ROOT / "native/src/runtime/paged_kv.c")]
    flags += ["-lm", "-o", str(binary)]
    subprocess.run(flags, text=True, capture_output=True, check=True)


class PagedDomainPlanTests(unittest.TestCase):
    def test_production_oracle_and_lifecycle(self):
        if not shutil.which("cc"):
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="volvox-paged-domain-") as temporary:
            unit = Path(temporary) / "test.c"
            unit.write_text(PREAMBLE)
            for profile, cuda in ((0, 0), (1, 0), (0, 1), (1, 1)):
                binary = Path(temporary) / f"test-{profile}-{cuda}"
                compile_fixture(unit, binary, profile, cuda, paged=True)
                result = subprocess.run([str(binary)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("PASS paged domain", result.stdout)
                print(result.stdout.strip())

    def test_exact_metadata_accounting(self):
        if not shutil.which("cc"):
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="volvox-paged-budget-") as temporary:
            unit = Path(temporary) / "test.c"
            unit.write_text(MEMORY_PROBE)
            for profile, cuda in ((0, 0), (1, 0), (0, 1), (1, 1)):
                output = []
                # In the task overlay the baseline is the frozen product source.
                # Integrated runs still test exact runtime layout and aligned
                # inventory; the independent delta table lives in the receipt.
                roots = (ROOT, SOURCE_ROOT) if ROOT != SOURCE_ROOT else (SOURCE_ROOT,)
                for index, source_root in enumerate(roots):
                    binary = Path(temporary) / f"budget-{profile}-{cuda}-{index}"
                    compile_fixture(unit, binary, profile, cuda, source_root=source_root)
                    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
                    output.append([list(map(int, line.split())) for line in result.stdout.splitlines()])
                if len(output) == 2:
                    for old, new in zip(*output):
                        self.assertEqual(old[:2], new[:2])
                        minimum = 5 * new[0] + new[1]
                        self.assertGreaterEqual(new[2] - old[2], minimum)
                        self.assertLessEqual(new[2] - old[2], minimum + 16)
                        self.assertGreater(new[3], old[3])
                print(json.dumps({"training": profile, "cuda": cuda, "layout_bytes": output[-1]}))


NATIVE_PAGED_ROWS_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "sequence_runtime.h"
#include "decode_projection.h"
typedef struct { T* tensor; size_t row_bytes; } VxCacheTensor;
typedef struct { void* host; size_t bytes; uint32_t bits[64]; } Device;
static Device devices[16]; static unsigned device_count, transfers, uploads;
static VxEngineState state; static T tensors[4]; static Node nodes[1];
static GraphNodeFusion fusions[1]; static Ref ports[2];
static const char* paged_name="kv";
VxEngineState* vx_engine_state_current(void) { return &state; }
T* t_find(const char* name) {
    for(unsigned i=0;i<4;i++) if(!strcmp(name,tensors[i].name)) return &tensors[i];
    return NULL;
}
static T* nin(Node* node,VxPortKind port) {
    for(int i=0;i<node->nin;i++) if(node->ins[i].port==port) return t_find(node->ins[i].name);
    return NULL;
}
int volvoxai_engine_tensor_is_model_weight_locked(const char* name) { return !strcmp(name,"weight"); }
int vx_incremental_tensor_dirty_before_node_locked(const T* tensor,int index) {
    (void)index; return !volvoxai_engine_tensor_is_model_weight_locked(tensor->name);
}
int vx_paged_bound_locked(void) { return !g_decode_rows.unpaged; }
int vx_paged_slot_pages_locked(int slot,VxDecodeSlotPages* out) { assert(slot==0); *out=g_decode_rows.pages[0]; return 1; }
int vx_paged_tensor_locked(const T* tensor) { return !strcmp(tensor->name,paged_name); }
int vx_sequence_rope_parameters(Node* node,const T* input,const T* output,VxSequenceRoPE* p) {
    (void)node;(void)input;(void)output;(void)p;return 0;
}
static Device* device(const void* host,size_t bytes) {
    for(unsigned i=0;i<device_count;i++) if(devices[i].host==host && bytes<=devices[i].bytes) return &devices[i];
    assert(!"unregistered device span"); return NULL;
}
static Device* register_device(void* host,size_t bytes) {
    assert(device_count<16 && bytes<=sizeof(devices[0].bits));
    Device* result=&devices[device_count++]; result->host=host; result->bytes=bytes;
    return result;
}
void vx_runtime_backend_mark_host(const void* host,size_t bytes,int weight) {
    assert(!weight); Device* target=device(host,bytes); memcpy(target->bits,host,bytes); ++uploads;
}
int vx_runtime_backend_row_index_transfer(const void* source,size_t source_bytes,const int32_t* indices,
        uint32_t rows,void* destination,size_t destination_bytes,uint32_t width,uint32_t full,uint32_t scatter) {
    Device *from=device(source,source_bytes),*to=device(destination,destination_bytes);
    const int32_t* selected=(const int32_t*)device(indices,(size_t)rows*4)->bits;
    assert(source!=destination && width && rows && full && (uint64_t)full*width*4<= (scatter?destination_bytes:source_bytes));
    assert((uint64_t)rows*width*4<= (scatter?source_bytes:destination_bytes));
    for(uint32_t r=0;r<rows;r++) {
        assert(selected[r]>=0 && (uint32_t)selected[r]<full);
        memcpy(to->bits+(scatter?(uint32_t)selected[r]:r)*width,
               from->bits+(scatter?r:(uint32_t)selected[r])*width,(size_t)width*4);
    }
    ++transfers; return 1;
}
'''

NATIVE_PAGED_ROWS_FIXTURE = r'''
static T tensor(const char* name,void* host,VxDataType dtype,int width) {
    T result={.dtype=dtype,.elem_size=4,.ndim=3,.shape={1,8,width},.numel=8*width,.data=host};
    strcpy(result.name,name);return result;
}
static void preserved_host(const uint32_t* host) {
    for(unsigned i=0;i<32;i++) assert(host[i]==0xababababu);
}
int main(void) {
    uint32_t bank_host[32],ordinary_host[32],weight[4]={0},staging[12][32],metadata[8];
    memset(bank_host,0xab,sizeof(bank_host));memset(ordinary_host,0xab,sizeof(ordinary_host));
    memset(staging,0,sizeof(staging));memset(metadata,0,sizeof(metadata));
    state.loaded=1;state.backend=VX_BACKEND_KIND_VULKAN;state.use_vulkan=1;
    state.decode_row_stage_reserved=1;state.decode_numeric_stage_count=12;
    for(unsigned i=0;i<12;i++) {
        state.decode_numeric_stage[i]=(unsigned char*)staging[i];state.decode_numeric_stage_bytes[i]=sizeof(staging[i]);
        register_device(staging[i],sizeof(staging[i]));
    }
    state.decode_metadata_stage=(unsigned char*)metadata;state.decode_metadata_stage_bytes=sizeof(metadata);
    register_device(metadata,sizeof(metadata));
    Device* bank=register_device(bank_host,sizeof(bank_host));
    Device* ordinary=register_device(ordinary_host,sizeof(ordinary_host));
    for(unsigned i=0;i<32;i++) {bank->bits[i]=0x3f000000u+i;ordinary->bits[i]=0x40000000u+i;}
    bank->bits[17]=0x7fc12345u;bank->bits[18]=0x80000000u;
    state.tensors=tensors;state.tensor_count=4;state.nodes=nodes;state.node_count=1;state.node_fusion=fusions;
    tensors[0]=tensor("kv",bank_host,T_F32,4);
    tensors[1]=tensor("ordinary",ordinary_host,T_F32,4);
    tensors[2]=(T){.dtype=T_F32,.elem_size=4,.ndim=2,.shape={4,4},.numel=16,.data=(float*)weight};strcpy(tensors[2].name,"weight");
    tensors[3]=tensor("out",ordinary_host,T_F32,4);
    nodes[0]=(Node){.operator_kind=VX_OP_LINEAR,.ins=ports,.nin=2,.nout=1};strcpy(nodes[0].out,"out");
    ports[0].port=VX_PORT_INPUT;strcpy(ports[0].name,"ordinary");
    ports[1].port=VX_PORT_WEIGHT;strcpy(ports[1].name,"weight");
    int page_table[]={2,0,3,1};int position=5;
    VxDecodeSlotPages pages={.page_table=page_table,.page_tokens=2,.pages_per_slot=4};
    assert(vx_decode_row_set_init(&state.decode_rows,1,&position,&pages)==VX_DECODE_ROW_SET_OK);state.decode_slots=1;
    VxDecodeRowSet rows_before=state.decode_rows;
    VxModernRowProjection p;
    assert(vx_native_plain_row_projection(&nodes[0],position,&p));
    state.use_vulkan=0;state.use_cuda=1;
    assert(!vx_native_plain_row_projection(&nodes[0],position,&p));
    state.use_cuda=0;state.use_vulkan=1;
    /* A prefix's logical rows [0..5] occupy physical [4,5,0,1,6,7]. */
    VxModernRowSpan span;
    assert(vx_modern_row_span(&tensors[0],1,0,6,0,&span));
    assert(vx_modern_device_transfer(&span,0));
    const int expected[]={4,5,0,1,6,7};
    Device* stage=device(staging[0],sizeof(staging[0]));
    for(unsigned r=0;r<6;r++) for(unsigned c=0;c<4;c++) assert(stage->bits[r*4+c]==bank->bits[expected[r]*4+c]);
    assert(stage->bits[1]==0x7fc12345u && stage->bits[2]==0x80000000u);
    preserved_host(bank_host);
    uint32_t before[32];memcpy(before,bank->bits,sizeof(before));
    assert(vx_modern_row_span(&tensors[0],1,5,1,3,&span));
    Device* output=device(staging[3],sizeof(staging[3]));
    for(unsigned c=0;c<4;c++) output->bits[c]=0x43000000u+c;
    assert(vx_modern_device_transfer(&span,1));
    for(unsigned i=0;i<32;i++) assert(bank->bits[i]==(i>=28?0x43000000u+i-28:before[i]));
    preserved_host(bank_host);
    /* Changing a recycled page table changes only paged addressing. */
    page_table[2]=1;
    assert(vx_modern_device_transfer(&span,1));
    for(unsigned c=0;c<4;c++) assert(bank->bits[12+c]==output->bits[c]);
    assert(vx_modern_row_span(&tensors[1],1,5,1,0,&span));
    assert(vx_modern_device_transfer(&span,0));
    for(unsigned c=0;c<4;c++) assert(stage->bits[c]==ordinary->bits[20+c]);
    /* Mapping/capacity failures issue no uploads or numerical transfers. */
    unsigned commands=transfers,marks=uploads;
    assert(vx_modern_row_span(&tensors[0],1,5,1,3,&span));
    page_table[2]=-1;assert(!vx_modern_device_transfer(&span,1));
    page_table[2]=4;assert(!vx_modern_device_transfer(&span,1));
    page_table[2]=1;state.decode_numeric_stage_bytes[3]=12;assert(!vx_modern_device_transfer(&span,1));
    state.decode_numeric_stage_bytes[3]=128;state.decode_metadata_stage_bytes=20;
    assert(vx_modern_row_span(&tensors[0],1,0,6,0,&span));assert(!vx_modern_device_transfer(&span,0));
    state.decode_metadata_stage_bytes=32;assert(transfers==commands && uploads==marks);
    assert(!memcmp(&rows_before,&state.decode_rows,sizeof(rows_before)));
    /* Initialize repacks and COW copies retain device bits; a host mirror
     * filled with unrelated values is never a source of activation bytes. */
    VxCacheTensor cached={.tensor=&tensors[0],.row_bytes=16};uint32_t placeholder[32];
    memset(placeholder,0xcc,sizeof(placeholder));int initial[]={0,1},repacked[]={4,5};
    memcpy(before,bank->bits,sizeof(before));
    assert(!vx_cache_transfer(&cached,initial,2,placeholder,0,0));
    assert(!vx_cache_transfer(&cached,repacked,2,placeholder,1,0));
    for(unsigned c=0;c<8;c++) assert(bank->bits[16+c]==before[c]);
    for(unsigned c=0;c<32;c++) assert(placeholder[c]==0xccccccccu);
    int cow_source[]={4,5},cow_target[]={2,3};
    assert(!vx_cache_transfer(&cached,cow_source,2,placeholder,0,0));
    assert(!vx_cache_transfer(&cached,cow_target,2,placeholder,1,0));
    for(unsigned c=0;c<8;c++) assert(bank->bits[8+c]==before[c]);
    memcpy(before,bank->bits,sizeof(before));memset(placeholder,0,sizeof(placeholder));
    assert(!vx_cache_transfer(&cached,cow_source,2,placeholder,1,1));
    for(unsigned c=0;c<32;c++) assert(bank->bits[c]==(c>=16 && c<24?0:before[c]));
    preserved_host(bank_host);
    commands=transfers;marks=uploads;int invalid[]={8};
    assert(vx_cache_transfer(&cached,invalid,1,placeholder,0,0));
    state.decode_numeric_stage_bytes[0]=28;assert(vx_cache_transfer(&cached,initial,2,placeholder,0,0));
    state.decode_numeric_stage_bytes[0]=128;state.decode_metadata_stage_bytes=4;
    assert(vx_cache_transfer(&cached,initial,2,placeholder,0,0));state.decode_metadata_stage_bytes=32;
    tensors[0].dtype=T_I32;assert(vx_cache_transfer(&cached,initial,2,placeholder,0,0));tensors[0].dtype=T_F32;
    assert(transfers==commands && uploads==marks);
    /* A second engine-state scope cannot see the first owner's staging. */
    VxEngineState saved=state;memset(&state,0,sizeof(state));state.use_opengl=1;
    assert(vx_cache_transfer(&cached,initial,2,placeholder,0,0));state=saved;
    assert(transfers==commands && !memcmp(&rows_before,&state.decode_rows,sizeof(rows_before)));
    vx_decode_row_set_dispose(&state.decode_rows);
    puts("PASS production portable paged prefix/write, repack/COW/erase bits, bounds and context ownership");
}
'''


class NativePagedRowTests(unittest.TestCase):
    def test_production_paged_transfers(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        runtime = root / "native/src/runtime"
        rows = (runtime / "engine_runtime_row_modern.inc").read_text()
        unit = NATIVE_PAGED_ROWS_PREAMBLE
        for name in ("vx_native_row_selected", "vx_modern_row_span", "vx_native_plain_row_projection", "vx_modern_device_transfer"):
            unit += function_source(rows, name)
        unit += function_source((runtime / "public_api_decode_cache.inc").read_text(), "vx_cache_transfer")
        unit += NATIVE_PAGED_ROWS_FIXTURE
        with tempfile.TemporaryDirectory(prefix="volvox-native-paged-rows-") as directory:
            source, binary = Path(directory) / "rows.c", Path(directory) / "rows"
            source.write_text(unit)
            built = subprocess.run([compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2",
                "-Wall", "-Wextra", "-Werror", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=0", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                "-DVOLVOXAI_ENABLE_VULKAN=1", "-DVOLVOXAI_ENABLE_OPENGL=1", "-DVOLVOXAI_ENABLE_METAL=0",
                *["-I" + str(root / path) for path in ("native/include", "native/third_party", "native/src",
                    "native/src/runtime", "native/src/kernels", "native/src/backends")], str(source),
                str(runtime / "decode_row_set.c"), str(runtime / "attention_mask.c"), "-lm", "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS production portable paged prefix/write", checked.stdout)


NATIVE_DENSE_ROWS_PREAMBLE = r'''
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "vx_lifecycle.h"
#include "llm_workspace.h"
static VxEngineState* current;
static size_t live_bytes, span_limit = 4096;
static int locks, configurations, forwarded;
VxEngineState* vx_engine_state_current(void) { return current; }
VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* state) {
    VxEngineStateScope scope = {.previous=current, .bound=state}; current=state; return scope;
}
void vx_engine_state_scope_leave(VxEngineStateScope scope) { current=scope.previous; }
static VxBackendKind vx_backend_manager_current(void) { return current->backend; }
int volvoxai_engine_model_route_lease_active(void) { return 0; }
void volvoxai_engine_model_lock(void) { ++locks; }
void volvoxai_engine_model_unlock(void) { --locks; }
void volvoxai_engine_metadata_lock(void) { ++locks; }
void volvoxai_engine_metadata_unlock(void) { --locks; }
int volvoxai_engine_tensor_is_model_weight_locked(const char* name) { (void)name; return 0; }
static T* nin(Node* node, VxPortKind port) { (void)node; (void)port; return NULL; }
static void* vx_arena_allocate(size_t bytes) { live_bytes+=bytes; return calloc(1,bytes); }
static void vx_arena_release(void* pointer) { free(pointer); }
'''

NATIVE_DENSE_ROWS_MODEL = r'''
struct VxModel { cJSON* logical_graph; VxDeclaredTensor* inputs; size_t input_count; };
struct VxCompiledModel { VxModel* model; VxBackendKind builtin_backend; };
struct VxExecutionContext {
    VxEngineState* engine_state; VxCompiledModel* compiled;
    VxDeclaredTensor* logical_tensors; size_t logical_tensor_count;
    VxDecodeRowMode decode_row_mode; uint32_t decode_slots;
    void* adapter; uint64_t allocated_bytes;
};
static const cJSON* vx_execution_document(const VxModel* model) {
    return cJSON_GetObjectItemCaseSensitive(model->logical_graph,"execution");
}
static const char* vx_execution_string(const cJSON* object,const char* key) {
    return cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(object,key));
}
static int vx_native_gpu_backend(VxBackendKind backend) {
    return backend==VX_BACKEND_KIND_CUDA || backend==VX_BACKEND_KIND_VULKAN ||
        backend==VX_BACKEND_KIND_OPENGL || backend==VX_BACKEND_KIND_METAL;
}
static int vx_adapter_is_base(const void* adapter) { return adapter==NULL; }
static int vx_context_cuda_row_host_masks_proven(const VxExecutionContext* context) {
    (void)context; return 1;
}
/* The hardware and allocator are test doubles. Production proof arithmetic,
 * stage capacities, transactional reservation and negotiation are verbatim. */
int volvoxai_engine_configure_dynamic_shape_domain(
        const VolvoxAIEngineMaximumTensor* maximums,size_t count) {
    assert(maximums && count==2); ++configurations;
    static size_t offsets[2]={0,32}, capacities[2]={32,128};
    VxDynamicShapeMaximumLayout* layout=&current->dynamic_shape_maximum_layout;
    layout->configured=1; layout->tensor_count=2;
    layout->physical_offsets=offsets; layout->physical_capacities=capacities;
    layout->physical_span_count=current->shape_policy.retain_activations?2:1;
    layout->arena_bytes=current->shape_policy.retain_activations?160:32;
    return 0;
}
int volvoxai_engine_forward(void) { ++forwarded; return 0; }
int volvoxai_engine_demote_preloaded_logical_tensors_locked(void) { return 0; }
int volvoxai_engine_retain_preloaded_model_weights_locked(void) { return 0; }
int volvoxai_engine_reserve_dynamic_shape_domain(void) {
    current->dynamic_arena_capacity_bytes=current->dynamic_shape_maximum_layout.arena_bytes;
    return 0;
}
'''

NATIVE_DENSE_ROWS_HARDWARE = r'''
static int vx_native_gpu_query_domain_limits(VxBackendKind backend,VxNativeGpuDomainLimits* limits) {
    (void)backend;
    *limits=(VxNativeGpuDomainLimits){.maximum_buffer_bytes=4096,
        .maximum_uniform_bytes=4096,.maximum_total_span_bytes=span_limit,
        .fixed_compute_allocation_bytes=4096,.fixed_staging_allocation_bytes=4096,
        .maximum_scratch_bytes=4096,.storage_alignment=16,
        .maximum_grid={65535,65535,65535},.maximum_block={256,256,64},
        .maximum_threads_per_block=256,.maximum_tensor_slots=128,
        .maximum_storage_bindings=16,.maximum_uniform_bindings=1};
    return 0;
}
typedef struct { unsigned char* tensors; } VxHybridRowPlan;
static int refused_node, probe_calls, last_probe;
static int incremental_plan_build_locked(void) { return 0; }
static int hybrid_row_plan_init(VxHybridRowPlan* plan) { plan->tensors=calloc((size_t)g_nt,1); return 0; }
static void hybrid_row_plan_reset(VxHybridRowPlan* plan) { memset(plan->tensors,0,(size_t)g_nt); }
static void hybrid_row_plan_deinit(VxHybridRowPlan* plan) { free(plan->tensors); }
static int hybrid_row_plan_expand_locked(int row,VxHybridRowPlan* plan) {
    ++probe_calls; last_probe=row; assert(plan->tensors[0]); return !refused_node;
}
static int device_row_backend_selected(void) { return vx_native_gpu_backend(current->backend); }
'''

NATIVE_DENSE_ROWS_FIXTURE = r'''
static void clear_state(VxEngineState* state,VxBackendKind backend,T* token) {
    for(unsigned i=0;i<VX_DECODE_NUMERIC_STAGE_COUNT;i++) free(state->decode_numeric_stage[i]);
    free(state->decode_metadata_stage); memset(state,0,sizeof(*state));
    state->backend=backend; state->loaded=1; state->tensor_count=1; state->tensors=token;
    state->decode_slots=1; live_bytes=0; configurations=forwarded=0;
}
static int prove(VxEngineState* state,VxDeclaredTensor* logical,VxBackendKind backend,
        uint32_t slots,uint64_t* arena,uint64_t* spans,uint64_t* count) {
    VxNativeGpuDomainLimits limits; uint64_t qg,ql,modern,scratch;
    int status=vx_native_gpu_maximum_layout_proof(state,logical,2,backend,slots,0,
        &limits,arena,spans,count,&qg,&ql,&modern,&scratch);
    if(!status) assert(!qg && !ql && !modern && !scratch);
    return status;
}
int main(void) {
    VxDeclaredTensor active={.name="active_lengths",.dtype=VX_DTYPE_I32,.rank=1,
        .minimums={1},.maximums={1},.maximum_byte_size=4};
    VxDeclaredTensor logical[2]={
        {.name="token_ids",.dtype=VX_DTYPE_I32,.rank=2,.maximums={1,8},.maximum_byte_size=32},
        {.name="hidden",.dtype=VX_DTYPE_F32,.rank=3,.maximums={1,8,4},.maximum_byte_size=128}};
    VxModel model={.inputs=&active,.input_count=1};
    model.logical_graph=cJSON_Parse("{\"execution\":{\"incremental\":{\"active_lengths_input\":\"active_lengths\"}},\"nodes\":[{\"opType\":\"Linear\"}]}");
    assert(model.logical_graph && !vx_native_gpu_incremental_workspace_domain_required(&model));
    uint32_t slots=0;
    assert(vx_native_gpu_cuda_decoder_slots_proven(&model,&slots) && slots==1);
    assert(!vx_native_gpu_portable_decoder_slots_proven(&model,VX_BACKEND_KIND_NATIVE_CPU,&slots));
    assert(!vx_native_gpu_portable_decoder_slots_proven(&model,VX_BACKEND_KIND_CUDA,&slots));
    assert(!vx_native_gpu_portable_decoder_slots_proven(&model,VX_BACKEND_KIND_WEBGPU,&slots));
    active.maximums[0]=2;
    assert(!vx_native_gpu_portable_decoder_slots_proven(&model,VX_BACKEND_KIND_VULKAN,&slots));
    active.maximums[0]=1; active.dtype=VX_DTYPE_F32;
    assert(!vx_native_gpu_portable_decoder_slots_proven(&model,VX_BACKEND_KIND_OPENGL,&slots));
    active.dtype=VX_DTYPE_I32;
    VxEngineState state={0}; current=&state;
    T token={.dtype=T_I32,.elem_size=4,.ndim=2,.shape={1,1},.numel=1,.is_graph_input=1};
    const VxBackendKind backends[]={VX_BACKEND_KIND_VULKAN,VX_BACKEND_KIND_OPENGL,VX_BACKEND_KIND_METAL};
    for(unsigned b=0;b<3;b++) {
        VxBackendKind backend=backends[b]; clear_state(&state,backend,&token);
        assert(vx_native_gpu_portable_decoder_slots_proven(&model,backend,&slots) && slots==1);
        /* Before reservation the original [1,1] token has no row-one probe. */
        probe_calls=0; assert(!hybrid_model_has_row_closure_locked() && !probe_calls);
        uint64_t arena=0,spans=0,count=0;
        assert(!prove(&state,logical,backend,slots,&arena,&spans,&count));
        assert(state.shape_policy.retain_activations && !state.dynamic_shape_maximum_layout.modern_incremental_rows);
        /* 160 retained bytes + twelve independent 16-byte stages + 68-byte
         * metadata; the device metadata span is independently aligned. */
        assert(arena==420 && spans==432 && count==15);
        span_limit=431;
        assert(prove(&state,logical,backend,slots,&arena,&spans,&count)==VX_NATIVE_GPU_LAYOUT_ARENA_EXHAUSTED);
        span_limit=4096;
        clear_state(&state,backend,&token);
        VxCompiledModel compiled={.model=&model,.builtin_backend=backend};
        VxExecutionContext context={.engine_state=&state,.compiled=&compiled,
            .logical_tensors=logical,.logical_tensor_count=2,.decode_row_mode=VX_DECODE_ROW_REQUIRED,
            .decode_slots=1,.allocated_bytes=37};
        assert(vx_context_prepare_native_gpu_dynamic_reservation(&context)==VX_STATUS_OK);
        assert(state.decode_row_stage_reserved && state.decode_numeric_stage_count==12);
        assert(state.shape_policy.retain_activations && !state.dynamic_shape_maximum_layout.modern_incremental_rows);
        assert(live_bytes==260 && context.allocated_bytes==37+420 && configurations==1 && forwarded==1);
        probe_calls=0; assert(hybrid_model_has_row_closure_locked() && last_probe==0 && probe_calls==1);
        refused_node=1; assert(!hybrid_model_has_row_closure_locked()); refused_node=0;
        clear_state(&state,backend,&token); context.decode_slots=2;
        assert(vx_context_prepare_native_gpu_dynamic_reservation(&context)==VX_STATUS_BACKEND_UNSUPPORTED);
        assert(!live_bytes && !configurations && !forwarded && !state.decode_row_stage_reserved);
        context.decode_slots=1; context.decode_row_mode=VX_DECODE_ROW_DISABLED; context.allocated_bytes=37;
        assert(vx_context_prepare_native_gpu_dynamic_reservation(&context)==VX_STATUS_OK);
        assert(!state.decode_row_stage_reserved && !state.shape_policy.retain_activations && context.allocated_bytes==69);
    }
    /* CUDA retains its four-stage capacity and existing fixed slot proof. */
    clear_state(&state,VX_BACKEND_KIND_CUDA,&token);
    uint64_t arena=0,spans=0,count=0;
    assert(!prove(&state,logical,VX_BACKEND_KIND_CUDA,1,&arena,&spans,&count));
    assert(arena==292 && spans==304 && count==7);
    clear_state(&state,VX_BACKEND_KIND_NATIVE_CPU,&token);
    cJSON_Delete(model.logical_graph);
    assert(!locks);
    puts("PASS production dense portable proof, retained reservation, singleton negotiation and unchanged CUDA4");
}
'''

NATIVE_DENSE_ROWS_LEGACY_FIXTURE = NATIVE_DENSE_ROWS_FIXTURE[:NATIVE_DENSE_ROWS_FIXTURE.index("int main(void)")] + r'''
int main(void) {
    VxDeclaredTensor active={.name="active_lengths",.dtype=VX_DTYPE_I32,.rank=1,
        .minimums={1},.maximums={1},.maximum_byte_size=4};
    VxDeclaredTensor logical[2]={
        {.name="token_ids",.dtype=VX_DTYPE_I32,.rank=2,.maximums={1,8},.maximum_byte_size=32},
        {.name="hidden",.dtype=VX_DTYPE_F32,.rank=3,.maximums={1,8,4},.maximum_byte_size=128}};
    VxModel model={.inputs=&active,.input_count=1};
    model.logical_graph=cJSON_Parse("{\"execution\":{\"incremental\":{\"active_lengths_input\":\"active_lengths\"}},\"nodes\":[{\"opType\":\"Linear\"}]}");
    uint32_t slots=0;
    assert(vx_native_gpu_portable_decoder_slots_proven(&model,VX_BACKEND_KIND_VULKAN,&slots) && slots==1);
    VxEngineState state={0}; current=&state;
    T token={.dtype=T_I32,.elem_size=4,.ndim=2,.shape={1,1},.numel=1,.is_graph_input=1};
    clear_state(&state,VX_BACKEND_KIND_VULKAN,&token);
    assert(vx_backend_manager_current()==VX_BACKEND_KIND_VULKAN && !nin(NULL,VX_PORT_INPUT));
    uint64_t arena=0,spans=0,count=0;
    /* An unavailable portable backend must retain the previous generic
     * proof control flow in CPU/WASM builds, rather than add dense staging. */
    assert(!prove(&state,logical,VX_BACKEND_KIND_VULKAN,slots,&arena,&spans,&count));
    assert(!state.shape_policy.retain_activations && arena==32 && spans==32 && count==1);
    assert(!hybrid_model_has_row_closure_locked() && !probe_calls);
    assert(vx_context_prepare_native_gpu_dynamic_reservation(NULL)==VX_STATUS_INVALID_ARGUMENT);
    clear_state(&state,VX_BACKEND_KIND_NATIVE_CPU,&token);
    cJSON_Delete(model.logical_graph);
    puts("PASS production feature-off legacy maximum layout preserved");
}
'''


class NativeDenseRowDomainTests(unittest.TestCase):
    def test_production_dense_row_domain(self):
        self._check_fixture(NATIVE_DENSE_ROWS_FIXTURE, portable=True, marker="PASS production dense portable proof")

    def test_feature_off_preserves_legacy_layout(self):
        self._check_fixture(NATIVE_DENSE_ROWS_LEGACY_FIXTURE, portable=False, marker="PASS production feature-off legacy")

    def _check_fixture(self, fixture, *, portable, marker):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        runtime = root / "native/src/runtime"
        proof = (runtime / "public_api_native_gpu_domain.inc").read_text()
        api = (runtime / "public_api.c").read_text()
        start = api.index("typedef struct VxDeclaredTensor {")
        declared = api[start:api.index("} VxDeclaredTensor;", start) + len("} VxDeclaredTensor;")]
        start = proof.index("typedef struct VxNativeGpuDomainLimits {")
        limits = proof[start:proof.index("static int vx_native_gpu_query_domain_limits", start)]
        unit = NATIVE_DENSE_ROWS_PREAMBLE + declared + NATIVE_DENSE_ROWS_MODEL
        for name in ("vx_native_gpu_incremental_workspace_domain_required",
                     "vx_native_gpu_cuda_decoder_slots_proven", "vx_native_gpu_portable_decoder_slots_proven"):
            unit += function_source(proof, name)
        unit += limits + NATIVE_DENSE_ROWS_HARDWARE
        model = (runtime / "engine_runtime_model.inc").read_text()
        arena = (runtime / "arena.c").read_text()
        unit += function_source(model, "volvoxai_engine_decode_row_stage_capacities")
        unit += function_source(arena, "volvoxai_engine_reserve_decode_row_stage")
        for name in ("vx_native_gpu_slot_capacity_proven", "vx_native_gpu_accumulate_resource_bytes",
                     "vx_native_gpu_align_resource_bytes", "vx_native_gpu_vulkan_scratch_capacity_proven",
                     "vx_native_gpu_maximum_layout_proof"):
            # Resource helper declarations precede their definitions.
            unit += function_source(proof[proof.rindex("static int " + name + "("):], name)
        unit += function_source((runtime / "public_api_execution_context.inc").read_text(),
                                "vx_context_prepare_native_gpu_dynamic_reservation")
        incremental = (runtime / "incremental_runtime.c").read_text()
        unit += function_source(incremental, "reserved_device_row_selected")
        unit += function_source(incremental, "hybrid_model_has_row_closure_locked")
        unit += fixture
        with tempfile.TemporaryDirectory(prefix="volvox-native-dense-rows-") as directory:
            source, binary = Path(directory) / "domain.c", Path(directory) / "domain"
            source.write_text(unit)
            command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=0", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                "-DVOLVOXAI_ENABLE_VULKAN=" + ("1" if portable else "0"),
                "-DVOLVOXAI_ENABLE_OPENGL=" + ("1" if portable else "0"), "-DVOLVOXAI_ENABLE_METAL=0",
                *["-I" + str(root / path) for path in ("native/include", "native/third_party", "native/src",
                    "native/src/runtime", "native/src/kernels", "native/src/backends")], str(source),
                str(root / "native/third_party/cJSON.c"), str(root / "native/src/generated/operator_vocabulary.c"),
                "-o", str(binary)]
            built = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn(marker, checked.stdout)


if __name__ == "__main__":
    unittest.main()

