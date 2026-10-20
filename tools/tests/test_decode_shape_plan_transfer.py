"""Compare first-prefill transactions with frozen production resolver calls.

The fixture supplies a seven-tensor shape graph and CPU backend services.
Production cache ownership, paged storage, input value checks, reference COW,
commit, cancellation and settlement run unchanged. This is a host transaction
test; physical model numerics and latency require separate CUDA qualification.
"""
from pathlib import Path
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import unittest

from tools.tests.test_incremental_prepared_bindings import FIXTURE
from tools.tests import function_source

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "native/src/runtime"
FIXTURES = ROOT / "tools/tests/fixtures"




def prepared_plan_type():
    public = (RUNTIME / "public_api.c").read_text()
    end = public.index("} VxPreparedShapePlan;") + len("} VxPreparedShapePlan;")
    return public[public.rfind("typedef struct {", 0, end):end] + "\n"


def transaction_source(candidate):
    cache = (RUNTIME / "public_api_decode_cache.inc").read_text()
    commit = (RUNTIME / "public_api_execution_context.inc").read_text()
    reference = (FIXTURES / "decode_shape_plan_transfer_reference.inc").read_text()
    unit = (FIXTURES / "decode_shape_plan_transfer_preamble.h").read_text()
    unit += prepared_plan_type()
    unit += cache[:cache.index("typedef struct {\n    VxExecutionContext* context;")]
    for name in ("vx_resolved_tensor_find", "vx_staged_binding_find",
                 "vx_native_gpu_preflight_i32_binding", "vx_native_gpu_preflight_finite_binding",
                 "vx_moe_preflight_route_binding"):
        unit += function_source(commit, name)
    values = (RUNTIME / "public_api_value_validation.inc").read_text()
    for name in ("vx_public_value_bounds", "vx_native_gpu_preflight_public_values"):
        unit += function_source(values, name)
    for name in ("vx_tensor_reference_needs_snapshot", "vx_prepare_tensor_references",
                 "vx_staged_binding_reused", "vx_compact_changed_bindings"):
        unit += function_source(commit, name)
    unit += function_source((RUNTIME / "public_api_resolved_shape_plan.inc").read_text(),
                            "vx_resolved_plan_create")
    if candidate:
        unit += function_source(commit, "vx_cache_prepared_shape_enabled")
        for name in ("vx_cache_prepared_shape_clear", "vx_cache_prepared_shape_take"):
            unit += function_source(cache, name)
    changed_cache = {"vx_cache_free", "vx_cache_configure_impl", "vx_cache_prepare_initialize"}
    for name in ("vx_cache_storage_free", "vx_cache_prefix_free", "vx_cache_free",
                 "vx_context_decode_cache_clear", "vx_cache_scratch", "vx_cache_transfer",
                 "vx_cache_tensor_layout", "vx_cache_note_erase", "vx_cache_consumers_validate",
                 "vx_cache_configure_impl", "vx_cache_configure", "vx_cache_prepare_initialize"):
        source = reference if not candidate and name in changed_cache else cache
        unit += function_source(source, name)
    unit += "static void vx_context_clear_decode_tracking(VxExecutionContext* c)" \
            "{vx_context_decode_cache_clear(c);}\n"
    unit += function_source(commit if candidate else reference, "vx_commit_builtin_bindings")
    unit += function_source((RUNTIME / "public_api_execute_result.inc").read_text(),
                            "vx_context_run_prepare")
    driver = (RUNTIME / "public_api_execution_driver.inc").read_text()
    for name in ("vx_execution_settle", "vx_execution_context_unreserve"):
        unit += function_source(driver, name)
    return unit + (FIXTURES / "decode_shape_plan_transfer_cases.c").read_text()


METADATA_CHECKS = r'''
int main(void) {
    VxDeclaredTensor inputs[4]={{"ids",2},{"pos",1},{"x",3},{"y",3}};
    VxModel model={4,inputs};
    size_t cases=0;uint64_t old=0,current=0;
    const uint64_t counts[]={7,491,627};
    for(size_t i=0;i<3;i++) {
        assert(old_peak(&model,counts[i],counts[i]+12,counts[i]/2,&old));
        assert(current_peak(&model,counts[i],counts[i]+12,counts[i]/2,&current));
        assert(current==old+sizeof(VxPreparedShapePlan));cases++;
    }
    assert(!current_peak(NULL,7,7,7,&current));cases++;
    assert(!current_peak(&model,0,7,7,&current));cases++;
    assert(!current_peak(&model,UINT64_MAX,7,7,&current));cases++;
    assert(!current_peak(&model,7,UINT64_MAX,7,&current));cases++;
    assert(!current_peak(&model,7,7,UINT64_MAX,&current));cases++;
    assert(!current_peak(&model,7,7,7,NULL));cases++;
    uint64_t total=UINT64_MAX-sizeof(VxPreparedShapePlan)+1;
    assert(!vx_native_gpu_accumulate_resource_bytes(&total,1,sizeof(VxPreparedShapePlan)));cases++;
    assert(total==UINT64_MAX-sizeof(VxPreparedShapePlan)+1);
    total=UINT64_MAX-sizeof(VxPreparedShapePlan);
    assert(vx_native_gpu_accumulate_resource_bytes(&total,1,sizeof(VxPreparedShapePlan)) &&
           total==UINT64_MAX);cases++;
    printf("PASS checked shape plan metadata: %zu cases; owner=%zu bytes\n",
           cases,sizeof(VxPreparedShapePlan));
}
'''


def metadata_source():
    unit = "#include <assert.h>\n#include <stdint.h>\n#include <stddef.h>\n" \
           "#include <limits.h>\n#include <stdio.h>\n#include <string.h>\n" \
           "typedef int VxMemoryLocation;\ntypedef struct VxCompiledModel VxCompiledModel;\n" \
           "typedef struct {const char* name;uint32_t rank;} VxDeclaredTensor;\n" \
           "typedef struct {size_t input_count;VxDeclaredTensor* inputs;} VxModel;\n"
    header = (RUNTIME / "engine_core.h").read_text()
    for name in ("VolvoxAIEngineResolvedTensor", "VolvoxAIEngineInputBinding",
                 "VolvoxAIEnginePhysicalSpan", "VolvoxAIEngineMaximumTensor"):
        end = header.index("} " + name + ";") + len("} " + name + ";")
        unit += header[header.rfind("typedef struct {", 0, end):end] + "\n"
    state = (RUNTIME / "runtime_state.h").read_text()
    end = state.index("} VxDynamicShapePlan;") + len("} VxDynamicShapePlan;")
    unit += state[state.rfind("typedef struct {", 0, end):end] + "\n"
    unit += prepared_plan_type() + "enum {VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY=4};\n"
    domain = (RUNTIME / "public_api_native_gpu_domain.inc").read_text()
    for name in ("vx_native_gpu_accumulate_resource_bytes", "vx_native_gpu_maximum_signature_bytes"):
        unit += function_source(domain, name)
    reference = (FIXTURES / "decode_shape_plan_transfer_reference.inc").read_text()
    name = "vx_native_gpu_dynamic_metadata_peak"
    unit += function_source(reference, name).replace(name, "old_peak")
    unit += function_source(domain, name).replace(name, "current_peak")
    return unit + METADATA_CHECKS


class DecodeShapePlanTransfer(unittest.TestCase):
    def compile_and_run(self, source, candidate=False):
        compiler = shutil.which(os.getenv("CC", "cc"))
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="volvox-shape-plan-") as directory:
            unit, binary = Path(directory) / "test.c", Path(directory) / "test"
            unit.write_text(source)
            command = [compiler, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra",
                       "-Werror", "-Wno-unused-parameter", f"-DCANDIDATE={int(candidate)}",
                       f"-I{RUNTIME}"]
            if os.getenv("VOLVOXAI_TEST_SANITIZE"):
                command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            command += shlex.split(os.getenv("VOLVOXAI_TEST_CFLAGS", ""))
            command += [str(unit), str(RUNTIME / "paged_kv.c"), "-lpthread", "-lm", "-o", str(binary)]
            built = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            return checked.stdout

    def test_first_prefill_preserves_numerics_precommit_and_cleanup(self):
        old = [json.loads(line) for line in self.compile_and_run(transaction_source(False)).splitlines()]
        current = [json.loads(line) for line in self.compile_and_run(transaction_source(True), True).splitlines()]
        self.assertEqual(len(old), 400)
        self.assertEqual(len(current), 400)
        transfers = fallbacks = avoided_failures = 0
        for before, after in zip(old, current):
            self.assertLessEqual(after["peak_plans"], 1)
            if before["case"] == 49 and after["status"] == 0:
                self.assertEqual((before["status"], before["commits"], after["commits"]), (2, 0, 1))
                avoided_failures += 1
                continue
            self.assertEqual({k: v for k, v in before.items() if k not in ("resolve_calls", "peak_plans")},
                             {k: v for k, v in after.items() if k not in ("resolve_calls", "peak_plans")})
            if before["case"] in (0, 1, 2, 3, 18) and before["status"] == 0:
                self.assertEqual((before["resolve_calls"], after["resolve_calls"]), (2, 1))
                transfers += 1
            if before["case"] in range(4, 17) and before["resolve_calls"] == 2:
                self.assertEqual(after["resolve_calls"], 2)
                fallbacks += 1
            if before["case"] in (29, 30, 31, 32, 33):
                self.assertEqual((before["resolve_calls"], after["resolve_calls"],
                                  before["commits"], after["commits"]), (1, 1, 0, 0))
        self.assertEqual((transfers, fallbacks, avoided_failures), (35, 91, 7))
        print("PASS shape plan transfer: 400 frozen comparisons; 35 transfers; 91 fallbacks; "
              "7 eliminated allocation failures; F64 numerics and exact bytes")

    def test_owner_metadata_peak_and_overflow(self):
        output = self.compile_and_run(metadata_source())
        self.assertIn("PASS checked shape plan metadata: 11 cases", output)
        print(output.strip())


DECODE_SESSION_CHECKS = r'''
static VxBackendKind selected_backend;
static int bad_page, bad_control, bad_geometry, refresh_calls;
VxBackendKind vx_backend_manager_current(void) {return selected_backend;}
int vx_adapter_run_has_versions(void) {return 0;}
int vx_adapter_run_has_target(const char* name) {(void)name;return 0;}
int volvoxai_engine_refresh_weight_caches(void) {refresh_calls++;assert(0);return -1;}
int volvoxai_engine_prepare_tensor_table_mutation(void) {assert(0);return -1;}
void volvoxai_engine_finish_tensor_table_mutation(void) {assert(0);}
void vx_runtime_backend_reset_transients(void) {}
int vx_paged_bound_locked(void) {return 0;}
int vx_paged_domain_supported_locked(const unsigned char* selected) {(void)selected;assert(0);return 0;}
int vk_sync_host_tensor(T* tensor) {(void)tensor;assert(0);return -1;}
int vx_runtime_backend_sync_host(const void* host,size_t bytes,int weight) {(void)host;(void)bytes;(void)weight;assert(0);return -1;}
int volvoxai_engine_execution_row_valid_locked(int row) {return row>=0 && row<4;}
VxDecodeRowSetStatus vx_paged_row_set_init_locked(VxDecodeRowSet* rows,int slots,const int* positions) {
    return vx_decode_row_set_init(rows,slots,positions,NULL);
}
static int live_compatible(Node* node,int index,int row) {
    (void)node;(void)index;proof_calls++;
    assert(g_decode_rows.slots==1 && g_decode_slots==1 && g_decode_rows.positions[0]==row);
    return !refuse && !bad_page && !bad_control && !bad_geometry && row>=0 && row<4;
}

typedef struct {
    int status, mode, last_mode, cache, prefilled, proof_count, dispatch_count;
    float values[6][12];
} Outcome;
static void run_case(int reference,int device,int automatic,int row,int roots,int failure,
                     int stepwise,Outcome* outcome) {
    g_current_engine_state=&contexts[0];
    selected_backend=device?VX_BACKEND_KIND_CUDA:VX_PORTABLE_BACKEND_KIND;
    graph_mode=device;device_rows=device;
    g_cache_valid=failure!=5;g_weight_caches_dirty=failure==6;g_hybrid_row_active=0;g_hybrid_prepared_row=-1;
    g_portable_selection_active=0;g_active_row=g_execution_row=-1;
    g_arena_detached=0;
    memset(g_dirty,0,6);memset(expected,0,6);
    g_dirty[0]=expected[0]=roots&1;g_dirty[1]=expected[1]=(roots>>1)&1;
    for(int tensor=0;tensor<6;tensor++)for(int k=0;k<12;k++)
        values[0][tensor][k]=(float)(tensor*3+k)*.125f;
    float numerical[6][12];memcpy(numerical,values[0],sizeof(numerical));
    for(int k=row*3;k<(row+1)*3 && k<12;k++) {
        if(roots&1)numerical[2][k]=numerical[0][k]+numerical[5][k%3];
        if(roots&2)numerical[3][k]=numerical[1][k]*2;
        numerical[4][k]=numerical[2][k]+numerical[3][k];
    }
    refuse=failure==1;bad_page=failure==2;bad_control=failure==3;bad_geometry=failure==4;
    proof_calls=calls=refresh_calls=0;
#if VOLVOXAI_ENABLE_CUDA
    g_use_cuda=device;
#endif
    VolvoxAIDecodeSession session={.magic=VOLVOXAI_DECODE_SESSION_MAGIC,
        .row_mode=automatic?VOLVOXAI_DECODE_ROW_AUTO:VOLVOXAI_DECODE_ROW_REQUIRED,
        .mode=VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW,.prefilled=1,.attached=1,.slots=1};
    g_active_decode_session=&session;
    VxDecodeRun run;
    int status=reference?decode_run_begin_reference_locked(&run,&session,0,&row,1,stepwise)
                        :decode_run_begin_locked(&run,&session,0,&row,1,stepwise);
    if(!status) {
        int node;
        while((node=vx_incremental_run_next_locked(&run.pass))>=0 && node<3) {
            assert(!vx_incremental_run_node_locked(&run.pass,node));
            expected[g_n[node].primary_output_index]=1;
        }
        assert(node==3);status=decode_run_end_locked(&run,1);
    }
    outcome->status=status;outcome->mode=session.mode;outcome->last_mode=session.last_execution_mode;
    outcome->cache=g_cache_valid;outcome->prefilled=session.prefilled;
    outcome->proof_count=proof_calls;outcome->dispatch_count=calls;
    memcpy(outcome->values,values[0],sizeof(outcome->values));
    assert(!g_decode_slots && !g_decode_rows.slots && g_execution_row==-1);
    if(!failure && row<4)assert(!status && !memcmp(numerical,values[0],sizeof(numerical)));
    if(failure && !automatic)assert(status==-1 && !calls && !session.prefilled && !g_cache_valid);
    assert(!refresh_calls);
    g_active_decode_session=NULL;refuse=bad_page=bad_control=bad_geometry=0;
}

static void session_cases(void) {
    int saved_proofs=0,comparisons=0;
    for(int device=0;device<2;device++)for(int automatic=0;automatic<2;automatic++)
    for(int row=0;row<4;row++)for(int roots=1;roots<4;roots++)
    for(int failure=0;failure<7;failure++)for(int stepwise=0;stepwise<2;stepwise++) {
        Outcome old={0},current={0};
        run_case(1,device,automatic,row,roots,failure,stepwise,&old);
        run_case(0,device,automatic,row,roots,failure,stepwise,&current);
        assert(old.status==current.status && old.mode==current.mode && old.last_mode==current.last_mode);
        assert(old.cache==current.cache && old.prefilled==current.prefilled);
        assert(old.dispatch_count==current.dispatch_count && !memcmp(old.values,current.values,sizeof(old.values)));
        if(device && !automatic && !failure) {
            int nodes=(roots==3?3:2);
            assert(old.proof_count==2*nodes && current.proof_count==nodes);
            saved_proofs+=old.proof_count-current.proof_count;
        } else assert(old.proof_count==current.proof_count);
        comparisons++;
    }
    /* A rejected physical cursor is still terminal before any dispatch. */
    Outcome old={0},current={0};
    run_case(1,1,0,4,1,0,0,&old);run_case(0,1,0,4,1,0,0,&current);
    assert(old.status==-1 && current.status==-1 && !old.dispatch_count && !current.dispatch_count);
    assert(!old.cache && !current.cache && !old.prefilled && !current.prefilled);
    printf("PASS decode session admission: %d ordinary-reference comparisons; %d duplicate node proofs removed; row numerics, CPU, AUTO fallback and terminal refusal\n",comparisons,saved_proofs);
}
'''
CHECKS = DECODE_SESSION_CHECKS


class DecodeSessionAdmission(unittest.TestCase):
    def test_required_device_admission_matches_ordinary_reference(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        fixture = FIXTURE[:FIXTURE.index("int main(void) {")]
        fixture = fixture.replace("void vx_decode_session_invalidate_cache_locked(void) {}", "")
        source = (RUNTIME / "decode_session.c").read_text()
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
        fixture = fixture.replace(
            '#include "../src/runtime/incremental_runtime.c"',
            f'#include "{RUNTIME / "incremental_runtime.c"}"\n' + production)
        fixture = fixture.replace("static int lookups, classifications, calls, graph_mode, proof_calls, refuse;",
                                  "static int lookups, classifications, calls, graph_mode, proof_calls, refuse;\nstatic int device_rows;")
        fixture = fixture.replace("int vx_runtime_backend_has_device_rows(void) {return graph_mode;}",
                                  "int vx_runtime_backend_has_device_rows(void) {return graph_mode && device_rows;}")
        fixture = fixture.replace("assert(!replay);", "(void)replay;")
        function = function_source(fixture, "vx_runtime_node_incremental_row_compatible")
        fixture = fixture.replace(function,
            "static int live_compatible(Node* node,int index,int row);\n" +
            "int vx_runtime_node_incremental_row_compatible(Node* node,int index,int row) {return live_compatible(node,index,row);}\n")
        fixture = fixture.replace("for(int i=0;i<12;i++) {\n        if(node->operator_kind",
                                  "for(int i=g_active_row<0?0:g_active_row*3;i<(g_active_row<0?12:(g_active_row+1)*3);i++) {\n        if(node->operator_kind")
        reference = (ROOT / "tools/tests/fixtures/decode_session_admission_reference.inc").read_text()
        fixture += reference + DECODE_SESSION_CHECKS + "\nint main(void) {setup(0);session_cases();vx_engine_state_release_graph_metadata(&contexts[0]);}\n"
        includes = ["native/tests", "native/include", "native/src", "native/src/runtime",
                    "native/src/kernels", "native/src/backends", "native/third_party",
                    "native/third_party/synurang/include", "runtime/generated/c"]
        with tempfile.TemporaryDirectory(prefix="volvox-decode-admission-") as temporary:
            unit = Path(temporary) / "test.c"
            unit.write_text(fixture)
            for training, cuda in ((0, 0), (1, 0), (0, 1), (1, 1)):
                binary = Path(temporary) / f"test-{training}-{cuda}"
                command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                           "-Wno-unused-function", "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                           f"-DVOLVOXAI_ENABLE_TRAINING={training}", f"-DVOLVOXAI_ENABLE_CUDA={cuda}",
                           "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_VULKAN=0",
                           "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
                if os.getenv("VOLVOXAI_TEST_SANITIZE"):
                    command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
                command += [f"-I{ROOT / path}" for path in includes]
                command += [str(unit), str(RUNTIME / "decode_row_set.c"), "-lm", "-o", str(binary)]
                built = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                checked = subprocess.run([str(binary)], text=True, capture_output=True)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
                self.assertIn("PASS decode session admission", checked.stdout)
                print(checked.stdout.strip())


DECODE_CACHE_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include "paged_kv.h"
typedef enum { VX_STATUS_OK, VX_STATUS_INVALID_ARGUMENT, VX_STATUS_OUT_OF_MEMORY,
    VX_STATUS_OVERLOADED, VX_STATUS_BACKEND_UNSUPPORTED, VX_STATUS_EXECUTION_FAILED } VxStatus;
enum { VX_OP_CROSS_SDPA=1, VX_OP_Q_SDPA, VX_OP_ADD, VX_PORT_K=1, VX_PORT_V,
    VX_PORT_Q, VX_NODE_PARAM_CAUSAL=1 };
typedef struct { char name[128]; int tensor_index,port; } Ref;
typedef struct { int operator_kind,nin,causal; Ref* ins; } Node;
typedef struct { char name[128]; void* data; int ndim,shape[8],is_graph_input;
    int64_t numel; size_t elem_size; } T;
typedef struct { T* tensors; int tensor_count; Node* nodes; int node_count; } VxEngineState;
typedef struct { const char* name; int rank,shape[8]; } VolvoxAIEngineResolvedTensor;
typedef struct { const char* name; } Output;
typedef struct { size_t output_count; Output* outputs; } Model;
typedef struct { Model* model; } Compiled;
typedef struct VxContextDecodeCache VxContextDecodeCache;
typedef struct { int decode_prefilled; VxContextDecodeCache* decode_cache;
    VxContextDecodeCache* execution_prepared_cache; size_t decode_slots;
    const uint32_t* decode_active_lengths; Compiled* compiled;
    size_t execution_memory_limit,allocated_bytes; uint64_t decode_cache_generation; } VxExecutionContext;
typedef struct { const char* const* tensors; size_t tensor_count;
    uint32_t page_tokens,max_pages; int has_max_pages,policy,clear_on_recycle; } VxDecodeCacheCommand;
static VxEngineState state;
#define g_nn state.node_count
#define g_n state.nodes
static int creates,binds,transfers,unbindings,lookups,comparisons;
static int fail_members,fail_create,fail_bind,fail_domain,fail_transfer,weight_index=-1;
static int tracked_strcmp(const char* a,const char* b) {comparisons++;return strcmp(a,b);}
#define strcmp tracked_strcmp
static VxEngineState* vx_engine_state_current(void) {return &state;}
static T* t_find(const char* name) {
    lookups++;
    for(int index=0;index<state.tensor_count;index++)
        if(!strcmp(state.tensors[index].name,name))return &state.tensors[index];
    return NULL;
}
static int vx_node_param_bool(const Node* node,int parameter,int fallback) {
    assert(parameter==VX_NODE_PARAM_CAUSAL);(void)fallback;return node->causal;
}
static int volvoxai_engine_tensor_is_model_weight_locked(const char* name) {
    return weight_index>=0 && !strcmp(name,state.tensors[weight_index].name);
}
static const VolvoxAIEngineResolvedTensor* vx_resolved_tensor_find(
    const VolvoxAIEngineResolvedTensor* resolved,size_t count,const char* name) {
    for(size_t index=0;index<count;index++)if(!strcmp(resolved[index].name,name))return &resolved[index];
    return NULL;
}
static void* tracked_calloc(size_t count,size_t bytes) {
    if(fail_members && count==(size_t)state.tensor_count && bytes==1)return NULL;
    return calloc(count,bytes);
}
#define calloc tracked_calloc
static VxPagedKVCache* tracked_create(const VxPagedKVOptions* options) {
    creates++;return fail_create?NULL:vx_paged_kv_create(options);
}
#define vx_paged_kv_create tracked_create
static int vx_paged_bind_locked(VxPagedKVCache* cache,int slots,const char* const* names,int count) {
    assert(cache && !slots && names && count>0);binds++;return fail_bind;
}
static int vx_paged_domain_supported_locked(const unsigned char* selected) {
    assert(!selected);return !fail_domain;
}
static void vx_paged_unbind_locked(void) {unbindings++;}
'''

DECODE_CACHE_CHECKS = r'''
#undef calloc
#undef vx_paged_kv_create
enum { MAX_TENSORS=66,MAX_NODES=128,ROWS=32,WIDTH=8 };
static T tensors[MAX_TENSORS];
static Node nodes[MAX_NODES];
static Ref inputs[MAX_NODES][4];
static uint32_t values[MAX_TENSORS][ROWS*WIDTH];
static const char* names[MAX_TENSORS];
static VolvoxAIEngineResolvedTensor resolved[MAX_TENSORS];
static Output output;
static Model model;
static Compiled compiled;
static uint32_t lengths[2]={3,5};
static int32_t positions[2]={2,4};
typedef struct {
    VxStatus status; int mutated,published,prepared,creates,binds,transfers;
    uint64_t generation,pool_bytes; size_t count,scratch_bytes,row_bytes;
    int slots,sequence,page_tokens,max_pages,policy;
    uint32_t data[MAX_TENSORS][ROWS*WIDTH];
    int lengths[2],page_table[16];
} Outcome;
static void setup(int count,int policy) {
    memset(tensors,0,sizeof(tensors));memset(nodes,0,sizeof(nodes));memset(inputs,0,sizeof(inputs));
    memset(resolved,0,sizeof(resolved));memset(&model,0,sizeof(model));memset(&output,0,sizeof(output));
    state=(VxEngineState){tensors,count+2,nodes,count+8};
    compiled.model=&model;
    creates=binds=transfers=unbindings=lookups=comparisons=0;
    fail_members=fail_create=fail_bind=fail_domain=fail_transfer=0;weight_index=-1;
    for(int index=0;index<state.tensor_count;index++) {
        T* tensor=&tensors[index];snprintf(tensor->name,sizeof(tensor->name),"tensor-%d",index);
        tensor->data=values[index];tensor->ndim=3;tensor->numel=ROWS*WIDTH;tensor->elem_size=4;
        tensor->shape[0]=2;tensor->shape[1]=16;tensor->shape[2]=WIDTH;
        names[index]=tensor->name;
        resolved[index]=(VolvoxAIEngineResolvedTensor){tensor->name,3,{2,16,WIDTH}};
        for(int element=0;element<ROWS*WIDTH;element++)values[index][element]=0x3f000000u+index*1024u+element;
        values[index][7]=0x7fc12345u;values[index][8]=0x80000000u;
    }
    for(int index=0;index<state.node_count;index++) {
        nodes[index]=(Node){index<count?(index&1?VX_OP_Q_SDPA:VX_OP_CROSS_SDPA):VX_OP_ADD,3,1,inputs[index]};
        for(int port=0;port<3;port++) {
            int selected=port==1 && index<count?index:count;
            strcpy(inputs[index][port].name,tensors[selected].name);
            inputs[index][port].tensor_index=selected;
            inputs[index][port].port=port==1?(index&1?VX_PORT_V:VX_PORT_K):VX_PORT_Q;
        }
    }
    (void)policy;
}
static void mutate(int mutation,int count,VxExecutionContext* context,VxDecodeCacheCommand* command,
    size_t* resolved_count) {
    int last=count-1;
    switch(mutation) {
    case 0:break;
    case 1:strcpy(inputs[last][1].name,tensors[count].name);break; /* no consumer */
    case 2:nodes[last].operator_kind=VX_OP_ADD;break;
    case 3:inputs[last][1].port=VX_PORT_Q;break;
    case 4:nodes[last].causal=0;break;
    case 5:strcpy(inputs[count][0].name,tensors[last].name);inputs[count][0].tensor_index=last;break;
    case 6:command->tensors=names+1;names[count]=names[1];break; /* duplicate selection */
    case 7:command->tensors=names+1;names[count]="absent";break;
    case 8:tensors[last].is_graph_input=1;break;
    case 9:weight_index=last;break;
    case 10:output.name=tensors[last].name;model.output_count=1;model.outputs=&output;break;
    case 11:tensors[last].ndim=1;break;
    case 12:tensors[last].shape[1]=15;resolved[last].shape[1]=15;break;
    case 13:tensors[last].numel--;break;
    case 14:tensors[last].data=NULL;break;
    case 15:*resolved_count=(size_t)last;break;
    case 16:for(int n=0;n<count;n++)inputs[n][1].tensor_index=-1;break;
    case 17:for(int n=0;n<count;n++)inputs[n][1].tensor_index=state.tensor_count+2;break;
    case 18:for(int n=0;n<count;n++)inputs[n][1].tensor_index=(n+1)%count;break;
    case 19: /* Private duplicate names still use the old first-by-name selection. */
        strcpy(tensors[count+1].name,tensors[last].name);inputs[last][1].tensor_index=count+1;break;
    case 20: /* A stale selected index on an unrelated input must not invent a consumer. */
        inputs[count][0].tensor_index=last;break;
    case 21: /* Two consumers: the final illegal use must still terminalize admission. */
        nodes[count].operator_kind=VX_OP_Q_SDPA;nodes[count].causal=1;
        strcpy(inputs[count][0].name,tensors[last].name);inputs[count][0].tensor_index=-1;break;
    case 22:command->page_tokens=0;break;
    case 23:command->max_pages=17;command->has_max_pages=1;break;
    case 24:context->decode_slots=1;break;
    case 25:context->decode_prefilled=0;break;
    case 26:fail_bind=1;break;
    case 27:fail_domain=1;break;
    case 28:fail_transfer=1;break;
    case 29:fail_create=1;break;
    case 30:context->execution_memory_limit=1;break;
    default:assert(0);
    }
}
static void capture(const VxExecutionContext* context,int mutated,VxStatus status,Outcome* out) {
    memset(out,0,sizeof(*out));out->status=status;out->mutated=mutated;
    out->published=context->decode_cache!=NULL;out->prepared=context->execution_prepared_cache!=NULL;
    out->generation=context->decode_cache_generation;out->creates=creates;out->binds=binds;out->transfers=transfers;
    const VxContextDecodeCache* cache=context->decode_cache?context->decode_cache:context->execution_prepared_cache;
    if(cache) {
        out->count=cache->count;out->pool_bytes=cache->pool_bytes;out->scratch_bytes=cache->scratch_bytes;
        out->slots=cache->slots;out->sequence=cache->sequence;out->page_tokens=cache->page_tokens;
        out->max_pages=cache->max_pages;out->policy=cache->policy;
        for(size_t index=0;index<cache->count;index++)out->row_bytes+=cache->tensors[index].row_bytes;
        memcpy(out->lengths,vx_paged_kv_lengths(cache->pages),sizeof(out->lengths));
        memcpy(out->page_table,vx_paged_kv_page_table(cache->pages),sizeof(out->page_table));
    }
    memcpy(out->data,values,sizeof(values));
}
static void run_case(int reference,int count,int policy,int mode,int mutation,Outcome* out) {
    setup(count,policy);
    VxExecutionContext context={.decode_prefilled=1,.decode_slots=2,.decode_active_lengths=lengths,
        .compiled=&compiled,.decode_cache_generation=7};
    VxDecodeCacheCommand command={.tensors=names,.tensor_count=(size_t)count,.page_tokens=2,.policy=policy,.clear_on_recycle=1};
    size_t resolved_count=(size_t)count;
    mutate(mutation,count,&context,&command,&resolved_count);
    int mutated=0;
    VxStatus status=reference?vx_cache_configure_reference(&context,&command,&mutated,
        mode?positions:NULL,resolved,resolved_count):vx_cache_configure_impl(&context,&command,&mutated,
        mode?positions:NULL,resolved,resolved_count);
    if(mode==2 && status==VX_STATUS_OK) {
        /* Execution allocates the resolved tensors between prepare and use. */
        for(int index=0;index<state.tensor_count;index++)tensors[index].data=values[index];
        /* Prepared activation retains its original no-revalidation contract. */
        nodes[0].operator_kind=VX_OP_ADD;
        status=reference?vx_cache_configure_reference(&context,&command,&mutated,NULL,NULL,0)
                        :vx_cache_configure_impl(&context,&command,&mutated,NULL,NULL,0);
    }
    capture(&context,mutated,status,out);
    vx_cache_free(context.decode_cache);vx_cache_free(context.execution_prepared_cache);
}
static void compare_cases(void) {
    int checked=0,precommit=0;
    const int counts[]={1,2,56,60,64};
    for(size_t size=0;size<sizeof(counts)/sizeof(counts[0]);size++)for(int policy=0;policy<2;policy++)
    for(int mode=0;mode<3;mode++)for(int mutation=0;mutation<31;mutation++) {
        if(counts[size]==1 && mutation==6)continue;
        Outcome old,current;
        run_case(1,counts[size],policy,mode,mutation,&old);
        run_case(0,counts[size],policy,mode,mutation,&current);
        if(old.status!=current.status || old.mutated!=current.mutated)
            fprintf(stderr,"count=%d policy=%d mode=%d mutation=%d old=%d/%d current=%d/%d\n",
                counts[size],policy,mode,mutation,old.status,old.mutated,current.status,current.mutated);
        assert(old.status==current.status && old.mutated==current.mutated);
        assert(old.published==current.published && old.prepared==current.prepared && old.generation==current.generation);
        assert(old.count==current.count && old.pool_bytes==current.pool_bytes && old.scratch_bytes==current.scratch_bytes);
        assert(old.slots==current.slots && old.sequence==current.sequence && old.row_bytes==current.row_bytes);
        assert(old.page_tokens==current.page_tokens && old.max_pages==current.max_pages && old.policy==current.policy);
        assert(old.binds==current.binds && old.transfers==current.transfers);
        assert(!memcmp(old.data,current.data,sizeof(old.data)) && !memcmp(old.lengths,current.lengths,sizeof(old.lengths)));
        assert(!memcmp(old.page_table,current.page_table,sizeof(old.page_table)));
        if(current.status==VX_STATUS_INVALID_ARGUMENT && mutation<=25) {
            int prior_prepared=mode==2 && mutation==25;
            assert(!current.mutated && !current.published && current.prepared==prior_prepared && !current.binds && !current.transfers);
            /* Shape, consumer and command mutants refuse before page/storage work. */
            if(!prior_prepared)assert(!current.creates);
            precommit++;
        }
        checked++;
    }
    printf("PASS cache configure admission: %d frozen-reference comparisons; %d precommit refusals; exact bytes, page tables, prepare and activation\n",checked,precommit);
}
static void allocation_cases(void) {
    setup(60,0);
    VxExecutionContext context={.decode_prefilled=1,.decode_slots=2,.decode_active_lengths=lengths,
        .compiled=&compiled,.decode_cache_generation=7};
    VxDecodeCacheCommand command={.tensors=names,.tensor_count=60,.page_tokens=2};
    int mutated=0;uint32_t original[MAX_TENSORS][ROWS*WIDTH];memcpy(original,values,sizeof(original));
    fail_members=1;
    assert(vx_cache_configure_impl(&context,&command,&mutated,NULL,NULL,0)==VX_STATUS_OUT_OF_MEMORY);
    assert(!mutated && !context.decode_cache && !context.execution_prepared_cache && !creates && !binds && !transfers);
    assert(context.decode_cache_generation==7 && !memcmp(original,values,sizeof(original)));
    fail_members=0;
    /* The existing execution byte budget applies to persistent buffers.
     * Transient host metadata must preserve direct activation's contract. */
    context.execution_memory_limit=(size_t)state.tensor_count-1;
    assert(vx_cache_configure_impl(&context,&command,&mutated,NULL,NULL,0)==VX_STATUS_OK);
    assert(mutated && context.decode_cache && context.decode_cache_generation==8);
    vx_cache_free(context.decode_cache);context.decode_cache=NULL;
    printf("PASS cache configure allocation: temporary-map failure refuses before mutation; original buffer budget contract retained\n");
}
int main(void) {compare_cases();allocation_cases();return 0;}
'''


def cache_admission_fixture_source():
    source = (RUNTIME / "public_api_decode_cache.inc").read_text()
    types = source[source.index("typedef struct {\n    T* tensor;"):
                   source.index("typedef struct {\n    VxExecutionContext* context;")]
    fixture = DECODE_CACHE_PREAMBLE + "typedef Compiled VxCompiledModel;\n" + prepared_plan_type() + types
    for name in ("vx_cache_prepared_shape_clear", "vx_cache_storage_free", "vx_cache_prefix_free", "vx_cache_free", "vx_cache_scratch",
                 "vx_cache_tensor_layout", "vx_cache_note_erase"):
        fixture += function_source(source, name)
    transfer = function_source(source, "vx_cache_transfer")
    transfer = transfer.replace("    size_t full_bytes =", "    transfers++;\n    if(fail_transfer)return -1;\n    size_t full_bytes =", 1)
    fixture += transfer
    fixture += function_source(source, "vx_cache_consumers_validate")
    fixture += function_source(source, "vx_cache_configure_impl")
    fixture += (ROOT / "tools/tests/fixtures/decode_cache_configure_reference.inc").read_text()
    return fixture + DECODE_CACHE_CHECKS


class DecodeCacheAdmission(unittest.TestCase):
    def test_single_graph_scan_matches_ordinary_configure(self):
        compiler = shutil.which(os.getenv("CC", "cc"))
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="volvox-cache-admission-") as temporary:
            unit = Path(temporary) / "test.c"
            unit.write_text(cache_admission_fixture_source())
            binary = Path(temporary) / "test"
            command = [compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-function", "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_CUDA=0",
                       "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
            if os.getenv("VOLVOXAI_TEST_SANITIZE"):
                command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            command += [f"-I{RUNTIME}", str(unit), str(RUNTIME / "paged_kv.c"), "-o", str(binary)]
            built = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("924 frozen-reference comparisons", checked.stdout)
            self.assertIn("PASS cache configure allocation", checked.stdout)
            print(checked.stdout.strip())


DECODE_STAGE_FIXTURE = r'''
static void clear_stages(void) {
    for (unsigned i = 0; i < VX_DECODE_NUMERIC_STAGE_COUNT; ++i) {
        vx_arena_release(test_state.decode_numeric_stage[i]);
        test_state.decode_numeric_stage[i] = NULL;
        test_state.decode_numeric_stage_bytes[i] = 0;
    }
    vx_arena_release(test_state.decode_metadata_stage);
    test_state.decode_metadata_stage = NULL; test_state.decode_metadata_stage_bytes = 0;
    test_state.decode_numeric_stage_count = 0; test_state.decode_row_stage_reserved = 0;
}

int main(void) {
    size_t capacities[VX_DECODE_NUMERIC_STAGE_COUNT];
    for (unsigned i = 0; i < VX_DECODE_NUMERIC_STAGE_COUNT; ++i) capacities[i] = 64u + i * 4u;
    size_t offsets[1] = {0u}, sizes[1] = {32u};
    VxDynamicShapeMaximumLayout maximum = {.physical_span_count = 1,
        .physical_offsets = offsets, .physical_capacities = sizes};
    const VxBackendKind backends[] = {VX_BACKEND_KIND_CUDA, VX_BACKEND_KIND_VULKAN,
        VX_BACKEND_KIND_OPENGL, VX_BACKEND_KIND_METAL};
    for (unsigned b = 0; b < 4u; ++b) {
        memset(&test_state, 0, sizeof(test_state)); test_state.loaded = 1; test_state.backend = backends[b];
        unsigned stages = vx_decode_numeric_stage_count(backends[b]);
        assert(stages == (b == 0u ? 4u : 12u));
        size_t count = 0u;
        assert(dynamic_shape_physical_span_count(&test_state, &maximum, &count) == 0 && count == 1u);
        allocation_calls = 0; fail_allocation = 3;
        assert(volvoxai_engine_reserve_decode_row_stage(capacities, 512u) == -2);
        assert(!live_allocations && !model_locks && !metadata_locks);
        assert(!test_state.decode_row_stage_reserved && !test_state.decode_numeric_stage_count);
        for (unsigned i = 0; i < VX_DECODE_NUMERIC_STAGE_COUNT; ++i)
            assert(!test_state.decode_numeric_stage[i] && !test_state.decode_numeric_stage_bytes[i]);
        allocation_calls = 0; fail_allocation = 0;
        assert(volvoxai_engine_reserve_decode_row_stage(capacities, 512u) == 0);
        assert(allocation_calls == stages + 1u && live_allocations == stages + 1u);
        assert(!model_locks && !metadata_locks);
        assert(test_state.decode_numeric_stage_count == stages && test_state.decode_row_stage_reserved);
        assert(dynamic_shape_physical_span_count(&test_state, &maximum, &count) == 0 && count == stages + 2u);
        void* arena = malloc(32u); assert(arena);
        VolvoxAIEnginePhysicalSpan spans[VX_DECODE_NUMERIC_STAGE_COUNT + 2u];
        assert(dynamic_shape_physical_spans_fill(&test_state, &maximum, arena, 32u, spans, count) == 0);
        for (unsigned i = 0; i < stages; ++i) {
            unsigned matches = 0;
            for (size_t j = 0; j < count; ++j)
                matches += spans[j].host == test_state.decode_numeric_stage[i] && spans[j].capacity_bytes == capacities[i];
            assert(matches == 1u);
        }
        assert(dynamic_shape_physical_spans_fill(&test_state, &maximum, arena, 32u, spans, count - 1u) != 0);
        unsigned published = test_state.decode_numeric_stage_count;
        test_state.decode_numeric_stage_count = published - 1u;
        assert(dynamic_shape_physical_span_count(&test_state, &maximum, &count) != 0);
        test_state.decode_numeric_stage_count = published;
        size_t saved = test_state.decode_numeric_stage_bytes[stages - 1u];
        test_state.decode_numeric_stage_bytes[stages - 1u] = 0;
        assert(dynamic_shape_physical_span_count(&test_state, &maximum, &count) != 0);
        test_state.decode_numeric_stage_bytes[stages - 1u] = saved;
        if (b == 0u) {
            test_state.decode_numeric_stage_bytes[4] = 4u;
            assert(dynamic_shape_physical_span_count(&test_state, &maximum, &count) != 0);
            test_state.decode_numeric_stage_bytes[4] = 0u;
        }
        void* original = test_state.decode_numeric_stage[0];
        assert(volvoxai_engine_reserve_decode_row_stage(capacities, 512u) != 0);
        assert(test_state.decode_numeric_stage[0] == original && live_allocations == stages + 1u);
        free(arena); clear_stages(); assert(!live_allocations);
    }
    test_state.backend = VX_BACKEND_KIND_VULKAN;
    capacities[11] = 0;
    assert(volvoxai_engine_reserve_decode_row_stage(capacities, 512u) != 0 && !live_allocations);
    capacities[11] = SIZE_MAX;
    assert(volvoxai_engine_reserve_decode_row_stage(capacities, 512u) != 0 && !live_allocations);
    test_state.backend = VX_BACKEND_KIND_NATIVE_CPU;
    assert(!vx_decode_numeric_stage_count(test_state.backend));
    assert(volvoxai_engine_reserve_decode_row_stage(capacities, 512u) != 0 && !live_allocations);
    puts("PASS production CUDA4/portable12 transaction, rollback, span identities and capacity guards");
    return 0;
}
'''


class DecodeStageDomainTests(unittest.TestCase):
    def test_production_capacity_uses_token_axis_and_conv_history(self):
        root = Path(__file__).resolve().parents[2]
        model = (root / "native/src/runtime/engine_runtime_model.inc").read_text()
        translation = r'''#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include "engine_internal.h"
static VxEngineState test_state;
static T tensors[3];
VxEngineState* vx_engine_state_current(void) { return &test_state; }
static VxBackendKind vx_backend_manager_current(void) { return test_state.backend; }
int volvoxai_engine_tensor_is_model_weight_locked(const char* name) { (void)name; return 0; }
static T* lookup(const char* name) {
    for (unsigned i = 0; i < 3u; ++i) if (!strcmp(tensors[i].name, name)) return &tensors[i];
    return NULL;
}
static T* nin(Node* node, VxPortKind port) {
    for (int i = 0; i < node->nin; ++i) if (node->ins[i].port == port) return lookup(node->ins[i].name);
    return NULL;
}
'''
        translation += function_source(model, "volvoxai_engine_decode_row_stage_capacities")
        translation += r'''
int main(void) {
    size_t bytes[VX_DECODE_NUMERIC_STAGE_COUNT] = {0}, metadata = 0;
    test_state.backend = VX_BACKEND_KIND_VULKAN; test_state.loaded = 1;
    VolvoxAIEngineMaximumTensor attention = {.name="q", .rank=4, .dtype=T_F32,
        .maximum_shape={1,1,32,16}, .maximum_byte_size=2048};
    assert(volvoxai_engine_decode_row_stage_capacities(&attention, 1u, 1u, bytes, &metadata) == 0);
    for (unsigned i = 0; i < 12u; ++i) assert(bytes[i] == 2048u);
    assert(metadata == 12u);
    /* Identical [1,1,*,*] shape is a different token axis only for actual
     * causal Conv2D topology; no attention-head width may be discarded. */
    strcpy(tensors[0].name, "conv.in"); strcpy(tensors[1].name, "conv.weight");
    strcpy(tensors[2].name, "conv.out");
    tensors[1].ndim=4; tensors[1].shape[0]=1; tensors[1].shape[1]=3;
    Ref inputs[2] = {0}, output = {0};
    inputs[0].port=VX_PORT_INPUT; strcpy(inputs[0].name, "conv.in");
    inputs[1].port=VX_PORT_WEIGHT; strcpy(inputs[1].name, "conv.weight");
    output.port=VX_PORT_OUT; strcpy(output.name, "conv.out");
    Node node = {.operator_kind=VX_OP_CONV_2D, .ins=inputs, .nin=2, .outs=&output, .nout=1};
    node.parsed_params.dilation[1]=3;
    test_state.nodes=&node; test_state.node_count=1;
    VolvoxAIEngineMaximumTensor conv[2] = {
        {.name="conv.in", .rank=4, .dtype=T_F32, .maximum_shape={1,1,8,16}, .maximum_byte_size=512},
        {.name="conv.out", .rank=4, .dtype=T_F32, .maximum_shape={1,1,8,16}, .maximum_byte_size=512}};
    assert(volvoxai_engine_decode_row_stage_capacities(conv, 2u, 1u, bytes, &metadata) == 0);
    for (unsigned i = 0; i < 12u; ++i) assert(bytes[i] == 7u * 16u * 4u);
    assert(metadata == 68u);
    /* A long checkpoint context reserves finite convolution history and
     * one activation row, rather than twelve copies of every token. */
    conv[0].maximum_shape[2]=conv[1].maximum_shape[2]=65536;
    conv[0].maximum_shape[3]=conv[1].maximum_shape[3]=4096;
    conv[0].maximum_byte_size=conv[1].maximum_byte_size=1073741824u;
    assert(volvoxai_engine_decode_row_stage_capacities(conv, 2u, 1u, bytes, &metadata) == 0);
    for (unsigned i = 0; i < 12u; ++i) assert(bytes[i] == 7u * 4096u * 4u);
    assert(metadata == (65536u * 2u + 1u) * 4u);
    assert(bytes[0] * 12u < conv[0].maximum_byte_size / 512u);
    /* Private GDN STATE is already resident and owns ping-pong workspace;
     * it must not masquerade as token-axis staging. */
    node.operator_kind=VX_OP_GATED_DELTA_NET; output.port=VX_PORT_STATE; strcpy(output.name,"state");
    VolvoxAIEngineMaximumTensor pair[2] = {attention,
        {.name="state", .rank=4, .dtype=T_F32, .maximum_shape={1,8,512,512}, .maximum_byte_size=8388608}};
    assert(volvoxai_engine_decode_row_stage_capacities(pair, 2u, 1u, bytes, &metadata) == 0);
    for (unsigned i = 0; i < 12u; ++i) assert(bytes[i] == 2048u);
    test_state.node_count=0;
    VolvoxAIEngineMaximumTensor mrope = {.name="positions", .rank=2, .dtype=T_I32,
        .maximum_shape={3,8}, .maximum_byte_size=96};
    assert(volvoxai_engine_decode_row_stage_capacities(&mrope, 1u, 1u, bytes, &metadata) == 0);
    for (unsigned i = 0; i < 12u; ++i) assert(bytes[i] == 12u);
    assert(metadata == 196u);
    /* CUDA still publishes exactly four slots and leaves a caller tail alone. */
    test_state.backend=VX_BACKEND_KIND_CUDA;
    for (unsigned i = 0; i < 12u; ++i) bytes[i]=99u;
    assert(volvoxai_engine_decode_row_stage_capacities(&attention, 1u, 1u, bytes, &metadata) == 0);
    for (unsigned i = 0; i < 4u; ++i) assert(bytes[i] == 2048u);
    for (unsigned i = 4; i < 12u; ++i) assert(bytes[i] == 99u);
    puts("PASS production true token axes, bounded dilated history, STATE exclusion and CUDA4");
    return 0;
}
'''
        self._compile_and_check(root, translation, "PASS production true token axes", portable=True)

    def test_production_transaction_and_spans(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        arena = (root / "native/src/runtime/arena.c").read_text()
        translation = '''#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine_internal.h"
static VxEngineState test_state;
static unsigned allocation_calls, fail_allocation, live_allocations;
static int model_locks, metadata_locks;
VxEngineState* vx_engine_state_current(void) { return &test_state; }
int volvoxai_engine_model_route_lease_active(void) { return 0; }
void volvoxai_engine_model_lock(void) { ++model_locks; }
void volvoxai_engine_model_unlock(void) { --model_locks; }
void volvoxai_engine_metadata_lock(void) { ++metadata_locks; }
void volvoxai_engine_metadata_unlock(void) { --metadata_locks; }
static void* vx_arena_allocate(size_t bytes) {
    ++allocation_calls;
    if (fail_allocation && allocation_calls == fail_allocation) return NULL;
    void* memory = calloc(1u, bytes);
    if (memory) ++live_allocations;
    return memory;
}
static void vx_arena_release(void* pointer) {
    if (pointer) { assert(live_allocations); --live_allocations; free(pointer); }
}
'''
        for name in ("volvoxai_engine_reserve_decode_row_stage", "dynamic_shape_physical_span_count",
                     "dynamic_shape_physical_span_compare", "dynamic_shape_physical_spans_fill"):
            translation += function_source(arena, name)
        translation += DECODE_STAGE_FIXTURE
        self._compile_and_check(root, translation, "PASS production CUDA4/portable12")

    def _compile_and_check(self, root, translation, marker, *, portable=False):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="volvox-decode-stage-") as directory:
            source, binary = Path(directory) / "domain.c", Path(directory) / "domain"
            source.write_text(translation)
            built = subprocess.run([compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_CUDA=0", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                "-DVOLVOXAI_ENABLE_VULKAN=" + ("1" if portable else "0"), "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0",
                *["-I" + str(root / path) for path in ("native/include", "native/third_party", "native/src",
                    "native/src/runtime", "native/src/kernels", "native/src/backends")], str(source),
                "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn(marker, checked.stdout)


class DecodeProjectionGeometryTest(unittest.TestCase):
    def test_typed_token_output_and_singleton_head_norm(self):
        compiler = shutil.which("cc") or shutil.which("clang")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        includes = [
            "native/include", "native/src", "native/src/runtime", "native/src/kernels",
            "native/src/backends", "native/src/api", "native/third_party",
            "native/third_party/synurang/include", "runtime/generated/c",
        ]
        with tempfile.TemporaryDirectory(prefix="volvox-row-geometry-") as directory:
            for profile in (0, 1):
                executable = Path(directory) / f"projection-{profile}"
                command = [compiler, "-std=c11", "-O2", "-Wall", "-Wextra",
                           "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                           f"-DVOLVOXAI_ENABLE_TRAINING={profile}",
                           "-DVOLVOXAI_ENABLE_CUDA=0", "-DVOLVOXAI_ENABLE_WEBGPU=0",
                           "-DVOLVOXAI_ENABLE_VULKAN=0",
                           "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0"]
                command += [f"-I{root / item}" for item in includes]
                command += [str(root / "native/tests/test_decode_projection_geometry.c"),
                            str(root / "native/src/runtime/attention_mask.c"),
                            "-o", str(executable)]
                compiled = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                executed = subprocess.run([str(executable)], capture_output=True, text=True)
                self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
                self.assertIn("6 singleton head norms PASS", executed.stdout)




if __name__ == "__main__":
    unittest.main()

