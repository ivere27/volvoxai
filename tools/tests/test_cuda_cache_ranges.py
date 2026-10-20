"""Exercise production CUDA page maintenance and owner lookup without a GPU."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source


RANGE_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include ZERO_RANGES_HEADER
typedef struct { void* host; size_t full_bytes,offset,bytes; } CudaGraphZeroRange;
typedef uintptr_t CUdeviceptr;
typedef struct { const void* host; unsigned char* device; size_t bytes;
                 int host_dirty,device_dirty; } CudaTensorSlot;
typedef struct { CudaTensorSlot* slot; CUdeviceptr device; } CudaTensorView;
typedef struct { int owned; } CudaContextGuard;
enum { CUDA_REPLAY_PASS_INELIGIBLE, CUDA_REPLAY_PASS_CAPTURE };
static struct { int pass,failed; } cuda_replay;
static int cuda_stream;
static CudaTensorSlot slots[4];
static unsigned nslots,uploads,waits,clears,copies,marks;
static int fail_enter,fail_leave,fail_copy,fail_zero_after=-1;
typedef struct { CUdeviceptr to,from; size_t bytes; int zero; } Pending;
static Pending pending[32]; static unsigned npending;
static void drain(void) {
    for(unsigned i=0;i<npending;i++) {
        if(pending[i].zero) memset((void*)pending[i].to,0,pending[i].bytes);
        else memcpy((void*)pending[i].to,(const void*)pending[i].from,pending[i].bytes);
    }
    npending=0;
}
static void cuda_replay_fail(void) { cuda_replay.failed=1; }
static int cuda_context_enter(CudaContextGuard* guard) { guard->owned=!fail_enter;return guard->owned; }
static int cuda_context_leave(CudaContextGuard* guard) { assert(guard->owned);return !fail_leave; }
static int cuda_ok(int status,const char* operation) { (void)operation;return status==0; }
static int mock_zero(CUdeviceptr to,unsigned char value,size_t bytes,int stream) {
    assert(!value && stream==cuda_stream);clears++;
    if(fail_zero_after==0)return 1;
    if(fail_zero_after>0)fail_zero_after--;
    assert(npending<32);pending[npending++]=(Pending){to,0,bytes,1};return 0;
}
static int (*p_cuMemsetD8Async)(CUdeviceptr,unsigned char,size_t,int)=mock_zero;
#define CUDA_FORWARD_FUNCTION(handle) 1
static int cuda_launch_in_owned_context(int function,unsigned gx,unsigned gy,unsigned gz,
    unsigned bx,unsigned by,unsigned bz,unsigned shared,void** args) {
    assert(function==1 && gy==1 && gz==1 && bx==256 && by==1 && bz==1 && !shared);
    assert(gx>1 && gx<=VX_CUDA_ZERO_RANGE_CAPACITY);clears++;
    if(fail_zero_after==0)return 0;
    VxCudaZeroRanges image=*(VxCudaZeroRanges*)args[0];
    for(unsigned i=0;i<gx;i++) {
        assert(npending<32);
        pending[npending++]=(Pending){image.ranges[i].address,0,image.ranges[i].bytes,1};
    }
    return 1;
}
static int cuda_trace_copy_device(CUdeviceptr to,CUdeviceptr from,size_t bytes,int stream) {
    assert(stream==cuda_stream);copies++;
    if(fail_copy)return 1;
    assert(npending<32);pending[npending++]=(Pending){to,from,bytes,0};return 0;
}
static void graph_mark_device(CudaTensorSlot* slot) { slot->device_dirty=1;slot->host_dirty=0;marks++; }
static int graph_ensure_input_view(const void* host,size_t bytes,int weight,CudaTensorView* view) {
    assert(!weight);
    for(unsigned i=0;i<nslots;i++) {
        uintptr_t start=(uintptr_t)host,base=(uintptr_t)slots[i].host;
        if(start<base || start-base>slots[i].bytes || bytes>slots[i].bytes-(start-base))continue;
        view->slot=&slots[i];view->device=(CUdeviceptr)slots[i].device+(start-base);
        if(slots[i].host_dirty) {
            drain();uploads++;waits++;
            memcpy(slots[i].device,slots[i].host,slots[i].bytes);
            slots[i].host_dirty=slots[i].device_dirty=0;
        }
        return 1;
    }
    return 0;
}
enum { T_F32, T_I32 };
typedef struct { void* data; int dtype;size_t elem_size;long numel; } T;
typedef struct { T* tensor;size_t row_bytes; } VxCacheTensor;
typedef struct { int decode_row_stage_reserved; } VxEngineState;
static VxEngineState state={1};
static VxEngineState* vx_engine_state_current(void) { return &state; }
static int g_use_cuda=1;
typedef struct { int max_pages,page_tokens;size_t count;VxCacheTensor* tensors;
                 unsigned char* erase;void* scratch;int* indices; } VxContextDecodeCache;
static int vx_cache_scratch(VxContextDecodeCache* cache,size_t rows,size_t bytes) {
    (void)cache;(void)rows;(void)bytes;assert(!g_use_cuda);return 0;
}
static int vx_cache_transfer(const VxCacheTensor* tensor,const int* indices,
    int count,void* packed,int scatter,int upload) {
    (void)tensor;(void)indices;(void)count;(void)packed;(void)scatter;(void)upload;
    assert(!"CUDA page work must not use a numeric staging owner");return -1;
}
'''

RANGE_FIXTURE = r'''
int main(void) {
    enum { ROWS=8,WIDTH=7,GUARD=5 };
    uint32_t host[ROWS*WIDTH+2*GUARD],device[ROWS*WIDTH+2*GUARD],before[ROWS*WIDTH+2*GUARD];
    /* A tensor can be an interior view of a larger owning allocation. Its
     * stale host mirror and every exterior guard must remain untouched. */
    memset(host,0xab,sizeof(host));
    for(unsigned i=0;i<sizeof(device)/4;i++)device[i]=0x3f000000u+i;
    device[GUARD+2*WIDTH]=0x7fc12345u;device[GUARD+2*WIDTH+1]=0x80000000u;
    slots[0]=(CudaTensorSlot){host,(unsigned char*)device,sizeof(device),0,1};nslots=1;
    T tensor={host+GUARD,T_F32,4,ROWS*WIDTH};
    VxCacheTensor cached={&tensor,WIDTH*4};unsigned char erase[4]={0,0,1,0};
    VxContextDecodeCache cache={4,2,1,&cached,erase,NULL,NULL};
    memcpy(before,device,sizeof(before));
    assert(!vx_cache_flush_erases(&cache));
    assert(clears==1 && pending[0].bytes==2*WIDTH*4 && !copies && !uploads && !waits && npending==1);
    assert(!erase[2] && !cache.scratch && !cache.indices);drain();
    for(unsigned i=0;i<sizeof(device)/4;i++)
        assert(device[i]==(i>=GUARD+4*WIDTH && i<GUARD+6*WIDTH?0:before[i]));
    for(unsigned i=0;i<sizeof(host)/4;i++)assert(host[i]==0xababababu);
    /* COW transfers exact bits, including NaN payloads and negative zero,
     * using one device copy and no index or vocabulary-sized staging upload. */
    memcpy(before,device,sizeof(before));
    assert(!vx_cache_copy_page(&cache,1,3));
    assert(copies==1 && pending[0].bytes==2*WIDTH*4 && npending==1 && !uploads && !waits);drain();
    for(unsigned i=0;i<sizeof(device)/4;i++)
        assert(device[i]==(i>=GUARD+6*WIDTH && i<GUARD+8*WIDTH?before[i-4*WIDTH]:before[i]));
    assert(device[GUARD+6*WIDTH]==0x7fc12345u && device[GUARD+6*WIDTH+1]==0x80000000u);
    /* Commands snapshot offsets and own only device ranges; changing cache
     * metadata or host bytes before completion cannot affect submitted work. */
    assert(!vx_cache_copy_page(&cache,3,0));
    memset(host,0xcc,sizeof(host));cache.page_tokens=1;drain();cache.page_tokens=2;
    assert(!memcmp(device+GUARD,device+GUARD+6*WIDTH,2*WIDTH*4));
    unsigned commands=clears+copies,old_marks=marks;
    assert(vx_cache_copy_page(&cache,-1,0));assert(vx_cache_copy_page(&cache,0,4));
    assert(!cuda_graph_zero_range(tensor.data,ROWS*WIDTH*4,SIZE_MAX,4));
    assert(!cuda_graph_copy_range(tensor.data,ROWS*WIDTH*4,0,4,8));
    assert(!cuda_graph_copy_range(tensor.data,ROWS*WIDTH*4,0,0,0));
    assert(clears+copies==commands && marks==old_marks && !npending);
    state.decode_row_stage_reserved=0;erase[1]=1;
    assert(vx_cache_flush_erases(&cache) && erase[1]);state.decode_row_stage_reserved=1;
    tensor.dtype=T_I32;assert(vx_cache_flush_erases(&cache) && erase[1]);tensor.dtype=T_F32;
    cached.row_bytes=SIZE_MAX-3;assert(vx_cache_flush_erases(&cache));cached.row_bytes=WIDTH*4;
    assert(clears+copies==commands && marks==old_marks && !npending);
    fail_enter=1;assert(vx_cache_flush_erases(&cache) && erase[1]);fail_enter=0;
    fail_zero_after=0;assert(vx_cache_flush_erases(&cache) && erase[1]);fail_zero_after=-1;
    assert(marks==old_marks && !npending);
    fail_copy=1;assert(vx_cache_copy_page(&cache,0,1));fail_copy=0;
    assert(marks==old_marks && !npending);
    /* A restoration failure follows accepted work: retain dirty ownership
     * and the pending erasure marker, letting the caller terminalize context. */
    fail_leave=1;assert(vx_cache_flush_erases(&cache) && erase[1]);
    assert(marks==old_marks+1 && npending==1 && slots[0].device_dirty);drain();fail_leave=0;
    cuda_replay.pass=CUDA_REPLAY_PASS_CAPTURE;commands=clears+copies;
    assert(vx_cache_flush_erases(&cache) && cuda_replay.failed && erase[1]);
    assert(clears+copies==commands && !npending);cuda_replay.pass=0;
    /* When the host is authoritative, first preserve its unselected bytes.
     * Subsequent page work remains device-only. A same-page copy is a no-op. */
    slots[0].host_dirty=1;slots[0].device_dirty=0;memset(host,0x5a,sizeof(host));
    assert(!vx_cache_flush_erases(&cache));drain();assert(uploads==1 && waits==1);
    for(unsigned i=0;i<sizeof(device)/4;i++)
        assert(device[i]==(i>=GUARD+2*WIDTH && i<GUARD+4*WIDTH?0:0x5a5a5a5au));
    commands=clears+copies;assert(!vx_cache_copy_page(&cache,0,0));
    assert(clears+copies==commands && uploads==1 && !npending);
    /* A batch accepts all ranges before submission. A launch failure retains
     * its page marker; a successful launch snapshots every descriptor. */
    uint32_t other_host[ROWS*WIDTH],other_device[ROWS*WIDTH];
    memset(other_host,0xab,sizeof(other_host));memset(other_device,0xee,sizeof(other_device));
    slots[1]=(CudaTensorSlot){other_host,(unsigned char*)other_device,sizeof(other_device),0,1};nslots=2;
    T other={other_host,T_F32,4,ROWS*WIDTH};VxCacheTensor pair[]={{&tensor,WIDTH*4},{&other,WIDTH*4}};
    cache.tensors=pair;cache.count=2;erase[2]=1;fail_zero_after=0;
    assert(vx_cache_flush_erases(&cache) && erase[2] && !npending);
    for(unsigned i=0;i<ROWS*WIDTH;i++)assert(other_device[i]==0xeeeeeeeeu);
    fail_zero_after=-1;unsigned old_clears=clears;
    assert(!vx_cache_flush_erases(&cache) && !erase[2] && npending==2);
    assert(clears==old_clears+1);pair[1].row_bytes=4;drain();pair[1].row_bytes=WIDTH*4;
    for(unsigned i=0;i<ROWS*WIDTH;i++)
        assert(other_device[i]==(i>=4*WIDTH && i<6*WIDTH?0:0xeeeeeeeeu));
    CudaGraphZeroRange ranges[]={{tensor.data,ROWS*WIDTH*4,0,4},
        {other.data,ROWS*WIDTH*4,0,4}};
    commands=clears+copies;old_marks=marks;
    ranges[1].offset=SIZE_MAX;assert(!cuda_graph_zero_ranges(ranges,2));
    ranges[1].offset=1;assert(!cuda_graph_zero_ranges(ranges,2));
    ranges[1].offset=0;ranges[1].bytes=3;assert(!cuda_graph_zero_ranges(ranges,2));
    assert(!cuda_graph_zero_ranges(ranges,0));
    assert(!cuda_graph_zero_ranges(ranges,VX_CUDA_ZERO_RANGE_CAPACITY+1));
    assert(clears+copies==commands && marks==old_marks && !npending);
    puts("PASS CUDA exact cache range erasure/COW, bit preservation, ownership and failures");
}
'''

LOOKUP_PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
enum { CUDA_GRAPH_MAX_TENSORS=1024,CUDA_GRAPH_SLOT_HASH_CAPACITY=2048 };
typedef struct { const void* host;uintptr_t device;size_t bytes;
                 size_t capacity;int domain_span,is_weight; } CudaTensorSlot;
typedef struct { CudaTensorSlot graph_slots[1024];int graph_slot_count;
                 uint16_t graph_slot_hash[CUDA_GRAPH_SLOT_HASH_CAPACITY];int graph_slot_hash_valid;
                 uint64_t graph_slot_epoch;int domain_enforced;
                 struct { uint64_t epoch;uint16_t order[CUDA_GRAPH_MAX_TENSORS];
                          uint16_t count;int disjoint; } owner_index; } Context;
typedef Context CudaContextState;
static Context first,second,*current=&first;static unsigned accessors;
static Context* cuda_context_state_current(void) { accessors++;return current; }
#define CUDA_CONTEXT_FIELD(field) (cuda_context_state_current()->field)
#define graph_slots CUDA_CONTEXT_FIELD(graph_slots)
#define graph_slot_count CUDA_CONTEXT_FIELD(graph_slot_count)
#define graph_slot_hash CUDA_CONTEXT_FIELD(graph_slot_hash)
#define graph_slot_hash_valid CUDA_CONTEXT_FIELD(graph_slot_hash_valid)
#define graph_slot_epoch CUDA_CONTEXT_FIELD(graph_slot_epoch)
'''

LOOKUP_FIXTURE = r'''
int main(void) {
    unsigned char host[4096];
    /* Many overlapping candidates exercise the slow containing-owner scan.
     * The narrowest valid owner wins, with exact-base preference preserved. */
    for(unsigned i=0;i<1024;i++)graph_slots[i]=(CudaTensorSlot){.host=host,.device=1,.bytes=4096-i};
    graph_slot_count=1024;accessors=0;
    assert(graph_find_containing_slot(host+2,4)==1023);
    assert(accessors<=8); /* bounded cost, independent of 1024 candidates */
    accessors=0;assert(graph_find_slot(host)==0 && accessors<=8);
    accessors=0;assert(graph_find_containing_slot(host,4)==0 && accessors<=8);
    graph_slots[1023].device=0;
    assert(graph_find_containing_slot(host+2,4)==1022);
    assert(graph_find_containing_slot(host+4095,2)==-1);
    assert(graph_find_containing_slot((void*)(UINTPTR_MAX-1),4)==-1);
    assert(graph_find_containing_slot(NULL,4)==-1);
    assert(graph_find_containing_slot(host,0)==-1);
    /* Hash exact hits and corrupt-entry fallback retain existing behavior;
     * accessor cost depends on lookup path, not the number of visited slots. */
    graph_slot_hash_valid=1;
    size_t bucket=graph_slot_hash_bucket(host);graph_slot_hash[bucket]=1;
    accessors=0;assert(graph_find_slot(host)==0 && accessors<=8);
    graph_slot_hash[bucket]=2048;accessors=0;
    assert(graph_find_slot(host)==0 && !graph_slot_hash_valid && accessors<=10);
    current=&second;graph_slots[0]=(CudaTensorSlot){.host=host+64,.device=2,.bytes=64};graph_slot_count=1;
    assert(graph_find_containing_slot(host+68,4)==0);
    assert(graph_find_containing_slot(host+2,4)==-1);
    current=&first;assert(graph_find_containing_slot(host+2,4)==1022);
    /* Maximum-domain intervals are disjoint even while logical extents reset
     * or grow. Reverse slot order forces real index sorting, not array order. */
    current->domain_enforced=1;graph_slot_epoch=2;
    for(unsigned i=0;i<1024;i++)graph_slots[i]=(CudaTensorSlot){
        .host=host+(1023-i)*4,.device=1,.bytes=4,.capacity=4,.domain_span=1};
    for(unsigned i=0;i<1024;i++) {
        accessors=0;assert(graph_find_containing_slot(host+i*4+1,2)==(int)(1023-i));
        assert(accessors<=8);
    }
    assert(current->owner_index.disjoint==1 && current->owner_index.count==1024);
    assert(current->owner_index.epoch==2);
    uint16_t order[1024];memcpy(order,current->owner_index.order,sizeof(order));
    graph_slots[512].bytes=0;
    assert(graph_find_containing_slot(host+511*4+1,1)==-1);
    graph_slots[512].bytes=2;
    assert(graph_find_containing_slot(host+511*4+1,1)==512);
    assert(graph_find_containing_slot(host+511*4+1,2)==-1);
    graph_slots[512].bytes=4;
    assert(!memcmp(order,current->owner_index.order,sizeof(order)) && current->owner_index.epoch==2);
    /* An epoch change rebuilds after slot relocation or compaction. Odd
     * counts, immutable weights, gaps and crossing ranges are also admitted. */
    graph_slot_count=521;graph_slot_epoch=3;
    graph_slots[20].domain_span=0;graph_slots[20].is_weight=1;
    assert(graph_find_containing_slot(host+1003*4+1,2)==20);
    assert(current->owner_index.count==521 && current->owner_index.epoch==3);
    assert(graph_find_containing_slot(host+1,1)==-1);
    assert(graph_find_containing_slot(host+1003*4+3,2)==-1);
    /* Any physical overlap disables the index for that epoch, preserving
     * minimum logical span and exact-base preference as slots grow live. */
    graph_slots[0]=(CudaTensorSlot){.host=host+2000,.device=1,.bytes=80,.capacity=80,.domain_span=1};
    graph_slots[1]=(CudaTensorSlot){.host=host+2004,.device=1,.bytes=0,.capacity=16,.domain_span=1};
    graph_slot_count=2;graph_slot_epoch=4;
    assert(graph_find_containing_slot(host+2008,4)==0 && current->owner_index.disjoint==-1);
    graph_slots[1].bytes=16;
    assert(graph_find_containing_slot(host+2008,4)==1);
    assert(graph_find_containing_slot(host+2004,4)==1);
    current=&second;
    assert(graph_find_containing_slot(host+68,4)==0 && !current->owner_index.disjoint);
    puts("PASS CUDA lookup owner selection, hash fallback, contexts and bounded TLS resolution");
}
'''


class CudaCacheRangeTests(unittest.TestCase):
    def compile_and_run(self, unit, marker):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="volvoxai-cuda-cache-ranges-") as directory:
            source, binary = Path(directory) / "test.c", Path(directory) / "test"
            source.write_text(unit)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(marker, result.stdout)

    def test_exact_page_erasure_and_cow(self):
        root = Path(__file__).resolve().parents[2]
        memory = (root / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        cache = (root / "native/src/runtime/public_api_decode_cache.inc").read_text()
        unit = RANGE_PREAMBLE + "\n#define VOLVOXAI_ENABLE_CUDA 1\n"
        unit = unit.replace("ZERO_RANGES_HEADER", '"' + str(root / "native/src/backends/cuda/cuda_zero_ranges.h") + '"')
        for name in ("cuda_graph_zero_range", "cuda_graph_zero_ranges", "cuda_graph_copy_range"):
            unit += function_source(memory, name)
        for name in ("vx_cache_cuda_page_range", "vx_cache_flush_erases", "vx_cache_copy_page"):
            unit += function_source(cache, name)
        self.compile_and_run(unit + RANGE_FIXTURE, "PASS CUDA exact cache range")

    def test_owner_lookup_uses_bounded_context_resolution(self):
        root = Path(__file__).resolve().parents[2]
        memory = (root / "native/src/backends/cuda/host/cuda_graph_memory_host.inc").read_text()
        unit = LOOKUP_PREAMBLE
        for name in ("graph_slot_hash_bucket", "graph_find_slot_linear", "graph_find_slot",
                     "graph_slot_is_resident_weight", "graph_owner_index_less", "graph_owner_index_sift",
                     "graph_owner_index_build", "graph_owner_index_find", "graph_find_containing_slot"):
            unit += function_source(memory, name)
        self.compile_and_run(unit + LOOKUP_FIXTURE, "PASS CUDA lookup owner selection")


if __name__ == "__main__":
    unittest.main()
