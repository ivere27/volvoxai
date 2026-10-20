"""Replay nonzero generated-ID chunks through production native sync functions."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests.test_webgpu_llm_workspace import function_source


PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef struct {
    const void* host;size_t bytes,offset;int host_dirty,device_dirty,is_weight;
    int owns_range,owns_buffer,is_alias;uint64_t shape_generation;unsigned buffer;
} Slot;
typedef Slot VkTensorSlot;
typedef Slot OglTensorSlot;
typedef Slot MetalTensorSlot;
typedef struct {int device_acquired,forward_active,implicit_forward;uint64_t shape_generation;} Context;
typedef Context OpenGLContextState;
typedef Context MetalContextState;
static Context state={1,0,0,7};
static Context* opengl_context_state_get(int create) {(void)create;return &state;}
static Context* metal_context_state_get(int create) {(void)create;return &state;}
static Slot graph_slots[4];static int graph_slot_count=1;
static uint64_t graph_shape_generation=7;
static int graph_forward_error,graph_forward_active,downloads;
static size_t last_offset,last_bytes;
static unsigned char device[256];
static int vk_is_ready(void) {return 1;}
static int ogl_ready(void) {return 1;}
static int metal_ready(void) {return 1;}
static int vk_staging_download(size_t offset,void* output,size_t bytes) {
    assert(offset+bytes<=sizeof(device));memcpy(output,device+offset,bytes);
    last_offset=offset;last_bytes=bytes;downloads++;return 1;
}
static int graph_flush_commands(void) {downloads++;return 0;}
static const void* mock_buffer_contents(unsigned buffer) {assert(buffer==1);return device+32;}
typedef unsigned GLenum;typedef intptr_t GLintptr;typedef intptr_t GLsizeiptr;
#define GL_SHADER_STORAGE_BUFFER 1u
#define GL_BUFFER_UPDATE_BARRIER_BIT 2u
#define GL_MAP_READ_BIT 4u
#define OPENGL_COMPUTE_API_DESKTOP 1u
static struct {unsigned api;} compute_capability={1};
static void opengl_device_lock(void) {}
static void opengl_device_unlock(void) {}
static int opengl_make_current_locked(void) {return 0;}
static void opengl_release_current_locked(void) {}
static void mock_barrier(unsigned flag) {(void)flag;}
static void mock_finish(void) {}
static void mock_bind(unsigned target,unsigned buffer) {(void)target;assert(buffer<=1);}
static void mock_get(unsigned target,GLintptr offset,GLsizeiptr bytes,void* output) {
    (void)target;assert(offset>=0 && bytes>0);assert((size_t)offset+(size_t)bytes<=64);
    memcpy(output,device+32+offset,(size_t)bytes);last_offset=(size_t)offset;last_bytes=(size_t)bytes;downloads++;
}
static void* mock_map(unsigned target,GLintptr offset,GLsizeiptr bytes,unsigned flag) {
    (void)target;(void)flag;assert(offset>=0 && bytes>0);assert((size_t)offset+(size_t)bytes<=64);
    last_offset=(size_t)offset;last_bytes=(size_t)bytes;downloads++;return device+32+offset;
}
static int mock_unmap(unsigned target) {(void)target;return 1;}
static void (*p_glMemoryBarrier)(unsigned)=mock_barrier;
static void (*p_glFinish)(void)=mock_finish;
static void (*p_glBindBuffer)(unsigned,unsigned)=mock_bind;
static void (*p_glGetBufferSubData)(unsigned,GLintptr,GLsizeiptr,void*)=mock_get;
static void* (*p_glMapBufferRange)(unsigned,GLintptr,GLsizeiptr,unsigned)=mock_map;
static int (*p_glUnmapBuffer)(unsigned)=mock_unmap;
'''

FIXTURE = r'''
int main(void) {
    int32_t backing[10]={0},ids[8]={3,2,7,11,5,9,4,6};
    int32_t* host=backing+1;size_t host_bytes=sizeof(ids);
    memcpy(device+32,ids,sizeof(ids));
    graph_slots[0]=(Slot){.host=host,.bytes=host_bytes,.offset=32,.device_dirty=1,
        .owns_range=1,.owns_buffer=1,.buffer=1,.shape_generation=7};
    /* Generate begins after a three-token prompt: the snapshot passes an
     * interior address for budgets1,1,2. Every device ID is nonzero. */
    assert(SYNC(host+2,4,0));assert(host[2]==7 && host[0]==0 && host[3]==0);
    assert(graph_slots[0].device_dirty && downloads==1);
    assert(SYNC(host+3,4,0));assert(host[3]==11 && graph_slots[0].device_dirty);
    assert(SYNC(host+4,8,0));assert(host[4]==5 && host[5]==9 && graph_slots[0].device_dirty);
#if !TEST_METAL
    assert(last_bytes==8);
#endif
    for(unsigned i=0;i<8;i++)assert(host[i]==(i>=2 && i<=5?ids[i]:0));
    assert(SYNC(host,host_bytes,0));assert(!memcmp(host,ids,sizeof(ids)) && !graph_slots[0].device_dirty);
    int before=downloads;assert(SYNC(host+2,4,0) && downloads==before);
    graph_slots[0].device_dirty=1;
    assert(!SYNC(host+7,8,0) && downloads==before);
    assert(!SYNC((const void*)(UINTPTR_MAX-1),4,0) && downloads==before);
    int32_t unknown=23;assert(SYNC(&unknown,4,0) && unknown==23 && downloads==before);
    /* A raw byte slice must preserve the exact device bits without clearing
     * whole-buffer ownership, even when it starts inside an I32 element. */
    memset(host,0,host_bytes);
    assert(SYNC((unsigned char*)host+9,5,0));
    assert(!memcmp((unsigned char*)host+9,device+41,5) && graph_slots[0].device_dirty);
    assert(((unsigned char*)host)[8]==0 && ((unsigned char*)host)[14]==0);
    /* The smallest current containing span wins over a larger arena span;
     * stale shape generations are ignored, and exact base still works. */
    graph_slots[1]=graph_slots[0];graph_slots[1].host=(unsigned char*)host-4;
    graph_slots[1].bytes=host_bytes+4;graph_slots[1].offset=28;
    graph_slots[1].shape_generation=6;graph_slot_count=2;
    assert(SYNC(host+6,4,0) && host[6]==4);
#if TEST_GL
    memset(host,0,host_bytes);graph_slots[0].device_dirty=1;
    compute_capability.api=2;p_glGetBufferSubData=NULL;
    assert(SYNC(host+2,8,0) && host[2]==7 && host[3]==11 && graph_slots[0].device_dirty);
#endif
    puts("PASS production native selected-ID chunk offsets and partial coherence");
}
'''


class NativeGenerationReadbackTests(unittest.TestCase):
    def test_production_native_selected_id_chunk_ranges(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        for backend, filename, name in (
            ("vulkan", "vulkan_engine.c", "vk_graph_sync_host"),
            ("opengl", "opengl_engine.c", "opengl_graph_sync_host"),
            ("metal", "metal_engine.m", "metal_graph_sync_host"),
        ):
            with self.subTest(backend=backend):
                source = (root / "native/src/backends" / filename).read_text()
                unit = PREAMBLE
                unit += function_source(source[source.rfind("static int graph_find_slot("):], "graph_find_slot")
                unit += function_source(source, "graph_find_containing_slot")
                if backend == "opengl":
                    unit += function_source(source, "read_buffer_range")
                sync = function_source(source, name)
                # Only the Objective-C accessor is adapted to the CPU fixture;
                # production range lookup and coherence logic remain verbatim.
                sync = sync.replace("[s->buffer contents]", "mock_buffer_contents(s->buffer)")
                unit += sync + "\n#define SYNC " + name + "\n"
                unit += "#define TEST_GL " + ("1\n" if backend == "opengl" else "0\n")
                unit += "#define TEST_METAL " + ("1\n" if backend == "metal" else "0\n")
                unit += FIXTURE
                with tempfile.TemporaryDirectory(prefix="volvoxai-native-generation-readback-") as temporary:
                    path, binary = Path(temporary) / "test.c", Path(temporary) / "test"
                    path.write_text(unit)
                    subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                                    "-Wno-unused-function", "-Wno-unused-variable", str(path), "-o", str(binary)], check=True)
                    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
                    self.assertIn("PASS production native selected-ID", result.stdout)


if __name__ == "__main__":
    unittest.main()
