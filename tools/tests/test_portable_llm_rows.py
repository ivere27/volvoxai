"""Compile production resident row staging against a strict transfer recorder."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from tools.llm_baseline.verify_portable_numeric import (
    compare, cross_reference, layernorm_reference,
)
from tools.tests import function_source




PREAMBLE = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include "llm_gpu_plan.h"
typedef struct {
    unsigned char* decode_numeric_stage[12];
    size_t decode_numeric_stage_bytes[12];
    int decode_row_stage_reserved;
} VxEngineState;
static VxEngineState owner;
static VxEngineState* vx_engine_state_current(void) {return &owner;}
static int gathers,scatters,numerical;
static const VxLLMOpArgs* original;
#define VX_LLM_ENTRY(name) mock##name
static int mock_graph_row_index_transfer(const void* source,size_t source_bytes,
        const int32_t* indices,uint32_t rows,void* destination,size_t destination_bytes,
        uint32_t width,uint32_t indexed_rows,uint32_t mode) {
    assert(source && destination && rows==1 && source!=destination);
    assert(indices[0]>=0 && (uint32_t)indices[0]<indexed_rows);
    assert((uint64_t)indexed_rows*width*4u<=(mode?destination_bytes:source_bytes));
    assert((uint64_t)width*4u<=(mode?source_bytes:destination_bytes));
    if(mode) {
        assert(source==owner.decode_numeric_stage[8] || source==owner.decode_numeric_stage[9] ||
            source==owner.decode_numeric_stage[10]);
        memcpy((float*)destination+(size_t)indices[0]*width,source,(size_t)width*4u);scatters++;
    } else {
        /* The source is the whole retained bank, not an interior host mirror. */
        int whole=source==original->mask || source==original->positions;
        for(unsigned i=0;i<8;i++)whole|=source==original->x[i];
        assert(whole);
        memcpy(destination,(const float*)source+(size_t)indices[0]*width,(size_t)width*4u);gathers++;
    }
    return 1;
}
static int mock_graph_llm_f32(const VxLLMOpArgs* row) {
    numerical++;assert(row->sequence==1 && row->query_offset==6);
    assert(row->y[0]==(float*)owner.decode_numeric_stage[8]);
    assert(row->x[0]==(float*)owner.decode_numeric_stage[0]);
    assert(row->x[0][0]==original->x[0][24]);
    if(row->kind==VX_OP_GATED_DELTA_NET) {
        assert(row->x[7]==original->y[1] && row->y[1]==original->y[1]);
        for(unsigned i=0;i<5;i++)assert(row->x[i]==(float*)owner.decode_numeric_stage[i]);
    } else if(row->kind==VX_OP_SPARSE_ATTENTION) {
        assert(row->x[1]==original->x[1] && row->x_elements[1]==original->x_elements[1]);
        assert(row->mask==(int32_t*)owner.decode_numeric_stage[1] && row->mask[0]==60);
        assert(row->positions==(int32_t*)owner.decode_numeric_stage[6] && row->positions[0]==106);
        assert(row->x[5]==(float*)owner.decode_numeric_stage[5]);
        assert(row->x[7]==(float*)owner.decode_numeric_stage[7]);
        assert(row->x[6]==original->x[6]);
    }
    for(uint64_t j=0;j<row->y_elements[0];j++)row->y[0][j]=1000.0f+(float)j;
    return 1;
}
'''
FIXTURE = r'''
int main(void) {
    unsigned char staging[12][256];
    owner.decode_row_stage_reserved=1;
    for(unsigned i=0;i<12;i++) {owner.decode_numeric_stage[i]=staging[i];owner.decode_numeric_stage_bytes[i]=256;}
    float input[8][32],output[32],state[16];
    for(unsigned i=0;i<8;i++)for(unsigned j=0;j<32;j++)input[i][j]=(float)(i*100+j);
    for(unsigned j=0;j<32;j++)output[j]=-99.0f;
    VxLLMOpArgs full={.kind=VX_OP_GATED_DELTA_NET,.batch=1,.sequence=8};
    for(unsigned i=0;i<8;i++) {full.x[i]=input[i];full.x_elements[i]=32;}
    full.x[7]=NULL;full.x_elements[7]=0;full.y[0]=output;full.y_elements[0]=32;full.y[1]=state;full.y_elements[1]=16;
    VxLLMOpArgs row=full;row.sequence=1;row.query_offset=6;
    for(unsigned i=0;i<5;i++) {row.x[i]+=24;row.x_elements[i]=4;}
    row.x[7]=state;row.x_elements[7]=16;row.y[0]+=24;row.y_elements[0]=4;original=&full;
    assert(mock_graph_llm_row_f32(&full,&row));assert(gathers==5 && scatters==1 && numerical==1);
    for(unsigned j=0;j<32;j++)assert(output[j]==(j>=24 && j<28 ? 1000.0f+(float)(j-24) : -99.0f));
    owner.decode_numeric_stage_bytes[8]=12;gathers=scatters=numerical=0;
    assert(!mock_graph_llm_row_f32(&full,&row) && !scatters && !numerical);
    owner.decode_numeric_stage_bytes[8]=256;
    int32_t mask[80],positions[8];for(unsigned j=0;j<80;j++)mask[j]=(int32_t)j;for(unsigned j=0;j<8;j++)positions[j]=100+j;
    full.kind=VX_OP_SPARSE_ATTENTION;full.x[7]=input[7];full.x_elements[7]=32;full.y[1]=NULL;full.y_elements[1]=0;
    full.mask=mask;full.mask_elements=80;full.positions=positions;full.position_elements=8;
    row=full;row.sequence=1;row.query_offset=6;
    row.x[0]+=24;row.x_elements[0]=4;row.x[5]+=24;row.x_elements[5]=4;row.x[7]+=24;row.x_elements[7]=4;
    row.y[0]+=24;row.y_elements[0]=4;row.mask+=60;row.mask_elements=10;row.positions+=6;row.position_elements=1;
    gathers=scatters=numerical=0;
    assert(mock_graph_llm_row_f32(&full,&row));assert(gathers==5 && scatters==1 && numerical==1);
    /* Position and mask do not overwrite query-dependent index weights. */
    assert(((int32_t*)staging[6])[0]==106 && ((float*)staging[7])[0]==input[7][24]);
    row.positions=positions+8;gathers=scatters=numerical=0;
    assert(!mock_graph_llm_row_f32(&full,&row) && !scatters && !numerical);
    owner.decode_row_stage_reserved=0;
    assert(!mock_graph_llm_row_f32(&full,&row));
    puts("PASS native resident modern row bases, independent stages, masks and retained rows");
}
'''


class PortableRowTests(unittest.TestCase):
    def test_native_resident_staging_exact_spans(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        source = (root / "native/src/backends/llm_gpu_native.inc").read_text()
        translation = PREAMBLE + function_source(source, "native_llm_row_input")
        # The entry uses a token-pasted private backend prefix, not a C name.
        start = source.index("int VX_LLM_ENTRY(_graph_llm_row_f32)")
        opening = source.index("{", start)
        depth, closing = 1, opening + 1
        while depth:
            depth += (source[closing] == "{") - (source[closing] == "}")
            closing += 1
        translation += source[start:closing] + "\n" + FIXTURE
        with tempfile.TemporaryDirectory(prefix="volvoxai-native-modern-rows-") as temporary:
            unit = Path(temporary) / "test.c"
            binary = Path(temporary) / "test"
            unit.write_text(translation)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(root / "native/src/backends"), str(unit), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("PASS native resident modern row", result.stdout)


    def test_webgpu_row_transfer_controls_and_first_row(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        source = (root / "native/src/backends/webgpu_row_plans.inc").read_text()
        translation = r'''
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef struct {const void* hosts[3];size_t bytes[3];uint32_t words[4],bound;} VxWebGpuNodePlan;
#define VX_SHADER_INFERENCE_ROW_INDEX_TRANSFER 1
static int validating,marks;
static void* allocations[32];static unsigned allocated;
static void* vx_webgpu_arena(size_t bytes) {
    assert(allocated<32);return allocations[allocated++]=calloc(1,bytes);
}
static void vx_webgpu_mark_host(void* user,void* host,size_t bytes,int weight) {
    (void)user;(void)host;(void)bytes;(void)weight;marks++;
}
static int vx_webgpu_bind(VxWebGpuNodePlan* p,const void* host,size_t bytes) {
    assert(p->bound<3 && host && bytes);p->hosts[p->bound]=host;p->bytes[p->bound++]=bytes;return 1;
}
static int vx_webgpu_params(VxWebGpuNodePlan* p,const uint32_t* values,unsigned count) {
    assert(count==4);memcpy(p->words,values,16);return 1;
}
static int vx_webgpu_grid(VxWebGpuNodePlan* p,int shader,uint32_t x,uint32_t y,uint32_t z) {
    assert(shader==1 && p->bound==3 && x && y==1 && z==1);
    uint32_t rows=p->words[0],width=p->words[1],full=p->words[2],scatter=p->words[3];
    const int* indices=p->hosts[1];
    for(uint32_t i=0;i<rows;i++) {
        assert(indices[i]>=0 && (uint32_t)indices[i]<full);
        memcpy((char*)p->hosts[2]+(size_t)(scatter?(uint32_t)indices[i]:i)*width,
            (const char*)p->hosts[0]+(size_t)(scatter?i:(uint32_t)indices[i])*width,width);
    }
    return 1;
}
'''
        translation += function_source(source, "vx_webgpu_modern_transfer")
        translation += function_source(source, "vx_webgpu_modern_input")
        translation += r'''
int main(void) {
    uint32_t full[32];for(unsigned i=0;i<32;i++)full[i]=0x7fa00000u+i;
    VxWebGpuNodePlan plans[8]={0};uint32_t used=0;const void* pointer=full+24;
    assert(vx_webgpu_modern_input(plans,&used,full,32,&pointer,4));
    assert(used==1 && pointer!=full+24 && plans[0].hosts[0]==full && plans[0].bytes[0]==128);
    assert(plans[0].words[1]==16 && plans[0].words[2]==8 && plans[0].words[3]==0);
    assert(((const int*)plans[0].hosts[1])[0]==6 && !memcmp(pointer,full+24,16));
    /* Row zero still needs an independent GPUBuffer: it has the same host
     * address as the full bank but a different capacity and lifetime. */
    pointer=full;assert(vx_webgpu_modern_input(plans,&used,full,32,&pointer,4));
    assert(used==2 && pointer!=full && !memcmp(pointer,full,16));
    pointer=full;assert(vx_webgpu_modern_input(plans,&used,full,32,&pointer,32));assert(used==2);
    pointer=full+32;assert(!vx_webgpu_modern_input(plans,&used,full,32,&pointer,4));
    uint32_t retained[32];for(unsigned i=0;i<32;i++)retained[i]=0xdeadbeefu;
    assert(vx_webgpu_modern_transfer(&plans[3],retained,32,(void*)plans[0].hosts[2],4,6,1,1));
    for(unsigned i=0;i<32;i++)assert(retained[i]==(i>=24 && i<28?full[i]:0xdeadbeefu));
    VxWebGpuNodePlan bad={0};assert(!vx_webgpu_modern_transfer(&bad,full,32,retained,4,8,1,0));
    assert(!vx_webgpu_modern_transfer(&bad,full,31,retained,4,0,1,0));
    validating=1;pointer=full+24;used=4;int prior=marks;
    assert(vx_webgpu_modern_input(plans,&used,full,32,&pointer,4));assert(marks==prior);
    for(unsigned i=0;i<allocated;i++)free(allocations[i]);
    puts("PASS WebGPU packed rows, exact bit transfers, first row and retained scatter");
}
'''
        with tempfile.TemporaryDirectory(prefix="volvoxai-webgpu-modern-rows-") as temporary:
            unit = Path(temporary) / "test.c"
            binary = Path(temporary) / "test"
            unit.write_text(translation)
            subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(unit), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("PASS WebGPU packed rows", result.stdout)


class PortableOracleTests(unittest.TestCase):
    def test_constant_layernorm_is_exact_affine_bias(self):
        x = np.full((3, 384), 16, np.float32)
        gamma = np.linspace(.5, 1.5, 384, dtype=np.float32)
        beta = np.linspace(-.1, .1, 384, dtype=np.float32)
        np.testing.assert_array_equal(layernorm_reference(x, gamma, beta, 1e-5),
                                      np.tile(beta.astype(np.float64), (3, 1)))

    def test_small_variance_layernorm_retains_centered_difference(self):
        delta = np.spacing(np.float32(16)) * 16
        x = np.full((1, 384), 16, np.float32)
        x[:, :192] += delta; x[:, 192:] -= delta
        actual = layernorm_reference(x, np.ones(384, np.float32), np.zeros(384, np.float32), 1e-5)
        expected = float(delta) / np.sqrt(float(delta) ** 2 + float(np.float32(1e-5)))
        np.testing.assert_allclose(actual[0, :192], expected, atol=0, rtol=1e-14)
        np.testing.assert_allclose(actual[0, 192:], -expected, atol=0, rtol=1e-14)

    def test_gqa_head_groups_and_batch_mask_are_independent(self):
        q = np.zeros((2, 1, 4, 1), np.float32)
        k = np.zeros((2, 2, 2, 1), np.float32)
        v = np.asarray([[[[1], [10]], [[3], [30]]], [[[5], [50]], [[7], [70]]]], np.float32)
        mask = np.asarray([[1, 0], [0, 1]], np.int32)
        result = cross_reference(q, k, v, mask, dict(scale=1, causal=False, mask_mode=2), [0])
        np.testing.assert_array_equal(result[:, 0, :, 0], [[1, 1, 10, 10], [7, 7, 70, 70]])

    def test_all_masked_attention_is_zero_without_division(self):
        q = np.ones((1, 2, 2, 3), np.float32)
        kv = np.ones((1, 5, 1, 3), np.float32)
        result = cross_reference(q, kv, kv, np.zeros((1, 2, 5), np.int32),
                                 dict(scale=1, causal=False, mask_mode=4), [0, 1])
        np.testing.assert_array_equal(result, np.zeros((1, 2, 2, 3)))

    def test_finite_extreme_negative_scores_keep_softmax_translation_invariance(self):
        q = np.full((1, 1, 2, 1), -1e19, np.float32)
        k = np.full((1, 2, 1, 1), 2e19, np.float32)
        v = np.asarray([[[[2]], [[4]]]], np.float32)
        result = cross_reference(q, k, v, np.ones(1, np.int32),
                                 dict(scale=1, causal=False, mask_mode=0), [0])
        np.testing.assert_array_equal(result, np.full((1, 1, 2, 1), 3))

    def test_nonfinite_classes_and_unexpected_nan_are_distinct(self):
        expected = np.asarray([np.nan, np.inf, -np.inf, 1], np.float64)
        self.assertTrue(compare(expected.astype(np.float32), expected, 1e-5, 1e-5)["pass_"])
        wrong = np.asarray([np.nan, np.inf, np.inf, np.nan], np.float32)
        self.assertFalse(compare(wrong, expected, 1e-5, 1e-5)["pass_"])


if __name__ == "__main__":
    unittest.main()
