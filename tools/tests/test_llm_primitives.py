import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np

from tools.tests import function_source
from tools.llm_reference import capture_sequence, verify_snapshot, _comparison

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
F32 = ct.POINTER(ct.c_float)
I32 = ct.POINTER(ct.c_int32)

class Args(ct.Structure):
    _fields_ = [(name, ct.c_int32) for name in (
        "kind batch sequence keys heads kv_heads key_dim value_dim streams groups rate window top_k iterations "
        "index_heads index_dim rotary_dim rotary_offset normalize causal interleaved mask_mode").split()] + [
        ("eps", ct.c_float), ("hc_eps", ct.c_float), ("scale", ct.c_float),
        ("x", F32 * 8), ("y", F32 * 3), ("mask", I32), ("positions", I32),
        ("x_elements", ct.c_uint64 * 8), ("y_elements", ct.c_uint64 * 3),
        ("mask_elements", ct.c_uint64), ("position_elements", ct.c_uint64),
        ("query_offset", ct.c_int32), ("first_unit", ct.c_uint64), ("unit_count", ct.c_uint64)]

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class ModernDecoderPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="volvox-llm-numerical-")
        library = Path(cls.directory.name) / "kernels.so"
        recompute = Path(cls.directory.name) / "recompute.c"
        recompute.write_text('#include <math.h>\n#define VX_LLM_CUDA_MATH 1\n#include "llm_math.h"\nvoid test_sparse_recompute(const VxLLMOpArgs* a) { for(uint64_t i=0;i<(uint64_t)a->batch*a->sequence;i++) vx_llm_sparse_index_unit(a,i,0); }\n')
        subprocess.run(["clang", "-shared", "-fPIC", "-O2", "-ffp-contract=off", "-I" + str(ROOT / "native/src/kernels"), str(recompute),
            str(ROOT / "native/src/kernels/sequence_ops.c"), "-lm", "-o", str(library)], check=True)
        cls.library = ct.CDLL(str(library))
        cls.library.test_sparse_recompute.argtypes = [ct.POINTER(Args)]
        cls.library.vx_llm_forward_f32.argtypes = [ct.POINTER(Args)]
        cls.library.vx_llm_forward_f32.restype = ct.c_int

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.rng = np.random.default_rng(94)

    def random(self, shape):
        return self.rng.normal(0, .25, shape).astype(np.float32)

    def run_kernel(self, kind, inputs, output_shapes, keep=None, positions=None, **params):
        recompute = params.pop("recompute", False)
        settings=dict(kind=kind,eps=1e-6,hc_eps=1e-6,scale=1)
        settings.update(params)
        args = Args(**settings)
        inputs = [None if x is None else np.ascontiguousarray(x, np.int32 if (kind==1034 and i==0) or (kind==1035 and i==3) or (kind==1025 and i==5 and not params.get("index_heads", 0)) else np.float32) for i,x in enumerate(inputs)]
        outputs = [np.full(shape, -1 if kind in (1032,1035) else np.nan, np.int32 if kind in (1032,1035) else np.float32) for shape in output_shapes]
        for i, value in enumerate(inputs):
            if value is not None:
                args.x[i] = value.ctypes.data_as(F32); args.x_elements[i] = value.size
        for i, value in enumerate(outputs):
            args.y[i] = value.ctypes.data_as(F32); args.y_elements[i] = value.size
        if keep is not None:
            keep=np.ascontiguousarray(keep,np.int32);args.mask=keep.ctypes.data_as(I32);args.mask_elements=keep.size
        if positions is not None:
            positions=np.ascontiguousarray(positions,np.int32);args.positions=positions.ctypes.data_as(I32);args.position_elements=positions.size
        if recompute: self.library.test_sparse_recompute(ct.byref(args))
        else: self.assertEqual(self.library.vx_llm_forward_f32(ct.byref(args)), 0)
        return outputs

    def test_sparse_indexer_candidates_ties_and_shared_attention_f64(self):
        for sequence, rate, top_k in ((31, 2, 3), (63, 4, 2), (127, 4, 7)):
            b, h, d = 2, 2, 3
            q, k, w = self.random((b,sequence,h,d)), self.random((b,sequence,d)), self.random((b,sequence,h))
            # Exact ReLU ties and signed weights coexist with ordinary scores.
            q[:,0:8] = 0
            score = np.einsum("bthd,bjd->bthj", q.astype(np.float64), k.astype(np.float64))
            score = (np.maximum(score, 0) * w.astype(np.float64)[...,None]).sum(2) / np.sqrt(d*h)
            expected = np.zeros((b,sequence,sequence), np.int32)
            candidates = np.zeros_like(expected)
            block, budget = 3, 2
            for bi in range(b):
                for t in range(sequence):
                    visible = (t+1)//rate
                    order = np.lexsort((np.arange(visible), -score[bi,t,:visible]))
                    expected[bi,t,order[:top_k]] = 1
                    if visible:
                        maxima = np.array([score[bi,t,j:min(j+block,visible)].max() for j in range(0,visible,block)])
                        maxima[-1] = np.inf
                        blocks = np.lexsort((np.arange(len(maxima)), -maxima))[:budget]
                        for selected in blocks: candidates[bi,t,selected*block:min((selected+1)*block,sequence//rate)] = 1
            settings = dict(batch=b,sequence=sequence,keys=sequence,heads=h,key_dim=d,rate=rate,top_k=top_k,window=block,groups=budget)
            for recompute in (False, True):
                actual, shared = self.run_kernel(1035,[q,k,w], [expected.shape,expected.shape], recompute=recompute, **settings)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(shared, candidates)
            restricted = np.zeros_like(expected)
            for bi in range(b):
                for t in range(sequence):
                    visible = (t+1)//rate
                    masked = np.where(candidates[bi,t,:visible], score[bi,t,:visible], -np.inf)
                    order = np.lexsort((np.arange(visible), -masked))
                    restricted[bi,t,order[:top_k]] = 1
            settings.update(window=0,groups=0)
            for recompute in (False,True):
                actual, = self.run_kernel(1035,[q,k,w,candidates],[expected.shape],recompute=recompute,**settings)
                np.testing.assert_array_equal(actual,restricted)
        # A compressed mask gates entries independently of ordinary KV masks.
        b,sequence,d=1,7,2
        q=self.random((b,sequence,1,d));k=self.random(q.shape);v=self.random(q.shape)
        bank=self.random((b,sequence,d));mask=np.zeros((b,sequence,sequence),np.int32)
        mask[:,:,1]=1;mask[:,:,6]=1  # Future bank entries must remain invisible.
        expected=np.zeros(q.shape,np.float64)
        for t in range(sequence):
            selected=[j for j in range((t+1)//2) if mask[0,t,j]]
            keys=np.concatenate((k[0,:t+1,0],bank[0,selected]),axis=0).astype(np.float64)
            values=np.concatenate((v[0,:t+1,0],bank[0,selected]),axis=0).astype(np.float64)
            expected[0,t,0]=softmax(keys@q[0,t,0].astype(np.float64))@values
        actual,=self.run_kernel(1025,[q,k,v,bank,None,mask], [q.shape],batch=b,sequence=sequence,keys=sequence,heads=1,kv_heads=1,key_dim=d,value_dim=d,rate=2,causal=1)
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-7)

    def test_gated_delta_state_and_grouped_heads(self):
        b,s,hk,hv,dk,dv = 2,4,2,4,3,5
        q,k = self.random((b,s,hk,dk)), self.random((b,s,hk,dk))
        v,a,beta = self.random((b,s,hv,dv)),self.random((b,s,hv)),self.random((b,s,hv))
        alog,dt = self.random((hv,)),self.random((hv,))
        state = self.random((b,hv,dk,dv))
        expected_state = state.astype(np.float64).copy()
        expected = np.empty_like(v)
        qn = q / np.sqrt((q.astype(np.float64)**2).sum(-1,keepdims=True)+1e-6)
        kn = k / np.sqrt((k.astype(np.float64)**2).sum(-1,keepdims=True)+1e-6)
        for t in range(s):
            for h in range(hv):
                decay = np.exp(-np.exp(alog[h]) * np.logaddexp(0, a[:,t,h]+dt[h]))
                expected_state[:,h] *= decay[:,None,None]
                key = kn[:,t,h//(hv//hk)]
                prediction = np.einsum('bd,bdv->bv',key,expected_state[:,h])
                correction = (v[:,t,h]-prediction)*sigmoid(beta[:,t,h])[:,None]
                expected_state[:,h] += key[:,:,None]*correction[:,None,:]
                expected[:,t,h] = np.einsum('bd,bdv->bv',qn[:,t,h//(hv//hk)],expected_state[:,h])/np.sqrt(dk)
        outputs = self.run_kernel(1024,[q,k,v,a,beta,alog,dt,state],[v.shape,state.shape],
            batch=b,sequence=s,heads=hv,kv_heads=hk,key_dim=dk,value_dim=dv,normalize=1)
        # Set the numerical contract's explicit query scale.
        scaled = outputs[0] / np.sqrt(dk)
        np.testing.assert_allclose(scaled,expected,atol=2e-6,rtol=2e-6)
        np.testing.assert_allclose(outputs[1],expected_state,atol=2e-6,rtol=2e-6)

    def test_sparse_gqa_independent_value_width_and_sinks(self):
        q,k,v = self.random((1,4,4,3)),self.random((1,4,2,3)),self.random((1,4,2,5))
        sinks = self.random((4,))
        expected = np.empty((1,4,4,5),np.float32)
        for t in range(4):
            for h in range(4):
                logits = k[0,:t+1,h//2].astype(np.float64) @ q[0,t,h]
                probabilities = softmax(np.r_[logits,sinks[h]])[:-1]
                expected[0,t,h] = probabilities @ v[0,:t+1,h//2]
        actual = self.run_kernel(1025,[q,k,v,None,sinks],[expected.shape],batch=1,sequence=4,keys=4,heads=4,kv_heads=2,key_dim=3,value_dim=5,causal=1)[0]
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)

    def test_overlapping_compression_and_rotary(self):
        for groups in (1,2):
            b,s,r,d = 2,7,2,4
            x,g,bias,w = self.random((b,s,groups*d)),self.random((b,s,groups*d)),self.random((r,groups*d)),self.random((d,))
            angles=np.arange(s)[:,None]*np.asarray([[.1,.3]])
            cos,sin=np.cos(angles).astype(np.float32),np.sin(angles).astype(np.float32)
            expected=np.zeros((b,s,d),np.float32)
            for window in range(s//r):
                kv=x[:,window*r:(window+1)*r,-d:]
                gates=g[:,window*r:(window+1)*r,-d:]+bias[None,:,-d:]
                if groups==2 and window:
                    kv=np.concatenate((x[:,(window-1)*r:window*r,:d],kv),1)
                    gates=np.concatenate((g[:,(window-1)*r:window*r,:d]+bias[None,:,:d],gates),1)
                result=(softmax(gates.astype(np.float64),1)*kv).sum(1)
                result=result/np.sqrt((result**2).mean(-1,keepdims=True)+1e-6)*w
                left,right=result[:,::2].copy(),result[:,1::2].copy()
                result[:,::2]=left*cos[window]-right*sin[window]
                result[:,1::2]=right*cos[window]+left*sin[window]
                expected[:,window]=result
            actual=self.run_kernel(1026,[x,g,bias,w,cos,sin],[expected.shape],batch=b,sequence=s,key_dim=d,groups=groups,rate=r,rotary_dim=d,interleaved=1)[0]
            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)

    def test_hyper_connections_sinkhorn_and_transposed_residual(self):
        b,s,m,d=1,3,3,4
        x=self.random((b,s,m,d));width=m*d;count=2*m+m*m
        w,bias,scales=self.random((count,width)),self.random((count,)),self.random((3,))
        flat=x.reshape(b,s,width).astype(np.float64)
        raw=(flat/np.sqrt((flat**2).mean(-1,keepdims=True)+1e-6))@w.T
        pre=sigmoid(raw[...,:m]*scales[0]+bias[:m])+1e-6
        post=2*sigmoid(raw[...,m:2*m]*scales[1]+bias[m:2*m])
        comb=softmax((raw[...,2*m:]*scales[2]+bias[2*m:]).reshape(b,s,m,m))+1e-6
        for iteration in range(5):
            if iteration: comb/=comb.sum(-1,keepdims=True)+1e-6
            comb/=comb.sum(-2,keepdims=True)+1e-6
        actual=self.run_kernel(1027,[x,w,bias,scales],[pre.shape,post.shape,comb.shape],batch=b,sequence=s,streams=m,key_dim=d,iterations=5)
        for got,want in zip(actual,(pre,post,comb)):np.testing.assert_allclose(got,want,atol=2e-6,rtol=2e-6)
        update=self.random((b,s,d))
        residual=self.run_kernel(1028,[x,update,actual[1],actual[2]],[x.shape],batch=b,sequence=s,streams=m,key_dim=d)[0]
        np.testing.assert_allclose(residual,post[...,None]*update[...,None,:]+np.einsum('bstu,bstd->bsud',comb,x),atol=2e-6,rtol=2e-6)
        headw,headbias,headscale=self.random((m,width)),self.random((m,)),self.random((1,))
        headraw=(flat/np.sqrt((flat**2).mean(-1,keepdims=True)+1e-6))@headw.T
        expected_head=((sigmoid(headraw*headscale[0]+headbias)+1e-6)[...,None]*x).sum(-2)
        head=self.run_kernel(1029,[x,headw,headbias,headscale],[expected_head.shape],batch=b,sequence=s,streams=m,key_dim=d)[0]
        np.testing.assert_allclose(head,expected_head,atol=2e-6,rtol=2e-6)

    def test_rotary_feature_offset_large_heads(self):
        x=self.random((2,3,2,512));r,offset=64,448
        for interleaved in (0,1):
            angles=self.random((2,3,r//2))
            cos,sin=np.cos(angles),np.sin(angles)
            if not interleaved:cos,sin=np.concatenate((cos,cos),-1),np.concatenate((sin,sin),-1)
            expected=x.copy()
            a=x[...,offset:offset+r]
            rotated=np.stack((-a[...,1::2],a[...,::2]),-1).reshape(a.shape) if interleaved else np.concatenate((-a[...,r//2:],a[...,:r//2]),-1)
            cc=np.repeat(cos,2,-1) if interleaved else cos
            ss=np.repeat(sin,2,-1) if interleaved else sin
            expected[...,offset:offset+r]=a*cc[:,:,None]+rotated*ss[:,:,None]
            actual=self.run_kernel(1030,[x,cos,sin],[x.shape],batch=2,sequence=3,heads=2,key_dim=512,rotary_dim=r,rotary_offset=offset,interleaved=interleaved)[0]
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)

    def test_signed_sqrt_zero_and_small_values(self):
        x=np.asarray([-9,-1e-9,-0.,0.,1e-9,4],np.float32)
        got=self.run_kernel(1033,[x],[x.shape],batch=1,sequence=1)[0]
        np.testing.assert_allclose(got,np.sign(x)*np.sqrt(np.maximum(np.abs(x),1e-6)),atol=0,rtol=0)
        self.assertFalse(np.signbit(got[2]))

    def test_embedding_shards_preserve_large_integer_ids(self):
        ids=np.asarray([[0,16777219,16777220,300000000]],np.int32)
        bank=self.random((2,4))
        got=self.run_kernel(1034,[ids,bank],[(1,4,4)],batch=1,sequence=1,key_dim=4,value_dim=2,rotary_offset=16777219,keys=300000001)[0]
        want=np.zeros((1,4,4),np.float32);want[0,1:3]=bank
        np.testing.assert_array_equal(got,want)

    def test_qsa_visible_groups_tail_and_stable_ties(self):
        b,s,h,d,r=2,7,2,4,2
        q,k,w=self.random((b,s,h,d)),self.random((b,s,d)),self.random((d,))
        cos=np.ones((b,s,d),np.float32);sin=np.zeros_like(cos)
        keep=np.ones((b,s),np.int32);keep[0,1]=0;keep[1,:2]=0
        want=np.zeros((b,s,s),np.int32)
        for batch in range(b):
            for query in range(s):
                visible=np.flatnonzero((np.arange(s)<=query)&keep[batch].astype(bool))
                complete=len(visible)//r
                if complete:
                    bank=k[batch,visible[:complete*r]].reshape(complete,r,d).astype(np.float64).mean(1)
                    bank=bank/np.sqrt((bank**2).mean(-1,keepdims=True)+1e-6)*w
                    scores=np.maximum(q[batch,query]@bank.T,0).sum(0)/np.sqrt(d)
                    winner=int(np.argmax(scores))
                    want[batch,query,visible[winner*r:(winner+1)*r]]=1
                want[batch,query,visible[complete*r:]]=1
        got=self.run_kernel(1032,[q,k,w,cos,sin],[(b,s,s)],keep=keep,batch=b,sequence=s,keys=s,heads=h,key_dim=d,rotary_dim=d,rate=r,top_k=1,causal=1,mask_mode=2)[0]
        np.testing.assert_array_equal(got,want)
        q.fill(0)
        tied=self.run_kernel(1032,[q,k,w,cos,sin],[(b,s,s)],keep=keep,batch=b,sequence=s,keys=s,heads=h,key_dim=d,rotary_dim=d,rate=r,top_k=1,causal=1,mask_mode=2)[0]
        # Equal compressed-block scores select the lowest block index.
        np.testing.assert_array_equal(np.flatnonzero(tied[0,-1]),[0,2])

    def test_sparse_index_ties_and_absolute_compression_positions(self):
        b,s,h,d=1,8,2,3
        q,k,v=self.random((b,s,h,d)),self.random((b,s,1,d)),self.random((b,s,1,d))
        bank=self.random((b,s,d));iq=np.zeros((b,s,2,2),np.float32);ik=self.random((b,s,2));iw=self.random((b,s,2))
        positions=np.asarray([7]*s,np.int32)
        want=np.empty((b,s,h,d),np.float32)
        for t in range(s):
            for head in range(h):
                # Logical local window remains causal despite absolute position7.
                local=np.arange(max(0,t-1),t+1)
                keys=np.concatenate((k[0,local,0],bank[0,:1]))
                values=np.concatenate((v[0,local,0],bank[0,:1]))
                want[0,t,head]=softmax(keys.astype(np.float64)@q[0,t,head])@values
        got=self.run_kernel(1025,[q,k,v,bank,None,iq,ik,iw],[want.shape],positions=positions,batch=b,sequence=s,keys=s,heads=h,kv_heads=1,key_dim=d,value_dim=d,causal=1,window=2,rate=4,top_k=1,index_heads=2,index_dim=2)[0]
        np.testing.assert_allclose(got,want,atol=2e-6,rtol=2e-6)

    def test_sparse_index_exact_ties_and_strict_scores_have_independent_f64_selection(self):
        # One-hot compressed values expose the actual selected indices in the
        # output. Equal index scores need not imply equal attention values.
        count = 15
        q = np.zeros((1, count, 1, count), np.float32)
        bank = np.eye(count, dtype=np.float32)[None]
        iq = np.ones((1, count, 1, 1), np.float32)
        iw = -np.ones_like(iq[..., 0])
        base = np.arange(1, count + 1, dtype=np.float32)
        base[[1, 4, 6, 7, 11]] = -1  # ReLU yields five exactly tied maxima.
        for top_k in (1, 3):
            for tied in (True, False):
                ik = base.copy()
                if not tied:
                    ik = np.linspace(0.25, 1, count, dtype=np.float32)
                    ik[7] = np.nextafter(ik[0], np.float32(0))
                scores = -np.maximum(ik.astype(np.float64), 0)
                selected = np.lexsort((np.arange(count), -scores))[:top_k]
                want = bank[0, selected].astype(np.float64).mean(axis=0)
                actual = self.run_kernel(1025, [q, q, q, bank, None, iq, ik[None, :, None], iw],
                    [q.shape], keep=np.zeros((1, count), np.int32),
                    positions=np.full(count, count - 1, np.int32), batch=1, sequence=count,
                    keys=count, heads=1, kv_heads=1, key_dim=count, value_dim=count,
                    causal=1, rate=1, top_k=top_k, index_heads=1, index_dim=1, mask_mode=2)[0]
                with self.subTest(top_k=top_k, tied=tied):
                    np.testing.assert_array_equal(np.flatnonzero(actual[0, -1, 0]), selected[np.argsort(selected)])
                    np.testing.assert_allclose(actual, np.broadcast_to(want, q.shape), atol=2e-7, rtol=2e-7)


class LlmRowPrefillDomainTests(unittest.TestCase):
    def test_production_row_uses_retained_capacity(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        source = r'''#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "engine_internal.h"
#include "llm_ops.h"
#include "packed_linear_ops.inc"
static VxEngineState state;
static T output;
static T packed_input,packed_weight,packed_scale;
static VxLLMOpArgs complete;
static int cuda_launches;
static VxLLMOpArgs cuda_args;
VxEngineState* vx_engine_state_current(void) {return &state;}
T* t_find(const char* name) {(void)name;return &output;}
void materialize_tensor_f32(T* tensor) {(void)tensor;}
int vx_llm_forward_f32(const VxLLMOpArgs* args) {(void)args;return 0;}
int cuda_graph_llm_f32(const VxLLMOpArgs* args) {cuda_launches++;cuda_args=*args;return 1;}
static T* nout(Node* node,VxPortKind port) {(void)node;return port==VX_PORT_OUT?&output:NULL;}
static T* nin(Node* node,VxPortKind port) {
    (void)node;
    return port==VX_PORT_INPUT?&packed_input:port==VX_PORT_WEIGHT?&packed_weight:
        port==VX_PORT_SCALE?&packed_scale:NULL;
}
'''
        runtime = (ROOT / "native/src/runtime/engine_runtime_llm.inc").read_text()
        source += function_source((ROOT / "native/src/kernels/llm_ops.inc").read_text(), "vx_llm_work_items")
        for name in ("vx_llm_operator", "vx_llm_tensor_valid"):
            source += function_source(runtime, name)
        source += function_source(runtime, "vx_engine_llm_node_args").replace(
            "int vx_engine_llm_node_args(", "static int production_node_args(", 1)
        source += r'''
int vx_engine_llm_node_args(Node* node,VxLLMOpArgs* args) {
    if(node->operator_kind==VX_OP_PACKED_LINEAR)return production_node_args(node,args);
    *args=complete;return 1;
}
'''
        source += function_source(runtime, "vx_engine_llm_row_args")
        status_start = runtime.index("enum { VX_LLM_NODE_SUBMITTED_ERROR")
        source += runtime[status_start:runtime.index(";", status_start) + 1] + "\n"
        source += function_source(runtime, "vx_llm_node_run")
        source += r'''
int main(void) {
    float inputs[5][128],results[128],retained[32];
    Node node={.operator_kind=VX_OP_GATED_DELTA_NET};
    output.ndim=4;output.shape[0]=1;output.shape[1]=16;output.shape[2]=2;output.shape[3]=4;
    complete.kind=VX_OP_GATED_DELTA_NET;complete.batch=1;complete.sequence=3;
    for(int i=0;i<5;i++) {complete.x[i]=inputs[i];complete.x_elements[i]=(uint64_t)16*(i<2?4:i==2?8:2);}
    complete.y[0]=results;complete.y_elements[0]=128;
    complete.y[1]=retained;complete.y_elements[1]=32;
    state.decode_slots=1;
    for(int prefix=3;prefix<=5;prefix+=2) {
        complete.sequence=prefix;
        for(int row=0;row<16;row++) {
            VxLLMOpArgs args;
            assert(vx_engine_llm_row_args(&node,row,&args)==1);
            assert(args.sequence==1 && args.query_offset==row);
            for(int i=0;i<5;i++) {
                unsigned width=i<2?4:i==2?8:2;
                assert(args.x_elements[i]==width && args.x[i]==inputs[i]+row*width);
            }
            assert(args.y_elements[0]==8 && args.y[0]==results+row*8);
            assert(args.x[7]==retained && args.y[1]==retained && args.x_elements[7]==32);
        }
    }
    VxLLMOpArgs args;
    assert(!vx_engine_llm_row_args(&node,16,&args));
    complete.y[1]=NULL;assert(!vx_engine_llm_row_args(&node,3,&args));
    complete.y[1]=retained;complete.x[7]=retained;
    assert(!vx_engine_llm_row_args(&node,3,&args));
    /* During prefill, explicit row probes must restore the full packed
     * geometry before selecting a future decode row. Weights stay shared. */
    memset(&complete,0,sizeof(complete));
    node.operator_kind=VX_OP_PACKED_LINEAR;
    output.ndim=3;output.shape[0]=1;output.shape[1]=16;output.shape[2]=8;output.numel=128;
    output.dtype=T_F32;output.data=results;
    packed_input=(T){.ndim=3,.shape={1,16,4},.numel=64,.dtype=T_F32,.data=inputs[0]};
    packed_weight=(T){.ndim=2,.shape={8,1},.numel=8,.dtype=T_I32,.data=inputs[1]};
    packed_scale=(T){.ndim=2,.shape={8,1},.numel=8,.dtype=T_F32,.data=inputs[2]};
    strcpy(output.name,"logits");
    node.parsed_params.cache[VX_NODE_PARAM_GROUP_SIZE].kind=VX_NODE_PARAM_I32;
    node.parsed_params.cache[VX_NODE_PARAM_GROUP_SIZE].value.i32=128;
    node.parsed_params.cache[VX_NODE_PARAM_ACTIVATION_BITS].kind=VX_NODE_PARAM_I32;
    node.parsed_params.cache[VX_NODE_PARAM_ACTIVATION_BITS].value.i32=8;
    state.active_row=-1;
    for(int prefix=3;prefix<=5;prefix+=2) {
        state.modern_prefill_rows=prefix;
        assert(vx_engine_llm_node_args(&node,&args)==1);
        assert(args.sequence==prefix && vx_llm_work_items(&args)==(uint64_t)prefix*8);
        /* Prefix work must not shrink the device owner's physical span. */
        assert(args.x_elements[0]==64 && args.y_elements[0]==128);
        for(int row=0;row<16;row++) {
            assert(vx_engine_llm_row_args(&node,row,&args)==1);
            assert(args.sequence==1 && args.query_offset==row);
            assert(args.x[0]==inputs[0]+row*4 && args.x_elements[0]==4);
            assert(args.y[0]==results+row*8 && args.y_elements[0]==8);
            assert(args.x[1]==inputs[1] && args.x_elements[1]==8);
            assert(args.x[2]==inputs[2] && args.x_elements[2]==8);
        }
        assert(!vx_engine_llm_row_args(&node,16,&args));
    }
    /* Generate may elide earlier LM-head rows only after the terminal
     * consumer proof. Physical buffers remain complete, and row probes must
     * discard that transient selection before addressing a decode row. */
    for(int bits=8;bits<=16;bits+=8) {
        node.parsed_params.cache[VX_NODE_PARAM_ACTIVATION_BITS].value.i32=bits;
        for(int i=0;i<64;i++)inputs[0][i]=(float)(i/4+1);
        for(int i=0;i<8;i++) {
            uint32_t word=0x88889999u;
            memcpy(&inputs[1][i],&word,sizeof(word));inputs[2][i]=1.0f;
        }
        state.modern_prefill_rows=5;state.generation_prefill_rows=5;
        state.generation_prefill_capacity=16;state.generation_prefill_logits="logits";
        assert(vx_engine_llm_node_args(&node,&args)==1);
        assert(args.sequence==5 && args.first_unit==32 && args.unit_count==8);
        assert(args.x[0]==inputs[0] && args.x_elements[0]==64);
        assert(args.y[0]==results && args.y_elements[0]==128);
        state.prefix_rows=5;cuda_launches=0;
        assert(vx_llm_node_run(&node,1)==1 && cuda_launches==1);
        assert(cuda_args.first_unit==32 && cuda_args.unit_count==8);
        assert(vx_llm_node_run(&node,0)==-1 && cuda_launches==1);
        state.generation_prefill_rows=4;
        assert(vx_llm_node_run(&node,1)==-1 && cuda_launches==1);
        state.generation_prefill_rows=5;state.active_row=4;
        assert(vx_llm_node_run(&node,1)==-1 && cuda_launches==1);
        state.active_row=-1;node.operator_kind=VX_OP_GATED_DELTA_NET;
        assert(vx_llm_node_run(&node,1)==-1 && cuda_launches==1);
        node.operator_kind=VX_OP_PACKED_LINEAR;state.prefix_rows=0;
        for(int i=0;i<128;i++)results[i]=-999.0f;
        float scratch[4];assert(vx_packed_linear_forward(&args,scratch)==0);
        for(int i=0;i<128;i++)assert(results[i]==(i>=32&&i<40?20.0f:-999.0f));
        for(int row=0;row<16;row++) {
            assert(vx_engine_llm_row_args(&node,row,&args)==1);
            assert(args.first_unit==0 && args.unit_count==0 && args.sequence==1);
            assert(args.x[0]==inputs[0]+row*4 && args.y[0]==results+row*8);
        }
        /* Ordinary Execute and incomplete/mismatched terminal proofs keep
         * every active logit row; they cannot inherit a Generate slice. */
        for(int mismatch=0;mismatch<4;mismatch++) {
            state.generation_prefill_logits=mismatch==0?NULL:mismatch==1?"other":"logits";
            state.generation_prefill_rows=mismatch==2?4:5;
            state.generation_prefill_capacity=mismatch==3?15:16;
            assert(vx_engine_llm_node_args(&node,&args)==1);
            assert(args.sequence==5 && args.first_unit==0 && args.unit_count==0);
        }
        state.generation_prefill_logits=NULL;state.generation_prefill_rows=0;
        state.generation_prefill_capacity=0;
        assert(vx_engine_llm_node_args(&node,&args)==1);
        for(int i=0;i<128;i++)results[i]=-999.0f;
        assert(vx_packed_linear_forward(&args,scratch)==0);
        for(int i=0;i<128;i++)assert(results[i]==(i<40?(float)(i/8+1)*4.0f:-999.0f));
    }
    puts("PASS production retained row capacity independent of prefill cap");
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-llm-row-domain-") as directory:
            unit, binary = Path(directory) / "row.c", Path(directory) / "row"
            unit.write_text(source)
            built = subprocess.run([compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2",
                "-Wall", "-Wextra", "-Werror", "-DVOLVOXAI_ENABLE_TRAINING=0",
                "-DVOLVOXAI_ENABLE_CUDA=1",
                "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0",
                *["-I" + str(ROOT / path) for path in ("native/include", "native/third_party", "native/src",
                    "native/src/runtime", "native/src/kernels", "native/src/backends")],
                str(unit), "-lm", "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS production retained row", run.stdout)


class SnapshotIdentityTest(unittest.TestCase):
    def test_actual_weight_bytes_and_config_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"model_type": "llama", "num_attention_heads": 4}
            (root / "config.json").write_text(json.dumps(config))
            (root / "model.safetensors").write_bytes(b"test weight identity")
            expected = {}
            for name in ("config.json", "model.safetensors"):
                raw = (root / name).read_bytes()
                expected[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            model = {"config": config, "assets": {"config.json": expected["config.json"]},
                     "weights": {"model.safetensors": expected["model.safetensors"]}}
            self.assertEqual(verify_snapshot(root, model), expected)
            (root / "model.safetensors").write_bytes(b"test weight identitY")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                verify_snapshot(root, model)


@unittest.skipIf(torch is None, "HF reference dependencies are not installed")
class NumericalOracleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)
        torch.set_num_threads(2)

    def test_llama_and_qwen_gqa_continuations_keep_independent_kv_snapshots(self):
        for config_type, model_type in ((LlamaConfig, LlamaForCausalLM),
                                        (Qwen3Config, Qwen3ForCausalLM)):
            with self.subTest(architecture=model_type.__name__):
                config = config_type(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=4, tie_word_embeddings=True, attention_dropout=0.0)
                config._attn_implementation = "eager"
                model = model_type(config).float().eval()
                ids = torch.tensor([[1, 3, 5]])
                tensors, observation = capture_sequence(model, ids, decode_steps=3,
                    tolerance={"absolute": 1e-5, "relative": 1e-5})
                self.assertEqual(len(observation["greedy_ids"]), 3)
                self.assertEqual(tensors["prefill.layer.000.key"].shape, (1, 2, 3, 4))
                self.assertEqual(tensors["decode.002.layer.000.key"].shape, (1, 2, 6, 4))
                torch.testing.assert_close(tensors["prefill.layer.000.key"],
                    tensors["decode.002.layer.000.key"][:, :, :3, :], rtol=0, atol=0)
                again, _ = capture_sequence(model, ids, decode_steps=3,
                    tolerance={"absolute": 1e-5, "relative": 1e-5})
                self.assertEqual(tensors.keys(), again.keys())
                for name in tensors:
                    self.assertTrue(torch.equal(tensors[name], again[name]), name)
                with torch.inference_mode():
                    tensors["decode.002.layer.000.key"].zero_()
                self.assertTrue(torch.any(tensors["prefill.layer.000.key"] != 0))

    def test_oracle_rejects_nonfinite_or_out_of_budget_values(self):
        expected = torch.tensor([1.0, 2.0])
        tolerance = {"absolute": 1e-5, "relative": 1e-5}
        with self.assertRaisesRegex(ValueError, "fixed tolerance"):
            _comparison(torch.zeros_like(expected), expected, tolerance)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _comparison(torch.tensor([float("nan"), 2.0]), expected, tolerance)


LLM_WORKSPACE_FIXTURE = r'''
static void tensor(unsigned index, const char* name, int b, int s, int h, int d) {
    T* t = &test_tensors[index];
    strcpy(t->name, name); t->ndim = 4; t->dtype = T_F32; t->elem_size = 4u;
    t->shape[0] = b; t->shape[1] = s; t->shape[2] = h; t->shape[3] = d;
    t->numel = (long)b * s * h * d;
}

static void node(Node* n, Ref ports[2], const char* value, const char* state) {
    memset(n, 0, sizeof(*n)); n->operator_kind = VX_OP_GATED_DELTA_NET;
    n->ins = &ports[0]; n->outs = &ports[1];
    n->nin = 1; n->ins[0].port = VX_PORT_V; strcpy(n->ins[0].name, value);
    if (state) { n->nout = 1; n->outs[0].port = VX_PORT_STATE; strcpy(n->outs[0].name, state); }
}

static void cached_predicate(const char* document, int expected) {
    VxModel model = {.logical_graph = cJSON_Parse(document)};
    assert(model.logical_graph);
    assert(vx_native_gpu_incremental_workspace_domain_required(&model) == expected);
    cJSON_Delete(model.logical_graph);
}

int main(void) {
    size_t bytes[2]; int job; Node a, b; Ref a_ports[2] = {0}, b_ports[2] = {0};
    uint64_t maximum[2] = {0u, 0u}, candidate[2], total, required;
    test_state.use_vulkan = 1;
    test_state.tensors = test_tensors; test_state.tensor_count = 4;
    tensor(0, "a.v", 1, 1, 8, 4); tensor(1, "a.state", 1, 8, 4, 4);
    tensor(2, "b.v", 1, 1, 1, 512); tensor(3, "b.state", 1, 1, 512, 512);
    node(&a, a_ports, "a.v", "a.state"); node(&b, b_ports, "b.v", "b.state");
    VolvoxAIEngineMaximumTensor descriptors[4] = {0};
    for (unsigned i = 0; i < 4; ++i) {
        descriptors[i].name = test_tensors[i].name;
        descriptors[i].rank = 4; descriptors[i].dtype = T_F32;
        memcpy(descriptors[i].maximum_shape, test_tensors[i].shape, 4u * sizeof(int));
        descriptors[i].maximum_byte_size = (size_t)test_tensors[i].numel * 4u;
    }
    /* The gates maximum belongs to A; the state maximum belongs to B. */
    descriptors[0].maximum_shape[1] = 4096;
    descriptors[0].maximum_byte_size *= 4096u;
    assert(dynamic_modern_workspace_bound(&a, descriptors, 4u, bytes, &job) == 0 && job);
    assert(bytes[0] == 524288u && bytes[1] == 512u);
    candidate[0] = bytes[0]; candidate[1] = bytes[1];
    assert(vx_llm_workspace_maxima(maximum, candidate));
    assert(dynamic_modern_workspace_bound(&b, descriptors, 4u, bytes, &job) == 0 && job);
    assert(bytes[0] == 16u && bytes[1] == 1048576u);
    candidate[0] = bytes[0]; candidate[1] = bytes[1];
    assert(vx_llm_workspace_maxima(maximum, candidate));
    assert(vx_llm_workspace_capacity(maximum, 1u, 8u, 2u, &total));
    assert(total == 1572880u && total > 1048576u + 16u + 16u);
    /* Maximum descriptors rather than bootstrap T.numel bound both slots. */
    descriptors[3].maximum_shape[0] = 2;
    descriptors[3].maximum_byte_size *= 2u;
    assert(dynamic_modern_workspace_bound(&b, descriptors, 4u, bytes, &job) == 0);
    assert(bytes[1] == 2097152u);
    b.nout = 0;
    assert(dynamic_modern_workspace_bound(&b, descriptors, 4u, bytes, &job) == 0 && !bytes[1]);
    b.nout = 1;
    descriptors[3].maximum_byte_size = (size_t)UINT32_MAX + 1u;
    assert(dynamic_modern_workspace_bound(&b, descriptors, 4u, bytes, &job) != 0);
    descriptors[0].maximum_shape[1] = 33554432;
    descriptors[0].maximum_byte_size = (size_t)33554432u * 8u * 4u * 4u;
    assert(dynamic_modern_workspace_bound(&a, descriptors, 4u, bytes, &job) != 0);
    test_state.use_vulkan = 0;
    assert(dynamic_modern_workspace_bound(&a, descriptors, 4u, bytes, &job) == 0);
    assert(!job && !bytes[0] && !bytes[1]);

    uint64_t slots[2] = {4u, 4u};
    assert(vx_llm_workspace_capacity(slots, 1u, 8u, 2u, &total) && total == 24u);
    /* Vulkan requires two independently aligned slots, plus two dummy ranges.
     * Aligning their summed eight bytes once would incorrectly admit 768. */
    assert(vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 24u, slots,
        0u, 256u, 1024u, &required) && required == 1024u);
    assert(!vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 24u, slots,
        0u, 256u, 1023u, &required));
    assert(!vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 24u, slots,
        0u, 256u, 768u, &required));
    assert(vx_native_gpu_vulkan_scratch_capacity_proven(4u, 8u, 24u, slots,
        256u, 256u, 1792u, &required) && required == 1792u);
    assert(!vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 23u, slots,
        0u, 256u, 1024u, &required));
    assert(!vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 24u, slots,
        1u, 256u, 2048u, &required));
    assert(!vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 24u, slots,
        0u, 0u, 2048u, &required));
    assert(!vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 0u, slots,
        0u, 256u, 2048u, &required));
    assert(!vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 24u, NULL,
        0u, 256u, 2048u, &required));
    slots[0] = slots[1] = 0u;
    assert(vx_native_gpu_vulkan_scratch_capacity_proven(0u, 0u, 16u, slots,
        0u, 256u, 512u, &required) && required == 512u);
    slots[0] = UINT64_MAX - 7u; slots[1] = 4u;
    assert(!vx_llm_workspace_capacity(slots, 1u, 8u, 2u, &total));
    maximum[0] = maximum[1] = 1u;
    candidate[0] = UINT64_MAX; candidate[1] = 1u;
    assert(!vx_llm_workspace_maxima(maximum, candidate));
    assert(maximum[0] == 1u && maximum[1] == 1u);

    cached_predicate("{\"execution\":{\"incremental\":{}},\"nodes\":[{\"opType\":\"GatedDeltaNet\"}]}", 1);
    cached_predicate("{\"execution\":{\"incremental\":{}},\"nodes\":[{\"opType\":\"MoEGatedMLP\"}]}", 1);
    cached_predicate("{\"execution\":{\"incremental\":{}},\"nodes\":[{\"opType\":\"MoELinear\"}]}", 1);
    cached_predicate("{\"execution\":{\"incremental\":{},\"generation\":{\"incremental_controls\":{}}},\"nodes\":[{\"opType\":\"MatMul\"}]}", 1);
    cached_predicate("{\"execution\":{\"incremental\":{}},\"nodes\":[{\"opType\":\"MatMul\"}]}", 0);
    cached_predicate("{\"execution\":{\"stateless\":{}},\"nodes\":[{\"opType\":\"GatedDeltaNet\"}]}", 0);
    cached_predicate("{\"execution\":{\"stateless\":{},\"generation\":{\"full_prefix\":{}}},\"nodes\":[{\"opType\":\"GatedDeltaNet\"}]}", 0);
    puts("PASS production persistent slots, maximum descriptors, alignment and cached admission");
    return 0;
}
'''


class LlmWorkspaceDomainTests(unittest.TestCase):
    def test_production_workspace_capacity(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        runtime = ROOT / "native/src/runtime"
        arena = (runtime / "arena.c").read_text()
        proof = (runtime / "public_api_native_gpu_domain.inc").read_text()
        translation = '''#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include "engine_internal.h"
#include "llm_workspace.h"
static VxEngineState test_state;
static T test_tensors[4];
VxEngineState* vx_engine_state_current(void) { return &test_state; }
T* t_find(const char* name) {
    for (int i = 0; i < test_state.tensor_count; ++i)
        if (!strcmp(test_tensors[i].name, name)) return &test_tensors[i];
    return NULL;
}
struct VxModel { cJSON* logical_graph; };
static const cJSON* vx_execution_document(const VxModel* model) {
    return cJSON_GetObjectItemCaseSensitive(model->logical_graph, "execution");
}
static const char* vx_execution_string(const cJSON* object, const char* key) {
    return cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(object, key));
}
'''
        for name in ("dynamic_maximum_descriptor_index", "dynamic_modern_maximum_input", "dynamic_modern_workspace_bound"):
            translation += function_source(arena, name)
        for name in ("vx_native_gpu_accumulate_resource_bytes", "vx_native_gpu_align_resource_bytes",
                     "vx_native_gpu_vulkan_scratch_capacity_proven", "vx_native_gpu_incremental_workspace_domain_required"):
            source = proof[proof.rindex("static int " + name + "("):]
            translation += function_source(source, name)
        translation += LLM_WORKSPACE_FIXTURE
        with tempfile.TemporaryDirectory(prefix="volvox-llm-workspace-") as directory:
            source, binary = Path(directory) / "domain.c", Path(directory) / "domain"
            source.write_text(translation)
            built = subprocess.run([compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2", "-Wall", "-Wextra", "-Werror",
                "-DVOLVOXAI_ENABLE_TRAINING=0", "-DVOLVOXAI_ENABLE_WEBGPU=0", "-DVOLVOXAI_ENABLE_CUDA=0",
                "-DVOLVOXAI_ENABLE_VULKAN=0", "-DVOLVOXAI_ENABLE_OPENGL=0", "-DVOLVOXAI_ENABLE_METAL=0",
                *["-I" + str(ROOT / path) for path in ("native/include", "native/third_party", "native/src",
                    "native/src/runtime", "native/src/kernels", "native/src/backends")], str(source),
                str(ROOT / "native/third_party/cJSON.c"), str(ROOT / "native/src/generated/operator_vocabulary.c"),
                "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS production persistent slots", checked.stdout)


if __name__ == "__main__":
    unittest.main()

