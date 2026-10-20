"""Small F32 modern-MoE public API oracles (CPU, or explicit CUDA backend).

VOLVOXAI_TEST_MOE_BACKEND=cuda python -m unittest python.tests.test_modern_moe
A requested backend must compile and execute; no fallback or hardware skips.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_fixture import stateless_graph
import volvoxai as vx


def save_weights(path, arrays):
    header, payload, offset = {}, [], 0
    for name, a in arrays.items():
        raw = a.tobytes(order="C")
        header[name] = {"dtype": "I32" if a.dtype == np.int32 else "F32",
                        "shape": list(a.shape), "data_offsets": [offset, offset + len(raw)]}
        payload.append(raw); offset += len(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + b"".join(payload))


def oracle(x, arrays, params, input_scale=False, limit=0, tokens=None):
    logits = x.astype(np.float64) @ arrays["router"].astype(np.float64)
    logits /= params.get("temperature", 1)
    scoring = params.get("scoring_func", "softmax")
    if scoring == "softmax":
        scores = np.exp(logits-logits.max(axis=-1,keepdims=True)); scores /= scores.sum(axis=-1,keepdims=True)
    elif scoring == "sigmoid": scores = 1/(1+np.exp(-logits))
    else: scores = np.sqrt(np.logaddexp(0,logits))
    selection = scores + arrays.get("correction",0)
    groups=params.get("num_groups",1); group_keep=params.get("topk_groups",groups)
    per_group=4//groups; m=params.get("group_score_top_k",1)
    ids=[]
    for row in range(len(x)):
        if tokens is not None: chosen=arrays["hash"][tokens[row]].tolist()
        else:
            group_scores=[sum(sorted(selection[row,g*per_group:(g+1)*per_group],reverse=True)[:m]) for g in range(groups)]
            active=sorted(range(groups),key=lambda g:(-group_scores[g],g))[:group_keep]
            # Selected logits preserve monotonic ordering before sigmoid saturation.
            ordering=selection[row] if "correction" in arrays else logits[row]
            chosen=sorted((e for e in range(4) if e//per_group in active),key=lambda e:(-ordering[e],e))[:2]
        ids.append(chosen)
    ids=np.asarray(ids,dtype=np.int32)
    routes=np.take_along_axis(scores,ids,axis=1)
    if params.get("normalize",True): routes/=routes.sum(axis=1,keepdims=True)+params.get("normalization_epsilon",0)
    routes*=params.get("routed_scaling_factor",1)
    y=np.zeros((len(x),2),dtype=np.float64)
    for row in range(len(x)):
        for slot,e in enumerate(ids[row]):
            value=x[row].astype(np.float64)*(routes[row,slot] if input_scale else 1)
            gate=value@arrays["gate"][e];up=value@arrays["up"][e]
            if limit: gate=np.minimum(gate,limit);up=np.clip(up,-limit,limit)
            hidden=gate/(1+np.exp(-gate))*up
            y[row]+=(hidden@arrays["down"][e])*(1 if input_scale else routes[row,slot])
    return ids.astype(np.float32),routes,y


class ModernMoeTest(unittest.TestCase):
    backend=os.environ.get("VOLVOXAI_TEST_MOE_BACKEND","cpu")

    def observe_route(self, session):
        self.assertEqual(session.backend, self.backend)
        if self.backend == "cuda":
            self.assertEqual(session.report.compilation.policy_mode,
                             vx.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE)
            self.assertEqual(session.report.compilation.operator_fallback,
                             vx.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)
        execute = session._engine.execute

        def observed(request, **kwargs):
            response = execute(request, **kwargs)
            self.assertEqual(response.report.backend, self.backend)
            self.assertFalse(response.report.fallback.operator_fallback_used)
            route = response.report.route
            self.assertTrue(route.builtin and route.attested)
            self.assertGreater(route.active_nodes, 0)
            self.assertEqual(route.selected_nodes, route.active_nodes)
            self.assertEqual((route.fallback_nodes, route.missing_nodes), (0, 0))
            return response

        session._engine.execute = observed

    def package(self,directory,params,*,input_scale=False,limit=0,hash_routes=False,correction=False):
        rng=np.random.default_rng(412)
        arrays={"router":np.array([[0,1,2,3],[.3,-.2,.5,-.7]],np.float32),
                "gate":rng.normal(0,.7,(4,2,3)).astype(np.float32),
                "up":rng.normal(0,.6,(4,2,3)).astype(np.float32),
                "down":rng.normal(0,.5,(4,3,2)).astype(np.float32)}
        router_inputs={"input":"x","weight":"router"}
        graph_inputs={"x":{"shape":["B",2],"dtype":"float32"}}
        if correction:
            arrays["correction"]=np.array([2.5,0,0,-1],np.float32)
            router_inputs["correction_bias"]="correction"
        if hash_routes:
            arrays["hash"]=np.array([[3,1],[0,2],[1,2]],np.int32)
            router_inputs.update(token_ids="tokens",hash_indices="hash")
            graph_inputs["tokens"]={"shape":["B"],"dtype":"int32"}
        out=lambda name,width:{"tensor":name,"shape":["B",width],"dtype":"float32"}
        gated_params={"route_input_scale":input_scale}
        if limit:gated_params["swiglu_limit"]=limit
        graph=stateless_graph({"format":"volvox-graph/v1","dimensions":{"B":{"min":1,"max":3}},
            "inputs":graph_inputs,"nodes":[
                {"id":"route","opType":"MoERouter","inputs":router_inputs,"outputs":{"indices":out("ids",2),"weights":out("routes",2)},"params":{"top_k":2,**params}},
                {"id":"experts","opType":"MoEGatedMLP","inputs":{"input":"x","gate_weight":"gate","up_weight":"up","down_weight":"down","route_indices":"ids","route_weights":"routes"},"outputs":{"out":out("y",2)},"params":gated_params}],
            "outputs":["ids","routes","y"]})
        (directory/"graph.json").write_text(json.dumps(graph))
        save_weights(directory/"model.safetensors",arrays)
        return arrays

    def test_model_families_full_and_changing_rows(self):
        cases=[({"scoring_func":"softmax","normalize":True},False,0,False,False),
               ({"scoring_func":"softmax","normalize":False},False,0,False,False),
               ({"scoring_func":"sigmoid","normalize":False},True,0,False,False),
               ({"scoring_func":"sigmoid","normalize":True,"num_groups":2,"topk_groups":1,"group_score_top_k":2,"routed_scaling_factor":2.5},False,0,False,True),
               ({"scoring_func":"sqrtsoftplus","normalize":True,"normalization_epsilon":1e-20,"routed_scaling_factor":1.5},False,.8,True,False)]
        for params,scaled,limit,hashed,corrected in cases:
            with self.subTest(params=params), tempfile.TemporaryDirectory() as tmp:
                arrays=self.package(Path(tmp),params,input_scale=scaled,limit=limit,hash_routes=hashed,correction=corrected)
                with vx.InferenceSession(tmp,backend=self.backend) as session:
                    self.observe_route(session)
                    for rows in (1,3,2):
                        x=np.array([[1,-2],[.5,3],[-1,-.25]],np.float32)[:rows]
                        tokens=np.array([0,2,1],np.int32)[:rows] if hashed else None
                        actual=session.run({"x":x,**({"tokens":tokens} if hashed else {})})
                        expected=oracle(x,arrays,params,scaled,limit,tokens)
                        np.testing.assert_array_equal(actual["ids"],expected[0])
                        np.testing.assert_allclose(actual["routes"],expected[1],atol=2e-5,rtol=2e-5)
                        np.testing.assert_allclose(actual["y"],expected[2],atol=2e-5,rtol=2e-5)

    def test_hash_token_range_refusal_before_next_valid_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            params={"scoring_func":"sqrtsoftplus"}
            arrays=self.package(Path(tmp),params,hash_routes=True)
            x=np.array([[1,-2]],np.float32)
            with vx.InferenceSession(tmp,backend=self.backend) as session:
                self.observe_route(session)
                if self.backend == "cuda":
                    with self.assertRaises(vx.VolvoxAIError) as denied:
                        session.run({"x":x,"tokens":np.array([3],np.int32)})
                    self.assertEqual(denied.exception.report.status,vx.pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT)
                tokens=np.array([1],np.int32)
                np.testing.assert_allclose(session.run({"x":x,"tokens":tokens})["y"],oracle(x,arrays,params,tokens=tokens)[2],atol=2e-5,rtol=2e-5)

if __name__ == "__main__":unittest.main()
