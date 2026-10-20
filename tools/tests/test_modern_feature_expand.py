import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools.exporter.capabilities import validate_graph
from tools.exporter.hf_deepseek import DeepSeekDecoderConfig, build_deepseek_decoder_graph
from tools.exporter.hf_extended import ExtendedDecoderConfig, build_extended_decoder_graph
from tools.exporter.portable_domain import head_attention_backend_supported
from tools.tests import function_source
from tools.llm_baseline.generate_modern_incremental_stress import tiny_config, tiny_deepseek_config, tiny_qwen4

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "tools/llm_baseline/run_native_capacity_worker.py"
sys.path.insert(0, str(WRAPPER.parent))
from run_native_capacity_worker import reserve_vulkan_allocations


class ModernFeatureExpandTests(unittest.TestCase):
    def test_real_projection_keeps_singleton_sequence(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        source = r'''#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "engine_internal.h"
#include "modern_row.h"
static VxEngineState state;
static Node nodes[1];
static GraphNodeFusion fusions[1];
static T tensors[2];
static Ref inputs[1], outputs[1];
static float input_data[128], output_data[256];
VxEngineState* vx_engine_state_current(void) {return &state;}
T* t_find(const char* name) {
    for(int i=0;i<2;i++) if(!strcmp(name,tensors[i].name)) return &tensors[i];
    return NULL;
}
static T* nin(Node* node,VxPortKind port) {
    for(int i=0;i<node->nin;i++) if(node->ins[i].port==port) return t_find(node->ins[i].name);
    return NULL;
}
static T* nout(Node* node,VxPortKind port) {
    for(int i=0;i<node->nout;i++) if(node->outs[i].port==port) return t_find(node->outs[i].name);
    return NULL;
}
static T* node_input_default(Node* node) {return nin(node,VX_PORT_INPUT);}
static int vx_decode_projection_invariant(const Node* node,const T* tensor) {
    (void)node;(void)tensor;return 0;
}
'''
        geometry = (root / "native/src/runtime/engine_runtime_row_modern.inc").read_text()
        source += function_source(geometry, "vx_modern_row_span")
        source += function_source(geometry, "vx_engine_modern_row_projection")
        source += r'''
static void shape(T* tensor,int sequence,int streams,float* data) {
    tensor->dtype=T_F32;tensor->elem_size=4;tensor->ndim=4;
    tensor->shape[0]=1;tensor->shape[1]=sequence;tensor->shape[2]=streams;tensor->shape[3]=16;
    tensor->numel=(long)sequence*streams*16;tensor->data=data;
}
int main(void) {
    state.nodes=nodes;state.node_count=1;state.node_fusion=fusions;state.decode_slots=1;
    strcpy(tensors[0].name,"input");strcpy(tensors[1].name,"output");
    strcpy(inputs[0].name,"input");inputs[0].port=VX_PORT_INPUT;
    strcpy(outputs[0].name,"output");outputs[0].port=VX_PORT_OUT;
    nodes[0].operator_kind=VX_OP_EXPAND;nodes[0].ins=inputs;nodes[0].nin=1;
    nodes[0].outs=outputs;nodes[0].nout=1;
    for(int sequence=1;sequence<=5;sequence+=4) {
        shape(&tensors[0],sequence,1,input_data);shape(&tensors[1],sequence,2,output_data);
        for(int row=0;row<sequence;row++) {
            VxModernRowProjection p;
            assert(vx_engine_modern_row_projection(nodes,row,&p));
            assert(p.inputs_count==1 && p.outputs_count==1 && !p.copy);
            assert(p.inputs[0].axis==1 && p.outputs[0].axis==1);
            assert(p.inputs[0].narrow.numel==16 && p.outputs[0].narrow.numel==32);
            assert(p.inputs[0].narrow.data==input_data+row*16);
            assert(p.outputs[0].narrow.data==output_data+row*32);
            assert(p.inputs[0].narrow.shape[1]==1 && p.outputs[0].narrow.shape[1]==1);
            assert(p.inputs[0].narrow.shape[2]==1 && p.outputs[0].narrow.shape[2]==2);
        }
        VxModernRowProjection p;assert(!vx_engine_modern_row_projection(nodes,sequence,&p));
    }
    /* Broadcasting a token or batch axis cannot be a feature-only row job. */
    shape(&tensors[0],1,1,input_data);shape(&tensors[1],5,2,output_data);
    VxModernRowProjection p;assert(!vx_engine_modern_row_projection(nodes,0,&p));
    shape(&tensors[0],5,3,input_data);assert(!vx_engine_modern_row_projection(nodes,0,&p));
    shape(&tensors[0],5,1,input_data);tensors[1].shape[0]=2;
    assert(!vx_engine_modern_row_projection(nodes,0,&p));
    puts("PASS production feature Expand token-axis, singleton and history offsets");
}
'''
        with tempfile.TemporaryDirectory(prefix="volvox-feature-expand-") as directory:
            unit, binary = Path(directory) / "expand.c", Path(directory) / "expand"
            unit.write_text(source)
            built = subprocess.run([compiler, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-O2",
                "-Wall", "-Wextra", "-Werror", "-DVOLVOXAI_ENABLE_TRAINING=0",
                *["-I" + str(root / path) for path in ("native/include", "native/third_party", "native/src",
                    "native/src/runtime", "native/src/kernels", "native/src/backends")],
                str(unit), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS production feature Expand", run.stdout)


TARGETS = ("native-cpu", "wasm", "webgpu", "backend:cuda",
           "backend:vulkan", "backend:opengl", "backend:metal")


class ModernGpuAdmissionTests(unittest.TestCase):
    def test_nine_family_graphs_keep_their_generation_contract(self):
        families = ("qwen3_moe", "qwen3_next", "qwen3_5_text",
                    "qwen3_5_moe_text", "llama4_text", "qwen4_exp_text",
                    "deepseek_v2", "deepseek_v3", "deepseek_v4")
        for family in families:
            deepseek = family.startswith("deepseek_")
            raw = tiny_deepseek_config(family) if deepseek else tiny_qwen4() if family == "qwen4_exp_text" else tiny_config(family)
            config = (DeepSeekDecoderConfig if deepseek else ExtendedDecoderConfig).from_hf(raw, max_context=16)
            source = {
                name: np.zeros(shape, np.int64 if name.endswith(("hash_indices", "tid2eid")) else np.float32)
                for name, shape in config.weight_shapes().items()
            }
            builder = build_deepseek_decoder_graph if deepseek else build_extended_decoder_graph
            graph, weights = builder(config, source, generate=True)
            generation = graph["execution"]["generation"]
            self.assertIn("full_prefix", generation)
            self.assertEqual(generation["initialize_entrypoint"], generation["transition_entrypoint"])
            for target in TARGETS:
                with self.subTest(family=family, target=target):
                    result = validate_graph(graph, (target,), weights=weights)
                    self.assertTrue(result.supported, [value.to_dict() for value in result.diagnostics])
            if family == "llama4_text":
                prefix = generation["full_prefix"]
                self.assertEqual(prefix["sequence_input"], "sequence_ids")
                self.assertEqual(prefix["chunk_mask_input"], "chunk_attention_mask")

    def test_compact_attention_and_head_rope_admission(self):
        for target in TARGETS:
            with self.subTest(target=target):
                self.assertTrue(head_attention_backend_supported("CrossSDPA", {"heads": 4, "kv_heads": 1}, target))
                self.assertTrue(head_attention_backend_supported("RoPE", {"heads": 4}, target))


class ModernIncrementalCorpusTests(unittest.TestCase):
    def test_v4_canonical_reference_preserves_raw_and_matches_native_long_prefixes(self):
        try:
            import torch
            from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM
        except ImportError:
            self.skipTest("independent HF DeepSeek V4 reference unavailable")
        from contextlib import ExitStack
        import base64
        from safetensors.numpy import save
        from tools.llm_baseline.generate_modern_incremental_stress import (
            deepseek_v4_reference_policy, incremental_fixture,
            initial_values, reference_rows)
        from tools.llm_baseline.verify_modern_incremental_native import replay, vx
        self.assertTrue(callable(DeepseekV4ForCausalLM))
        with ExitStack() as scope:
            model, config, graph, weights = incremental_fixture("deepseek_v4", scope, capacity=128, seed=746)
            prompt = [2 + (row * 7 + row // 5) % 30 for row in range(31)]
            indexer = model.model.layers[2].self_attn.compressor.indexer
            raw_selected = []
            with ExitStack() as observation:
                observation.callback(indexer.register_forward_hook(
                    lambda module, args, out: raw_selected.append(out.detach().numpy().copy())).remove)
                raw, _ = reference_rows(model, prompt, 1)
            selected, observed_scores = [], []
            with deepseek_v4_reference_policy(model, "canonical-lowest-index"):
                with ExitStack() as observation:
                    observation.callback(indexer.scorer.register_forward_hook(
                        lambda module, args, out: observed_scores.append(out.detach().numpy().copy())).remove)
                    observation.callback(indexer.register_forward_hook(
                        lambda module, args, out: selected.append(out.detach().numpy().copy())).remove)
                    canonical, _ = reference_rows(model, prompt, 1)
                scores = observed_scores[0][0, -1]
                # The retained failing seed remains a zero-gap case, without an epsilon tie rule.
                np.testing.assert_array_equal(np.flatnonzero(scores == scores.max()), [1, 4, 6, 7, 11])
                self.assertEqual(selected[0][0, -1, 0], 1)
                if raw_selected[0][0, -1, 0] != 1:
                    self.assertGreater(float(np.max(np.abs(raw[0] - canonical[0]))), 2e-4)
                non_tied = 0
                for row in range(len(prompt)):
                    ready = (row + 1) // indexer.compress_rate
                    ordered = np.sort(observed_scores[0][0, row, :ready])
                    if ready == 1 or (ready > 1 and ordered[-1] > ordered[-2]):
                        np.testing.assert_array_equal(selected[0][0, row], raw_selected[0][0, row])
                        non_tied += 1
                self.assertGreater(non_tied, 0)
                for length in (31, 63, 127):
                    prefix = [2 + (row * 7 + row // 5) % 30 for row in range(length)]
                    expected, ids = reference_rows(model, prefix, 2)
                    values = initial_values(graph, config, prefix)
                    case = {"name":f"deepseek_v4-prompt{length}", "graph":graph,
                        "reference_policy":"canonical-lowest-index",
                        "weight_shards":[base64.b64encode(save(weights, metadata={"format":"pt"})).decode()],
                        "inputs":[{"name":name, "shape":list(value.shape), "dtype":str(value.dtype),
                            "data":base64.b64encode(value.tobytes()).decode()} for name,value in values.items()],
                        "expected_logits":[value.tolist() for value in expected], "expected_ids":ids, "budgets":[1, 1]}
                    for profile in ("volvoxai", "volvoxai-lite"):
                        with self.subTest(length=length, profile=profile), vx.open_library(ROOT / "native" / f"lib{profile}.so") as host:
                            self.assertTrue(replay(case, host, "cpu")["pass"])
            restored, _ = reference_rows(model, prompt, 1)
            np.testing.assert_array_equal(restored, raw)
            with self.assertRaisesRegex(RuntimeError, "unwind"):
                with deepseek_v4_reference_policy(model, "canonical-lowest-index"):
                    raise RuntimeError("unwind")
            np.testing.assert_array_equal(reference_rows(model, prompt, 1)[0], raw)
            with self.assertRaises(ValueError):
                with deepseek_v4_reference_policy(model, "unknown"):
                    pass
            with self.assertRaises(ValueError):
                with deepseek_v4_reference_policy(torch.nn.Linear(2, 2), "canonical-lowest-index"):
                    pass

    def test_single_candidate_indexer_keeps_small_capacity_oracle(self):
        try:
            from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM
        except ImportError:
            self.skipTest("independent HF DeepSeek V4 reference unavailable")
        self.assertTrue(callable(DeepseekV4ForCausalLM))
        with tempfile.TemporaryDirectory(prefix="volvox-modern-single-index-") as directory:
            corpus, receipt = Path(directory) / "corpus.json", Path(directory) / "receipt.json"
            process = subprocess.run([sys.executable, "-B",
                str(ROOT / "tools/llm_baseline/generate_modern_incremental_stress.py"),
                "--out", str(corpus), "--receipt", str(receipt), "--capacity", "16",
                "--prompts", "3", "--steps", "1", "--budgets", "1",
                "--families", "deepseek_v4", "--with-boundary"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stdout)
            result = json.loads(receipt.read_text())
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["reference_policy"], "raw-hf")
            # Longer boundary rows still collect their real top-two gaps.
            gaps = result["deepseek_v4_indexer_gaps"]
            self.assertGreater(gaps["count"], 0)
            self.assertEqual(gaps["count"], len(gaps["values"]))
            self.assertTrue(all(np.isfinite(value) and value >= 0 for value in gaps["values"]))
            cases = json.loads(corpus.read_text())["cases"]
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["name"], "deepseek_v4-prompt3")
            for scope, count in ((cases[0], 1), (cases[0]["boundary"], 2)):
                logits = np.asarray(scope["expected_logits"])
                self.assertEqual(logits.shape[0], count)
                self.assertTrue(np.isfinite(logits).all())
                self.assertEqual(logits.argmax(axis=-1).tolist(), scope["expected_ids"])


class NativeCapacityWorkerTests(unittest.TestCase):
    def test_vulkan_base_and_scratch_are_included_and_fixed_pair_is_per_owner(self):
        estimate = {"host_scheduling_bytes": 100, "device_scheduling_bytes": 200, "per_context_scheduling_bytes": 50}
        plan = {"live_contexts": 2, "compile_probe_scheduling_estimate": copy.deepcopy(estimate),
                "scheduling_estimate": copy.deepcopy(estimate)}
        reserve_vulkan_allocations(plan, 1024)
        pair = (1024 + 32) << 20
        self.assertEqual(plan["compile_probe_scheduling_estimate"]["device_scheduling_bytes"], 200 + pair)
        self.assertEqual(plan["scheduling_estimate"]["host_scheduling_bytes"], 100 + 2 * pair)
        self.assertEqual(plan["scheduling_estimate"]["device_scheduling_bytes"], 200 + 2 * pair)
        self.assertEqual(plan["scheduling_estimate"]["per_context_scheduling_bytes"], 50 + pair)
        self.assertEqual(plan["vulkan_fixed_allocation_reserve"]["usable_graph_region_before_alignment_bytes"], (1024 - 128 - 1) << 20)

    def fixture(self, directory, script):
        root = Path(directory)
        (root / "tools/llm_baseline").mkdir(parents=True)
        (root / "tools/llm_baseline/measure_native_capacity.py").write_text(script)
        (root / "native").mkdir()
        for name in ("libvolvoxai.so", "libsynurang_module_host.so"):
            (root / "native" / name).write_bytes(b"test-only, never loaded")
        plan = root / "plan.json"
        plan.write_text(json.dumps({"pins": {"model": "smollm2-135m", "backend": "cpu", "profile": "full"},
                                    "capacity": 128, "live_contexts": 1}))
        out = root / "build/llm/worker"
        return root, plan, out

    def command(self, root, plan, out):
        return [sys.executable, str(WRAPPER), "--root", str(root), "--out", str(out),
                "--model", "smollm2-135m", "--backend", "cpu", "--stage", "memory", "--plan", str(plan)]

    def test_compile_only_receipt_cannot_qualify_execution(self):
        fake = """import json, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])
out.write_text(json.dumps({'status': 'PROOF_CAPTURED', 'system_swap_delta': {}, 'private_tmp_remaining': []}))
"""
        with tempfile.TemporaryDirectory(prefix="volvoxai-remaining-validation-test-") as directory:
            root, plan, out = self.fixture(directory, fake)
            process = subprocess.run(self.command(root, plan, out), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, timeout=10)
            result = json.loads((out / "result.json").read_text())
            self.assertEqual(process.returncode, 1, process.stdout)
            self.assertIn("does not qualify the requested stage", result["error"])
            self.assertEqual(result["temporary_owned_paths_remaining"], [])

    def test_timeout_terminates_worker_and_cleans_private_snapshot(self):
        fake = """import os, pathlib, sys, time
tmp = pathlib.Path(sys.argv[sys.argv.index('--tmpdir') + 1])
(tmp / 'owned-snapshot').write_bytes(b'test owner')
pathlib.Path(__file__).parents[2].joinpath('worker-pid').write_text(str(os.getpid()))
time.sleep(30)
"""
        with tempfile.TemporaryDirectory(prefix="volvoxai-remaining-validation-test-") as directory:
            root, plan, out = self.fixture(directory, fake)
            process = subprocess.run(self.command(root, plan, out) + ["--timeout-seconds", "1"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
            result = json.loads((out / "result.json").read_text())
            self.assertEqual(process.returncode, 1, process.stdout)
            self.assertIn("owned worker timed out", result["error"])
            self.assertEqual(result["temporary_owned_paths_remaining"], [])
            worker_pid = int((root / "worker-pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(worker_pid, 0)


if __name__ == "__main__":
    unittest.main()

