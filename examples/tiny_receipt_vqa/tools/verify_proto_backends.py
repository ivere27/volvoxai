#!/usr/bin/env python3
"""Verify split VQA conversion, explicit KV decoding and C PTQ through proto.

The corpus stores the exact resized pixels and question token IDs shared by
native, WASM and WebGPU. ORT supplies independent component fixtures and full
autoregressive references. Calibration images are disjoint from all evaluation
images by normalized content, and decoder calibration uses generated prefixes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time
import unicodedata

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
import volvoxai as vx  # noqa: E402
from examples.tiny_receipt_vqa.tools.tiny_receipt_tokenizer import TinyReceiptTokenizer  # noqa: E402

pb = vx.pb
HEIGHT, WIDTH = 320, 672
DTYPES = {pb.DataType.DATA_TYPE_F32: np.dtype("float32"),
          pb.DataType.DATA_TYPE_I32: np.dtype("int32")}
ROLES = ("encoder", "decoder")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def clean(text):
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", str(text)).strip())


def normalize(gray):
    return np.ascontiguousarray(((gray.astype(np.float32) / 255 - .5) / .5).reshape(1, 1, HEIGHT, WIDTH))


def tokenizer(package):
    manifest = json.loads((package / "package_manifest.json").read_text())
    return TinyReceiptTokenizer.from_documents(manifest["tokenizer"], json.loads((package / "vocab.json").read_text()))


def attest(report, backend):
    assert report.backend == backend, report.to_dict()
    assert report.route and report.route.attested and report.route.provider == backend, report.to_dict()
    assert report.route.active_nodes == report.route.selected_nodes, report.to_dict()
    assert report.route.missing_nodes == report.route.fallback_nodes == 0, report.to_dict()
    assert not (report.fallback and report.fallback.operator_fallback_used), report.to_dict()
    assert not (report.compilation and report.compilation.tier_fallback_used), report.to_dict()


def narrow(values):
    return np.ascontiguousarray(values, dtype=np.float32 if values.dtype.kind == "f" else np.int32)


def tensors(feed, mapping):
    return [pb.Tensor(name=mapping[name], shape=list(value.shape),
                      dtype=pb.DataType.DATA_TYPE_F32 if value.dtype.kind == "f" else pb.DataType.DATA_TYPE_I32,
                      inline=narrow(value).tobytes()) for name, value in feed.items()]


def save_tensors(path, feed):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata, offset = {}, 0
    with path.with_suffix(".bin").open("wb") as stream:
        for name, value in feed.items():
            value = narrow(value)
            metadata[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                              "offset": offset, "bytes": value.nbytes}
            stream.write(value.tobytes())
            offset += value.nbytes
    save_json(path.with_suffix(".json"), metadata)


def read_tensors(path):
    data = path.with_suffix(".bin").read_bytes()
    return {name: np.frombuffer(data, dtype=item["dtype"], count=item["bytes"] // 4,
                               offset=item["offset"]).reshape(item["shape"]).copy()
            for name, item in json.loads(path.with_suffix(".json").read_text()).items()}


class Runner:
    def __init__(self, package, backend="cpu", profile="inference", threads=1):
        self.host = vx.open_library(profile)
        self.client = vx.VxInferenceServiceClient(self.host)
        self.backend = backend
        self.manifest = json.loads((package / "package_manifest.json").read_text())
        self.contexts, self.reports, self.executions = {}, {}, Counter()
        try:
            runtime = self.client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=threads))
            for role in ROLES:
                assets = self.manifest["graphs"][role]
                model = self.client.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                    package=pb.ModelPackage(graph_document=(package / assets["graph"]["path"]).read_bytes(),
                        weight_shards=[(package / assets["weights"]["path"]).read_bytes()])))
                compiled = self.client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
                    policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                        backends=[backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
                attest(compiled.report, backend)
                self.reports[role] = {"compilation": compiled.report.to_dict()}
                self.contexts[role] = self.client.create_execution_context(pb.CreateExecutionContextRequest(compiled_model_id=compiled.compiled_model_id)).context_id
        except BaseException:
            self.close()
            raise

    def run(self, role, feed):
        contract = self.manifest["graphs"][role]
        assert set(feed) == set(contract["inputs"])
        execution = self.client.execute(pb.ExecuteRequest(context_id=self.contexts[role],
                            inputs=tensors(feed, contract["inputs"])))
        try:
            attest(execution.report, self.backend)
            self.reports[role].setdefault("execution", execution.report.to_dict())
            deadline = time.monotonic() + 120
            while True:
                info = self.client.get_result(pb.ResultRef(result_id=execution.result_id))
                if info.state != pb.ResultState.RESULT_STATE_PENDING:
                    assert info.state == pb.ResultState.RESULT_STATE_READY, info.to_dict()
                    break
                assert time.monotonic() < deadline, "result timed out"
                time.sleep(.001)
            result = {}
            for semantic, name in contract["outputs"].items():
                output = self.client.read_output(pb.ReadOutputRequest(result_id=execution.result_id, name=name)).tensor
                value = np.frombuffer(output.inline, dtype=DTYPES[output.dtype]).copy().reshape(output.shape)
                assert np.isfinite(value).all(), name
                result[semantic] = value
            self.executions[role] += 1
            return result
        finally:
            self.client.release_result(pb.ResultRef(result_id=execution.result_id))

    def close(self):
        self.host.close()


class OrtRunner:
    def __init__(self, source, variant, optimize=True):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        if not optimize:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.quantized_optimizations = optimize and variant == "int8"
        suffix = "_int8" if variant == "int8" else ""
        self.sessions = {role: ort.InferenceSession(str(source / f"{role}_model{suffix}.onnx"),
            opts, providers=["CPUExecutionProvider"]) for role in ROLES}

    def run(self, role, feed):
        session = self.sessions[role]
        dtype = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
        inputs = {item.name: np.asarray(feed[item.name], dtype=dtype[item.type]) for item in session.get_inputs()}
        return {item.name: narrow(value) for item, value in zip(session.get_outputs(), session.run(None, inputs))}

    def close(self):
        self.sessions.clear()


def encoder_feed(image, ids, family=-1):
    return {"image": image, "question_ids": np.asarray([ids], np.int32),
            "question_position_ids": np.arange(len(ids), dtype=np.int32)[None],
            "family_ids": np.asarray([family], np.int32)}


def decoder_feed(encoded, past_length=1):
    result = {name: value for name, value in encoded.items() if name.startswith("cross_") or name == "memory_padding_mask"}
    result.update(decoder_input_ids=np.asarray([[1]], np.int32), position_ids=np.asarray([0], np.int32),
                  family_ids=encoded["selected_family_ids"],
                  past_padding_mask=np.ones((1, past_length), np.int32))
    for layer in range(4):
        for kind in ("k", "v"):
            result[f"past_{kind}_{layer}"] = np.zeros((1, 8, past_length, 40), np.float32)
    return result


def advance(feed, output, token, position, check_prefix=True):
    past = feed["past_padding_mask"].shape[1]
    assert output["present_padding_mask"].shape == (1, past + 1)
    assert np.array_equal(output["present_padding_mask"][:, :past], feed["past_padding_mask"])
    assert np.all(output["present_padding_mask"][:, past:] == 0)
    for layer in range(4):
        for kind in ("k", "v"):
            before, after = f"past_{kind}_{layer}", f"present_{kind}_{layer}"
            assert output[after].shape == (1, 8, past + 1, 40)
            if check_prefix:
                assert np.array_equal(output[after][:, :, :past], feed[before]), f"cache prefix changed: {after}"
            feed[before] = output[after]
    feed["past_padding_mask"] = output["present_padding_mask"]
    feed["decoder_input_ids"] = np.asarray([[token]], np.int32)
    feed["position_ids"] = np.asarray([position + 1], np.int32)


def generate(runner, image, ids, observe=None):
    enc_feed = encoder_feed(image, ids)
    encoded = runner.run("encoder", enc_feed)
    if observe:
        observe("encoder", enc_feed, 0)
    # Independent original ONNX decoding starts with genuinely empty KV.
    feed = decoder_feed(encoded, 0 if isinstance(runner, OrtRunner) else 1)
    tokens = [1]
    for position in range(191):
        if observe:
            observe("decoder", feed, position)
        output = runner.run("decoder", feed)
        assert output["logits"].shape == (1, 1, 1536)
        token = int(output["logits"].argmax(-1)[0, 0])
        tokens.append(token)
        advance(feed, output, token, position,
                not (isinstance(runner, OrtRunner) and runner.quantized_optimizations))
        if token == 2:
            break
    return {"tokens": tokens, "family": int(encoded["selected_family_ids"][0]), "eos": tokens[-1] == 2}


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    result = {}
    for name, value in actual.items():
        reference = expected[name]
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.isfinite(value).all() and np.isfinite(reference).all(), name
        delta = value.astype(np.float64) - reference.astype(np.float64)
        result[name] = {"max_abs": float(np.abs(delta).max()), "rmse": float(np.sqrt(np.mean(delta ** 2))),
                        "exact": bool(np.array_equal(value, reference)),
                        "close_1e_4": bool(np.allclose(value, reference, atol=1e-4, rtol=1e-4))}
    if "logits" in actual:
        result["logits"]["argmax_equal"] = bool(np.array_equal(actual["logits"].argmax(-1), expected["logits"].argmax(-1)))
    return result


def prepare(args):
    from PIL import Image
    corpus = args.work / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(args.work / "fp32")
    def gray(path):
        with Image.open(path) as image:
            return np.asarray(image.convert("L").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))
    all_eval = sorted(args.eval_annotations.glob("*.json"))
    assert all_eval
    seen_eval, records = set(), []
    with (corpus / "images.u8").open("wb") as stream:
        for path in all_eval:
            pixels = gray(args.eval_images / f"{path.stem}.jpg")
            seen_eval.add(hashlib.sha256(normalize(pixels).tobytes()).hexdigest())
            stream.write(pixels.tobytes())
            doc = json.loads(path.read_text())
            records.append({"id": path.stem, "question_ids": tok.encode(clean(doc["question"]), add_eos=True, max_len=192),
                            "answer": clean(doc["answer"])})
    candidates = sorted(args.calibration_annotations.glob("*.json"))
    random.Random(args.seed).shuffle(candidates)
    selected, seen_calibration = [], set()
    with (corpus / "calibration-images.u8").open("wb") as stream:
        for path in candidates:
            pixels = gray(args.calibration_images / f"{path.stem}.jpg")
            identity = hashlib.sha256(normalize(pixels).tobytes()).hexdigest()
            if identity in seen_eval or identity in seen_calibration:
                continue
            doc = json.loads(path.read_text())
            seen_calibration.add(identity)
            selected.append({"id": path.stem, "question_ids": tok.encode(clean(doc["question"]), add_eos=True, max_len=192)})
            stream.write(pixels.tobytes())
            if len(selected) == args.calibration_count:
                break
    assert len(selected) == args.calibration_count
    manifest = {"schema": "volvoxai.vqa-proto-corpus/v1", "seed": args.seed, "evaluation": records,
                "calibration": selected, "calibration_eval_overlap": 0,
                "all_evaluation_input_sha256": sorted(seen_eval), "calibration_input_sha256": sorted(seen_calibration),
                "images_sha256": digest(corpus / "images.u8"), "calibration_images_sha256": digest(corpus / "calibration-images.u8")}
    save_json(corpus / "corpus.json", manifest)
    print(f"prepared {len(records)} evaluation and {len(selected)} disjoint calibration images", flush=True)


def fixtures(args):
    corpus = json.loads((args.work / "corpus/corpus.json").read_text())
    images = np.memmap(args.work / "corpus/images.u8", dtype=np.uint8, mode="r", shape=(len(corpus["evaluation"]), HEIGHT, WIDTH))
    cases = []
    for index, family in enumerate(range(-1, 8)):
        cases.append({"id": str(index), "family": family, "Q": [1, 16, 64, 192, 8, 16, 64, 192, 8][index],
                      "P": [1, 2, 8, 191, 2, 64, 1, 191, 2][index]})
    for variant in ("fp32", "int8"):
        # ORT's QDQ optimizer can move quantization across Concat and change
        # the public float KV prefix. Component references use the ONNX graph
        # as authored; end-to-end references separately exercise default ORT.
        runner = OrtRunner(args.source, variant, optimize=False)
        try:
            for case in cases:
                folder = args.work / "fixtures" / variant / case["id"]
                ids = np.resize(corpus["evaluation"][int(case["id"])]["question_ids"], case["Q"]).tolist()
                feed = encoder_feed(normalize(images[int(case["id"])]), ids, case["family"])
                encoded = runner.run("encoder", feed)
                save_tensors(folder / "encoder-inputs", feed)
                save_tensors(folder / "encoder-outputs", encoded)
                feed = decoder_feed(encoded, case["P"])
                feed["position_ids"][:] = case["P"] - 1
                feed["past_padding_mask"][:, 1:] = 0
                rng = np.random.default_rng(20260913 + int(case["id"]))
                for name, value in feed.items():
                    if name.startswith("past_k_") or name.startswith("past_v_"):
                        value[:, :, 1:] = rng.normal(0, .1, value[:, :, 1:].shape)
                decoded = runner.run("decoder", feed)
                advance(dict(feed), decoded, 1, case["P"] - 1, check_prefix=variant != "int8")
                save_tensors(folder / "decoder-inputs", feed)
                save_tensors(folder / "decoder-outputs", decoded)
        finally:
            runner.close()
    save_json(args.work / "fixtures/cases.json", cases)
    print("generated nine component cases per ONNX pair; all router families, Q/P growth and shrink", flush=True)


def score(predictions, records, tok):
    correct, formed = 0, 0
    for prediction, record in zip(predictions, records):
        match = re.search(r"<answer>(.*?)</answer>", tok.decode(prediction["tokens"]))
        formed += match is not None
        correct += bool(match and clean(match.group(1)) == record["answer"])
    return {"answer_exact": correct, "well_formed": formed, "samples": len(predictions),
            "answer_exact_rate": correct / len(predictions), "eos": sum(item["eos"] for item in predictions)}


def run(args):
    corpus = json.loads((args.work / "corpus/corpus.json").read_text())
    count = min(args.limit or len(corpus["evaluation"]), len(corpus["evaluation"]))
    images = np.memmap(args.work / "corpus/images.u8", dtype=np.uint8, mode="r", shape=(len(corpus["evaluation"]), HEIGHT, WIDTH))
    report = {"schema": "volvoxai.vqa-proto-verification/v1", "backend": args.backend, "profile": args.profile,
              "samples": count, "cpu_threads": args.threads, "corpus_sha256": digest(args.work / "corpus/corpus.json"), "variants": {}}
    if args.backend != "ort":
        report["library_sha256"] = digest(vx.find_library(args.profile))
    for variant in args.variants:
        runner, entry = None, {"status": "failed", "stage": "load"}
        report["variants"][variant] = entry
        try:
            package = args.work / variant
            entry["packages"] = {role: {name: digest(package / role / name) for name in ("graph.json", "model.safetensors")} for role in ROLES}
            runner = OrtRunner(args.source, variant, not args.ort_disable_optimizations) if args.backend == "ort" else Runner(package, args.backend, args.profile, args.threads)
            if args.backend == "ort":
                entry["ort_graph_optimizations"] = "disabled" if args.ort_disable_optimizations else "all"
            if args.backend != "ort" and not args.skip_fixtures:
                entry["stage"], entry["fixtures"] = "component fixtures", {}
                reference_variant = variant if variant in ("fp32", "int8") else "fp32"
                for case in json.loads((args.work / "fixtures/cases.json").read_text()):
                    for role in ROLES:
                        folder = args.work / "fixtures" / reference_variant / case["id"]
                        feed = read_tensors(folder / f"{role}-inputs")
                        output = runner.run(role, feed)
                        entry["fixtures"][f"{role}-{case['id']}"] = compare(output, read_tensors(folder / f"{role}-outputs"))
                        if role == "decoder":
                            advance(feed, output, 1, case["P"] - 1, check_prefix=variant != "int8")
            entry["stage"], entry["predictions"] = "autoregressive", []
            started = time.perf_counter()
            for index, record in enumerate(corpus["evaluation"][:count]):
                prediction = generate(runner, normalize(images[index]), record["question_ids"])
                entry["predictions"].append(prediction)
                if (index + 1) % 16 == 0 or index + 1 == count:
                    entry["seconds"] = time.perf_counter() - started
                    print(f"{args.backend}/{args.profile}/{variant}: {index + 1}/{count}, {entry['seconds']:.1f}s", flush=True)
                    save_json(args.out, report)
            entry.update(status="executed", stage="complete", seconds=time.perf_counter() - started,
                         score=score(entry["predictions"], corpus["evaluation"], tokenizer(args.work / "fp32")))
            if isinstance(runner, Runner):
                entry["routes"], entry["executions"] = runner.reports, dict(runner.executions)
            print(variant, json.dumps(entry["score"]), flush=True)
        except Exception as error:
            entry["error"] = str(error)
            print(variant, type(error).__name__, str(error), flush=True)
        finally:
            if runner:
                runner.close()
            save_json(args.out, report)
    return int(any(item["status"] != "executed" for item in report["variants"].values()))


def calibrate(args):
    corpus = json.loads((args.work / "corpus/corpus.json").read_text())
    images = np.memmap(args.work / "corpus/calibration-images.u8", dtype=np.uint8, mode="r", shape=(len(corpus["calibration"]), HEIGHT, WIDTH))
    package, output = args.work / "fp32", args.work / "ptq"
    output.mkdir(exist_ok=True)
    runner = Runner(package, "cpu", "full", args.threads)
    host = vx.open_library("full")
    report = {"schema": "volvoxai.vqa-proto-ptq/v1", "library_sha256": digest(vx.find_library("full")), "roles": {}}
    try:
        inference, quantization = vx.VxInferenceServiceClient(host), vx.VxQuantizationServiceClient(host)
        runtime = inference.create_runtime(pb.CreateRuntimeRequest(cpu_threads=args.threads))
        plans, counts, maps = {}, Counter(), {}
        for role in ROLES:
            target = output / role
            target.mkdir(exist_ok=True)
            graph_bytes, weights = (package / role / "graph.json").read_bytes(), (package / role / "model.safetensors").read_bytes()
            graph = json.loads(graph_bytes)
            selected = [node["id"] for node in graph["nodes"] if node["opType"] in ("Conv2D", "Linear", "Gemm")]
            config = pb.PtqAuthoringConfig(activation_dtype=pb.DataType.DATA_TYPE_I8,
                activation_scheme=pb.PtqScheme.PTQ_SCHEME_ASYMMETRIC, weight_dtype=pb.DataType.DATA_TYPE_I8,
                selected_nodes=selected)
            authored = quantization.author_ptq_template(pb.AuthorPtqTemplateRequest(
                source_graph=graph_bytes, weight_shards=[weights], config=config))
            (target / "author-config.pb").write_bytes(config.to_bytes())
            (target / "template.graph.json").write_bytes(authored.template_graph)
            model = inference.load_model(pb.LoadModelRequest(runtime_id=runtime.runtime_id,
                package=pb.ModelPackage(graph_document=graph_bytes, weight_shards=[weights])))
            request = pb.CreatePtqPlanRequest(model_id=model.model_id, template_graph=authored.template_graph,
                observers=list(authored.observers), layers=list(authored.layers), profile_names=["default"])
            plans[role] = quantization.create_ptq_plan(request).ptq_plan_id
            request.model_id = 0
            (target / "plan.pb").write_bytes(request.to_bytes())
            maps[role] = runner.manifest["graphs"][role]["inputs"]
            report["roles"][role] = {"selected_nodes": selected, "quantized_nodes": authored.quantized_nodes,
                "retained_float_nodes": authored.retained_float_nodes, "observers": len(authored.observers)}
        started = time.perf_counter()
        if args.replay_calibration:
            for role in ROLES:
                paths = sorted((args.work / "calibration" / role).glob("*.pb"))
                assert paths, f"missing calibration requests: {role}"
                for path in paths:
                    request = pb.CalibratePtqPlanRequest.from_bytes(path.read_bytes())
                    request.ptq_plan_id = plans[role]
                    quantization.calibrate_ptq_plan(request)
                    counts[role] += 1
                    if counts[role] % 32 == 0:
                        print(f"C PTQ replay {role}: {counts[role]}/{len(paths)}", flush=True)
        for index, record in enumerate([] if args.replay_calibration else corpus["calibration"]):
            def observe(role, feed, position):
                # Three actual prefixes per receipt cover initial and later KV;
                # generated tokens are independent of heldout target answers.
                if role == "decoder" and position not in (0, 4, 12):
                    return
                request = pb.CalibratePtqPlanRequest(ptq_plan_id=plans[role], profile_name="default",
                    sample_name=f"calibration-{index}-step-{position}", sample_count=1, inputs=tensors(feed, maps[role]))
                quantization.calibrate_ptq_plan(request)
                request.ptq_plan_id = 0
                folder = args.work / "calibration" / role
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{counts[role]:04d}.pb").write_bytes(request.to_bytes())
                counts[role] += 1
            generate(runner, normalize(images[index]), record["question_ids"], observe)
            if (index + 1) % 16 == 0:
                print(f"C PTQ calibration {index + 1}/{len(corpus['calibration'])}, {time.perf_counter() - started:.1f}s", flush=True)
        for role in ROLES:
            state = quantization.inspect_ptq_plan(pb.PtqPlanRef(ptq_plan_id=plans[role]))
            assert state.coverage.complete and state.calibration_samples == counts[role]
            packed = quantization.write_ptq_package(pb.WritePtqPackageRequest(ptq_plan_id=plans[role]))
            (output / role / "graph.json").write_bytes(packed.graph)
            (output / role / "model.safetensors").write_bytes(packed.weights)
            report["roles"][role].update(calibration_samples=counts[role], coverage_complete=True, plan=state.to_dict(),
                graph_sha256=digest(output / role / "graph.json"), weights_sha256=digest(output / role / "model.safetensors"))
        manifest = json.loads((package / "package_manifest.json").read_text())
        manifest["variant"] = {"requested": "native-c-ptq", "activation": "I8 asymmetric", "selected_operators": ["Conv2D", "Linear", "Gemm"]}
        manifest["validation"] = {"calibration_eval_overlap": 0, "strict_runtime_execution": "not-run"}
        for role in ROLES:
            for kind, name in (("graph", "graph.json"), ("weights", "model.safetensors")):
                asset = output / role / name
                manifest["graphs"][role][kind] = {"path": f"{role}/{name}", "bytes": asset.stat().st_size, "sha256": digest(asset)}
            manifest["graphs"][role].pop("export_report", None)
        save_json(output / "package_manifest.json", manifest)
        for name in ("config.json", "vocab.json"):
            shutil.copyfile(package / name, output / name)
        report["seconds"] = time.perf_counter() - started
        save_json(args.work / "native-ptq.json", report)
    finally:
        runner.close()
        host.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fixtures", "run", "calibrate"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--eval-images", type=Path)
    parser.add_argument("--eval-annotations", type=Path)
    parser.add_argument("--calibration-images", type=Path)
    parser.add_argument("--calibration-annotations", type=Path)
    parser.add_argument("--calibration-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--backend", choices=("ort", "cpu", "cuda", "vulkan", "opengl"), default="cpu")
    parser.add_argument("--profile", choices=("inference", "full"), default="inference")
    parser.add_argument("--variants", nargs="+", default=["fp32", "int8", "ptq", "ptq-wasm"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="zero runs every evaluation record")
    parser.add_argument("--skip-fixtures", action="store_true")
    parser.add_argument("--ort-disable-optimizations", action="store_true")
    parser.add_argument("--replay-calibration", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.limit < 0 or args.calibration_count < 1 or args.threads < 1:
        parser.error("counts must be positive; limit may be zero")
    args.work = args.work.resolve()
    return globals()[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
