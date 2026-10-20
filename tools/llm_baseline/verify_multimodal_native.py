#!/usr/bin/env python3
"""Pinned raw JPEG/PCM workflows through the public native CPU API.

Whisper uses the official HF NumPy feature branch and an explicit fixed caller
greedy policy, not the complete HF Whisper.generate transcription pipeline.
All packages are ordinary graphs; media preprocessing stays in this caller.
"""
from __future__ import annotations

from google.protobuf.json_format import MessageToDict

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import volvoxai as vx
from tools.whisper_audio import prepare_whisper_features
from tools.receipt_image import prepare_receipt_image
from tools.exporter.execution_contract import stateless_execution, tensor_feedback_execution, full_prefix_decoder_execution

pb = vx.pb


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): value.update(block)
    return value.hexdigest()


def array_hash(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compare(actual, expected, *, atol, rtol, rms=None):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    delta = np.asarray(actual, np.float64) - np.asarray(expected, np.float64)
    error = float(np.sqrt(np.mean(delta * delta)))
    if rms is not None: assert error <= rms, (error, rms)
    return {"shape": list(actual.shape), "maximum_absolute_error": float(np.abs(delta).max()),
        "rms_error": error, "atol": atol, "rtol": rtol, "actual_sha256": array_hash(actual),
        "reference_sha256": array_hash(expected)}


def tensors(inputs, *, borrowed=False):
    result = []
    for name, value in inputs.items():
        assert value.flags.c_contiguous and value.dtype in (np.dtype("float32"), np.dtype("int32"))
        fields = {"name": name, "shape": list(value.shape), "dtype":
            pb.DataType.DATA_TYPE_F32 if value.dtype.kind == "f" else pb.DataType.DATA_TYPE_I32}
        if borrowed:
            fields["borrowed"] = pb.BorrowedBuffer(resource=pb.NativeResource(
                kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST, handle=value.ctypes.data,
                size_bytes=value.nbytes), length_bytes=value.nbytes)
        else: fields["inline"] = value.tobytes()
        result.append(pb.Tensor(**fields))
    return result


def attest(report, backend="cpu"):
    assert report.backend == backend and report.route.attested and report.route.provider == backend
    assert report.route.active_nodes == report.route.selected_nodes
    assert report.route.fallback_nodes == report.route.missing_nodes == 0
    assert not (report.fallback and report.fallback.operator_fallback_used)
    return {"active_nodes": report.route.active_nodes, "selected_nodes": report.route.selected_nodes,
        "digest": str(report.route.digest)}


class Session:
    def __init__(self, profile, directory, *, library_dir=None, backend="cpu"):
        self.resources = ExitStack()
        self.library_dir = library_dir or ROOT / "native"
        self.backend = backend
        self.host = self.resources.enter_context(vx.open_library(self.library_dir /
            ("libvolvoxai-lite.so" if profile == "inference" else "libvolvoxai.so")))
        try:
            platform = vx.VxPlatformServiceClient(self.host).get_platform_info(pb.Empty())
            expected = (pb.BuildProfile.BUILD_PROFILE_INFERENCE if profile == "inference"
                else pb.BuildProfile.BUILD_PROFILE_FULL)
            assert platform.build_profile == expected, "loaded library profile differs from requested profile"
            self.platform = MessageToDict(platform, preserving_proto_field_name=True)
        except BaseException:
            self.resources.close()
            raise
        self.client = vx.VxInferenceServiceClient(self.host)
        self.buffers = vx.VxBufferServiceClient(self.host)
        self.runtime = self.client.create_runtime(pb.CreateRuntimeRequest(cpu_threads=1)).runtime_id
        self.resources.callback(self.client.release_runtime, pb.RuntimeRef(runtime_id=self.runtime))
        self.directory, self.models, self.routes = Path(directory), {}, []
        self.encoder_calls = 0
        self.owned_inventory = []

    def model(self, role, graph, weights):
        from safetensors.numpy import save_file
        path = self.directory / role
        path.mkdir()
        (path / "graph.json").write_text(json.dumps(graph, allow_nan=False))
        save_file(weights, str(path / "model.safetensors"))
        return self.load(role, path / "graph.json", path / "model.safetensors")

    def load(self, role, graph_path, weights_path):
        model = self.client.load_model(pb.LoadModelRequest(runtime_id=self.runtime,
            graph_path=str(graph_path), weight_paths=[str(weights_path)]))
        self.resources.callback(self.client.release_model, pb.ModelRef(model_id=model.model_id))
        compiled = self.client.compile_model(pb.CompileModelRequest(model_id=model.model_id,
            policy=pb.BackendPolicy(mode=pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
                backends=[self.backend], operator_fallback=pb.OperatorFallback.OPERATOR_FALLBACK_FORBID)))
        self.routes.append(attest(compiled.report, self.backend))
        self.resources.callback(self.client.release_compiled_model,
            pb.CompiledModelRef(compiled_model_id=compiled.compiled_model_id))
        self.models[role] = compiled.compiled_model_id
        return {"graph_sha256": sha(graph_path), "weights_sha256": sha(weights_path),
            "weights_bytes": Path(weights_path).stat().st_size}

    def context(self, role, *, generate=False, steps=3, eos_ids=()):
        options = {"autoregressive": pb.AutoregressiveExecution(sampling=pb.SamplingMode.SAMPLING_MODE_GREEDY,
            max_steps=steps, eos_ids=list(eos_ids))} if generate else {"graph": pb.GraphExecution()}
        context = self.client.create_execution_context(pb.CreateExecutionContextRequest(
            compiled_model_id=self.models[role], **options)).context_id
        self.resources.callback(self.client.release_execution_context, pb.ExecutionContextRef(context_id=context))
        return context

    def execute(self, context, inputs, *, encoder=False, owned=False, entrypoint="evaluate", reuse=()):
        packet = self.client.execute(pb.ExecuteRequest(context_id=context,
            inputs=pb.InputBindings(values=tensors(inputs, borrowed=True), reuse=list(reuse)),
            scope=pb.ExecutionScope(entrypoint=entrypoint, full=pb.Empty()),
            outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER if owned
                else pb.OutputStorage.OUTPUT_STORAGE_INLINE)))
        self.routes.append(attest(packet.report, self.backend))
        self.encoder_calls += int(encoder)
        if owned:
            self.resources.callback(self.buffers.release_buffers,
                pb.BufferRefs(buffer_ids=[value.buffer.buffer_id for value in packet.outputs]))
            for value in packet.outputs:
                info = self.buffers.get_buffer_info(pb.BufferHandle(buffer_id=value.buffer.buffer_id))
                self.owned_inventory.append({"name": value.name, "payload_bytes": value.buffer.length_bytes,
                    "allocation_bytes": info.size_bytes, "kind": int(info.kind), "read_only": info.read_only})
        return packet

    def read(self, tensor):
        dtype = np.float32 if tensor.dtype == pb.DataType.DATA_TYPE_F32 else np.int32
        if not tensor.HasField("buffer"): return np.frombuffer(tensor.inline, dtype=dtype).reshape(tensor.shape).copy()
        value = np.empty(tensor.shape, dtype=dtype)
        self.buffers.copy_tensors(pb.CopyTensorsRequest(sources=[tensor], into=[pb.BorrowedBuffer(
            resource=pb.NativeResource(kind=pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST,
                handle=value.ctypes.data, size_bytes=value.nbytes), length_bytes=value.nbytes)]))
        return value

    def close(self): self.resources.close()


def pinned_audio(snapshot, manifest, *, download=False):
    pin = manifest["audio"]
    for name, asset in {**pin["assets"], **pin["weights"]}.items():
        path = snapshot / name
        if not path.exists() and download:
            snapshot.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".download")
            url = asset.get("url") or f"https://huggingface.co/{pin['repo_id']}/resolve/{pin['revision']}/{name}"
            try:
                with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as destination:
                    for block in iter(lambda: source.read(1 << 20), b""): destination.write(block)
                assert temporary.stat().st_size == asset["bytes"] and sha(temporary) == asset["sha256"], name
                temporary.replace(path)
            finally: temporary.unlink(missing_ok=True)
        assert path.stat().st_size == asset["bytes"] and sha(path) == asset["sha256"], name
    return {"repo_id": pin["repo_id"], "revision": pin["revision"],
        "assets": {name: asset["sha256"] for name, asset in {**pin["assets"], **pin["weights"]}.items()}}


def pinned_image(package, manifest):
    """Verify the upstream JPEG/ONNX/processor assets and converted package."""
    pin = manifest["image"]
    for asset in pin["assets"].values():
        path = ROOT / asset["local_path"]
        assert path.stat().st_size == asset["bytes"] and sha(path) == asset["sha256"], str(path)
    source = ROOT / "examples/tiny_receipt_vqa/benchmarks/work/source"
    upstream_path = source / "inference.py"
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location("pinned_receipt_inference", upstream_path)
        upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    finally: sys.path.pop(0)
    upstream_tokenizer = upstream.BPE1536Tokenizer(json.loads((source / "vocab.json").read_text()))
    package_manifest = json.loads((package / "package_manifest.json").read_text())
    assets = package_manifest["graphs"]
    for role in ("encoder", "decoder"):
        for kind in ("graph", "weights"):
            asset = assets[role][kind]; path = package / asset["path"]
            assert path.stat().st_size == asset["bytes"] and sha(path) == asset["sha256"], str(path)
    provenance = {"repo_id": pin["repo_id"], "revision": pin["revision"],
        "source_assets": {name: asset["sha256"] for name, asset in pin["assets"].items()},
        "package_manifest_sha256": sha(package / "package_manifest.json"),
        "package_assets": {role: {kind: assets[role][kind] for kind in ("graph", "weights")}
            for role in ("encoder", "decoder")}}
    return source, upstream, upstream_tokenizer, package_manifest, provenance


def image_workflow(args, manifest, receipt):
    from examples.tiny_receipt_vqa.tools import verify_proto_backends as vqa
    pin = manifest["image"]
    package = args.vqa_package
    source, upstream, upstream_tokenizer, package_manifest, provenance = pinned_image(package, manifest)
    candidate_tokenizer = vqa.tokenizer(package)
    assets = package_manifest["graphs"]
    graphs = {role: json.loads((package / assets[role]["graph"]["path"]).read_text())
        for role in ("encoder", "decoder")}
    graphs["encoder"]["execution"] = stateless_execution(graphs["encoder"]["inputs"], graphs["encoder"]["outputs"])
    mapping = assets["decoder"]["inputs"]
    invariants = [name for name in mapping if name.startswith("cross_") or name in ("family_ids", "memory_padding_mask")]
    feedback = {mapping[f"past_{kind}_{index}"]: assets["decoder"]["outputs"][f"present_{kind}_{index}"]
        for index in range(4) for kind in ("k", "v")}
    feedback[mapping["past_padding_mask"]] = assets["decoder"]["outputs"]["present_padding_mask"]
    graphs["decoder"]["execution"] = tensor_feedback_execution(graphs["decoder"]["inputs"],
        graphs["decoder"]["outputs"], feedback=feedback, position_input=mapping["position_ids"],
        invariant_inputs=[mapping[name] for name in invariants])
    entry = {"pin": {"repo_id": pin["repo_id"], "revision": pin["revision"]},
        "source_assets": {name: asset["sha256"] for name, asset in pin["assets"].items()},
        "package_manifest_sha256": sha(package / "package_manifest.json"),
        "package_assets": provenance["package_assets"],
        "processor": "Pillow grayscale L, bilinear 672x320; (pixel/255-.5)/.5; pinned byte-fallback BPE1536",
        "generation_policy": "caller greedy explicit KV via decoder Execute; EOS=2; BOS=1; first-index ties; maximum 191 new IDs",
        "cases": []}
    receipt["image"] = entry
    for filename, question in (("receipt_en.jpg", "What is the store name?"), ("receipt_ko.jpg", "가게 이름은 무엇인가요?")):
        image_path = source / "examples" / filename
        prepared = time.perf_counter_ns()
        image = prepare_receipt_image(image_path)
        ids = candidate_tokenizer.encode(vqa.clean(question), add_eos=True, max_len=192)
        prepare_ns = time.perf_counter_ns() - prepared
        np.testing.assert_array_equal(image, upstream.preprocess_image(image_path))
        np.testing.assert_array_equal(ids, upstream_tokenizer.encode(upstream.clean_text(question), 192).reshape(-1))
        reference = vqa.OrtRunner(source, "fp32")
        try:
            expected_encoder = reference.run("encoder", vqa.encoder_feed(image, ids))
            feed = vqa.decoder_feed(expected_encoder, 0)
            expected_ids, expected_logits = [], []
            for position in range(191):
                output = reference.run("decoder", feed)
                expected_logits.append(output["logits"])
                token = int(output["logits"].argmax(-1)[0, 0]); expected_ids.append(token)
                vqa.advance(feed, output, token, position)
                if token == 2: break
        finally: reference.close()
        assert expected_ids[-1] == 2, filename
        case = {"jpeg": filename, "jpeg_sha256": sha(image_path), "question": question, "question_ids": ids,
            "processor_exact": True, "pixels_sha256": array_hash(image), "caller_prepare_ns": prepare_ns,
            "expected_ids": expected_ids, "expected_text": candidate_tokenizer.decode([1, *expected_ids]), "profiles": []}
        entry["cases"].append(case)
        for profile in args.profiles:
            with tempfile.TemporaryDirectory(prefix="volvox-h-image-", dir=args.scratch) as directory:
                session = Session(profile, directory, library_dir=args.library_dir, backend=args.backend)
                try:
                    packages = {}
                    for role in ("encoder", "decoder"):
                        graph_path = Path(directory) / f"{role}.graph.json"
                        graph_path.write_text(json.dumps(graphs[role], allow_nan=False))
                        packages[role] = session.load(role, graph_path, package / assets[role]["weights"]["path"])
                    ec, dc = session.context("encoder"), session.context("decoder")
                    encoder_inputs = {assets["encoder"]["inputs"][name]: value
                        for name, value in vqa.encoder_feed(image, ids).items()}
                    packet = session.execute(ec, encoder_inputs, encoder=True, owned=True)
                    by_name = {value.name: value for value in packet.outputs}
                    encoded = {name: session.read(by_name[tensor]) for name, tensor in assets["encoder"]["outputs"].items()}
                    encoder_hashes = {name: array_hash(value) for name, value in encoded.items()}
                    encoder_checks = {}
                    for name, actual in encoded.items():
                        if actual.dtype.kind == "f":
                            encoder_checks[name] = compare(actual, expected_encoder[name], atol=1e-4, rtol=1e-4)
                        else: np.testing.assert_array_equal(actual, expected_encoder[name])
                    # Media arrays and accepted invariant decoder inputs are reusable
                    # independently of the encoder's OWNED output snapshots.
                    untouched_pixels = image.copy(); image[:] = 0
                    feed = vqa.decoder_feed(encoded, 1)
                    observed, checks, retained = [], [], []
                    for index, logits in enumerate(expected_logits):
                        reuse = [mapping[name] for name in invariants] if index else []
                        inputs = {mapping[name]: value for name, value in feed.items() if not index or name not in invariants}
                        packet = session.execute(dc, inputs, entrypoint="transition" if index else "initialize", reuse=reuse)
                        retained.append(packet.outputs[0])
                        outputs = {value.name: session.read(value) for value in packet.outputs}
                        actual = {name: outputs[tensor] for name, tensor in assets["decoder"]["outputs"].items()}
                        checks.append(compare(actual["logits"], logits, atol=1e-4, rtol=1e-4))
                        token = int(actual["logits"].argmax(-1)[0, 0]); observed.append(token)
                        vqa.advance(feed, actual, token, index)
                        if not index:
                            for name in invariants: feed[name][:] = 0
                    assert observed == expected_ids
                    for name, tensor in assets["encoder"]["outputs"].items():
                        actual = session.read(by_name[tensor])
                        assert array_hash(actual) == encoder_hashes[name], name
                        if actual.dtype.kind == "f": compare(actual, expected_encoder[name], atol=1e-4, rtol=1e-4)
                        else: np.testing.assert_array_equal(actual, expected_encoder[name])
                    assert session.encoder_calls == 1
                    snapshots = [value.inline for value in retained]
                    result = {"profile": profile, "platform": session.platform, "packages": packages, "encoder": encoder_checks, "logits": checks,
                        "generated_ids": observed, "eos": observed[-1] == 2, "encoder_execute_calls": 1,
                        "conditioning_snapshot_and_reuse": True, "routes": session.routes}
                    result["owned_outputs"] = session.owned_inventory
                    image[:] = untouched_pixels
                finally: session.close()
                assert snapshots == [value.inline for value in retained]
                result["inline_survives_all_owner_retirement"] = True
                case["profiles"].append(result)
                print(f"VQA {filename} {profile}: encoder/logits/{len(observed)} IDs PASS", flush=True)


def codebook_workflow(args, receipt):
    """Qualify a fixed categorical-ID contract, not a pretrained media model."""
    from tools.exporter.hf_graph import GraphBuilder
    b = GraphBuilder(8, ["logits", "generated"])
    b.add("transitions", np.array([[0, 0, 3], [4, 0, 0], [0, 5, 0]], np.float32))
    b.tensor("logits", "Embedding", {"input": "token_ids", "weight": "transitions"}, [1, "S", 3])
    b.tensor("generated", "ArgMax", {"input": "logits"}, [1, "S"], {"axis": -1, "keepdims": False}, "int32")
    generator, gw = b.finish(full_prefix_decoder_execution(b.graph["inputs"], b.graph["outputs"],
        token_input="token_ids", token_output="generated", keep_input="attention_mask", position_input="position_ids",
        logits_output="logits", id_domain_size=3))
    codebook = np.array([[.25, -.5, 1, 0], [-1, .125, 0, .75], [.5, 1, -.25, .375]], np.float32)
    b = GraphBuilder(1, ["decoded_features"], {"codes": {"shape": [1, 3], "dtype": "int32"}})
    b.graph["dimensions"] = {}
    b.add("codebook", codebook)
    b.tensor("decoded_features", "Embedding", {"input": "codes", "weight": "codebook"}, [1, 3, 4])
    decoder, dw = b.finish()
    expected_ids = [0, 2, 1]
    expected_features = np.stack((codebook[0], codebook[2], codebook[1]))[None]
    entry = {"qualification": "synthetic fixed single-codebook categorical fixture; no tokenizer",
        "expected_ids": expected_ids, "expected_features": expected_features.tolist(), "profiles": [],
        "excluded": ["pretrained nontext checkpoint", "multiple codebooks", "audio/video rendering quality"]}
    receipt["single_codebook"] = entry
    for profile in args.profiles:
        with tempfile.TemporaryDirectory(prefix="volvox-h-codes-", dir=args.scratch) as directory:
            session = Session(profile, directory, library_dir=args.library_dir, backend=args.backend)
            try:
                packages = {"generator": session.model("generator", generator, gw), "tensor_decoder": session.model("tensor_decoder", decoder, dw)}
                gc = session.context("generator", generate=True)
                packet = session.client.generate(pb.GenerateRequest(context_id=gc,
                    initial_inputs=pb.GenerationInputs(values=tensors({"token_ids": np.array([[1]], np.int32),
                        "attention_mask": np.ones((1, 1), np.int32), "position_ids": np.zeros(1, np.int32)})),
                    budget=pb.GenerationBudget(max_steps=3),
                    outputs=pb.OutputOptions(storage=pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER)))
                session.routes.append(attest(packet.report, args.backend))
                output = packet.outputs[0]
                session.resources.callback(session.buffers.release_buffers, pb.BufferRefs(buffer_ids=[output.buffer.buffer_id]))
                codes = session.read(output)
                assert codes.reshape(-1).tolist() == expected_ids
                packet = session.execute(session.context("tensor_decoder"), {"codes": codes})
                decoded = session.read(packet.outputs[0])
                np.testing.assert_array_equal(decoded, expected_features)
                retained = packet.outputs[0]; snapshot = retained.inline
                result = {"profile": profile, "platform": session.platform, "packages": packages, "generated_ids": expected_ids,
                    "decoded_features_sha256": array_hash(decoded), "exact_numpy_decoder_match": True,
                    "routes": session.routes}
            finally: session.close()
            assert retained.inline == snapshot
            result["inline_survives_all_owner_retirement"] = True
            entry["profiles"].append(result)
            print(f"single-codebook {profile}: Generate/ID-to-tensor PASS", flush=True)


def audio_workflow(args, manifest, receipt):
    from tools.exporter.hf_decoder import read_hf_checkpoint
    from tools.exporter.hf_whisper import WhisperConfig, build_whisper_encoder_graph, build_whisper_decoder_graph
    import torch
    import transformers
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer
    from transformers.generation.logits_process import SuppressTokensLogitsProcessor
    torch.set_num_threads(1)
    pin = pinned_audio(args.whisper_snapshot, manifest, download=args.download_whisper)
    processor = manifest["audio"]["processor_config"]
    hf_processor = WhisperFeatureExtractor.from_pretrained(args.whisper_snapshot, local_files_only=True)
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.whisper_snapshot, local_files_only=True,
        attn_implementation="eager", dtype=torch.float32).eval()
    tokenizer = WhisperTokenizer.from_pretrained(args.whisper_snapshot, local_files_only=True,
        language="english", task="transcribe", predict_timestamps=False)
    prompt = tokenizer.prefix_tokens
    source, weights, _ = read_hf_checkpoint(args.whisper_snapshot)
    config = WhisperConfig.from_hf(source, max_context=16)
    suppressed = sorted(set(source["suppress_tokens"]) | set(range(50364, config.vocabulary)))
    encoder, ew = build_whisper_encoder_graph(config, weights)
    decoder, dw = build_whisper_decoder_graph(config, weights, generate=True, suppressed_ids=suppressed)
    del weights
    clock = np.arange(24000, dtype=np.float32) / np.float32(16000)
    pcm = (.2 * np.sin(2 * np.pi * 220 * clock) + .05 * np.sin(2 * np.pi * 730 * clock)).astype(np.float32)
    raw_hash = array_hash(pcm)
    preparation = time.perf_counter_ns()
    features = prepare_whisper_features(pcm, processor)
    prepare_ns = time.perf_counter_ns() - preparation
    padded = np.zeros((1, 480000), np.float32); padded[0, :len(pcm)] = pcm
    expected_features = hf_processor._np_extract_fbank_features(padded, "cpu")
    torch_features = hf_processor(pcm, sampling_rate=16000, return_tensors="np").input_features
    feature_check = compare(features, expected_features, atol=1e-5, rtol=1e-5)
    with torch.inference_mode():
        reference_memory = hf_model.model.encoder(torch.from_numpy(expected_features)).last_hidden_state
        prefix = list(prompt); expected_ids, reference_logits = [], []
        policy = SuppressTokensLogitsProcessor(suppressed)
        for _ in range(args.steps):
            tokens = torch.tensor([prefix])
            logits = hf_model(encoder_outputs=(reference_memory,), decoder_input_ids=tokens, use_cache=False).logits
            reference_logits.append(logits.numpy())
            selected = int(policy(tokens, logits[:, -1].clone()).argmax(-1)[0])
            expected_ids.append(selected); prefix.append(selected)
            if selected == source["eos_token_id"]: break
    entry = {"pin": pin, "transformers_version": transformers.__version__, "reference_model_source_sha256":
        sha(sys.modules[hf_model.__class__.__module__].__file__), "pcm": {"kind": "deterministic two-tone mono F32",
            "samples": len(pcm), "sampling_rate": 16000, "sha256": raw_hash},
        "processor": {"branch": "official HF NumPy CPU", "comparison": feature_check,
            "torch_default_max_abs_difference": float(np.abs(features - torch_features).max())},
        "generation_policy": {"name": "caller fixed English/transcribe/no-timestamps greedy",
            "prompt_ids": prompt, "suppress_ids": suppressed, "begin_suppress_tokens": [],
            "eos_id": source["eos_token_id"],
            "reference": "HF forward plus official SuppressTokensLogitsProcessor",
            "excluded": ["default HF Whisper.generate", "VAD", "beam search", "automatic language detection",
                "temperature fallback", "timestamp generation", "transcription quality qualification"]},
        "expected_ids": expected_ids, "expected_text_raw": tokenizer.decode(expected_ids, skip_special_tokens=True),
        "caller_prepare_ns": prepare_ns, "profiles": []}
    receipt["audio"] = entry
    pcm[:] = 0
    assert array_hash(features) == feature_check["actual_sha256"]
    for profile in args.profiles:
        with tempfile.TemporaryDirectory(prefix="volvox-h-audio-", dir=args.scratch) as directory:
            session = Session(profile, directory, library_dir=args.library_dir, backend=args.backend)
            try:
                started = time.perf_counter_ns()
                packages = {"encoder": session.model("encoder", encoder, ew), "decoder": session.model("decoder", decoder, dw)}
                ec = session.context("encoder")
                packet = session.execute(ec, {"input_features": features}, encoder=True, owned=True)
                memory_tensor = packet.outputs[0]
                memory = session.read(memory_tensor)
                memory_snapshot_hash = array_hash(memory)
                memory_check = compare(memory, reference_memory.numpy(), atol=2e-4, rtol=2e-4, rms=2e-5)
                dc = session.context("decoder")
                checks = []
                prefix = list(prompt)
                for index, expected in enumerate(reference_logits):
                    inputs = {"token_ids": np.array([prefix], np.int32), "position_ids": np.arange(len(prefix), dtype=np.int32),
                        "attention_mask": np.ones((1, len(prefix)), np.int32), "encoder_memory": memory}
                    outputs = session.execute(dc, inputs).outputs
                    actual = next(session.read(value) for value in outputs if value.name == "logits")
                    checks.append(compare(actual, expected, atol=2e-3, rtol=1e-3, rms=5e-4))
                    prefix.append(expected_ids[index])
                gc = session.context("decoder", generate=True, steps=args.steps, eos_ids=[source["eos_token_id"]])
                initial = tensors({"token_ids": np.array([prompt], np.int32), "position_ids": np.arange(len(prompt), dtype=np.int32),
                    "attention_mask": np.ones((1, len(prompt)), np.int32)}, borrowed=False)
                initial.append(memory_tensor)
                observed, retained = [], []
                for index, budget in enumerate((1, args.steps - 1)):
                    if budget <= 0: continue
                    output = session.client.generate(pb.GenerateRequest(context_id=gc,
                        initial_inputs=pb.GenerationInputs(values=initial) if index == 0 else None,
                        budget=pb.GenerationBudget(max_steps=budget)))
                    session.routes.append(attest(output.report, args.backend))
                    value = output.outputs[0]; retained.append(value)
                    observed.extend(session.read(value).reshape(-1).tolist())
                    # Reusing caller media/features/memory cannot alter accepted
                    # conditioning or its independent encoder output snapshot.
                    memory[:] = 0
                    if not output.progress.can_continue: break
                assert observed == expected_ids, (profile, observed, expected_ids)
                owned_memory = session.read(memory_tensor)
                assert array_hash(owned_memory) == memory_snapshot_hash
                compare(owned_memory, reference_memory.numpy(), atol=2e-4, rtol=2e-4, rms=2e-5)
                # A second decoder consumes caller-owned conditioning. Mutating
                # its original host bytes between chunks cannot change the
                # accepted immutable context snapshot or rerun the encoder.
                bc = session.context("decoder", generate=True, steps=args.steps, eos_ids=[source["eos_token_id"]])
                borrowed_values = {"token_ids": np.array([prompt], np.int32), "position_ids": np.arange(len(prompt), dtype=np.int32),
                    "attention_mask": np.ones((1, len(prompt)), np.int32), "encoder_memory": owned_memory}
                borrowed_ids = []
                for index, budget in enumerate((1, args.steps - 1)):
                    packet = session.client.generate(pb.GenerateRequest(context_id=bc,
                        initial_inputs=pb.GenerationInputs(values=tensors(borrowed_values, borrowed=True)) if not index else None,
                        budget=pb.GenerationBudget(max_steps=budget)))
                    session.routes.append(attest(packet.report, args.backend))
                    retained.extend(packet.outputs)
                    borrowed_ids.extend(session.read(packet.outputs[0]).reshape(-1).tolist())
                    owned_memory[:] = 0
                    if not packet.progress.can_continue: break
                assert borrowed_ids == expected_ids
                assert array_hash(session.read(memory_tensor)) == memory_snapshot_hash
                assert session.encoder_calls == 1
                results = {"profile": profile, "platform": session.platform, "library_sha256": sha(args.library_dir /
                    ("libvolvoxai-lite.so" if profile == "inference" else "libvolvoxai.so")), "packages": packages,
                    "encoder": memory_check, "logits": checks, "generated_ids": observed,
                    "encoder_execute_calls": session.encoder_calls, "routes": session.routes,
                    "borrowed_conditioning_snapshot_and_reuse": True, "owned_outputs": session.owned_inventory,
                    "accuracy_workflow_ns": time.perf_counter_ns() - started,
                    "metrics_scope": "single accuracy workflow including loading, compile and output capture; not latency qualification"}
                snapshots = [value.inline for value in retained]
            finally: session.close()
            assert [value.inline for value in retained] == snapshots
            results["inline_survives_all_owner_retirement"] = True
            entry["profiles"].append(results)
            print(f"Whisper {profile}: encoder/logits/{len(observed)} IDs PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whisper-snapshot", type=Path, default=Path("/tmp/volvoxai-h-stage-whisper-169d4a"))
    parser.add_argument("--download-whisper", action="store_true", help="fetch missing pinned files into the explicit snapshot; verify every SHA256")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--library-dir", type=Path, default=ROOT / "native")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "opengl", "metal"), default="cpu")
    parser.add_argument("--vqa-package", type=Path, default=ROOT / "build/receipt-vqa-ptq-head-29ffb7e/fp32")
    parser.add_argument("--workflows", nargs="+", choices=("audio", "image", "codebook"), default=("audio", "image", "codebook"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("inference", "full"), default=("inference", "full"))
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.steps <= 8: parser.error("steps must be 2..8")
    if args.backend != "cpu" and args.profiles != ["full"]: parser.error("native GPU requires --profiles full")
    manifest = json.loads((ROOT / "tools/llm_baseline/multimodal_sensor.json").read_text())
    receipt = {"format": "volvox-multimodal-native/v1", "status": "RUNNING",
        "scope": "selected pinned Whisper/VQA caller workflows and synthetic single-codebook; native CPU",
        "harness_sha256": sha(__file__), "baseline_manifest_sha256": sha(ROOT / "tools/llm_baseline/multimodal_sensor.json"),
        "memory": {"namespace": "whole Python process incl HF oracle, NumPy, native owners; lifetime high-water RSS"}}
    try:
        if "audio" in args.workflows: audio_workflow(args, manifest, receipt)
        if "image" in args.workflows: image_workflow(args, manifest, receipt)
        if "codebook" in args.workflows: codebook_workflow(args, receipt)
        receipt["status"] = "PASS"
    except BaseException as error:
        receipt["status"] = "FAIL"; receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        receipt["memory"]["process_lifetime_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__": main()
