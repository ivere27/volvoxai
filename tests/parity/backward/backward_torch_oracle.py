#!/usr/bin/env python3
"""PyTorch backward oracle for the cross-tier gradient parity harness.

For each case freshly dumped by run_backward.mjs it rebuilds the exact forward,
runs cross-entropy + backward, and atomically publishes a versioned artifact plus
a manifest linked to the current CPU/WASM campaign. This is the genuine "PyTorch
autograd vs VolvoxAI autograd" check.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CASE_IDS = ("layernorm", "linear", "mlp")
MANIFEST_DIR = os.path.join(OUT, "manifests")
TORCH_MANIFEST = os.path.join(MANIFEST_DIR, "torch.json")
ARTIFACT_SCHEMA = "volvoxai.parity-artifact"
MANIFEST_SCHEMA = "volvoxai.parity-run-manifest"
SCHEMA_VERSION = 1


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def cleanup_outputs():
    for case_id in CASE_IDS:
        try:
            os.unlink(os.path.join(OUT, f"{case_id}.torch.json"))
        except FileNotFoundError:
            pass
    try:
        os.unlink(TORCH_MANIFEST)
    except FileNotFoundError:
        pass
    try:
        os.unlink(os.path.join(OUT, "backward_matrix.md"))
    except FileNotFoundError:
        pass


def read_campaign():
    manifests = []
    for tier in ("cpu", "wasm"):
        path = os.path.join(MANIFEST_DIR, f"{tier}.json")
        with open(path, encoding="utf-8") as source:
            manifest = json.load(source)
        if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != SCHEMA_VERSION):
            raise RuntimeError(f"invalid {tier} backward manifest schema")
        if manifest.get("outcome") != "success":
            raise RuntimeError(f"{tier} backward producer did not succeed")
        manifests.append(manifest)
    cpu, wasm = manifests
    if cpu.get("runId") != wasm.get("runId"):
        raise RuntimeError("cpu/wasm backward manifests belong to different runs")
    if cpu.get("fingerprint", {}).get("digest") != wasm.get("fingerprint", {}).get("digest"):
        raise RuntimeError("cpu/wasm backward manifests have different fingerprints")
    return cpu


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def load(d, name, shape, grad):
    a = np.fromfile(os.path.join(d, f"{name}.f32"), dtype=np.float32).reshape(shape)
    return torch.tensor(a, requires_grad=grad)


def forward(arch, x, w, dims):
    if arch == "linear":
        return x @ w["W"] + w["b"]
    if arch == "mlp":
        h = F.gelu(x @ w["W1"] + w["b1"])          # default approximate='none' (erf) == VolvoxAI CPU GELU
        return h @ w["W2"] + w["b2"]
    if arch == "layernorm":
        ln = F.layer_norm(x, [dims["D"]], w["g"], w["be"], eps=1e-5)
        return ln @ w["W"] + w["b"]
    raise ValueError(arch)


def main():
    cleanup_outputs()
    try:
        global np, torch, F, signature, write_json_atomic
        import numpy as np
        import torch
        import torch.nn.functional as F

        sys.path.insert(0, os.path.join(HERE, "..", "external"))
        from sigutil import signature, write_json_atomic

        campaign = read_campaign()
        command = "backward torch-oracle"
        started_at = now_iso()
        results = []
        for case_id in CASE_IDS:
            directory = os.path.join(OUT, case_id)
            meta_path = os.path.join(directory, "meta.json")
            with open(meta_path, encoding="utf-8") as source:
                meta = json.load(source)
            if meta.get("id") != case_id:
                raise RuntimeError(f"backward fixture id mismatch for {case_id}")
            shapes, input_name = meta["shapes"], meta["input"]
            x = load(directory, input_name, shapes[input_name], grad=False)
            weights = {
                name: load(directory, name, shapes[name], grad=True)
                for name in meta["trainable"]
            }
            logits = forward(meta["arch"], x, weights, meta["dims"])
            loss = F.cross_entropy(logits, torch.tensor(meta["targets"], dtype=torch.long))
            loss.backward()
            learning_rate = 0.1  # must match run_backward.mjs
            sigs = {
                name: signature(weights[name].grad.detach().numpy(), topk=0, shape=shapes[name])
                for name in meta["trainable"]
            }
            step = {
                name: signature(
                    (weights[name].detach() - learning_rate * weights[name].grad.detach()).numpy(),
                    topk=0,
                    shape=shapes[name],
                )
                for name in meta["trainable"]
            }
            artifact_path = os.path.join(OUT, f"{case_id}.torch.json")
            created_at = now_iso()
            loss_value = float(loss.detach())
            write_json_atomic(artifact_path, {
                "schema": ARTIFACT_SCHEMA,
                "version": SCHEMA_VERSION,
                "provenance": {
                    "runId": campaign["runId"],
                    "command": command,
                    "fingerprint": campaign["fingerprint"]["digest"],
                    "case": case_id,
                    "tier": "torch",
                    "createdAt": created_at,
                },
                "payload": {
                    "id": case_id,
                    "tier": "torch",
                    "loss": loss_value,
                    "sigs": sigs,
                    "stepSigs": step,
                },
            })
            results.append({
                "case": case_id,
                "tier": "torch",
                "status": "success",
                "artifact": {
                    "path": f"{case_id}.torch.json",
                    **sha256_file(artifact_path),
                },
            })
            print(f"torch-oracle {case_id}: loss={loss_value:.6f} grads + sgd-step")

        jobs = [
            {"case": case_id, "tier": "torch", "expectation": "required"}
            for case_id in CASE_IDS
        ]
        write_json_atomic(TORCH_MANIFEST, {
            "schema": MANIFEST_SCHEMA,
            "version": SCHEMA_VERSION,
            "runId": campaign["runId"],
            "command": command,
            "fingerprint": campaign["fingerprint"],
            "selection": {
                "cases": list(CASE_IDS),
                "tiers": ["torch"],
                "jobs": jobs,
            },
            "producer": {"kind": "python", "backend": "torch"},
            "startedAt": started_at,
            "completedAt": now_iso(),
            "outcome": "success",
            "results": results,
        })
    except Exception:
        cleanup_outputs()
        raise


if __name__ == "__main__":
    main()
