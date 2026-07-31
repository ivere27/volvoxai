#!/usr/bin/env python3
"""PyTorch greedy-decode oracle for the autoregressive decode-path parity check.

Reads the prompt written by decode_parity.mjs (out/prompt.json), runs the same greedy
algorithm (argmax of the last-position logits, append, repeat) on the source HF model
roneneldan/TinyStories-1M, and writes the generated token sequence to out/torch.json.
"""
import json
import hashlib
import os
import sys
import uuid
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(HERE, "..", "external"))
from sigutil import write_json_atomic  # noqa: E402

CASE_ID = "tinystories-decode"
SCHEMA_VERSION = 1
ARTIFACT_SCHEMA = "volvoxai.parity-artifact"
MANIFEST_SCHEMA = "volvoxai.parity-run-manifest"


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def remove_old_torch_outputs():
    for name in ["torch.json", "torch_run.json"]:
        try:
            os.unlink(os.path.join(OUT, name))
        except FileNotFoundError:
            pass


def current_prompt():
    manifest_path = os.path.join(OUT, "run.json")
    prompt_path = os.path.join(OUT, "prompt.json")
    manifest = json.load(open(manifest_path))
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != SCHEMA_VERSION or
            manifest.get("outcome") != "success" or not manifest.get("completedAt")):
        raise ValueError("decode producer manifest is missing, unfinished, or unsuccessful")
    fingerprint = manifest.get("fingerprint", {})
    digest = fingerprint.get("digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("decode producer manifest has no valid fingerprint")
    result = next((item for item in manifest.get("results", [])
                   if item.get("case") == CASE_ID and item.get("tier") == "prompt" and
                   item.get("status") == "success"), None)
    if result is None or result.get("artifact", {}).get("path") != "prompt.json":
        raise ValueError("decode producer manifest has no current prompt artifact")
    actual = sha256_file(prompt_path)
    if actual != {key: result["artifact"][key] for key in ["sha256", "size"]}:
        raise ValueError("prompt artifact bytes do not match the decode producer manifest")
    artifact = json.load(open(prompt_path))
    provenance = artifact.get("provenance", {})
    if (artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("version") != SCHEMA_VERSION or
            provenance.get("runId") != manifest.get("runId") or
            provenance.get("fingerprint") != digest or provenance.get("case") != CASE_ID or
            provenance.get("tier") != "prompt" or provenance.get("command") != manifest.get("command")):
        raise ValueError("prompt artifact does not belong to the current decode producer run")
    return manifest, artifact.get("payload", {})


def main():
    remove_old_torch_outputs()
    parent_manifest, spec = current_prompt()
    prompt, n_new = list(spec["prompt"]), int(spec["nNew"])
    if not prompt or n_new <= 0:
        raise ValueError("decode prompt and generation length must be non-empty")
    started_at = utc_now()
    policy = json.load(open(os.path.join(ROOT, "tests", "parity", "policy.json")))
    external = policy["models"]["tinystories_1m"]["external"]
    model = AutoModelForCausalLM.from_pretrained(
        external["repo"], revision=external["revision"]
    )
    model.eval()
    ids, gen = list(prompt), []
    with torch.no_grad():
        for _ in range(n_new):
            logits = model(input_ids=torch.tensor([ids], dtype=torch.long)).logits[0, -1, :]
            nxt = int(torch.argmax(logits))
            ids.append(nxt)
            gen.append(nxt)
    if len(gen) != n_new or any(not isinstance(token, int) or token < 0 for token in gen):
        raise ValueError("Torch oracle returned an invalid generated-token sequence")

    command = "decode torch oracle"
    run_id = str(uuid.uuid4())
    created_at = utc_now()
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "version": SCHEMA_VERSION,
        "provenance": {
            "runId": run_id,
            "command": command,
            "fingerprint": parent_manifest["fingerprint"]["digest"],
            "case": CASE_ID,
            "tier": "torch",
            "createdAt": created_at,
        },
        "payload": {"backend": "torch", "tokens": gen},
    }
    artifact_path = os.path.join(OUT, "torch.json")
    write_json_atomic(artifact_path, artifact, indent=2)
    artifact_digest = sha256_file(artifact_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "version": SCHEMA_VERSION,
        "runId": run_id,
        "command": command,
        "fingerprint": parent_manifest["fingerprint"],
        "selection": {
            "cases": [CASE_ID],
            "tiers": ["torch"],
            "jobs": [{"case": CASE_ID, "tier": "torch", "expectation": "required"}],
        },
        "producer": {
            "kind": "python",
            "backend": "torch",
            "parentRunId": parent_manifest["runId"],
        },
        "startedAt": started_at,
        "completedAt": utc_now(),
        "outcome": "success",
        "results": [{
            "case": CASE_ID,
            "tier": "torch",
            "status": "success",
            "artifact": {"path": "torch.json", **artifact_digest},
        }],
    }
    write_json_atomic(os.path.join(OUT, "torch_run.json"), manifest, indent=2)
    print(f"torch decode ({n_new} tokens): {gen}")


if __name__ == "__main__":
    main()
