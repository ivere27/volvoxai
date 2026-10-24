#!/usr/bin/env python3
"""Prepare pinned, fixed WikiText-2 and ARC-Easy test subsets before evaluation.

Requires transformers and pyarrow only for preparation. The runtime evaluator
consumes the resulting JSON token IDs without a tokenizer or dataset library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(repo, revision, filename, destination):
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    return {"url": url, "path": str(destination), "sha256": digest(destination),
            "bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wiki-tokens", type=int, default=16384)
    parser.add_argument("--block-tokens", type=int, default=1024)
    parser.add_argument("--arc-questions", type=int, default=100)
    args = parser.parse_args()
    if args.wiki_tokens <= 0 or not 0 < args.block_tokens <= 2048 or args.arc_questions <= 0:
        parser.error("Positive token/question counts and block size <= 2048 are required")
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    wiki_file = out / "wikitext2-test.parquet"
    arc_file = out / "arc-easy-test.parquet"
    sources = {
        "wikitext": download("Salesforce/wikitext", WIKITEXT_REVISION,
                             "wikitext-2-raw-v1/test-00000-of-00001.parquet", wiki_file),
        "arc": download("allenai/ai2_arc", ARC_REVISION,
                        "ARC-Easy/test-00000-of-00001.parquet", arc_file),
    }
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if tokenizer.bos_token_id is None:
        raise ValueError("The pinned TinyLlama tokenizer must declare BOS")
    tokenizer_files = {p.name: digest(p) for p in sorted(args.tokenizer.glob("*"))
                       if p.name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")}
    wiki_rows = parquet.read_table(wiki_file).to_pylist()
    # Preserve source row bytes and explicit separator semantics from the earlier
    # runtime benchmark. Fixed first-N selection has no dependency on model scores.
    wiki_text = "\n\n".join(row["text"] for row in wiki_rows)
    wiki_ids = tokenizer.encode(wiki_text, add_special_tokens=False)
    if len(wiki_ids) < args.wiki_tokens:
        raise ValueError("WikiText test corpus is shorter than the requested token budget")
    cases = []
    for offset in range(0, args.wiki_tokens, args.block_tokens):
        targets = wiki_ids[offset:min(offset + args.block_tokens, args.wiki_tokens)]
        cases.append({"id": f"wikitext2-test-{offset:06d}", "initial_ids": [tokenizer.bos_token_id],
                      "teacher_ids": targets, "capacity": len(targets), "execution": "prefill",
                      "dataset": "wikitext2-test", "source_token_start": offset,
                      "source_token_stop": offset + len(targets)})
    arc_rows = parquet.read_table(arc_file).to_pylist()
    if len(arc_rows) < args.arc_questions:
        raise ValueError("ARC test corpus is shorter than the requested question budget")
    selected_questions = []
    for index, row in enumerate(arc_rows[:args.arc_questions]):
        prompt = f"Question: {row['question']}\nAnswer:"
        initial = tokenizer.encode(prompt, add_special_tokens=True)
        texts, labels = row["choices"]["text"], row["choices"]["label"]
        answer = labels.index(row["answerKey"])
        selected_questions.append({"row_index": index, "id": row["id"], "question": row["question"],
                                   "choices": row["choices"], "answer_index": answer})
        for choice, text in enumerate(texts):
            joined = tokenizer.encode(prompt + " " + text, add_special_tokens=True)
            if joined[:len(initial)] != initial:
                raise ValueError(f"Unstable context/continuation token boundary for ARC {row['id']}")
            targets = joined[len(initial):]
            length = len(initial) + len(targets) - 1
            if not targets or length > 2048 or not text:
                raise ValueError(f"Unsupported ARC continuation length for {row['id']}")
            cases.append({"id": f"arc-easy-test-{index:04d}-choice-{choice}",
                          "initial_ids": initial, "teacher_ids": targets,
                          "capacity": length, "execution": "prefill", "dataset": "arc-easy-test",
                          "task": {"id": row["id"], "choice_index": choice, "choice_count": len(texts),
                                   "answer_index": answer, "normalization_chars": len(text)}})
    policy = {
        "schema": "volvoxai-quantization-quality-policy/v1",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model_revision": MODEL_REVISION,
        "tokenizer_files_sha256": tokenizer_files, "vocab_size": len(tokenizer),
        "execution": "prefill", "context_capacity": max(case["capacity"] for case in cases),
        "sources": sources,
        "language_model": {"dataset": "Salesforce/wikitext", "revision": WIKITEXT_REVISION,
                           "subset": "wikitext-2-raw-v1", "split": "test",
                           "selection": "First N tokenizer tokens from all test rows joined by two newlines; preserve each original row",
                           "scored_tokens": args.wiki_tokens, "block_tokens": args.block_tokens,
                           "context_policy": "Disjoint target blocks; reset each block with BOS; predict all target tokens",
                           "scope": "Fixed blocked subset, not full-corpus WikiText perplexity"},
        "multiple_choice": {"dataset": "allenai/ai2_arc", "revision": ARC_REVISION,
                            "subset": "ARC-Easy", "split": "test", "questions": args.arc_questions,
                            "selection": "First N questions in pinned parquet row order; no filtering",
                            "few_shot_examples": 0, "prompt_template": "Question: {question}\nAnswer:",
                            "continuation_template": " {choice_text}",
                            "normalization": "Unicode character count of choice_text; excludes inserted leading space",
                            "tie_break": "first choice in dataset order", "chat_template": False},
        "calibration": {"used": False, "reason": "RTN W4/W8 artifacts require no calibration or evaluation-derived fitting"},
        "comparison": {"reference": "dense F32", "weights": ["W4 RTN group128", "W8 RTN group128"],
                       "packed_arithmetic": ["dynamic-i8", "f16", "f32", "f32-fast"],
                       "scope": "Exploratory paired comparison; no equivalence margin or adoption threshold"},
        "cases": cases,
    }
    policy_path = out / "quality-policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n")
    (out / "arc-selected-questions.json").write_text(json.dumps(selected_questions, indent=2, ensure_ascii=False) + "\n")
    receipt = {"policy_sha256": digest(policy_path), "case_count": len(cases),
               "wiki_scored_tokens": args.wiki_tokens, "arc_questions": args.arc_questions,
               "arc_choice_cases": sum(bool(case.get("task")) for case in cases),
               "arc_scored_tokens": sum(len(case["teacher_ids"]) for case in cases if case.get("task")),
               "max_capacity": policy["context_capacity"], "total_test_rows": {"wiki": len(wiki_rows), "arc": len(arc_rows)}}
    (out / "preparation-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
