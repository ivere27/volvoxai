#!/usr/bin/env python3
"""Measure only VQA, using the shared receipt measurement driver."""
from run_receipt_proto_benchmarks import main

if __name__ == "__main__":
    raise SystemExit(main(model="vqa"))
