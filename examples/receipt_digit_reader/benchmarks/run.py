#!/usr/bin/env python3
"""Measure only the digit reader, using the shared receipt measurement driver."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.tiny_receipt_vqa.benchmarks.run_receipt_proto_benchmarks import main

if __name__ == "__main__":
    raise SystemExit(main(model="digit"))
