#!/usr/bin/env python3
"""Compare NumPy and native tensor outputs for the receipt digit reader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tiny_receipt_vqa" / "benchmarks"))
from native_tensor_handoff import main

if __name__ == "__main__":
    sys.argv.extend(["--model", "digit"])
    main()
