#!/usr/bin/env python3
"""Run the complete Kokoro model on VolvoxAI for a fresh phoneme request."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.kokoro.complete import main


if __name__ == "__main__":
    main()
