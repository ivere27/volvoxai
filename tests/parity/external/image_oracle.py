#!/usr/bin/env python3
"""Real-image EfficientDet oracle: decode dog.jpg/cat.jpg once, write the exact
input fixture the VolvoxAI tiers will read, run the source ONNX in ONNX Runtime,
and dump full scores/boxes + the top detections (with labels). Feeding *identical*
decoded bytes to every runtime makes the comparison apples-to-apples.

Usage:  python3 tests/parity/external/image_oracle.py <image.jpg> <int8|fp32>
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_OUT = os.path.join(ROOT, "tests", "parity", "out", "imgfix")
LABELS = open(os.path.join(ROOT, "models/efficientdet_lite0_int8/labels.txt")).read().splitlines()
ONNX = {
    "int8": "models/efficientdet_lite0_int8/efficientdet_lite0.onnx",
    "fp32": "models/efficientdet_lite0_fp32/efficientdet_lite0_float32.onnx",
}


def top_detections(scores, boxes, k=10, tag=""):
    sc = scores.reshape(-1, 90)
    bx = boxes.reshape(-1, 4)
    best_cls = sc.argmax(1)
    best = sc.max(1)
    order = np.argsort(-best)[:k]
    print(f"{tag} top-{k} detections (anchor: label score box):")
    for r, a in enumerate(order):
        c = int(best_cls[a])
        name = LABELS[c] if c < len(LABELS) else str(c)
        print(f"  {r+1:2d}. #{a:<6d} {name:<14s} {best[a]:.4f}  {np.round(bx[a], 3).tolist()}")


def main():
    img, model = sys.argv[1], sys.argv[2]
    out = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_OUT
    os.makedirs(out, exist_ok=True)

    def publish(array, name):
        final = os.path.join(out, name)
        temporary = f"{final}.{os.getpid()}.tmp"
        try:
            array.tofile(temporary)
            os.replace(temporary, final)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    im = Image.open(os.path.join(ROOT, img)).convert("RGB").resize((320, 320), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8).reshape(1, 320, 320, 3)

    if model == "int8":
        publish(arr, "input0.u8")                           # fixture VolvoxAI reads
        feed = {"serving_default_images:0": arr}
    else:
        f = (arr.astype(np.float32) / 255.0)                # zero-one normalization
        publish(f, "input0.f32")
        feed = {"serving_default_images:0": f}

    sess = ort.InferenceSession(os.path.join(ROOT, ONNX[model]), providers=["CPUExecutionProvider"])
    scores, boxes = sess.run(["StatefulPartitionedCall:1", "StatefulPartitionedCall:0"], feed)
    publish(scores.astype(np.float32), "onnx.scores.f32")
    publish(boxes.astype(np.float32), "onnx.boxes.f32")
    print(f"onnx {model} on {os.path.basename(img)}: scores{list(scores.shape)} boxes{list(boxes.shape)}")
    top_detections(scores, boxes, tag="ONNX")


if __name__ == "__main__":
    main()
