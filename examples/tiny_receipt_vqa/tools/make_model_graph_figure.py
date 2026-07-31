#!/usr/bin/env python3
"""Generate a publication figure of the TinyReceiptVQA W8A8 model graph (SVG)."""
import html
from pathlib import Path

W, H = 1360, 838
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "model_graph.svg"
S = []
def esc(t): return html.escape(str(t))

def box(x, y, w, h, lines, fill, stroke, rx=10, tfs=14, fs=12.5, sw=2):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    cx = x + w/2
    n = len(lines)
    total = tfs + (n-1)*(fs+3)
    ty = y + h/2 - total/2 + tfs*0.8
    for i, ln in enumerate(lines):
        if i == 0:
            S.append(f'<text x="{cx}" y="{ty:.1f}" font-size="{tfs}" font-weight="700" '
                     f'text-anchor="middle" fill="#0f172a">{esc(ln)}</text>')
            ty += fs + 5
        else:
            S.append(f'<text x="{cx}" y="{ty:.1f}" font-size="{fs}" text-anchor="middle" '
                     f'fill="#334155">{esc(ln)}</text>')
            ty += fs + 3

def arrow(x1, y1, x2, y2, color="#475569", dash=False, sw=2.2):
    d = ' stroke-dasharray="6 5"' if dash else ''
    S.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
             f'stroke-width="{sw}"{d} marker-end="url(#ah)"/>')

def label(x, y, t, fs=11.5, color="#64748b", anchor="middle", italic=False, bold=False):
    st = ' font-style="italic"' if italic else ''
    fw = ' font-weight="700"' if bold else ''
    S.append(f'<text x="{x}" y="{y}" font-size="{fs}" text-anchor="{anchor}" '
             f'fill="{color}"{st}{fw}>{esc(t)}</text>')

# palette
C_IN   = ("#f1f5f9", "#64748b")
C_STEM = ("#dbeafe", "#2563eb")
C_MEM  = ("#e0f2fe", "#0284c7")
C_ENC  = ("#dcfce7", "#16a34a")
C_DEC  = ("#fef3c7", "#d97706")
C_HEAD = ("#ede9fe", "#7c3aed")
C_OUT  = ("#fce7f3", "#db2777")
C_DET  = ("#f8fafc", "#94a3b8")

S.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Arial, sans-serif">')
S.append('<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="3" '
         'orient="auto" markerUnits="strokeWidth">'
         '<path d="M0,0 L8,3 L0,6 z" fill="#475569"/></marker></defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# title
label(W/2, 30, "TinyReceiptVQA — W8A8 Encoder–Decoder Graph (272 nodes)", fs=19, color="#0f172a", bold=True)
label(W/2, 50, "d_model=320  ·  heads=8  ·  head_dim=40  ·  encoder×6  ·  decoder×4  ·  img_tokens=210  ·  vocab=760  ·  all ops int8 (W8A8)",
      fs=12, color="#64748b")

# ---- main pipeline ----
# row A (encoder path)
box(30,  78, 170, 74, ["Receipt image", "[1, 320, 672, 1]", "grayscale"], *C_IN)
box(238, 70, 214, 90, ["① CNN Stem", "44 nodes", "13×(Conv+GroupNorm+SiLU)", "+4 ResBlocks, ÷32 → 210 tokens"], *C_STEM)
box(492,150,150, 84, ["Concat → memory", "[402 × 320]", "210 img + 192 q"], *C_MEM)
box(690, 74, 190, 96, ["② Encoder × 6", "112 nodes", "Self-Attention (8 heads)", "+ FFN (×4)"], *C_ENC)

# question feeding concat
box(238,196,214, 50, ["Question  q_ids [1,192]", "→ token + pos embed"], *C_IN, tfs=12.5, fs=11)

arrow(200, 115, 236, 115)                 # image -> stem
arrow(452, 128, 500, 168)                 # stem -> concat
arrow(360, 196, 470, 196)                 # question -> concat (up into)
arrow(345, 196, 545, 172)                 # question diagonal into concat bottom
arrow(642, 180, 700, 150)                 # concat -> encoder

# row C (decoder path)
box(492,270,150, 62, ["Answer so far", "y_ids [1,192]", "→ embed"], *C_IN, tfs=12.5, fs=11)
box(690,262,214,104, ["③ Decoder × 4", "108 nodes", "Masked Self-Attn + Cross-Attn", "+ FFN + phone Adapter"], *C_DEC)
box(944,278,150, 78, ["④ Head", "QArgMax  (1 node)"], *C_HEAD)
box(1134,286,150,62, ["token_ids", "next char"], *C_OUT, tfs=13, fs=11)

arrow(642, 300, 688, 308)                 # y_ids -> decoder
arrow(904, 314, 942, 314)                 # decoder -> head
arrow(1094, 316, 1132, 316)               # head -> token
# cross-attention encoder -> decoder
arrow(785, 170, 785, 262, color="#0284c7", dash=True)
label(795, 218, "cross-attention", fs=11, color="#0284c7", anchor="start", italic=True)
label(795, 232, "(memory K/V)", fs=11, color="#0284c7", anchor="start", italic=True)
# autoregressive loop
S.append('<path d="M1209,348 C1209,410 700,410 567,410 L567,334" fill="none" '
         'stroke="#db2777" stroke-width="2.2" stroke-dasharray="6 5" marker-end="url(#ah)"/>')
label(890, 405, "autoregressive  ×76 steps", fs=11.5, color="#db2777", italic=True)

# ================= DETAIL: encoder layer =================
def chain(x0, y, boxes, panel_title, panel_x, panel_w, ptitle_color):
    box(panel_x, y-26, panel_w, 118, [""], *C_DET, rx=12)
    label(panel_x+14, y-6, panel_title, fs=13.5, color=ptitle_color, anchor="start", bold=True)
    cx = x0
    centers = []
    for (lines, col) in boxes:
        w = col[2]
        b = (col[0], col[1])
        box(cx, y+14, w, 54, lines, b[0], b[1], rx=8, tfs=11.5, fs=10)
        centers.append((cx, cx+w))
        cx += w + 26
    for i in range(len(centers)-1):
        arrow(centers[i][1], y+41, centers[i+1][0], y+41, sw=2)
    return centers

# encoder layer row (full width)
enc_boxes = [
    (["input"], (C_IN[0],C_IN[1],72)),
    (["LayerNorm"], (C_ENC[0],C_ENC[1],96)),
    (["Self-Attn", "QSDPA · 8 heads"], (C_ENC[0],C_ENC[1],128)),
    (["⊕ residual"], ("#fff7ed","#ea580c",96)),
    (["LayerNorm"], (C_ENC[0],C_ENC[1],96)),
    (["FFN", "QLin 320→1280", "→ GELU → 1280→320"], (C_ENC[0],C_ENC[1],160)),
    (["⊕ residual"], ("#fff7ed","#ea580c",96)),
    (["output"], (C_IN[0],C_IN[1],78)),
]
chain(48, 468, enc_boxes, "Encoder layer  (×6):  self-attention block  +  feed-forward block", 30, 1300, C_ENC[1])

# decoder layer row (full width)
dec_boxes = [
    (["input"], (C_IN[0],C_IN[1],60)),
    (["LN"], (C_DEC[0],C_DEC[1],48)),
    (["Masked", "Self-Attn"], (C_DEC[0],C_DEC[1],96)),
    (["⊕"], ("#fff7ed","#ea580c",42)),
    (["LN"], (C_DEC[0],C_DEC[1],48)),
    (["Cross-Attn", "→ memory"], (C_MEM[0],C_MEM[1],104)),
    (["⊕"], ("#fff7ed","#ea580c",42)),
    (["LN"], (C_DEC[0],C_DEC[1],48)),
    (["FFN"], (C_DEC[0],C_DEC[1],72)),
    (["⊕"], ("#fff7ed","#ea580c",42)),
    (["Adapter", "320→64→320"], (C_HEAD[0],C_HEAD[1],108)),
    (["⊕"], ("#fff7ed","#ea580c",42)),
    (["output"], (C_IN[0],C_IN[1],66)),
]
chain(48, 590, dec_boxes, "Decoder layer  (×4):  masked self-attn  +  cross-attn (to memory)  +  FFN  +  family adapter", 30, 1300, C_DEC[1])

# ================= notes =================
ny = 700
box(30, ny, 1300, 96, [""], "#f8fafc", "#cbd5e1", rx=12)
label(48, ny+26, "Key structural features", fs=13.5, color="#0f172a", anchor="start", bold=True)
notes = [
    "•  Every Linear is LoRA-split in the graph:  y = base(x) + B(A(x))  (rank 8)  —  base + lora_a + lora_b + Add.  This is why there are 121 QLinear / 55 QAdd nodes.",
    "•  A hard router picks one of 8 family graphs (phone / address / store / item_row / item_math / item_lookup / math / other); each bakes in its own task adapter.",
    "•  All compute is W8A8: int8 weights (symmetric per-output-channel) + int8 activations; central refs bind every edge to safetensors scale/zero-point tensors.",
    "•  Stages ①② (image+question → memory, 156 nodes) depend only on inputs; the autoregressive loop currently re-runs the whole graph per token → caching target.",
]
yy = ny+46
for t in notes:
    label(48, yy, t, fs=11.3, color="#334155", anchor="start")
    yy += 17

S.append('</svg>')
OUTPUT_PATH.write_text("\n".join(S), encoding="utf-8")
print(f"wrote {OUTPUT_PATH}")
