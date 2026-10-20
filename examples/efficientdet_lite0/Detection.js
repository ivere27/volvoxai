// Application policy: retain the original Lite0 anchor decode and class-aware NMS.
const W = 320, H = 320, C = 90;
const clamp01 = value => Math.max(0, Math.min(1, value));
const anchors = [];
for (const stride of [8, 16, 32, 64, 128]) {
  for (let y = 0; y < Math.ceil(H / stride); y++) for (let x = 0; x < Math.ceil(W / stride); x++) {
    for (const scale of [1, Math.pow(2, 1 / 3), Math.pow(2, 2 / 3)]) for (const ratio of [1, 2, 0.5]) {
      const size = 4 * stride * scale, sr = Math.sqrt(ratio);
      anchors.push({y: (y + 0.5) * stride / H, x: (x + 0.5) * stride / W,
        h: size / sr / H, w: size * sr / W});
    }
  }
}

function decodeBox(raw, index) {
  const a = anchors[index];
  if (!a) return null;
  const yCenter = raw[0] / 10 * a.h + a.y, xCenter = raw[1] / 10 * a.w + a.x;
  const bh = Math.exp(Math.max(-6, Math.min(6, raw[2] / 5))) * a.h;
  const bw = Math.exp(Math.max(-6, Math.min(6, raw[3] / 5))) * a.w;
  const ymin = clamp01(yCenter - bh / 2), xmin = clamp01(xCenter - bw / 2);
  const ymax = clamp01(yCenter + bh / 2), xmax = clamp01(xCenter + bw / 2);
  return xmax <= xmin || ymax <= ymin ? null : {ymin, xmin, ymax, xmax};
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin));
  const ih = Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
  const inter = iw * ih, aa = (a.xmax - a.xmin) * (a.ymax - a.ymin);
  const ba = (b.xmax - b.xmin) * (b.ymax - b.ymin);
  return inter / Math.max(1e-9, aa + ba - inter);
}

export function decodeDetections(scores, boxes, labels, {scoreThreshold = 0.2, nmsThreshold = 0.45, maxDetections = 12} = {}) {
  if (!scores || !boxes) return [];
  const count = Math.min(Math.floor(boxes.length / 4), anchors.length);
  if (scores.length < count * C) throw new Error('EfficientDet scores do not cover the box anchors');
  const candidates = [], fallback = [];
  for (let d = 0; d < count; d++) {
    let cls = -1, score = -Infinity;
    for (let c = 0; c < C; c++) {
      if (!labels[c] || labels[c] === '???') continue;
      const value = scores[d * C + c];
      if (value > score) { score = value; cls = c; }
    }
    if (cls < 0) continue;
    const box = decodeBox(boxes.subarray(d * 4, d * 4 + 4), d);
    if (!box) continue;
    const det = {index: d, score, cls, label: labels[cls] || String(cls), ...box};
    if (score >= scoreThreshold) candidates.push(det);
    fallback.push(det);
  }
  const pool = candidates.length ? candidates : fallback;
  pool.sort((a, b) => b.score - a.score);
  const selected = [];
  for (const cand of pool.slice(0, 300)) {
    if (!selected.some(picked => cand.cls === picked.cls && iou(cand, picked) > nmsThreshold)) selected.push(cand);
    if (selected.length >= maxDetections) break;
  }
  return selected;
}

const colors = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#9333ea', '#0891b2', '#ea580c',
  '#4f46e5', '#059669', '#ca8a04', '#db2777', '#7c3aed', '#0284c7', '#65a30d', '#c026d3',
  '#0d9488', '#e11d48', '#b45309', '#4338ca', '#15803d'];
export const classColor = cls => colors[cls % colors.length];
