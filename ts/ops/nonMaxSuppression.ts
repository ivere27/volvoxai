export function _cpuNonMaxSuppression(node) {

    const boxes = node.inputs.boxes;
    const scores = node.inputs.scores;
    const out = node.outputs.out;
    
    let max_output_boxes_per_class = 0;
    if (node.inputs.max_output_boxes_per_class) max_output_boxes_per_class = node.inputs.max_output_boxes_per_class.buffer[0];
    let iou_threshold = 0.5;
    if (node.inputs.iou_threshold) iou_threshold = node.inputs.iou_threshold.buffer[0];
    let score_threshold = 0.0;
    if (node.inputs.score_threshold) score_threshold = node.inputs.score_threshold.buffer[0];

    const num_batches = boxes.shape[0];
    const spatial_dimension = boxes.shape[1];
    const num_classes = scores.shape[1];
    
    let outIdx = 0;
    for (let b = 0; b < num_batches; b++) {
        for (let c = 0; c < num_classes; c++) {
            const candidates: Array<{
                s: number;
                score: number;
                y1: number;
                x1: number;
                y2: number;
                x2: number;
            }> = [];
            for (let s = 0; s < spatial_dimension; s++) {
                const score = scores.buffer[b * (num_classes * spatial_dimension) + c * spatial_dimension + s];
                if (score >= score_threshold) {
                    const y1 = boxes.buffer[b * (spatial_dimension * 4) + s * 4 + 0];
                    const x1 = boxes.buffer[b * (spatial_dimension * 4) + s * 4 + 1];
                    const y2 = boxes.buffer[b * (spatial_dimension * 4) + s * 4 + 2];
                    const x2 = boxes.buffer[b * (spatial_dimension * 4) + s * 4 + 3];
                    candidates.push({s, score, y1, x1, y2, x2});
                }
            }
            candidates.sort((a, b_) => b_.score - a.score);
            const selected: typeof candidates = [];
            for (let i = 0; i < candidates.length && selected.length < max_output_boxes_per_class; i++) {
                const cand = candidates[i];
                let keep = true;
                for (let j = 0; j < selected.length; j++) {
                    const sel = selected[j];
                    const xx1 = Math.max(cand.x1, sel.x1);
                    const yy1 = Math.max(cand.y1, sel.y1);
                    const xx2 = Math.min(cand.x2, sel.x2);
                    const yy2 = Math.min(cand.y2, sel.y2);
                    const w = Math.max(0, xx2 - xx1);
                    const h = Math.max(0, yy2 - yy1);
                    const inter = w * h;
                    const areaCand = (cand.x2 - cand.x1) * (cand.y2 - cand.y1);
                    const areaSel = (sel.x2 - sel.x1) * (sel.y2 - sel.y1);
                    const iou = inter / (areaCand + areaSel - inter);
                    if (iou > iou_threshold) {
                        keep = false;
                        break;
                    }
                }
                if (keep) {
                    selected.push(cand);
                    if (outIdx < out.buffer.length / 3) {
                        out.buffer[outIdx * 3 + 0] = b;
                        out.buffer[outIdx * 3 + 1] = c;
                        out.buffer[outIdx * 3 + 2] = cand.s;
                        outIdx++;
                    }
                }
            }
        }
    }
    while (outIdx < out.buffer.length / 3) {
        out.buffer[outIdx * 3 + 0] = -1;
        out.buffer[outIdx * 3 + 1] = -1;
        out.buffer[outIdx * 3 + 2] = -1;
        outIdx++;
    }
  }
