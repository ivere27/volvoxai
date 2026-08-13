/* Non-maximum suppression and its IoU helper. */

static double vx_kernel_non_max_iou(const void* boxes, int boxes_dtype,
                                    int spatial, int batch, int first, int second) {
    long first_base = ((long)batch * spatial + first) * 4;
    long second_base = ((long)batch * spatial + second) * 4;
    double first_y1 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 0);
    double first_x1 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 1);
    double first_y2 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 2);
    double first_x2 = vx_kernel_value_at(boxes, boxes_dtype, first_base + 3);
    double second_y1 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 0);
    double second_x1 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 1);
    double second_y2 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 2);
    double second_x2 = vx_kernel_value_at(boxes, boxes_dtype, second_base + 3);
    double xx1 = vx_kernel_max(first_x1, second_x1);
    double yy1 = vx_kernel_max(first_y1, second_y1);
    double xx2 = vx_kernel_min(first_x2, second_x2);
    double yy2 = vx_kernel_min(first_y2, second_y2);
    double width = vx_kernel_max(0.0, xx2 - xx1);
    double height = vx_kernel_max(0.0, yy2 - yy1);
    double intersection = width * height;
    double first_area = (first_x2 - first_x1) * (first_y2 - first_y1);
    double second_area = (second_x2 - second_x1) * (second_y2 - second_y1);
    return intersection / (first_area + second_area - intersection);
}

/* Allocation-free greedy NMS.  Selected indices are retained in the output
 * rows while a class is being processed, avoiding a runtime heap dependency. */
void non_max_suppression_typed(const void* boxes, const void* scores, void* output,
                               int batches, int spatial, int classes, int output_records,
                               int boxes_dtype, int scores_dtype, int output_dtype,
                               double max_output_boxes_per_class, double iou_threshold,
                               double score_threshold) {
    if (!boxes || !scores || !output || batches < 0 || spatial < 0 || classes < 0 ||
        output_records < 0 || !vx_kernel_dtype_supported(boxes_dtype) ||
        !vx_kernel_dtype_supported(scores_dtype) ||
        !vx_kernel_dtype_supported(output_dtype)) return;

    int output_index = 0;
    for (int batch = 0; batch < batches && output_index < output_records; batch++) {
        for (int class_index = 0; class_index < classes && output_index < output_records;
             class_index++) {
            int class_start = output_index;
            for (int selected = 0;
                 (double)selected < max_output_boxes_per_class && output_index < output_records;
                 selected++) {
                int best = -1;
                double best_score = 0.0;
                for (int candidate = 0; candidate < spatial; candidate++) {
                    long score_index = ((long)batch * classes + class_index) * spatial + candidate;
                    double score = vx_kernel_value_at(scores, scores_dtype, score_index);
                    if (!(score >= score_threshold)) continue;
                    int suppressed = 0;
                    for (int prior = 0; prior < selected; prior++) {
                        int previous = (int)vx_kernel_value_at(output, output_dtype,
                            ((long)(class_start + prior) * 3) + 2);
                        if (vx_kernel_non_max_iou(boxes, boxes_dtype, spatial, batch,
                                candidate, previous) > iou_threshold) {
                            suppressed = 1;
                            break;
                        }
                    }
                    /* The source candidate order is ascending spatial index;
                     * not replacing equal scores reproduces stable JS sort. */
                    if (!suppressed && (best < 0 || score > best_score)) {
                        best = candidate;
                        best_score = score;
                    }
                }
                if (best < 0) break;
                vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 0, batch);
                vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 1, class_index);
                vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 2, best);
                output_index++;
            }
        }
    }
    while (output_index < output_records) {
        vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 0, -1);
        vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 1, -1);
        vx_kernel_store_index(output, output_dtype, (long)output_index * 3 + 2, -1);
        output_index++;
    }
}
