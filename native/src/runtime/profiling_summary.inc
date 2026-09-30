/* Included by profiling.c. GetTraceSummary: deterministic aggregates of a
 * READY trace, computed from the records ListTraceEvents returns. Rows never
 * mix host and device time or node, program and call layers, because those
 * intervals overlap. Grouped records are sorted by their key, so each group
 * is one contiguous run. */

static _Thread_local VxTraceSummaryGrouping vx_summary_sort_grouping;

static int vx_summary_record_selected(const VxTraceRecord* r, VxTraceSummaryGrouping grouping) {
    if (r->kind == VX_TRACE_EVENT_MEMORY) return 0;
    switch (grouping) {
    default:
        return 0;
    case VX_TRACE_SUMMARY_GROUPING_NODE:
    case VX_TRACE_SUMMARY_GROUPING_OPERATOR:
        return r->kind != VX_TRACE_EVENT_HOST_OPERATION && r->schedule_index >= 0 &&
            !r->program[0] && r->activity == VX_TRACE_ACTIVITY_COMPUTE;
    case VX_TRACE_SUMMARY_GROUPING_OPERATION:
        return r->kind == VX_TRACE_EVENT_HOST_OPERATION;
    case VX_TRACE_SUMMARY_GROUPING_ACTIVITY:
        return r->activity == VX_TRACE_ACTIVITY_COPY || r->activity == VX_TRACE_ACTIVITY_SUBMIT ||
            r->activity == VX_TRACE_ACTIVITY_SYNCHRONIZE || r->activity == VX_TRACE_ACTIVITY_COMPLETION;
    }
}
static int vx_summary_u64(uint64_t a, uint64_t b) { return a < b ? -1 : a > b; }
static int vx_summary_lineage(const VxExecutionIdentity* a, const VxExecutionIdentity* b) {
    int c = vx_summary_u64(a->model_id, b->model_id);
    if (!c) c = vx_summary_u64(a->compiled_model_id, b->compiled_model_id);
    if (!c) c = vx_summary_u64(a->context_id, b->context_id);
    return c;
}
/* The share scope: rows compared with each other in `share`. */
static int vx_summary_scope(const VxTraceRecord* a, const VxTraceRecord* b, VxTraceSummaryGrouping grouping) {
    int c = 0;
    if (grouping != VX_TRACE_SUMMARY_GROUPING_OPERATOR && grouping != VX_TRACE_SUMMARY_GROUPING_ACTIVITY)
        c = vx_summary_lineage(&a->identity, &b->identity);
    if (!c && grouping == VX_TRACE_SUMMARY_GROUPING_NODE) c = vx_summary_u64(a->plan_id, b->plan_id);
    if (!c) c = strcmp(a->backend, b->backend);
    if (!c && grouping != VX_TRACE_SUMMARY_GROUPING_OPERATION) c = (int)a->phase - (int)b->phase;
    if (!c) c = (a->kind == VX_TRACE_EVENT_DEVICE) - (b->kind == VX_TRACE_EVENT_DEVICE);
    if (!c && grouping == VX_TRACE_SUMMARY_GROUPING_ACTIVITY) c = (int)a->activity - (int)b->activity;
    return c;
}
static int vx_summary_compare_records(const void* left, const void* right) {
    const VxTraceRecord* a = *(const VxTraceRecord* const*)left;
    const VxTraceRecord* b = *(const VxTraceRecord* const*)right;
    VxTraceSummaryGrouping grouping = vx_summary_sort_grouping;
    int c = vx_summary_scope(a, b, grouping);
    if (!c && grouping == VX_TRACE_SUMMARY_GROUPING_NODE) c = a->schedule_index - b->schedule_index;
    if (!c) c = strcmp(a->name, b->name);
    if (!c) c = vx_summary_u64(a->sequence, b->sequence); /* stable, deterministic */
    return c;
}
static int vx_summary_same_group(const VxTraceRecord* a, const VxTraceRecord* b, VxTraceSummaryGrouping grouping) {
    return !vx_summary_scope(a, b, grouping) && !strcmp(a->name, b->name) &&
        (grouping != VX_TRACE_SUMMARY_GROUPING_NODE || a->schedule_index == b->schedule_index);
}
static int vx_summary_compare_u64(const void* left, const void* right) {
    return vx_summary_u64(*(const uint64_t*)left, *(const uint64_t*)right);
}
static int vx_summary_compare_rows(const void* left, const void* right) {
    const VxTraceSummaryRow* a = left;
    const VxTraceSummaryRow* b = right;
    int c = vx_summary_u64(b->total_ns, a->total_ns);
    if (!c) c = strcmp(a->name, b->name);
    if (!c) c = a->schedule_index - b->schedule_index;
    if (!c) c = vx_summary_u64(a->plan_id, b->plan_id);
    return c;
}
static const VxExecutionPlan* vx_summary_plan(const VxTraceView* view, uint64_t plan_id) {
    return plan_id && plan_id <= view->plan_count ? vx_trace_plan_at(view, (size_t)(plan_id - 1)) : NULL;
}
/* Rates need a known cost and time that measures that work. Decode steps run
 * only part of the plan's rows, so their plan cost does not apply. */
static int vx_summary_rate_applies(const VxTraceRecord* r, const VxTraceRecord* const* operations,
    size_t operation_count) {
    if (r->kind != VX_TRACE_EVENT_DEVICE && strcmp(r->backend, "cpu") && strcmp(r->backend, "wasm"))
        return 0;
    for (size_t i = 0; i < operation_count; i++)
        if (operations[i]->identity.execution_id == r->identity.execution_id && r->identity.execution_id)
            return strncmp(operations[i]->name, "Decode", 6) != 0;
    return 1;
}

int vx_trace_summarize(const VxTraceView* view, VxTraceSummaryGrouping grouping,
    size_t max_rows, VxTraceSummary* summary) {
    memset(summary, 0, sizeof(*summary));
    if (!view->records && view->count) return -1;
    size_t selected = 0, operation_count = 0;
    for (size_t i = 0; i < view->count; i++) {
        const VxTraceRecord* r = &view->records[i];
        if (r->kind == VX_TRACE_EVENT_HOST_OPERATION) operation_count++;
        if (!vx_summary_record_selected(r, grouping)) continue;
        if (r->truncated) summary->excluded++; else selected++;
    }
    const VxTraceRecord** records = calloc(selected ? selected : 1, sizeof(*records));
    const VxTraceRecord** operations = calloc(operation_count ? operation_count : 1, sizeof(*operations));
    uint64_t* durations = calloc(selected ? selected : 1, sizeof(*durations));
    VxTraceSummaryRow* rows = calloc(selected ? selected : 1, sizeof(*rows));
    if (!records || !operations || !durations || !rows) {
        free(records); free(operations); free(durations); free(rows);
        return -1;
    }
    size_t n = 0, o = 0;
    for (size_t i = 0; i < view->count; i++) {
        const VxTraceRecord* r = &view->records[i];
        if (r->kind == VX_TRACE_EVENT_HOST_OPERATION) operations[o++] = r;
        if (vx_summary_record_selected(r, grouping) && !r->truncated) records[n++] = r;
    }
    vx_summary_sort_grouping = grouping;
    qsort(records, n, sizeof(*records), vx_summary_compare_records);
    size_t row_count = 0;
    for (size_t begin = 0, end; begin < n; begin = end) {
        end = begin + 1;
        while (end < n && vx_summary_same_group(records[begin], records[end], grouping)) end++;
        const VxTraceRecord* first = records[begin];
        VxTraceSummaryRow* row = &rows[row_count++];
        row->identity = first->identity;
        row->identity.execution_id = 0;
        row->plan_id = grouping == VX_TRACE_SUMMARY_GROUPING_NODE ? first->plan_id : 0;
        row->backend = first->backend; row->name = first->name;
        row->phase = first->phase; row->activity = first->activity;
        row->device = first->kind == VX_TRACE_EVENT_DEVICE;
        row->schedule_index = grouping == VX_TRACE_SUMMARY_GROUPING_NODE ? first->schedule_index : -1;
        row->output = grouping == VX_TRACE_SUMMARY_GROUPING_NODE ? first->output : "";
        row->fused = grouping == VX_TRACE_SUMMARY_GROUPING_NODE && first->fused;
        int rates = grouping == VX_TRACE_SUMMARY_GROUPING_NODE || grouping == VX_TRACE_SUMMARY_GROUPING_OPERATOR;
        double flops = 0.0, bytes = 0.0;
        for (size_t i = begin; i < end; i++) {
            const VxTraceRecord* r = records[i];
            durations[i - begin] = r->duration_ns;
            row->total_ns += r->duration_ns;
            row->copy_bytes += r->copy_bytes;
            if (!rates) continue;
            const VxExecutionPlan* plan = vx_summary_plan(view, r->plan_id);
            const VxExecutionPlanStep* step = plan && r->schedule_index < (int)plan->step_count
                ? &plan->steps[r->schedule_index] : NULL;
            if (!step || step->cost_status == VX_COST_STATUS_UNKNOWN || !step->cost_status ||
                !vx_summary_rate_applies(r, operations, o)) { rates = 0; continue; }
            flops += 2.0 * (double)step->cost_macs;
            bytes += (double)step->cost_input_bytes + (double)step->cost_output_bytes;
            if (grouping == VX_TRACE_SUMMARY_GROUPING_NODE) row->step = step;
        }
        row->count = end - begin;
        qsort(durations, row->count, sizeof(*durations), vx_summary_compare_u64);
        row->min_ns = durations[0]; row->max_ns = durations[row->count - 1];
        row->median_ns = durations[(row->count + 1) / 2 - 1];
        row->p95_ns = durations[(row->count * 95 + 99) / 100 - 1];
        if (rates && row->total_ns) {
            row->has_rates = 1;
            row->flops_per_second = flops * 1e9 / (double)row->total_ns;
            row->bytes_per_second = bytes * 1e9 / (double)row->total_ns;
        }
        if (grouping == VX_TRACE_SUMMARY_GROUPING_NODE && !row->step) {
            const VxExecutionPlan* plan = vx_summary_plan(view, first->plan_id);
            if (plan && first->schedule_index < (int)plan->step_count) row->step = &plan->steps[first->schedule_index];
        }
        /* Groups sharing a scope are adjacent: accumulate the scope total into
         * share now and divide once the scope ends. */
        row->share = (double)row->total_ns;
    }
    /* Rows are in key order, so rows of one share scope are contiguous. */
    for (size_t begin = 0, end; begin < row_count; begin = end) {
        end = begin + 1;
        const VxTraceRecord key = {.identity = rows[begin].identity, .plan_id = rows[begin].plan_id,
            .backend = rows[begin].backend, .phase = rows[begin].phase, .activity = rows[begin].activity,
            .kind = rows[begin].device ? VX_TRACE_EVENT_DEVICE : VX_TRACE_EVENT_HOST_NODE};
        double total = rows[begin].share;
        while (end < row_count) {
            const VxTraceRecord other = {.identity = rows[end].identity, .plan_id = rows[end].plan_id,
                .backend = rows[end].backend, .phase = rows[end].phase, .activity = rows[end].activity,
                .kind = rows[end].device ? VX_TRACE_EVENT_DEVICE : VX_TRACE_EVENT_HOST_NODE};
            if (vx_summary_scope(&key, &other, grouping)) break;
            total += rows[end++].share;
        }
        for (size_t i = begin; i < end; i++) rows[i].share = total > 0.0 ? rows[i].share / total : 0.0;
    }
    qsort(rows, row_count, sizeof(*rows), vx_summary_compare_rows);
    if (max_rows && row_count > max_rows) { row_count = max_rows; summary->truncated = 1; }
    free(records); free(operations); free(durations);
    summary->rows = rows;
    summary->count = row_count;
    return 0;
}
void vx_trace_summary_free(VxTraceSummary* summary) {
    free(summary->rows);
    memset(summary, 0, sizeof(*summary));
}
