/* Performance choices contain no tensor pointers or admission proofs. The
 * process cache survives module release; each context freezes its own choices
 * before capture. Its lock never protects Driver/module lifetime. */
#define CUDA_MATMUL_LOCAL_CHOICES 32u
#define CUDA_MATMUL_SHARED_CHOICES 64u
#define CUDA_MATMUL_TUNING_BUDGET_NS UINT64_C(50000000)
#define CUDA_MATMUL_TUNING_REPEATS 8u

typedef struct {
    int device;
    CudaMatmulChoice choice;
} CudaMatmulSharedChoice;

static struct {
    pthread_mutex_t mutex;
    uint32_t count;
    uint64_t elapsed_ns;
    CudaMatmulSharedChoice choices[CUDA_MATMUL_SHARED_CHOICES];
} cuda_matmul_choices = { .mutex = PTHREAD_MUTEX_INITIALIZER };

static int cuda_matmul_choice_matches(const CudaMatmulChoice* choice,
    uint32_t rows, uint32_t input_width, uint32_t output_width, int out_in) {
    return choice->rows == rows && choice->input_width == input_width &&
        choice->output_width == output_width && choice->out_in == (uint32_t)out_in;
}

static int cuda_matmul_tuning_prepare(CudaContextState* state) {
    if (!state->matmul_workspace || !p_cuEventElapsedTime) return 1;
    CUresult status = cuda_memory_allocate(&state->matmul_tuning.output,
        VX_CUDA_MATMUL_CALIBRATION_OUTPUT_BYTES);
    if (status == CUDA_ERROR_OUT_OF_MEMORY) return 1;
    if (!cuda_ok(status, "cuMemAlloc(matmul calibration)")) return 0;
    status = p_cuEventCreate(&state->matmul_tuning.start, 0u);
    if (status == CUDA_ERROR_OUT_OF_MEMORY) return 1;
    if (!cuda_ok(status, "cuEventCreate(matmul calibration start)")) return 0;
    status = p_cuEventCreate(&state->matmul_tuning.end, 0u);
    if (status == CUDA_ERROR_OUT_OF_MEMORY) return 1;
    return cuda_ok(status, "cuEventCreate(matmul calibration end)");
}

/* An unresolved wait retains every owner for the ordinary cleanup retry. */
static int cuda_matmul_tuning_release(CudaContextState* state) {
    CudaMatmulTuning* tuning = &state->matmul_tuning;
    if (!tuning->output && !tuning->start && !tuning->end) return 1;
    CudaContextGuard guard;
    if (!cuda_context_enter(&guard)) return 0;
    int ok = cuda_ok(cuda_trace_wait_stream(cuda_stream),
        "cuStreamSynchronize(matmul calibration)");
    if (ok && tuning->start) {
        ok = cuda_ok(p_cuEventDestroy(tuning->start), "cuEventDestroy(matmul calibration start)");
        if (ok) tuning->start = NULL;
    }
    if (ok && tuning->end) {
        ok = cuda_ok(p_cuEventDestroy(tuning->end), "cuEventDestroy(matmul calibration end)");
        if (ok) tuning->end = NULL;
    }
    if (ok && tuning->output) {
        ok = cuda_ok(cuda_memory_free(tuning->output), "cuMemFree(matmul calibration)");
        if (ok) tuning->output = 0;
    }
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}

static int cuda_matmul_tuning_launch(CudaContextState* state,
    CUdeviceptr input, CUdeviceptr weight, CUdeviceptr bias,
    uint32_t rows, uint32_t input_width, uint32_t output_width,
    uint32_t out_in, uint32_t split_k_slices) {
    CUdeviceptr output = state->matmul_tuning.output;
    CUdeviceptr argument_storage = 0;
    if (!split_k_slices) {
        void* arguments[8] = {&input, &weight, &bias, &output,
            &rows, &input_width, &output_width, &argument_storage};
        return cuda_ok(p_cuLaunchKernel(out_in ? CUDA_FORWARD_FUNCTION(k_matmul_out_in)
                : CUDA_FORWARD_FUNCTION(k_matmul),
            (output_width + 15u) / 16u, (rows + 15u) / 16u, 1u,
            8u, 16u, 1u, 0u, cuda_stream, arguments, NULL),
            "cuLaunchKernel(matmul calibration tiled)");
    }
    void* partial_arguments[8] = {&input, &weight, &state->matmul_workspace,
        &rows, &input_width, &output_width, &split_k_slices, &argument_storage};
    void* reduce_arguments[11] = {&input, &weight, &bias,
        &state->matmul_workspace, &output, &rows, &input_width,
        &output_width, &out_in, &split_k_slices, &argument_storage};
    return cuda_ok(p_cuLaunchKernel(out_in ? CUDA_FORWARD_FUNCTION(k_matmul_split_out_in)
            : CUDA_FORWARD_FUNCTION(k_matmul_split),
        (output_width + 31u) / 32u, 1u, split_k_slices, 8u, 16u, 1u,
        0u, cuda_stream, partial_arguments, NULL), "cuLaunchKernel(matmul calibration split)") &&
        cuda_ok(p_cuLaunchKernel(CUDA_FORWARD_FUNCTION(k_matmul_split_reduce),
        (rows * output_width + 127u) / 128u, 1u, 1u, 128u, 1u, 1u,
        0u, cuda_stream, reduce_arguments, NULL), "cuLaunchKernel(matmul calibration reduction)");
}

static int cuda_matmul_tuning_measure(CudaContextState* state,
    CUdeviceptr input, CUdeviceptr weight, CUdeviceptr bias,
    uint32_t rows, uint32_t input_width, uint32_t output_width,
    uint32_t out_in, uint32_t split_k_slices, float* milliseconds) {
    if (!cuda_ok(p_cuEventRecord(state->matmul_tuning.start, cuda_stream),
            "cuEventRecord(matmul calibration start)")) return 0;
    for (unsigned int repeat = 0u; repeat < CUDA_MATMUL_TUNING_REPEATS; repeat++)
        if (!cuda_matmul_tuning_launch(state, input, weight, bias,
                rows, input_width, output_width, out_in, split_k_slices)) return 0;
    return cuda_ok(p_cuEventRecord(state->matmul_tuning.end, cuda_stream),
            "cuEventRecord(matmul calibration end)") &&
        cuda_ok(p_cuEventSynchronize(state->matmul_tuning.end),
            "cuEventSynchronize(matmul calibration)") &&
        cuda_ok(p_cuEventElapsedTime(milliseconds, state->matmul_tuning.start,
            state->matmul_tuning.end), "cuEventElapsedTime(matmul calibration)") &&
        isfinite(*milliseconds) && *milliseconds > 0.0f;
}

static float cuda_matmul_tuning_median(float a, float b, float c) {
    if (a > b) { float value = a; a = b; b = value; }
    if (b > c) { float value = b; b = c; c = value; }
    return a > b ? a : b;
}

/* The input, weights and optional bias are current, proved read-only views.
 * Only this context's calibration output and partial workspace are written.
 * Raw launches intentionally bypass production replay and Trace recording. */
static int cuda_matmul_choose_slices(CudaContextState* state,
    CUdeviceptr input, CUdeviceptr weight, CUdeviceptr bias,
    uint32_t rows, uint32_t input_width, uint32_t output_width,
    int out_in, uint32_t* selected) {
    CudaMatmulTuning* tuning = &state->matmul_tuning;
    *selected = 0u;
    for (uint32_t index = 0u; index < tuning->count; index++)
        if (cuda_matmul_choice_matches(&tuning->choices[index],
                rows, input_width, output_width, out_in)) {
            *selected = tuning->choices[index].split_k_slices;
            return 1;
        }
    if (tuning->count >= CUDA_MATMUL_LOCAL_CHOICES) return 1;
    uint32_t slices = vx_cuda_matmul_split_default_slices(rows, input_width,
        output_width, cuda_device_state.multiprocessors,
        cuda_device_state.matmul_resident_blocks[out_in]);
    /* Unknown capacity keeps the tiled route. A new key during an existing
     * replay must not change its launch plan. */
    if (!cuda_device_state.multiprocessors ||
        !cuda_device_state.matmul_resident_blocks[out_in] ||
        state->replay_active->pass == CUDA_REPLAY_PASS_CAPTURE ||
        state->replay_active->pass == CUDA_REPLAY_PASS_VALIDATE ||
        state->replay_active->capture_active) {
        tuning->choices[tuning->count++] =
            (CudaMatmulChoice){rows, input_width, output_width, (uint32_t)out_in, 0u};
        return 1;
    }
    if (pthread_mutex_trylock(&cuda_matmul_choices.mutex) == 0) {
        int found = 0;
        for (uint32_t index = 0u; index < cuda_matmul_choices.count; index++) {
            CudaMatmulSharedChoice* entry = &cuda_matmul_choices.choices[index];
            if (entry->device == cuda_device_state.selected_device_index &&
                cuda_matmul_choice_matches(&entry->choice,
                    rows, input_width, output_width, out_in)) {
                slices = entry->choice.split_k_slices;
                found = 1;
                break;
            }
        }
        int can_measure = !found && cuda_matmul_choices.count < CUDA_MATMUL_SHARED_CHOICES &&
            cuda_matmul_choices.elapsed_ns < CUDA_MATMUL_TUNING_BUDGET_NS &&
            tuning->output && tuning->start && tuning->end &&
            state->forward_scope_active && state->forward_context_ready &&
            !state->forward_context_failed && !state->trace_pass.scope;
#if VOLVOXAI_ENABLE_TRAINING
        can_measure = can_measure && !state->training_active;
#endif
        if (can_measure) {
            uint32_t candidates[5] = {0u}, count = 1u;
            for (uint32_t candidate = 2u; candidate <= 16u; candidate *= 2u)
                if (vx_cuda_matmul_split_workspace_bytes(rows, input_width, output_width, candidate) &&
                    cuda_launch_dimensions_valid((output_width + 31u) / 32u, 1u, candidate,
                        8u, 16u, 1u, 17152u)) candidates[count++] = candidate;
            uint64_t begin = vx_trace_now_ns();
            float times[5][3] = {{0}};
            int ok = 1, complete = 1;
            for (uint32_t round = 0u; round < 3u && ok && complete; round++)
                for (uint32_t ordinal = 0u; ordinal < count; ordinal++) {
                    uint32_t index = round == 1u ? count - 1u - ordinal : ordinal;
                    if (cuda_matmul_choices.elapsed_ns + vx_trace_now_ns() - begin >=
                            CUDA_MATMUL_TUNING_BUDGET_NS) { complete = 0; break; }
                    if (!round) ok = cuda_matmul_tuning_launch(state, input, weight, bias,
                        rows, input_width, output_width, (uint32_t)out_in, candidates[index]);
                    if (ok) ok = cuda_matmul_tuning_measure(state, input, weight, bias,
                        rows, input_width, output_width, (uint32_t)out_in,
                        candidates[index], &times[index][round]);
                    if (!ok) break;
                }
            uint64_t elapsed = vx_trace_now_ns() - begin;
            cuda_matmul_choices.elapsed_ns += elapsed;
            tuning->elapsed_ns += elapsed;
            tuning->calibrations++;
            if (!ok) {
                state->forward_context_failed = 1;
                (void)pthread_mutex_unlock(&cuda_matmul_choices.mutex);
                return 0;
            }
            if (complete) {
                float medians[5];
                float best = INFINITY;
                for (uint32_t index = 0u; index < count; index++) {
                    medians[index] = cuda_matmul_tuning_median(
                        times[index][0], times[index][1], times[index][2]);
                    if (medians[index] < best) best = medians[index];
                }
                /* Prefer no split, then fewer slices, when differences are small. */
                for (uint32_t index = 0u; index < count; index++)
                    if (medians[index] <= best * 1.03f) { slices = candidates[index]; break; }
            }
            CudaMatmulSharedChoice* entry = &cuda_matmul_choices.choices[cuda_matmul_choices.count++];
            entry->device = cuda_device_state.selected_device_index;
            entry->choice = (CudaMatmulChoice){rows, input_width, output_width, (uint32_t)out_in, slices};
        }
        (void)pthread_mutex_unlock(&cuda_matmul_choices.mutex);
    }
    tuning->choices[tuning->count++] =
        (CudaMatmulChoice){rows, input_width, output_width, (uint32_t)out_in, slices};
    *selected = slices;
    return 1;
}
