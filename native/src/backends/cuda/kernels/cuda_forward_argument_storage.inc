/* This private helper is only for proven global device buffers. Shared/local
 * workspace pointers must retain their own address space. */
template <typename T>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE T* vx_cuda_global_pointer(T* value) {
#if defined(__clang__) && defined(__CUDA_ARCH__)
    typedef T GlobalValue __attribute__((address_space(1)));
    static_assert(sizeof(T*) == 8 && sizeof(unsigned long long) == 8,
                  "CUDA global pointers require a 64-bit device address");
    return (T*)(GlobalValue*)(unsigned long long)value;
#else
    return value;
#endif
}

/* Prepared incremental graphs bind one stable record per kernel. The host
 * refreshes all records in one ordered upload before executing the graph.
 * Every scalar or pointer occupies an eight-byte slot, matching the generated
 * PTX argument image. Ordinary launches pass NULL and use their direct args.
 * Numerical bodies are shared; this changes submission, not arithmetic. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_restore_arguments(
        const unsigned char*) {}

template <typename T>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_restore_argument(
        const unsigned char* storage, T& value) {
    value = *reinterpret_cast<const T*>(storage);
}

template <typename T>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_restore_argument(
        const unsigned char* storage, T*& value) {
    value = vx_cuda_global_pointer(*reinterpret_cast<T* const*>(storage));
}

template <typename T, typename... Remaining>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_restore_arguments(
        const unsigned char* storage, T& value, Remaining&... remaining) {
    static_assert(sizeof(T) <= 8 && alignof(T) <= 8,
                  "prepared CUDA argument must fit an aligned scalar slot");
    vx_cuda_restore_argument(storage, value);
    vx_cuda_restore_arguments(storage + 8, remaining...);
}
