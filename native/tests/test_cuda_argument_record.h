/* Feed extracted production kernels the same aligned scalar image used by
 * the driver. Stored calls deliberately use invalid direct arguments, proving
 * that current records supply every operand and geometry value. */
template <class... Values> struct alignas(8) CudaArgumentRecord {
    std::array<unsigned char, 8 * sizeof...(Values)> bytes{};
    explicit CudaArgumentRecord(Values... values) {
        unsigned slot = 0;
        auto store = [&](auto value) {
            static_assert(sizeof(value) <= 8, "scalar argument slot");
            std::memcpy(bytes.data() + 8 * slot++, &value, sizeof(value));
        };
        (store(values), ...);
    }
};
template <class... Values> static auto cuda_argument_record(Values... values) {
    return CudaArgumentRecord<Values...>(values...);
}
