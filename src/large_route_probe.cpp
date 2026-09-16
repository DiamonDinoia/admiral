// Process-wide storage for the serial large-route line seam (override + probe switch).
// The override slot must have exactly ONE address per process: the engine ships in a
// shared library while test binaries build with hidden visibility, so a header-local
// static would split into one copy per module and the library would never see a test's
// injected line. See include/admiral/detail/four_step_large.hpp for the seam's contract.
#include <admiral/detail/four_step_large.hpp>

#include <atomic>
#include <cstdlib>

namespace admiral {
namespace detail {

namespace {
[[nodiscard]] std::atomic<std::size_t>& override_slot(std::size_t elem_bytes) {
    static std::atomic<std::size_t> f64{0}, f32{0};
    return elem_bytes == 16 ? f64 : f32;
}
}  // namespace

std::size_t large_route_serial_override(std::size_t elem_bytes) {
    return override_slot(elem_bytes).load(std::memory_order_relaxed);
}

void set_large_route_serial_override(std::size_t elem_bytes, std::size_t bytes) {
    override_slot(elem_bytes).store(bytes, std::memory_order_relaxed);
}

bool large_route_probe_disabled() {
    static const bool off = [] {
        const char* e = std::getenv("ADM_LARGE_ROUTE_PROBE");
        return e != nullptr && e[0] == '0' && e[1] == '\0';
    }();
    return off;
}

}  // namespace detail
}  // namespace admiral
