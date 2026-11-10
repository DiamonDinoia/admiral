// Binding rule 4: output bits must not depend on the buffer address. This is the committed
// nondet2 `align_sweep.cpp` protocol: every (in_off, out_off) in {0..120 step 8}^2 runs one
// transform on offset buffers and the output bytes must be bit-identical across the grid.
// Any differing cell exits non-zero. ctest registers one run per dispatch tier through
// ADM_TEST_TIER (tier_select.cpp pins the engine tier before main).

#include <admiral/admiral.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kBoundary = 128;
constexpr int kOffsets[16] = {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120};

class align_arena {
public:
    explicit align_arena(std::size_t bytes) : m_store(bytes + 2 * kBoundary) {
        const auto addr = reinterpret_cast<std::uintptr_t>(m_store.data());
        m_base = (kBoundary - addr % kBoundary) % kBoundary;
    }
    [[nodiscard]] void* at(int off) { return m_store.data() + m_base + off; }

private:
    std::vector<unsigned char> m_store;
    std::size_t m_base;
};

template<typename T>
[[nodiscard]] std::vector<std::complex<T>> c2c_input(std::size_t n) {
    std::vector<std::complex<T>> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto u = static_cast<double>((i * 2654435761u) % 2000000) / 1000000.0 - 1.0;
        v[i] = std::complex<T>(static_cast<T>(u), T(0));
    }
    return v;
}

template<typename T>
[[nodiscard]] std::vector<T> real_input(std::size_t n) {
    std::vector<T> v(n);
    std::uint64_t s = 0x243F6A8885A308D3ull;
    for (std::size_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v[i] = static_cast<T>(static_cast<double>(s % 2000000) / 1000000.0 - 1.0);
    }
    return v;
}

// Runs `exec(in_ptr, out_ptr)` at every (in_off, out_off) of the 16x16 grid and counts the
// cells whose output bytes differ from the offset-(0, 0) cell.
template<typename T, typename Exec>
[[nodiscard]] int run_grid(const char* kind, std::size_t n, const void* tmpl, std::size_t in_bytes,
                           std::size_t out_bytes, Exec&& exec) {
    align_arena in(in_bytes), out(out_bytes);
    std::vector<unsigned char> base(out_bytes), got(out_bytes);
    int bad = 0;
    bool first = true;
    for (int in_off : kOffsets) {
        for (int out_off : kOffsets) {
            std::memcpy(in.at(in_off), tmpl, in_bytes);
            std::memset(out.at(out_off), 0, out_bytes);
            exec(in.at(in_off), out.at(out_off));
            std::memcpy(got.data(), out.at(out_off), out_bytes);
            if (first) {
                base = got;
                first = false;
            } else if (base != got) {
                ++bad;
            }
        }
    }
    std::printf("%s %s N=%6zu: %d of 256 cells differ\n", bad ? "FAIL" : "ok  ", kind, n, bad);
    return bad;
}

// In place (src == dst) through the 16 offsets: the aliasing wiring must give the same bits
// at every offset of the one buffer.
template<typename T, typename Exec>
[[nodiscard]] int run_grid_inplace(std::size_t n, const void* tmpl, std::size_t bytes,
                                   Exec&& exec) {
    align_arena buf(bytes);
    std::vector<unsigned char> base(bytes), got(bytes);
    int bad = 0;
    bool first = true;
    for (int off : kOffsets) {
        std::memcpy(buf.at(off), tmpl, bytes);
        exec(buf.at(off), buf.at(off));
        std::memcpy(got.data(), buf.at(off), bytes);
        if (first) {
            base = got;
            first = false;
        } else if (base != got) {
            ++bad;
        }
    }
    std::printf("%s c2c-inplace %s N=%6zu: %d of 16 cells differ\n", bad ? "FAIL" : "ok  ",
                sizeof(T) == sizeof(double) ? "f64" : "f32", n, bad);
    return bad;
}

template<typename T>
[[nodiscard]] int sweep_type() {
    const admiral::options opts{.nthreads = 0, .eff = admiral::effort::estimate};
    const char* tname = sizeof(T) == sizeof(double) ? "f64" : "f32";
    int bad = 0;
    for (std::size_t n = 16; n <= 65536; n *= 2) {
        {
            const auto in = c2c_input<T>(n);
            const admiral::strides_plan<T> plan(n, 1, 1, 0, 1, 0, opts);
            char kind[16];
            std::snprintf(kind, sizeof kind, "c2c %s", tname);
            bad += run_grid<T>(kind, n, in.data(), n * sizeof(std::complex<T>),
                               n * sizeof(std::complex<T>),
                               [&](const void* src, void* dst) {
                                   plan.forward(static_cast<const std::complex<T>*>(src),
                                                static_cast<std::complex<T>*>(dst), T(1));
                               });
            bad += run_grid_inplace<T>(n, in.data(), n * sizeof(std::complex<T>),
                                       [&](const void* src, void* dst) {
                                           plan.forward(static_cast<const std::complex<T>*>(src),
                                                        static_cast<std::complex<T>*>(dst), T(1));
                                       });
        }
        {
            const auto in = real_input<T>(n);
            const admiral::plan_r2c<T> plan(n, opts);
            char kind[16];
            std::snprintf(kind, sizeof kind, "r2c %s", tname);
            bad += run_grid<T>(kind, n, in.data(), n * sizeof(T),
                               (n / 2 + 1) * sizeof(std::complex<T>),
                               [&](const void* src, void* dst) {
                                   plan.forward(static_cast<const T*>(src),
                                                static_cast<std::complex<T>*>(dst), T(1));
                               });
        }
    }
    return bad;
}

}  // namespace

int main() {
    const int bad = sweep_type<double>() + sweep_type<float>();
    if (bad) {
        std::printf("FAIL: %d alignment cells differ\n", bad);
        return 1;
    }
    std::printf("PASS: all alignment cells bit-identical\n");
    return 0;
}
