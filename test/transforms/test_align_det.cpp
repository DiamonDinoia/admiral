// Output bits must not depend on where the input or output buffer sits in memory.
// Every c2c and r2c transform runs at every 8-byte offset 0..120 from a 128-byte
// boundary for both buffers and the outputs must be bit-identical across the grid.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <admiral/admiral.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
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

// Runs `exec(in_ptr, out_ptr)` on the 16x16 (input, output) offset grid and counts the
// cells whose output bytes differ from the offset-(0, 0) cell.
template<typename T, typename Exec>
void run_grid(const void* tmpl, std::size_t in_bytes, std::size_t out_elems, Exec&& exec) {
    const std::size_t out_bytes = out_elems * sizeof(std::complex<T>);
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
    INFO("alignment cells that differ: " << bad << " of 256");
    REQUIRE(bad == 0);
}

// In place (src == dst) through the same offsets: the aliasing wiring and the last-pass
// overlap block must give the same bits at every offset of the one buffer.
template<typename T, typename Exec>
void run_grid_inplace(const void* tmpl, std::size_t bytes, Exec&& exec) {
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
    INFO("in-place alignment cells that differ: " << bad << " of 16");
    REQUIRE(bad == 0);
}

}

TEMPLATE_TEST_CASE("c2c output bits do not depend on buffer alignment", "[align][det]", double,
                   float) {
    const admiral::options opts{.nthreads = 0, .eff = admiral::effort::estimate};
    for (std::size_t n = 16; n <= 65536; n *= 2) {
        const auto in = c2c_input<TestType>(n);
        const admiral::strides_plan<TestType> plan(n, 1, 1, 0, 1, 0, opts);
        INFO("c2c n = " << n);
        run_grid<TestType>(in.data(), n * sizeof(std::complex<TestType>), n,
                           [&](const void* src, void* dst) {
            plan.forward(static_cast<const std::complex<TestType>*>(src),
                         static_cast<std::complex<TestType>*>(dst));
        });
    }
}

TEMPLATE_TEST_CASE("c2c in-place output bits do not depend on buffer alignment", "[align][det]",
                   double, float) {
    const admiral::options opts{.nthreads = 0, .eff = admiral::effort::estimate};
    for (std::size_t n = 16; n <= 65536; n *= 2) {
        const auto in = c2c_input<TestType>(n);
        const admiral::strides_plan<TestType> plan(n, 1, 1, 0, 1, 0, opts);
        INFO("c2c n = " << n);
        run_grid_inplace<TestType>(in.data(), n * sizeof(std::complex<TestType>),
                                   [&](const void* src, void* dst) {
            plan.forward(static_cast<const std::complex<TestType>*>(src),
                         static_cast<std::complex<TestType>*>(dst));
        });
    }
}

TEMPLATE_TEST_CASE("r2c output bits do not depend on buffer alignment", "[align][det]", double,
                   float) {
    const admiral::options opts{.nthreads = 0, .eff = admiral::effort::estimate};
    for (std::size_t n = 16; n <= 65536; n *= 2) {
        const auto in = real_input<TestType>(n);
        const admiral::plan_r2c<TestType> plan(n, opts);
        INFO("r2c n = " << n);
        run_grid<TestType>(in.data(), n * sizeof(TestType), n / 2 + 1,
                           [&](const void* src, void* dst) {
            plan.forward(static_cast<const TestType*>(src),
                         static_cast<std::complex<TestType>*>(dst));
        });
    }
}
