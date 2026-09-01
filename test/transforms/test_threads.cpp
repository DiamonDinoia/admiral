#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/plan.hpp>
#include <admiral/detail/thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

// MSVC spells both the attribute and the aligned allocator differently, and its blocks must go
// back through `_aligned_free`.
#if defined(_MSC_VER)
#include <malloc.h>
#define COUNTED_NOINLINE __declspec(noinline)
#else
#define COUNTED_NOINLINE [[gnu::noinline]]
#endif

// Global replacements count every allocation, libadmiral.so included. `noinline` keeps free()
// out of the caller, where gcc's -Wmismatched-new-delete pairs it with the builtin operator new.
namespace {
std::atomic<long> g_alloc_count{0};

COUNTED_NOINLINE void* counted_alloc(std::size_t n, std::size_t align) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
    void* p = ::_aligned_malloc(n ? n : 1, align);
    if (p == nullptr) throw std::bad_alloc{};
#else
    void* p = nullptr;
    if (::posix_memalign(&p, align, n ? n : 1) != 0) throw std::bad_alloc{};
#endif
    return p;
}
COUNTED_NOINLINE void counted_free(void* p) noexcept {
#if defined(_MSC_VER)
    ::_aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

void* operator new(std::size_t n) { return counted_alloc(n, alignof(std::max_align_t)); }
void* operator new[](std::size_t n) { return counted_alloc(n, alignof(std::max_align_t)); }
void* operator new(std::size_t n, std::align_val_t a) {
    return counted_alloc(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
    return counted_alloc(n, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { counted_free(p); }
void operator delete[](void* p) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t) noexcept { counted_free(p); }
void operator delete(void* p, std::align_val_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { counted_free(p); }

// Execute a plan<long double> at n <= SBO_MAX must not heap-allocate scratch;
// at n > SBO_MAX the soa_scratch heap path must fire.
// plan<long double> routes exclusively through scalar_nd_c2c<long double>, which
// uses soa_scratch<std::complex<long double>, 1> in every execute path.
TEST_CASE("soa_scratch: small-n execute uses zero heap allocations; large-n uses nonzero",
          "[threads][alloc]") {
    // Small: n=64 <= SBO_MAX=4096 — all scratch on stack.
    {
        INFO("small n=64");
        admiral::plan<long double> p({64});
        std::vector<std::complex<long double>> v(64, {1, 0});
        p.forward(v.data());  // warmup: flushes any lazy init
        const long before = g_alloc_count.load(std::memory_order_relaxed);
        p.forward(v.data());
        const long after = g_alloc_count.load(std::memory_order_relaxed);
        REQUIRE(after == before);
    }
    // Large: n=8192 > SBO_MAX=4096 — soa_scratch falls back to heap.
    {
        INFO("large n=8192");
        admiral::plan<long double> p({8192});
        std::vector<std::complex<long double>> v(8192, {1, 0});
        p.forward(v.data());  // warmup
        const long before = g_alloc_count.load(std::memory_order_relaxed);
        p.forward(v.data());
        const long after = g_alloc_count.load(std::memory_order_relaxed);
        REQUIRE(after > before);
    }
}

#if ADM_THREADS

#include <thread>

namespace {

std::string shape_str(const std::vector<std::size_t>& s) {
    std::string r;
    for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "x" : "") + std::to_string(s[i]);
    return r;
}

constexpr std::size_t kNthreads = 4;

template<typename T>
double forecast_tol(std::size_t N) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    return 16.0 * u * std::log2(static_cast<double>(N));
}

}

TEMPLATE_TEST_CASE("c2c N-D nthreads=1 vs 4 agrees within the FFT rounding floor",
                   "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes = {
        {4096},
        {1 << 20},
        {1 << 21},
        {64, 512},
        {16, 8192},
        {67, 512},
        {8, 8, 512},
    };
    for (const auto& shape : shapes) {
        INFO("shape=" << shape_str(shape) << " prec=" << (sizeof(T) == 4 ? "f32" : "f64"));
        std::size_t Ntot = 1;
        for (auto e : shape) Ntot *= e;
        const auto in = make_input<T>(Ntot, 0xC2C0u);
        const admiral::span<const std::size_t> sp(shape.data(), shape.size());

        admiral::plan<T> serial(sp);
        admiral::plan<T> threaded(sp, {kNthreads});

        const double tol = forecast_tol<T>(Ntot);
        auto a = in, b = in;
        serial.forward(a.data());
        threaded.forward(b.data());
        REQUIRE(relerrtwonorm(a, b) < tol);

        serial.inverse(a.data());
        threaded.inverse(b.data());
        REQUIRE(relerrtwonorm(a, b) < tol);
    }
}

TEMPLATE_TEST_CASE("r2c/c2r N-D nthreads=1 vs 4 agrees within the FFT rounding floor",
                   "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes = {
        {64, 512},
        {16, 8192},
        {8, 8, 512},
        {512, 513},
    };
    for (const auto& shape : shapes) {
        INFO("shape=" << shape_str(shape) << " prec=" << (sizeof(T) == 4 ? "f32" : "f64"));
        const admiral::span<const std::size_t> sp(shape.data(), shape.size());
        admiral::plan_r2c<T> serial(sp);
        admiral::plan_r2c<T> threaded(sp, {kNthreads});

        const auto rin = make_real_input<T>(serial.real_size(), 0x2C20u);
        const double tol = forecast_tol<T>(serial.real_size());

        std::vector<std::complex<T>> ca(serial.cplx_size()), cb(serial.cplx_size());
        serial.forward(rin.data(), ca.data());
        threaded.forward(rin.data(), cb.data());
        REQUIRE(relerrtwonorm(ca, cb) < tol);

        auto ca_in = ca, cb_in = cb;
        std::vector<T> ra(serial.real_size()), rb(serial.real_size());
        serial.inverse(ca_in.data(), ra.data());
        threaded.inverse(cb_in.data(), rb.data());
        REQUIRE(relerrtwonorm(ra, rb) < tol);
    }
}

TEMPLATE_TEST_CASE("nthreads=0 auto-select matches serial", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape = {16, 8192};
    const admiral::span<const std::size_t> sp(shape.data(), shape.size());
    const auto in = make_input<T>(16 * 8192, 0xA705u);

    admiral::plan<T> serial(sp);
    admiral::plan<T> autop(sp, {0});

    auto a = in, b = in;
    serial.forward(a.data());
    autop.forward(b.data());
    REQUIRE(relerrtwonorm(a, b) < forecast_tol<T>(16 * 8192));
}

TEST_CASE("parallel_for propagates a body exception, then resets", "[threads]") {
    admiral::detail::thread_pool pool(4);

    std::atomic<int> ran{0};
    REQUIRE_THROWS_AS(pool.parallel_for(64, [&](std::size_t b, std::size_t, std::size_t) {
        ++ran;
        if (b == 0) throw std::runtime_error("chunk failed");
    }), std::runtime_error);
    REQUIRE(ran.load() > 0);

    std::atomic<std::size_t> sum{0};
    REQUIRE_NOTHROW(pool.parallel_for(64, [&](std::size_t b, std::size_t e, std::size_t) {
        for (std::size_t i = b; i < e; ++i) sum += i;
    }));
    REQUIRE(sum.load() == std::size_t{64 * 63 / 2});
}

TEST_CASE("parallel_for with fewer units than threads leaves workers idle", "[threads]") {
    admiral::detail::thread_pool pool(4);
    std::atomic<int> ran{0};
    std::atomic<std::size_t> sum{0};
    pool.parallel_for(2, [&](std::size_t b, std::size_t e, std::size_t) {
        ++ran;
        for (std::size_t i = b; i < e; ++i) sum += i;
    });
    REQUIRE(ran.load() == 2);
    REQUIRE(sum.load() == 1);
}

TEST_CASE("threaded unfused four_step_large agrees across nthreads (double)",
          "[threads][fourstep]") {
    constexpr std::size_t N = 810000;
    REQUIRE(std::string(admiral::detail::plan_impl<double>(N, true, 4).route_name())
            == "four_step_large");
    const auto in = make_input<double>(N, 1234u);
    admiral::plan<double> p2({N}, {2}), p4({N}, {4});
    auto a = in;
    p2.forward(admiral::span(a));
    auto b = in;
    p4.forward(admiral::span(b));
    require_close(b, a, forecast_tol<double>(N));
    p4.inverse(admiral::span(b));
    require_close(b, in, forecast_tol<double>(N));
}

TEST_CASE("threaded out-of-place four_step_large matches serial (double)",
          "[threads][fourstep]") {
    constexpr std::size_t N = 2097152;
    const auto in = make_input<double>(N, 77u);
    admiral::plan<double> p1({N}), p4({N}, {4});
    std::vector<std::complex<double>> o1(N), o4(N);
    p1.forward(in.data(), o1.data());
    p4.forward(in.data(), o4.data());
    require_close(o4, o1, forecast_tol<double>(N));
}

TEST_CASE("resolve_nthreads wake law: serial floor, knee, pocket, pow2, cap", "[threads]") {
    using admiral::detail::dhat_ns;
    using admiral::detail::has_single_bit;
    using admiral::detail::kAutoSerialElems;
    using admiral::detail::kPocketOnsetNs;
    using admiral::detail::kPocketTierNs;
    using admiral::detail::resolve_nthreads;
    using admiral::detail::wake_family_for;
    using admiral::detail::wake_family_row;

    const std::size_t P = resolve_nthreads(0);
    const std::size_t C0 = admiral::detail::cores_per_socket();
    REQUIRE(P >= 1);
    REQUIRE(C0 >= 1);
    const wake_family_row& fam = wake_family_for(P, C0);

    REQUIRE(resolve_nthreads(0, kAutoSerialElems - 1, 3, 1e9, 1) == 1);
    REQUIRE(resolve_nthreads(0, std::size_t{1} << 30, 0, 1e9, 1) == 1);

    REQUIRE(resolve_nthreads(3) == 3);
    REQUIRE(resolve_nthreads(3, kAutoSerialElems - 1, 0, 0, 0) == 3);

    for (std::size_t lg = 15; lg < 34; ++lg)
        for (const std::size_t K : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}})
            for (const unsigned cls : {0u, 1u, 2u})
                for (const double w : {1e4, 1e7, 1e12}) {
                    const std::size_t nt =
                        resolve_nthreads(0, std::size_t{1} << lg, K, w, cls);
                    REQUIRE(nt <= P);
                    REQUIRE((nt == 1 || has_single_bit(nt)));
                }

    for (const unsigned cls : {0u, 1u, 2u}) {
        const std::size_t knee = fam.knee[cls][1];
        const std::size_t want = knee <= P ? knee : std::size_t{1}
                                   << (admiral::detail::bit_width(P) - 1);
        REQUIRE(resolve_nthreads(0, std::size_t{1} << 30, 1, 1e12, cls) == want);
    }

    const auto law_eval = [&](std::size_t total, std::size_t K, double w_ns, unsigned cls) {
        const bool deep = double(total) * 16.0 > double(fam.gateMB[cls]) * 1e6;
        const std::size_t knee = fam.knee[cls][deep];
        std::size_t best = 1;
        double best_t = w_ns;
        for (std::size_t nt = 2; nt <= P; nt <<= 1) {
            const double work = w_ns / double(std::min(nt, knee));
            const double gh = work / double(K);
            const bool pocket = gh < kPocketOnsetNs && nt >= C0 / 2;
            double gap = pocket ? std::max(gh, kPocketTierNs) : gh;
            double dh = dhat_ns(fam, nt, gap);
            for (int i = 0; i < 2; ++i) {
                const double t = work + double(K) * dh;
                gap = t / double(K);
                if (pocket) gap = std::max(gap, kPocketTierNs);
                dh = dhat_ns(fam, nt, gap);
            }
            if (const double t = work + double(K) * dh; t < best_t) { best_t = t; best = nt; }
        }
        return best;
    };
    for (const auto& c : {std::size_t{5}, std::size_t{1}, std::size_t{2}})
        for (const unsigned cls : {0u, 1u, 2u})
            for (const double w : {1e5, 9e6, 3e9}) {
                INFO("K=" << c << " cls=" << cls << " w=" << w);
                REQUIRE(resolve_nthreads(0, std::size_t{1} << 24, c, w, cls) ==
                        law_eval(std::size_t{1} << 24, c, w, cls));
            }

    const auto fsl_pick = [&](std::size_t lg) {
        const std::size_t n = std::size_t{1} << lg;
        const auto sp = admiral::detail::choose_large_split(n);
        const double cyc =
            sp.valid() ? admiral::detail::kFourStepOverhead *
                             (double(sp.n1) * admiral::detail::line_work_cyc<double>(sp.n2) +
                              double(sp.n2) * admiral::detail::line_work_cyc<double>(sp.n1))
                       : admiral::detail::line_work_cyc<double>(n);
        return resolve_nthreads(0, n, 5, cyc / admiral::detail::core_cyc_per_ns(), 0);
    };
    const auto sq2_pick = [&](std::size_t e) {
        const std::size_t n = std::size_t{1} << e;
        const double cyc = 2.0 * double(n) * admiral::detail::line_work_cyc<double>(n);
        return resolve_nthreads(0, n * n, 2, cyc / admiral::detail::core_cyc_per_ns(), 1);
    };
    if (P == 128 && C0 == 64) {
        REQUIRE(fsl_pick(21) == 16);
        REQUIRE(fsl_pick(22) == 32);
        REQUIRE(sq2_pick(11) == 16);
        REQUIRE(sq2_pick(13) == 32);
    }
    if (P == 64 && C0 == 32) {
        REQUIRE(fsl_pick(20) == 8);
        REQUIRE(fsl_pick(21) == 16);
        REQUIRE(sq2_pick(11) == 32);
        REQUIRE(sq2_pick(12) == 64);
    }
    if (P == 96 && C0 == 48) {
        REQUIRE(fsl_pick(20) == 16);
        REQUIRE(fsl_pick(21) == 32);
        REQUIRE(sq2_pick(10) == 32);
        REQUIRE(sq2_pick(13) == 64);
    }
}
// ---------------------------------------------------------------------------
// Concurrent-execute safety: two threads on ONE plan with separate buffers
// must produce results identical to single-threaded execution.
// ---------------------------------------------------------------------------

namespace {

// Spin barrier so both threads enter the critical section simultaneously.
struct spin_barrier {
    explicit spin_barrier(int n) : cnt(n) {}
    void arrive() { cnt.fetch_sub(1, std::memory_order_release); }
    void wait()   { while (cnt.load(std::memory_order_acquire) != 0) {} }
    std::atomic<int> cnt;
};

// Run body0 and body1 concurrently; both spin until the other is ready.
template<typename B0, typename B1>
void concurrent_pair(B0&& body0, B1&& body1) {
    spin_barrier bar(2);
    std::thread t0([&]{ bar.arrive(); bar.wait(); body0(); });
    std::thread t1([&]{ bar.arrive(); bar.wait(); body1(); });
    t0.join();
    t1.join();
}

} // namespace

TEMPLATE_TEST_CASE("concurrent execute on one plan: correctness at nthreads=1 and nthreads=4",
                   "[threads][concurrent]", float, double) {
    // ---- 1-D c2c (four_step_large at nt=4) ----
    {
        constexpr std::size_t N = 1u << 20;
        const auto in0 = make_input<TestType>(N, 0xAAAAu);
        const auto in1 = make_input<TestType>(N, 0xBBBBu);
        const auto tol = forecast_tol<TestType>(N);
        for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
            INFO("1-D c2c N=" << N << " nt=" << nt);
            admiral::plan<TestType> p({N}, {nt});
            auto ref0 = in0, ref1 = in1;
            p.forward(ref0.data());
            p.forward(ref1.data());
            auto d0 = in0, d1 = in1;
            concurrent_pair([&]{ p.forward(d0.data()); },
                            [&]{ p.forward(d1.data()); });
            REQUIRE(relerrtwonorm(d0, ref0) < tol);
            REQUIRE(relerrtwonorm(d1, ref1) < tol);
        }
    }

    // ---- N-D c2c ----
    {
        const std::vector<std::size_t> shape = {64, 512};
        const std::size_t Ntot = 64 * 512;
        const auto in0 = make_input<TestType>(Ntot, 0xCCCCu);
        const auto in1 = make_input<TestType>(Ntot, 0xDDDDu);
        const auto tol = forecast_tol<TestType>(Ntot);
        for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
            INFO("N-D c2c {64,512} nt=" << nt);
            admiral::plan<TestType> p(shape, {nt});
            auto ref0 = in0, ref1 = in1;
            p.forward(ref0.data());
            p.forward(ref1.data());
            auto d0 = in0, d1 = in1;
            concurrent_pair([&]{ p.forward(d0.data()); },
                            [&]{ p.forward(d1.data()); });
            REQUIRE(relerrtwonorm(d0, ref0) < tol);
            REQUIRE(relerrtwonorm(d1, ref1) < tol);
        }
    }

    // ---- plan_r2c (multi-row to exercise tile_scratch_) ----
    {
        const std::vector<std::size_t> shape = {4, 1024};
        const auto tol = forecast_tol<TestType>(4 * 1024);
        for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
            INFO("plan_r2c {4,1024} nt=" << nt);
            admiral::plan_r2c<TestType> p(shape, {nt});
            const auto rin0 = make_real_input<TestType>(p.real_size(), 0xEEEEu);
            const auto rin1 = make_real_input<TestType>(p.real_size(), 0xFFFFu);
            std::vector<std::complex<TestType>> ref0(p.cplx_size()), ref1(p.cplx_size());
            p.forward(rin0.data(), ref0.data());
            p.forward(rin1.data(), ref1.data());
            std::vector<std::complex<TestType>> c0(p.cplx_size()), c1(p.cplx_size());
            concurrent_pair([&]{ p.forward(rin0.data(), c0.data()); },
                            [&]{ p.forward(rin1.data(), c1.data()); });
            REQUIRE(relerrtwonorm(c0, ref0) < tol);
            REQUIRE(relerrtwonorm(c1, ref1) < tol);
        }
    }

    // ---- plan_r2r (exercises v_ and spec_ buffers) ----
    {
        constexpr std::size_t N = 512;
        constexpr std::size_t rows = 4;
        const auto tol = forecast_tol<TestType>(N);
        for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
            INFO("plan_r2r N=" << N << " rows=" << rows << " nt=" << nt);
            admiral::plan_r2r<TestType> p(N, admiral::r2r_kind::dct2, rows, {nt});
            const std::size_t sz = p.size();
            const auto in0 = make_real_input<TestType>(sz, 0x1111u);
            const auto in1 = make_real_input<TestType>(sz, 0x2222u);
            std::vector<TestType> ref0(sz), ref1(sz), d0(sz), d1(sz);
            p.forward(in0.data(), ref0.data());
            p.forward(in1.data(), ref1.data());
            concurrent_pair([&]{ p.forward(in0.data(), d0.data()); },
                            [&]{ p.forward(in1.data(), d1.data()); });
            REQUIRE(relerrtwonorm(d0, ref0) < tol);
            REQUIRE(relerrtwonorm(d1, ref1) < tol);
        }
    }

    // ---- axis_plan ----
    {
        const std::vector<std::size_t> shape = {16, 512};
        const std::size_t Ntot = 16 * 512;
        const auto tol = forecast_tol<TestType>(512);
        for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
            INFO("axis_plan {16,512} axis=0 nt=" << nt);
            admiral::axis_plan<TestType> p(shape, 0, true, {nt});
            const auto in0 = make_input<TestType>(Ntot, 0x3333u);
            const auto in1 = make_input<TestType>(Ntot, 0x4444u);
            auto ref0 = in0, ref1 = in1;
            p.execute(ref0.data(), {}, {});
            p.execute(ref1.data(), {}, {});
            auto d0 = in0, d1 = in1;
            concurrent_pair([&]{ p.execute(d0.data(), {}, {}); },
                            [&]{ p.execute(d1.data(), {}, {}); });
            REQUIRE(relerrtwonorm(d0, ref0) < tol);
            REQUIRE(relerrtwonorm(d1, ref1) < tol);
        }
    }

    // ---- axis_plan, innermost axis: unit stride picks a different route ----
    {
        const std::vector<std::size_t> shape = {16, 512};
        const std::size_t Ntot = 16 * 512;
        const auto tol = forecast_tol<TestType>(512);
        for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
            INFO("axis_plan {16,512} axis=1 nt=" << nt);
            admiral::axis_plan<TestType> p(shape, 1, true, {nt});
            const auto in0 = make_input<TestType>(Ntot, 0xA3A3u);
            const auto in1 = make_input<TestType>(Ntot, 0xB4B4u);
            auto ref0 = in0, ref1 = in1;
            p.execute(ref0.data(), {}, {});
            p.execute(ref1.data(), {}, {});
            auto d0 = in0, d1 = in1;
            concurrent_pair([&]{ p.execute(d0.data(), {}, {}); },
                            [&]{ p.execute(d1.data(), {}, {}); });
            REQUIRE(relerrtwonorm(d0, ref0) < tol);
            REQUIRE(relerrtwonorm(d1, ref1) < tol);
        }
    }

    // ---- strides_plan ----
    {
        constexpr std::size_t len = 512, nbatch = 4;
        const std::size_t Ntot = len * nbatch;
        const auto tol = forecast_tol<TestType>(len);
        for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
            INFO("strides_plan len=" << len << " nbatch=" << nbatch << " nt=" << nt);
            admiral::strides_plan<TestType> p(len, nbatch, 1, len, 1, len, {nt});
            const auto in0 = make_input<TestType>(Ntot, 0x5555u);
            const auto in1 = make_input<TestType>(Ntot, 0x6666u);
            std::vector<std::complex<TestType>> ref0(Ntot), ref1(Ntot);
            p.forward(in0.data(), ref0.data());
            p.forward(in1.data(), ref1.data());
            std::vector<std::complex<TestType>> d0(Ntot), d1(Ntot);
            concurrent_pair([&]{ p.forward(in0.data(), d0.data()); },
                            [&]{ p.forward(in1.data(), d1.data()); });
            REQUIRE(relerrtwonorm(d0, ref0) < tol);
            REQUIRE(relerrtwonorm(d1, ref1) < tol);
        }
    }

    // ---- two distinct plans, constructed and executed concurrently ----
    {
        const std::vector<std::size_t> shape = {64, 512};
        const std::size_t Ntot = 64 * 512;
        const admiral::span<const std::size_t> sp(shape.data(), shape.size());
        const auto in0 = make_input<TestType>(Ntot, 0xE1E1u);
        const auto in1 = make_input<TestType>(Ntot, 0xF2F2u);
        const auto tol = forecast_tol<TestType>(Ntot);
        auto ref0 = in0, ref1 = in1;
        {
            admiral::plan<TestType> ref(sp, {std::size_t{4}});
            ref.forward(ref0.data());
            ref.forward(ref1.data());
        }
        auto d0 = in0, d1 = in1;
        auto build_and_run = [&sp](std::vector<std::complex<TestType>>& d) {
            admiral::plan<TestType> own(sp, {std::size_t{4}});
            own.forward(d.data());
        };
        concurrent_pair([&]{ build_and_run(d0); }, [&]{ build_and_run(d1); });
        REQUIRE(relerrtwonorm(d0, ref0) < tol);
        REQUIRE(relerrtwonorm(d1, ref1) < tol);
    }

    // ---- strides_plan on the slab route: in_stride != 1, in_dist == 1, out_dist != 1 ----
    {
        for (std::size_t len : {std::size_t{64}, std::size_t{96}, std::size_t{256},
                                std::size_t{512}}) {
            for (std::size_t nbatch : {std::size_t{4}, std::size_t{8}}) {
                const std::size_t Ntot = len * nbatch;
                const auto tol = forecast_tol<TestType>(len);
                for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
                    INFO("strides_plan slab len=" << len << " nbatch=" << nbatch
                                                  << " nt=" << nt);
                    admiral::strides_plan<TestType> p(len, nbatch, nbatch, 1, 1, len, {nt});
                    const auto in0 = make_input<TestType>(Ntot, 0x7777u);
                    const auto in1 = make_input<TestType>(Ntot, 0x8888u);
                    std::vector<std::complex<TestType>> ref0(Ntot), ref1(Ntot);
                    p.forward(in0.data(), ref0.data());
                    p.forward(in1.data(), ref1.data());
                    std::vector<std::complex<TestType>> d0(Ntot), d1(Ntot);
                    concurrent_pair([&]{ p.forward(in0.data(), d0.data()); },
                                    [&]{ p.forward(in1.data(), d1.data()); });
                    REQUIRE(relerrtwonorm(d0, ref0) < tol);
                    REQUIRE(relerrtwonorm(d1, ref1) < tol);
                }
            }
        }
    }

    // ---- Bluestein and Rader routes ----
    {
        // Find a size that routes to bluestein at estimate effort.
        std::size_t N_blue = 0;
        for (std::size_t n = 40; n < 300 && N_blue == 0; ++n)
            if (std::string(admiral::detail::plan_impl<TestType>(n, true).route_name()) ==
                "bluestein")
                N_blue = n;
        REQUIRE(N_blue != 0);

        // Find a size that routes to rader.
        std::size_t N_rader = 0;
        for (std::size_t n = 5; n < 200 && N_rader == 0; ++n)
            if (std::string(admiral::detail::plan_impl<TestType>(n, true).route_name()) == "rader")
                N_rader = n;
        REQUIRE(N_rader != 0);

        for (std::size_t N : {N_blue, N_rader}) {
            const std::string rname = admiral::detail::plan_impl<TestType>(N, true).route_name();
            const auto tol = forecast_tol<TestType>(N);
            INFO("route=" << rname << " N=" << N);
            for (std::size_t nt : {std::size_t{1}, std::size_t{4}}) {
                INFO("  nt=" << nt);
                admiral::plan<TestType> p({N}, {nt});
                const auto in0 = make_input<TestType>(N, 0x7777u);
                const auto in1 = make_input<TestType>(N, 0x8888u);
                auto ref0 = in0, ref1 = in1;
                p.forward(ref0.data());
                p.forward(ref1.data());
                auto d0 = in0, d1 = in1;
                concurrent_pair([&]{ p.forward(d0.data()); },
                                [&]{ p.forward(d1.data()); });
                REQUIRE(relerrtwonorm(d0, ref0) < tol);
                REQUIRE(relerrtwonorm(d1, ref1) < tol);
            }
        }
    }

    // ---- distinct plans are always independent (control) ----
    {
        constexpr std::size_t N = 65536;
        const auto in = make_input<TestType>(N, 0x9999u);
        const auto tol = forecast_tol<TestType>(N);
        admiral::plan<TestType> p0({N}, {4}), p1({N}, {4});
        auto d0 = in, d1 = in;
        auto ref = in;
        p0.forward(ref.data());
        concurrent_pair([&]{ p0.forward(d0.data()); },
                        [&]{ p1.forward(d1.data()); });
        REQUIRE(relerrtwonorm(d0, ref) < tol);
        REQUIRE(relerrtwonorm(d1, ref) < tol);
    }
}

#endif
