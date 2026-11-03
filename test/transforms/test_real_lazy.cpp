// A c2r-only plan must never build the forward c2c twin: the twin costs half of a fresh
// plan's small-N build and c2r runs only the lazy inverse. One shared real plan must also
// serve concurrent r2c and c2r callers bit-equal to serial runs.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/admiral.hpp>
#include <admiral/detail/config.hpp>

#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#if ADM_THREADS
#include <thread>
#endif
#include <vector>

// Allocation counting, the shape test_threads.cpp established: the replacements stay out
// of line, so GCC does not report -Wmismatched-new-delete on the malloc/free pair.
namespace {
std::atomic<long> g_allocs{0};
[[gnu::noinline]] void count_alloc() noexcept {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

[[gnu::noinline]] void* operator new(std::size_t n) {
    count_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
[[gnu::noinline]] void* operator new[](std::size_t n) { return ::operator new(n); }
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

constexpr std::size_t kN = 1009;

// Budget: the c2r-only lifecycle (ctor + two inverse calls) builds one c2c twin, the
// inverse. The forward twin costs one whole plan_impl build of allocations on top of
// that. Measured on this test: fix 8, base 14 (see REPORT.md); the budget sits between.
constexpr long kC2rOnlyAllocBudget = 10;

// Deterministic route election: effort::measure times candidates and can elect
// different routes per plan, which would break bit equality.
const admiral::options kOpts{0, admiral::effort::estimate};

}  // namespace

TEMPLATE_TEST_CASE("a c2r-only plan never builds the forward c2c twin", "[real][lazy]",
                   float, double) {
    {
        // Warm-up: the first plan in the process pays one-time dispatch and route costs.
        admiral::plan_r2c<TestType> warm(kN);
        std::vector<TestType> x(kN, TestType(1));
        std::vector<std::complex<TestType>> sp(warm.cplx_size());
        std::vector<TestType> y(kN);
        warm.forward(x.data(), sp.data());
        warm.inverse(sp.data(), y.data());
    }
    const std::size_t nspec = kN / 2 + 1;
    std::vector<std::complex<TestType>> spec(nspec);
    for (std::size_t i = 0; i < nspec; ++i)
        spec[i] = {TestType(i % 13), TestType(i % 7)};
    std::vector<TestType> out(kN);
    const long a0 = g_allocs.load(std::memory_order_relaxed);
    {
        admiral::plan_r2c<TestType> p(kN, kOpts);
        p.inverse(spec.data(), out.data());
        p.inverse(spec.data(), out.data());
    }
    const long used = g_allocs.load(std::memory_order_relaxed) - a0;
    std::printf("c2r_only_allocs n=%zu count=%ld budget=%ld\n", kN, used,
                kC2rOnlyAllocBudget);
    INFO("c2r-only lifecycle allocations " << used);
    REQUIRE(used <= kC2rOnlyAllocBudget);
}

#if ADM_THREADS

namespace {

// The yield-bearing barrier test_threads.cpp established: a bare spin convoys under TSan
// and oversubscription and livelocks the lazy-build race.
struct spin_barrier {
    explicit spin_barrier(int n) : cnt(n) {}
    void arrive() { cnt.fetch_sub(1, std::memory_order_release); }
    void wait() {
        while (cnt.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
    std::atomic<int> cnt;
};

}  // namespace

TEMPLATE_TEST_CASE("8 threads run r2c and c2r on one shared plan bit-equal to serial",
                   "[real][threads]", float, double) {
    constexpr std::size_t kReps = 4;
    constexpr std::size_t kHalf = 4;  // 4 r2c threads + 4 c2r threads
    for (const std::size_t n : {std::size_t{97}, std::size_t{256}}) {
        const std::size_t nspec = n / 2 + 1;
        std::vector<TestType> x(n);
        for (std::size_t i = 0; i < n; ++i)
            x[i] = TestType((i * 7) % 31) * TestType(0.5) - TestType((i * 13) % 17);
        std::vector<std::complex<TestType>> spec0(nspec);
        for (std::size_t i = 0; i < nspec; ++i)
            spec0[i] = {TestType(i % 13), TestType(i % 7)};
        std::vector<std::complex<TestType>> ref_r2c(nspec);
        std::vector<TestType> ref_c2r(n);
        {
            admiral::plan_r2c<TestType> p(n, kOpts);
            p.forward(x.data(), ref_r2c.data());
            p.inverse(spec0.data(), ref_c2r.data());
        }
        for (std::size_t rep = 0; rep < kReps; ++rep) {
            // A fresh plan per repetition races the lazy twin builds on first use.
            admiral::plan_r2c<TestType> p(n, kOpts);
            std::vector<std::vector<std::complex<TestType>>> outs(
                kHalf, std::vector<std::complex<TestType>>(nspec));
            std::vector<std::vector<TestType>> routs(kHalf, std::vector<TestType>(n));
            std::vector<std::vector<std::complex<TestType>>> specs(kHalf, spec0);
            spin_barrier bar(int(2 * kHalf));
            std::vector<std::thread> ts;
            ts.reserve(2 * kHalf);
            for (std::size_t t = 0; t < 2 * kHalf; ++t) {
                const bool fwd = t < kHalf;
                ts.emplace_back([&, t, fwd] {
                    bar.arrive();
                    bar.wait();
                    if (fwd) p.forward(x.data(), outs[t].data());
                    else     p.inverse(specs[t - kHalf].data(), routs[t - kHalf].data());
                });
            }
            for (auto& th : ts) th.join();
            for (std::size_t t = 0; t < kHalf; ++t) {
                INFO("n=" << n << " rep=" << rep << " r2c thread " << t);
                REQUIRE(outs[t] == ref_r2c);
                INFO("n=" << n << " rep=" << rep << " c2r thread " << t);
                REQUIRE(routs[t] == ref_c2r);
            }
        }
    }
}

#endif  // ADM_THREADS
