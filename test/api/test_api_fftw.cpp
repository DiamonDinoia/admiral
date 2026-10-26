#include <catch2/catch_test_macros.hpp>
#include <climits>
#include <cstdint>
#include <cmath>
#include <complex>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#endif

// Both failure-injection cases below fork a child under a hard RLIMIT_AS cap. That mechanism
// is Linux/glibc only: darwin's setrlimit refuses the cap, and asan/tsan reserve a shadow
// address space the cap kills at startup. The execute case additionally calls malloc_trim and
// reads /proc/self/statm.
#if defined(__linux__) && defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__) && \
    !defined(__SANITIZE_THREAD__)
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define ADM_TEST_FFTW_SANITIZED 1
#endif
#endif
#if !defined(ADM_TEST_FFTW_SANITIZED)
#define ADM_TEST_FFTW_RLIMIT_FAILURE 1
#include <cstdlib>
#include <fstream>
#include <limits>
#include <malloc.h>
#include <new>
#endif
#endif

#include <admiral/admiral.hpp>
#include <admiral/detail/scratch.hpp>
#include <admiral/fftw3.h>

#include "utils/reference.hpp"

#include <string>

namespace {
std::vector<std::complex<double>> signal(int N) {
    std::vector<std::complex<double>> v(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) v[static_cast<std::size_t>(i)] = {std::sin(0.3 * i), std::cos(0.17 * i)};
    return v;
}
}

TEST_CASE("fftw shim: forward matches native unnormalized forward", "[fftw]") {
    for (int N : {8, 12, 15, 64, 100}) {
        auto in = signal(N);
        std::vector<std::complex<double>> shim(static_cast<std::size_t>(N)), native = in;

        fftw_plan p = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(in.data()),
                                       reinterpret_cast<fftw_complex*>(shim.data()),
                                       FFTW_FORWARD, FFTW_ESTIMATE);
        REQUIRE(p != nullptr);
        fftw_execute(p);
        fftw_destroy_plan(p);

        admiral::plan<double> np(static_cast<std::size_t>(N));
        np.forward(admiral::span(native), 1.0);

        require_close(shim, native, fft_tol<double>());
    }
}

TEST_CASE("fftw shim: forward then backward is N*x (unnormalized)", "[fftw]") {
    const int N = 96;
    auto x = signal(N);
    std::vector<std::complex<double>> buf = x;

    fftw_plan fwd = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(buf.data()),
                                     reinterpret_cast<fftw_complex*>(buf.data()),
                                     FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(buf.data()),
                                     reinterpret_cast<fftw_complex*>(buf.data()),
                                     FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);
    fftw_execute(bwd);
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);

    for (std::size_t i = 0; i < x.size(); ++i) x[i] *= double(N);
    require_close(buf, x, fft_tol<double>(2.0));
}

TEST_CASE("fftw shim: 2-D plan and execute_dft on fresh arrays", "[fftw]") {
    const int n0 = 6, n1 = 10, N = n0 * n1;
    auto a = signal(N), b = signal(N);
    for (auto& z : b) z += std::complex<double>(1.0, -0.5);
    std::vector<std::complex<double>> oa(static_cast<std::size_t>(N)), ob(static_cast<std::size_t>(N));

    fftw_plan p = fftw_plan_dft_2d(n0, n1, reinterpret_cast<fftw_complex*>(a.data()),
                                   reinterpret_cast<fftw_complex*>(oa.data()),
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_execute_dft(p, reinterpret_cast<fftw_complex*>(b.data()),
                     reinterpret_cast<fftw_complex*>(ob.data()));
    fftw_destroy_plan(p);

    admiral::plan<double> np({std::size_t(n0), std::size_t(n1)});
    auto ra = a, rb = b;
    np.forward(admiral::span(ra), 1.0);
    np.forward(admiral::span(rb), 1.0);
    require_close(ra, oa, fft_tol<double>());
    require_close(rb, ob, fft_tol<double>());
}

TEST_CASE("fftw shim: 3-D plan and out-of-place inverse", "[fftw]") {
    const int n0 = 4, n1 = 6, n2 = 5, N = n0 * n1 * n2;
    auto in = signal(N);
    std::vector<std::complex<double>> spec(static_cast<std::size_t>(N)),
                                      back(static_cast<std::size_t>(N));

    fftw_plan fwd = fftw_plan_dft_3d(n0, n1, n2, reinterpret_cast<fftw_complex*>(in.data()),
                                     reinterpret_cast<fftw_complex*>(spec.data()),
                                     FFTW_FORWARD, FFTW_ESTIMATE);
    REQUIRE(fwd != nullptr);
    fftw_plan bwd = fftw_plan_dft_3d(n0, n1, n2, reinterpret_cast<fftw_complex*>(spec.data()),
                                     reinterpret_cast<fftw_complex*>(back.data()),
                                     FFTW_BACKWARD, FFTW_ESTIMATE);
    REQUIRE(bwd != nullptr);
    fftw_execute(fwd);
    fftw_execute(bwd);
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);

    admiral::plan<double> np({std::size_t(n0), std::size_t(n1), std::size_t(n2)});
    auto ref = in;
    np.forward(admiral::span(ref), 1.0);
    require_close(spec, ref, fft_tol<double>());

    for (auto& z : in) z *= double(N);
    require_close(back, in, fft_tol<double>(2.0));
}

TEST_CASE("fftw shim: rejects bad ranks and an overflowing shape", "[fftw]") {
    const int n[3] = {4, 4, 4};
    REQUIRE(fftw_plan_dft(0, n, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE) == nullptr);
    REQUIRE(fftw_plan_dft(1, nullptr, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE) == nullptr);
    const int zero[2] = {4, 0};
    REQUIRE(fftw_plan_dft(2, zero, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE) == nullptr);
    const int huge[3] = {1 << 30, 1 << 30, 1 << 30};
    REQUIRE(fftw_plan_dft(3, huge, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE) == nullptr);
}

TEST_CASE("fftw shim: single-precision 2-D, 3-D and execute_dft", "[fftw]") {
    const int n0 = 4, n1 = 6, n2 = 5;
    for (const bool three_d : {false, true}) {
        const int N = three_d ? n0 * n1 * n2 : n0 * n1;
        std::vector<std::complex<float>> a(static_cast<std::size_t>(N)),
                                        b(static_cast<std::size_t>(N)),
                                        oa(static_cast<std::size_t>(N)),
                                        ob(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
            a[static_cast<std::size_t>(i)] = {std::sin(0.3f * float(i)), std::cos(0.17f * float(i))};
            b[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] + std::complex<float>(1.f, -0.5f);
        }

        fftwf_plan p = three_d
            ? fftwf_plan_dft_3d(n0, n1, n2, reinterpret_cast<fftwf_complex*>(a.data()),
                                reinterpret_cast<fftwf_complex*>(oa.data()), FFTW_FORWARD, FFTW_ESTIMATE)
            : fftwf_plan_dft_2d(n0, n1, reinterpret_cast<fftwf_complex*>(a.data()),
                                reinterpret_cast<fftwf_complex*>(oa.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        REQUIRE(p != nullptr);
        fftwf_execute(p);
        fftwf_execute_dft(p, reinterpret_cast<fftwf_complex*>(b.data()),
                          reinterpret_cast<fftwf_complex*>(ob.data()));
        fftwf_destroy_plan(p);

        std::vector<std::size_t> shape = three_d
            ? std::vector<std::size_t>{std::size_t(n0), std::size_t(n1), std::size_t(n2)}
            : std::vector<std::size_t>{std::size_t(n0), std::size_t(n1)};
        admiral::plan<float> np{admiral::span<const std::size_t>(shape)};
        auto ra = a, rb = b;
        np.forward(admiral::span(ra), 1.0f);
        np.forward(admiral::span(rb), 1.0f);
        require_close(ra, oa, fft_tol<float>());
        require_close(rb, ob, fft_tol<float>());
    }
}

TEST_CASE("fftw shim: allocator wrappers, both precisions", "[fftw]") {
    auto* zd = fftw_alloc_complex(16);
    auto* rd = fftw_alloc_real(16);
    auto* md = fftw_malloc(64);
    REQUIRE(zd != nullptr);
    REQUIRE(rd != nullptr);
    REQUIRE(md != nullptr);
    zd[0][0] = 1.0; zd[15][1] = 2.0; rd[15] = 3.0;
    REQUIRE(zd[0][0] == 1.0);
    REQUIRE(rd[15] == 3.0);
    fftw_free(zd);
    fftw_free(rd);
    fftw_free(md);
    fftw_cleanup();

    auto* rf = fftwf_alloc_real(16);
    auto* mf = fftwf_malloc(64);
    REQUIRE(rf != nullptr);
    REQUIRE(mf != nullptr);
    rf[15] = 3.f;
    REQUIRE(rf[15] == 3.f);
    fftwf_free(rf);
    fftwf_free(mf);
    fftwf_cleanup();
}

TEST_CASE("fftw shim: aligned malloc and single precision", "[fftw]") {
    auto* buf = fftwf_alloc_complex(32);
    REQUIRE(buf != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(buf) % admiral::detail::span_align<float> == 0);
    for (int i = 0; i < 32; ++i) { buf[i][0] = float(i); buf[i][1] = 0.f; }

    fftwf_plan fwd = fftwf_plan_dft_1d(32, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan bwd = fftwf_plan_dft_1d(32, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(fwd);
    fftwf_execute(bwd);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);

    std::vector<std::complex<float>> got(32), want(32);
    for (int i = 0; i < 32; ++i) {
        got[static_cast<std::size_t>(i)] = {buf[i][0], buf[i][1]};
        want[static_cast<std::size_t>(i)] = {float(i) * 32.f, 0.f};
    }
    require_close(got, want, fft_tol<float>(2.0));
    fftwf_free(buf);
}

TEST_CASE("fftw shim: null plans and null arrays are no-ops, malloc(0) is valid",
          "[fftw]") {
    REQUIRE_NOTHROW(fftw_execute(nullptr));
    REQUIRE_NOTHROW(fftwf_execute(nullptr));
    REQUIRE_NOTHROW(fftw_execute_dft(nullptr, nullptr, nullptr));
    REQUIRE_NOTHROW(fftwf_execute_dft(nullptr, nullptr, nullptr));

    fftw_complex* in = static_cast<fftw_complex*>(fftw_malloc(32 * sizeof(fftw_complex)));
    fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(32 * sizeof(fftw_complex)));
    REQUIRE(in != nullptr);
    REQUIRE(out != nullptr);
    fftw_plan p = fftw_plan_dft_1d(32, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    REQUIRE(p != nullptr);
    REQUIRE_NOTHROW(fftw_execute_dft(p, nullptr, out));
    fftw_destroy_plan(p);

    void* z = fftw_malloc(0);
    void* zf = fftwf_malloc(0);
    REQUIRE(z != nullptr);
    REQUIRE(zf != nullptr);
    fftw_free(z);
    fftw_free(zf);
    fftw_free(in);
    fftw_free(out);
}

TEST_CASE("fftw shim: FFTW_MEASURE plans run and stay correct", "[fftw]") {
    constexpr int N = 60;
    fftw_complex* in = static_cast<fftw_complex*>(fftw_malloc(N * sizeof(fftw_complex)));
    fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(N * sizeof(fftw_complex)));
    fftw_complex* alt = static_cast<fftw_complex*>(fftw_malloc(N * sizeof(fftw_complex)));
    REQUIRE(in != nullptr);
    REQUIRE(out != nullptr);
    REQUIRE(alt != nullptr);
    for (int i = 0; i < N; ++i) {
        in[i][0] = std::sin(i * 0.7);
        in[i][1] = std::cos(i * 0.3);
    }
    fftw_plan est = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    REQUIRE(est != nullptr);
    fftw_execute(est);
    for (const unsigned flags : {FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE,
                                 FFTW_MEASURE | FFTW_DESTROY_INPUT}) {
        CAPTURE(flags);
        fftw_plan mea = fftw_plan_dft_1d(N, in, alt, FFTW_FORWARD, flags);
        REQUIRE(mea != nullptr);
        fftw_execute(mea);
        constexpr double kPlanEquivTol = 1e-9;
        double err = 0.0;
        for (int i = 0; i < N; ++i)
            err += std::hypot(out[i][0] - alt[i][0], out[i][1] - alt[i][1]);
        REQUIRE(err <= kPlanEquivTol);
        fftw_destroy_plan(mea);
    }
    fftw_destroy_plan(est);
    fftw_free(in);
    fftw_free(out);
    fftw_free(alt);
}

#if defined(ADM_TEST_FFTW_RLIMIT_FAILURE)
namespace {

// Child exit codes other than 76 skip nothing: 2 means the child could not read its own
// /proc/self/statm, 3 a plan/handle check failed, 4 an exception escaped the code under
// test, and 1 is valgrind's documented answer to the throw probe below (its operator new
// exits(1) where a plain one throws). 76 is the positive skip marker; the whole child body
// runs inside the try, so the forked Catch2 runner is unreachable and no code it could
// produce (42, 66, ...) is ever observed.
constexpr int kSkipExit = 76;

// Runs, before any code under test, the two probes that decide whether a hard RLIMIT_AS
// cap can inject a failure here at all. A child that returns meets the cap with a plain
// glibc allocator and a plain-sized footprint. First probe: operator new of a size that
// fails on every runtime. With plain glibc the failure is a catchable bad_alloc, which is
// exactly the channel a capped allocation failure must use; valgrind's replacement exits(1)
// inside the probe, the runtime's own deterministic statement that it cannot express that
// channel. Second probe: the process's own address-space use. Past 1 GiB an instrumenting
// runtime (SDE maps ~1.25 GiB) dominates the address space, so a cap tight enough to fail
// the test allocation starves the instrumentation's own allocator first and no subsequent
// death can be attributed to the code under test.
void rlimit_probes_or_skip() {
    // Standard signal dispositions: without them an inherited Catch2 fatal handler would
    // print a bogus failure report on the child's own death, into the parent's capture.
    std::signal(SIGABRT, SIG_DFL);
    std::signal(SIGSEGV, SIG_DFL);
    try {
        // 2^60 fails on every runtime (mmap is hard-bounded far below it, whatever the
        // overcommit policy), and stays below gcc's max-object-size diagnostic and below
        // clang's compile-time constant array size limit (2^62 elements exceeds it).
#if INTPTR_MAX > INT32_MAX
        constexpr std::size_t kHugeBytes = std::size_t{1} << 60;
#else
        constexpr std::size_t kHugeBytes = static_cast<std::size_t>(PTRDIFF_MAX) - 4096;
#endif
        char* probe = new char[kHugeBytes];
        // A volatile store through the result: the new cannot be elided or assumed
        // non-null. A plain store is dead to clang, which elides the allocation at -O2
        // and would make the skip path fire on every plain clang release build.
        volatile char* const written = probe;
        written[0] = 1;
        delete[] probe;  // never reached in practice
        _exit(kSkipExit);  // an allocator this unbounded cannot be capped
    } catch (const std::bad_alloc&) {
    }
    std::ifstream statm("/proc/self/statm");
    std::size_t pages = 0;
    statm >> pages;
    if (pages == 0) _exit(2);
    const std::size_t vm = pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    if (vm >= (std::size_t{1} << 30)) _exit(kSkipExit);
}

[[noreturn]] void capped_plan_child() {
    try {
        rlimit_probes_or_skip();
        const struct rlimit cap{1u << 30, 1u << 30};
        if (setrlimit(RLIMIT_AS, &cap) != 0) _exit(kSkipExit);  // no cap can be applied
        // The cap must refuse an allocation larger than itself; if it does not, it cannot
        // fail the planner's shape vector either.
        void* probe = std::malloc((std::size_t{1} << 30) + (std::size_t{256} << 20));
        // clang's InstCombine folds `malloc(N) != nullptr` to true at -O1 and deletes the
        // call outright (verified 18.1.8); the asm round-trip keeps which value came back
        // a runtime question.
        asm volatile("" : "+r"(probe));
        if (probe != nullptr) {
            std::free(probe);
            _exit(kSkipExit);
        }
        int n[1] = {8};
        const fftw_plan p = fftw_plan_dft(INT_MAX, n, nullptr, nullptr, FFTW_FORWARD,
                                          FFTW_ESTIMATE);
        const fftwf_plan q = fftwf_plan_dft(INT_MAX, n, nullptr, nullptr, FFTW_FORWARD,
                                            FFTW_ESTIMATE);
        _exit(p == nullptr && q == nullptr ? 0 : 3);
    } catch (...) {
        _exit(4);
    }
}

[[noreturn]] void capped_execute_child() {
    try {
        rlimit_probes_or_skip();
        constexpr int n = 1 << 17;
        auto* in = static_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
        auto* out = static_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
        fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        if (in == nullptr || out == nullptr || p == nullptr) _exit(2);
        malloc_trim(0);
        std::ifstream statm("/proc/self/statm");
        std::size_t pages = 0;
        statm >> pages;
        if (pages == 0) _exit(2);
        const rlim_t cap = pages * static_cast<rlim_t>(sysconf(_SC_PAGESIZE)) + (rlim_t{2} << 20);
        const struct rlimit lim{cap, cap};
        if (setrlimit(RLIMIT_AS, &lim) != 0) _exit(kSkipExit);
        // A 1 GiB request dwarfs the 2 MiB headroom; if even it succeeds, the cap cannot
        // fail the engine's 4 MiB scratch request either.
        void* probe = std::malloc(std::size_t{1} << 30);
        asm volatile("" : "+r"(probe));  // see the plan case: clang folds malloc != nullptr
        if (probe != nullptr) {
            std::free(probe);
            _exit(kSkipExit);
        }
        fftw_execute(p);
        fftw_execute_dft(p, in, out);
        _exit(0);
    } catch (...) {
        _exit(4);
    }
}

}  // namespace

TEST_CASE("fftw shim: planner allocation failure returns NULL, nothing escapes the C ABI",
          "[fftw]") {
    // INT_MAX ranks a 17 GiB shape vector, allocated before any engine code runs. In a
    // forked child under a 1 GiB address-space cap the allocation throws deterministically
    // on a plain Linux host, and the planner's failure channel is a NULL plan. The child
    // skips only on its positive pre-test marker (76, or valgrind's own exit 1); every
    // other fate fails: exit 3 is a non-NULL plan under the cap, exit 4 an exception
    // crossing the C ABI, any signal a runtime abort outside the noexcept execute case.
    const pid_t pid = fork();
    REQUIRE(pid != -1);
    if (pid == 0) capped_plan_child();
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    CAPTURE(status);
    if (WIFEXITED(status) && WEXITSTATUS(status) == kSkipExit)
        SKIP("RLIMIT_AS cannot fail an allocation of the code under test in this runtime");
    if (WIFEXITED(status) && WEXITSTATUS(status) == 1)
        SKIP("operator new aborts instead of throwing in this runtime (probed pre-test)");
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("fftw shim: an execute-time engine failure aborts under noexcept", "[fftw]") {
    // At 2^17 f64 the iterative dif route asks the heap for 4 MiB of scratch once per
    // execute. The child trims the allocator's free headroom, then caps its address space
    // just above live use so that request throws bad_alloc. fftw_execute is noexcept, so
    // the child must die by SIGABRT; before the noexcept fix the exception escaped the
    // C ABI and the child exited 4. 2^17 stays clear of snmalloc, whose first >= 32 MiB
    // allocation would abort the capped child for a reason of its own. The child skips only
    // on its positive pre-test marker (76, or valgrind's own exit 1); every other fate --
    // exit 0 (a swallowed engine failure), exit 4 (an escaping exception), or any signal
    // other than SIGABRT -- is a failure.
    const pid_t pid = fork();
    REQUIRE(pid != -1);
    if (pid == 0) capped_execute_child();
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    CAPTURE(status);
    if (WIFEXITED(status)) {
        if (WEXITSTATUS(status) == kSkipExit)
            SKIP("RLIMIT_AS cannot fail an allocation of the code under test in this runtime");
        if (WEXITSTATUS(status) == 1)
            SKIP("operator new aborts instead of throwing in this runtime (probed pre-test)");
    }
    REQUIRE(WIFSIGNALED(status));
    REQUIRE(WTERMSIG(status) == SIGABRT);
}
#endif
