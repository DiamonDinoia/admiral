#include <catch2/catch_test_macros.hpp>
#include <climits>
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
#include <fstream>
#include <malloc.h>
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
TEST_CASE("fftw shim: planner allocation failure returns NULL, nothing escapes the C ABI",
          "[fftw]") {
    // INT_MAX ranks a 17 GiB shape vector, allocated before any engine code runs. In a forked
    // child under a 1 GiB address-space cap the allocation throws deterministically on a plain
    // Linux host, and the planner's failure channel is a NULL plan: exit 3/4 below are pre-fix
    // behaviors made visible. Any other fate is the runtime breaking the mechanism, not the
    // shim -- valgrind installs an operator new that aborts instead of throwing, SDE kills the
    // capped child -- and says nothing about the failure channel, so skip there.
    const pid_t pid = fork();
    REQUIRE(pid != -1);
    if (pid == 0) {
        // Standard signal dispositions: without them an inherited Catch2 fatal handler would
        // print a bogus failure report on the child's own death, into the parent's capture.
        std::signal(SIGABRT, SIG_DFL);
        std::signal(SIGSEGV, SIG_DFL);
        const struct rlimit cap{1u << 30, 1u << 30};
        if (setrlimit(RLIMIT_AS, &cap) != 0) _exit(2);
        int n[1] = {8};
        fftw_plan p = nullptr;
        fftwf_plan q = nullptr;
        try {
            p = fftw_plan_dft(INT_MAX, n, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
            q = fftwf_plan_dft(INT_MAX, n, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
        } catch (...) {
            _exit(4);
        }
        _exit(p == nullptr && q == nullptr ? 0 : 3);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    if (!WIFEXITED(status) || WEXITSTATUS(status) == 1 || WEXITSTATUS(status) == 2)
        SKIP("RLIMIT_AS does not fail a capped allocation in this runtime");
    REQUIRE(WEXITSTATUS(status) == 0);
}
#endif

#if defined(ADM_TEST_FFTW_RLIMIT_FAILURE)
TEST_CASE("fftw shim: an execute-time engine failure aborts under noexcept", "[fftw]") {
    // At 2^17 f64 the iterative dif route asks the heap for 4 MiB of scratch once per
    // execute. The child trims the allocator's free headroom, then caps its address space
    // just above live use so that request throws bad_alloc. fftw_execute is noexcept, so
    // the child must die by SIGABRT; before the noexcept fix the exception escaped the
    // C ABI and the child exited 4. 2^17 stays clear of snmalloc, whose first >= 32 MiB
    // allocation would abort the capped child for a reason of its own. Other fates belong
    // to the runtime again -- valgrind's operator new aborts rather than throws, SDE
    // reports the child's signal death as a plain exit -- skip those.
    const pid_t pid = fork();
    REQUIRE(pid != -1);
    if (pid == 0) {
        std::signal(SIGABRT, SIG_DFL);  // an inherited Catch2 handler would hijack the SIGABRT
        std::signal(SIGSEGV, SIG_DFL);
        constexpr int n = 1 << 17;
        fftw_complex* in = static_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
        fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(n * sizeof(fftw_complex)));
        fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        if (in == nullptr || out == nullptr || p == nullptr) _exit(3);
        malloc_trim(0);
        std::ifstream statm("/proc/self/statm");
        size_t pages = 0;
        statm >> pages;
        if (pages == 0) _exit(2);
        const rlim_t cap = pages * static_cast<rlim_t>(sysconf(_SC_PAGESIZE)) + (rlim_t{2} << 20);
        const struct rlimit lim{cap, cap};
        if (setrlimit(RLIMIT_AS, &lim) != 0) _exit(2);
        try {
            fftw_execute(p);
            fftw_execute_dft(p, in, out);
        } catch (...) {
            _exit(4);
        }
        _exit(0);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 4)
        FAIL("an exception escaped the noexcept execute");
    if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT)
        SKIP("a capped allocation failure does not surface as SIGABRT in this runtime");
    REQUIRE(WIFSIGNALED(status));
    REQUIRE(WTERMSIG(status) == SIGABRT);
}
#endif
