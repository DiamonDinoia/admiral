#include <admiral/admiral.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <complex>
#include <cstdio>
#include <numbers>
#include <algorithm>
#include <optional>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <ducc0/fft/fft.h>
#include <nanobench.h>
#include <admiral/detail/vecpass.hpp>  // vp:: vecpass kernel for the --vpass probe
#ifdef ADM_BENCH_FFTW
#include <fftw3.h>                 // optional FFTW reference (-DFFT_BENCH_FFTW=ON)
#endif

// All benchmark helpers below have internal linkage (this is a single-TU
// executable); only main() is external.
namespace {

// Wrapper for ducc0 FFT - isolates ducc0 API details. Templated on precision so
// float and double race against the same ducc0 c2c entry point.
template<typename T>
std::vector<std::complex<T>> ducc0_forward_fft(const std::vector<std::complex<T>>& input,
                                               size_t nthreads = 1) {
    using namespace ducc0;

    const size_t N = input.size();
    std::vector<std::complex<T>> output(N);

    detail_fft::shape_t shape = {N};
    auto in_view = detail_mav::cfmav<std::complex<T>>(input.data(), shape);
    auto out_view = detail_mav::vfmav<std::complex<T>>(output.data(), shape);

    // forward=true, fct=1.0
    detail_fft::c2c(in_view, out_view, {0}, true, T(1), nthreads);

    return output;
}

template<typename T>
std::vector<std::complex<T>> ducc0_inverse_fft(const std::vector<std::complex<T>>& input,
                                               size_t nthreads = 1) {
    using namespace ducc0;

    const size_t N = input.size();
    std::vector<std::complex<T>> output(N);

    detail_fft::shape_t shape = {N};
    auto in_view = detail_mav::cfmav<std::complex<T>>(input.data(), shape);
    auto out_view = detail_mav::vfmav<std::complex<T>>(output.data(), shape);

    // forward=false, fct=1/N for proper inverse scaling
    detail_fft::c2c(in_view, out_view, {0}, false, T(1) / T(N), nthreads);

    return output;
}

// N-D ducc0 c2c over all axes (row-major, axes={0..rank-1}), mirroring the 1D
// wrappers. The tensor is contiguous with the last axis fastest — the same
// layout our N-D plan transforms. fct=1 (forward) / 1/Ntot (inverse) matches
// our scaling. Rank is a runtime property of `shape`; 2D/3D/ND all share this.
template<typename T>
std::vector<std::complex<T>> ducc0_forward_fft_nd(const std::vector<std::complex<T>>& input,
                                                  const std::vector<std::size_t>& shape,
                                                  size_t nthreads = 1) {
    using namespace ducc0;
    std::vector<std::complex<T>> output(input.size());
    detail_fft::shape_t sh(shape.begin(), shape.end());
    detail_fft::shape_t axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view = detail_mav::cfmav<std::complex<T>>(input.data(), sh);
    auto out_view = detail_mav::vfmav<std::complex<T>>(output.data(), sh);
    detail_fft::c2c(in_view, out_view, axes, true, T(1), nthreads);
    return output;
}

template<typename T>
std::vector<std::complex<T>> ducc0_inverse_fft_nd(const std::vector<std::complex<T>>& input,
                                                  const std::vector<std::size_t>& shape,
                                                  size_t nthreads = 1) {
    using namespace ducc0;
    std::vector<std::complex<T>> output(input.size());
    detail_fft::shape_t sh(shape.begin(), shape.end());
    detail_fft::shape_t axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view = detail_mav::cfmav<std::complex<T>>(input.data(), sh);
    auto out_view = detail_mav::vfmav<std::complex<T>>(output.data(), sh);
    detail_fft::c2c(in_view, out_view, axes, false, T(1) / T(input.size()), nthreads);
    return output;
}

// N-D ducc0 r2c / c2r over all axes: real tensor <-> half-spectrum complex tensor
// (innermost extent halved to n/2+1), the same layout our r2c produces. r2c is
// unscaled forward; c2r is the 1/Ntot-scaled inverse (round-trip identity).
template<typename T>
std::vector<std::complex<T>> ducc0_r2c_nd(const std::vector<T>& in,
                                          const std::vector<std::size_t>& shape,
                                          size_t nthreads = 1) {
    using namespace ducc0;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    std::size_t Nc = 1; for (auto e : cshape) Nc *= e;
    std::vector<std::complex<T>> out(Nc);
    detail_fft::shape_t rsh(shape.begin(), shape.end());
    detail_fft::shape_t csh(cshape.begin(), cshape.end());
    detail_fft::shape_t axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view  = detail_mav::cfmav<T>(in.data(), rsh);
    auto out_view = detail_mav::vfmav<std::complex<T>>(out.data(), csh);
    detail_fft::r2c(in_view, out_view, axes, /*forward=*/true, T(1), nthreads);
    return out;
}

template<typename T>
std::vector<T> ducc0_c2r_nd(const std::vector<std::complex<T>>& in,
                            const std::vector<std::size_t>& shape,
                            size_t nthreads = 1) {
    using namespace ducc0;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    std::size_t Nreal = 1; for (auto e : shape) Nreal *= e;
    std::vector<T> out(Nreal);
    detail_fft::shape_t rsh(shape.begin(), shape.end());
    detail_fft::shape_t csh(cshape.begin(), cshape.end());
    detail_fft::shape_t axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view  = detail_mav::cfmav<std::complex<T>>(in.data(), csh);
    auto out_view = detail_mav::vfmav<T>(out.data(), rsh);
    detail_fft::c2r(in_view, out_view, axes, /*forward=*/false, T(1) / T(Nreal), nthreads);
    return out;
}

#ifdef ADM_BENCH_FFTW
// ADM_BENCH_FFTW_MEASURE=1 in the env flips FFTW planning to FFTW_MEASURE —
// FFTW's fair (tuned) ceiling for 1-D sweeps; too slow for large N-D shapes,
// so the default stays ESTIMATE (a pessimistic bound that flatters us).
inline unsigned fftw_plan_flag() {
    return std::getenv("ADM_BENCH_FFTW_MEASURE") ? FFTW_MEASURE : FFTW_ESTIMATE;
}
// Reusable FFTW c2c plans for a shape and precision T. Plans are built ONCE in
// the constructor and reused across every timed rep — the fair analogue of
// ducc0's internal plan cache and our plan-reuse. See the ctor for the
// FFTW_ESTIMATE-vs-MEASURE tradeoff. fftw_complex/fftwf_complex are
// layout-compatible with std::complex<T>, so the caller's buffers are
// reinterpret_cast in place. Inverse is scaled by 1/N to match our (and ducc0's)
// round-trip convention.
template<typename T>
class fftw_c2c {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    using cpx_t  = std::conditional_t<std::is_same_v<T, double>, fftw_complex, fftwf_complex>;
    using plan_t = std::conditional_t<std::is_same_v<T, double>, fftw_plan, fftwf_plan>;
public:
    // 1D convenience: a rank-1 shape.
    explicit fftw_c2c(std::size_t N, int nthreads = 1) : fftw_c2c(std::vector<std::size_t>{N}, nthreads) {}

    // General-rank c2c: one plan built with fftw_plan_dft(rank,dims,..) covers
    // 1D/2D/3D/ND in a single path. Plans are reused across every timed rep (the
    // fair analogue of ducc0's cached plan and our plan-reuse).
    // ponytail: FFTW_ESTIMATE (not FFTW_MEASURE) "for now" — MEASURE planning is
    // very slow on the large N-D shapes (1024^2, 128^3). ESTIMATE gives FFTW a
    // heuristic (non-tuned) plan, so its timing is a PESSIMISTIC bound: real FFTW
    // with MEASURE is faster, and our fftw_* ratios here flatter us. Flip back to
    // FFTW_MEASURE for a fair ceiling once the shape set is settled.
    // nthreads > 1 threads the plan (needs the fftw3_threads libs; -DFFT_BENCH_THREADS).
    explicit fftw_c2c(const std::vector<std::size_t>& shape, [[maybe_unused]] int nthreads = 1)
        : N_([&]{ std::size_t n = 1; for (auto e : shape) n *= e; return n; }()),
          in_(N_), out_(N_) {
        std::vector<int> dims(shape.begin(), shape.end());
        const int rank = static_cast<int>(dims.size());
#ifdef ADM_BENCH_THREADS
        if (nthreads > 1) {
            if constexpr (std::is_same_v<T, double>) { static int ok = fftw_init_threads();  (void)ok; fftw_plan_with_nthreads(nthreads); }
            else                                     { static int ok = fftwf_init_threads(); (void)ok; fftwf_plan_with_nthreads(nthreads); }
        }
#endif
        if constexpr (std::is_same_v<T, double>) {
            fwd_ = fftw_plan_dft(rank, dims.data(), cpx(in_), cpx(out_), FFTW_FORWARD,  fftw_plan_flag());
            inv_ = fftw_plan_dft(rank, dims.data(), cpx(in_), cpx(out_), FFTW_BACKWARD, fftw_plan_flag());
        } else {
            fwd_ = fftwf_plan_dft(rank, dims.data(), cpx(in_), cpx(out_), FFTW_FORWARD,  fftw_plan_flag());
            inv_ = fftwf_plan_dft(rank, dims.data(), cpx(in_), cpx(out_), FFTW_BACKWARD, fftw_plan_flag());
        }
    }
    ~fftw_c2c() {
        if constexpr (std::is_same_v<T, double>) { fftw_destroy_plan(fwd_);  fftw_destroy_plan(inv_); }
        else                                     { fftwf_destroy_plan(fwd_); fftwf_destroy_plan(inv_); }
    }
    fftw_c2c(const fftw_c2c&) = delete;
    fftw_c2c& operator=(const fftw_c2c&) = delete;

    // Forward c2c (unnormalized, matches our forward): in_ <- x, run, return out_.
    const std::vector<std::complex<T>>& forward(const std::vector<std::complex<T>>& x) {
        std::copy(x.begin(), x.end(), in_.begin());
        exec(fwd_);
        return out_;
    }
    // Forward then inverse with 1/N scaling -> out_ ~= x.
    const std::vector<std::complex<T>>& roundtrip(const std::vector<std::complex<T>>& x) {
        std::copy(x.begin(), x.end(), in_.begin());
        exec(fwd_);
        std::copy(out_.begin(), out_.end(), in_.begin());
        exec(inv_);
        const T s = T(1) / static_cast<T>(N_);
        for (auto& v : out_) v *= s;
        return out_;
    }
private:
    static cpx_t* cpx(std::vector<std::complex<T>>& v) { return reinterpret_cast<cpx_t*>(v.data()); }
    static void exec(plan_t p) {
        if constexpr (std::is_same_v<T, double>) fftw_execute(p);
        else                                     fftwf_execute(p);
    }
    std::size_t N_;
    std::vector<std::complex<T>> in_, out_;
    plan_t fwd_{}, inv_{};
};

// Reusable FFTW r2c/c2r plans for a real N-D shape (same ESTIMATE-vs-MEASURE
// tradeoff as fftw_c2c). The complex half-spectrum has the innermost extent
// halved to n/2+1. c2r consumes its complex input, so round-trip stages through
// a private buffer; the 1/Ntot scale matches our (and ducc0's) convention.
template<typename T>
class fftw_r2c {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    using cpx_t  = std::conditional_t<std::is_same_v<T, double>, fftw_complex, fftwf_complex>;
    using plan_t = std::conditional_t<std::is_same_v<T, double>, fftw_plan, fftwf_plan>;
public:
    explicit fftw_r2c(const std::vector<std::size_t>& shape, [[maybe_unused]] int nthreads = 1) {
        Nreal_ = 1; for (auto e : shape) Nreal_ *= e;
        std::vector<std::size_t> cshape(shape);
        cshape.back() = shape.back() / 2 + 1;
        Nc_ = 1; for (auto e : cshape) Nc_ *= e;
        rin_.resize(Nreal_); cout_.resize(Nc_); cin_.resize(Nc_); rout_.resize(Nreal_);
        std::vector<int> dims(shape.begin(), shape.end());
        const int rank = static_cast<int>(dims.size());
#ifdef ADM_BENCH_THREADS
        if (nthreads > 1) {
            if constexpr (std::is_same_v<T, double>) { static int ok = fftw_init_threads();  (void)ok; fftw_plan_with_nthreads(nthreads); }
            else                                     { static int ok = fftwf_init_threads(); (void)ok; fftwf_plan_with_nthreads(nthreads); }
        }
#endif
        if constexpr (std::is_same_v<T, double>) {
            fwd_ = fftw_plan_dft_r2c(rank, dims.data(), rin_.data(),  cpx(cout_), fftw_plan_flag());
            inv_ = fftw_plan_dft_c2r(rank, dims.data(), cpx(cin_),    rout_.data(), fftw_plan_flag());
        } else {
            fwd_ = fftwf_plan_dft_r2c(rank, dims.data(), rin_.data(), cpx(cout_), fftw_plan_flag());
            inv_ = fftwf_plan_dft_c2r(rank, dims.data(), cpx(cin_),   rout_.data(), fftw_plan_flag());
        }
    }
    ~fftw_r2c() {
        if constexpr (std::is_same_v<T, double>) { fftw_destroy_plan(fwd_);  fftw_destroy_plan(inv_); }
        else                                     { fftwf_destroy_plan(fwd_); fftwf_destroy_plan(inv_); }
    }
    fftw_r2c(const fftw_r2c&) = delete;
    fftw_r2c& operator=(const fftw_r2c&) = delete;

    const std::vector<std::complex<T>>& forward(const std::vector<T>& x) {
        std::copy(x.begin(), x.end(), rin_.begin());
        exec_r2c();
        return cout_;
    }
    const std::vector<T>& roundtrip(const std::vector<T>& x) {
        std::copy(x.begin(), x.end(), rin_.begin());
        exec_r2c();
        std::copy(cout_.begin(), cout_.end(), cin_.begin());
        exec_c2r();
        const T s = T(1) / static_cast<T>(Nreal_);
        for (auto& v : rout_) v *= s;
        return rout_;
    }
private:
    static cpx_t* cpx(std::vector<std::complex<T>>& v) { return reinterpret_cast<cpx_t*>(v.data()); }
    void exec_r2c() { if constexpr (std::is_same_v<T, double>) fftw_execute(fwd_); else fftwf_execute(fwd_); }
    void exec_c2r() { if constexpr (std::is_same_v<T, double>) fftw_execute(inv_); else fftwf_execute(inv_); }
    std::size_t Nreal_, Nc_;
    std::vector<T> rin_, rout_;
    std::vector<std::complex<T>> cout_, cin_;
    plan_t fwd_{}, inv_{};
};
#endif  // ADM_BENCH_FFTW

// ----------------------------------------------------------------------------
// Accuracy gate (§E). The benchmark only ever TIMED transforms; it never
// checked that the transform was actually computed. A plan dispatched to an
// unmatched radix (the retracted radix-16/32/64 "wins") ran a no-op and timed
// as physically-impossible-fast garbage. Every timed plan now first has to
// reproduce a reference DFT to within tolerance, or it is rejected — never
// timed, never reported as a win.
// ----------------------------------------------------------------------------

// Naive O(N^2) reference DFT, evaluated in double precision regardless of T so
// it is the trusted ground truth for both f32 and f64 plans. Forward,
// unnormalized: X[k] = sum_n x[n] e^{-2*pi*i*k*n/N}, matching our forward
// convention. The angle uses the exact integer turn fraction ((k*n) mod N)/N so
// the reference stays accurate even at the largest swept sizes.
template<typename T>
std::vector<std::complex<double>>
reference_forward_dft(const std::vector<std::complex<T>>& in) {
    const std::size_t N = in.size();
    std::vector<std::complex<double>> out(N);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (std::size_t k = 0; k < N; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (std::size_t n = 0; n < N; ++n) {
            const double ang = -two_pi * static_cast<double>((k * n) % N) / static_cast<double>(N);
            acc += std::complex<double>(static_cast<double>(in[n].real()),
                                        static_cast<double>(in[n].imag()))
                 * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        out[k] = acc;
    }
    return out;
}

// Relative L2 (Frobenius) error of a computed transform vs the reference:
// ||got - ref||_2 / ||ref||_2. A correctly computed mixed-radix transform sits
// near machine eps; a transform that didn't run has O(1) error.
template<typename T>
double l2_rel_error(const std::vector<std::complex<T>>& got,
                    const std::vector<std::complex<double>>& ref) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double dr = static_cast<double>(got[i].real()) - ref[i].real();
        const double di = static_cast<double>(got[i].imag()) - ref[i].imag();
        num += dr * dr + di * di;
        den += ref[i].real() * ref[i].real() + ref[i].imag() * ref[i].imag();
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Default acceptance tolerance per precision. Generous enough that any correct
// transform in the sweep passes (error grows only ~sqrt(log N) above eps),
// tight enough that an un-run / wrong transform (O(1) error) is always caught.
template<typename T>
constexpr double default_accuracy_tol() {
    return std::is_same_v<T, float> ? 1e-3 : 1e-9;
}

// Performance data structure
struct BenchmarkResult {
    size_t size;
    std::string type;
    std::string prec;   // "f32" or "f64"
    double fft_fwd_ms;
    double fft_rt_ms;
    double ducc0_fwd_ms;
    double ducc0_rt_ms;

    double fwd_ratio() const { return fft_fwd_ms / ducc0_fwd_ms; }
    double rt_ratio() const { return fft_rt_ms / ducc0_rt_ms; }

    bool is_power_of_2() const {
        return std::has_single_bit(size);
    }

    bool is_prime() const {
        if (size <= 1) return false;
        if (size <= 3) return true;
        if (size % 2 == 0 || size % 3 == 0) return false;
        for (size_t i = 5; i * i <= size; i += 6) {
            if (size % i == 0 || size % (i + 2) == 0) return false;
        }
        return true;
    }

    std::string category() const {
        if (is_power_of_2()) return "Power-of-2";
        if (is_prime()) return "Prime";
        return "Composite";
    }
};

std::vector<BenchmarkResult> all_results;

// Timing utility backed by nanobench: returns the MEDIAN wall-clock time per
// call in milliseconds. nanobench auto-tunes epoch iteration counts, warms up,
// runs multiple epochs and reports the median (robust to outliers / scheduler
// noise) instead of a single mean over a fixed loop.
template<typename Func>
double time_execution(const char* name, Func&& func) {
    ankerl::nanobench::Bench b;
    b.output(nullptr);   // suppress nanobench's own table; we print our report
    b.warmup(10);
    b.run(name, std::forward<Func>(func));
    // Measure::elapsed is seconds per op; convert to milliseconds.
    return b.results().back().median(ankerl::nanobench::Result::Measure::elapsed) * 1e3;
}

// Per-call timing backed by nanobench for the --compare gate. Returns the BEST
// (least-contended) epoch in microseconds, plus nanobench's Median Absolute
// Percent Error (MdAPE) over the epochs as a
// stability flag. Self-stabilizing instead of requiring a quiet machine: start
// at a 1ms epoch floor and, while the reading is noisy (err > target), DOUBLE
// the per-epoch time so more calls are averaged per epoch, retrying up to a cap.
// This trades a little wall-clock on contended sizes for a trustworthy ratio
// without waiting for a quiet window. Epochs map to `reps`; `min_iters>0` forces
// an exact minEpochIterations and disables adaptation.
// us = best-epoch wall-clock (for human-readable display only). cyc = best-epoch
// per-process CPU cycles (0 if perf counters unavailable). err = MdAPE of the
// metric actually used for the ratio. CPU cycles are the trustworthy metric here:
// a per-process perf counter advances only while THIS process runs at whatever
// clock, so it is invariant to both the powersave governor's floating frequency
// and to other processes stealing the core. Wall-clock conflates work with clock
// and is only a fallback when counters are inaccessible.
struct NbStat { double us; double cyc; double err; };
template<typename Func>
NbStat nb_measure(const char* name, int reps, long min_iters, Func&& func) {
    using ankerl::nanobench::Result;
    constexpr double kTarget = 0.05;   // 5% MdAPE -> considered stable
    constexpr int kMaxMs = 64;         // cap per-epoch growth (1,2,...,64ms)
    NbStat stat{0.0, 0.0, 1e9};
    for (int ms = 1; ms <= kMaxMs; ms *= 2) {
        ankerl::nanobench::Bench b;
        b.output(nullptr);
        b.warmup(10);
        if (reps > 0) b.epochs(static_cast<std::size_t>(reps));
        if (min_iters > 0) b.minEpochIterations(static_cast<uint64_t>(min_iters));
        else b.minEpochTime(std::chrono::milliseconds(ms));
        b.run(name, func);   // re-invoked per retry; don't move from func
        const auto& r = b.results().back();
        const double cyc = r.minimum(Result::Measure::cpucycles);
        // -ffinite-math-only makes std::isfinite a no-op (and clang -Werror flags
        // it), so bound with the max() sentinel instead of an inf/nan check.
        const bool have_cyc = cyc > 0.0 && cyc < std::numeric_limits<double>::max();
        // Stabilize on (and later compare with) cycles when available, else elapsed.
        const double err = have_cyc ? r.medianAbsolutePercentError(Result::Measure::cpucycles)
                                    : r.medianAbsolutePercentError(Result::Measure::elapsed);
        stat = {r.minimum(Result::Measure::elapsed) * 1e6, have_cyc ? cyc : 0.0, err};
        if (min_iters > 0 || stat.err <= kTarget) break;
    }
    return stat;
}

std::vector<std::size_t> parse_size_list(const std::string& s) {
    std::vector<std::size_t> sizes;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        if (comma != pos) {
            sizes.push_back(std::stoul(s.substr(pos, comma - pos)));
        }
        pos = comma + 1;
    }
    return sizes;
}

// Parse "RxC,RxCxD,..." into a list of N-D shapes (each a vector of extents) for
// the N-D compare mode. Any rank; "16x256" is 2D, "64x64x64" is 3D, etc.
std::vector<std::vector<std::size_t>> parse_nd_shape_list(const std::string& s) {
    std::vector<std::vector<std::size_t>> shapes;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        const std::string tok = s.substr(pos, comma - pos);
        if (!tok.empty()) {
            std::vector<std::size_t> shape;
            size_t p = 0;
            while (p < tok.size()) {
                size_t x = tok.find('x', p);
                if (x == std::string::npos) x = tok.size();
                shape.push_back(std::stoul(tok.substr(p, x - p)));
                p = x + 1;
            }
            shapes.push_back(std::move(shape));
        }
        pos = comma + 1;
    }
    return shapes;
}

std::vector<unsigned> parse_radix_list(const std::string& s) {
    std::vector<unsigned> radices;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t sep = s.find_first_of(",-", pos);
        if (sep == std::string::npos) sep = s.size();
        if (sep != pos) {
            radices.push_back(static_cast<unsigned>(std::stoul(s.substr(pos, sep - pos))));
        }
        pos = sep + 1;
    }
    return radices;
}

admiral::detail::dif_factor_plan make_dif_factor_plan(const std::vector<unsigned>& radices,
                                                  unsigned base_n = 0) {
    admiral::detail::dif_factor_plan plan;
    for (unsigned radix : radices) {
        plan.push(radix);
    }
    plan.base_n = base_n;
    return plan;
}

std::string join_radices(const std::vector<unsigned>& radices) {
    std::ostringstream os;
    for (std::size_t i = 0; i < radices.size(); ++i) {
        if (i != 0) os << '-';
        os << radices[i];
    }
    return os.str();
}

void enumerate_dif_radix_sequences_impl(std::size_t remaining,
                                        std::vector<unsigned>& current,
                                        std::vector<std::vector<unsigned>>& out) {
    // 16/32 only where the engine dispatch admits them (32-reg ISAs).
    static constexpr unsigned kRadices[] = {2, 3, 4, 5, 7, 8, 11, 16, 32};
    static constexpr std::size_t kNumRadices =
        admiral::detail::dif_wide_radices ? 9u : 7u;
    if (remaining == 1) {
        out.push_back(current);
        return;
    }
    for (std::size_t ri = 0; ri < kNumRadices; ++ri) {
        const unsigned radix = kRadices[ri];
        if (remaining % radix != 0) continue;
        current.push_back(radix);
        enumerate_dif_radix_sequences_impl(remaining / radix, current, out);
        current.pop_back();
    }
}

std::vector<std::vector<unsigned>> enumerate_dif_radix_sequences(std::size_t n) {
    std::vector<std::vector<unsigned>> sequences;
    std::vector<unsigned> current;
    if (n > 1) enumerate_dif_radix_sequences_impl(n, current, sequences);
    return sequences;
}

template<typename T>
admiral::detail::plan_impl<T> make_factor_sweep_plan(std::size_t n,
                                                 bool is_forward,
                                                 const std::vector<unsigned>& radices) {
    const auto plan = make_dif_factor_plan(radices);
    return admiral::detail::plan_impl<T>(n, is_forward, &plan);
}

template<typename T>
void factor_sweep_size(std::size_t N, const std::vector<unsigned>& radices, int reps) {
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    auto fwd_plan = make_factor_sweep_plan<T>(N, true, radices);
    auto inv_plan = make_factor_sweep_plan<T>(N, false, radices);
    std::vector<std::complex<T>> buf(N);

    // Accuracy gate (§E/§D): a decomposition that doesn't reproduce the reference
    // DFT is reported as FAIL and never timed, so the auto-tuner can only ever
    // pick the empirically fastest *correct* plan.
    std::copy(data.begin(), data.end(), buf.begin());
    fwd_plan.execute(std::span(buf));
    const double l2err = l2_rel_error<T>(buf, reference_forward_dft<T>(data));
    if (!(l2err <= default_accuracy_tol<T>())) {
        std::cout << N << ',' << ((sizeof(T) == 4) ? "f32" : "f64") << ','
                  << join_radices(radices)
                  << ",0,0,0,0,0,0,0,"
                  << std::scientific << std::setprecision(6) << l2err << std::defaultfloat
                  << ",FAIL\n";
        return;
    }

    volatile T sink = T(0);
    const NbStat fft_fwd = nb_measure("factor_fft_fwd", reps, 0, [&]() {
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_plan.execute(std::span(buf));
        sink += buf[N / 2].real();
    });
    const NbStat fft_rt = nb_measure("factor_fft_rt", reps, 0, [&]() {
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_plan.execute(std::span(buf));
        inv_plan.execute(std::span(buf));
        sink += buf[N / 2].real();
    });
    const NbStat ducc_fwd = nb_measure("factor_ducc_fwd", reps, 0, [&]() {
        auto out = ducc0_forward_fft<T>(data);
        sink += out[N / 2].real();
    });
    const NbStat ducc_rt = nb_measure("factor_ducc_rt", reps, 0, [&]() {
        auto fwd = ducc0_forward_fft<T>(data);
        auto inv = ducc0_inverse_fft<T>(fwd);
        sink += inv[N / 2].real();
    });
    (void)sink;

    // CPU cycles when available (frequency- & drift-invariant — wall-clock
    // factor ranking was measured NOT to transfer to cyc, the metric that
    // governs the win/lose call). Falls back to wall only if counters are off.
    const bool use_cyc = fft_fwd.cyc > 0.0 && ducc_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && ducc_rt.cyc > 0.0;
    const double fwd_ratio = use_cyc ? fft_fwd.cyc / ducc_fwd.cyc : fft_fwd.us / ducc_fwd.us;
    const double rt_ratio = use_cyc ? fft_rt.cyc / ducc_rt.cyc : fft_rt.us / ducc_rt.us;
    const double max_err =
        std::max(std::max(fft_fwd.err, fft_rt.err), std::max(ducc_fwd.err, ducc_rt.err));

    std::cout << N << ','
              << ((sizeof(T) == 4) ? "f32" : "f64") << ','
              << join_radices(radices) << ','
              << std::fixed << std::setprecision(6)
              << fft_fwd.us << ','
              << fft_rt.us << ','
              << ducc_fwd.us << ','
              << ducc_rt.us << ','
              << fwd_ratio << ','
              << rt_ratio << ','
              << max_err << ','
              << std::scientific << std::setprecision(6) << l2err << std::defaultfloat
              << ",OK"
              << '\n';
}

template<typename T>
void factor_sweep_precision(const std::vector<std::size_t>& sizes, int reps) {
    for (std::size_t N : sizes) {
        const auto sequences = enumerate_dif_radix_sequences(N);
        if (sequences.empty()) {
            std::cerr << "factor-sweep: size " << N
                      << " has no factorization using candidate DIF radices\n";
            continue;
        }
        for (const auto& radices : sequences) {
            factor_sweep_size<T>(N, radices, reps);
        }
    }
}

// Single-pass microbench (Phase 1): time dif_pass / dif_pass_last directly to
// perf-ground the per-pass cost surface c_pass(radix, ido, l1, prec). The DIF
// pass cost is NON-MONOTONE in ido: vectorized (cheap) at ido>=W, scalar (the
// "valley") at 1<ido<W, vectorized again at ido==1 via the lane-over-b last pass.
// --perf-iters=K runs a bare K-call loop (no nanobench) so external `perf stat`
// reads almost-pure kernel PMU; otherwise reports nanobench cyc + cyc/element.
template<typename T>
void pass_microbench(unsigned IP, std::size_t ido, std::size_t l1, bool last,
                     int reps, long inner, long perf_iters) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t span = static_cast<std::size_t>(IP) * ido * l1;  // complex elems
    std::vector<T> ccre(span), ccim(span), chre(span), chim(span);
    std::vector<std::complex<T>> out(span);
    const std::size_t tsz = std::max<std::size_t>(1, (IP - 1) * ido);
    std::vector<T> twre(tsz), twim(tsz);
    for (std::size_t i = 0; i < span; ++i) {
        ccre[i] = std::sin(T(i) * T(0.1)); ccim[i] = std::cos(T(i) * T(0.1));
    }
    for (std::size_t i = 0; i < tsz; ++i) {
        twre[i] = std::cos(T(i) * T(0.017)); twim[i] = std::sin(T(i) * T(0.017));
    }
    volatile T sink = T(0);
    auto mid = [&]<int IPv>() {
        admiral::detail::dif_pass<T, true, IPv>(ccre.data(), ccim.data(), chre.data(),
                                            chim.data(), l1, ido, twre.data(), twim.data());
    };
    auto lst = [&]<int IPv>() {
        admiral::detail::dif_pass_last<T, true, IPv>(ccre.data(), ccim.data(), out.data(),
                                                 l1, 1, twre.data(), twim.data());
    };
    auto call = [&]() {
        switch (IP) {
#define PASS_CASE(N) case N: if (last) lst.template operator()<N>(); else mid.template operator()<N>(); break
            PASS_CASE(2); PASS_CASE(3); PASS_CASE(4); PASS_CASE(5);
            PASS_CASE(7); PASS_CASE(8); PASS_CASE(11);
            PASS_CASE(16); PASS_CASE(32);
            PASS_CASE(9); PASS_CASE(15);  // probe-only: composite-odd merged radices (WS7 P5)
#undef PASS_CASE
            default: std::cerr << "--pass: unsupported radix " << IP << "\n"; return;
        }
        sink += last ? out[span / 2].real() : chre[span / 2];
    };
    if (perf_iters > 0) {
        for (long i = 0; i < 200; ++i) call();               // warm
        for (long i = 0; i < perf_iters; ++i) call();        // measured by external perf
        (void)sink;
        return;
    }
    const NbStat st = nb_measure("pass", reps, inner, call);
    const char* regime = last ? "last(lane-b)"
                       : (ido == 1) ? "ido1(scalar)"
                       : (ido >= W) ? "vec(ido>=W)"
                                    : "valley(1<ido<W)";
    std::printf("PASS prec=%s IP=%2u ido=%5zu l1=%6zu span=%8zu  %-13s  cyc=%10.1f cyc/elem=%7.3f us=%8.3f\n",
                (sizeof(T) == 4 ? "f32" : "f64"), IP, ido, l1, span, regime,
                st.cyc, st.cyc / static_cast<double>(span), st.us);
    (void)sink;
}

template<typename T>
void benchmark_size(size_t N, const std::string& type) {
    // Create test data
    std::vector<std::complex<T>> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
    }

    // Build the plans ONCE (twiddles generated here, during planning). This is
    // the fair comparison vs ducc0, which caches its c2c plans internally; the
    // timed loop measures execute() only, with zero trig per call.
    admiral::plan<T> fwd_plan(N);
    admiral::plan<T> inv_plan(N);

    // Benchmark fft forward (copy-in to match ducc0's out-of-place call, then
    // in-place execute on the plan-owned dispatch).
    std::vector<std::complex<T>> fft_output(N);
    double fft_fwd_time = time_execution("fft_fwd", [&]() {
        std::copy(data.begin(), data.end(), fft_output.begin());
        fwd_plan.forward(std::span(fft_output));
        ankerl::nanobench::doNotOptimizeAway(fft_output.data());
    });

    // Benchmark fft round-trip (forward then inverse, in place).
    std::vector<std::complex<T>> fft_temp1(N);
    double fft_rt_time = time_execution("fft_rt", [&]() {
        std::copy(data.begin(), data.end(), fft_temp1.begin());
        fwd_plan.forward(std::span(fft_temp1));
        inv_plan.inverse(std::span(fft_temp1));
        ankerl::nanobench::doNotOptimizeAway(fft_temp1.data());
    });

    // Benchmark ducc0 forward FFT (via wrapper)
    double ducc0_fwd_time = time_execution("ducc0_fwd", [&]() {
        auto result = ducc0_forward_fft<T>(data);
        ankerl::nanobench::doNotOptimizeAway(result.data());
    });

    // Benchmark ducc0 round-trip (via wrapper)
    double ducc0_rt_time = time_execution("ducc0_rt", [&]() {
        auto fwd = ducc0_forward_fft<T>(data);
        auto inv = ducc0_inverse_fft<T>(fwd);
        ankerl::nanobench::doNotOptimizeAway(inv.data());
    });

    // Store result
    BenchmarkResult result;
    result.size = N;
    result.type = type;
    result.prec = (sizeof(T) == 4) ? "f32" : "f64";
    result.fft_fwd_ms = fft_fwd_time;
    result.fft_rt_ms = fft_rt_time;
    result.ducc0_fwd_ms = ducc0_fwd_time;
    result.ducc0_rt_ms = ducc0_rt_time;
    all_results.push_back(result);

    // Print per-size results
    std::cout << std::setw(6) << N << " "
              << std::setw(5) << result.prec << " "
              << std::setw(12) << type << " | "
              << std::setw(11) << std::fixed << std::setprecision(4) << fft_fwd_time << " ms | "
              << std::setw(11) << fft_rt_time << " ms | "
              << std::setw(11) << ducc0_fwd_time << " ms | "
              << std::setw(11) << ducc0_rt_time << " ms | "
              << std::setw(9) << std::setprecision(2) << result.fwd_ratio() << "x | "
              << std::setw(9) << result.rt_ratio() << "x\n";
}

// Statistical summary for a category
struct CategoryStats {
    std::string name;
    size_t count;
    double avg_fwd_ratio;
    double avg_rt_ratio;
    double min_fwd_ratio;
    double max_fwd_ratio;
    double min_rt_ratio;
    double max_rt_ratio;
};

CategoryStats compute_category_stats(const std::string& category, const std::string& prec) {
    CategoryStats stats;
    stats.name = category;
    stats.count = 0;

    std::vector<double> fwd_ratios, rt_ratios;

    for (const auto& result : all_results) {
        if (result.category() == category && result.prec == prec) {
            fwd_ratios.push_back(result.fwd_ratio());
            rt_ratios.push_back(result.rt_ratio());
            stats.count++;
        }
    }

    if (stats.count == 0) return stats;

    // Compute statistics
    double sum_fwd = 0.0, sum_rt = 0.0;
    for (double r : fwd_ratios) sum_fwd += r;
    for (double r : rt_ratios) sum_rt += r;

    stats.avg_fwd_ratio = sum_fwd / static_cast<double>(fwd_ratios.size());
    stats.avg_rt_ratio = sum_rt / static_cast<double>(rt_ratios.size());

    stats.min_fwd_ratio = *std::min_element(fwd_ratios.begin(), fwd_ratios.end());
    stats.max_fwd_ratio = *std::max_element(fwd_ratios.begin(), fwd_ratios.end());
    stats.min_rt_ratio = *std::min_element(rt_ratios.begin(), rt_ratios.end());
    stats.max_rt_ratio = *std::max_element(rt_ratios.begin(), rt_ratios.end());

    return stats;
}

void print_performance_report() {
    std::cout << "\n\n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "                         PERFORMANCE REPORT                            \n";
    std::cout << "═══════════════════════════════════════════════════════════════════════\n\n";

    // Per-precision breakdown: float and double are reported separately so a
    // regression in one precision is never masked by the other's average.
    for (const char* prec : {"f64", "f32"}) {
        std::vector<const BenchmarkResult*> rs;
        for (const auto& r : all_results) if (r.prec == prec) rs.push_back(&r);
        if (rs.empty()) continue;

        const char* prec_name = (std::string(prec) == "f64") ? "double" : "float";

        double total_fwd_ratio = 0.0, total_rt_ratio = 0.0;
        for (const auto* r : rs) {
            total_fwd_ratio += r->fwd_ratio();
            total_rt_ratio += r->rt_ratio();
        }

        std::cout << "Overall Summary [" << prec_name << "]:\n";
        std::cout << "  Total benchmarks: " << rs.size() << "\n";
        std::cout << "  Average forward ratio: " << std::fixed << std::setprecision(2)
                  << (total_fwd_ratio / static_cast<double>(rs.size())) << "x\n";
        std::cout << "  Average forward+inverse ratio: "
                  << (total_rt_ratio / static_cast<double>(rs.size())) << "x\n";
        std::cout << "\n";

        // Category breakdown
        std::cout << "Performance by Category [" << prec_name << "]:\n";
        std::cout << "───────────────────────────────────────────────────────────────────────\n";
        std::cout << std::setw(15) << "Category" << " | "
                  << std::setw(6) << "Count" << " | "
                  << "Forward Ratio (fft/ducc0) | "
                  << "Fwd+Inv Ratio (fft/ducc0)\n";
        std::cout << std::setw(15) << "" << " | "
                  << std::setw(6) << "" << " | "
                  << "Avg    Min    Max           | "
                  << "Avg    Min    Max\n";
        std::cout << "───────────────────────────────────────────────────────────────────────\n";

        for (const auto& cat : {"Power-of-2", "Prime", "Composite"}) {
            auto stats = compute_category_stats(cat, prec);
            if (stats.count == 0) continue;

            std::cout << std::setw(15) << stats.name << " | "
                      << std::setw(6) << stats.count << " | "
                      << std::setw(5) << std::setprecision(2) << stats.avg_fwd_ratio << "x "
                      << std::setw(5) << stats.min_fwd_ratio << "x "
                      << std::setw(5) << stats.max_fwd_ratio << "x       | "
                      << std::setw(5) << stats.avg_rt_ratio << "x "
                      << std::setw(5) << stats.min_rt_ratio << "x "
                      << std::setw(5) << stats.max_rt_ratio << "x\n";
        }
        std::cout << "\n";

        // Biggest ducc0 advantages for this precision (ratio > 1.0 means ducc0 faster)
        std::vector<BenchmarkResult> ducc0_wins;
        for (const auto* r : rs) if (r->fwd_ratio() > 1.0) ducc0_wins.push_back(*r);
        if (!ducc0_wins.empty()) {
            std::sort(ducc0_wins.begin(), ducc0_wins.end(),
                      [](const auto& a, const auto& b) { return a.fwd_ratio() > b.fwd_ratio(); });
            std::cout << "ducc0 faster [" << prec_name << "] (forward FFT):\n";
            for (size_t i = 0; i < std::min(size_t(8), ducc0_wins.size()); ++i) {
                const auto& r = ducc0_wins[i];
                std::cout << "  Size " << std::setw(6) << r.size
                          << " (" << std::setw(12) << r.type << "): "
                          << std::setprecision(2) << r.fwd_ratio() << "x\n";
            }
            std::cout << "\n";
        } else {
            std::cout << "ducc0 faster [" << prec_name << "]: NONE — fft wins every size.\n\n";
        }
    }

    std::cout << "═══════════════════════════════════════════════════════════════════════\n";
    std::cout << "\nInterpretation:\n";
    std::cout << "  - Ratio = fft time / ducc0 time\n";
    std::cout << "  - Ratio > 1.0: ducc0 is faster (fft takes more time)\n";
    std::cout << "  - Ratio < 1.0: fft is faster\n";
    std::cout << "  - Power-of-2: ducc0 excels with radix-2 optimizations\n";
    std::cout << "  - Prime: More level playing field, both use similar methods\n";
    std::cout << "  - Composite: Performance depends on factorization\n";
}

// Single-size profiling mode: run ONLY our forward FFT execute() in a tight
// loop so a profiler attributes ~all cycles to the library, not ducc0 or the
// report machinery. Enabled by --size=N (optionally --iters=M).
template<typename T>
int profile_single_size(std::size_t N, long iters) {
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
    }
    admiral::plan<T> fwd_plan(N);
    std::vector<std::complex<T>> buf(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long it = 0; it < iters; ++it) {
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_plan.forward(std::span(buf));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Touch the result so the loop is not optimized away.
    volatile double sink = static_cast<double>(buf[N / 2].real() + buf[N / 2].imag());
    (void)sink;
    std::cout << "size=" << N << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " iters=" << iters
              << " total_ms=" << std::fixed << std::setprecision(3) << total_ms
              << " per_call_us=" << std::setprecision(4) << (total_ms * 1e3 / double(iters))
              << "\n";
    return 0;
}

// Theoretical peak flops/cycle for type T at this build's SIMD width: two FMA
// units, each xsimd::batch<T>::size lanes wide, 2 flops per FMA (mul + add).
// AVX2: 16 (f64, 4 lanes) / 32 (f32, 8 lanes). Frequency-invariant, so a %peak
// derived from flops/cycle is consistent with the cycle-true ratio.
template<typename T>
consteval double peak_flops_per_cycle() {
    return 2.0 * static_cast<double>(xsimd::batch<T>::size) * 2.0;
}

// FFTW/ducc convention: a complex FFT of size N costs ~5*N*log2(N) flops. Used
// only for throughput reporting (GFLOPS, %peak), not for the win/lose ratio.
inline double fft_flops(std::size_t N) {
    return 5.0 * static_cast<double>(N) * std::log2(static_cast<double>(N));
}

// Paired nanobench compare mode. For each size we build the plans ONCE (warm,
// plan-reuse — fair vs ducc0's cached c2c), then time both our execute() and
// ducc0's c2c. Forward and round-trip, per precision.
// ratio = fft_time / ducc0_time; <1.0 means this FFT wins.
template<typename T>
bool compare_min_of_n(std::size_t N, int reps, long inner,
                      const std::vector<unsigned>* factor_override = nullptr,
                      double tol = default_accuracy_tol<T>(), int nthreads = 1) {
    // ponytail: our 1D transform stays serial (single-transform MT is a measured
    // NO-GO: large-N iterative_dif is DRAM-bandwidth-bound — 32 MiB f32 runs 47.5%
    // backend-bound at 73% LLC-miss, so extra cores add no bandwidth; smaller N is
    // cache-resident where per-pass fork/join barriers dwarf the µs transform).
    // nthreads only threads the ducc0/FFTW references here, and ducc0 itself
    // disables threading for a single 1-D transform, so this mainly exposes FFTW's
    // 1-D threading — a 1D --nthreads>1 row is not an apples-to-apples win/lose.
    const std::size_t nt = static_cast<std::size_t>(nthreads);
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    const std::optional<admiral::detail::dif_factor_plan> override_plan =
        factor_override ? std::optional<admiral::detail::dif_factor_plan>(make_dif_factor_plan(*factor_override))
                        : std::nullopt;
    admiral::detail::plan_impl<T> fwd_plan(N, true, override_plan ? &*override_plan : nullptr);
    admiral::detail::plan_impl<T> inv_plan(N, false, override_plan ? &*override_plan : nullptr);
    std::vector<std::complex<T>> buf(N);

    // Accuracy gate FIRST: a plan that didn't actually compute the transform is
    // rejected here and never timed, so its bogus (fast) ratio can never be
    // reported as a win. This is the check that would have caught the retracted
    // radix-16/32/64 "wins".
    std::copy(data.begin(), data.end(), buf.begin());
    fwd_plan.execute(std::span(buf));
    // ponytail: naive O(N^2) reference takes hours above 64K; large sizes gate by
    // round-trip error instead, and FFTW below cross-checks against our gated forward.
    constexpr std::size_t kNaiveRefMaxN = 65536;
    double l2err;
    if (N <= kNaiveRefMaxN) {
        l2err = l2_rel_error<T>(buf, reference_forward_dft<T>(data));
    } else {
        std::vector<std::complex<T>> rt(buf.begin(), buf.end());
        inv_plan.execute(std::span(rt));
        std::vector<std::complex<double>> ref_d(N);
        for (std::size_t i = 0; i < N; ++i)
            ref_d[i] = std::complex<double>(static_cast<double>(data[i].real()),
                                            static_cast<double>(data[i].imag()));
        l2err = l2_rel_error<T>(rt, ref_d);
    }
    if (!(l2err <= tol)) {
        std::cout << "CMP size=" << std::setw(5) << N
                  << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
                  << " l2err=" << std::scientific << std::setprecision(2) << l2err
                  << " tol=" << tol << std::defaultfloat
                  << "  <== FAIL (inaccurate, not timed)\n";
        return false;
    }
    std::vector<std::complex<double>> gated_fwd(N);
    for (std::size_t i = 0; i < N; ++i)
        gated_fwd[i] = std::complex<double>(static_cast<double>(buf[i].real()),
                                            static_cast<double>(buf[i].imag()));

    volatile T sink = T(0);
    const NbStat fft_fwd = nb_measure("fft_fwd", reps, inner, [&]() {
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_plan.execute(std::span(buf));
        sink += buf[N / 2].real();
    });
    const NbStat fft_rt = nb_measure("fft_rt", reps, inner, [&]() {
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_plan.execute(std::span(buf));
        inv_plan.execute(std::span(buf));
        sink += buf[N / 2].real();
    });
    const NbStat ducc_fwd = nb_measure("ducc_fwd", reps, inner, [&]() {
        auto out = ducc0_forward_fft<T>(data, nt);
        sink += out[N / 2].real();
    });
    const NbStat ducc_rt = nb_measure("ducc_rt", reps, inner, [&]() {
        auto fwd = ducc0_forward_fft<T>(data, nt);
        auto inv = ducc0_inverse_fft<T>(fwd, nt);
        sink += inv[N / 2].real();
    });
#ifdef ADM_BENCH_FFTW
    // Optional FFTW reference (plan reused across reps, same as ducc0; see
    // fftw_c2c for the ESTIMATE-vs-MEASURE tradeoff). Accuracy-gated: a
    // mis-scaled/inaccurate FFTW result is reported and its timing skipped.
    fftw_c2c<T> fftw(N, nthreads);
    const double fftw_l2 = (N <= kNaiveRefMaxN)
        ? l2_rel_error<T>(fftw.forward(data), reference_forward_dft<T>(data))
        : l2_rel_error<T>(fftw.forward(data), gated_fwd);
    const bool fftw_ok = fftw_l2 <= tol;
    NbStat fftw_fwd{0, 0, 0}, fftw_rt{0, 0, 0};
    if (fftw_ok) {
        fftw_fwd = nb_measure("fftw_fwd", reps, inner, [&]() { sink += fftw.forward(data)[N / 2].real(); });
        fftw_rt  = nb_measure("fftw_rt",  reps, inner, [&]() { sink += fftw.roundtrip(data)[N / 2].real(); });
    }
#endif
    (void)sink;

    // Ratio from CPU cycles when perf counters are available (frequency- and
    // contention-invariant), else fall back to wall-clock. `metric` documents
    // which was used so a reader never mistakes a clock-skewed elapsed ratio for a
    // cycle-true one. Threaded (nthreads>1): FORCE wall-clock — the per-process
    // cycle counter only sees the calling thread, so it undercounts MT work.
    const bool use_cyc = nthreads == 1
                      && fft_fwd.cyc > 0.0 && ducc_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && ducc_rt.cyc > 0.0;
    const double fwd_ratio = use_cyc ? fft_fwd.cyc / ducc_fwd.cyc : fft_fwd.us / ducc_fwd.us;
    const double rt_ratio = use_cyc ? fft_rt.cyc / ducc_rt.cyc : fft_rt.us / ducc_rt.us;
    const char* metric = use_cyc ? "cyc" : "wall";
    // Worst stability across the four readings; >5% => treat the ratio as suspect.
    const double max_err =
        std::max(std::max(fft_fwd.err, fft_rt.err), std::max(ducc_fwd.err, ducc_rt.err));
    constexpr double kUnstable = 0.05;
    const bool unstable = max_err > kUnstable;
    const bool lose = !(fwd_ratio < 1.0 && rt_ratio < 1.0);

    // Throughput on the forward transform, for ours and ducc0:
    //   GFLOPS  = flops / (us * 1e3)        -- wall-clock, familiar units
    //   flops/cycle (+ %peak)               -- frequency-invariant (use_cyc only)
    const double flops = fft_flops(N);
    const double fft_gflops = flops / (fft_fwd.us * 1e3);
    const double ducc_gflops = flops / (ducc_fwd.us * 1e3);
    constexpr double peak = peak_flops_per_cycle<T>();
    const double fft_fpc = use_cyc ? flops / fft_fwd.cyc : 0.0;
    const double ducc_fpc = use_cyc ? flops / ducc_fwd.cyc : 0.0;

    std::cout << "CMP size=" << std::setw(5) << N
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << metric
              << std::fixed
              << " fft_fwd_us=" << std::setprecision(4) << std::setw(10) << fft_fwd.us
              << " ducc_fwd_us=" << std::setw(10) << ducc_fwd.us
              << " fwd_ratio=" << std::setprecision(3) << std::setw(7) << fwd_ratio
              << " | fft_rt_us=" << std::setprecision(4) << std::setw(10) << fft_rt.us
              << " ducc_rt_us=" << std::setw(10) << ducc_rt.us
              << " rt_ratio=" << std::setprecision(3) << std::setw(7) << rt_ratio
              << " err=" << std::setprecision(1) << std::setw(4) << (max_err * 100.0) << "%"
              << " l2err=" << std::scientific << std::setprecision(1) << l2err << std::defaultfloat
              << std::setprecision(2)
              << " || fft_GFLOPS=" << std::setw(7) << fft_gflops
              << " ducc_GFLOPS=" << std::setw(7) << ducc_gflops;
    if (use_cyc) {
        std::cout << " fft_f/c=" << std::setw(5) << fft_fpc
                  << "(" << std::setprecision(0) << std::setw(3) << (100.0 * fft_fpc / peak) << "%pk)"
                  << std::setprecision(2) << " ducc_f/c=" << std::setw(5) << ducc_fpc
                  << "(" << std::setprecision(0) << std::setw(3) << (100.0 * ducc_fpc / peak) << "%pk)";
    }
    std::cout << (unstable ? "  <== UNSTABLE" : "")
              << (lose ? "  <== LOSE" : "")
              << "\n";
#ifdef ADM_BENCH_FFTW
    if (fftw_ok) {
        const bool ucf = fft_fwd.cyc > 0.0 && fftw_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && fftw_rt.cyc > 0.0;
        const double fwd_r = ucf ? fft_fwd.cyc / fftw_fwd.cyc : fft_fwd.us / fftw_fwd.us;
        const double rt_r  = ucf ? fft_rt.cyc  / fftw_rt.cyc  : fft_rt.us  / fftw_rt.us;
        std::cout << "  FFTW size=" << std::setw(5) << N
                  << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
                  << " m=" << (ucf ? "cyc" : "wall") << std::fixed
                  << " fftw_fwd_us=" << std::setprecision(4) << std::setw(10) << fftw_fwd.us
                  << " fftw_rt_us=" << std::setw(10) << fftw_rt.us
                  << " fft/fftw_fwd=" << std::setprecision(3) << std::setw(7) << fwd_r
                  << " fft/fftw_rt=" << std::setw(7) << rt_r
                  << (fwd_r < 1.0 ? "  fft<=FFTW(fwd)" : "  FFTW<fft(fwd)") << "\n";
    } else {
        std::cout << "  FFTW size=" << N << " <== FFTW accuracy FAIL (l2="
                  << std::scientific << std::setprecision(2) << fftw_l2 << std::defaultfloat
                  << "), timing skipped\n";
    }
#endif
    return !(unstable || lose);
}

// ---- In-process interleaved A/B of two DIF factorizations (trustworthy baseline) ----
// Comparing two factorizations across two PROCESSES is the drift trap: per-process
// turbo residency and (when perf counters are unavailable) wall-clock frequency make
// the cross-process ratio untrustworthy. This measures plan A and plan B in ONE
// process, interleaved round-by-round (A and B adjacent in time so any residual drift
// cancels), and reports the cycle-true A/B ratio directly. Both plans are forced onto
// iterative_dif by plan_impl (the override path), so the comparison is the pure
// factorization effect. ducc0 is timed each round as a stable absolute anchor.
inline double median_of(std::vector<double> v) {
    // -ffinite-math-only makes NaN UB (clang -Werror flags quiet_NaN); use a
    // large finite sentinel for the degenerate empty-input case instead.
    if (v.empty()) return std::numeric_limits<double>::max();
    const auto mid = static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[static_cast<std::size_t>(mid)];
}
// Median absolute deviation about the median — robust spread, in the ratio's own units.
inline double mad_of(const std::vector<double>& v) {
    const double m = median_of(v);
    std::vector<double> d;
    d.reserve(v.size());
    for (double x : v) d.push_back(std::abs(x - m));
    return median_of(d);
}
inline double geomean_of(const std::vector<double>& v) {
    double s = 0.0; for (double x : v) s += std::log(x);
    return std::exp(s / static_cast<double>(v.size()));
}

// base_a/base_b: optional codelet-terminal size per side (dif_factor_plan::base_n).
// 0 = plain chain. This makes the tool the role-swapped arbiter for terminal
// candidates — TERMAB is NOT role-swapped and inflates forced-B by 5-15%.
template<typename T>
bool compare_factors_ab(std::size_t N,
                        const std::vector<unsigned>& fa,
                        const std::vector<unsigned>& fb,
                        int rounds, int reps, long inner,
                        double tol = default_accuracy_tol<T>(),
                        unsigned base_a = 0, unsigned base_b = 0) {
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    std::vector<std::complex<T>> buf(N);
    volatile T sink = T(0);
    auto geomean = [](const std::vector<double>& v) {
        double s = 0.0; for (double x : v) s += std::log(x);
        return std::exp(s / static_cast<double>(v.size()));
    };

    // One measurement phase: build the FIRST and SECOND plan objects in that heap
    // allocation order, warm them, then time `rounds` of first-vs-second with the
    // per-round position alternated. Returns the per-round first/second cyc ratios
    // (fwd & rt) plus first/ducc and second/ducc. The plan allocated FIRST carries a
    // ~2-3% layout/first-touch advantage that follows the OBJECT (warm-up and
    // position-alternation can't remove it); the caller cancels it by running this
    // twice with the roles swapped (sqrt of the ratio-of-ratios).
    struct phase { std::vector<double> fs, rt, fd, sd; bool any_wall; };
    auto run_phase = [&](const std::vector<unsigned>& ff, unsigned bf,
                         const std::vector<unsigned>& fs_, unsigned bs) -> phase {
        const auto pf = make_dif_factor_plan(ff, bf);
        const auto ps = make_dif_factor_plan(fs_, bs);
        admiral::detail::plan_impl<T> f_fwd(N, true, &pf), f_inv(N, false, &pf);
        admiral::detail::plan_impl<T> s_fwd(N, true, &ps), s_inv(N, false, &ps);
        auto t_fwd = [&](admiral::detail::plan_impl<T>& p) {
            return nb_measure("ab_fwd", reps, inner, [&]() {
                std::copy(data.begin(), data.end(), buf.begin());
                p.execute(std::span(buf)); sink += buf[N / 2].real();
            });
        };
        auto t_rt = [&](admiral::detail::plan_impl<T>& pf2, admiral::detail::plan_impl<T>& pi) {
            return nb_measure("ab_rt", reps, inner, [&]() {
                std::copy(data.begin(), data.end(), buf.begin());
                pf2.execute(std::span(buf)); pi.execute(std::span(buf)); sink += buf[N / 2].real();
            });
        };
        for (int w = 0; w < 3; ++w) {   // pre-fault + warm caches for both
            std::copy(data.begin(), data.end(), buf.begin()); f_fwd.execute(std::span(buf)); f_inv.execute(std::span(buf));
            std::copy(data.begin(), data.end(), buf.begin()); s_fwd.execute(std::span(buf)); s_inv.execute(std::span(buf));
            sink += buf[0].real();
        }
        phase p; p.any_wall = false;
        for (int r = 0; r < rounds; ++r) {
            NbStat ff_s, sf_s, fr_s, sr_s;
            if ((r & 1) == 0) { ff_s = t_fwd(f_fwd); sf_s = t_fwd(s_fwd); fr_s = t_rt(f_fwd, f_inv); sr_s = t_rt(s_fwd, s_inv); }
            else              { sf_s = t_fwd(s_fwd); ff_s = t_fwd(f_fwd); sr_s = t_rt(s_fwd, s_inv); fr_s = t_rt(f_fwd, f_inv); }
            const NbStat df = nb_measure("ab_ducc", reps, inner, [&]() {
                auto out = ducc0_forward_fft<T>(data); sink += out[N / 2].real();
            });
            const bool cyc = ff_s.cyc > 0 && sf_s.cyc > 0 && fr_s.cyc > 0 && sr_s.cyc > 0 && df.cyc > 0;
            p.any_wall = p.any_wall || !cyc;
            const auto M = [&](const NbStat& s) { return cyc ? s.cyc : s.us; };
            p.fs.push_back(M(ff_s) / M(sf_s));
            p.rt.push_back(M(fr_s) / M(sr_s));
            p.fd.push_back(M(ff_s) / M(df));
            p.sd.push_back(M(sf_s) / M(df));
        }
        return p;
    };

    // Accuracy gate both plans (build once, transform, compare to the reference DFT)
    // before spending time on the two timing phases.
    {
        const auto pa = make_dif_factor_plan(fa, base_a);
        const auto pb = make_dif_factor_plan(fb, base_b);
        auto accurate = [&](const admiral::detail::dif_factor_plan& p) {
            admiral::detail::plan_impl<T> fp(N, true, &p);
            std::copy(data.begin(), data.end(), buf.begin());
            fp.execute(std::span(buf));
            return l2_rel_error<T>(buf, reference_forward_dft<T>(data)) <= tol;
        };
        if (!accurate(pa) || !accurate(pb)) {
            std::cout << "ABCMP size=" << N << "  <== FAIL (a or b inaccurate, not timed)\n";
            return false;
        }
    }

    const phase ab = run_phase(fa, base_a, fb, base_b);   // A allocated first
    const phase ba = run_phase(fb, base_b, fa, base_a);   // B allocated first
    (void)sink;

    // sqrt(M_ab / M_ba) cancels the per-object allocated-first advantage γ exactly:
    //   M_ab = (tA·γ)/tB,  M_ba = (tB·γ)/tA  =>  M_ab/M_ba = (tA/tB)^2.
    const double mab_f = geomean(ab.fs), mba_f = geomean(ba.fs);
    const double mab_r = geomean(ab.rt), mba_r = geomean(ba.rt);
    const double mfwd = std::sqrt(mab_f / mba_f);
    const double mrt  = std::sqrt(mab_r / mba_r);
    // Spread: round-to-round noise (MAD) plus the disagreement between the two phases
    // (|M_ab·M_ba − 1|/2 ≈ residual after cancellation) — both fold into the floor a
    // result must clear.
    const double spread = std::max(mad_of(ab.fs), mad_of(ba.fs))
                        + 0.5 * std::abs(mab_f * mba_f - 1.0);
    const bool any_wall = ab.any_wall || ba.any_wall;
    // A/ducc, B/ducc: A is "first" in ab and "second" in ba — geomean over both roles
    // averages out γ for the anchor too.
    const double a_ducc = std::sqrt(geomean(ab.fd) * geomean(ba.sd));
    const double b_ducc = std::sqrt(geomean(ab.sd) * geomean(ba.fd));

    // "A robustly faster" requires the corrected edge to clear the spread (2*MAD-ish)
    // on BOTH fwd and rt — a sub-spread delta is noise, not a result.
    const bool robust_a = (mfwd < 1.0 - 2.0 * spread) && (mrt < 1.0);
    const bool robust_b = (mfwd > 1.0 + 2.0 * spread) && (mrt > 1.0);
    const char* verdict = robust_a ? "A faster (robust)"
                        : robust_b ? "B faster (robust)"
                                   : "tie (within noise)";
    std::cout << "ABCMP size=" << std::setw(5) << N
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << (any_wall ? "WALL!" : "cyc") << std::fixed << std::setprecision(3)
              << " A=[" << join_radices(fa) << (base_a ? "+" + std::to_string(base_a) : "")
              << "] B=[" << join_radices(fb) << (base_b ? "+" + std::to_string(base_b) : "") << "]"
              << " | fwd A/B=" << std::setw(6) << mfwd
              << " (A/ducc=" << a_ducc << " B/ducc=" << b_ducc << ")"
              << " | rt A/B=" << std::setw(6) << mrt
              << " | rounds=" << rounds << "x2 spread=" << std::setprecision(1)
              << (spread * 100.0) << "%"
              << "  <== " << verdict
              << (any_wall ? "  [perf counters UNAVAILABLE — ratio is wall-clock, NOT trustworthy]" : "")
              << "\n";
    return true;
}

// Correctness-only sweep (§E, for CI). Builds the DEFAULT plan for N (the exact
// routing --compare and production use), forward-transforms, and checks the
// result against the reference DFT. No timing. Returns true iff accurate.
template<typename T>
bool verify_size(std::size_t N, double tol) {
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    admiral::detail::plan_impl<T> fwd_plan(N, true, nullptr);
    std::vector<std::complex<T>> buf(data);
    fwd_plan.execute(std::span(buf));
    const double l2err = l2_rel_error<T>(buf, reference_forward_dft<T>(data));
    const bool ok = l2err <= tol;
    std::cout << (ok ? "PASS " : "FAIL ")
              << "size=" << std::setw(5) << N
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " l2err=" << std::scientific << std::setprecision(2) << l2err
              << " tol=" << tol << std::defaultfloat << "\n";
    return ok;
}

// Format a shape as "RxCxD..." for the compare output.
std::string shape_to_string(const std::vector<std::size_t>& shape) {
    std::string s;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) s += 'x';
        s += std::to_string(shape[i]);
    }
    return s;
}

// N-D paired compare (arbitrary rank). Builds a reusable admiral::plan<T>(shape)
// ONCE (warm, plan-reuse — fair vs ducc0's cached c2c), then times our
// plan.execute() against ducc0's N-D c2c and, when -DFFT_BENCH_FFTW is set,
// FFTW's general-rank plan. Forward and round-trip, per precision. The innermost
// axis reuses the 1D plan_impl (expect ~1D parity); smooth outer axes take the
// batched DIF column path. ratio = fft/ref; <1.0 means this FFT wins.
template<typename T>
bool compare_nd(const std::vector<std::size_t>& shape, int reps, long inner, int nthreads = 1) {
    std::size_t Ntot = 1;
    for (auto e : shape) Ntot *= e;
    std::vector<std::complex<T>> data(Ntot);
    for (std::size_t i = 0; i < Ntot; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    const std::size_t nt = static_cast<std::size_t>(nthreads);
    admiral::plan<T> p(std::span<const std::size_t>(shape.data(), shape.size()), nt);
    std::vector<std::complex<T>> buf(Ntot);

    volatile T sink = T(0);
    // Out-of-place, matching ducc0's call below (reads `data`, writes fresh output):
    // an apples-to-apples comparison. The old path std::copy'd `data`->buf then ran
    // in place — charging our plan a serial full-tensor reset-copy that ducc0 (also
    // out-of-place) never pays. Our OOP execute folds input-preservation into the
    // threaded row pass, so this is both fair and faster.
    const NbStat fft_fwd = nb_measure("fftnd_fwd", reps, inner, [&]() {
        p.forward(data.data(), buf.data());
        sink += buf[Ntot / 2].real();
    });
    const NbStat fft_rt = nb_measure("fftnd_rt", reps, inner, [&]() {
        p.forward(data.data(), buf.data());  // OOP fwd (data preserved)
        p.inverse(buf.data());               // inv in place on buf
        sink += buf[Ntot / 2].real();
    });
    const NbStat ducc_fwd = nb_measure("duccnd_fwd", reps, inner, [&]() {
        auto out = ducc0_forward_fft_nd<T>(data, shape, nt);
        sink += out[Ntot / 2].real();
    });
    const NbStat ducc_rt = nb_measure("duccnd_rt", reps, inner, [&]() {
        auto fwd = ducc0_forward_fft_nd<T>(data, shape, nt);
        auto inv = ducc0_inverse_fft_nd<T>(fwd, shape, nt);
        sink += inv[Ntot / 2].real();
    });
#ifdef ADM_BENCH_FFTW
    fftw_c2c<T> fftw(shape, nthreads);
    NbStat fftw_fwd{0, 0, 0}, fftw_rt{0, 0, 0};
    fftw_fwd = nb_measure("fftwnd_fwd", reps, inner, [&]() { sink += fftw.forward(data)[Ntot / 2].real(); });
    fftw_rt  = nb_measure("fftwnd_rt",  reps, inner, [&]() { sink += fftw.roundtrip(data)[Ntot / 2].real(); });
#endif
    (void)sink;

    // Threaded (nthreads>1): force wall-clock — the per-process cycle counter only
    // sees the calling thread and undercounts the workers.
    const bool use_cyc = nthreads == 1
                      && fft_fwd.cyc > 0.0 && ducc_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && ducc_rt.cyc > 0.0;
    const double fwd_ratio = use_cyc ? fft_fwd.cyc / ducc_fwd.cyc : fft_fwd.us / ducc_fwd.us;
    const double rt_ratio = use_cyc ? fft_rt.cyc / ducc_rt.cyc : fft_rt.us / ducc_rt.us;
    const char* metric = use_cyc ? "cyc" : "wall";
    const double max_err =
        std::max(std::max(fft_fwd.err, fft_rt.err), std::max(ducc_fwd.err, ducc_rt.err));
    constexpr double kUnstable = 0.05;
    const bool unstable = max_err > kUnstable;
    const bool lose = !(fwd_ratio < 1.0 && rt_ratio < 1.0);

    std::cout << "CMPND " << std::setw(16) << shape_to_string(shape)
              << " (N=" << std::setw(9) << Ntot << ")"
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << metric
              << std::fixed
              << " fft_fwd_us=" << std::setprecision(4) << std::setw(10) << fft_fwd.us
              << " ducc_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fwd_ratio
              << " | fft_rt_us=" << std::setprecision(4) << std::setw(10) << fft_rt.us
              << " ducc_rt_ratio=" << std::setprecision(3) << std::setw(7) << rt_ratio;
#ifdef ADM_BENCH_FFTW
    {
        const bool fftw_cyc = use_cyc && fftw_fwd.cyc > 0.0 && fftw_rt.cyc > 0.0;
        const double fw = fftw_cyc ? fft_fwd.cyc / fftw_fwd.cyc : fft_fwd.us / fftw_fwd.us;
        const double rw = fftw_cyc ? fft_rt.cyc / fftw_rt.cyc : fft_rt.us / fftw_rt.us;
        std::cout << " | fftw_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fw
                  << " fftw_rt_ratio=" << std::setw(7) << rw;
    }
#endif
    std::cout << " err=" << std::setprecision(1) << std::setw(4) << (max_err * 100.0) << "%"
              << (unstable ? "  <== UNSTABLE" : "")
              << (lose ? "  <== LOSE (vs ducc0)" : "")
              << "\n";
    return !(unstable || lose);
}

// N-D r2c/c2r paired compare. Builds a reusable admiral::plan_r2c<T>(shape) ONCE,
// then times r2c (forward) and r2c->c2r (round-trip) against ducc0's r2c/c2r
// and, when built -DFFT_BENCH_FFTW, FFTW's. Accuracy-gated: our r2c must match
// ducc0's r2c and the round-trip must reproduce the real input, or it is flagged
// UNSTABLE. ratio = fft/ref; <1.0 means this FFT wins.
template<typename T>
bool compare_nd_r2c(const std::vector<std::size_t>& shape, int reps, long inner, int nthreads = 1) {
    std::size_t Nreal = 1;
    for (auto e : shape) Nreal *= e;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    std::size_t Nc = 1;
    for (auto e : cshape) Nc *= e;

    std::vector<T> real_in(Nreal);
    for (std::size_t i = 0; i < Nreal; ++i) real_in[i] = std::sin(T(i) * T(0.1)) + std::cos(T(i) * T(0.03));

    const std::size_t nt = static_cast<std::size_t>(nthreads);
    admiral::plan_r2c<T> p(std::span<const std::size_t>(shape.data(), shape.size()), nt);
    std::vector<std::complex<T>> cbuf(Nc);
    std::vector<T> rbuf(Nreal);

    // Accuracy: our r2c vs ducc0 r2c (half-spectrum), and round-trip identity.
    p.forward(real_in.data(), cbuf.data());
    const auto ref_c = ducc0_r2c_nd<T>(real_in, shape);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < Nc; ++i) {
        num += std::norm(static_cast<std::complex<double>>(cbuf[i] - ref_c[i]));
        den += std::norm(static_cast<std::complex<double>>(ref_c[i]));
    }
    const double fwd_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    std::vector<std::complex<T>> rt_c = cbuf;
    p.inverse(rt_c.data(), rbuf.data());
    double rtnum = 0.0, rtden = 0.0;
    for (std::size_t i = 0; i < Nreal; ++i) {
        rtnum += double(rbuf[i] - real_in[i]) * double(rbuf[i] - real_in[i]);
        rtden += double(real_in[i]) * double(real_in[i]);
    }
    const double rt_l2 = rtden > 0.0 ? std::sqrt(rtnum / rtden) : std::sqrt(rtnum);
    const bool inaccurate = !(fwd_l2 <= 1e-3 && rt_l2 <= 1e-3);

    volatile T sink = T(0);
    const NbStat fft_fwd = nb_measure("r2c_fwd", reps, inner, [&]() {
        p.forward(real_in.data(), cbuf.data());
        sink += cbuf[Nc / 2].real();
    });
    const NbStat fft_rt = nb_measure("r2c_rt", reps, inner, [&]() {
        p.forward(real_in.data(), cbuf.data());
        std::copy(cbuf.begin(), cbuf.end(), rt_c.begin());
        p.inverse(rt_c.data(), rbuf.data());
        sink += rbuf[Nreal / 2];
    });
    const NbStat ducc_fwd = nb_measure("ducc_r2c_fwd", reps, inner, [&]() {
        auto out = ducc0_r2c_nd<T>(real_in, shape, nt);
        sink += out[Nc / 2].real();
    });
    const NbStat ducc_rt = nb_measure("ducc_r2c_rt", reps, inner, [&]() {
        auto c = ducc0_r2c_nd<T>(real_in, shape, nt);
        auto r = ducc0_c2r_nd<T>(c, shape, nt);
        sink += r[Nreal / 2];
    });
#ifdef ADM_BENCH_FFTW
    fftw_r2c<T> fftw(shape, nthreads);
    NbStat fftw_fwd{0, 0, 0}, fftw_rt{0, 0, 0};
    fftw_fwd = nb_measure("fftw_r2c_fwd", reps, inner, [&]() { sink += fftw.forward(real_in)[Nc / 2].real(); });
    fftw_rt  = nb_measure("fftw_r2c_rt",  reps, inner, [&]() { sink += fftw.roundtrip(real_in)[Nreal / 2]; });
#endif
    (void)sink;

    // Threaded (nthreads>1): force wall-clock — the per-process cycle counter only
    // sees the calling thread and undercounts the workers.
    const bool use_cyc = nthreads == 1
                      && fft_fwd.cyc > 0.0 && ducc_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && ducc_rt.cyc > 0.0;
    const double fwd_ratio = use_cyc ? fft_fwd.cyc / ducc_fwd.cyc : fft_fwd.us / ducc_fwd.us;
    const double rt_ratio = use_cyc ? fft_rt.cyc / ducc_rt.cyc : fft_rt.us / ducc_rt.us;
    const char* metric = use_cyc ? "cyc" : "wall";
    const double max_err =
        std::max(std::max(fft_fwd.err, fft_rt.err), std::max(ducc_fwd.err, ducc_rt.err));
    const bool unstable = max_err > 0.05 || inaccurate;
    const bool lose = !(fwd_ratio < 1.0 && rt_ratio < 1.0);

    std::cout << "R2CND " << std::setw(16) << shape_to_string(shape)
              << " (N=" << std::setw(9) << Nreal << ")"
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << metric
              << std::fixed
              << " fft_fwd_us=" << std::setprecision(4) << std::setw(10) << fft_fwd.us
              << " ducc_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fwd_ratio
              << " | fft_rt_us=" << std::setprecision(4) << std::setw(10) << fft_rt.us
              << " ducc_rt_ratio=" << std::setprecision(3) << std::setw(7) << rt_ratio;
#ifdef ADM_BENCH_FFTW
    {
        const bool fftw_cyc = use_cyc && fftw_fwd.cyc > 0.0 && fftw_rt.cyc > 0.0;
        const double fw = fftw_cyc ? fft_fwd.cyc / fftw_fwd.cyc : fft_fwd.us / fftw_fwd.us;
        const double rw = fftw_cyc ? fft_rt.cyc / fftw_rt.cyc : fft_rt.us / fftw_rt.us;
        std::cout << " | fftw_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fw
                  << " fftw_rt_ratio=" << std::setw(7) << rw;
    }
#endif
    std::cout << " l2=" << std::scientific << std::setprecision(1) << std::max(fwd_l2, rt_l2)
              << std::defaultfloat
              << (unstable ? "  <== UNSTABLE" : "")
              << (lose ? "  <== LOSE (vs ducc0)" : "")
              << "\n";
    return !(unstable || lose);
}

// ============================================================================
// WS-A: trustworthy engine A/B (the --robust gate).
//
// The sequential compare_nd/compare_nd_r2c above time each engine in its own
// nb_measure block, back-to-back — the frequency/measurement-order artifact that
// produced the false 13-19x f32 r2c ratio (see [[rdtscp-minofn-frequency-trap]]).
// This generalizes compare_factors_ab's trustworthy protocol to an engine A/B:
//   * interleaved per-round timing (A and B adjacent in time, order alternated),
//   * role-swap the two phases and sqrt(mAB/mBA)-cancel the per-object first-touch
//     bias exactly (see [[trustworthy-factors-ab-tool]]),
//   * cycle-invariant metric (cpucycles, wall only as a flagged fallback),
//   * a spread/noise floor a result must clear to read as a win,
//   * an identity control (ours-vs-ours must read ~1.000 or the harness is rejected).
// An engine is a pair of "run once" thunks (fwd, rt); the maker allocates the
// engine's own state so role-swap actually swaps allocation order.
// ============================================================================
struct ab_engine { std::function<void()> fwd, rt; };

// Core: two engine makers already wired to their I/O. Prints one line; returns the
// sqrt-cancelled fwd A/B ratio (<1 => A faster). `spread_out` gets the noise floor.
template<typename MakeA, typename MakeB>
double engine_ab_core(const char* tag, const std::string& shape_str, const char* prec,
                      const char* nameA, const char* nameB,
                      MakeA&& makeA, MakeB&& makeB,
                      int rounds, int reps, long inner, double* spread_out = nullptr) {
    struct phase { std::vector<double> fs, rt; bool any_wall; };
    auto run_phase = [&](auto&& makeFirst, auto&& makeSecond) -> phase {
        ab_engine first  = makeFirst();    // allocated first (carries the γ bias)
        ab_engine second = makeSecond();
        for (int w = 0; w < 3; ++w) { first.fwd(); first.rt(); second.fwd(); second.rt(); }
        phase p; p.any_wall = false;
        for (int r = 0; r < rounds; ++r) {
            NbStat ff, sf, fr, sr;
            if ((r & 1) == 0) {
                ff = nb_measure("ab_ff", reps, inner, first.fwd);
                sf = nb_measure("ab_sf", reps, inner, second.fwd);
                fr = nb_measure("ab_fr", reps, inner, first.rt);
                sr = nb_measure("ab_sr", reps, inner, second.rt);
            } else {
                sf = nb_measure("ab_sf", reps, inner, second.fwd);
                ff = nb_measure("ab_ff", reps, inner, first.fwd);
                sr = nb_measure("ab_sr", reps, inner, second.rt);
                fr = nb_measure("ab_fr", reps, inner, first.rt);
            }
            const bool cyc = ff.cyc > 0 && sf.cyc > 0 && fr.cyc > 0 && sr.cyc > 0;
            p.any_wall = p.any_wall || !cyc;
            const auto M = [&](const NbStat& s) { return cyc ? s.cyc : s.us; };
            p.fs.push_back(M(ff) / M(sf));
            p.rt.push_back(M(fr) / M(sr));
        }
        return p;
    };

    const phase ab = run_phase(makeA, makeB);   // A allocated first
    const phase ba = run_phase(makeB, makeA);   // B allocated first
    const double mab_f = geomean_of(ab.fs), mba_f = geomean_of(ba.fs);
    const double mab_r = geomean_of(ab.rt), mba_r = geomean_of(ba.rt);
    const double mfwd = std::sqrt(mab_f / mba_f);   // (tA/tB), γ cancelled
    const double mrt  = std::sqrt(mab_r / mba_r);
    const double spread = std::max(mad_of(ab.fs), mad_of(ba.fs))
                        + 0.5 * std::abs(mab_f * mba_f - 1.0);
    if (spread_out) *spread_out = spread;
    const bool any_wall = ab.any_wall || ba.any_wall;
    const bool robust_a = (mfwd < 1.0 - 2.0 * spread) && (mrt < 1.0);
    const bool robust_b = (mfwd > 1.0 + 2.0 * spread) && (mrt > 1.0);
    const char* verdict = robust_a ? "A faster (robust)"
                        : robust_b ? "B faster (robust)"
                                   : "tie (within noise)";
    std::cout << tag << " " << std::setw(16) << shape_str
              << " prec=" << prec << " m=" << (any_wall ? "WALL!" : "cyc")
              << std::fixed << std::setprecision(3)
              << " " << nameA << "/" << nameB
              << " fwd=" << std::setw(6) << mfwd
              << " rt=" << std::setw(6) << mrt
              << " | rounds=" << rounds << "x2 spread=" << std::setprecision(1)
              << (spread * 100.0) << "%  <== " << verdict
              << (any_wall ? "  [perf counters UNAVAILABLE — wall, NOT trustworthy]" : "")
              << std::defaultfloat << "\n";
    return mfwd;
}

// c2c robust A/B: our admiral::plan vs ducc0 (and FFTW when built), with the mandatory
// ours-vs-ours identity control. Returns false if the identity control is rejected.
template<typename T>
bool compare_nd_robust(const std::vector<std::size_t>& shape, int rounds, int reps, long inner) {
    std::size_t Ntot = 1;
    for (auto e : shape) Ntot *= e;
    std::vector<std::complex<T>> data(Ntot);
    for (std::size_t i = 0; i < Ntot; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
    std::span<const std::size_t> sp(shape.data(), shape.size());
    const std::string ss = shape_to_string(shape);
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    volatile T sink = T(0);

    auto makeOurs = [&]() {
        auto plan = std::make_shared<admiral::plan<T>>(sp);
        auto buf  = std::make_shared<std::vector<std::complex<T>>>(Ntot);
        ab_engine e;
        e.fwd = [&, plan, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            plan->forward(buf->data());
            sink += (*buf)[Ntot / 2].real();
        };
        e.rt = [&, plan, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            plan->forward(buf->data());
            plan->inverse(buf->data());
            sink += (*buf)[Ntot / 2].real();
        };
        return e;
    };
    auto makeDucc = [&]() {
        ab_engine e;
        e.fwd = [&]() { auto o = ducc0_forward_fft_nd<T>(data, shape); sink += o[Ntot / 2].real(); };
        e.rt  = [&]() {
            auto f = ducc0_forward_fft_nd<T>(data, shape);
            auto v = ducc0_inverse_fft_nd<T>(f, shape);
            sink += v[Ntot / 2].real();
        };
        return e;
    };

    double id_spread = 0.0;
    const double id = engine_ab_core("IDENT", ss, prec, "ours", "ours2",
                                     makeOurs, makeOurs, rounds, reps, inner, &id_spread);
    engine_ab_core("ABND ", ss, prec, "ours", "ducc0",
                   makeOurs, makeDucc, rounds, reps, inner);
#ifdef ADM_BENCH_FFTW
    auto makeFftw = [&]() {
        auto f = std::make_shared<fftw_c2c<T>>(shape);
        ab_engine e;
        e.fwd = [&, f]() { sink += f->forward(data)[Ntot / 2].real(); };
        e.rt  = [&, f]() { sink += f->roundtrip(data)[Ntot / 2].real(); };
        return e;
    };
    engine_ab_core("ABND ", ss, prec, "ours", "fftw ",
                   makeOurs, makeFftw, rounds, reps, inner);
#endif
    (void)sink;
    const bool id_ok = std::abs(id - 1.0) <= std::max(0.03, 2.0 * id_spread);
    if (!id_ok)
        std::cout << "  <== IDENTITY CONTROL REJECTED (ours/ours=" << std::fixed
                  << std::setprecision(3) << id << ", must be ~1.000) — harness untrustworthy\n";
    return id_ok;
}

// r2c robust A/B: our plan_r2c vs ducc0 r2c/c2r (and FFTW when built), identity-gated.
template<typename T>
bool compare_nd_r2c_robust(const std::vector<std::size_t>& shape, int rounds, int reps, long inner) {
    std::size_t Nreal = 1;
    for (auto e : shape) Nreal *= e;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    std::size_t Nc = 1;
    for (auto e : cshape) Nc *= e;
    std::vector<T> real_in(Nreal);
    for (std::size_t i = 0; i < Nreal; ++i) real_in[i] = std::sin(T(i) * T(0.1)) + std::cos(T(i) * T(0.03));
    std::span<const std::size_t> sp(shape.data(), shape.size());
    const std::string ss = shape_to_string(shape);
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    volatile T sink = T(0);

    auto makeOurs = [&]() {
        auto plan = std::make_shared<admiral::plan_r2c<T>>(sp);
        auto cbuf = std::make_shared<std::vector<std::complex<T>>>(Nc);
        auto rbuf = std::make_shared<std::vector<T>>(Nreal);
        ab_engine e;
        e.fwd = [&, plan, cbuf]() {
            plan->forward(real_in.data(), cbuf->data());
            sink += (*cbuf)[Nc / 2].real();
        };
        e.rt = [&, plan, cbuf, rbuf]() {
            plan->forward(real_in.data(), cbuf->data());
            plan->inverse(cbuf->data(), rbuf->data());
            sink += (*rbuf)[Nreal / 2];
        };
        return e;
    };
    auto makeDucc = [&]() {
        ab_engine e;
        e.fwd = [&]() { auto o = ducc0_r2c_nd<T>(real_in, shape); sink += o[Nc / 2].real(); };
        e.rt  = [&]() {
            auto c = ducc0_r2c_nd<T>(real_in, shape);
            auto r = ducc0_c2r_nd<T>(c, shape);
            sink += r[Nreal / 2];
        };
        return e;
    };

    double id_spread = 0.0;
    const double id = engine_ab_core("IDENT", ss, prec, "ours", "ours2",
                                     makeOurs, makeOurs, rounds, reps, inner, &id_spread);
    engine_ab_core("R2CAB", ss, prec, "ours", "ducc0",
                   makeOurs, makeDucc, rounds, reps, inner);
#ifdef ADM_BENCH_FFTW
    auto makeFftw = [&]() {
        auto f = std::make_shared<fftw_r2c<T>>(shape);
        ab_engine e;
        e.fwd = [&, f]() { sink += f->forward(real_in)[Nc / 2].real(); };
        e.rt  = [&, f]() { sink += f->roundtrip(real_in)[Nreal / 2]; };
        return e;
    };
    engine_ab_core("R2CAB", ss, prec, "ours", "fftw ",
                   makeOurs, makeFftw, rounds, reps, inner);
#endif
    (void)sink;
    const bool id_ok = std::abs(id - 1.0) <= std::max(0.03, 2.0 * id_spread);
    if (!id_ok)
        std::cout << "  <== IDENTITY CONTROL REJECTED (ours/ours=" << std::fixed
                  << std::setprecision(3) << id << ", must be ~1.000) — harness untrustworthy\n";
    return id_ok;
}

// ============================================================================
// Phase 0 harness: batched four-step (catalog-factor) decomposition sweep.
//
// Times N = N1*N2 executed as the SIMD-batched four-step (four_step_batched_ct,
// the transpose-scatter form) against the DEFAULT plan for N and ducc0. Pure
// measurement scaffolding — no production routing. four_step_batched_ct is
// compile-time-sized and requires N1 % W == 0 && N2 % W == 0, so the dispatcher
// instantiates a curated set of W-divisible splits and the per-precision helper
// SKIPs splits that do not satisfy the width constraint for that T.
//
// The fsb path is timed as a DROP-IN: deinterleave (AoS->planar) + batched
// four-step + reinterleave (planar->AoS), matching ducc0's interleaved-complex
// I/O, so the boundary swizzle cost is included (it is part of any real routing).
// ============================================================================
template<typename T, unsigned N1, unsigned N2>
void bench_fsb(int reps, long inner) {
    using admiral::detail::four_step_batched_ct;
    using admiral::detail::build_four_step_twiddles_v;
    constexpr unsigned W = static_cast<unsigned>(xsimd::batch<T>::size);
    const char* pc = (sizeof(T) == 4) ? "f32" : "f64";
    if constexpr (N1 % W != 0 || N2 % W != 0) {
        std::cout << "FSB " << N1 << "x" << N2 << " (N=" << (N1 * N2) << ") prec=" << pc
                  << "  <== SKIP (factor not multiple of W=" << W << ")\n";
        return;
    } else {
        constexpr std::size_t N = std::size_t(N1) * N2;
        std::vector<std::complex<T>> data(N);
        for (std::size_t i = 0; i < N; ++i)
            data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

        // V-contiguous twiddles (fwd + inv), split into planar re/im.
        const auto twf = build_four_step_twiddles_v<T, true>(N1, N2, W);
        const auto twi = build_four_step_twiddles_v<T, false>(N1, N2, W);
        std::vector<T> twf_re(N), twf_im(N), twi_re(N), twi_im(N);
        for (std::size_t i = 0; i < N; ++i) {
            twf_re[i] = twf[i].real(); twf_im[i] = twf[i].imag();
            twi_re[i] = twi[i].real(); twi_im[i] = twi[i].imag();
        }
        std::vector<T> are(N), aim(N), bre(N), bim(N), Gre(N), Gim(N);
        auto deinterleave = [&](const std::vector<std::complex<T>>& src) {
            for (std::size_t i = 0; i < N; ++i) { are[i] = src[i].real(); aim[i] = src[i].imag(); }
        };

        // Accuracy gate (forward): fsb result vs reference DFT.
        deinterleave(data);
        four_step_batched_ct<N1, N2, T, true>(are.data(), aim.data(), bre.data(), bim.data(),
                                              twf_re.data(), twf_im.data(), Gre.data(), Gim.data());
        std::vector<std::complex<T>> got(N);
        for (std::size_t i = 0; i < N; ++i) got[i] = std::complex<T>(bre[i], bim[i]);
        const double l2err = l2_rel_error<T>(got, reference_forward_dft<T>(data));
        if (!(l2err <= default_accuracy_tol<T>())) {
            std::cout << "FSB " << N1 << "x" << N2 << " (N=" << N << ") prec=" << pc
                      << " l2err=" << std::scientific << std::setprecision(2) << l2err
                      << std::defaultfloat << "  <== FAIL (inaccurate)\n";
            return;
        }

        // Default plan for N (production routing) + ducc0, for reference ratios.
        admiral::detail::plan_impl<T> def_fwd(N, true, nullptr);
        std::vector<std::complex<T>> dbuf(N);

        volatile T sink = T(0);
        const NbStat fsb_fwd = nb_measure("fsb_fwd", reps, inner, [&]() {
            deinterleave(data);
            four_step_batched_ct<N1, N2, T, true>(are.data(), aim.data(), bre.data(), bim.data(),
                                                  twf_re.data(), twf_im.data(), Gre.data(), Gim.data());
            for (std::size_t i = 0; i < N; ++i) got[i] = std::complex<T>(bre[i], bim[i]);
            sink += got[N / 2].real();
        });
        const NbStat def_fwdt = nb_measure("def_fwd", reps, inner, [&]() {
            std::copy(data.begin(), data.end(), dbuf.begin());
            def_fwd.execute(std::span(dbuf));
            sink += dbuf[N / 2].real();
        });
        const NbStat ducc_fwd = nb_measure("ducc_fwd", reps, inner, [&]() {
            auto out = ducc0_forward_fft<T>(data);
            sink += out[N / 2].real();
        });
        (void)sink;

        const bool use_cyc = fsb_fwd.cyc > 0.0 && def_fwdt.cyc > 0.0 && ducc_fwd.cyc > 0.0;
        const double fsb_vs_ducc = use_cyc ? fsb_fwd.cyc / ducc_fwd.cyc : fsb_fwd.us / ducc_fwd.us;
        const double def_vs_ducc = use_cyc ? def_fwdt.cyc / ducc_fwd.cyc : def_fwdt.us / ducc_fwd.us;
        const double fsb_vs_def  = use_cyc ? fsb_fwd.cyc / def_fwdt.cyc : fsb_fwd.us / def_fwdt.us;
        const double max_err = std::max({fsb_fwd.err, def_fwdt.err, ducc_fwd.err});
        std::cout << "FSB " << std::setw(3) << N1 << "x" << std::setw(3) << N2
                  << " (N=" << std::setw(6) << N << ") prec=" << pc
                  << " m=" << (use_cyc ? "cyc" : "wall") << std::fixed
                  << " fsb/ducc=" << std::setprecision(3) << std::setw(7) << fsb_vs_ducc
                  << " def/ducc=" << std::setw(7) << def_vs_ducc
                  << " fsb/def=" << std::setw(7) << fsb_vs_def
                  << " err=" << std::setprecision(1) << std::setw(4) << (max_err * 100.0) << "%"
                  << (max_err > 0.05 ? "  <== UNSTABLE" : "")
                  << (fsb_vs_def < 1.0 ? "  <== FSB beats default" : "")
                  << "\n";
    }
}

template<typename T>
void dispatch_fsb(int reps, long inner) {
    // Curated W-divisible 2-factor splits. The helper SKIPs any split whose
    // factors are not a multiple of this precision's W. Dense low-N coverage to
    // map the f32 batched-four-step win boundary found in the first sweep.
    bench_fsb<T, 8, 16>(reps, inner);    // 128
    bench_fsb<T, 16, 16>(reps, inner);   // 256
    bench_fsb<T, 8, 32>(reps, inner);    // 256 (alt split)
    bench_fsb<T, 8, 40>(reps, inner);    // 320
    bench_fsb<T, 8, 48>(reps, inner);    // 384
    bench_fsb<T, 16, 24>(reps, inner);   // 384 (alt split)
    bench_fsb<T, 8, 56>(reps, inner);    // 448
    bench_fsb<T, 16, 32>(reps, inner);   // 512
    bench_fsb<T, 8, 64>(reps, inner);    // 512 (alt split)
    bench_fsb<T, 24, 24>(reps, inner);   // 576
    bench_fsb<T, 16, 40>(reps, inner);   // 640
    bench_fsb<T, 16, 48>(reps, inner);   // 768
    bench_fsb<T, 24, 32>(reps, inner);   // 768 (alt split)
    bench_fsb<T, 24, 40>(reps, inner);   // 960
    bench_fsb<T, 32, 32>(reps, inner);   // 1024
    bench_fsb<T, 16, 64>(reps, inner);   // 1024 (alt split)
    bench_fsb<T, 32, 64>(reps, inner);   // 2048
    bench_fsb<T, 64, 64>(reps, inner);   // 4096
    bench_fsb<T, 20, 36>(reps, inner);   // 720  (f64 only)
    bench_fsb<T, 36, 36>(reps, inner);   // 1296 (f64 only)
    bench_fsb<T, 48, 48>(reps, inner);   // 2304
    bench_fsb<T, 60, 60>(reps, inner);   // 3600 (f64 only)
    bench_fsb<T, 40, 40>(reps, inner);   // 1600
    bench_fsb<T, 64, 128>(reps, inner);  // 8192 (v4 f64 W=8 probe)
    bench_fsb<T, 128, 64>(reps, inner);  // 8192 (alt split)
    bench_fsb<T, 128, 128>(reps, inner); // 16384
}

// ============================================================================
// vecpass probe (ducc0 cfftp_vecpass architecture): N = W*M, M = A*B, lanes =
// W independent radix-W-peel sub-transforms. Times the clean vp::vpass_forward
// against the default plan and ducc0. f64 only (W=4 specialization). N=1260.
// ============================================================================
template<typename T, unsigned A, unsigned B>
void bench_vpass(int reps, long inner) {
    using V = xsimd::batch<T>;
    constexpr unsigned W = static_cast<unsigned>(V::size);
    constexpr unsigned M = A * B;
    constexpr std::size_t N = std::size_t(W) * M;
    if constexpr (W != 4 && W != 8 && W != 16) {
        std::cout << "VPASS " << N << " <== SKIP (combine specialized W in {4,8,16})\n";
        return;
    } else {
        const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
        const double vtol = (sizeof(T) == 4) ? 1e-5 : 1e-13;
        std::vector<std::complex<T>> data(N);
        for (std::size_t i = 0; i < N; ++i)
            data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.07) + T(1)));

        // Phase 3 twist twiddles: twN[k2*W + l] = W_N^{l*k2} (per-lane, N scalars).
        // Phase 2 multipass twiddles are built internally by vpass_forward.
        std::vector<T> twN_re(N), twN_im(N);
        for (unsigned k2 = 0; k2 < M; ++k2)
            for (unsigned l = 0; l < W; ++l) {
                auto [sn, cs] = admiral::detail::portable_trig::sincos_turns<true>(
                    static_cast<unsigned long>(l) * k2, static_cast<unsigned long>(N));
                twN_re[k2 * W + l] = T(cs); twN_im[k2 * W + l] = T(sn);
            }

        std::vector<T> re(N), im(N);
        std::vector<std::complex<T>> vout(N);
        vp::multipass_tables<T> tab;
        tab.template build<true>(M);  // hoist sincos + factor plan outside timed loop
        // Working V-planes, allocated once and reused across reps (kernel-only probe).
        std::vector<V> cr(M), ci(M), nr(M), ni(M);
        auto deint = [&]() { for (std::size_t i = 0; i < N; ++i) { re[i] = data[i].real(); im[i] = data[i].imag(); } };

        // accuracy gate: vpass_forward now writes AoS std::complex<T>* directly
        deint();
        vp::vpass_forward<T, true>(re.data(), im.data(), vout.data(),
                                   twN_re.data(), twN_im.data(), tab,
                                   cr.data(), ci.data(), nr.data(), ni.data());
        const double l2 = l2_rel_error<T>(vout, reference_forward_dft<T>(data));
        if (!(l2 <= vtol)) {
            std::cout << "VPASS " << A << "x" << B << " (N=" << N << ") prec=" << prec
                      << " l2err=" << l2 << " <== FAIL\n";
            return;
        }

        admiral::detail::plan_impl<T> def_fwd(N, true, nullptr);
        std::vector<std::complex<T>> dbuf(N);
        volatile T sink = T(0);
        const NbStat vp_t = nb_measure("vpass", reps, inner, [&]() {
            deint();
            vp::vpass_forward<T, true>(re.data(), im.data(), vout.data(),
                                       twN_re.data(), twN_im.data(), tab,
                                   cr.data(), ci.data(), nr.data(), ni.data());
            // AoS output: sink a real part so the compiler can't dead-strip.
            sink += vout[N / 2].real();
        });
        const NbStat def_t = nb_measure("def", reps, inner, [&]() {
            std::copy(data.begin(), data.end(), dbuf.begin());
            def_fwd.execute(std::span(dbuf));
            sink += dbuf[N / 2].real();
        });
        const NbStat duc_t = nb_measure("ducc", reps, inner, [&]() {
            auto out = ducc0_forward_fft<T>(data);
            sink += out[N / 2].real();
        });
        (void)sink;
        const bool uc = vp_t.cyc > 0 && def_t.cyc > 0 && duc_t.cyc > 0;
        const double vp_d = uc ? vp_t.cyc / duc_t.cyc : vp_t.us / duc_t.us;
        const double de_d = uc ? def_t.cyc / duc_t.cyc : def_t.us / duc_t.us;
        const double vp_de = uc ? vp_t.cyc / def_t.cyc : vp_t.us / def_t.us;
        std::cout << "VPASS " << A << "x" << B << " (N=" << N << ") prec=" << prec << " m=" << (uc ? "cyc" : "wall")
                  << std::fixed << std::setprecision(3)
                  << " vp/ducc=" << vp_d << " def/ducc=" << de_d << " vp/def=" << vp_de
                  << std::scientific << std::setprecision(2) << " l2=" << l2
                  << std::fixed << std::setprecision(3)
                  << (vp_de < 1.0 ? "  <== vpass beats default" : "")
                  << (vp_d < 1.0 ? "  <== beats ducc0" : "") << "\n";
    }
}

// Per-codelet throughput microbench (Phase 1). Times codelet_dispatch<T> (the
// shipped compiled straight-line codelet, un-normalized forward) in isolation for
// every catalog size 2..64, accuracy-gated vs the reference DFT before timing.
// Emits CSV: the raw cyc/call is exactly what codelet_cost_cyc[] stores; cyc_per_n
// and cyc_per_nlogn normalize by data size so codelets are comparable across N.
template<typename T>
void codelet_sweep(int reps, long inner, bool with_ducc) {
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    const double tol = default_accuracy_tol<T>();
    for (std::size_t N = 2; N <= 64; ++N) {
        if (!admiral::detail::is_codelet_catalog(N)) continue;
        std::vector<std::complex<T>> data(N);
        for (std::size_t i = 0; i < N; ++i)
            data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

        std::vector<std::complex<T>> buf(data);
        admiral::detail::codelet_dispatch<T, true>(buf.data(), buf.data(), N);
        const double l2 = l2_rel_error<T>(buf, reference_forward_dft<T>(data));
        if (!(l2 <= tol)) {
            std::cout << N << "," << prec << ",FAIL,l2=" << std::scientific << l2
                      << std::defaultfloat << "\n";
            continue;
        }

        volatile T sink = T(0);
        const NbStat cs = nb_measure("codelet", reps, inner, [&]() {
            std::copy(data.begin(), data.end(), buf.begin());
            admiral::detail::codelet_dispatch<T, true>(buf.data(), buf.data(), N);
            sink += buf[N / 2].real();
        });
        NbStat dc{0, 0, 0};
        if (with_ducc) {
            dc = nb_measure("ducc", reps, inner, [&]() {
                auto out = ducc0_forward_fft<T>(data);
                sink += out[N / 2].real();
            });
        }
        (void)sink;
        const bool uc = cs.cyc > 0.0;
        const double metric = uc ? cs.cyc : cs.us;       // cyc when counters present
        const double lgn = std::log2(double(N));
        const double ratio = (with_ducc && uc && dc.cyc > 0.0) ? cs.cyc / dc.cyc
                            : (with_ducc && dc.us > 0.0) ? cs.us / dc.us : 0.0;
        std::cout << N << "," << prec << "," << (uc ? "cyc" : "wall")
                  << std::fixed << std::setprecision(2)
                  << "," << metric
                  << "," << cs.us
                  << "," << metric / double(N)
                  << "," << metric / (double(N) * lgn)
                  << "," << (with_ducc ? dc.cyc : 0.0)
                  << std::setprecision(3) << "," << ratio
                  << std::setprecision(1) << "," << (cs.err * 100.0) << "%"
                  << std::scientific << std::setprecision(1) << "," << l2 << std::defaultfloat
                  << (cs.err > 0.05 ? ",UNSTABLE" : "")
                  << "\n";
    }
}

// Optimal-decomposition report (Phase 2). For each N: the route the planner picks,
// the model-optimal route + factor split over the full 2..64 codelet catalog, and
// all candidate model costs (the shipped f64 cost model — select_route is f64-only,
// so the route is the same for either T). Flags every N where the model-optimal route
// differs from the planner's choice. CSV to stdout.
template<typename T>
void decomp_report(const std::vector<std::size_t>& sizes) {
    namespace d = admiral::detail;
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    for (std::size_t N : sizes) {
        if (N < 2) continue;
        d::plan_impl<T> plan(N, true);
        const char* route = plan.route_name();
        const d::four_step_split used = plan.four_step_split_used();

        // Candidate model costs. A large FINITE sentinel marks an infeasible route
        // (not std::infinity: the codebase builds with -ffinite-math-only, where
        // isinf() always returns false and inf comparisons are UB).
        constexpr double INF = 1e30;
        const d::four_step_split fs = d::choose_four_step_split(N);
        const double fs_cost = fs.valid() ? d::four_step_cost(fs.n1, fs.n2) : INF;
        const bool smooth = std::has_single_bit(N) || d::is_codelet_supported(N);
        const double dif_cost = smooth ? d::dif_model_cost(N) : INF;
        const double cod_cost = d::is_codelet_catalog(N) ? d::codelet_cost_cyc[N] : INF;
        const double blue_cost = d::bluestein_model_cost(N);
        const bool prime = d::ct_is_prime(static_cast<unsigned>(N));
        const double rader_cost = (prime && N > 64 && d::rader_supported(N))
            ? 2.0 * d::estimated_plan_cost(N - 1) + 17.0 * double(N) : INF;

        // Model-optimal route = argmin over feasible candidate costs.
        struct C { double c; const char* name; } cands[] = {
            {cod_cost, "codelet"}, {dif_cost, "iterative_dif"},
            {fs_cost, "four_step"}, {rader_cost, "rader"}, {blue_cost, "bluestein"}};
        const C* best = &cands[0];
        for (const C& c : cands) if (c.c < best->c) best = &c;

        // Where base_cost_table has a measured entry (2..64 + extras like 120) the
        // planner routes from it, so its winner is optimal; the analytic candidates
        // above don't model good_thomas and are unreliable here.
        const d::base_cost_entry meas = d::base_cost_for<T>(N);
        const bool table_routed = meas.cyc >= 0.f;
        const char* opt_route = best->name;
        double meas_cyc = -1.0;
        if (table_routed) {
            switch (meas.form) {
                case d::base_form::codelet:       opt_route = "codelet"; break;
                case d::base_form::iterative_dif: opt_route = "iterative_dif"; break;
                case d::base_form::good_thomas:   opt_route = "good_thomas"; break;
            }
            meas_cyc = static_cast<double>(meas.cyc);
        }

        char split[16] = "-";
        if (std::string(route).rfind("four_step", 0) == 0 && used.valid())
            std::snprintf(split, sizeof split, "%zux%zu", used.n1, used.n2);
        char opt[16] = "-";
        if (std::string(opt_route) == "four_step" && fs.valid())
            std::snprintf(opt, sizeof opt, "%zux%zu", fs.n1, fs.n2);

        const bool mismatch = std::string(opt_route) != route
            // four_step vs four_step_batched are the same family — not a mismatch.
            && !(std::string(opt_route) == "four_step"
                 && std::string(route) == "four_step_batched");
        auto fc = [](double v) {
            if (v >= 1e29) return std::string("inf");
            std::ostringstream os; os << std::fixed << std::setprecision(0) << v; return os.str();
        };
        std::cout << N << "," << prec << "," << route << "," << split
                  << "," << opt_route << "," << opt
                  << "," << fc(cod_cost) << "," << fc(dif_cost) << "," << fc(fs_cost)
                  << "," << fc(rader_cost) << "," << fc(blue_cost)
                  << "," << (meas_cyc < 0.0 ? std::string("-") : fc(meas_cyc))
                  << (mismatch ? ",MISMATCH" : ",") << "\n";
    }
}

// ============================================================================
// Route A/B: interleaved DEFAULT vs FORCED four-step (H2: L2-latency-band test).
//
// Hypothesis H2: Bailey four-step (SIMD-batched leaves) beats the iterative DIF
// pass chain at pow2 N=8192..65536, where the multi-pass DIF chain strides
// through a whole-transform SoA buffer on each pass (L2-latency-bound). The
// four-step reduces to two full-width SIMD-leaf passes + one O(N) transpose.
// Old verdict "four-step loses smooth N" was measured at N~128..4096 (compute-
// bound, f64 W=4); the L2-latency band at large pow2 is a different regime.
//
// This mode calls four_step_batched_ct DIRECTLY (test-only scaffolding), not via
// plan routing. Zero library code changes; zero behavior change when unused.
//
// Accuracy gate: forward pass vs O(N^2) reference DFT for N<=4096; skipped at
// larger N (too slow); kernel_batched correctness is inherited from the
// smaller-N tests already run by --fsb and the test suite.
//
// Splits (W-divisible for both f32/f64, closest-to-sqrt):
//   1024  = 32*32    below-band context: expected LOSE confirms harness polarity
//   2048  = 32*64    below-band context
//   4096  = 64*64    below-band context
//   8192  = 64*128   H2 target band begins
//   16384 = 128*128
//   32768 = 128*256
//   65536 = 256*256
// ============================================================================

// One-size interleaved A/B of the default plan vs four_step_batched_ct<N1,N2>.
// Skipped at compile time (if constexpr) when W-divisibility is not met so the
// static_assert inside four_step_batched_ct never fires for the skip case.
template<typename T, unsigned N1, unsigned N2>
void route_ab_fsb_one(int rounds, int reps, long inner) {
    constexpr std::size_t N = std::size_t(N1) * N2;
    constexpr unsigned W = static_cast<unsigned>(xsimd::batch<T>::size);
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";

    if constexpr (N1 % W != 0 || N2 % W != 0) {
        std::cout << "ROUTEAB N=" << N << " prec=" << prec
                  << " split=" << N1 << "x" << N2
                  << "  <== SKIP (factors not divisible by W=" << W << ")\n";
    } else {
        using admiral::detail::four_step_batched_ct;
        using admiral::detail::build_four_step_twiddles_v;

        // Build V-contiguous twiddle tables once (plan setup, outside timed loop).
        const std::vector<std::complex<T>> twv_fwd =
            build_four_step_twiddles_v<T, true>(N1, N2, W);
        const std::vector<std::complex<T>> twv_inv =
            build_four_step_twiddles_v<T, false>(N1, N2, W);
        std::vector<T> twfre(N), twfim(N), twire(N), twiim(N);
        for (std::size_t i = 0; i < N; ++i) {
            twfre[i] = twv_fwd[i].real(); twfim[i] = twv_fwd[i].imag();
            twire[i] = twv_inv[i].real(); twiim[i] = twv_inv[i].imag();
        }

        std::vector<std::complex<T>> data(N);
        for (std::size_t i = 0; i < N; ++i)
            data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

        // Accuracy gate for N<=4096: the O(N^2) reference DFT is too slow beyond that.
        // kernel_batched<128/256> correctness is inherited from smaller-N --fsb tests.
        if constexpr (N <= 4096) {
            std::vector<T> gare(N), gaim(N), gbre(N), gbim(N), gGre(N), gGim(N);
            for (std::size_t i = 0; i < N; ++i) {
                gare[i] = data[i].real(); gaim[i] = data[i].imag();
            }
            four_step_batched_ct<N1, N2, T, true>(
                gare.data(), gaim.data(), gbre.data(), gbim.data(),
                twfre.data(), twfim.data(), gGre.data(), gGim.data());
            std::vector<std::complex<T>> got(N);
            for (std::size_t i = 0; i < N; ++i)
                got[i] = std::complex<T>(gbre[i], gbim[i]);
            const double l2 = l2_rel_error<T>(got, reference_forward_dft<T>(data));
            if (!(l2 <= default_accuracy_tol<T>())) {
                std::cout << "ROUTEAB N=" << N << " prec=" << prec
                          << "  <== ACCURACY FAIL (l2=" << std::scientific
                          << std::setprecision(2) << l2 << std::defaultfloat << ")\n";
                return;
            }
        }

        volatile T sink = T(0);
        const std::string shape_str = std::to_string(N);

        // Engine A: default plan_impl (selects iterative_dif for pow2 N).
        // Matches compare_min_of_n: std::copy into buf then execute in place.
        auto makeDefault = [&data, &sink]() {
            // ponytail: rebind N — make_shared's forwarding refs odr-use the enclosing
            // constexpr, which lambdas can't touch uncaptured.
            constexpr std::size_t n = N;
            auto fwd = std::make_shared<admiral::detail::plan_impl<T>>(n, true,  nullptr);
            auto inv = std::make_shared<admiral::detail::plan_impl<T>>(n, false, nullptr);
            auto buf = std::make_shared<std::vector<std::complex<T>>>(n);
            ab_engine e;
            e.fwd = [&data, &sink, fwd, buf]() {
                std::copy(data.begin(), data.end(), buf->begin());
                fwd->execute(std::span(*buf));
                sink += (*buf)[N / 2].real();
            };
            e.rt = [&data, &sink, fwd, inv, buf]() {
                std::copy(data.begin(), data.end(), buf->begin());
                fwd->execute(std::span(*buf));
                inv->execute(std::span(*buf));
                sink += (*buf)[N / 2].real();
            };
            return e;
        };

        // Engine B: forced four_step_batched_ct (test-only, not production routing).
        // Matches four_step_batched_plan::execute: deinterleave -> compute ->
        // reinterleave (fwd); fwd -> inv -> 1/N scale (rt).
        // Per-engine heap buffers: role-swap in engine_ab_core needs independent
        // allocations for the gamma (first-touch bias) cancellation to be exact.
        auto makeFourStep = [&data, &sink, &twfre, &twfim, &twire, &twiim]() {
            constexpr std::size_t n = N;  // see makeDefault
            auto fre = std::make_shared<std::vector<T>>(n);
            auto fim = std::make_shared<std::vector<T>>(n);
            auto gre = std::make_shared<std::vector<T>>(n);
            auto gim = std::make_shared<std::vector<T>>(n);
            auto Hre = std::make_shared<std::vector<T>>(n);   // scratch G buffer
            auto Him = std::make_shared<std::vector<T>>(n);
            auto out = std::make_shared<std::vector<std::complex<T>>>(n);
            ab_engine e;
            e.fwd = [&data, &sink, &twfre, &twfim,
                     fre, fim, gre, gim, Hre, Him, out]() {
                for (std::size_t i = 0; i < N; ++i) {
                    (*fre)[i] = data[i].real();
                    (*fim)[i] = data[i].imag();
                }
                four_step_batched_ct<N1, N2, T, true>(
                    fre->data(), fim->data(), gre->data(), gim->data(),
                    twfre.data(), twfim.data(), Hre->data(), Him->data());
                for (std::size_t i = 0; i < N; ++i)
                    (*out)[i] = std::complex<T>((*gre)[i], (*gim)[i]);
                sink += (*out)[N / 2].real();
            };
            e.rt = [&data, &sink, &twfre, &twfim, &twire, &twiim,
                    fre, fim, gre, gim, Hre, Him]() {
                // Forward: data -> (gre, gim)
                for (std::size_t i = 0; i < N; ++i) {
                    (*fre)[i] = data[i].real();
                    (*fim)[i] = data[i].imag();
                }
                four_step_batched_ct<N1, N2, T, true>(
                    fre->data(), fim->data(), gre->data(), gim->data(),
                    twfre.data(), twfim.data(), Hre->data(), Him->data());
                // Inverse: (gre, gim) -> (fre, fim)
                four_step_batched_ct<N1, N2, T, false>(
                    gre->data(), gim->data(), fre->data(), fim->data(),
                    twire.data(), twiim.data(), Hre->data(), Him->data());
                // 1/N normalization (matches plan_impl::apply_inverse_scale)
                const T s = T(1) / static_cast<T>(N);
                for (std::size_t i = 0; i < N; ++i) {
                    (*fre)[i] *= s;
                    (*fim)[i] *= s;
                }
                sink += (*fre)[N / 2];
            };
            return e;
        };

        engine_ab_core("ROUTEAB", shape_str, prec, "def", "fs_bat",
                       makeDefault, makeFourStep, rounds, reps, inner);
        (void)sink;
    }
}

// Dispatcher: runtime N -> compile-time four_step_batched_ct<N1,N2> instantiation.
template<typename T>
void route_ab_four_step(const std::vector<std::size_t>& sizes,
                        int rounds, int reps, long inner) {
    for (std::size_t N : sizes) {
        switch (N) {
            case 1024:  route_ab_fsb_one<T,  32,  32>(rounds, reps, inner); break;
            case 2048:  route_ab_fsb_one<T,  32,  64>(rounds, reps, inner); break;
            case 4096:  route_ab_fsb_one<T,  64,  64>(rounds, reps, inner); break;
            case 8192:  route_ab_fsb_one<T,  64, 128>(rounds, reps, inner); break;
            case 16384: route_ab_fsb_one<T, 128, 128>(rounds, reps, inner); break;
            case 32768: route_ab_fsb_one<T, 128, 256>(rounds, reps, inner); break;
            case 65536: route_ab_fsb_one<T, 256, 256>(rounds, reps, inner); break;
            default:
                std::cout << "ROUTEAB N=" << N
                          << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
                          << "  <== no split configured"
                          << " (supported: 1024 2048 4096 8192 16384 32768 65536)\n";
        }
    }
}

// ============================================================================
// --route-ab-dif: default route vs forced iterative_dif, interleaved A/B.
//
// Engine A: plan_impl<T>(N, fwd/inv, nullptr) — whatever route the planner
//   picks (vecpass / four_step_batched / iterative_dif / ...).
// Engine B: plan_impl<T>(N, fwd/inv, &dif_plan) — forces iterative_dif via
//   the library's own build_dif_factor_plan<T>(N), or an explicit --factors=
//   chain if supplied.
//
// Output prefix: RABDIF. nameA encodes the actual route selected by the
// default plan so the output is self-documenting. Ratio <1 ⇒ default faster.
// ============================================================================
template<typename T>
void route_ab_dif_one(std::size_t N,
                      const std::vector<unsigned>& factors_override,
                      int rounds, int reps, long inner) {
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    // Build the DIF plan for engine B. Use the library's DP chain by default;
    // fall back to the user-supplied --factors= chain if one was given.
    const admiral::detail::dif_factor_plan dif_plan =
        factors_override.empty()
            ? admiral::detail::build_dif_factor_plan<T>(N)
            : make_dif_factor_plan(factors_override);

    // Determine which route the default planner picks (diagnostic only; the
    // probe plan is a throwaway and is NOT timed).
    const char* def_route = [&]() -> const char* {
        admiral::detail::plan_impl<T> probe(N, true, nullptr);
        return probe.route_name();
    }();

    const std::string shape_str = std::to_string(N);
    volatile T sink = T(0);

    // Engine A: default plan (route selected by the planner).
    auto makeDefault = [&data, &sink, N]() {
        auto fwd = std::make_shared<admiral::detail::plan_impl<T>>(N, true,  nullptr);
        auto inv = std::make_shared<admiral::detail::plan_impl<T>>(N, false, nullptr);
        auto buf = std::make_shared<std::vector<std::complex<T>>>(N);
        ab_engine e;
        e.fwd = [&data, &sink, fwd, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        e.rt = [&data, &sink, fwd, inv, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            inv->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        return e;
    };

    // Engine B: forced iterative_dif (non-null dif_plan override).
    auto makeDif = [&data, &sink, N, &dif_plan]() {
        auto fwd = std::make_shared<admiral::detail::plan_impl<T>>(N, true,  &dif_plan);
        auto inv = std::make_shared<admiral::detail::plan_impl<T>>(N, false, &dif_plan);
        auto buf = std::make_shared<std::vector<std::complex<T>>>(N);
        ab_engine e;
        e.fwd = [&data, &sink, fwd, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        e.rt = [&data, &sink, fwd, inv, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            inv->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        return e;
    };

    // nameA encodes the planner's route choice for traceability in the output.
    const std::string nameA = std::string("def[") + def_route + "]";
    engine_ab_core("RABDIF", shape_str, prec, nameA.c_str(), "dif",
                   makeDefault, makeDif, rounds, reps, inner);
    (void)sink;
}

// ============================================================================
// --base-cost: per-size, per-form absolute cycle measurement.
//
// For each (N, T) pair, measures every eligible kernel form (codelet,
// iterative_dif, good_thomas) in paired-interleaved rounds to keep thermal / turbo
// state consistent across forms. Takes min over rounds per form.
//
// Correctness guard: each forced plan is verified against reference_forward_dft
// before timing; mismatched forms print BASECOST-VERIFY-FAIL and are skipped.
//
// Output (one line per form, machine-parseable):
//   BASECOST size= <N> prec=<p> form=<f> cyc= <c> ns= <n>
// ============================================================================
template<typename T>
void base_cost_size(std::size_t N, int rounds, int reps, long inner) {
    namespace d = admiral::detail;
    using plan_t = d::plan_impl<T>;
    using rk     = typename plan_t::route_kind;
    const char* prec  = (sizeof(T) == 4) ? "f32" : "f64";
    // Slightly tighter than default_accuracy_tol for f64 (1e-9 -> 1e-10) to
    // catch a wrong-form dispatch that still lands close to the right answer.
    const double tol  = (sizeof(T) == 4) ? 1e-3 : 1e-10;

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    // Reference: exact DFT (O(N^2), fine for small N).
    const auto ref = reference_forward_dft<T>(data);

    // Eligible forms for this (N, T) pair.
    struct Form { const char* name; rk kind; };
    std::vector<Form> forms;
    if (d::is_codelet_catalog(N))
        forms.push_back({"codelet",       rk::codelet});
    forms.push_back(    {"iterative_dif", rk::iterative_dif});
    if (plan_t::good_thomas_available(N))
        forms.push_back({"good_thomas",           rk::good_thomas});

    // Build and verify each form; keep only those that pass.
    struct FormState {
        const char*                              name;
        std::shared_ptr<plan_t>                  plan;
        std::shared_ptr<std::vector<std::complex<T>>> buf;
    };
    std::vector<FormState> states;
    states.reserve(forms.size());
    for (const auto& f : forms) {
        auto pl  = std::make_shared<plan_t>(N, true, f.kind);
        auto buf = std::make_shared<std::vector<std::complex<T>>>(N);
        std::copy(data.begin(), data.end(), buf->begin());
        pl->execute(std::span(*buf));
        const double l2 = l2_rel_error<T>(*buf, ref);
        if (l2 > tol) {
            std::cout << "BASECOST-VERIFY-FAIL size= " << N << " prec=" << prec
                      << " form=" << f.name << " l2=" << l2 << "\n";
            continue;
        }
        states.push_back({f.name, std::move(pl), std::move(buf)});
    }
    if (states.empty()) return;

    const std::size_t K = states.size();
    volatile T sink = T(0);

    // Warmup all forms.
    for (auto& s : states) {
        for (int w = 0; w < 3; ++w) {
            std::copy(data.begin(), data.end(), s.buf->begin());
            s.plan->execute(std::span(*s.buf));
            sink += (*s.buf)[N / 2].real();
        }
    }

    // Per-form best (min) measurements across rounds.
    // max() (not infinity()) as the "no measurement yet" min sentinel: fast-math
    // (-ffinite-math-only) makes infinity()/isinf UB and clang -Werror rejects them.
    std::vector<double> best_cyc(K, std::numeric_limits<double>::max());
    std::vector<double> best_us (K, std::numeric_limits<double>::max());

    for (int r = 0; r < rounds; ++r) {
        // Rotate the start index each round to interleave allocation/cache order.
        for (std::size_t ki = 0; ki < K; ++ki) {
            const std::size_t idx = (ki + static_cast<std::size_t>(r)) % K;
            auto& s = states[idx];
            const NbStat st = nb_measure("bc_fwd", reps, inner, [&]() {
                std::copy(data.begin(), data.end(), s.buf->begin());
                s.plan->execute(std::span(*s.buf));
                sink += (*s.buf)[N / 2].real();
            });
            if (st.cyc > 0.0) best_cyc[idx] = std::min(best_cyc[idx], st.cyc);
            best_us[idx] = std::min(best_us[idx], st.us);
        }
    }

    // Emit one line per form.
    for (std::size_t i = 0; i < K; ++i) {
        const double cyc_out = best_cyc[i] >= std::numeric_limits<double>::max() ? 0.0 : best_cyc[i];
        const double ns_out  = best_us[i] * 1000.0;
        std::cout << "BASECOST size= " << N << " prec=" << prec
                  << " form=" << states[i].name
                  << " cyc= " << std::fixed << std::setprecision(1) << cyc_out
                  << " ns= " << std::setprecision(1) << ns_out
                  << std::defaultfloat << "\n";
    }
    (void)sink;
}

// ============================================================================
// --terminal-ab: default DIF chain vs DIF chain with a base codelet terminal.
//
// Engine A: default plan (no terminal, standard iterative_dif).
// Engine B: forced dif plan whose factor chain is the DP prefix that reduces
//   N to k; base_n=k tells the driver to finish each group with codelet_dispatch.
//   Construction: take build_dif_factor_plan<T>(N), walk the radix chain until
//   the running product equals N/k (remaining == k), then stop and set base_n.
//
// Correctness: B's forward output is verified against reference_forward_dft
// (L2 < tol) before any timing.  On failure the size is skipped with a message.
//
// Output (machine-parseable, single-space):
//   TERMAB size= N base= k prec=<p> ratio= <B/A>
// ratio < 1 means terminal wins.
// ============================================================================
template<typename T>
void terminal_ab_one(std::size_t N, std::size_t k, int rounds, int reps, long inner,
                     const std::vector<unsigned>& tchain = {}) {
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";

    // k==1: no terminal — engine B is the forced PLAIN chain given by --tchain
    // (base_n=0). Isolates chain-structure effects from the terminal itself.
    if (k != 1 && (k < 2 || k > 64)) {
        std::cerr << "--terminal-ab: base k=" << k << " must be 1 or in [2,64]\n";
        return;
    }
    if (N % k != 0) {
        std::cerr << "--terminal-ab: k=" << k << " does not divide N=" << N << "\n";
        return;
    }
    if (k != 1 && !admiral::detail::is_codelet_catalog(k)) {
        std::cerr << "--terminal-ab: k=" << k << " is not in the codelet catalog\n";
        return;
    }

    // Build engine B's plan: an explicit --tchain prefix if given, otherwise
    // walk the DP chain until remaining == k.
    const auto dp_plan = admiral::detail::build_dif_factor_plan<T>(N);
    admiral::detail::dif_factor_plan term_plan;
    if (!tchain.empty()) {
        std::size_t rem = N;
        for (const unsigned r : tchain) {
            if (rem % r != 0) {
                std::cerr << "--terminal-ab: tchain factor " << r
                          << " does not divide remaining " << rem << "\n";
                return;
            }
            term_plan.push(r);
            rem /= r;
        }
        if (rem != k) {
            std::cerr << "--terminal-ab: tchain reduces N=" << N << " to " << rem
                      << ", not base k=" << k << "\n";
            return;
        }
        term_plan.base_n = (k == 1) ? 0u : static_cast<unsigned>(k);
    } else {
        std::size_t rem = N;
        bool found = false;
        for (std::size_t i = 0; i < dp_plan.count && !found; ++i) {
            if (rem == k) { found = true; break; }
            if (rem % dp_plan.radices[i] != 0) break;
            term_plan.push(dp_plan.radices[i]);
            rem /= dp_plan.radices[i];
            if (rem == k) { found = true; }
        }
        if (!found) {
            std::cerr << "--terminal-ab: the DP chain for N=" << N << " (prec=" << prec
                      << ") never reduces to k=" << k
                      << " (chain factors: ";
            for (std::size_t i = 0; i < dp_plan.count; ++i) {
                if (i) std::cerr << '-';
                std::cerr << dp_plan.radices[i];
            }
            std::cerr << ")\n";
            return;
        }
        term_plan.base_n = (k == 1) ? 0u : static_cast<unsigned>(k);
    }

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    // Correctness gate: engine B must reproduce the reference DFT.
    {
        const auto ref = reference_forward_dft<T>(data);
        std::vector<std::complex<T>> buf(N);
        admiral::detail::plan_impl<T> fwd_b(N, true, &term_plan);
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_b.execute(std::span(buf));
        const double l2 = l2_rel_error<T>(buf, ref);
        const double tol = default_accuracy_tol<T>();
        if (!(l2 <= tol)) {
            std::cout << "TERMAB-VERIFY-FAIL size= " << N << " base= " << k
                      << " prec=" << prec << " l2=" << l2 << "\n";
            return;
        }
    }

    volatile T sink = T(0);

    // Engine A: default plan.
    auto makeDefault = [&data, &sink, N]() {
        auto fwd = std::make_shared<admiral::detail::plan_impl<T>>(N, true,  nullptr);
        auto inv = std::make_shared<admiral::detail::plan_impl<T>>(N, false, nullptr);
        auto buf = std::make_shared<std::vector<std::complex<T>>>(N);
        ab_engine e;
        e.fwd = [&data, &sink, fwd, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        e.rt = [&data, &sink, fwd, inv, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            inv->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        return e;
    };

    // Engine B: forced dif with terminal base codelet.
    auto makeTerm = [&data, &sink, N, &term_plan]() {
        auto fwd = std::make_shared<admiral::detail::plan_impl<T>>(N, true,  &term_plan);
        auto inv = std::make_shared<admiral::detail::plan_impl<T>>(N, false, &term_plan);
        auto buf = std::make_shared<std::vector<std::complex<T>>>(N);
        ab_engine e;
        e.fwd = [&data, &sink, fwd, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        e.rt = [&data, &sink, fwd, inv, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(std::span(*buf));
            inv->execute(std::span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        return e;
    };

    const std::string shape_str = std::to_string(N);
    double spread = 0.0;
    const double ratio = engine_ab_core("TERMAB", shape_str, prec, "default", "terminal",
                                        makeDefault, makeTerm, rounds, reps, inner, &spread);
    std::cout << "TERMAB size= " << N << " base= " << k << " prec=" << prec
              << " ratio= " << std::fixed << std::setprecision(3) << ratio
              << std::defaultfloat << "\n";
    (void)sink;
}

}  // anonymous namespace

int main(int argc, char** argv) {
    // Correctness-only verification sweep (§E, CI gate):
    //   --verify [--prec=f32|f64|both] [--sizes=a,b,c] [--tol=eps]
    // Builds the default plan for each size and checks it against a reference
    // DFT. Returns nonzero if ANY size is inaccurate. Default size list spans
    // the whole catalog [2..64], the large-N decomposition sizes, the weak set,
    // and assorted primes/composites so every routing path is exercised.
    {
        bool verify = false;
        std::string v_prec = "both";
        double tol_override = -1.0;
        std::vector<std::size_t> sizes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--verify") verify = true;
            else if (arg.rfind("--prec=", 0) == 0) v_prec = arg.substr(7);
            else if (arg.rfind("--tol=", 0) == 0) tol_override = std::stod(arg.substr(6));
            else if (arg.rfind("--sizes=", 0) == 0) sizes = parse_size_list(arg.substr(8));
        }
        if (verify) {
            if (sizes.empty()) {
                for (std::size_t n = 2; n <= 64; ++n) sizes.push_back(n);  // full catalog
                for (std::size_t n : {128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u,
                                      96u, 120u, 192u, 210u, 360u, 720u, 1000u, 2520u,
                                      67u, 121u, 127u, 251u, 13u, 17u, 19u, 23u, 29u, 31u})
                    sizes.push_back(n);
            }
            bool ok = true;
            auto run = [&](auto tag) {
                using T = decltype(tag);
                const double tol = tol_override < 0.0 ? default_accuracy_tol<T>() : tol_override;
                for (std::size_t N : sizes) ok = verify_size<T>(N, tol) && ok;
            };
            if (v_prec == "f64" || v_prec == "both") run(double{});
            if (v_prec == "f32" || v_prec == "both") run(float{});
            std::cout << (ok ? "VERIFY: all sizes PASS\n" : "VERIFY: FAILURES present\n");
            return ok ? 0 : 1;
        }
    }

    // Nanobench N-D compare gate:
    //   --compare-nd [--prec=f32|f64|both] [--reps=N] [--inner=M] [--nthreads=N]
    //                [--shapes=RxC,RxCxD,..] [--r2c] [--fail-on-lose]
    //   --compare-2d is a thin alias (same handler, any rank in --shapes).
    // --nthreads=N (default 1) threads our plan + ducc0(N) + FFTW(N); with N>1 the
    // ratio metric is forced to wall-clock (CPU-cycle counting is per-thread).
    // Default shape list = pow2 squares/cubes (DIF column path) + 7-smooth +
    // inner-vs-outer rectangles + a 4D smoke shape. Ratios are vs ducc0 and,
    // when built -DFFT_BENCH_FFTW, vs FFTW. --r2c swaps in the r2c/c2r paths.
    {
        bool compare_nd_mode = false;
        bool fail_on_lose = false;
        bool r2c = false;
        bool robust = false;
        std::string cmp_prec = "both";
        int reps = 9;
        int rounds = 7;
        int nthreads = 1;
        long inner = 0;
        std::vector<std::vector<std::size_t>> shapes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--compare-nd" || arg == "--compare-2d") compare_nd_mode = true;
            else if (arg == "--r2c") r2c = true;
            else if (arg == "--robust") robust = true;
            else if (arg == "--fail-on-lose") fail_on_lose = true;
            else if (arg.rfind("--prec=", 0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--rounds=", 0) == 0) rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--nthreads=", 0) == 0) nthreads = std::stoi(arg.substr(11));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--shapes=", 0) == 0) shapes = parse_nd_shape_list(arg.substr(9));
        }
        if (compare_nd_mode) {
            if (shapes.empty())
                shapes = {{256, 256}, {512, 512}, {1024, 1024},   // pow2 squares
                          {64, 64, 64}, {128, 128, 128},          // pow2 cubes
                          {60, 60}, {120, 120},                   // 7-smooth
                          {16, 256}, {256, 16}, {1024, 64},       // inner-vs-outer
                          {8, 8, 8, 8}};                          // 4D smoke
            bool ok = true;
            // ponytail: --nthreads applies to the sequential compare only; the
            // --robust A/B stays single-threaded (its trustworthy interleaving is a
            // separate axis). Warn rather than silently ignore.
            if (robust && nthreads > 1)
                std::cerr << "note: --nthreads is ignored with --robust (serial A/B)\n";
            auto run = [&](auto tag) {
                using T = decltype(tag);
                for (const auto& shape : shapes)
                    ok = (robust ? (r2c ? compare_nd_r2c_robust<T>(shape, rounds, reps, inner)
                                        : compare_nd_robust<T>(shape, rounds, reps, inner))
                                 : (r2c ? compare_nd_r2c<T>(shape, reps, inner, nthreads)
                                        : compare_nd<T>(shape, reps, inner, nthreads))) && ok;
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return (fail_on_lose && !ok) ? 1 : 0;
        }
    }

    // Nanobench compare gate:
    //   --compare [--prec=f32|f64|both] [--reps=N] [--inner=M] [--nthreads=N]
    //             [--sizes=a,b,c] [--factors=r-r-r] [--tol=eps] [--fail-on-lose]
    // --nthreads>1 threads only the ducc0/FFTW references (our 1-D transform stays
    // serial — single-transform MT is a measured NO-GO, DRAM-bound; see
    // compare_min_of_n) and forces wall-clock.
    // Default size list = full sweep union. Every plan is accuracy-gated vs a
    // reference DFT before timing (--tol overrides the per-precision default).
    {
        bool compare = false;
        bool fail_on_lose = false;
        std::string cmp_prec = "both";
        int reps = 9;
        int nthreads = 1;
        long inner = 0;
        double tol_override = -1.0;   // <0 => per-precision default
        std::vector<std::size_t> sizes;
        std::vector<unsigned> factor_override;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--compare") compare = true;
            else if (arg == "--fail-on-lose") fail_on_lose = true;
            else if (arg.rfind("--prec=", 0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--nthreads=", 0) == 0) nthreads = std::stoi(arg.substr(11));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--tol=", 0) == 0) tol_override = std::stod(arg.substr(6));
            else if (arg.rfind("--sizes=", 0) == 0) {
                sizes = parse_size_list(arg.substr(8));
            } else if (arg.rfind("--factors=", 0) == 0) {
                factor_override = parse_radix_list(arg.substr(10));
            }
        }
        if (compare) {
            if (sizes.empty())
                sizes = {2,4,8,16,32,64,128,256,512,1024,2048,4096,
                         3,5,7,11,13,17,31,
                         6,10,12,15,20,24,30,100,
                         36,48,60,90,120,210,360,720,1000,2520,
                         67,121,127,251};
            bool ok = true;
            auto run = [&](auto tag) {
                using T = decltype(tag);
                const double tol = tol_override < 0.0 ? default_accuracy_tol<T>() : tol_override;
                for (std::size_t N : sizes)
                    // inner=0 => nanobench auto-tunes epoch length (~1ms floor):
                    // short, and the reported err flags any unstable reading.
                    ok = compare_min_of_n<T>(N, reps, inner,
                                             factor_override.empty() ? nullptr : &factor_override,
                                             tol, nthreads) && ok;
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return (fail_on_lose && !ok) ? 1 : 0;
        }
    }

    // In-process interleaved A/B of two DIF factorizations — the trustworthy
    // factor-ordering baseline (replaces cross-process --factors comparison):
    //   --factors-ab=Ra-Rb-..[+k]:Sa-Sb-..[+k] [--prec=..] [--reps=N] [--rounds=K] [--inner=M]
    // Both factorizations must multiply to the same N; an optional +k suffix adds a
    // codelet terminal of size k (chain-factors x k == N). Reports the cycle-true A/B
    // ratio (A=first list, B=second), median over K rounds, with the round-to-round
    // spread so a sub-noise delta reads as a tie.
    {
        std::string ab_arg;
        std::string cmp_prec = "both";
        int reps = 9, rounds = 7;
        long inner = 0;
        double tol_override = -1.0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--factors-ab=", 0) == 0) ab_arg = arg.substr(13);
            else if (arg.rfind("--prec=", 0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--rounds=", 0) == 0) rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--tol=", 0) == 0) tol_override = std::stod(arg.substr(6));
        }
        if (!ab_arg.empty()) {
            const auto colon = ab_arg.find(':');
            if (colon == std::string::npos) {
                std::cerr << "--factors-ab needs two ':'-separated factor lists\n";
                return 1;
            }
            // Each side is Ra-Rb-..[+k] where +k appends a codelet terminal of
            // size k (dif_factor_plan::base_n) — e.g. 16-16+64 at N=16384.
            auto split_base = [](std::string side, unsigned& base) {
                const auto plus = side.find('+');
                base = 0;
                if (plus != std::string::npos) {
                    base = static_cast<unsigned>(std::stoul(side.substr(plus + 1)));
                    side.resize(plus);
                }
                return side;
            };
            unsigned base_a = 0, base_b = 0;
            const auto fa = parse_radix_list(split_base(ab_arg.substr(0, colon), base_a));
            const auto fb = parse_radix_list(split_base(ab_arg.substr(colon + 1), base_b));
            auto prod = [](const std::vector<unsigned>& f, unsigned base) {
                std::size_t p = base ? base : 1; for (unsigned r : f) p *= r; return p;
            };
            for (const unsigned b : {base_a, base_b}) {
                if (b != 0 && !admiral::detail::is_codelet_catalog(b)) {
                    std::cerr << "--factors-ab: terminal +" << b
                              << " is not in the codelet catalog\n";
                    return 1;
                }
            }
            const std::size_t na = prod(fa, base_a), nb = prod(fb, base_b);
            if (na != nb || na == 0) {
                std::cerr << "--factors-ab: the two factorizations multiply to "
                          << na << " vs " << nb << " — must match\n";
                return 1;
            }
            auto run = [&](auto tag) {
                using T = decltype(tag);
                const double tol = tol_override < 0.0 ? default_accuracy_tol<T>() : tol_override;
                compare_factors_ab<T>(na, fa, fb, rounds, reps, inner, tol, base_a, base_b);
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return 0;
        }
    }

#ifdef ADM_BENCH_FFTW
    // Interleaved fft<->FFTW A/B (the trustworthy cross-library baseline):
    //   --fftw-ab --sizes=a,b,c [--prec=..] [--reps=N] [--rounds=K] [--inner=M]
    // nanobench "cpucycles" is TSC-based (constant rate == wall clock), so
    // single-shot compare runs carry measurement-ORDER frequency bias (cold
    // turbo flatters whatever runs first, up to ~30% at memory-bound sizes).
    // This alternates fft/FFTW order round-by-round like --factors-ab so the
    // drift cancels; report = median ratio over rounds + spread.
    // ADM_BENCH_FFTW_MEASURE=1 flips FFTW planning to its tuned ceiling.
    {
        bool fftw_ab = false;
        std::string cmp_prec = "both";
        int reps = 9, rounds = 9;
        long inner = 0;
        double tol_override = -1.0;
        std::vector<std::size_t> sizes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--fftw-ab") fftw_ab = true;
            else if (arg.rfind("--prec=", 0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--rounds=", 0) == 0) rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--tol=", 0) == 0) tol_override = std::stod(arg.substr(6));
            else if (arg.rfind("--sizes=", 0) == 0) sizes = parse_size_list(arg.substr(8));
        }
        if (fftw_ab) {
            if (sizes.empty()) { std::cerr << "--fftw-ab needs --sizes=\n"; return 1; }
            auto run = [&](auto tag) {
                using T = decltype(tag);
                const double tol = tol_override < 0.0 ? default_accuracy_tol<T>() : tol_override;
                for (std::size_t N : sizes) {
                    std::vector<std::complex<T>> data(N), buf(N);
                    for (std::size_t i = 0; i < N; ++i)
                        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
                    admiral::detail::plan_impl<T> fwd(N, true), inv(N, false);
                    std::copy(data.begin(), data.end(), buf.begin());
                    fwd.execute(std::span(buf));
                    const double l2 = l2_rel_error<T>(buf, reference_forward_dft<T>(data));
                    fftw_c2c<T> fftw(N);
                    const double fl2 = l2_rel_error<T>(fftw.forward(data), reference_forward_dft<T>(data));
                    if (!(l2 <= tol) || !(fl2 <= tol)) {
                        std::cout << "FFTWAB size=" << N << " ACCURACY FAIL (fft=" << l2
                                  << " fftw=" << fl2 << "), skipped\n";
                        continue;
                    }
                    volatile T sink = T(0);
                    auto t_fft_fwd = [&]() { return nb_measure("fab_f", reps, inner, [&]() {
                        std::copy(data.begin(), data.end(), buf.begin());
                        fwd.execute(std::span(buf)); sink += buf[N / 2].real(); }); };
                    auto t_fft_rt = [&]() { return nb_measure("fab_r", reps, inner, [&]() {
                        std::copy(data.begin(), data.end(), buf.begin());
                        fwd.execute(std::span(buf)); inv.execute(std::span(buf));
                        sink += buf[N / 2].real(); }); };
                    auto t_ftw_fwd = [&]() { return nb_measure("fab_wf", reps, inner, [&]() {
                        sink += fftw.forward(data)[N / 2].real(); }); };
                    auto t_ftw_rt = [&]() { return nb_measure("fab_wr", reps, inner, [&]() {
                        sink += fftw.roundtrip(data)[N / 2].real(); }); };
                    for (int w = 0; w < 3; ++w) {   // warm both paths
                        std::copy(data.begin(), data.end(), buf.begin());
                        fwd.execute(std::span(buf)); inv.execute(std::span(buf));
                        sink += fftw.roundtrip(data)[N / 2].real();
                    }
                    std::vector<double> fr, rr;
                    bool any_wall = false;
                    for (int r = 0; r < rounds; ++r) {
                        NbStat af, bf, ar, br;
                        if ((r & 1) == 0) { af = t_fft_fwd(); bf = t_ftw_fwd(); ar = t_fft_rt(); br = t_ftw_rt(); }
                        else              { bf = t_ftw_fwd(); af = t_fft_fwd(); br = t_ftw_rt(); ar = t_fft_rt(); }
                        const bool cyc = af.cyc > 0 && bf.cyc > 0 && ar.cyc > 0 && br.cyc > 0;
                        any_wall = any_wall || !cyc;
                        fr.push_back(cyc ? af.cyc / bf.cyc : af.us / bf.us);
                        rr.push_back(cyc ? ar.cyc / br.cyc : ar.us / br.us);
                    }
                    auto spread = [](std::vector<double> v) {
                        auto [mn, mx] = std::minmax_element(v.begin(), v.end());
                        return (*mx - *mn) / *mn;
                    };
                    const double fm = median_of(fr), rm = median_of(rr);
                    std::cout << "FFTWAB size=" << std::setw(6) << N
                              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
                              << " m=" << (any_wall ? "wall" : "tsc") << std::fixed
                              << " fft/fftw fwd=" << std::setprecision(3) << fm
                              << " rt=" << rm
                              << " spread=" << std::setprecision(1) << spread(fr) * 100.0 << "%/"
                              << spread(rr) * 100.0 << "%"
                              << " rounds=" << rounds
                              << (fm < 1.0 && rm < 1.0 ? "  <== WIN" : fm < 1.0 ? "  <== fwd-win" : "")
                              << "\n";
                }
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return 0;
        }
    }
#endif

    // Closed-form cost-model diagnostic (NON-baking — never writes a table):
    //   --cost-audit=N[:Sa-Sb-..] [--prec=..] [--rounds=K] ...
    // Prints the closed-form DP pick build_dif_factor_plan<T>(N). If a candidate
    // measured-best ordering is given after ':', also runs the role-swapped,
    // cycle-true A/B (DP-pick vs candidate) so a divergence between the closed
    // form and the measured optimum is visible — to refine the *formula*, not to
    // emit an override. Reuses --factors-ab's compare_factors_ab verbatim.
    {
        std::string ca_arg;
        std::string cmp_prec = "both";
        int reps = 9, rounds = 15;
        long inner = 0;
        double tol_override = -1.0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--cost-audit=", 0) == 0) ca_arg = arg.substr(13);
            else if (arg.rfind("--prec=", 0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--rounds=", 0) == 0) rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--tol=", 0) == 0) tol_override = std::stod(arg.substr(6));
        }
        if (!ca_arg.empty()) {
            const auto colon = ca_arg.find(':');
            const std::size_t N = std::strtoul(ca_arg.substr(0, colon).c_str(), nullptr, 10);
            std::vector<unsigned> cand;
            if (colon != std::string::npos) cand = parse_radix_list(ca_arg.substr(colon + 1));
            auto fmt = [](const std::vector<unsigned>& f) {
                std::string s;
                for (std::size_t i = 0; i < f.size(); ++i) { if (i) s += '-'; s += std::to_string(f[i]); }
                return s.empty() ? std::string("none") : s;
            };
            auto run = [&](auto tag) {
                using T = decltype(tag);
                const auto p = admiral::detail::build_dif_factor_plan<T>(N);
                std::vector<unsigned> dp(p.radices.begin(), p.radices.begin() + p.count);
                std::cout << (sizeof(T) == 8 ? "f64" : "f32") << " N=" << N
                          << " closed-form-DP=" << fmt(dp);
                if (!cand.empty()) std::cout << " candidate=" << fmt(cand);
                std::cout << "\n";
                if (!cand.empty()) {
                    const double tol = tol_override < 0.0 ? default_accuracy_tol<T>() : tol_override;
                    compare_factors_ab<T>(N, dp, cand, rounds, reps, inner, tol);
                }
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return 0;
        }
    }

    // Single-pass microbench (Phase 1, cost-surface grounding):
    //   --pass=IP,ido,l1 [--prec=f32|f64] [--last] [--reps=N] [--inner=M] [--perf-iters=K]
    // Times dif_pass<T,true,IP> (or dif_pass_last with --last/ido==1) directly.
    {
        std::string pass_arg;
        std::string pp = "f64";
        bool last = false;
        int reps = 13;
        long inner = 0, perf_iters = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--pass=", 0) == 0) pass_arg = arg.substr(7);
            else if (arg == "--last") last = true;
            else if (arg.rfind("--prec=", 0) == 0) pp = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--perf-iters=", 0) == 0) perf_iters = std::stol(arg.substr(13));
        }
        if (!pass_arg.empty()) {
            const auto f = parse_radix_list(pass_arg);   // reuse the '-'/',' splitter
            if (f.size() != 3) {
                std::cerr << "--pass needs IP,ido,l1 (3 comma/dash-separated values)\n";
                return 1;
            }
            const unsigned IP = f[0];
            const std::size_t ido = f[1], l1 = f[2];
            if (last || ido == 1) {
                if (pp == "f32") pass_microbench<float>(IP, 1, l1, true, reps, inner, perf_iters);
                else             pass_microbench<double>(IP, 1, l1, true, reps, inner, perf_iters);
            } else {
                if (pp == "f32") pass_microbench<float>(IP, ido, l1, false, reps, inner, perf_iters);
                else             pass_microbench<double>(IP, ido, l1, false, reps, inner, perf_iters);
            }
            return 0;
        }
    }

    // Per-codelet throughput microbench (Phase 1):
    //   --codelet-sweep [--prec=f32|f64|both] [--reps=N] [--inner=M] [--no-ducc]
    // Emits CSV (size,prec,metric,cyc,us,cyc_per_n,cyc_per_nlogn,ducc_cyc,ratio,
    // err,l2) for every catalog size 2..64. The cyc column feeds codelet_cost_cyc
    // recalibration; cyc_per_n / cyc_per_nlogn normalize by data size.
    {
        bool csweep = false;
        std::string cs_prec = "both";
        int reps = 12;
        long inner = 0;
        bool with_ducc = true;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--codelet-sweep") csweep = true;
            else if (arg == "--no-ducc") with_ducc = false;
            else if (arg.rfind("--prec=", 0) == 0) cs_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
        }
        if (csweep) {
            std::cout << "size,prec,metric,cyc,us,cyc_per_n,cyc_per_nlogn,ducc_cyc,ratio,err,l2\n";
            if (cs_prec == "f64" || cs_prec == "both") codelet_sweep<double>(reps, inner, with_ducc);
            if (cs_prec == "f32" || cs_prec == "both") codelet_sweep<float>(reps, inner, with_ducc);
            return 0;
        }
    }

    // Optimal-decomposition report (Phase 2):
    //   --decomp-report [--prec=f32|f64|both] [--sizes=a,b,c | --range=lo-hi]
    // CSV: N,prec,planner_route,split,model_best_route,opt_split,cod,dif,fs,rader,
    // blue,meas,flag. model_best_route/meas come from the measured base_cost_table
    // where it has an entry, else the analytic argmin. MISMATCH = optimal != route.
    {
        bool report = false;
        std::string rp_prec = "both";
        std::vector<std::size_t> sizes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--decomp-report") report = true;
            else if (arg.rfind("--prec=", 0) == 0) rp_prec = arg.substr(7);
            else if (arg.rfind("--sizes=", 0) == 0) sizes = parse_size_list(arg.substr(8));
            else if (arg.rfind("--range=", 0) == 0) {
                const std::string r = arg.substr(8);
                const auto dash = r.find('-');
                const std::size_t lo = std::stoul(r.substr(0, dash));
                const std::size_t hi = std::stoul(r.substr(dash + 1));
                for (std::size_t n = lo; n <= hi; ++n) sizes.push_back(n);
            }
        }
        if (report) {
            if (sizes.empty())
                for (std::size_t n = 2; n <= 2048; ++n) sizes.push_back(n);
            std::cout << "N,prec,planner_route,split,model_best_route,opt_split,"
                         "cod,dif,fs,rader,blue,meas,flag\n";
            if (rp_prec == "f64" || rp_prec == "both") decomp_report<double>(sizes);
            if (rp_prec == "f32" || rp_prec == "both") decomp_report<float>(sizes);
            return 0;
        }
    }

    // Benchmark-only DIF factor order sweep:
    //   --factor-sweep --sizes=a,b,c [--prec=f32|f64|both] [--reps=N]
    // Emits CSV to stdout. The factor override itself is intentionally isolated
    // in make_factor_sweep_plan() so the public API and --compare path are not
    // coupled to benchmark-only experimentation.
    {
        bool factor_sweep = false;
        std::string sweep_prec = "both";
        int reps = 9;
        std::vector<std::size_t> sizes;
        std::vector<unsigned> factor_override;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--factor-sweep") factor_sweep = true;
            else if (arg.rfind("--prec=", 0) == 0) sweep_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--sizes=", 0) == 0) sizes = parse_size_list(arg.substr(8));
            else if (arg.rfind("--factors=", 0) == 0) factor_override = parse_radix_list(arg.substr(10));
        }
        if (factor_sweep) {
            if (sizes.empty()) {
                std::cerr << "--factor-sweep requires --sizes=a,b,c\n";
                return 2;
            }
            if (!(sweep_prec == "f32" || sweep_prec == "f64" || sweep_prec == "both")) {
                std::cerr << "--prec must be f32, f64, or both\n";
                return 2;
            }
            std::cout << "size,prec,radices,fft_fwd_us,fft_rt_us,ducc_fwd_us,ducc_rt_us,fwd_ratio,rt_ratio,err,l2err,status\n";
            if (!factor_override.empty()) {
                if (sweep_prec == "f64" || sweep_prec == "both") {
                    for (std::size_t N : sizes) factor_sweep_size<double>(N, factor_override, reps);
                }
                if (sweep_prec == "f32" || sweep_prec == "both") {
                    for (std::size_t N : sizes) factor_sweep_size<float>(N, factor_override, reps);
                }
                return 0;
            }
            if (sweep_prec == "f64" || sweep_prec == "both") {
                factor_sweep_precision<double>(sizes, reps);
            }
            if (sweep_prec == "f32" || sweep_prec == "both") {
                factor_sweep_precision<float>(sizes, reps);
            }
            return 0;
        }
    }

    // Phase 0 batched-four-step decomposition sweep:
    //   --fsb [--prec=f32|f64|both] [--reps=N] [--inner=M]
    // Times curated W-divisible 2-factor splits as four_step_batched_ct vs the
    // default plan and ducc0. Measurement scaffolding only.
    {
        bool fsb = false;
        std::string fsb_prec = "both";
        int reps = 12;
        long inner = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--fsb") fsb = true;
            else if (arg.rfind("--prec=", 0) == 0) fsb_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
        }
        if (fsb) {
            if (fsb_prec == "f64" || fsb_prec == "both") dispatch_fsb<double>(reps, inner);
            if (fsb_prec == "f32" || fsb_prec == "both") dispatch_fsb<float>(reps, inner);
            return 0;
        }
    }

    // Route A/B: default plan vs forced four-step (H2 L2-band hypothesis):
    //   --route-ab [--sizes=a,b,c] [--prec=f32|f64|both]
    //              [--rounds=K] [--reps=N] [--inner=M]
    // Default sizes = below-band context (1024..4096) + H2 target band (8192..65536).
    // Output prefix: ROUTEAB. Verdict: "A faster" => default wins; "B faster" => four-step wins.
    // NOTE: test-only scaffolding — four_step_batched_ct called directly, not via plan routing.
    {
        bool route_ab = false;
        std::string cmp_prec = "both";
        int reps = 9, rounds = 9;
        long inner = 0;
        std::vector<std::size_t> sizes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--route-ab") route_ab = true;
            else if (arg.rfind("--prec=", 0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--rounds=", 0) == 0) rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--sizes=", 0) == 0) sizes = parse_size_list(arg.substr(8));
        }
        if (route_ab) {
            if (sizes.empty())
                sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
            if (cmp_prec == "f64" || cmp_prec == "both")
                route_ab_four_step<double>(sizes, rounds, reps, inner);
            if (cmp_prec == "f32" || cmp_prec == "both")
                route_ab_four_step<float>(sizes, rounds, reps, inner);
            return 0;
        }
    }

    // Per-size base-kernel cost measurement:
    //   --base-cost=<comma-separated sizes> [--prec=f32|f64|both]
    //               [--rounds=K (default 6)] [--reps=N] [--inner=M]
    // For each (size, precision) pair, determines the eligible kernel forms
    // (codelet, iterative_dif, good_thomas), verifies correctness, then measures them
    // in paired-interleaved rounds (form rotation each round) and reports the
    // min-over-rounds absolute cycle/ns cost per form.
    // Output prefix: BASECOST (one line per form, machine-parseable).
    {
        std::string bc_arg;
        std::string bc_prec = "both";
        int bc_rounds = 6;
        int bc_reps   = 9;
        long bc_inner = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if      (arg.rfind("--base-cost=", 0) == 0) bc_arg    = arg.substr(12);
            else if (arg.rfind("--prec=",      0) == 0) bc_prec   = arg.substr(7);
            else if (arg.rfind("--rounds=",    0) == 0) bc_rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--reps=",      0) == 0) bc_reps   = std::stoi(arg.substr(7));
            else if (arg.rfind("--inner=",     0) == 0) bc_inner  = std::stol(arg.substr(8));
        }
        if (!bc_arg.empty()) {
            const auto sizes = parse_size_list(bc_arg);
            auto run = [&](auto tag) {
                using T = decltype(tag);
                for (std::size_t N : sizes)
                    base_cost_size<T>(N, bc_rounds, bc_reps, bc_inner);
            };
            if (bc_prec == "f64" || bc_prec == "both") run(double{});
            if (bc_prec == "f32" || bc_prec == "both") run(float{});
            return 0;
        }
    }

    // Default-route vs forced-dif A/B:
    //   --route-ab-dif=<N> [--prec=f32|f64|both] [--rounds=K] [--reps=N]
    //                      [--inner=M] [--factors=r1-r2-...]
    // Engine A: default plan (whatever route the planner picks: vecpass/fsb/dif).
    // Engine B: forced iterative_dif, using the library's own build_dif_factor_plan
    //   unless --factors= supplies an explicit radix chain.
    // Output prefix: RABDIF. Ratio < 1 => default route faster than forced-dif.
    {
        std::size_t rab_dif_N = 0;
        std::string cmp_prec = "both";
        int reps = 15, rounds = 15;
        long inner = 0;
        std::vector<unsigned> factors_override;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--route-ab-dif=", 0) == 0) rab_dif_N = std::stoul(arg.substr(15));
            else if (arg.rfind("--prec=",   0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=",   0) == 0) reps   = std::stoi(arg.substr(7));
            else if (arg.rfind("--rounds=", 0) == 0) rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--inner=",  0) == 0) inner  = std::stol(arg.substr(8));
            else if (arg.rfind("--factors=",0) == 0) factors_override = parse_radix_list(arg.substr(10));
        }
        if (rab_dif_N > 0) {
            if (cmp_prec == "f64" || cmp_prec == "both")
                route_ab_dif_one<double>(rab_dif_N, factors_override, rounds, reps, inner);
            if (cmp_prec == "f32" || cmp_prec == "both")
                route_ab_dif_one<float>(rab_dif_N, factors_override, rounds, reps, inner);
            return 0;
        }
    }

    // Terminal base-codelet A/B:
    //   --terminal-ab=<N> --base=<k> [--prec=f32|f64|both]
    //                     [--rounds=K] [--reps=N] [--inner=M]
    // Engine A: default plan.
    // Engine B: forced dif with codelet_dispatch terminal for groups of size k.
    // k must be in [2,64] (codelet catalog), divide N, and align with the DP chain.
    // Output: TERMAB line (ratio < 1 => terminal wins).
    {
        std::size_t tab_N = 0;
        std::size_t tab_k = 0;
        std::string tab_prec = "both";
        std::vector<unsigned> tab_chain;
        int tab_rounds = 15, tab_reps = 9;
        long tab_inner = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if      (arg.rfind("--terminal-ab=", 0) == 0) tab_N = std::stoul(arg.substr(14));
            else if (arg.rfind("--base=",        0) == 0) tab_k = std::stoul(arg.substr(7));
            else if (arg.rfind("--tchain=",      0) == 0) {
                std::string s = arg.substr(9);
                for (std::size_t pos = 0; pos < s.size();) {
                    const std::size_t c = s.find(',', pos);
                    tab_chain.push_back(static_cast<unsigned>(
                        std::stoul(s.substr(pos, c - pos))));
                    if (c == std::string::npos) break;
                    pos = c + 1;
                }
            }
            else if (arg.rfind("--prec=",        0) == 0) tab_prec   = arg.substr(7);
            else if (arg.rfind("--rounds=",      0) == 0) tab_rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--reps=",        0) == 0) tab_reps   = std::stoi(arg.substr(7));
            else if (arg.rfind("--inner=",       0) == 0) tab_inner  = std::stol(arg.substr(8));
        }
        if (tab_N > 0) {
            if (tab_k == 0) { std::cerr << "--terminal-ab requires --base=<k>\n"; return 1; }
            if (tab_prec == "f64" || tab_prec == "both")
                terminal_ab_one<double>(tab_N, tab_k, tab_rounds, tab_reps, tab_inner, tab_chain);
            if (tab_prec == "f32" || tab_prec == "both")
                terminal_ab_one<float>(tab_N, tab_k, tab_rounds, tab_reps, tab_inner, tab_chain);
            return 0;
        }
    }

    // vecpass probe: --vpass [--reps=N]. f64 N=1260 = 4*(63*5).
    {
        bool vpass = false;
        int reps = 15;
        long inner = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--vpass") vpass = true;
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
        }
        if (vpass) {
            bench_vpass<double, 63, 5>(reps, inner);    // f64 1260 = 4*(63*5)  (ref: loses)
            // f64 probes — at v4 f64 W=8 these give N=8*M:
            bench_vpass<double, 63, 10>(reps, inner);   // f64 v4:N=5040  = 8*630
            bench_vpass<double, 128, 8>(reps, inner);   // f64 v4:N=8192  = 8*1024
            bench_vpass<double, 128, 16>(reps, inner);  // f64 v4:N=16384 = 8*2048
            bench_vpass<double, 128, 32>(reps, inner);  // f64 v4:N=32768 = 8*4096
            bench_vpass<double, 256, 32>(reps, inner);  // f64 v4:N=65536 = 8*8192
            // f32 eligible set (N%W==0, M=N/W is 11-smooth). Map the actual winning set.
            bench_vpass<float,  63, 5>(reps, inner);    // f32 W8:2520  W16:5040  = W*315
            bench_vpass<float,  27, 6>(reps, inner);    // f32 W8:1296  W16:2592  = W*162
            bench_vpass<float,  63, 4>(reps, inner);    // f32 W8:2016  W16:4032  = W*252
            bench_vpass<float,  63, 8>(reps, inner);    // f32 W8:4032  W16:8064  = W*504
            bench_vpass<float,  63, 10>(reps, inner);   // f32 W8:5040  W16:10080 = W*630  (headline; WIN)
            bench_vpass<float,  63, 15>(reps, inner);   // f32 W8:7560  W16:15120 = W*945  (we already win vs ducc)
            bench_vpass<float, 128, 8>(reps, inner);    // f32 W8:8192  W16:16384 = W*1024 (pow2; we already win)
            bench_vpass<float,  63, 20>(reps, inner);   // f32 W8:10080 W16:20160 = W*1260
            bench_vpass<float,  63, 30>(reps, inner);   // f32 W8:15120 W16:30240 = W*1890
            bench_vpass<float,  63, 40>(reps, inner);   // f32 W8:20160 W16:40320 = W*2520
            return 0;
        }
    }

    // Single-size profiling mode: --size=N [--iters=M].
    {
        std::size_t prof_size = 0;
        long prof_iters = 0;
        std::string prof_prec = "f64";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--size=", 0) == 0) prof_size = std::stoul(arg.substr(7));
            else if (arg.rfind("--iters=", 0) == 0) prof_iters = std::stol(arg.substr(8));
            else if (arg.rfind("--prec=", 0) == 0) prof_prec = arg.substr(7);
        }
        if (prof_size > 0) {
            if (prof_iters <= 0) {
                // Default iteration count scaled so total work is roughly constant.
                prof_iters = prof_size <= 64 ? 5000000L
                           : prof_size <= 1024 ? 500000L
                           : 100000L;
            }
            return (prof_prec == "f32" || prof_prec == "float")
                       ? profile_single_size<float>(prof_size, prof_iters)
                       : profile_single_size<double>(prof_size, prof_iters);
        }
    }

    std::cout << "FFT Benchmark: fft vs ducc0 Comparison\n";
    std::cout << "=====================================================================================\n";

    // Run the full sweep once per precision so float is measured as a
    // first-class citizen alongside double.
    auto run_sweep = [](auto tag) {
        using T = decltype(tag);
        const char* prec_name = (sizeof(T) == 4) ? "float" : "double";
        std::cout << "\n#####################################################################################\n";
        std::cout << "# Precision: " << prec_name << "\n";
        std::cout << "#####################################################################################\n";
        std::cout << "  Size  Prec        Type |     FFT     |     FFT     |   ducc0     |   ducc0     | Fwd Ratio | Inv Ratio\n";
        std::cout << "                         |  Forward    |  Fwd+Inv    |  Forward    |  Fwd+Inv    | (fft/ducc)| (fft/ducc)\n";
        std::cout << "-------------------------------------------------------------------------------------\n";

        // Power-of-2 sizes
        std::cout << "\nPower-of-2 sizes:\n";
        for (size_t N : {2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u})
            benchmark_size<T>(N, "pow2");

        // Prime sizes
        std::cout << "\nPrime sizes:\n";
        for (size_t N : {3u, 5u, 7u, 11u, 13u, 17u, 31u})
            benchmark_size<T>(N, "prime");

        // Composite sizes
        std::cout << "\nComposite sizes:\n";
        for (size_t N : {6u, 10u, 12u, 15u, 20u, 24u, 30u, 100u})
            benchmark_size<T>(N, "composite");

        // Larger 7-smooth mixed-radix sizes (iterative DIF pass-chain).
        std::cout << "\nMixed-radix (7-smooth) sizes:\n";
        for (size_t N : {36u, 48u, 60u, 90u, 120u, 210u, 360u, 720u, 1000u, 2520u})
            benchmark_size<T>(N, "7-smooth");

        // Bluestein sizes: N > 64, non-pow2 and non-7-smooth (chirp-z over a padded pow2).
        std::cout << "\nBluestein (large prime / non-7-smooth) sizes:\n";
        for (size_t N : {67u, 121u, 127u, 251u})
            benchmark_size<T>(N, "bluestein");
    };

    run_sweep(double{});
    run_sweep(float{});

    std::cout << "\n====================================================================================\n";
    std::cout << "Implementation Details:\n";
    std::cout << "fft (this library):\n";
    std::cout << "  - General-purpose FFT for all sizes\n";
    std::cout << "  - No SIMD, no hand-crafted kernels\n";
    std::cout << "  - Single-threaded, simple and readable\n";
    std::cout << "  - C++20 with std::bit optimizations\n";
    std::cout << "\nducc0:\n";
    std::cout << "  - Production-grade FFT library (used by numpy/scipy)\n";
    std::cout << "  - Mixed-radix with SIMD optimizations\n";
    std::cout << "  - Single-threaded for fair comparison\n";
    std::cout << "  - Optimized for various architectures\n";
    std::cout << "\nRatio interpretation:\n";
    std::cout << "  - Ratio = fft time / ducc0 time\n";
    std::cout << "  - Ratio < 1.0: fft is faster\n";
    std::cout << "  - Ratio > 1.0: ducc0 is faster\n";
    std::cout << "  - Ratio ≈ 1.0: Similar performance\n";

    // Print detailed performance report
    print_performance_report();

    return 0;
}
