#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <nanobench.h>

#ifdef ADM_BENCH_FFTW
#include <cstdlib>
#include <fftw3.h>
#include <type_traits>
#endif

namespace bench {

template<typename T>
constexpr double default_accuracy_tol() {
    return std::is_same_v<T, float> ? 1e-3 : 1e-9;
}

struct NbStat { double us; double cyc; double err; };

inline constexpr double kStableMdape = 0.05;

template<typename T>
void pass_microbench(unsigned IP, std::size_t ido, std::size_t l1, bool last,
                     int reps, long inner, long perf_iters);

template<typename Func>
NbStat nb_measure(const char* name, int reps, long min_iters, Func&& func) {
    using ankerl::nanobench::Result;
    constexpr int kMaxMs = 64;
    NbStat stat{0.0, 0.0, 1e9};
    for (int ms = 1; ms <= kMaxMs; ms *= 2) {
        ankerl::nanobench::Bench b;
        b.output(nullptr);
        b.warmup(10);
        if (reps > 0) b.epochs(static_cast<std::size_t>(reps));
        if (min_iters > 0) b.minEpochIterations(static_cast<uint64_t>(min_iters));
        else b.minEpochTime(std::chrono::milliseconds(ms));
        b.run(name, func);
        const auto& r = b.results().back();
        const double cyc = r.minimum(Result::Measure::cpucycles);
        const bool have_cyc = cyc > 0.0 && cyc < std::numeric_limits<double>::max();
        const double err = have_cyc ? r.medianAbsolutePercentError(Result::Measure::cpucycles)
                                    : r.medianAbsolutePercentError(Result::Measure::elapsed);
        stat = {r.minimum(Result::Measure::elapsed) * 1e6, have_cyc ? cyc : 0.0, err};
        if (min_iters > 0 || stat.err <= kStableMdape) break;
    }
    return stat;
}

inline double median_of(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::max();
    const auto mid = static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[static_cast<std::size_t>(mid)];
}

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

struct ab_engine { std::function<void()> fwd, rt; };

template<typename MakeA, typename MakeB>
double engine_ab_core(const char* tag, const std::string& shape_str, const char* prec,
                      const char* nameA, const char* nameB,
                      MakeA&& makeA, MakeB&& makeB,
                      int rounds, int reps, long inner, double* spread_out = nullptr,
                      bool force_wall = false) {
    struct phase { std::vector<double> fs, rt; bool any_wall; };
    auto run_phase = [&](auto&& makeFirst, auto&& makeSecond) -> phase {
        ab_engine first  = makeFirst();
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
            const bool cyc = !force_wall && ff.cyc > 0 && sf.cyc > 0 && fr.cyc > 0 && sr.cyc > 0;
            p.any_wall = p.any_wall || !cyc;
            const auto M = [&](const NbStat& s) { return cyc ? s.cyc : s.us; };
            p.fs.push_back(M(ff) / M(sf));
            p.rt.push_back(M(fr) / M(sr));
        }
        return p;
    };

    const phase ab = run_phase(makeA, makeB);
    const phase ba = run_phase(makeB, makeA);
    const double mab_f = geomean_of(ab.fs), mba_f = geomean_of(ba.fs);
    const double mab_r = geomean_of(ab.rt), mba_r = geomean_of(ba.rt);
    const double mfwd = std::sqrt(mab_f / mba_f);
    const double mrt  = std::sqrt(mab_r / mba_r);
    const double spread = std::max(mad_of(ab.fs), mad_of(ba.fs))
                        + 0.5 * std::abs(mab_f * mba_f - 1.0);
    if (spread_out) *spread_out = spread;
    const bool counters_lost = !force_wall && (ab.any_wall || ba.any_wall);
    const bool robust_a = (mfwd < 1.0 - 2.0 * spread) && (mrt < 1.0);
    const bool robust_b = (mfwd > 1.0 + 2.0 * spread) && (mrt > 1.0);
    const char* verdict = robust_a ? "A faster (robust)"
                        : robust_b ? "B faster (robust)"
                                   : "tie (within noise)";
    std::cout << tag << " " << std::setw(16) << shape_str
              << " prec=" << prec
              << " m=" << (counters_lost ? "WALL!" : force_wall ? "wall" : "cyc")
              << std::fixed << std::setprecision(3)
              << " " << nameA << "/" << nameB
              << " fwd=" << std::setw(6) << mfwd
              << " rt=" << std::setw(6) << mrt
              << " | rounds=" << rounds << "x2 spread=" << std::setprecision(1)
              << (spread * 100.0) << "%  <== " << verdict
              << (counters_lost ? "  [perf counters UNAVAILABLE: wall, NOT trustworthy]" : "")
              << std::defaultfloat << "\n";
    return mfwd;
}

template<typename T>
bool compare_nd(const std::vector<std::size_t>& shape, int reps, long inner, int nthreads = 1);

template<typename T>
bool compare_nd_r2c(const std::vector<std::size_t>& shape, int reps, long inner, int nthreads = 1);

template<typename T>
bool compare_nd_robust(const std::vector<std::size_t>& shape, int rounds, int reps, long inner,
                       int nthreads = 1);

template<typename T>
bool compare_nd_r2c_robust(const std::vector<std::size_t>& shape, int rounds, int reps, long inner,
                           int nthreads = 1);

extern template bool compare_nd<float>(const std::vector<std::size_t>&, int, long, int);
extern template bool compare_nd<double>(const std::vector<std::size_t>&, int, long, int);
extern template bool compare_nd_r2c<float>(const std::vector<std::size_t>&, int, long, int);
extern template bool compare_nd_r2c<double>(const std::vector<std::size_t>&, int, long, int);
extern template bool compare_nd_robust<float>(const std::vector<std::size_t>&, int, int, long, int);
extern template bool compare_nd_robust<double>(const std::vector<std::size_t>&, int, int, long,
                                               int);
extern template bool compare_nd_r2c_robust<float>(const std::vector<std::size_t>&, int, int, long,
                                                  int);
extern template bool compare_nd_r2c_robust<double>(const std::vector<std::size_t>&, int, int, long,
                                                   int);

#ifdef ADM_BENCH_FFTW
inline unsigned fftw_plan_flag() {
    return std::getenv("ADM_BENCH_FFTW_ESTIMATE") ? FFTW_ESTIMATE : FFTW_MEASURE;
}

inline void fftw_load_wisdom() {
    [[maybe_unused]] static const bool once = [] {
        const char* p = std::getenv("ADM_BENCH_FFTW_WISDOM");
        if (!p) return false;
        static const std::string d = std::string(p) + ".d";
        static const std::string f = std::string(p) + ".f";
        fftw_import_wisdom_from_filename(d.c_str());
        fftwf_import_wisdom_from_filename(f.c_str());
        std::atexit([] {
            fftw_export_wisdom_to_filename(d.c_str());
            fftwf_export_wisdom_to_filename(f.c_str());
        });
        return true;
    }();
}

inline void fftw_apply_timelimit(std::size_t total_elems) {
#ifdef ADM_BENCH_FFTW
    static const double tl = [] {
        const char* e = std::getenv("ADM_BENCH_FFTW_TIMELIMIT");
        return e ? std::atof(e) : 0.0;
    }();
    static const std::size_t floor_n = [] {
        const char* e = std::getenv("ADM_BENCH_FFTW_TIMELIMIT_MIN_ELEMS");
        return e ? static_cast<std::size_t>(std::strtoull(e, nullptr, 10)) : std::size_t{1};
    }();
    if (total_elems >= floor_n) {
        if (tl > 0.0) { fftw_set_timelimit(tl); fftwf_set_timelimit(tl); }
    } else {
        fftw_set_timelimit(-1.0);
        fftwf_set_timelimit(-1.0);
    }
#endif
}

#ifdef ADM_BENCH_THREADS
template<typename T>
inline void fftw_plan_threads(std::size_t nthreads) {
    if constexpr (std::is_same_v<T, double>) {
        static const int ready = fftw_init_threads();
        fftw_plan_with_nthreads(ready ? static_cast<int>(nthreads) : 1);
    } else {
        static const int ready = fftwf_init_threads();
        fftwf_plan_with_nthreads(ready ? static_cast<int>(nthreads) : 1);
    }
}
#endif

template<typename T>
class fftw_c2c {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    using cpx_t  = std::conditional_t<std::is_same_v<T, double>, fftw_complex, fftwf_complex>;
    using plan_t = std::conditional_t<std::is_same_v<T, double>, fftw_plan, fftwf_plan>;
public:
    explicit fftw_c2c(std::size_t N, int nthreads = 1) : fftw_c2c(std::vector<std::size_t>{N}, nthreads) {}

    explicit fftw_c2c(const std::vector<std::size_t>& shape, [[maybe_unused]] int nthreads = 1)
        : N_([&]{ std::size_t n = 1; for (auto e : shape) n *= e; return n; }()),
          in_(N_), out_(N_) {
        fftw_load_wisdom();
        fftw_apply_timelimit(N_);
        std::vector<int> dims(shape.begin(), shape.end());
        const int rank = static_cast<int>(dims.size());
#ifdef ADM_BENCH_THREADS
        if (nthreads > 1) fftw_plan_threads<T>(static_cast<std::size_t>(nthreads));
#endif
        if constexpr (std::is_same_v<T, double>) {
            fwd_ = fftw_plan_dft(rank, dims.data(), cpx(in_), cpx(out_), FFTW_FORWARD,  fftw_plan_flag());
            inv_ = fftw_plan_dft(rank, dims.data(), cpx(out_), cpx(in_), FFTW_BACKWARD, fftw_plan_flag());
        } else {
            fwd_ = fftwf_plan_dft(rank, dims.data(), cpx(in_), cpx(out_), FFTW_FORWARD,  fftw_plan_flag());
            inv_ = fftwf_plan_dft(rank, dims.data(), cpx(out_), cpx(in_), FFTW_BACKWARD, fftw_plan_flag());
        }
    }
    ~fftw_c2c() {
        if constexpr (std::is_same_v<T, double>) { fftw_destroy_plan(fwd_);  fftw_destroy_plan(inv_); }
        else                                     { fftwf_destroy_plan(fwd_); fftwf_destroy_plan(inv_); }
    }
    fftw_c2c(const fftw_c2c&) = delete;
    fftw_c2c& operator=(const fftw_c2c&) = delete;

    const std::vector<std::complex<T>>& forward(const std::vector<std::complex<T>>& x) {
        std::copy(x.begin(), x.end(), in_.begin());
        exec(fwd_);
        return out_;
    }
    const std::vector<std::complex<T>>& forward_into(const std::vector<std::complex<T>>& x) {
        if constexpr (std::is_same_v<T, double>) fftw_execute_dft(fwd_, cpx_of(x), cpx(out_));
        else                                     fftwf_execute_dft(fwd_, cpx_of(x), cpx(out_));
        return out_;
    }
    [[nodiscard]] bool alignment_ok(const std::vector<std::complex<T>>& x) const {
        const auto a = [](const std::vector<std::complex<T>>& v) {
            auto* p = const_cast<std::complex<T>*>(v.data());
            if constexpr (std::is_same_v<T, double>)
                return fftw_alignment_of(reinterpret_cast<double*>(p));
            else
                return fftwf_alignment_of(reinterpret_cast<float*>(p));
        };
        return x.size() == in_.size() && a(x) == a(in_);
    }
    const std::vector<std::complex<T>>& roundtrip(const std::vector<std::complex<T>>& x) {
        std::copy(x.begin(), x.end(), in_.begin());
        exec(fwd_);
        exec(inv_);
        const T s = T(1) / static_cast<T>(N_);
        for (auto& v : in_) v *= s;
        return in_;
    }
private:
    static cpx_t* cpx(std::vector<std::complex<T>>& v) { return reinterpret_cast<cpx_t*>(v.data()); }
    static cpx_t* cpx_of(const std::vector<std::complex<T>>& v) {
        return reinterpret_cast<cpx_t*>(const_cast<std::complex<T>*>(v.data()));
    }
    static void exec(plan_t p) {
        if constexpr (std::is_same_v<T, double>) fftw_execute(p);
        else                                     fftwf_execute(p);
    }
    std::size_t N_;
    std::vector<std::complex<T>> in_, out_;
    plan_t fwd_{}, inv_{};
};

template<typename T>
class fftw_r2c {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    using cpx_t  = std::conditional_t<std::is_same_v<T, double>, fftw_complex, fftwf_complex>;
    using plan_t = std::conditional_t<std::is_same_v<T, double>, fftw_plan, fftwf_plan>;
public:
    explicit fftw_r2c(const std::vector<std::size_t>& shape, [[maybe_unused]] int nthreads = 1) {
        Nreal_ = 1; for (auto e : shape) Nreal_ *= e;
        fftw_load_wisdom();
        fftw_apply_timelimit(Nreal_);
        std::vector<std::size_t> cshape(shape);
        cshape.back() = shape.back() / 2 + 1;
        Nc_ = 1; for (auto e : cshape) Nc_ *= e;
        rin_.resize(Nreal_); cout_.resize(Nc_); cin_.resize(Nc_); rout_.resize(Nreal_);
        std::vector<int> dims(shape.begin(), shape.end());
        const int rank = static_cast<int>(dims.size());
#ifdef ADM_BENCH_THREADS
        if (nthreads > 1) fftw_plan_threads<T>(static_cast<std::size_t>(nthreads));
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
#endif

}
