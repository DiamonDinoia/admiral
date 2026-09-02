#include <admiral/admiral.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <complex>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <optional>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <ducc0/fft/fft.h>
#include <nanobench.h>
#include <admiral/detail/four_step.hpp>
#include <admiral/detail/math.hpp>
#include <admiral/detail/plan.hpp>
#include <admiral/detail/twiddles.hpp>

#include "bench_harness.hpp"

using namespace bench;
using admiral::span;

namespace {

template<typename T>
std::vector<std::complex<T>> ducc0_forward_fft(const std::vector<std::complex<T>>& input,
                                               size_t nthreads = 1) {
    using namespace ducc0;

    const size_t N = input.size();
    std::vector<std::complex<T>> output(N);

    detail_fft::shape_t shape = {N};
    auto in_view = detail_mav::cfmav<std::complex<T>>(input.data(), shape);
    auto out_view = detail_mav::vfmav<std::complex<T>>(output.data(), shape);

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

    detail_fft::c2c(in_view, out_view, {0}, false, T(1) / T(N), nthreads);

    return output;
}

template<typename T>
std::vector<std::complex<double>>
reference_forward_dft(const std::vector<std::complex<T>>& in) {
    const std::size_t N = in.size();
    std::vector<std::complex<double>> out(N);
    constexpr double two_pi = 2.0 * admiral::detail::numbers::pi_v<double>;
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

constexpr std::size_t kNaiveRefMaxN = 65536;

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

struct BenchmarkResult {
    size_t size;
    std::string type;
    std::string prec;
    double fft_fwd_ms;
    double fft_rt_ms;
    double ducc0_fwd_ms;
    double ducc0_rt_ms;

    double fwd_ratio() const { return fft_fwd_ms / ducc0_fwd_ms; }
    double rt_ratio() const { return fft_rt_ms / ducc0_rt_ms; }

    bool is_power_of_2() const {
        return admiral::detail::has_single_bit(size);
    }

    bool is_prime() const { return admiral::detail::ct_is_prime(size); }

    std::string category() const {
        if (is_power_of_2()) return "Power-of-2";
        if (is_prime()) return "Prime";
        return "Composite";
    }
};

std::vector<BenchmarkResult> all_results;

template<typename Func>
double time_execution(const char* name, Func&& func) {
    ankerl::nanobench::Bench b;
    b.output(nullptr);
    b.warmup(10);
    b.run(name, std::forward<Func>(func));
    return b.results().back().median(ankerl::nanobench::Result::Measure::elapsed) * 1e3;
}

std::vector<std::size_t> parse_size_list(const std::string& s) {
    std::vector<std::size_t> sizes;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        if (comma != pos) {
            const std::string item = s.substr(pos, comma - pos);
            const size_t dash = item.find('-');
            if (dash == std::string::npos || dash == 0) {
                sizes.push_back(std::stoul(item));
            } else {
                for (std::size_t n = std::stoul(item.substr(0, dash));
                     n <= std::stoul(item.substr(dash + 1)); ++n)
                    sizes.push_back(n);
            }
        }
        pos = comma + 1;
    }
    return sizes;
}

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

admiral::detail::dif_factor_plan make_dif_factor_plan(const std::vector<unsigned>& radices) {
    admiral::detail::dif_factor_plan plan;
    for (unsigned radix : radices) {
        plan.push(radix);
    }
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
    return admiral::detail::plan_impl<T>(n, is_forward, 1, &plan);
}

template<typename T>
void factor_sweep_size(std::size_t N, const std::vector<unsigned>& radices, int reps) {
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    auto fwd_plan = make_factor_sweep_plan<T>(N, true, radices);
    auto inv_plan = make_factor_sweep_plan<T>(N, false, radices);
    std::vector<std::complex<T>> buf(N);

    std::copy(data.begin(), data.end(), buf.begin());
    fwd_plan.execute(span(buf));
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
        fwd_plan.execute(span(buf));
        sink += buf[N / 2].real();
    });
    const NbStat fft_rt = nb_measure("factor_fft_rt", reps, 0, [&]() {
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_plan.execute(span(buf));
        inv_plan.execute(span(buf));
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

template<typename T>
void benchmark_size(size_t N, const std::string& type) {
    std::vector<std::complex<T>> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
    }

    admiral::plan<T> fwd_plan(N);
    admiral::plan<T> inv_plan(N);

    std::vector<std::complex<T>> fft_output(N);
    double fft_fwd_time = time_execution("fft_fwd", [&]() {
        std::copy(data.begin(), data.end(), fft_output.begin());
        fwd_plan.forward(span(fft_output));
        ankerl::nanobench::doNotOptimizeAway(fft_output.data());
    });

    std::vector<std::complex<T>> fft_temp1(N);
    double fft_rt_time = time_execution("fft_rt", [&]() {
        std::copy(data.begin(), data.end(), fft_temp1.begin());
        fwd_plan.forward(span(fft_temp1));
        inv_plan.inverse(span(fft_temp1));
        ankerl::nanobench::doNotOptimizeAway(fft_temp1.data());
    });

    double ducc0_fwd_time = time_execution("ducc0_fwd", [&]() {
        auto result = ducc0_forward_fft<T>(data);
        ankerl::nanobench::doNotOptimizeAway(result.data());
    });

    double ducc0_rt_time = time_execution("ducc0_rt", [&]() {
        auto fwd = ducc0_forward_fft<T>(data);
        auto inv = ducc0_inverse_fft<T>(fwd);
        ankerl::nanobench::doNotOptimizeAway(inv.data());
    });

    BenchmarkResult result;
    result.size = N;
    result.type = type;
    result.prec = (sizeof(T) == 4) ? "f32" : "f64";
    result.fft_fwd_ms = fft_fwd_time;
    result.fft_rt_ms = fft_rt_time;
    result.ducc0_fwd_ms = ducc0_fwd_time;
    result.ducc0_rt_ms = ducc0_rt_time;
    all_results.push_back(result);

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

    const auto mean = [](const std::vector<double>& v) {
        return std::reduce(v.begin(), v.end()) / static_cast<double>(v.size());
    };
    stats.avg_fwd_ratio = mean(fwd_ratios);
    stats.avg_rt_ratio = mean(rt_ratios);

    const auto [fmin, fmax] = std::minmax_element(fwd_ratios.begin(), fwd_ratios.end());
    const auto [rmin, rmax] = std::minmax_element(rt_ratios.begin(), rt_ratios.end());
    stats.min_fwd_ratio = *fmin;
    stats.max_fwd_ratio = *fmax;
    stats.min_rt_ratio = *rmin;
    stats.max_rt_ratio = *rmax;

    return stats;
}

void print_performance_report() {
    std::cout << "\n\n";
    std::cout << "=======================================================================\n";
    std::cout << "                         PERFORMANCE REPORT                            \n";
    std::cout << "=======================================================================\n\n";

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

        std::cout << "Performance by Category [" << prec_name << "]:\n";
        std::cout << "-----------------------------------------------------------------------\n";
        std::cout << std::setw(15) << "Category" << " | "
                  << std::setw(6) << "Count" << " | "
                  << "Forward Ratio (fft/ducc0) | "
                  << "Fwd+Inv Ratio (fft/ducc0)\n";
        std::cout << std::setw(15) << "" << " | "
                  << std::setw(6) << "" << " | "
                  << "Avg    Min    Max           | "
                  << "Avg    Min    Max\n";
        std::cout << "-----------------------------------------------------------------------\n";

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
            std::cout << "ducc0 faster [" << prec_name << "]: NONE, fft wins every size.\n\n";
        }
    }

    std::cout << "=======================================================================\n";
    std::cout << "\nInterpretation:\n";
    std::cout << "  - Ratio = fft time / ducc0 time\n";
    std::cout << "  - Ratio > 1.0: ducc0 is faster (fft takes more time)\n";
    std::cout << "  - Ratio < 1.0: fft is faster\n";
    std::cout << "  - Power-of-2: ducc0 excels with radix-2 optimizations\n";
    std::cout << "  - Prime: More level playing field, both use similar methods\n";
    std::cout << "  - Composite: Performance depends on factorization\n";
}

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
        fwd_plan.forward(span(buf));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    volatile double sink = static_cast<double>(buf[N / 2].real() + buf[N / 2].imag());
    (void)sink;
    std::cout << "size=" << N << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " iters=" << iters
              << " total_ms=" << std::fixed << std::setprecision(3) << total_ms
              << " per_call_us=" << std::setprecision(4) << (total_ms * 1e3 / double(iters))
              << "\n";
    return 0;
}

template<typename T>
constexpr double peak_flops_per_cycle() {
    return 2.0 * static_cast<double>(xsimd::batch<T>::size) * 2.0;
}

inline double fft_flops(std::size_t N) {
    return 5.0 * static_cast<double>(N) * std::log2(static_cast<double>(N));
}

template<typename T>
bool compare_min_of_n(std::size_t N, int reps, long inner,
                      const std::vector<unsigned>* factor_override = nullptr,
                      double tol = default_accuracy_tol<T>(), int nthreads = 1,
                      int adm_nthreads = 1,
                      admiral::effort adm_eff = admiral::effort::estimate) {
    const std::size_t nt = static_cast<std::size_t>(nthreads);
    const std::size_t adm_nt =
        admiral::detail::resolve_nthreads(static_cast<std::size_t>(adm_nthreads));
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    const std::optional<admiral::detail::dif_factor_plan> override_plan =
        factor_override ? std::optional<admiral::detail::dif_factor_plan>(make_dif_factor_plan(*factor_override))
                        : std::nullopt;
    admiral::detail::plan_impl<T> fwd_plan(N, true, adm_nt, override_plan ? &*override_plan : nullptr, adm_eff);
    admiral::detail::plan_impl<T> inv_plan(N, false, adm_nt, override_plan ? &*override_plan : nullptr, adm_eff);
    std::vector<std::complex<T>> buf(N);

    std::copy(data.begin(), data.end(), buf.begin());
    fwd_plan.execute(span(buf));
    double l2err;
    if (N <= kNaiveRefMaxN) {
        l2err = l2_rel_error<T>(buf, reference_forward_dft<T>(data));
    } else {
        std::vector<std::complex<T>> rt(buf.begin(), buf.end());
        inv_plan.execute(span(rt));
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
        fwd_plan.execute(span(buf));
        sink += buf[N / 2].real();
    });
    const NbStat fft_rt = nb_measure("fft_rt", reps, inner, [&]() {
        std::copy(data.begin(), data.end(), buf.begin());
        fwd_plan.execute(span(buf));
        inv_plan.execute(span(buf));
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

    const bool use_cyc = nthreads == 1 && adm_nt == 1
                      && fft_fwd.cyc > 0.0 && ducc_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && ducc_rt.cyc > 0.0;
    const double fwd_ratio = use_cyc ? fft_fwd.cyc / ducc_fwd.cyc : fft_fwd.us / ducc_fwd.us;
    const double rt_ratio = use_cyc ? fft_rt.cyc / ducc_rt.cyc : fft_rt.us / ducc_rt.us;
    const char* metric = use_cyc ? "cyc" : "wall";
    const double max_err =
        std::max(std::max(fft_fwd.err, fft_rt.err), std::max(ducc_fwd.err, ducc_rt.err));
    const bool unstable = max_err > bench::kStableMdape;
    const bool lose = !(fwd_ratio < 1.0 && rt_ratio < 1.0);

    const double flops = fft_flops(N);
    const double fft_gflops = flops / (fft_fwd.us * 1e3);
    const double ducc_gflops = flops / (ducc_fwd.us * 1e3);
    constexpr double peak = peak_flops_per_cycle<T>();
    const double fft_fpc = use_cyc ? flops / fft_fwd.cyc : 0.0;
    const double ducc_fpc = use_cyc ? flops / ducc_fwd.cyc : 0.0;

    std::cout << "CMP size=" << std::setw(5) << N
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << metric << " adm_t=" << adm_nt
              << (adm_eff == admiral::effort::measure    ? " adm_M"
                  : adm_eff == admiral::effort::automatic ? " adm_A"
                                                          : "")
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
        const bool ucf = use_cyc && fft_fwd.cyc > 0.0 && fftw_fwd.cyc > 0.0
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

template<typename T>
bool compare_factors_ab(std::size_t N,
                        const std::vector<unsigned>& fa,
                        const std::vector<unsigned>& fb,
                        int rounds, int reps, long inner,
                        double tol = default_accuracy_tol<T>()) {
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    std::vector<std::complex<T>> buf(N);
    volatile T sink = T(0);

    struct phase { std::vector<double> fs, rt, fd, sd; bool any_wall; };
    auto run_phase = [&](const std::vector<unsigned>& ff,
                         const std::vector<unsigned>& fs_) -> phase {
        const auto pf = make_dif_factor_plan(ff);
        const auto ps = make_dif_factor_plan(fs_);
        admiral::detail::plan_impl<T> f_fwd(N, true, 1, &pf), f_inv(N, false, 1, &pf);
        admiral::detail::plan_impl<T> s_fwd(N, true, 1, &ps), s_inv(N, false, 1, &ps);
        auto t_fwd = [&](admiral::detail::plan_impl<T>& p) {
            return nb_measure("ab_fwd", reps, inner, [&]() {
                std::copy(data.begin(), data.end(), buf.begin());
                p.execute(span(buf)); sink += buf[N / 2].real();
            });
        };
        auto t_rt = [&](admiral::detail::plan_impl<T>& pf2, admiral::detail::plan_impl<T>& pi) {
            return nb_measure("ab_rt", reps, inner, [&]() {
                std::copy(data.begin(), data.end(), buf.begin());
                pf2.execute(span(buf)); pi.execute(span(buf)); sink += buf[N / 2].real();
            });
        };
        for (int w = 0; w < 3; ++w) {
            std::copy(data.begin(), data.end(), buf.begin()); f_fwd.execute(span(buf)); f_inv.execute(span(buf));
            std::copy(data.begin(), data.end(), buf.begin()); s_fwd.execute(span(buf)); s_inv.execute(span(buf));
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

    {
        const auto pa = make_dif_factor_plan(fa);
        const auto pb = make_dif_factor_plan(fb);
        auto accurate = [&](const admiral::detail::dif_factor_plan& p) {
            admiral::detail::plan_impl<T> fp(N, true, 1, &p);
            std::copy(data.begin(), data.end(), buf.begin());
            fp.execute(span(buf));
            return l2_rel_error<T>(buf, reference_forward_dft<T>(data)) <= tol;
        };
        if (!accurate(pa) || !accurate(pb)) {
            std::cout << "ABCMP size=" << N << "  <== FAIL (a or b inaccurate, not timed)\n";
            return false;
        }
    }

    const phase ab = run_phase(fa, fb);
    const phase ba = run_phase(fb, fa);
    (void)sink;

    const double mab_f = geomean_of(ab.fs), mba_f = geomean_of(ba.fs);
    const double mab_r = geomean_of(ab.rt), mba_r = geomean_of(ba.rt);
    const double mfwd = std::sqrt(mab_f / mba_f);
    const double mrt  = std::sqrt(mab_r / mba_r);
    const double spread = std::max(mad_of(ab.fs), mad_of(ba.fs))
                        + 0.5 * std::abs(mab_f * mba_f - 1.0);
    const bool any_wall = ab.any_wall || ba.any_wall;
    const double a_ducc = std::sqrt(geomean_of(ab.fd) * geomean_of(ba.sd));
    const double b_ducc = std::sqrt(geomean_of(ab.sd) * geomean_of(ba.fd));

    const bool robust_a = (mfwd < 1.0 - 2.0 * spread) && (mrt < 1.0);
    const bool robust_b = (mfwd > 1.0 + 2.0 * spread) && (mrt > 1.0);
    const char* verdict = robust_a ? "A faster (robust)"
                        : robust_b ? "B faster (robust)"
                                   : "tie (within noise)";
    std::cout << "ABCMP size=" << std::setw(5) << N
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << (any_wall ? "WALL!" : "cyc") << std::fixed << std::setprecision(3)
              << " A=[" << join_radices(fa) << "] B=[" << join_radices(fb) << "]"
              << " | fwd A/B=" << std::setw(6) << mfwd
              << " (A/ducc=" << a_ducc << " B/ducc=" << b_ducc << ")"
              << " | rt A/B=" << std::setw(6) << mrt
              << " | rounds=" << rounds << "x2 spread=" << std::setprecision(1)
              << (spread * 100.0) << "%"
              << "  <== " << verdict
              << (any_wall ? "  [perf counters UNAVAILABLE: ratio is wall-clock, NOT trustworthy]" : "")
              << "\n";
    return true;
}

template<typename T>
bool verify_size(std::size_t N, double tol) {
    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    admiral::detail::plan_impl<T> fwd_plan(N, true);
    std::vector<std::complex<T>> buf(data);
    fwd_plan.execute(span(buf));
    const double l2err = l2_rel_error<T>(buf, reference_forward_dft<T>(data));
    const bool ok = l2err <= tol;
    std::cout << (ok ? "PASS " : "FAIL ")
              << "size=" << std::setw(5) << N
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " l2err=" << std::scientific << std::setprecision(2) << l2err
              << " tol=" << tol << std::defaultfloat << "\n";
    return ok;
}

template<typename T>
void codelet_sweep(int reps, long inner, bool with_ducc) {
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    const double tol = default_accuracy_tol<T>();
    for (const std::size_t N : admiral::detail::CODELET_CATALOG_SIZES) {
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
        const double metric = uc ? cs.cyc : cs.us;
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
                  << (cs.err > bench::kStableMdape ? ",UNSTABLE" : "")
                  << "\n";
    }
}

template<typename T>
void fs_split_sweep(const std::vector<std::size_t>& sizes, int reps) {
    namespace d = admiral::detail;
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    const double tol = default_accuracy_tol<T>();
    for (std::size_t N : sizes) {
        const d::four_step_split def = d::choose_four_step_split(N);
        if (!def.valid()) continue;
        std::vector<std::complex<T>> data(N);
        for (std::size_t i = 0; i < N; ++i)
            data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
        const auto ref = reference_forward_dft<T>(data);
        for (std::size_t n1 = 2; n1 <= N / 2; ++n1) {
            if (N % n1 != 0) continue;
            const std::size_t n2 = N / n1;
            if (n1 > 64 || n2 > 64) continue;
            if (!d::is_codelet_catalog(n1) || !d::is_codelet_catalog(n2)) continue;
            const auto tw = d::build_four_step_twiddles<T, true>(n1, n2);
            std::vector<std::complex<T>> out(N), G(N);
            d::four_step_execute<T, true>(data.data(), out.data(), n1, n2, tw.data(), G.data());
            const double l2 = l2_rel_error<T>(out, ref);
            if (!(l2 <= tol)) continue;
            volatile T sink = T(0);
            const NbStat s = nb_measure("fs", reps, 0, [&]() {
                d::four_step_execute<T, true>(data.data(), out.data(), n1, n2, tw.data(), G.data());
                sink += out[N / 2].real();
            });
            (void)sink;
            const bool uc = s.cyc > 0.0;
            std::cout << N << "," << prec << "," << n1 << "," << n2
                      << std::fixed << std::setprecision(1)
                      << "," << (uc ? s.cyc : s.us) << "," << s.us
                      << "," << (s.err * 100.0)
                      << "," << ((n1 == def.n1 && n2 == def.n2) ? 1 : 0) << "\n";
        }
    }
}

template<typename T>
void decomp_report(const std::vector<std::size_t>& sizes) {
    namespace d = admiral::detail;
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    for (std::size_t N : sizes) {
        if (N < 2) continue;
        d::plan_impl<T> plan(N, true);
        const char* route = plan.route_name();
        const d::four_step_split used = plan.four_step_split_used();

        constexpr double INF = 1e30;
        const d::four_step_split fs = d::choose_four_step_split(N);
        const double fs_cost = fs.valid() ? d::gate_four_step_cost(fs.n1, fs.n2) : INF;
        const bool smooth = admiral::detail::has_single_bit(N) || d::is_codelet_supported(N);
        const double dif_cost = smooth ? d::dif_model_cost(N) : INF;
        const double cod_cost = (N <= d::kFourStepLeafMax && d::is_codelet_catalog(N))
                                    ? d::gate_leaf_cyc(N) : INF;
        const double blue_cost = d::bluestein_model_cost(N);
        const bool prime = d::ct_is_prime(static_cast<unsigned>(N));
        const double rader_cost = (prime && d::rader_supported(N)) ? d::rader_model_cost(N) : INF;

        struct C { double c; const char* name; } cands[] = {
            {cod_cost, "codelet"}, {dif_cost, "iterative_dif"},
            {fs_cost, "four_step"}, {rader_cost, "rader"}, {blue_cost, "bluestein"}};
        const C* best = &cands[0];
        for (const C& c : cands) if (c.c < best->c) best = &c;

        const d::base_cost_entry meas = d::base_cost_for<T>(N);
        const bool table_routed = meas.cyc >= 0.f;
        const char* opt_route = best->name;
        double meas_cyc = -1.0;
        if (table_routed) {
            switch (meas.form) {
                case d::base_form::codelet:       opt_route = "codelet"; break;
                case d::base_form::iterative_dif: opt_route = "iterative_dif"; break;
                case d::base_form::good_thomas:   opt_route = "good_thomas"; break;
                case d::base_form::four_step:         opt_route = "four_step"; break;
                case d::base_form::four_step_batched: opt_route = "four_step_batched"; break;
                case d::base_form::rader:             opt_route = "rader"; break;
                case d::base_form::bluestein:         opt_route = "bluestein"; break;
            }
            meas_cyc = static_cast<double>(meas.cyc);
        }

        char split[48] = "-";
        if (std::string(route).rfind("four_step", 0) == 0 && used.valid())
            std::snprintf(split, sizeof split, "%zux%zu", used.n1, used.n2);
        char opt[48] = "-";
        if (std::string(opt_route) == "four_step" && fs.valid())
            std::snprintf(opt, sizeof opt, "%zux%zu", fs.n1, fs.n2);

        const bool mismatch = std::string(opt_route) != route
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

template<typename T>
void route_ab_dif_one(std::size_t N,
                      const std::vector<unsigned>& factors_override,
                      int rounds, int reps, long inner) {
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    const admiral::detail::dif_factor_plan dif_plan =
        factors_override.empty()
            ? admiral::detail::build_dif_factor_plan<T>(N)
            : make_dif_factor_plan(factors_override);

    const char* def_route = [&]() -> const char* {
        admiral::detail::plan_impl<T> probe(N, true);
        return probe.route_name();
    }();

    const std::string shape_str = std::to_string(N);
    volatile T sink = T(0);

    auto makeDefault = [&data, &sink, N]() {
        auto fwd = std::make_shared<admiral::detail::plan_impl<T>>(N, true);
        auto inv = std::make_shared<admiral::detail::plan_impl<T>>(N, false);
        auto buf = std::make_shared<std::vector<std::complex<T>>>(N);
        ab_engine e;
        e.fwd = [&data, &sink, fwd, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        e.rt = [&data, &sink, fwd, inv, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(span(*buf));
            inv->execute(span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        return e;
    };

    auto makeDif = [&data, &sink, N, &dif_plan]() {
        auto fwd = std::make_shared<admiral::detail::plan_impl<T>>(N, true,  1, &dif_plan);
        auto inv = std::make_shared<admiral::detail::plan_impl<T>>(N, false, 1, &dif_plan);
        auto buf = std::make_shared<std::vector<std::complex<T>>>(N);
        ab_engine e;
        e.fwd = [&data, &sink, fwd, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        e.rt = [&data, &sink, fwd, inv, buf]() {
            std::copy(data.begin(), data.end(), buf->begin());
            fwd->execute(span(*buf));
            inv->execute(span(*buf));
            sink += (*buf)[buf->size() / 2].real();
        };
        return e;
    };

    const std::string nameA = std::string("def[") + def_route + "]";
    engine_ab_core("RABDIF", shape_str, prec, nameA.c_str(), "dif",
                   makeDefault, makeDif, rounds, reps, inner);
    (void)sink;
}

template<typename T>
void base_cost_size(std::size_t N, int rounds, int reps, long inner) {
    namespace d = admiral::detail;
    using plan_t = d::plan_impl<T>;
    using rk     = typename plan_t::route_kind;
    const char* prec  = (sizeof(T) == 4) ? "f32" : "f64";
    constexpr double kFormDispatchTolF64 = 1e-10;
    const double tol = (sizeof(T) == 4) ? default_accuracy_tol<T>() : kFormDispatchTolF64;

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    const auto ref = reference_forward_dft<T>(data);

    struct Form { const char* name; rk kind; };
    static constexpr Form kForms[] = {
        {"codelet", rk::codelet},         {"iterative_dif", rk::iterative_dif},
        {"good_thomas", rk::good_thomas}, {"four_step", rk::four_step},
        {"four_step_batched", rk::four_step_batched},
        {"rader", rk::rader},             {"bluestein", rk::bluestein}};
    std::vector<Form> forms;
    for (const Form& f : kForms)
        if (plan_t::route_available(f.kind, N)) forms.push_back(f);

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
        pl->execute(span(*buf));
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

    for (auto& s : states) {
        for (int w = 0; w < 3; ++w) {
            std::copy(data.begin(), data.end(), s.buf->begin());
            s.plan->execute(span(*s.buf));
            sink += (*s.buf)[N / 2].real();
        }
    }

    std::vector<double> best_cyc(K, std::numeric_limits<double>::max());
    std::vector<double> best_us (K, std::numeric_limits<double>::max());
    std::vector<double> best_err(K, 1.0);

    for (int r = 0; r < rounds; ++r) {
        for (std::size_t ki = 0; ki < K; ++ki) {
            const std::size_t idx = (ki + static_cast<std::size_t>(r)) % K;
            auto& s = states[idx];
            const NbStat st = nb_measure("bc_fwd", reps, inner, [&]() {
                std::copy(data.begin(), data.end(), s.buf->begin());
                s.plan->execute(span(*s.buf));
                sink += (*s.buf)[N / 2].real();
            });
            if (st.cyc > 0.0 && st.cyc < best_cyc[idx]) {
                best_cyc[idx] = st.cyc;
                best_err[idx] = st.err;
            }
            best_us[idx] = std::min(best_us[idx], st.us);
        }
    }

    for (std::size_t i = 0; i < K; ++i) {
        const double cyc_out = best_cyc[i] >= std::numeric_limits<double>::max() ? 0.0 : best_cyc[i];
        const double ns_out  = best_us[i] * 1000.0;
        std::cout << "BASECOST size= " << N << " prec=" << prec
                  << " form=" << states[i].name
                  << " cyc= " << std::fixed << std::setprecision(1) << cyc_out
                  << " ns= " << std::setprecision(1) << ns_out
                  << " err= " << std::setprecision(3) << best_err[i]
                  << std::defaultfloat << "\n";
    }
    (void)sink;
}

template<typename T>
void chain_sweep(std::size_t N, int rounds, int reps, long inner, std::size_t max_chains) {
    using plan_t = admiral::detail::plan_impl<T>;
    namespace d      = admiral::detail;
    const char* prec = sizeof(T) == 8 ? "f64" : "f32";

    std::vector<unsigned> pool(d::dif_candidate_radices.begin(), d::dif_candidate_radices.end());
    pool.insert(pool.end(), d::dif_generic_radices.begin(), d::dif_generic_radices.end());
    std::vector<std::vector<unsigned>> chains;
    std::vector<unsigned>              cur;
    std::size_t                        enumerated = 0;
    auto rec = [&](auto&& self, std::size_t rest) -> void {
        if (rest == 1) {
            ++enumerated;
            if (const auto p = make_dif_factor_plan(cur); plan_t::dif_chain_shape_ok(N, p))
                chains.push_back(cur);
            return;
        }
        for (const unsigned r : pool)
            if (rest % r == 0) { cur.push_back(r); self(self, rest / r); cur.pop_back(); }
    };
    rec(rec, N);
    if (chains.empty()) {
        std::cout << "CHAINSWEEP size= " << N << " prec=" << prec << " <== no runnable chain\n";
        return;
    }
    const std::size_t found = chains.size();
    if (found > max_chains) {
        std::vector<std::vector<unsigned>> keep;
        for (std::size_t i = 0; i < max_chains; ++i) keep.push_back(chains[i * found / max_chains]);
        const d::dif_chain_list cands = d::dif_chain_candidates<T>(N);
        for (std::size_t ci = 0; ci < cands.count; ++ci) {
            const d::dif_factor_plan& fp = cands[ci];
            std::vector<unsigned> c;
            for (std::size_t i = 0; i < fp.count; ++i) c.push_back(static_cast<unsigned>(fp[i]));
            if (!c.empty() && std::find(keep.begin(), keep.end(), c) == keep.end()
                && std::find(chains.begin(), chains.end(), c) != chains.end())
                keep.push_back(std::move(c));
        }
        chains.swap(keep);
    }

    if (rounds == 0) {
        for (const auto& c : chains) {
            const auto            fp = make_dif_factor_plan(c);
            std::array<double, 9> col{};
            double                fitted = 0.0;
            std::size_t           n = N, veto = 0;
            for (const unsigned r : c) {
                const std::size_t ido = n / r;
                if (ido > 1 && ido < xsimd::batch<T>::size) ++veto;
                if constexpr (d::dif_surface_is_analytic<T>) {
                    const std::size_t idx = d::dif_cost_index(r);
                    const double      B =
                        static_cast<double>(d::dif_pass_footprint_bytes<T>(N, r, ido));
                    if (idx < d::dif_cost_radices.size()) {
                        const double A = d::dif_surface_t<T>::arith[idx];
                        if (ido == 1) { col[8] += 1.0; col[7] += A / double(N); }
                        else {
                            const double m =
                                B <= 2048.0 * 1024.0 ? d::dif_interior_kernel_mult(r) : 1.0;
                            col[0] += m;
                            col[1] += std::sqrt(A) * m;
                            if (ido < xsimd::batch<T>::size) col[2] += m / double(ido);
                            col[6] += 1.0 / double(N);
                        }
                        if (B > 48.0 * 1024.0 && B <= 2048.0 * 1024.0) col[3] += 1.0;
                        else if (B > 2048.0 * 1024.0) { col[4] += 1.0; col[5] += std::sqrt(A); }
                    }
                }
                n /= r;
            }
            if constexpr (d::dif_surface_is_analytic<T>)
                for (std::size_t i = 0; i < 9; ++i) fitted += d::dif_surface_t<T>::c[i] * col[i];
            const double model = d::dif_chain_cost<T>(N, fp) / double(N);
            std::cout << "CHAINMODEL size= " << N << " prec=" << prec
                      << " chain=" << join_radices(c) << " model= " << std::fixed
                      << std::setprecision(4) << model << " veto= " << veto << " res= "
                      << (d::dif_surface_is_analytic<T> ? model - fitted : 0.0) << " col=";
            for (const double v : col) std::cout << " " << std::setprecision(6) << v;
            std::cout << std::defaultfloat << "\n";
        }
        return;
    }

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
    const auto   ref = reference_forward_dft<T>(data);
    const double tol = default_accuracy_tol<T>();

    struct Arm {
        const std::vector<unsigned>*                  chain;
        std::unique_ptr<plan_t>                       plan;
        std::vector<std::complex<T>>                  buf;
        double                                        cyc = std::numeric_limits<double>::max();
        double                                        us  = std::numeric_limits<double>::max();
        double                                        err = 1.0;
    };
    std::vector<Arm> arms;
    arms.reserve(chains.size() + 1);
    auto add = [&](const std::vector<unsigned>& c) {
        const auto fp = make_dif_factor_plan(c);
        Arm        a{&c, std::make_unique<plan_t>(N, true, 1, &fp), std::vector<std::complex<T>>(N)};
        std::copy(data.begin(), data.end(), a.buf.begin());
        a.plan->execute(span(a.buf));
        if (l2_rel_error<T>(a.buf, ref) <= tol) arms.push_back(std::move(a));
    };
    for (const auto& c : chains) add(c);
    if (arms.empty()) {
        std::cout << "CHAINSWEEP size= " << N << " prec=" << prec << " <== all chains inaccurate\n";
        return;
    }
    add(*arms.front().chain);
    const std::size_t dup = arms.size() - 1;

    volatile T sink = T(0);
    for (auto& a : arms)
        for (int w = 0; w < 3; ++w) {
            std::copy(data.begin(), data.end(), a.buf.begin());
            a.plan->execute(span(a.buf));
            sink += a.buf[N / 2].real();
        }
    for (int r = 0; r < rounds; ++r)
        for (std::size_t k = 0; k < arms.size(); ++k) {
            Arm&         a  = arms[(k + static_cast<std::size_t>(r)) % arms.size()];
            const NbStat st = nb_measure("chain_fwd", reps, inner, [&]() {
                std::copy(data.begin(), data.end(), a.buf.begin());
                a.plan->execute(span(a.buf));
                sink += a.buf[N / 2].real();
            });
            if (st.cyc > 0.0 && st.cyc < a.cyc) { a.cyc = st.cyc; a.err = st.err; }
            a.us = std::min(a.us, st.us);
        }
    (void)sink;

    const double gamma = arms[dup].cyc > 0.0 ? arms[dup].cyc / arms[0].cyc : 0.0;
    std::cout << "CHAINSWEEP-ENV size= " << N << " prec=" << prec
              << " timed=" << arms.size() - 1 << " kept=" << chains.size()
              << " runnable=" << found << " enumerated=" << enumerated << " gamma=" << std::fixed
              << std::setprecision(3) << gamma << std::defaultfloat << "\n";
    for (std::size_t i = 0; i < dup; ++i)
        std::cout << "CHAIN size= " << N << " prec=" << prec
                  << " chain=" << join_radices(*arms[i].chain) << " cyc= " << std::fixed
                  << std::setprecision(1) << arms[i].cyc << " cpe= " << std::setprecision(4)
                  << arms[i].cyc / static_cast<double>(N) << " ns= " << std::setprecision(1)
                  << arms[i].us * 1000.0 << " err= " << std::setprecision(3) << arms[i].err
                  << " model= " << std::setprecision(1)
                  << d::dif_chain_cost<T>(N, make_dif_factor_plan(*arms[i].chain))
                  << std::defaultfloat << "\n";
}

template<typename T>
void model_dump(std::size_t lo, std::size_t hi) {
    using plan_t = admiral::detail::plan_impl<T>;
    namespace d      = admiral::detail;
    const char* prec = sizeof(T) == 8 ? "f64" : "f32";
    for (std::size_t N = lo; N <= hi; ++N) {
        auto emit = [&](const char* form, double cyc) {
            std::cout << "MODEL size= " << N << " prec=" << prec << " form=" << form
                      << " cyc= " << std::fixed << std::setprecision(2) << cyc
                      << std::defaultfloat << "\n";
        };
        if (N <= d::kFourStepLeafMax && d::is_codelet_catalog(N))
            emit("codelet", d::gate_leaf_cyc(N));
        if (plan_t::route_available(plan_t::route_kind::iterative_dif, N)) {
            emit("iterative_dif", d::dif_chain_cost<T>(N, d::dif_elected_chain<T>(N)));
        }
        if (const auto s = d::choose_four_step_split(N); s.valid())
            emit("four_step", d::gate_four_step_cost(s.n1, s.n2));
        if (d::rader_supported(N))
            emit("rader", d::rader_model_cost(N));
        emit("bluestein", d::bluestein_model_cost(N));
    }
}

}

int main(int argc, char** argv) {
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
                for (std::size_t n = 2; n <= 64; ++n) sizes.push_back(n);
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
                shapes = {{256, 256}, {512, 512}, {1024, 1024},
                          {64, 64, 64}, {128, 128, 128},
                          {60, 60}, {120, 120},
                          {16, 256}, {256, 16}, {1024, 64},
                          {8, 8, 8, 8}};
            bool ok = true;
            auto run = [&](auto tag) {
                using T = decltype(tag);
                for (const auto& shape : shapes)
                    ok = (robust
                              ? (r2c ? compare_nd_r2c_robust<T>(shape, rounds, reps, inner, nthreads)
                                     : compare_nd_robust<T>(shape, rounds, reps, inner, nthreads))
                              : (r2c ? compare_nd_r2c<T>(shape, reps, inner, nthreads)
                                     : compare_nd<T>(shape, reps, inner, nthreads))) && ok;
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return (fail_on_lose && !ok) ? 1 : 0;
        }
    }

    {
        bool compare = false;
        bool fail_on_lose = false;
        std::string cmp_prec = "both";
        int reps = 9;
        int nthreads = 1;
        int adm_nthreads = 1;
        admiral::effort adm_eff = admiral::effort::estimate;
        long inner = 0;
        double tol_override = -1.0;
        std::vector<std::size_t> sizes;
        std::vector<unsigned> factor_override;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--compare") compare = true;
            else if (arg == "--fail-on-lose") fail_on_lose = true;
            else if (arg.rfind("--prec=", 0) == 0) cmp_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--nthreads=", 0) == 0) nthreads = std::stoi(arg.substr(11));
            else if (arg.rfind("--adm-nthreads=", 0) == 0) adm_nthreads = std::stoi(arg.substr(15));
            else if (arg.rfind("--adm-effort=", 0) == 0) {
                const std::string v = arg.substr(13);
                adm_eff = v == "measure"                          ? admiral::effort::measure
                          : (v == "auto" || v == "automatic")     ? admiral::effort::automatic
                          : v == "estimate"                       ? admiral::effort::estimate
                                                                  : throw std::invalid_argument(
                                                                        "--adm-effort=" + v);
            }
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
                sizes = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,
                         3,5,7,11,13,17,31,
                         6,10,12,15,20,24,30,100,
                         36,48,60,90,120,210,360,720,1000,2520,
                         67,121,127,251};
            bool ok = true;
            auto run = [&](auto tag) {
                using T = decltype(tag);
                const double tol = tol_override < 0.0 ? default_accuracy_tol<T>() : tol_override;
                for (std::size_t N : sizes)
                    ok = compare_min_of_n<T>(N, reps, inner,
                                             factor_override.empty() ? nullptr : &factor_override,
                                             tol, nthreads, adm_nthreads, adm_eff) && ok;
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return (fail_on_lose && !ok) ? 1 : 0;
        }
    }

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
            const auto fa = parse_radix_list(ab_arg.substr(0, colon));
            const auto fb = parse_radix_list(ab_arg.substr(colon + 1));
            const auto prod = [](const std::vector<unsigned>& f) {
                std::size_t q = 1; for (unsigned r : f) q *= r; return q;
            };
            const std::size_t na = prod(fa), nb = prod(fb);
            if (na != nb || na == 0) {
                std::cerr << "--factors-ab: the two factorizations multiply to "
                          << na << " vs " << nb << ", must match\n";
                return 1;
            }
            auto run = [&](auto tag) {
                using T = decltype(tag);
                const double tol = tol_override < 0.0 ? default_accuracy_tol<T>() : tol_override;
                compare_factors_ab<T>(na, fa, fb, rounds, reps, inner, tol);
            };
            if (cmp_prec == "f64" || cmp_prec == "both") run(double{});
            if (cmp_prec == "f32" || cmp_prec == "both") run(float{});
            return 0;
        }
    }

#ifdef ADM_BENCH_FFTW
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
                    fwd.execute(span(buf));
                    fftw_c2c<T> fftw(N);
                    if (!fftw.alignment_ok(data)) {
                        std::cout << "FFTWAB size=" << N << " VOID (fftw alignment mismatch)\n";
                        continue;
                    }
                    double l2, fl2;
                    if (N <= kNaiveRefMaxN) {
                        const auto ref = reference_forward_dft<T>(data);
                        l2 = l2_rel_error<T>(buf, ref);
                        fl2 = l2_rel_error<T>(fftw.forward_into(data), ref);
                    } else {
                        std::vector<std::complex<double>> data_d(N);
                        for (std::size_t i = 0; i < N; ++i)
                            data_d[i] = std::complex<double>(static_cast<double>(data[i].real()),
                                                             static_cast<double>(data[i].imag()));
                        std::vector<std::complex<T>> rt(buf.begin(), buf.end());
                        inv.execute(span(rt));
                        l2 = l2_rel_error<T>(rt, data_d);
                        fl2 = l2_rel_error<T>(fftw.roundtrip(data), data_d);
                    }
                    if (!(l2 <= tol) || !(fl2 <= tol)) {
                        std::cout << "FFTWAB size=" << N << " ACCURACY FAIL (fft=" << l2
                                  << " fftw=" << fl2 << "), skipped\n";
                        continue;
                    }
                    volatile T sink = T(0);
                    auto t_fft_fwd = [&]() { return nb_measure("fab_f", reps, inner, [&]() {
                        fwd.execute(data.data(), buf.data()); sink += buf[N / 2].real(); }); };
                    auto t_fft_rt = [&]() { return nb_measure("fab_r", reps, inner, [&]() {
                        std::copy(data.begin(), data.end(), buf.begin());
                        fwd.execute(span(buf)); inv.execute(span(buf));
                        sink += buf[N / 2].real(); }); };
                    auto t_ftw_fwd = [&]() { return nb_measure("fab_wf", reps, inner, [&]() {
                        sink += fftw.forward_into(data)[N / 2].real(); }); };
                    auto t_ftw_rt = [&]() { return nb_measure("fab_wr", reps, inner, [&]() {
                        sink += fftw.roundtrip(data)[N / 2].real(); }); };
                    for (int w = 0; w < 3; ++w) {
                        std::copy(data.begin(), data.end(), buf.begin());
                        fwd.execute(span(buf)); inv.execute(span(buf));
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
                std::vector<unsigned> dp(p.radices.begin(),
                                         p.radices.begin()
                                             + static_cast<std::ptrdiff_t>(p.count));
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

    {
        std::string range, md_prec = "both";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--model-dump=", 0) == 0) range = arg.substr(13);
            else if (arg.rfind("--prec=", 0) == 0) md_prec = arg.substr(7);
        }
        if (!range.empty()) {
            const auto dash = range.find('-');
            const std::size_t lo = std::stoul(range.substr(0, dash));
            const std::size_t hi = dash == std::string::npos ? lo
                                                            : std::stoul(range.substr(dash + 1));
            if (md_prec == "f64" || md_prec == "both") model_dump<double>(lo, hi);
            if (md_prec == "f32" || md_prec == "both") model_dump<float>(lo, hi);
            return 0;
        }
    }

    {
        std::vector<std::size_t> sizes;
        std::string              cs_prec = "both";
        int                      reps = 9, rounds = 7;
        long                     inner = 0;
        std::size_t              max_chains = 96;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--chain-sweep=", 0) == 0) sizes = parse_size_list(arg.substr(14));
            else if (arg.rfind("--prec=", 0) == 0) cs_prec = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--rounds=", 0) == 0) rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--inner=", 0) == 0) inner = std::stol(arg.substr(8));
            else if (arg.rfind("--max-chains=", 0) == 0) max_chains = std::stoul(arg.substr(13));
        }
        if (!sizes.empty()) {
            for (const std::size_t N : sizes) {
                if (cs_prec == "f64" || cs_prec == "both") chain_sweep<double>(N, rounds, reps, inner, max_chains);
                if (cs_prec == "f32" || cs_prec == "both") chain_sweep<float>(N, rounds, reps, inner, max_chains);
            }
            return 0;
        }
    }

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
            const auto f = parse_radix_list(pass_arg);
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

    {
        bool ssweep = false;
        std::string sp = "both";
        int reps = 12;
        std::vector<std::size_t> sizes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--fs-split-sweep") ssweep = true;
            else if (arg.rfind("--prec=", 0) == 0) sp = arg.substr(7);
            else if (arg.rfind("--reps=", 0) == 0) reps = std::stoi(arg.substr(7));
            else if (arg.rfind("--sizes=", 0) == 0) sizes = parse_size_list(arg.substr(8));
        }
        if (ssweep) {
            if (sizes.empty())
                for (std::size_t n = 65; n <= 512; ++n) sizes.push_back(n);
            std::cout << "size,prec,n1,n2,cyc,us,err,def\n";
            if (sp == "f64" || sp == "both") fs_split_sweep<double>(sizes, reps);
            if (sp == "f32" || sp == "both") fs_split_sweep<float>(sizes, reps);
            return 0;
        }
    }

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

    {
        std::string bc_arg;
        std::string bc_prec = "both";
        std::string bc_out;
        int bc_rounds = 6;
        int bc_reps   = 9;
        long bc_inner = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if      (arg.rfind("--base-cost=", 0) == 0) bc_arg    = arg.substr(12);
            else if (arg.rfind("--out=",       0) == 0) bc_out    = arg.substr(6);
            else if (arg.rfind("--prec=",      0) == 0) bc_prec   = arg.substr(7);
            else if (arg.rfind("--rounds=",    0) == 0) bc_rounds = std::stoi(arg.substr(9));
            else if (arg.rfind("--reps=",      0) == 0) bc_reps   = std::stoi(arg.substr(7));
            else if (arg.rfind("--inner=",     0) == 0) bc_inner  = std::stol(arg.substr(8));
        }
        if (!bc_arg.empty()) {
            const auto sizes = parse_size_list(bc_arg);
            std::ofstream bc_file;
            std::streambuf* bc_saved = nullptr;
            if (!bc_out.empty()) {
                bc_file.open(bc_out);
                if (!bc_file) {
                    std::cerr << "cannot write " << bc_out << "\n";
                    return 1;
                }
                bc_saved = std::cout.rdbuf(bc_file.rdbuf());
            }
            auto run = [&](auto tag) {
                using T = decltype(tag);
                namespace d = admiral::detail;
                std::cout << "BASECOST-ENV arch=" << d::build_arch
                          << " compiler=" << d::build_compiler
                          << " major=" << d::build_compiler_major
                          << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
                          << " w=" << d::build_width<T>
                          << " regs=" << d::build_vector_regs
                          << " uarch=" << d::build_uarch << "\n";
                for (std::size_t N : sizes)
                    base_cost_size<T>(N, bc_rounds, bc_reps, bc_inner);
            };
            if (bc_prec == "f64" || bc_prec == "both") run(double{});
            if (bc_prec == "f32" || bc_prec == "both") run(float{});
            if (bc_saved) std::cout.rdbuf(bc_saved);
            return 0;
        }
    }

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
                constexpr long kProfItersTiny = 5000000L;
                constexpr long kProfItersSmall = 500000L;
                constexpr long kProfItersLarge = 100000L;
                prof_iters = prof_size <= 64 ? kProfItersTiny
                           : prof_size <= 1024 ? kProfItersSmall
                           : kProfItersLarge;
            }
            return (prof_prec == "f32" || prof_prec == "float")
                       ? profile_single_size<float>(prof_size, prof_iters)
                       : profile_single_size<double>(prof_size, prof_iters);
        }
    }

    std::cout << "FFT Benchmark: fft vs ducc0 Comparison\n";
    std::cout << "=====================================================================================\n";

    auto run_sweep = [](auto tag) {
        using T = decltype(tag);
        const char* prec_name = (sizeof(T) == 4) ? "float" : "double";
        std::cout << "\n#####################################################################################\n";
        std::cout << "# Precision: " << prec_name << "\n";
        std::cout << "#####################################################################################\n";
        std::cout << "  Size  Prec        Type |     FFT     |     FFT     |   ducc0     |   ducc0     | Fwd Ratio | Inv Ratio\n";
        std::cout << "                         |  Forward    |  Fwd+Inv    |  Forward    |  Fwd+Inv    | (fft/ducc)| (fft/ducc)\n";
        std::cout << "-------------------------------------------------------------------------------------\n";

        std::cout << "\nPower-of-2 sizes:\n";
        for (size_t N : {2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u})
            benchmark_size<T>(N, "pow2");

        std::cout << "\nPrime sizes:\n";
        for (size_t N : {3u, 5u, 7u, 11u, 13u, 17u, 31u})
            benchmark_size<T>(N, "prime");

        std::cout << "\nComposite sizes:\n";
        for (size_t N : {6u, 10u, 12u, 15u, 20u, 24u, 30u, 100u})
            benchmark_size<T>(N, "composite");

        std::cout << "\nMixed-radix (7-smooth) sizes:\n";
        for (size_t N : {36u, 48u, 60u, 90u, 120u, 210u, 360u, 720u, 1000u, 2520u})
            benchmark_size<T>(N, "7-smooth");

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
    std::cout << "  - xsimd kernels, cost-model routing, compiled codelets\n";
    std::cout << "  - Single-threaded here (threads are opt-in per plan)\n";
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

    print_performance_report();

    return 0;
}
