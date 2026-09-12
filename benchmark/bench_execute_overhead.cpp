#include <admiral/admiral.hpp>

#include <ducc0/fft/fft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "bench_harness.hpp"

namespace {

std::string shape_to_string(const std::vector<std::size_t>& shape) {
    std::string s;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) s += 'x';
        s += std::to_string(shape[i]);
    }
    return s;
}

}

namespace bench {

// Per-execute overhead for tiny N-D (bucket A): every existing timed mode in this tree reuses a
// constructed plan but folds fixed dispatch cost and kernel cost into one wall-time number. This
// target reports the one unit that is exact and comparable across shapes: retired instructions
// per p.forward() call. Cycles ride along for cross-reference against wall-clock claims.
template<typename T>
bool bench_execute_overhead(const std::vector<std::size_t>& shape, int reps, long inner) {
    std::size_t Ntot = 1;
    for (auto e : shape) Ntot *= e;
    std::vector<std::complex<T>> in(Ntot), out(Ntot), ref(Ntot);
    for (std::size_t i = 0; i < Ntot; ++i)
        in[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    const admiral::plan<T> p(admiral::span<const std::size_t>(shape.data(), shape.size()),
                             {.nthreads = 1, .eff = admiral::effort::measure});

    volatile T sink = T(0);
    const auto call_once = [&]() {
        p.forward(in.data(), out.data());
        sink += out[Ntot / 2].real();
    };
    const NbStat call = nb_measure("exec_overhead", reps, inner, call_once);

    // Same-binary control: the identical call, measured ten times with the arm order rotated.
    // Retired instructions along a fixed path are deterministic, so an A/A instruction RATIO is
    // structurally 1.0 and proves nothing. The control is that instr is bit-stable round to round
    // (spread 0) while cyc is not; the cycle spread is the floor for any cycle ratio read here.
    double imin = 0.0, imax = 0.0, cmin = 0.0, cmax = 0.0;
    for (int r = 0; r < 10; ++r) {
        const NbStat s = nb_measure(r & 1 ? "ctrl_b" : "ctrl_a", reps, inner, call_once);
        if (r == 0) { imin = imax = s.instr; cmin = cmax = s.cyc; continue; }
        imin = std::min(imin, s.instr); imax = std::max(imax, s.instr);
        cmin = std::min(cmin, s.cyc);   cmax = std::max(cmax, s.cyc);
    }
    const double instr_spread = imin > 0.0 ? imax / imin - 1.0 : 0.0;
    const double cyc_spread = cmin > 0.0 ? cmax / cmin - 1.0 : 0.0;

    // Positive control: `out` must match ducc0's reference within the precision's tolerance,
    // so a broken route or a stale scratch buffer turns `correct` false, not silently 1.0.
    ducc0::fmav_info::shape_t sh(shape.begin(), shape.end()), axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view = ducc0::cfmav<std::complex<T>>(in.data(), sh);
    auto ref_view = ducc0::vfmav<std::complex<T>>(ref.data(), sh);
    ducc0::c2c(in_view, ref_view, axes, true, T(1), std::size_t{1});
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < Ntot; ++i) {
        num += std::norm(static_cast<std::complex<double>>(out[i] - ref[i]));
        den += std::norm(static_cast<std::complex<double>>(ref[i]));
    }
    const double l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    const bool correct = l2 <= default_accuracy_tol<T>();

    std::cout << "OVERHEAD " << std::setw(12) << shape_to_string(shape)
              << " (N=" << std::setw(6) << Ntot << ")"
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << std::fixed
              << " call_cyc=" << std::setprecision(1) << std::setw(9) << call.cyc
              << " call_instr=" << std::setprecision(1) << std::setw(9) << call.instr
              << " instr_spread=" << std::setprecision(4) << instr_spread
              << " cyc_spread=" << std::setprecision(4) << cyc_spread
              << " l2err=" << std::scientific << std::setprecision(2) << l2
              << (correct ? "" : "  <== CORRECTNESS CONTROL FAILED")
              << "\n";
    return correct;
}

template bool bench_execute_overhead<float>(const std::vector<std::size_t>&, int, long);
template bool bench_execute_overhead<double>(const std::vector<std::size_t>&, int, long);

}
