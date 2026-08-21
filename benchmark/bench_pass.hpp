#pragma once

// The --pass single-pass microbench. Instantiating dif_pass / dif_pass_last over the
// whole radix set dominates a benchmarks-ON build, so the definition lives here and
// each precision gets its own TU (bench_pass_{f,d}.cpp).
#include "bench_harness.hpp"

#include <admiral/detail/dif_passes.hpp>  // dif_pass, dif_pass_last
#include <admiral/detail/twiddles.hpp>    // dif_radix_set

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <vector>

#include <poet/poet.hpp>
#include "admiral/detail/simd.hpp"

namespace bench {

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
    auto mid = [&]<std::size_t IPv>() {
        // A middle SoA pass carries no direction. The inverse rides the same code in
        // swapped domain (butterfly.hpp). Only the boundary passes still take Forward.
        admiral::detail::dif_pass<T, IPv>(ccre.data(), ccim.data(), chre.data(),
                                          chim.data(), l1, ido, twre.data(), twim.data(),
                                          1, 1);  // contiguous SoA: unit element strides
    };
    auto lst = [&]<std::size_t IPv>() {
        admiral::detail::dif_pass_last<T, true, IPv>(ccre.data(), ccim.data(), out.data(),
                                                 l1, 1, twre.data(), twim.data());
    };
    auto call = [&]() {
        // The engine's own radix set and dispatch: --pass measures what ships. An
        // unsupported radix is a command-line error to report, not an exception.
        const bool matched = poet::dispatch(
            [&]<std::size_t IPv>() {
                if (last) lst.template operator()<IPv>();
                else mid.template operator()<IPv>();
                return true;
            },
            poet::dispatch_param<admiral::detail::dif_radix_set>{IP});
        if (!matched) {
            std::cerr << "--pass: unsupported radix " << IP << "\n";
            return;
        }
        sink += last ? out[span / 2].real() : chre[span / 2];
    };
    if (perf_iters > 0) {
        // Enough to fault the buffers in and settle the frequency before perf attaches;
        // the measured loop below is what an external `perf stat` counts.
        constexpr long kWarmReps = 200;
        for (long i = 0; i < kWarmReps; ++i) call();
        for (long i = 0; i < perf_iters; ++i) call();        // measured by external perf
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
}

}  // namespace bench
