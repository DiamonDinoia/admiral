#pragma once

#include "bench_harness.hpp"

#include <admiral/detail/dif_passes.hpp>
#include <admiral/detail/twiddles.hpp>

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
struct bench_mid {
    const T* ccre; const T* ccim; T* chre; T* chim;
    std::size_t l1, ido; const T* twre; const T* twim;
    template<std::size_t IPv>
    void operator()() const {
        admiral::detail::dif_pass<T, IPv>(ccre, ccim, chre, chim, l1, ido, twre, twim,
                                          1, 1);
    }
};

template<typename T>
struct bench_lst {
    const T* ccre; const T* ccim; std::complex<T>* out;
    std::size_t l1; const T* twre; const T* twim;
    template<std::size_t IPv>
    void operator()() const {
        admiral::detail::dif_pass_last<T, true, IPv>(ccre, ccim, out, l1, 1, twre, twim);
    }
};

template<typename T>
struct bench_call {
    const bench_mid<T>* mid; const bench_lst<T>* lst; bool last;
    template<std::size_t IPv>
    bool operator()() const {
        if (last) (*lst).template operator()<IPv>();
        else (*mid).template operator()<IPv>();
        return true;
    }
};

template<typename T>
void pass_microbench(unsigned IP, std::size_t ido, std::size_t l1, bool last,
                     int reps, long inner, long perf_iters) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t span = static_cast<std::size_t>(IP) * ido * l1;
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
    const bench_mid<T> mid{ccre.data(), ccim.data(), chre.data(), chim.data(),
                           l1, ido, twre.data(), twim.data()};
    const bench_lst<T> lst{ccre.data(), ccim.data(), out.data(), l1, twre.data(), twim.data()};
    auto call = [&]() {
        const bool matched = poet::dispatch(
            bench_call<T>{&mid, &lst, last},
            poet::dispatch_param<admiral::detail::dif_radix_set>{IP});
        if (!matched) {
            std::cerr << "--pass: unsupported radix " << IP << "\n";
            return;
        }
        sink += last ? out[span / 2].real() : chre[span / 2];
    };
    if (perf_iters > 0) {
        constexpr long kWarmReps = 200;
        for (long i = 0; i < kWarmReps; ++i) call();
        for (long i = 0; i < perf_iters; ++i) call();
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

}
