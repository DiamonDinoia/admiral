#pragma once

// ============================================================================
// Four-step (two-factor Cooley-Tukey) driver for N = N1 * N2 with BOTH factors
// in the compiled codelet catalog (N1, N2 <= 64). This is the decomposition that
// actually exploits the big spill-free codelets: instead of a chain of radix<=8
// DIF passes, a size-N transform is two batches of big-codelet transforms with a
// single twiddle multiply between them (e.g. 2048 = 32*64, 4096 = 64*64,
// 1024 = 32*32).
//
// Index map (decimation in time):
//   input   n = n2*N1 + n1   (n1 in [0,N1), n2 in [0,N2))
//   output  k = k1*N2 + k2   (k1 in [0,N1), k2 in [0,N2))
//   X[k1*N2+k2] = sum_{n1} W_N1^{n1 k1} * W_N^{n1 k2} *
//                 ( sum_{n2} x[n2*N1+n1] W_N2^{n2 k2} )
//   inner   : N1 size-N2 DFTs over the strided input  -> G[n1*N2 + k2]
//   twiddle : G[n1*N2+k2] *= W_N^{n1 k2}
//   outer   : N2 size-N1 DFTs over the G columns      -> X[k1*N2+k2]
//
// The leaf transforms call the compiled codelet_dispatch (any catalog size), so
// each leaf is the internally-SIMD straight-line codelet. Twiddles are plan-owned
// (built once, exact integer turn reduction). UN-normalized (inverse 1/N applied
// by the caller), matching the other drivers.
//
// ROUTING: for SMOOTH N this naive form is slower than iterative_dif —
// it gathers each leaf scalar-wise and re-pays codelet_dispatch's AoS<->SoA
// deinterleave on every one of the N1+N2 leaf calls, with a scalar std::complex
// twist. So plan.hpp keeps iterative_dif for the smooth sizes. But for
// the NON-11-smooth composites that would otherwise hit Bluestein (factors are
// two <=64 catalog codelets, e.g. 143=11x13, 289=17x17, 338=13x26), this
// COMPOSITION OF CODELETS beats Bluestein whenever its calibrated cost model
// (four_step_beats_bluestein, below) says so — Bluestein pads to
// next_pow2(2N-1) and runs 3 big pow2 FFTs, which is dearer than a couple dozen
// small spill-free codelets when the leaves are cheap or the pad jumps.
// select_route routes here when cost-gated.
//
// A fully SIMD-batched four-step (four_step_batched_ct below) would lift even the
// smooth path, but needs N1%W==0 && N2%W==0 (W-tail handling otherwise).
// ============================================================================

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "codelet.hpp"       // kernel_batched, xsimd::batch (batched leaves)
#include "math.hpp"          // codelet_dispatch, is_codelet_catalog
#include "portable_trig.hpp" // sincos_turns

namespace admiral {
namespace detail {

struct four_step_split {
    std::size_t n1 = 0;
    std::size_t n2 = 0;
    [[nodiscard]] constexpr bool valid() const { return n1 != 0 && n2 != 0; }
};

// Calibrated relative cost of a size-p catalog codelet: measured forward
// codelet_dispatch cycles. Index by p; entries 0/1 and any non-catalog p are
// unused (the four-step split only ever indexes catalog sizes). This is the
// planner's "compile-time wisdom" cost model — relative magnitudes (Rader primes
// are more expensive than similar-size composites) are what drive the
// four-step-vs-Bluestein decision; absolute machine cycles cancel in the comparison.
inline constexpr double codelet_cost_cyc[65] = {
    0, 0, 13.8, 45.0, 24.4, 62.9, 114.8, 74.7, 92.4, 103.8, 102.0, 102.6, 130.7,
    272.6, 133.1, 136.0, 119.1, 294.6, 193.1, 436.2, 168.0, 177.0, 185.1, 437.1,
    249.9, 182.9, 274.9, 265.8, 217.6, 496.6, 253.5, 629.4, 269.0, 286.4, 335.8,
    293.0, 337.8, 759.8, 470.5, 408.3, 331.7, 765.0, 375.7, 784.6, 388.2, 405.7,
    663.5, 1526.7, 402.8, 398.0, 384.4, 522.2, 431.0, 997.7, 575.4, 491.3, 505.7,
    691.6, 726.3, 1584.7, 500.8, 1124.1, 807.4, 574.8, 457.2};

// Modeled cycle cost of executing N as a four-step with leaves (n1,n2): the two
// leaf passes are n1 size-n2 codelets + n2 size-n1 codelets. The 1.13 factor
// absorbs the scalar gather/twist/transpose overhead.
[[nodiscard]] inline constexpr double four_step_cost(std::size_t n1, std::size_t n2) {
    return 1.13 * (double(n1) * codelet_cost_cyc[n2] + double(n2) * codelet_cost_cyc[n1]);
}

// Pick the COST-OPTIMAL two-factor split N = N1*N2 with both factors in the
// codelet catalog (<= 64), per the calibrated leaf-cost model (a balanced split
// is only optimal when leaves are equal-cost; an unbalanced split with a cheap
// composite leaf can win). Returns {0,0} if no such split exists.
[[nodiscard]] inline four_step_split choose_four_step_split(std::size_t N) {
    four_step_split best{};
    double best_cost = -1.0;
    for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
        if (N % n1 != 0) continue;
        const std::size_t n2 = N / n1;
        if (n1 > 64 || n2 > 64) continue;
        if (!is_codelet_catalog(n1) || !is_codelet_catalog(n2)) continue;
        const double c = four_step_cost(n1, n2);
        if (best_cost < 0.0 || c < best_cost) {
            best_cost = c;
            best = {n1, n2};
        }
    }
    return best;
}

// True iff N can be executed by a single (non-recursive) four-step.
[[nodiscard]] inline bool four_step_supported(std::size_t N) {
    return N > 64 && choose_four_step_split(N).valid();
}

// Cost-gated routing decision: prefer the four-step codelet composition over
// Bluestein (bluestein_model_cost) only when the model says it is cheaper.
[[nodiscard]] inline bool four_step_beats_bluestein(std::size_t N) {
    const four_step_split sp = choose_four_step_split(N);
    if (!sp.valid()) return false;
    return four_step_cost(sp.n1, sp.n2) < bluestein_model_cost(N);
}

// Plan-owned twiddle table W_N^{n1*k2}, n1 in [0,N1), k2 in [0,N2): N entries,
// laid out row-major [n1*N2 + k2]. Exact integer turn reduction.
template<typename T, bool Forward>
[[nodiscard]] inline std::vector<std::complex<T>>
build_four_step_twiddles(std::size_t N1, std::size_t N2) {
    const std::size_t N = N1 * N2;
    std::vector<std::complex<T>> tw(N);
    for (std::size_t n1 = 0; n1 < N1; ++n1) {
        for (std::size_t k2 = 0; k2 < N2; ++k2) {
            const auto [sn, cs] = portable_trig::sincos_turns<Forward>(n1 * k2, N);
            tw[n1 * N2 + k2] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
    }
    return tw;
}

// Execute the four-step on AoS complex<T>[N1*N2]. in == out is in-place;
// in != out reads `in` (fully consumed by the inner pass before the outer pass
// writes `out`). `G` is caller-owned scratch of length N1*N2. UN-normalized.
template<typename T, bool Forward>
void four_step_execute(const std::complex<T>* in, std::complex<T>* out,
                       std::size_t N1, std::size_t N2,
                       const std::complex<T>* tw, std::complex<T>* G) {
    std::complex<T> tmp[64];  // leaf buffer; N1,N2 <= 64 by construction

    // inner: N1 size-N2 DFTs over strided input, then twiddle into G.
    for (std::size_t n1 = 0; n1 < N1; ++n1) {
        for (std::size_t n2 = 0; n2 < N2; ++n2) tmp[n2] = in[n2 * N1 + n1];
        codelet_dispatch<T, Forward>(tmp, tmp, N2);
        std::complex<T>* Grow = G + n1 * N2;
        const std::complex<T>* twrow = tw + n1 * N2;
        for (std::size_t k2 = 0; k2 < N2; ++k2) Grow[k2] = tmp[k2] * twrow[k2];
    }

    // outer: N2 size-N1 DFTs over the G columns, scatter to output.
    for (std::size_t k2 = 0; k2 < N2; ++k2) {
        for (std::size_t n1 = 0; n1 < N1; ++n1) tmp[n1] = G[n1 * N2 + k2];
        codelet_dispatch<T, Forward>(tmp, tmp, N1);
        for (std::size_t k1 = 0; k1 < N1; ++k1) out[k1 * N2 + k2] = tmp[k1];
    }
}

// ============================================================================
// BATCHED four-step (the competitive form). The N1 inner / N2 outer leaf
// transforms are run W at a time across SIMD lanes via kernel_batched<N>, so the
// DFT work is W-wide instead of N1+N2 scalar codelet_dispatch calls. Data is kept
// planar (split re/im); the only scalar O(N) traffic is the entry deinterleave,
// the single transpose-store between passes (inherent to four-step), and the exit
// reinterleave. Requires N1 % W == 0 and N2 % W == 0 (both leaf sizes a multiple
// of the SIMD width) so every group is full — guarded by four_step_batched_supported.
//
// Lane mapping (W = V::size):
//   inner: group over n1 (lanes = base..base+W-1). input layout n = n2*N1+n1 has
//          n1 contiguous, so the W-lane load is contiguous. After the size-N2 DFT
//          and the per-lane W_N^{n1 k2} twist, lane l (n1=base+l) is scattered to
//          G[(base+l)*N2 + k2]  (the transpose; strided store).
//   outer: group over k2 (lanes = base2..base2+W-1). G layout [n1*N2+k2] has k2
//          contiguous, so the W-lane load is contiguous; the size-N1 DFT output
//          stores contiguously to out[k1*N2 + base2..].
// ============================================================================

// Per-lane twiddle table in V-contiguous layout: entry [(g*N2 + k2)*W + l] holds
// W_N^{(g*W+l)*k2}, so the inner pass loads one contiguous V per (group g, k2).
// N1 must be a multiple of W (the caller guarantees this).
template<typename T, bool Forward>
[[nodiscard]] inline std::vector<std::complex<T>>
build_four_step_twiddles_v(std::size_t N1, std::size_t N2, std::size_t W) {
    const std::size_t N = N1 * N2;
    std::vector<std::complex<T>> tw(N);
    const std::size_t groups = N1 / W;
    for (std::size_t g = 0; g < groups; ++g) {
        for (std::size_t k2 = 0; k2 < N2; ++k2) {
            for (std::size_t l = 0; l < W; ++l) {
                const std::size_t n1 = g * W + l;
                const auto [sn, cs] = portable_trig::sincos_turns<Forward>(n1 * k2, N);
                tw[(g * N2 + k2) * W + l] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
            }
        }
    }
    return tw;
}

// Compile-time-sized batched four-step on planar buffers. in/out are length-N
// planar (re,im); Gre/Gim are length-N planar scratch; twvre/twvim are the
// V-contiguous twiddle planes from build_four_step_twiddles_v. out may alias in.
template<unsigned N1, unsigned N2, typename T, bool Forward, typename V = xsimd::batch<T>>
void four_step_batched_ct(const T* in_re, const T* in_im, T* out_re, T* out_im,
                          const T* twvre, const T* twvim, T* Gre, T* Gim) {
    constexpr unsigned W = static_cast<unsigned>(V::size);
    // Parametric width gate: this batched four-step needs both leaves a multiple
    // of the SIMD width. The dispatch switch below instantiates every (N1,N2) in
    // the split table regardless of W, so a hard static_assert would break the
    // build at wider W (e.g. W=16 on AVX-512, where the W=8-tuned splits 8/24/40
    // are not multiples of 16). Guard the body instead: width-incompatible
    // instantiations become an empty no-op that is never selected at runtime
    // (four_step_batched_supported applies the same n%W==0 gate). This is
    // width-generic — the same guard admits AVX2 (W=8), SSE (W=4), NEON, etc.
    if constexpr (N1 % W == 0 && N2 % W == 0) {
    // INNER: N1 size-N2 DFTs, W columns (n1) per group; twist + transpose to G.
    // The W-lane scatter to G is the four-step transpose; we tile k2 into W-wide
    // WxW tiles and transpose each in registers (vunpck/vperm). N2 % W == 0 by
    // static_assert, so every tile is full (no scalar remainder).
    for (unsigned base = 0; base < N1; base += W) {
        V xv[N2], iv[N2], ov[N2], oi[N2];
        for (unsigned n2 = 0; n2 < N2; ++n2) {
            xv[n2] = V::load_unaligned(in_re + n2 * N1 + base);
            iv[n2] = V::load_unaligned(in_im + n2 * N1 + base);
        }
        kernel_batched<N2, T, Forward, V>::apply(xv, iv, 1, ov, oi);
        const unsigned g = base / W;
        // Twist in place: lane l (n1 = base+l) of ov[k2] *= W_N^{(base+l)*k2}.
        for (unsigned k2 = 0; k2 < N2; ++k2) {
            const unsigned tvi = (g * N2 + k2) * W;
            const V wr = V::load_unaligned(twvre + tvi);
            const V wi = V::load_unaligned(twvim + tvi);
            const V gr = ov[k2] * wr - oi[k2] * wi;
            const V gi = ov[k2] * wi + oi[k2] * wr;
            ov[k2] = gr;
            oi[k2] = gi;
        }
        // Transpose-scatter: each WxW tile {ov[k0..k0+W)} has row=k2, lane=n1;
        // after transpose ov[k0+a] holds (n1=base+a, k2=k0..k0+W) across lanes —
        // a contiguous run into G row (base+a). vunpck/vperm, NOT vmaskmov/vgather.
        for (unsigned k0 = 0; k0 < N2; k0 += W) {
            xsimd::transpose(ov + k0, ov + k0 + W);
            xsimd::transpose(oi + k0, oi + k0 + W);
            for (unsigned a = 0; a < W; ++a) {
                ov[k0 + a].store_unaligned(Gre + (base + a) * N2 + k0);
                oi[k0 + a].store_unaligned(Gim + (base + a) * N2 + k0);
            }
        }
    }

    // OUTER: N2 size-N1 DFTs, W frequencies (k2) per group; contiguous in and out.
    for (unsigned base2 = 0; base2 < N2; base2 += W) {
        V xv[N1], iv[N1], ov[N1], oi[N1];
        for (unsigned n1 = 0; n1 < N1; ++n1) {
            xv[n1] = V::load_unaligned(Gre + n1 * N2 + base2);
            iv[n1] = V::load_unaligned(Gim + n1 * N2 + base2);
        }
        kernel_batched<N1, T, Forward, V>::apply(xv, iv, 1, ov, oi);
        for (unsigned k1 = 0; k1 < N1; ++k1) {
            ov[k1].store_unaligned(out_re + k1 * N2 + base2);
            oi[k1].store_unaligned(out_im + k1 * N2 + base2);
        }
    }
    }  // if constexpr (N1 % W == 0 && N2 % W == 0)
}

// ============================================================================
// Batched-four-step ROUTE (f32 only). For f32 the batched four-step runs two
// full-width kernel_batched passes on N in {128,256,384,448,512,640,768}, where
// iterative_dif's early/late radix passes underutilize the 8-wide f32 register.
// f64 table is empty (W=4 and W=8 both fall back to iterative_dif).
// ============================================================================

// Optimal split per N (f32 only, width-keyed). {0,0} ⇒ not a
// batched-four-step size.
//
// W == 4 (e.g. SSE2/NEON f32): route is off; returns {} for all N.
//
// W == 8: small-band table only (128–768). At W=16 these splits are not
//   multiples of 16, so the route is off above W=8 too.
// W==8 small-band splits (128–768). Single source of truth for both the split
// lookup below and the runtime→compile-time dispatch further down, so adding a
// size is a one-line edit both consumers pick up (no dual-maintenance drift).
inline constexpr std::array<four_step_split, 7> fsb_splits{{
    {8, 16}, {16, 16}, {16, 24}, {8, 56}, {16, 32}, {16, 40}, {16, 48},
}};

template<typename T>
[[nodiscard]] constexpr four_step_split fsb_split_for([[maybe_unused]] std::size_t N) {
    if constexpr (sizeof(T) == 4) {
        constexpr std::size_t W = xsimd::batch<T>::size;
        if constexpr (W != 8) return {};
        for (const four_step_split s : fsb_splits)
            if (s.n1 * s.n2 == N) return s;
    }
    return {};
}

template<typename T>
[[nodiscard]] constexpr bool four_step_batched_supported(std::size_t N) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const four_step_split sp = fsb_split_for<T>(N);
    // Only select the route where both leaves are a multiple of the active SIMD
    // width (see four_step_batched_ct's parametric guard). The current split
    // table is W=8-tuned, so on AVX-512 (W=16) these fall back to iterative_dif
    // until the table is re-tuned per-ISA.
    return sp.valid() && (sp.n1 % W == 0) && (sp.n2 % W == 0);
}

// Largest batched-four-step N (sizes the execute() stack scratch must hold).
inline constexpr std::size_t FSB_MAX_N = 768;

// Expand fsb_splits into one guarded four_step_batched_ct call per entry.
template<typename T, bool Forward, std::size_t... I>
inline void fsb_dispatch_pack(std::index_sequence<I...>, unsigned N1, unsigned N2,
        const T* ire, const T* iim, T* ore, T* oim,
        const T* twre, const T* twim, T* Gre, T* Gim) {
    (((N1 == fsb_splits[I].n1 && N2 == fsb_splits[I].n2)
          ? four_step_batched_ct<static_cast<unsigned>(fsb_splits[I].n1),
                                 static_cast<unsigned>(fsb_splits[I].n2), T, Forward>(
                ire, iim, ore, oim, twre, twim, Gre, Gim)
          : void()),
     ...);
}

// Runtime (N1,N2) -> compile-time four_step_batched_ct instantiation. Only the
// f32 measured splits are instantiated (the whole body is if-constexpr'd away for
// f64, so no four_step_batched_ct<...,double> is ever generated).
template<typename T, bool Forward>
inline void four_step_batched_dispatch([[maybe_unused]] unsigned N1, [[maybe_unused]] unsigned N2,
        [[maybe_unused]] const T* ire, [[maybe_unused]] const T* iim,
        [[maybe_unused]] T* ore, [[maybe_unused]] T* oim,
        [[maybe_unused]] const T* twre, [[maybe_unused]] const T* twim,
        [[maybe_unused]] T* Gre, [[maybe_unused]] T* Gim) {
    if constexpr (sizeof(T) == 4) {
        // Match (N1,N2) against fsb_splits and instantiate that leaf pair. Free
        // helper (no capturing lambda: an IILE closure materializes on the stack).
        fsb_dispatch_pack<T, Forward>(std::make_index_sequence<fsb_splits.size()>{},
                                      N1, N2, ire, iim, ore, oim, twre, twim, Gre, Gim);
    }
}

// Plan-owned batched-four-step state: the chosen split + direction-specific
// V-contiguous planar twiddles (built once). execute() is a drop-in on
// interleaved complex: deinterleave -> batched four-step -> reinterleave.
// UN-normalized (plan_impl applies 1/N for inverse), matching the other routes.
template<typename T>
struct four_step_batched_plan {
    unsigned n1 = 0, n2 = 0;
    bool forward = true;
    std::vector<T> twre, twim;  // V-contiguous planar twiddles for `forward`
    // Band-B sizes (N > FSB_MAX_N): one plan-owned block carved into are/aim/
    // Gre/Gim — a per-execute malloc+first-touch would eat the four-step
    // win. Mutable ⇒ concurrent execute() on ONE plan instance races (same
    // discipline as the Bluestein/Rader plan-owned scratch; distinct plans are
    // independent).
    mutable std::vector<T> big_scratch;

    four_step_batched_plan(std::size_t N, bool fwd) : forward(fwd) {
        const four_step_split sp = fsb_split_for<T>(N);
        n1 = static_cast<unsigned>(sp.n1);
        n2 = static_cast<unsigned>(sp.n2);
        constexpr unsigned W = static_cast<unsigned>(xsimd::batch<T>::size);
        const auto tw = fwd ? build_four_step_twiddles_v<T, true>(n1, n2, W)
                            : build_four_step_twiddles_v<T, false>(n1, n2, W);
        twre.resize(N);
        twim.resize(N);
        for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
        if (N > FSB_MAX_N) big_scratch.resize(4 * N);
    }

    // in == out is in-place; in != out reads `in` (preserved, consumed by the
    // entry deinterleave) and writes `out` at the exit reinterleave.
    void execute(const std::complex<T>* in, std::complex<T>* out) const {
        const std::size_t N = std::size_t(n1) * n2;
        if (N > FSB_MAX_N) {
            T* are = big_scratch.data();
            run(in, out, are, are + N, are + 2 * N, are + 3 * N, N);
            return;
        }
        // Stack planar scratch (N <= FSB_MAX_N): in/out alias (are,aim), G scratch.
        alignas(xsimd::batch<T>::arch_type::alignment()) T are[FSB_MAX_N], aim[FSB_MAX_N], Gre[FSB_MAX_N], Gim[FSB_MAX_N];
        run(in, out, are, aim, Gre, Gim, N);
    }

private:
    void run(const std::complex<T>* in, std::complex<T>* out,
             T* are, T* aim, T* Gre, T* Gim, std::size_t N) const {
        for (std::size_t i = 0; i < N; ++i) { are[i] = in[i].real(); aim[i] = in[i].imag(); }
        if (forward)
            four_step_batched_dispatch<T, true>(n1, n2, are, aim, are, aim, twre.data(), twim.data(), Gre, Gim);
        else
            four_step_batched_dispatch<T, false>(n1, n2, are, aim, are, aim, twre.data(), twim.data(), Gre, Gim);
        for (std::size_t i = 0; i < N; ++i) out[i] = std::complex<T>(are[i], aim[i]);
    }
};

} // namespace detail
} // namespace admiral
