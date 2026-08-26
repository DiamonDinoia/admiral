#pragma once

// ============================================================================
// Generic compile-time-recursive 1D DFT codelet: kernel<N, T, Forward>. N factors at
// compile time into radix r and cofactor M = N/r; a size-N transform composes r
// sub-transforms of size M plus a radix-r butterfly. Layout is PLANAR (split re/im);
// the std::complex AoS API de-interleaves on entry and re-interleaves on exit.
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <poet/poet.hpp>
#include "simd.hpp"

#include <admiral/detail/codelet_max.hpp>  // CODELET_CATALOG_MAX
#include "cxx_compat.hpp"  // ADM_CONSTEVAL, detail::bit_width, detail::has_single_bit

#include "butterfly.hpp"  // dif_butterfly (symmetric odd-radix + Cooley-Tukey DIF pow2)
#include "ct_math.hpp"  // ct_sincos_t/turns, smallest_radix, codelet_radix
#include "simd_swizzle.hpp"  // lane_lt (shared compile-time lane predicate)
#include "macros.hpp"

namespace admiral {
namespace detail {

// ----------------------------------------------------------------------------
// Combine stage (forward-only; see kernel<N,T,false>) for compile-time N = R*M.
//
// Given planar (yre, yim) with R sub-DFTs of length M (block q at [q*M,(q+1)*M)),
// for each j in [0, M):
//   (1) inter-stage twiddle  t_q = W_N^{q j} * y[q*M + j]   (q=0: trivial),
//   (2) radix-R DFT:         out_l = sum_q W_R^{q l} t_q,
//   (3) emit sink(l*M + j, out_r, out_i)  (default sink: store to yre/yim).
//
// Arithmetic is split-real/imag SoA. V is an xsimd::batch (W consecutive j)
// or T (scalar, j compile-time). Scalar chunks constant-fold the twiddle VALUE
// (apply_stage_twiddle: W==1 skip, W==+-i swap/negate, diagonal shared factor)
// from the same ct_sincos_turns the table bakes; vector chunks load the table.
// ----------------------------------------------------------------------------

// Consteval inter-stage twiddle table for compile-time N, radix R:
// W_N^{q j} = exp(sign 2*pi*i q j / N), layout (q-1)*M+j, q in [1,R), j in [0,M=N/R).
// Part selects real (false) or imag (true). No runtime trig, no heap.
template<unsigned N, unsigned R, typename T, bool Imag>
ADM_CONSTEVAL std::array<T, (R - 1) * (N / R)> make_twiddle_table() {
    std::array<T, (R - 1) * (N / R)> a{};
    constexpr unsigned M = N / R;
    for (unsigned q = 1; q < R; ++q) {
        for (unsigned j = 0; j < M; ++j) {
            const ct_sincos_t w = ct_sincos_turns(/*conjugate=*/true, q * j, N);
            a[(q - 1) * M + j] = Imag ? static_cast<T>(w.s) : static_cast<T>(w.c);
        }
    }
    return a;
}

// sink(std::size_t p, V outr, V outi) storing to yre/yim (unit element base p).
template<typename T>
struct yre_sink {
    T* yre;
    T* yim;
    template<typename V>
    ADM_ALWAYS_INLINE void operator()(std::size_t p, V outr, V outi) const {
        if constexpr (std::is_same_v<V, T>) {
            yre[p] = outr;
            yim[p] = outi;
        } else {
            outr.store_unaligned(yre + p);
            outi.store_unaligned(yim + p);
        }
    }
};

template<unsigned R, typename T, typename V, typename Sink>
ADM_ALWAYS_INLINE void bfly_chunk(const T* ADM_RESTRICT twre,
                                  const T* ADM_RESTRICT twim,
                                  const T* yre, const T* yim,
                                  std::size_t M, std::size_t j, Sink&& sink) {
    // Gather one sample per sub-block, applying the inter-stage twiddle.
    V tr[R], ti[R];
    poet::static_for<0, R>([&](const auto q) {
        const V ar = V::load_unaligned(yre + q * M + j);
        const V ai = V::load_unaligned(yim + q * M + j);
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const V wr = V::load_unaligned(twre + (q - 1) * M + j);
            const V wi = V::load_unaligned(twim + (q - 1) * M + j);
            tr[q] = wr * ar - wi * ai;
            ti[q] = wr * ai + wi * ar;
        }
    });

    // Radix-R DFT via the UNIFIED butterfly: odd R -> radix_sym_dft (halves
    // multiplies), pow2 R -> pow2_dif_butterfly (twiddle elision), else naive.
    // emit carries output index l, preserving sink(l*M + j) regardless of order.
    dif_butterfly<T, R, V>(tr, ti,
        [&](auto L, V outr, V outi) { sink(L * M + j, outr, outi); });
}

// Scalar combine chunk at compile-time j: twiddles fold by value, no table.
// Inputs are NOT restrict: the cofactor combine reads rows stored moments
// earlier in the same inlined region; restrict lets GCC reorder loads above stores.
template<unsigned R, unsigned N, std::size_t J, typename T, typename Sink>
ADM_ALWAYS_INLINE void bfly_chunk_scalar_ct(const T* yre, const T* yim,
                                            Sink&& sink) {
    constexpr std::size_t M = N / R;
    T tr[R], ti[R];
    poet::static_for<0, R>([&](const auto q) {
        const T ar = yre[q * M + J];
        const T ai = yim[q * M + J];
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const auto [wr, wi] = apply_stage_twiddle<T, N, decltype(q)::value * J, T>(ar, ai);
            tr[q] = wr;
            ti[q] = wi;
        }
    });
    dif_butterfly<T, R, T>(tr, ti,
        [&](auto L, T outr, T outi) { sink(L * M + J, outr, outi); });
}

// Narrowest SIMD width >= 2 with a native batch at this T (tail descent starts here).
template<typename T, std::size_t Wt = 2>
[[nodiscard]] ADM_CONSTEVAL std::size_t min_sized_tail_width() {
    if constexpr (!std::is_void_v<xsimd::make_sized_batch_t<T, Wt>>) return Wt;
    else return min_sized_tail_width<T, Wt * 2>();
}

// Compile-time-M radix-R combine: full chunks at native width, then the sub-W
// remainder (rem = M % W < W) by its binary expansion: one batch per set bit
// at existing widths, value-folded scalar chunks for the rest. Every j is a
// compile-time constant, so no combine work survives as an out-of-line call.
template<unsigned R, unsigned N, typename T, typename Sink>
ADM_ALWAYS_INLINE void radix_butterfly_ct(T* ADM_RESTRICT yre, T* ADM_RESTRICT yim,
                                          Sink&& sink) {
    constexpr std::size_t M = N / R;
    static constexpr auto twre = make_twiddle_table<N, R, T, false>();
    static constexpr auto twim = make_twiddle_table<N, R, T, true>();
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    // Spill-aware unroll: R-input combine holds ~2R+10 live batches (tr/ti +
    // butterfly accumulators). U = registers / peak_live (dif_pass_unroll model).
    constexpr std::size_t U = dif_pass_unroll<R>();
    constexpr std::size_t nfull = M / W;
    if constexpr (nfull > 0) {
        // Force-inline only when 2R+10 live batches fit the vector file (R<=4);
        // At R=8 out-of-line wins, because an inlined spilling combine merges live ranges
        // with caller (see kernel_should_noinline).
        if constexpr (2u * R + 10u <= poet::vector_register_count() + 2u) {
            poet::dynamic_for<U>(std::size_t{0}, nfull, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
                bfly_chunk<R, T, batch>(twre.data(), twim.data(), yre, yim, M, b * W, sink);
            });
        } else {
            poet::dynamic_for<U>(std::size_t{0}, nfull, [&](std::size_t b) {
                bfly_chunk<R, T, batch>(twre.data(), twim.data(), yre, yim, M, b * W, sink);
            });
        }
    }
    constexpr std::size_t rem = M - nfull * W;
    poet::static_for<1, detail::bit_width(W)>([&](auto S) {
        constexpr std::size_t Wt = W >> S;
        using Vt = xsimd::make_sized_batch_t<T, Wt>;
        if constexpr (Wt >= 2 && !std::is_void_v<Vt> && (rem & Wt) != 0) {
            bfly_chunk<R, T, Vt>(twre.data(), twim.data(), yre, yim, M,
                                 nfull * W + (rem & ~(2 * Wt - 1)), sink);
        }
    });
    // Scalar residue: the low bits below the narrowest existing batch (0-1
    // elements f64, 0-3 f32). Value-folded twiddles (J compile-time).
    constexpr std::size_t smask = min_sized_tail_width<T>() - 1;
    constexpr std::size_t nscal = rem & smask;
    constexpr std::size_t jscal = nfull * W + (rem & ~smask);
    poet::static_for<0, nscal>([&](auto I) {
        bfly_chunk_scalar_ct<R, N, jscal + decltype(I)::value, T>(yre, yim, sink);
    });
}

template<unsigned R, unsigned N, typename T>
ADM_ALWAYS_INLINE void radix_butterfly_ct(T* ADM_RESTRICT yre, T* ADM_RESTRICT yim) {
    radix_butterfly_ct<R, N, T>(yre, yim, yre_sink<T>{yre, yim});
}

// ----------------------------------------------------------------------------
// Rader prime codelet (chiplet): size-p DFT (prime p > 11) has no small-radix
// factorization. Rader's algorithm expresses it as a length-(p-1) cyclic
// convolution evaluated with kernel<p-1> codelets.
//
// With g a primitive root mod p:  a_q = x[g^q],   b'_j = w_p^{g^{-j}},
//   X[g^{-m}] - x[0] = (a (*) b')_m = IDFT_{p-1}( DFT_{p-1}(a) . DFT_{p-1}(b') ),
//   X[0] = sum_n x[n].
// DFT_{p-1}(b') is data-independent, baked consteval in make_rader_bhat.
// ----------------------------------------------------------------------------

// Consteval forward DFT_{p-1} of the Rader b' sequence (convolution kernel),
// real (Imag=false) or imag (Imag=true). Pure compile-time double; no runtime trig.
// Pre-divided by L: the convolution ends in an unnormalized inverse kernel<L>, and by
// linearity the fold costs nothing (compile-time table) and deletes 2L runtime
// multiplies per Rader codelet.
template<unsigned P, typename T, bool Imag>
ADM_CONSTEVAL std::array<T, P - 1> make_rader_bhat() {
    constexpr unsigned L = P - 1;
    constexpr std::size_t g = ct_primitive_root(P);
    struct cd { double re; double im; };
    const std::size_t ginv = ct_powmod(g, P - 2, P);   // g^{-1} mod p
    std::array<cd, L> bp{};
    for (unsigned j = 0; j < L; ++j) {
        const std::size_t e = ct_powmod(ginv, j, P);   // g^{-j} mod p
        const ct_sincos_t w = ct_sincos_turns(/*conjugate=*/true, e, P);  // forward sign of w_p in b'
        bp[j] = {w.c, w.s};
    }
    std::array<T, L> out{};
    for (unsigned k = 0; k < L; ++k) {
        double sr = 0.0, si = 0.0;
        for (unsigned j = 0; j < L; ++j) {
            // exp(-2*pi*i j k / L): c=cos, s=sin(-angle)
            const ct_sincos_t e = ct_sincos_turns(/*conjugate=*/true, j * k % L, L);
            sr += bp[j].re * e.c - bp[j].im * e.s;
            si += bp[j].re * e.s + bp[j].im * e.c;
        }
        out[k] = static_cast<T>((Imag ? si : sr) / L);
    }
    return out;
}

// ----------------------------------------------------------------------------
// Cofactor-SIMD batched codelet (kernel_batched / rader_apply_batched).
//
// When N = r*M and r <= SIMD width W, the r size-M sub-transforms are independent
// and the DIT-decimated input x[r*j+q] is contiguous in lane order: run all r as ONE
// Wc-wide batched size-M transform, twiddles broadcast (they depend only on j).
// Combine vectorizes over LANES (one j per chunk, twiddles folded by value), NOT
// over consecutive j as in radix_butterfly_ct.
// cofactor_batch_width picks the narrowest ISA register >= r; when Wc>r, a
// compile-time masked load zeroes the high lanes.
// ----------------------------------------------------------------------------

// Narrowest power-of-two SIMD width >= R with a native batch, capped at native
// width. Packs R independent sub-transforms into the smallest register, which leaves
// the fewest idle lanes. The recursion walks W as a template argument so
// make_sized_batch_t can probe each candidate. Native width is always available, so the
// recursion always terminates.
template<typename T, std::size_t R, std::size_t W = 1>
[[nodiscard]] ADM_CONSTEVAL std::size_t cofactor_batch_width() {
    if constexpr (W >= xsimd::batch<T>::size)
        return xsimd::batch<T>::size;  // native width is always available, so stop here
    else if constexpr (W >= R && !std::is_void_v<xsimd::make_sized_batch_t<T, W>>)
        return W;                      // narrowest available width that holds R
    else
        return cofactor_batch_width<T, R, W * 2>();
}

// Radix-R combine, batched layout: one frequency j (compile-time), twiddles
// folded by value (apply_stage_twiddle) instead of broadcast table loads.
// `sink(p, outr, outi)` lets the top-level combine store anywhere (e.g. the DP
// terminal interleaves straight to AoS `data`) instead of round-tripping through
// yre/yim.
template<unsigned R, unsigned N, std::size_t J, typename T, typename V, typename Sink>
ADM_ALWAYS_INLINE void bfly_chunk_batched_ct(const V* yre, const V* yim,
                                             Sink&& sink) {
    constexpr std::size_t M = N / R;
    V tr[R], ti[R];
    poet::static_for<0, R>([&](const auto q) {
        const V ar = yre[q * M + J];
        const V ai = yim[q * M + J];
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const auto [wr, wi] = apply_stage_twiddle<T, N, decltype(q)::value * J, V>(ar, ai);
            tr[q] = wr;
            ti[q] = wi;
        }
    });
    dif_butterfly<T, R, V>(tr, ti,
        [&](auto L, V outr, V outi) { sink(L * M + J, outr, outi); });
}

template<unsigned R, unsigned N, typename T, typename V, typename Sink>
ADM_ALWAYS_INLINE void radix_butterfly_batched_ct(const V* ADM_RESTRICT yre,
                                                  const V* ADM_RESTRICT yim,
                                                  Sink&& sink) {
    constexpr std::size_t M = N / R;
    poet::static_for<0, M>([&](auto J) {
        bfly_chunk_batched_ct<R, N, decltype(J)::value, T, V>(yre, yim, sink);
    });
}

// sink(p, outr, outi) storing to arrays of batches (batched combine default).
template<typename V>
struct batch_sink {
    V* yre;
    V* yim;
    ADM_ALWAYS_INLINE void operator()(std::size_t p, V outr, V outi) const {
        yre[p] = outr;
        yim[p] = outi;
    }
};

// Rader codelet, batched over W lanes (one size-P DFT per lane).
// Mirror of rader_apply with T -> V; data-independent Bhat broadcast across lanes.
// `xstride` indexes the batch array. Output yre/yim: contiguous size-P block.
template<unsigned P, typename T, typename V = xsimd::batch<T>>
void rader_apply_batched(const V* xre, const V* xim, std::size_t xstride,
                         V* yre, V* yim);

// kernel_batched<N, T, Forward, V>: V::size independent size-N DFTs, one per lane.
// Same compile-time recursion as kernel<N> with every element a batch V.
// V defaults to native batch; cofactor-SIMD path picks the narrowest ISA batch
// >= cofactor count (see cofactor_batch_width).
template<unsigned N, typename T, bool Forward, typename V = xsimd::batch<T>>
struct kernel_batched {
    static_assert(Forward, "inverse routes through the swapped-domain specialization");
    static constexpr unsigned r = codelet_radix(N);
    static constexpr unsigned M = N / r;
    static_assert(r * M == N, "codelet_radix(N) must divide N exactly");

    // Cut cofactor recursion at small leaves: a flat register-resident
    // dif_butterfly_terminal avoids the inter-level yre/yim round-trip the cofactor
    // combine pays. Threshold: 2N live batches must fit the file (a pow2 leaf is
    // scheduled for the whole file; every other leaf needs twiddle/address headroom).
    // Bit test rather than std::has_single_bit: gcc 14.2 rejects the library call in
    // this member initializer as used-before-definition at 16 vector registers.
    static constexpr bool flat_leaf =
        N >= 2 && 2u * N <= (((N & (N - 1u)) == 0u)
                                 ? poet::vector_register_count()
                                 : usable_vector_regs(poet::vector_register_count()));

    static void apply(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim) {
        if constexpr (is_rader_prime(N)) {
            rader_apply_batched<N, T, V>(xre, xim, xstride, yre, yim);
            return;
        } else if constexpr (flat_leaf) {
            V tr[N], ti[N];
            poet::static_for<0, N>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly_terminal<T, N, V>(tr, ti, [&](auto K, V re, V im) {
                yre[K] = re;
                yim[K] = im;
            });
        } else {
            poet::static_for<0, r>([&](const auto q) {
                kernel_batched<M, T, true, V>::apply(xre + q * xstride, xim + q * xstride,
                                                         xstride * r, yre + q * M, yim + q * M);
            });
            radix_butterfly_batched_ct<r, N, T, V>(yre, yim, batch_sink<V>{yre, yim});
        }
    }

    // Same transform, but the FINAL combine emits via sink(p, outr, outi) instead
    // of storing to yre/yim. The caller fuses its store (scale + AoS interleave) into
    // the last butterfly. yre/yim stay as scratch. Each output index sinks once.
    template<typename Sink>
    static void apply_sink(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim,
                           Sink&& sink) {
        if constexpr (is_rader_prime(N)) {
            // Rader writes via data-dependent index maps; no last-combine hook.
            // Run whole, then forward outputs.
            rader_apply_batched<N, T, V>(xre, xim, xstride, yre, yim);
            poet::static_for<0, N>([&](auto P) { sink(P, yre[P], yim[P]); });
        } else if constexpr (flat_leaf) {
            V tr[N], ti[N];
            poet::static_for<0, N>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly_terminal<T, N, V>(tr, ti,
                [&](auto K, V re, V im) { sink(K, re, im); });
        } else {
            poet::static_for<0, r>([&](const auto q) {
                kernel_batched<M, T, true, V>::apply(xre + q * xstride, xim + q * xstride,
                                                         xstride * r, yre + q * M, yim + q * M);
            });
            radix_butterfly_batched_ct<r, N, T, V>(yre, yim, sink);
        }
    }
};

template<typename T, typename V>
struct kernel_batched<1, T, true, V> {
    static void apply(const V* xre, const V* xim, std::size_t /*xstride*/, V* yre, V* yim) {
        yre[0] = xre[0];
        yim[0] = xim[0];
    }
};

// DIRECTION-FREE INTERIOR. Everything above computes the FORWARD DFT only; the
// inverse rides the same code in swapped domain. With z = a+bi, swap(z) = b+ai
// = i*conj(z), so by linearity
//     swap(fwd(swap x)) = conj(fwd(conj x)) = inv(x)
// exactly. The inverse sums the same terms in the forward tree's association, so ULP
// cells move: accuracy-neutral, not unchanged.
template<unsigned N, typename T, typename V>
struct kernel_batched<N, T, false, V> {
    static void apply(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim) {
        kernel_batched<N, T, true, V>::apply(xim, xre, xstride, yim, yre);
    }
    template<typename Sink>
    static void apply_sink(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim,
                           Sink&& sink) {
        kernel_batched<N, T, true, V>::apply_sink(xim, xre, xstride, yim, yre,
            [&](std::size_t p, V outr, V outi) { sink(p, outi, outr); });
    }
};

// Forward declaration: Rader codelet calls forward/inverse kernel<p-1>, defined
// after kernel<N> (like kernel_apply_boundary).
template<unsigned P, typename T>
void rader_apply(const T* xre, const T* xim, std::size_t xstride, T* yre, T* yim);

// ----------------------------------------------------------------------------
// kernel<N, T, Forward>: size-N DFT, planar split real/imag.
//
//   apply(xre, xim, xstride, yre, yim)
//
// Reads N inputs x[k] = (xre,xim)[k*xstride], writes DFT to (yre,yim)[0..N).
// Output must not alias input. Forward: exp(-2*pi*i*kn/N); inverse: + (swapped domain).
//
// Fully compile-time-recursive (all radices unrolled). The tests validate this
// building block; the runtime driver reuses radix_butterfly for large N.
// ----------------------------------------------------------------------------

// Should the size-M sub-transform be a noinline boundary instead of inlined?
// Both conditions required:
//   (1) sub-transform combine spills: peak_live = 2r+10 > vector_register_count()
//       (16 YMM on AVX2, 32 ZMM on AVX-512);
//   (2) M >= kNoinlineMinSize: amortizes call overhead (tiny leaves: call cost > win).
// noinline boundary caps each level to its own register allocation.
inline constexpr unsigned kNoinlineMinSize = 16;

// Between SIMD width and the catalog max: below the width no batched work
// amortizes the call; above the catalog the boundary could never fire.
static_assert(kNoinlineMinSize >= xsimd::batch<float>::size && kNoinlineMinSize <= CODELET_CATALOG_MAX,
              "kNoinlineMinSize must sit between the SIMD width and the catalog max");

[[nodiscard]] ADM_CONSTEVAL bool kernel_should_noinline(std::size_t M) {
    return M >= kNoinlineMinSize
        && 2 * codelet_radix(M) + 10 > poet::vector_register_count();
}

// Cofactor-SIMD profitability for peeled cofactor M and radix R: run R size-M
// sub-transforms across Wc lanes. Profitable when kernel<M> is scalar-dominated AND
// the batch is nearly full. Eligible M:
//   * Rader primes (M>13 prime): batching R copies amortizes per-copy scalar cost.
//     Rejected if 2*R <= Wc, because scalar rader_apply<M> runs its kernel<M-1>
//     convolution at full width and wins.
//   * Odd composites / small odd primes (3,5,7,9,11,15,21,27,...): no radix-2/4
//     factor; kernel<M> combine is scalar.
//   * Even M at least one batch tile wide: the scalar recursion runs at non-unit
//     stride, which costs more than the lanes the batch leaves idle.
// Width/utilization gate (masked-load cost):
//   * Wc==R:            full batch, no mask.
//   * 2*R <= Wc:        half-idle batch, reject.
//   * Wc*sizeof(T)<=16: 128-bit masked load, cheap.
//   * else:             256-bit masked load, require R*M >= kMaskedLoad256MinWork.
inline constexpr std::size_t kMaskedLoad256MinWork = 27;

template<typename T, unsigned R>
[[nodiscard]] ADM_CONSTEVAL bool cofactor_simd_profitable(std::size_t M) {
    if (is_rader_prime(M)) {
        // Rader-prime: batch R copies across lanes UNLESS half-idle
        // (f32 R=2 -> Wc=4): scalar rader_apply<M> runs kernel<M-1> at full ymm.
        constexpr std::size_t Wc = cofactor_batch_width<T, R>();
        if constexpr (2u * R <= Wc) return false;  // f32 R=2: prefer scalar rader_apply (full ymm)
        return true;
    }
    // pow2_m covers cofactors narrower than a batch tile, which even_m cannot admit: with
    // Wc == R the load is a full batch, so M=4 pays no mask. M=8 needs 2*M live batches to
    // fit, hence 32-reg ISAs only. There is no M=16 arm: Wc == R forces R <= 16, so
    // M=16 >= Wc and even_m already admits it. odd_m keeps the noinline guard and pow2_m
    // does not, because that predicate models scalar spill pressure.
    const bool odd_m = M >= 3 && (M % 2 != 0) && !kernel_should_noinline(M);
    const bool pow2_m = (M == 4 || (M == 8 && 2 * M < poet::vector_register_count())) &&
                        cofactor_batch_width<T, R>() == R;
    // Even cofactor filling at least one batch tile, past what odd_m/pow2_m admit.
    const bool even_m = (M % 2 == 0) && M >= cofactor_batch_width<T, R>();
    if (!(odd_m || pow2_m || even_m)) return false;
    constexpr std::size_t Wc = cofactor_batch_width<T, R>();
    if (Wc == R) return true;                  // full batch, no masked load: always wins
    if (2u * R <= Wc) return false;            // low-half mask: half-idle batch
    if (Wc * sizeof(T) <= 16u) return true;    // 128-bit hardware-masked load: cheap
    return R * M >= kMaskedLoad256MinWork;     // 256-bit masked load: needs enough work
}

// Peel radix for kernel<N>: prefer a pow2 r with odd cofactor M=N/r so the
// cofactor-SIMD path fires (kernel_batched<M> across r lanes) instead of the scalar
// recursion. Gated on an r-lane batch existing and cofactor_simd_profitable; where
// either fails this equals codelet_radix_for. 64 -> 8x8: r=4's flat-16 batched leaf
// holds 2*16 batches, r=8's flat-8 holds 2*8 plus a combine under the 2r+10 budget.
template<typename T>
[[nodiscard]] ADM_CONSTEVAL std::size_t kernel_peel_radix(std::size_t N) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (N == 64 && 8 <= W && cofactor_simd_profitable<T, 8u>(8)) return 8;
    if (N % 8 == 0 && (N / 8) % 2 == 1 && 8 <= W && cofactor_simd_profitable<T, 8u>(N / 8))
        return 8;
    if (N % 4 == 0 && (N / 4) % 2 == 1 && 4 <= W && cofactor_simd_profitable<T, 4u>(N / 4))
        return 4;
    return codelet_radix_for<T>(N);
}

// Noinline boundary forward declaration (defined after kernel<N> to name
// kernel<N>::apply). Attribute lives on this declaration.
template<unsigned N, typename T>
ADM_NOINLINE void kernel_apply_boundary(const T* xre, const T* xim, std::size_t xstride,
                                        T* yre, T* yim);

template<unsigned N, typename T, bool Forward>
struct kernel {
    static_assert(Forward, "inverse routes through the swapped-domain specialization");
    // codelet_radix_for overridden by the odd-cofactor pow2 peel when the
    // cofactor-SIMD path fires (see kernel_peel_radix).
    static constexpr unsigned r = kernel_peel_radix<T>(N);
    static constexpr unsigned M = N / r;
    // Exact factorization: a non-dividing peeled radix silently drops or duplicates
    // inputs. codelet_radix is a fixpoint divisor of N (or N for a Rader prime,
    // M==1), so this always holds. The assert pins the planner contract at compile time.
    static_assert(r * M == N, "codelet_radix(N) must divide N exactly");

    // Flat register-resident leaf, admitted where it beats the cofactor combine.
    // 2N live scalars must fit the file. A pow2 leaf stops at 8 (pow2_dif_butterfly
    // needs ~N/2 extra temps); every other leaf runs to 13, the largest non-Rader
    // prime. M > r defers to cofactor-SIMD; at M <= r the flat leaf wins, so 12 (4x3)
    // and 9 (3x3) stay flat and 10 (2x5) does not.
    // Bit test, for the reason kernel_batched::flat_leaf gives.
    static constexpr bool flat_leaf =
        N >= 2 && (((N & (N - 1u)) == 0u) ? N <= 8u : N <= 13u)
        && 2u * N <= poet::vector_register_count()
        && !(M > r && r <= xsimd::batch<T>::size && cofactor_simd_profitable<T, r>(M));

    // Whole forward transform, emitting each output via sink(p, outr, outi):
    // (l*M + j) from the last combine, lane width V or scalar T. apply() below
    // stores to yre/yim; apply_sink() forwards the caller's store (codelet_apply's
    // AoS interleave) without an output SoA round-trip. Each output index sinks once.
    template<typename Sink>
    static void apply_impl(const T* xre, const T* xim, std::size_t xstride,
                           T* yre, T* yim, Sink&& sink) {
        // Prime > 11: use the Rader chiplet instead of a flat radix-N unroll.
        // if constexpr discards recursion + twiddle-fold below.
        if constexpr (is_rader_prime(N)) {
            rader_apply<N, T>(xre, xim, xstride, yre, yim);
            poet::static_for<0, N>([&](auto P) { sink(P, yre[P], yim[P]); });
            return;
        }
        // Flat leaf: one dif_butterfly over registers, no inter-level yre/yim
        // round-trip (dispatch-bound; same fix as kernel_batched).
        if constexpr (flat_leaf) {
            T tr[N], ti[N];
            poet::static_for<0, N>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly<T, N, T>(tr, ti, [&](auto K, T re, T im) { sink(K, re, im); });
            return;
        }
        // Cofactor-SIMD path: when M is a Rader prime or odd scalar-dominated cofactor
        // and r <= W, run the r size-M sub-transforms as ONE Wc-wide batched transform
        // (one per lane), then combine.
        // At unit stride x[r*j..r*j+r-1] is contiguous: r==Wc -> one full batch load;
        // r<Wc -> low lanes packed (no OOB). Unit stride always holds here.
        if constexpr (r <= xsimd::batch<T>::size && cofactor_simd_profitable<T, r>(M)) {
            // Narrowest ISA register >= r (cofactor_batch_width).
            // Low r lanes carry data; Wc>r -> compile-time masked load zeroes high lanes.
            constexpr std::size_t Wc = cofactor_batch_width<T, r>();
            using V = xsimd::make_sized_batch_t<T, Wc>;
            static_assert(!std::is_void_v<V>, "no SIMD batch wide enough for cofactor r");
            static_assert(Wc >= r, "cofactor batch must hold all r sub-transforms");
            using A = typename V::arch_type;
            // Lanes [0, r) hold a sub-transform; [r, Wc) are masked off.
            if (xstride == 1) {
                // ovr/ovi padded to whole Wc-tiles so leftover M%Wc columns use
                // the masked-tile transpose path instead of per-element extracts.
                // Pad only when M >= Wc (masked-tile path taken).
                constexpr std::size_t Mp =
                    (M % Wc != 0 && M >= Wc) ? ((M / Wc + 1) * Wc) : M;
                V ovr[Mp], ovi[Mp];
                {
                    poet::static_for<0, Mp - M>([&](auto P) {
                        // Guard: static_for instantiates with P=0 even when Mp==M;
                        // ovr[M] is OOB, which clang reports as -Warray-bounds.
                        if constexpr (M + P < Mp) {
                            ovr[M + P] = V(T(0));
                            ovi[M + P] = V(T(0));
                        }
                    });
                    if constexpr (Wc == r) {
                        // The direct arm reads xre/xim as aligned batch arrays, which is
                        // legal only below the alignment check. The barrier pins every
                        // memory access below the test, so no compiler hoists a read
                        // above it. It emits no instructions of its own; it does
                        // constrain scheduling around the test.
                        asm volatile("" ::: "memory");
                        // Full batch: the batched input layout is byte-identical to the
                        // planar source, so forward it when V-aligned instead of staging
                        // (gcc lowers the staging loop to out-of-line memcpy PLT calls).
                        const bool kb_aligned =
                            (reinterpret_cast<std::uintptr_t>(xre) % alignof(V) == 0) &&
                            (reinterpret_cast<std::uintptr_t>(xim) % alignof(V) == 0);
                        if (kb_aligned) {
                            kernel_batched<M, T, true, V>::apply(
                                reinterpret_cast<const V*>(xre),
                                reinterpret_cast<const V*>(xim), 1, ovr, ovi);
                        } else {
                            V xv[M], iv[M];
                            for (std::size_t j = 0; j < M; ++j) {
                                xv[j] = V::load_unaligned(xre + r * j);
                                iv[j] = V::load_unaligned(xim + r * j);
                            }
                            kernel_batched<M, T, true, V>::apply(xv, iv, 1, ovr, ovi);
                        }
                    } else {
                        V xv[M], iv[M];
                        for (std::size_t j = 0; j < M; ++j) {
                            constexpr auto mask = xsimd::make_batch_bool_constant<T, lane_lt<r>, A>();
                            xv[j] = V::load(xre + r * j, mask, xsimd::unaligned_mode{});
                            iv[j] = V::load(xim + r * j, mask, xsimd::unaligned_mode{});
                        }
                        kernel_batched<M, T, true, V>::apply(xv, iv, 1, ovr, ovi);
                    }
                }
                // Combine r blocks fused per Wc-tile: the M output batches are an
                // M×Wc matrix whose in-register Wc×Wc transpose (vunpck/vperm) IS
                // the gather y[q*M+j]. The combine consumes the transposed rows and
                // emits via sink with no yre/yim round-trip. xsimd::transpose needs
                // exactly Wc batches per tile; only rows [0,r) are real.
                constexpr std::size_t nft = M / Wc;  // number of full Wc-wide tiles
                static constexpr auto twre = make_twiddle_table<N, r, T, false>();
                static constexpr auto twim = make_twiddle_table<N, r, T, true>();
                if constexpr (nft > 0) {
                    poet::static_for<0, nft>([&](auto Tt) {
                        constexpr std::size_t k0 = decltype(Tt)::value * Wc;
                        xsimd::transpose(&ovr[k0], &ovr[k0] + Wc);
                        xsimd::transpose(&ovi[k0], &ovi[k0] + Wc);
                        V tr[r], ti[r];
                        poet::static_for<0, r>([&](const auto q) {
                            if constexpr (q == 0) {
                                tr[0] = ovr[k0];
                                ti[0] = ovi[k0];
                            } else {
                                const V ar = ovr[k0 + q];
                                const V ai = ovi[k0 + q];
                                const V wr = V::load_unaligned(twre.data() + (q - 1) * M + k0);
                                const V wi = V::load_unaligned(twim.data() + (q - 1) * M + k0);
                                tr[q] = wr * ar - wi * ai;
                                ti[q] = wr * ai + wi * ar;
                            }
                        });
                        dif_butterfly<T, r, V>(tr, ti,
                            [&](auto L, V outr, V outi) { sink(L * M + k0, outr, outi); });
                    });
                }
                // Leftover columns [nft*Wc, M): masked/scalar row stores to
                // yre/yim, then one value-folded scalar combine chunk per column.
                if constexpr (M % Wc != 0 && M >= Wc) {
                    // Masks feeding a transpose are the sanctioned exception to the
                    // sized-batch tail policy. Gated on M >= Wc, at least one full
                    // tile. For tile-less tiny cofactors (e.g. M=3,Wc=4 in
                    // rader<13>'s kernel<12>) the scalar scatter wins.
                    constexpr auto smask = xsimd::make_batch_bool_constant<T, lane_lt<M % Wc>, A>();
                    constexpr std::size_t k0 = nft * Wc;
                    xsimd::transpose(&ovr[k0], &ovr[k0] + Wc);
                    xsimd::transpose(&ovi[k0], &ovi[k0] + Wc);
                    poet::static_for<0, r>([&](const auto q) {
                        ovr[k0 + q].store(yre + q * M + k0, smask, xsimd::unaligned_mode{});
                        ovi[k0 + q].store(yim + q * M + k0, smask, xsimd::unaligned_mode{});
                    });
                } else if constexpr (M % Wc != 0) {
                    T sr[Wc], si[Wc];
                    for (std::size_t k = nft * Wc; k < M; ++k) {
                        ovr[k].store_unaligned(sr);
                        ovi[k].store_unaligned(si);
                        poet::static_for<0, r>([&](const auto q) {
                            yre[q * M + k] = sr[q];
                            yim[q * M + k] = si[q];
                        });
                    }
                }
                // Leftover columns [nft*Wc, M) (count C < Wc): descend existing
                // sized batches by C's bits (one chunk per set bit), then
                // value-folded scalar chunks for the sub-width residue. Pure
                // scalar loses. One 4-wide chunk beats four scalar radix-8
                // butterflies.
                constexpr std::size_t C = M - nft * Wc;
                poet::static_for<1, detail::bit_width(Wc)>([&](auto S) {
                    constexpr std::size_t Wt = Wc >> S;
                    using Vt = xsimd::make_sized_batch_t<T, Wt>;
                    if constexpr (Wt >= 2 && !std::is_void_v<Vt> && (C & Wt) != 0) {
                        bfly_chunk<r, T, Vt>(twre.data(), twim.data(), yre, yim, M,
                                             nft * Wc + (C & ~(2 * Wt - 1)), sink);
                    }
                });
                constexpr std::size_t smask = min_sized_tail_width<T>() - 1;
                constexpr std::size_t nscal = C & smask;
                poet::static_for<0, nscal>([&](auto I) {
                    constexpr std::size_t J = nft * Wc + (C & ~smask) + decltype(I)::value;
                    bfly_chunk_scalar_ct<r, N, J, T>(yre, yim, sink);
                });
                return;
            }
        }
        // Recurse: r sub-DFTs of size M, decimated by residue mod r, written into
        // r contiguous output blocks. Above pressure threshold, route through a
        // noinline boundary for an isolated register allocation.
        poet::static_for<0, r>([&](const auto q) {
            if constexpr (kernel_should_noinline(M)) {
                kernel_apply_boundary<M, T>(xre + q * xstride, xim + q * xstride,
                                                     xstride * r, yre + q * M, yim + q * M);
            } else {
                kernel<M, T, true>::apply(xre + q * xstride, xim + q * xstride,
                                             xstride * r, yre + q * M, yim + q * M);
            }
        });
        // Combine r blocks (consteval twiddle table; M==1: all twiddles are ones,
        // combine is a pure radix-N DFT). Value-folded scalar residue.
        radix_butterfly_ct<r, N, T>(yre, yim, sink);
    }

    static void apply(const T* xre, const T* xim, std::size_t xstride,
                      T* yre, T* yim) {
        apply_impl(xre, xim, xstride, yre, yim, yre_sink<T>{yre, yim});
    }

    // Same transform, but outputs emit via sink(p, outr, outi). The caller fuses
    // its store (AoS interleave) into the last combine. yre/yim: scratch only.
    // Each output index sinks exactly once, as a value T or a sized batch V.
    template<typename Sink>
    static void apply_sink(const T* xre, const T* xim, std::size_t xstride,
                           T* yre, T* yim, Sink&& sink) {
        apply_impl(xre, xim, xstride, yre, yim,
                   static_cast<Sink&&>(sink));
    }
};

// Size-1 DFT is the identity (strided copy).
template<typename T>
struct kernel<1, T, true> {
    static void apply(const T* xre, const T* xim, std::size_t /*xstride*/,
                      T* yre, T* yim) {
        yre[0] = xre[0];
        yim[0] = xim[0];
    }
    template<typename Sink>
    static void apply_sink(const T* xre, const T* xim, std::size_t /*xstride*/,
                           T* /*yre*/, T* /*yim*/, Sink&& sink) {
        sink(std::size_t{0}, xre[0], xim[0]);
    }
};

// Inverse via swapped domain; see the kernel_batched note above.
template<unsigned N, typename T>
struct kernel<N, T, false> {
    static void apply(const T* xre, const T* xim, std::size_t xstride, T* yre, T* yim) {
        kernel<N, T, true>::apply(xim, xre, xstride, yim, yre);
    }
    template<typename Sink>
    static void apply_sink(const T* xre, const T* xim, std::size_t xstride, T* yre, T* yim,
                           Sink&& sink) {
        kernel<N, T, true>::apply_sink(xim, xre, xstride, yim, yre,
            [&](std::size_t p, auto outr, auto outi) { sink(p, outi, outr); });
    }
};

// Noinline boundary: thin forwarder that forces an isolated register allocation
// for the size-N sub-transform (see kernel_should_noinline).
template<unsigned N, typename T>
ADM_NOINLINE void kernel_apply_boundary(const T* xre, const T* xim, std::size_t xstride,
                                        T* yre, T* yim) {
    kernel<N, T, true>::apply(xre, xim, xstride, yre, yim);
}

// Consteval g^q mod P table for Rader input gather (q=0..L-1, L=P-1).
// Shared by rader_apply and rader_apply_batched; templated on P/L only to avoid
// duplicate instantiations across T/V.
template<unsigned P, unsigned L = P - 1>
ADM_CONSTEVAL std::array<std::size_t, L> rader_gpow_table() {
    constexpr std::size_t g = ct_primitive_root(P);
    std::array<std::size_t, L> t{};
    std::size_t e = 1;
    for (unsigned q = 0; q < L; ++q) {
        t[q] = e;
        e = e * g % P;
    }
    return t;
}

// Consteval g^{-m} mod P table for Rader output scatter (m=0..L-1, L=P-1).
// Shared by rader_apply and rader_apply_batched.
template<unsigned P, unsigned L = P - 1>
ADM_CONSTEVAL std::array<std::size_t, L> rader_ginvpow_table() {
    constexpr std::size_t g = ct_primitive_root(P);
    const std::size_t gi = ct_powmod(g, P - 2, P);
    std::array<std::size_t, L> t{};
    std::size_t e = 1;
    for (unsigned m = 0; m < L; ++m) {
        t[m] = e;
        e = e * gi % P;
    }
    return t;
}

// Rader prime codelet: size-P DFT (P prime > 11) as a length-(P-1) cyclic
// convolution via forward+inverse kernel<P-1>. noinline: gather/scatter + two
// sub-transforms get an isolated register allocation.
template<unsigned P, typename T>
ADM_NOINLINE void rader_apply(const T* xre, const T* xim, std::size_t xstride,
                              T* yre, T* yim) {
    constexpr unsigned L = P - 1;

    // Input gather index g^q mod p and output scatter index g^{-m} mod p.
    static constexpr std::array<std::size_t, L> gpow = rader_gpow_table<P>();
    static constexpr std::array<std::size_t, L> ginvpow = rader_ginvpow_table<P>();
    static constexpr std::array<T, L> Bre = make_rader_bhat<P, T, false>();
    static constexpr std::array<T, L> Bim = make_rader_bhat<P, T, true>();

    // Gather a_q = x[g^q] with input stride.
    const T x0r = xre[0];
    const T x0i = xim[0];
    T are[L], aim[L];
    for (unsigned q = 0; q < L; ++q) {
        const std::size_t idx = gpow[q] * xstride;
        are[q] = xre[idx];
        aim[q] = xim[idx];
    }

    // A = DFT_L(a); P = A .* Bhat (Bhat carries the 1/L); c = IDFT_L(P).
    T Are[L], Aim[L];
    kernel<L, T, true>::apply(are, aim, 1, Are, Aim);

    // Pointwise P = A .* Bhat, SIMD-vectorized in planar SoA layout
    // (hand-rolled; xsimd::batch<complex> is slow), scalar tail for L%W remainder.
    T Pre[L], Pim[L];
    using batch = xsimd::batch<T>;
    constexpr std::size_t Wd = batch::size;
    constexpr std::size_t Lv = L - L % Wd;   // vectorized prefix
    for (std::size_t k = 0; k < Lv; k += Wd) {
        const batch ar = batch::load_unaligned(&Are[k]);
        const batch ai = batch::load_unaligned(&Aim[k]);
        const batch br = batch::load_unaligned(&Bre[k]);
        const batch bi = batch::load_unaligned(&Bim[k]);
        (ar * br - ai * bi).store_unaligned(&Pre[k]);
        (ar * bi + ai * br).store_unaligned(&Pim[k]);
    }
    // if constexpr, not a runtime tail: when L % Wd == 0 the tail is dead but gcc-14
    // still range-analyses it and reports Pre[L] out of bounds (-Werror=array-bounds,
    // hit at W=4 by P=13/17/53/61). Compile-time bounds delete it instead.
    if constexpr (L % Wd != 0) {
        for (std::size_t k = Lv; k < L; ++k) {
            Pre[k] = Are[k] * Bre[k] - Aim[k] * Bim[k];
            Pim[k] = Are[k] * Bim[k] + Aim[k] * Bre[k];
        }
    }

    T cre[L], cim[L];
    kernel<L, T, false>::apply(Pre, Pim, 1, cre, cim);

    // X[0] = sum_n x[n] = x0 + sum_q a_q, and the forward kernel already computed
    // that sum as A[0], so no separate accumulation and no length-L serial chain.
    yre[0] = x0r + Are[0];
    yim[0] = x0i + Aim[0];
    for (unsigned m = 0; m < L; ++m) {
        yre[ginvpow[m]] = x0r + cre[m];
        yim[ginvpow[m]] = x0i + cim[m];
    }
}

// Batched Rader codelet: rader_apply with T -> V; Bhat broadcast across lanes.
// One size-P DFT per lane; forward/inverse length-(P-1) transforms use kernel_batched.
template<unsigned P, typename T, typename V>
ADM_NOINLINE void rader_apply_batched(const V* xre, const V* xim, std::size_t xstride,
                                      V* yre, V* yim) {
    constexpr unsigned L = P - 1;

    static constexpr std::array<std::size_t, L> gpow = rader_gpow_table<P>();
    static constexpr std::array<std::size_t, L> ginvpow = rader_ginvpow_table<P>();
    static constexpr std::array<T, L> Bre = make_rader_bhat<P, T, false>();
    static constexpr std::array<T, L> Bim = make_rader_bhat<P, T, true>();

    const V x0r = xre[0];
    const V x0i = xim[0];
    V are[L], aim[L];
    for (unsigned q = 0; q < L; ++q) {
        const std::size_t idx = gpow[q] * xstride;
        are[q] = xre[idx];
        aim[q] = xim[idx];
    }

    V Are[L], Aim[L];
    kernel_batched<L, T, true, V>::apply(are, aim, 1, Are, Aim);

    // Pointwise complex multiply P = A .* Bhat (Bhat carries the 1/L), broadcast.
    V Pre[L], Pim[L];
    for (unsigned k = 0; k < L; ++k) {
        const V br(Bre[k]);
        const V bi(Bim[k]);
        Pre[k] = Are[k] * br - Aim[k] * bi;
        Pim[k] = Are[k] * bi + Aim[k] * br;
    }

    V cre[L], cim[L];
    kernel_batched<L, T, false, V>::apply(Pre, Pim, 1, cre, cim);

    // A[0] is already sum_q a_q; see rader_apply.
    yre[0] = x0r + Are[0];
    yim[0] = x0i + Aim[0];
    for (unsigned m = 0; m < L; ++m) {
        yre[ginvpow[m]] = x0r + cre[m];
        yim[ginvpow[m]] = x0i + cim[m];
    }
}

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"

