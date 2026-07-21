#pragma once

// ============================================================================
// Generic, template-metaprogrammed 1D DFT codelet: kernel<N, T, Forward>.
//
// A SINGLE generic template computes a size-N DFT by compile-time recursion
// (one body, no hand-written per-size kernels). N is factored at compile time
// into a small radix r and cofactor M = N/r; the size-N transform is composed
// from r sub-transforms of size M plus one generic radix-r butterfly.
//
// Data layout is PLANAR (split real/imag arrays of T) so the arithmetic maps
// directly onto hand-written real/imag SIMD batches later -- we deliberately
// avoid xsimd::batch<std::complex<T>> (measured slow). The public std::complex
// AoS API is preserved by de-interleaving on entry and re-interleaving on exit
// (see the AoS driver).
// ============================================================================

#include <array>
#include <bit>
#include <cstddef>
#include <numbers>
#include <type_traits>

#include <poet/poet.hpp>
#include <xsimd/xsimd.hpp>

#include "butterfly.hpp"  // dif_butterfly (symmetric odd-radix + split-radix pow2)
#include "ct_math.hpp"  // ct_sincos_t/turns, smallest_radix, codelet_radix
#include "macros.hpp"

namespace admiral {
namespace detail {

// ----------------------------------------------------------------------------
// radix_butterfly<R, T, Forward>: the generic Cooley-Tukey combine stage.
//
// Given a planar buffer (yre, yim) holding R sub-DFTs of length M laid out as R
// contiguous blocks (block q occupies [q*M, (q+1)*M)), this applies, for every
// j in [0, M):
//   (1) the inter-stage twiddle  t_q = W_N^{q j} * y[q*M + j]   (q = 0 trivial),
//   (2) a radix-R DFT across the R samples,  out_l = sum_q W_R^{q l} t_q,
//   (3) scatter out_l to y[l*M + j].
//
// The radix-R matrix W_R^{q l} is constant-folded at compile time (one generic
// static_for double-sum, specialized for every R with no per-radix source). The
// inter-stage twiddles W_N^{q j} come from a contiguous table laid out over j for
// each q in [1, R): index (q-1)*M + j. For the compile-time kernel<N> this table
// is itself consteval-generated (constexpr std::array, no runtime trig); for the
// runtime driver it is precomputed once per plan. Either way the SAME vectorized
// body below consumes it.
//
// Arithmetic is split real/imag SoA (hand-written, NOT xsimd::batch<std::complex>,
// which is slow): one chunk is either a full xsimd::batch<T> of W consecutive j or
// a single scalar j (V = batch or T; identical body since xsimd overloads the
// operators), so the radix-R butterfly is written exactly once.
// ----------------------------------------------------------------------------

template<unsigned R, bool Forward, typename T, typename V>
ADM_ALWAYS_INLINE void bfly_chunk(const T* ADM_RESTRICT twre,
                                  const T* ADM_RESTRICT twim,
                                  T* ADM_RESTRICT yre, T* ADM_RESTRICT yim,
                                  std::size_t M, std::size_t j) {
    auto load = [](const T* p) -> V {
        if constexpr (std::is_same_v<V, T>) return *p;
        else return V::load_unaligned(p);
    };
    auto store = [](T* p, V v) {
        if constexpr (std::is_same_v<V, T>) *p = v;
        else v.store_unaligned(p);
    };

    // Gather one sample per sub-block, applying the inter-stage twiddle.
    V tr[R], ti[R];
    poet::static_for<0, static_cast<int>(R)>([&](auto Q) {
        constexpr unsigned q = Q;
        const V ar = load(yre + q * M + j);
        const V ai = load(yim + q * M + j);
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const V wr = load(twre + (q - 1) * M + j);
            const V wi = load(twim + (q - 1) * M + j);
            tr[q] = wr * ar - wi * ai;
            ti[q] = wr * ai + wi * ar;
        }
    });

    // Radix-R DFT across the R samples via the UNIFIED butterfly (shared with the
    // iterative-DIF path): odd R routes through the low-multiply symmetric kernel
    // (radix_sym_dft, halves the multiplies), pow2 R through the split-radix kernel
    // (pow2_dif_butterfly, twiddle elision), the rest through the naive matrix.
    // emit carries the output index l, so the store layout y[l*M + j] is preserved
    // regardless of emission order.
    dif_butterfly<T, Forward, static_cast<int>(R), V>(tr, ti,
        [&](auto L, V outr, V outi) {
            store(yre + L * M + j, outr);
            store(yim + L * M + j, outi);
        });
}

// Vectorized radix-R combine over the full j range, table-driven twiddles.
template<unsigned R, typename T, bool Forward>
ADM_ALWAYS_INLINE void radix_butterfly_v(T* ADM_RESTRICT yre, T* ADM_RESTRICT yim,
                                         std::size_t M,
                                         const T* ADM_RESTRICT twre,
                                         const T* ADM_RESTRICT twim) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    // Spill-aware unroll. A radix-R combine holds ~2R+10 live batches (the 2R-wide
    // gather `tr/ti` plus the butterfly's pair-sums/diffs/accumulators), so the
    // unroll factor U = registers / peak_live (dif_pass_unroll — the same model the
    // iterative DIF passes use) is the largest batch block that still fits the
    // vector register file. On AVX2 (16 YMM) this is U=1 for every catalog radix:
    // an R-input DFT's live set already exceeds the file, so unrolling would only
    // multiply the unavoidable spills; on AVX-512 (32 ZMM) the cheap radices get
    // U>1. dynamic_for<U> emits the U-wide unrolled block plus a compile-time-
    // dispatched leftover of < U full batches; the sub-W remainder stays a scalar
    // loop so we never load past M when M is not a multiple of W.
    constexpr std::size_t U = dif_pass_unroll<static_cast<int>(R)>();
    const std::size_t nfull = M / W;
    // Force-inline the chunk lambda only while the combine's live set (2R+10
    // batches) roughly fits the vector file: R<=4 (kills the call + push %rbp
    // in the hot loop); R=8 — the spilling combine inlined into the parent
    // merges its live range with the caller's, so the out-of-line clone is the
    // better allocation there (same rationale as kernel_should_noinline).
    if constexpr (2u * R + 10u <= poet::vector_register_count() + 2u) {
        poet::dynamic_for<U>(std::size_t{0}, nfull, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
            bfly_chunk<R, Forward, T, batch>(twre, twim, yre, yim, M, b * W);
        });
    } else {
        poet::dynamic_for<U>(std::size_t{0}, nfull, [&](std::size_t b) {
            bfly_chunk<R, Forward, T, batch>(twre, twim, yre, yim, M, b * W);
        });
    }
    // Sized-batch tail: cover the sub-W remainder with progressively narrower
    // ISA batches (W/2, W/4, ..., 2) before falling back to scalar columns —
    // each width fires at most once since the remainder halves. Widths without
    // a native batch (make_sized_batch_t void, e.g. 2-wide f32) fall through.
    // At M=15 this turns the f32 W=16 combine from 15 scalar radix-R DFTs into
    // 8+4-wide batches + 3 scalars (f64 W=8: 8+4+2+1).
    std::size_t j = nfull * W;
    poet::static_for<1, std::bit_width(W)>([&](auto S) {
        constexpr std::size_t Wt = W >> S;
        using Vt = xsimd::make_sized_batch_t<T, Wt>;
        if constexpr (Wt >= 2 && !std::is_void_v<Vt>) {
            if (j + Wt <= M) {
                bfly_chunk<R, Forward, T, Vt>(twre, twim, yre, yim, M, j);
                j += Wt;
            }
        }
    });
    for (; j < M; ++j) {
        bfly_chunk<R, Forward, T, T>(twre, twim, yre, yim, M, j);
    }
}

// consteval generation of the inter-stage twiddle table for a compile-time size
// N with radix R: W_N^{q j} = exp(sign 2*pi*i q j / N), laid out as (q-1)*M + j for
// q in [1, R), j in [0, M = N/R). Part selects real (false) or imag (true). The
// result is a constexpr std::array, so kernel<N> carries no runtime trig.
template<unsigned N, unsigned R, bool Forward, typename T, bool Imag>
consteval std::array<T, (R - 1) * (N / R)> make_twiddle_table() {
    std::array<T, (R - 1) * (N / R)> a{};
    constexpr unsigned M = N / R;
    constexpr int sign = Forward ? -1 : 1;
    for (unsigned q = 1; q < R; ++q) {
        for (unsigned j = 0; j < M; ++j) {
            const ct_sincos_t w = ct_sincos_turns(
                sign * static_cast<long long>(q) * static_cast<long long>(j),
                static_cast<long long>(N));
            a[(q - 1) * M + j] = Imag ? static_cast<T>(w.s) : static_cast<T>(w.c);
        }
    }
    return a;
}

// ----------------------------------------------------------------------------
// Rader prime codelet (chiplet): a size-p DFT (prime p > 11) has no small-radix
// factorization, so the generic recursion would emit a flat O(p^2) radix-p
// butterfly that spills. Rader's algorithm instead expresses it as a length-(p-1)
// cyclic CONVOLUTION, which we evaluate with the existing small-radix kernel<p-1>
// codelets — i.e. the prime codelet is itself a *combination of small transforms*.
//
// With g a primitive root mod p:  a_q = x[g^q],   b'_j = w_p^{g^{-j}},
//   X[g^{-m}] - x[0] = (a (*) b')_m = IDFT_{p-1}( DFT_{p-1}(a) . DFT_{p-1}(b') ),
//   X[0] = sum_n x[n].
// DFT_{p-1}(b') is data-independent, so it is baked consteval here (make_rader_bhat);
// the two length-(p-1) transforms at runtime are forward/inverse kernel<p-1>.
// ----------------------------------------------------------------------------

// Consteval forward DFT_{p-1} of the Rader b' sequence (the convolution kernel),
// real part (Imag=false) or imag part (Imag=true). Pure compile-time double math;
// no runtime trig, no heap (mirrors make_twiddle_table).
template<unsigned P, bool Forward, typename T, bool Imag>
consteval std::array<T, P - 1> make_rader_bhat() {
    constexpr unsigned L = P - 1;
    constexpr unsigned g = ct_primitive_root(P);
    constexpr int sp = Forward ? -1 : 1;     // sign of w_p in b'
    struct cd { double re; double im; };
    const unsigned ginv = ct_powmod(g, P - 2, P);   // g^{-1} mod p
    std::array<cd, L> bp{};
    for (unsigned j = 0; j < L; ++j) {
        const unsigned e = ct_powmod(ginv, j, P);    // g^{-j} mod p
        const ct_sincos_t w = ct_sincos_turns(sp * static_cast<long long>(e),
                                              static_cast<long long>(P));
        bp[j] = {w.c, w.s};                           // exp(sp 2*pi*i e/p)
    }
    std::array<T, L> out{};
    for (unsigned k = 0; k < L; ++k) {
        double sr = 0.0, si = 0.0;
        for (unsigned j = 0; j < L; ++j) {
            // exp(-2*pi*i j k / L): c=cos, s=sin(-angle)
            const ct_sincos_t e = ct_sincos_turns(-static_cast<long long>((j * k) % L),
                                                  static_cast<long long>(L));
            sr += bp[j].re * e.c - bp[j].im * e.s;
            si += bp[j].re * e.s + bp[j].im * e.c;
        }
        out[k] = Imag ? static_cast<T>(si) : static_cast<T>(sr);
    }
    return out;
}

// ----------------------------------------------------------------------------
// Cofactor-SIMD batched codelet (kernel_batched / rader_apply_batched).
//
// A single small/prime DFT has no batch dimension, so kernel<p> (Rader) compiles
// to scalar code with heavy spills. But when a composite N = r * M is decomposed
// with r == the SIMD width W, the r sub-transforms of size M are INDEPENDENT and
// the DIT-decimated input x[r*j + q] is already contiguous in lane order. So the
// r = W sub-transforms run as ONE W-wide batched size-M transform: every scalar
// becomes an xsimd::batch<T> with one independent transform per lane, inter-stage
// twiddles broadcast across lanes (they depend only on the frequency j, not the
// transform index), and loads are contiguous (no gather). This vectorizes the
// entire sub-transform — Rader chiplet included — across the W cofactor copies.
// kernel<N>::apply scatters the lanes back into the r contiguous output blocks and
// runs the unchanged radix-r combine.
//
// One lane = one independent transform. The combine here is vectorized over
// LANES (one j per call, twiddle broadcast), NOT over consecutive j as in
// radix_butterfly_v — but it reuses the same V-generic dif_butterfly.
//
// The batch type is NOT fixed to the native width: with r < W most lanes would
// sit idle (e.g. r=2 on an 8-wide f32 register = 75% waste). cofactor_batch_width
// instead probes, at compile time, the narrowest ISA register wide enough to hold
// the r cofactor sub-transforms (xsimd::make_sized_batch_t / is_void over the
// power-of-two widths), so r=2 f64 runs on a 2-wide SSE batch (full) and r=4 f32
// on a 4-wide SSE batch (full) — combining ISAs to minimize idle lanes. When the
// chosen width still exceeds r (e.g. r=3), the load is a compile-time masked load
// (batch_bool_constant: the W-r high lanes are never touched) and those lanes
// carry zeros harmlessly through the linear transform.
// ----------------------------------------------------------------------------

// Narrowest power-of-two SIMD width >= R for which some supported ISA provides a
// native batch, capped at the native width. This is the batch width the cofactor
// path runs at: it packs the R independent sub-transforms into the smallest
// register that holds them, so no wider-ISA lanes idle. R <= native width is a
// precondition (the caller only takes this path when r <= xsimd::batch<T>::size),
// and the native width is always available, so the recursion always terminates.
// W is walked as a template argument (not a mutable local) so make_sized_batch_t
// can probe each candidate width; it is only queried for W < native width.
template<typename T, std::size_t R, std::size_t W = 1>
[[nodiscard]] consteval std::size_t cofactor_batch_width() {
    if constexpr (W >= xsimd::batch<T>::size)
        return xsimd::batch<T>::size;  // native width is always available — stop here
    else if constexpr (W >= R && !std::is_void_v<xsimd::make_sized_batch_t<T, W>>)
        return W;                      // narrowest available width that holds R
    else
        return cofactor_batch_width<T, R, W * 2>();
}

// Radix-R combine for the batched layout: one frequency j, twiddles broadcast.
// Outputs go through `sink(p, outr, outi)` (p = flat output index) so the
// TOP-LEVEL combine of a batched codelet can store anywhere (e.g. the DP
// terminal interleaves straight to AoS `data`) instead of round-tripping
// through the yre/yim output arrays. The plain overloads below store to
// yre/yim, preserving the original in-place behavior.
template<unsigned R, bool Forward, typename T, typename V, typename Sink>
ADM_ALWAYS_INLINE void bfly_chunk_batched(const T* ADM_RESTRICT twre,
                                          const T* ADM_RESTRICT twim,
                                          const V* ADM_RESTRICT yre,
                                          const V* ADM_RESTRICT yim,
                                          std::size_t M, std::size_t j, Sink&& sink) {
    V tr[R], ti[R];
    poet::static_for<0, static_cast<int>(R)>([&](auto Q) {
        constexpr unsigned q = Q;
        const V ar = yre[q * M + j];
        const V ai = yim[q * M + j];
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const V wr(twre[(q - 1) * M + j]);  // broadcast scalar twiddle to all lanes
            const V wi(twim[(q - 1) * M + j]);
            tr[q] = wr * ar - wi * ai;
            ti[q] = wr * ai + wi * ar;
        }
    });
    dif_butterfly<T, Forward, static_cast<int>(R), V>(tr, ti,
        [&](auto L, V outr, V outi) { sink(L * M + j, outr, outi); });
}

template<unsigned R, bool Forward, typename T, typename V = xsimd::batch<T>>
ADM_ALWAYS_INLINE void bfly_chunk_batched(const T* ADM_RESTRICT twre,
                                          const T* ADM_RESTRICT twim,
                                          V* ADM_RESTRICT yre,
                                          V* ADM_RESTRICT yim,
                                          std::size_t M, std::size_t j) {
    bfly_chunk_batched<R, Forward, T, V>(twre, twim, yre, yim, M, j,
        [&](std::size_t p, V outr, V outi) {
            yre[p] = outr;
            yim[p] = outi;
        });
}

template<unsigned R, typename T, bool Forward, typename V, typename Sink>
ADM_ALWAYS_INLINE void radix_butterfly_batched(const V* ADM_RESTRICT yre,
                                               const V* ADM_RESTRICT yim,
                                               std::size_t M,
                                               const T* ADM_RESTRICT twre,
                                               const T* ADM_RESTRICT twim,
                                               Sink&& sink) {
    for (std::size_t j = 0; j < M; ++j)
        bfly_chunk_batched<R, Forward, T, V>(twre, twim, yre, yim, M, j, sink);
}

template<unsigned R, typename T, bool Forward, typename V = xsimd::batch<T>>
ADM_ALWAYS_INLINE void radix_butterfly_batched(V* ADM_RESTRICT yre,
                                               V* ADM_RESTRICT yim,
                                               std::size_t M,
                                               const T* ADM_RESTRICT twre,
                                               const T* ADM_RESTRICT twim) {
    for (std::size_t j = 0; j < M; ++j)
        bfly_chunk_batched<R, Forward, T, V>(twre, twim, yre, yim, M, j);
}

// Rader prime codelet, batched over W lanes (one independent size-P DFT per lane).
// Mirror of rader_apply with T -> xsimd::batch<T> and the data-independent Bhat
// broadcast across lanes. `xstride` strides the batch-array index. Output yre/yim
// is the contiguous size-P block.
template<unsigned P, typename T, bool Forward, typename V = xsimd::batch<T>>
void rader_apply_batched(const V* xre, const V* xim, std::size_t xstride,
                         V* yre, V* yim);

// kernel_batched<N, T, Forward, V>: V::size independent size-N DFTs, one per SIMD
// lane. Same compile-time recursion as kernel<N> but every element is a batch V.
// V defaults to the native batch; the cofactor-SIMD call site instead picks the
// narrowest ISA batch wide enough for the cofactor count (see cofactor_batch_width).
template<unsigned N, typename T, bool Forward, typename V = xsimd::batch<T>>
struct kernel_batched {
    static constexpr unsigned r = codelet_radix(N);
    static constexpr unsigned M = N / r;
    static_assert(r * M == N, "codelet_radix(N) must divide N exactly");

    // Cut the cofactor recursion at small power-of-two leaves: emit a single flat,
    // register-resident dif_butterfly instead of recursing N->...->1 and writing
    // every level's result to the yre/yim OUTPUT memory only to read it back. Those
    // inter-level round-trips are pure dispatch-bound uops (the leaf is
    // front-end/uop-throughput-bound, not FP-port-bound).
    //
    // Threshold is register-budget-derived (ISA-adaptive, like dif_pass_unroll): a
    // flat size-N leaf holds its 2N input batches (N re + N im) live, so it stays in
    // the vector file when 2N <= vector_register_count() -- N<=8 on AVX2 (16 YMM),
    // N<=16 on AVX-512 (32 ZMM). Beyond that a flat leaf spills catastrophically.
    static constexpr bool flat_leaf =
        (N & (N - 1)) == 0 && N >= 2 && 2u * N <= poet::vector_register_count();

    static void apply(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim) {
        if constexpr (is_rader_prime(N)) {
            rader_apply_batched<N, T, Forward, V>(xre, xim, xstride, yre, yim);
            return;
        } else if constexpr (flat_leaf) {
            V tr[N], ti[N];
            poet::static_for<0, static_cast<int>(N)>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly<T, Forward, static_cast<int>(N), V>(tr, ti, [&](auto K, V re, V im) {
                yre[K] = re;
                yim[K] = im;
            });
        } else {
            poet::static_for<0, static_cast<int>(r)>([&](auto Q) {
                constexpr unsigned q = Q;
                kernel_batched<M, T, Forward, V>::apply(xre + q * xstride, xim + q * xstride,
                                                        xstride * r, yre + q * M, yim + q * M);
            });
            static constexpr auto twre = make_twiddle_table<N, r, Forward, T, false>();
            static constexpr auto twim = make_twiddle_table<N, r, Forward, T, true>();
            radix_butterfly_batched<r, T, Forward, V>(yre, yim, M, twre.data(), twim.data());
        }
    }

    // Same transform, but the FINAL combine's outputs go through
    // `sink(p, outr, outi)` (p = output index in [0,N)) instead of being stored
    // to yre/yim — the caller fuses its own store (scale + AoS interleave) into
    // the last butterfly and skips the output-array round-trip. yre/yim are
    // still required as scratch for the sub-transform results. Every output
    // index is sinked exactly once.
    template<typename Sink>
    static void apply_sink(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim,
                           Sink&& sink) {
        if constexpr (is_rader_prime(N)) {
            // Rader writes through data-dependent index maps; no last-combine to
            // hook. Run it whole, then forward the outputs.
            rader_apply_batched<N, T, Forward, V>(xre, xim, xstride, yre, yim);
            poet::static_for<0, static_cast<int>(N)>([&](auto P) { sink(P, yre[P], yim[P]); });
        } else if constexpr (flat_leaf) {
            V tr[N], ti[N];
            poet::static_for<0, static_cast<int>(N)>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly<T, Forward, static_cast<int>(N), V>(tr, ti,
                [&](auto K, V re, V im) { sink(K, re, im); });
        } else {
            poet::static_for<0, static_cast<int>(r)>([&](auto Q) {
                constexpr unsigned q = Q;
                kernel_batched<M, T, Forward, V>::apply(xre + q * xstride, xim + q * xstride,
                                                        xstride * r, yre + q * M, yim + q * M);
            });
            static constexpr auto twre = make_twiddle_table<N, r, Forward, T, false>();
            static constexpr auto twim = make_twiddle_table<N, r, Forward, T, true>();
            radix_butterfly_batched<r, T, Forward, V>(yre, yim, M, twre.data(), twim.data(),
                                                      sink);
        }
    }
};

template<typename T, bool Forward, typename V>
struct kernel_batched<1, T, Forward, V> {
    static void apply(const V* xre, const V* xim, std::size_t /*xstride*/, V* yre, V* yim) {
        yre[0] = xre[0];
        yim[0] = xim[0];
    }
};

// Forward declaration: the Rader codelet calls forward/inverse kernel<p-1>, so it
// is defined after kernel<N> (like kernel_apply_boundary).
template<unsigned P, typename T, bool Forward>
void rader_apply(const T* xre, const T* xim, std::size_t xstride, T* yre, T* yim);

// ----------------------------------------------------------------------------
// kernel<N, T, Forward>: size-N DFT, planar split real/imag.
//
//   apply(xre, xim, xstride, yre, yim)
//
// Reads N complex inputs x[k] = (xre, xim)[k * xstride], k in [0, N), and writes
// the DFT to the contiguous planar block (yre, yim)[0..N). Output must not alias
// input. `Forward == true` uses exp(-2*pi*i*kn/N); inverse uses +.
//
// This is the fully compile-time-recursive form (all radices and the recursion
// unrolled). It is the building block validated by tests; the runtime driver
// reuses radix_butterfly directly for bounded instantiation on large N.
// ----------------------------------------------------------------------------

// Should the recursive call into a size-M sub-transform be a register-allocation
// BARRIER (a noinline call) rather than inlined into its caller? Two conditions,
// both required:
//   (1) the sub-transform's own radix-r combine already spills — its live set
//       peak_live = 2r+10 exceeds the vector register file
//       (poet::vector_register_count(): 16 YMM on AVX2, 32 ZMM on AVX-512); and
//   (2) the sub-transform is large enough (M >= kNoinlineMinSize) that its work
//       amortizes the call overhead — at tiny leaves the call cost erases the win.
// When (1) holds but the level is inlined, the child's spilling live set
// accumulates into the parent's; the noinline boundary caps each level to its own
// allocation. Below the threshold the level stays fully inlined (current codegen).
inline constexpr unsigned kNoinlineMinSize = 16;

// Policy constant: below the SIMD width there is no batched work to amortize
// the call, and above the ≤64 catalog max the boundary could never fire.
static_assert(kNoinlineMinSize >= xsimd::batch<float>::size && kNoinlineMinSize <= 64,
              "kNoinlineMinSize must sit between the SIMD width and the catalog max");

[[nodiscard]] consteval bool kernel_should_noinline(unsigned M) {
    return M >= kNoinlineMinSize
        && 2u * codelet_radix(M) + 10u > poet::vector_register_count();
}

// Cofactor-SIMD profitability for the peeled cofactor M and radix R (codelet.hpp
// kernel<N> path). The lane-batched path runs the R size-M sub-transforms across the
// lanes of an Wc-wide batch; it is profitable when kernel<M> would otherwise be
// scalar-dominated AND the batch is wide and well-utilized enough to pay for the
// load/transpose/scatter machinery. Eligible M:
//   * Rader primes (M>11 prime): an expensive straight-line chiplet; batching the R
//     copies across lanes amortizes the per-copy scalar cost. Subject to the SAME
//     low-half-mask gate as the odd composites (below): the narrow f32 R=2 (Wc=4,
//     2 idle lanes) batch is NOT profitable — the scalar rader_apply<M> recursion
//     runs its kernel<M-1> conv at full ymm width.
//   * ODD composite / small odd prime M (3,5,7,9,11,15,21,27,...): no radix-2/4 factor,
//     so kernel<M>'s combine has a tiny frequency count and runs essentially scalar —
//     batching the R copies vectorizes it. Even/smooth M (2,4,6,8,...) already vectorize
//     in the scalar-recursion path, so routing them through Wc would only waste lanes.
// The width/utilization gate: the masked load used when Wc > R (the R sub-transforms
// fill only the low R of Wc lanes) — a low-half mask (active lanes <= Wc/2) leaves
// half the batch idle. Contiguous SSE masks load natively (movss/movlps), but
// the half-idle-batch cost still gates.
// So for the odd-composite branch:
//   * Full batch (Wc==R): no mask.
//   * Low-half mask (2*R <= Wc, i.e. f32 R=2 -> Wc=4): reject — half-idle batch.
//     Rader primes also rejected here (scalar rader_apply<M> path preferred —
//     see the is_rader_prime branch).
//   * 128-bit hardware-masked load (Wc*sizeof<=16, f32 R=3): cheap.
//   * 256-bit hardware-masked load (f64 R=3, f32 R=5/7): the wider load's fixed cost
//     needs enough work to amortize — require R*M >= 27.
template<typename T, unsigned R>
[[nodiscard]] consteval bool cofactor_simd_profitable(unsigned M) {
    if (is_rader_prime(M)) {
        // Rader-prime cofactor: batch the R copies across lanes (below) UNLESS the
        // batch is the low-half-mask case (f32 R=2 -> Wc=4, 2 idle lanes), where the
        // scalar rader_apply<M> recursion runs its inner kernel<M-1> conv at full ymm
        // width and is preferred.
        constexpr std::size_t Wc = cofactor_batch_width<T, R>();
        if constexpr (2u * R <= Wc) return false;  // f32 R=2: prefer scalar rader_apply (full ymm)
        return true;
    }
    // Power-of-two M is eligible when the batch is FULL (Wc == R) AND the
    // kernel_batched<M> recursion does not hit the AVX2 register-file ceiling:
    //
    //   M == 4: flat leaf (2*4=8 live batches), 8 < 16 regs.
    //           kernel<16>'s scalar path is 'fully scalar' (no radix-2/4 factor);
    //           routing through kernel_batched<4> vectorizes it.
    //
    //   M == 16: NOT a flat leaf (2*16=32 > 16 AVX2 regs → recurses into
    //            kernel_batched<4>); peak live = 8 (flat-leaf pass) or 12
    //            (radix-4 butterfly), both under 16 AVX2 regs.  N=64
    //            (kernel<64>, r=4, M=16) must route through kernel_batched<16>
    //            to vectorize all four sub-transforms: a direct kernel<64> at
    //            xstride=4 takes the scalar path (cofactor SIMD's xstride==1
    //            gate is never reached from kernel<64>).
    //
    //   M == 8: flat leaf holds 2*8=16 live batches — the ENTIRE 16-reg AVX2
    //           file (twiddle/address spill), but only half of a 32-reg AVX-512
    //           file. Gate on register headroom (2*M < regs) instead of
    //           excluding outright: AVX2 behavior unchanged (16 < 16 false),
    //           32-reg ISAs admit it.
    //
    //   M >= 32 EXCLUDED: kernel_batched<32> uses r=codelet_radix(32)=8;
    //           radix-8 butterfly (bfly_chunk_batched<8>) holds 16 live batches
    //           = same ceiling; excluded by the same register model.
    //
    // !kernel_should_noinline(M) is NOT applied to pow2_m: that predicate
    // models scalar spill pressure (live scalar scalars > register file); in
    // the batched path each 'register' is a SIMD batch and the peak live count
    // is bounded differently (see above).  odd_m keeps the guard because the
    // scalar recursion model still applies there.
    const bool odd_m = M >= 3u && (M % 2u != 0u) && !kernel_should_noinline(M);
    const bool pow2_m = (M == 4u || M == 16u ||
                         (M == 8u && 2u * M < poet::vector_register_count())) &&
                        cofactor_batch_width<T, R>() == R;
    if (!(odd_m || pow2_m)) return false;
    constexpr std::size_t Wc = cofactor_batch_width<T, R>();
    if (Wc == R) return true;                  // full batch, no masked load: always wins
    if (2u * R <= Wc) return false;            // low-half mask: half-idle batch
    if (Wc * sizeof(T) <= 16u) return true;    // 128-bit hardware-masked load: cheap
    return R * M >= 27u;                        // 256-bit masked load: needs enough work
}

// Peel radix for kernel<N>: prefer a power-of-two r whose cofactor M = N/r is
// ODD, so the cofactor-SIMD path below fires (kernel_batched<M> across r lanes)
// instead of the scalar recursion an even non-pow2 cofactor falls into. Gated on
// the ISA having an r-lane batch and on cofactor_simd_profitable — where either
// gate fails this is exactly codelet_radix_for (no behavior change). Pow2 N are
// unaffected: their N/r cofactors are even (or M=1, rejected by the
// profitability predicate).
template<typename T>
[[nodiscard]] consteval unsigned kernel_peel_radix(unsigned N) {
    constexpr unsigned W = static_cast<unsigned>(xsimd::batch<T>::size);
    if (N % 8u == 0u && (N / 8u) % 2u == 1u && 8u <= W && cofactor_simd_profitable<T, 8u>(N / 8u))
        return 8u;
    if (N % 4u == 0u && (N / 4u) % 2u == 1u && 4u <= W && cofactor_simd_profitable<T, 4u>(N / 4u))
        return 4u;
    return codelet_radix_for<T>(N);
}

// Forward declaration of the noinline recursion boundary (defined after kernel<N>
// so it can name kernel<N>::apply). The attribute lives on this declaration.
template<unsigned N, typename T, bool Forward>
ADM_NOINLINE void kernel_apply_boundary(const T* xre, const T* xim, std::size_t xstride,
                                        T* yre, T* yim);

template<unsigned N, typename T, bool Forward>
struct kernel {
    // Precision- and width-aware: codelet_radix_for overridden by the odd-cofactor
    // pow2 peel where the cofactor-SIMD path fires (see kernel_peel_radix).
    static constexpr unsigned r = kernel_peel_radix<T>(N);
    static constexpr unsigned M = N / r;
    // The factorization must be exact: a non-dividing peeled radix would silently
    // drop or duplicate sub-transform inputs. codelet_radix is a fixpoint divisor
    // of N (or N itself for a Rader prime, giving M==1), so this always holds —
    // the assert pins the planner contract at compile time for every catalog N.
    static_assert(r * M == N, "codelet_radix(N) must divide N exactly");

    static void apply(const T* xre, const T* xim, std::size_t xstride,
                      T* yre, T* yim) {
        // Prime > 11: no small-radix factorization — use the Rader chiplet
        // (length-(N-1) convolution via kernel<N-1>) instead of a flat radix-N
        // unroll. if constexpr discards the recursion+twiddle-table machinery
        // below for these sizes (codelet_radix(N) would be N, M would be 1).
        if constexpr (is_rader_prime(N)) {
            rader_apply<N, T, Forward>(xre, xim, xstride, yre, yim);
            return;
        }
        // Cut the recursion at a small power-of-two leaf: emit a flat,
        // register-resident dif_butterfly instead of recursing N->...->1 and writing
        // every level's result to the yre/yim OUTPUT memory only to read it back (the
        // same dispatch-bound inter-level round-trip, and the same fix, as
        // kernel_batched above). Register-budget threshold (2N scalar values live):
        // N<=8 on AVX2, N<=16 on AVX-512.
        if constexpr ((N & (N - 1)) == 0 && N >= 2 && 2u * N <= poet::vector_register_count()) {
            T tr[N], ti[N];
            poet::static_for<0, static_cast<int>(N)>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly<T, Forward, static_cast<int>(N), T>(tr, ti, [&](auto K, T re, T im) {
                yre[K] = re;
                yim[K] = im;
            });
            return;
        }
        // Cofactor-SIMD path: when the cofactor M is cofactor_simd_eligible (a Rader
        // prime, or an odd M whose kernel<M> is scalar-dominated) and the peeled radix
        // r (<= SIMD width W) gives r INDEPENDENT size-M sub-transforms, run them as ONE
        // batched size-M transform (one sub-transform per lane, r of Wc lanes used) —
        // vectorizing the otherwise-scalar sub-transform across the r copies — then
        // scatter lanes into the r output blocks and combine as usual. At unit input
        // stride the r decimated inputs x[r*j .. r*j+r-1] are contiguous, so when
        // r == Wc the load is a single contiguous batch; for r < Wc the r values are
        // packed into the low lanes (safe, no OOB at the tail). Fires for the Rader-prime
        // composites plus the odd-cofactor composites. Unit stride holds: such N are
        // never a sub-transform here.
        if constexpr (r <= xsimd::batch<T>::size && cofactor_simd_profitable<T, r>(M)) {
            // Run the r sub-transforms in the narrowest ISA register that holds
            // them (see cofactor_batch_width): r=2 f64 -> 2-wide, r=4 f32 -> 4-wide,
            // etc. Only the low r lanes carry data; if Wc > r they are filled by a
            // compile-time masked load (the high lanes are never read).
            constexpr std::size_t Wc = cofactor_batch_width<T, r>();
            using V = xsimd::make_sized_batch_t<T, Wc>;
            static_assert(!std::is_void_v<V>, "no SIMD batch wide enough for cofactor r");
            static_assert(Wc >= r, "cofactor batch must hold all r sub-transforms");
            using A = typename V::arch_type;
            // Generator: lanes [0, r) hold a sub-transform; [r, Wc) are masked off.
            struct lane_active { static constexpr bool get(std::size_t i, std::size_t) { return i < r; } };
            if (xstride == 1) {
                // ovr/ovi are padded to a whole number of Wc-tiles so the leftover
                // M%Wc columns can go through the same in-register transpose as the
                // full tiles (masked row stores) instead of per-element extracts.
                // Pad only when the masked-tile path below is taken (M >= Wc).
                constexpr std::size_t Mp =
                    (M % Wc != 0 && M >= Wc) ? ((M / Wc + 1) * Wc) : M;
                V xv[M], iv[M], ovr[Mp], ovi[Mp];
                poet::static_for<0, static_cast<int>(Mp - M)>([&](auto P) {
                    // Guard the write: static_for's is_invocable_v probe
                    // instantiates this body with P=0 even when Mp==M (zero-trip),
                    // and ovr[M] is then out of bounds — clang -Warray-bounds errors.
                    if constexpr (M + P < Mp) {
                        ovr[M + P] = V(T(0));
                        ovi[M + P] = V(T(0));
                    }
                });
                for (std::size_t j = 0; j < M; ++j) {
                    if constexpr (Wc == r) {
                        xv[j] = V::load_unaligned(xre + r * j);  // full register: contiguous load
                        iv[j] = V::load_unaligned(xim + r * j);
                    } else {
                        constexpr auto mask = xsimd::make_batch_bool_constant<T, lane_active, A>();
                        xv[j] = V::load(xre + r * j, mask, xsimd::unaligned_mode{});
                        iv[j] = V::load(xim + r * j, mask, xsimd::unaligned_mode{});
                    }
                }
                kernel_batched<M, T, Forward, V>::apply(xv, iv, 1, ovr, ovi);
                // Scatter: lane q (sub-transform q) of ovr[k] -> output yre[q*M + k].
                // The M output batches form an M×Wc matrix whose columns are the r
                // output blocks; emitting them transposed turns N scalar stores into
                // one in-register Wc×Wc transpose (xsimd::transpose = vunpck/vperm,
                // NOT vmaskmov) per Wc-tile + a full-width store per output row. Only
                // the low r rows are real (lanes [r,Wc) are masked-off idle), so we
                // store rows [0,r) and skip the idle ones; the M%Wc leftover columns
                // (< Wc, runs once) stay scalar. xsimd::transpose needs exactly Wc
                // batches, so it runs on each contiguous &ovr[k0..k0+Wc) tile.
                constexpr std::size_t nft = M / Wc;  // number of full Wc-wide tiles
                for (std::size_t t = 0; t < nft; ++t) {
                    const std::size_t k0 = t * Wc;
                    xsimd::transpose(&ovr[k0], &ovr[k0] + Wc);
                    xsimd::transpose(&ovi[k0], &ovi[k0] + Wc);
                    poet::static_for<0, static_cast<int>(r)>([&](auto Q) {
                        constexpr unsigned q = Q;
                        ovr[k0 + q].store_unaligned(yre + q * M + k0);
                        ovi[k0 + q].store_unaligned(yim + q * M + k0);
                    });
                }
                if constexpr (M % Wc != 0 && M >= Wc) {
                    // Leftover columns: transpose the padded final tile and store
                    // each of the r real rows under a compile-time column mask
                    // (M%Wc active lanes) — masks feeding a transpose are the
                    // sanctioned exception to the sized-batch tail policy. Gated on
                    // at least one full tile existing (M >= Wc): for tile-less tiny
                    // cofactors (e.g. M=3, Wc=4 inside rader<13>'s kernel<12>) the
                    // pad+transpose+mask chain loses to the unrolled scalar scatter.
                    struct col_tail_active {
                        static constexpr bool get(std::size_t i, std::size_t) { return i < M % Wc; }
                    };
                    constexpr auto smask = xsimd::make_batch_bool_constant<T, col_tail_active, A>();
                    const std::size_t k0 = nft * Wc;
                    xsimd::transpose(&ovr[k0], &ovr[k0] + Wc);
                    xsimd::transpose(&ovi[k0], &ovi[k0] + Wc);
                    poet::static_for<0, static_cast<int>(r)>([&](auto Q) {
                        constexpr unsigned q = Q;
                        ovr[k0 + q].store(yre + q * M + k0, smask, xsimd::unaligned_mode{});
                        ovi[k0 + q].store(yim + q * M + k0, smask, xsimd::unaligned_mode{});
                    });
                } else if constexpr (M % Wc != 0) {
                    T sr[Wc], si[Wc];
                    for (std::size_t k = nft * Wc; k < M; ++k) {
                        ovr[k].store_unaligned(sr);
                        ovi[k].store_unaligned(si);
                        poet::static_for<0, static_cast<int>(r)>([&](auto Q) {
                            constexpr unsigned q = Q;
                            yre[q * M + k] = sr[q];
                            yim[q * M + k] = si[q];
                        });
                    }
                }
                static constexpr auto twre = make_twiddle_table<N, r, Forward, T, false>();
                static constexpr auto twim = make_twiddle_table<N, r, Forward, T, true>();
                radix_butterfly_v<r, T, Forward>(yre, yim, M, twre.data(), twim.data());
                return;
            }
        }
        // Recurse: r sub-DFTs of size M, decimated by residue mod r in the input,
        // written into r contiguous output blocks of length M. Above the
        // register-pressure threshold the sub-transform is routed through a
        // noinline boundary so its spills get an independent allocation.
        poet::static_for<0, static_cast<int>(r)>([&](auto Q) {
            constexpr unsigned q = Q;
            if constexpr (kernel_should_noinline(M)) {
                kernel_apply_boundary<M, T, Forward>(xre + q * xstride, xim + q * xstride,
                                                     xstride * r, yre + q * M, yim + q * M);
            } else {
                kernel<M, T, Forward>::apply(xre + q * xstride, xim + q * xstride,
                                             xstride * r, yre + q * M, yim + q * M);
            }
        });
        // Combine the r blocks with the generic radix-r butterfly, using the
        // compile-time-folded inter-stage twiddle table for this size (no runtime
        // trig, no heap). For M == 1 (e.g. prime direct DFT) the table is all-ones
        // and the combine is a pure radix-N DFT.
        static constexpr auto twre = make_twiddle_table<N, r, Forward, T, false>();
        static constexpr auto twim = make_twiddle_table<N, r, Forward, T, true>();
        radix_butterfly_v<r, T, Forward>(yre, yim, M, twre.data(), twim.data());
    }
};

// Base case: size-1 DFT is the identity (a strided copy).
template<typename T, bool Forward>
struct kernel<1, T, Forward> {
    static void apply(const T* xre, const T* xim, std::size_t /*xstride*/,
                      T* yre, T* yim) {
        yre[0] = xre[0];
        yim[0] = xim[0];
    }
};

// Noinline recursion boundary: a thin forwarder whose only job is to be a call
// the optimizer cannot inline, so the size-N sub-transform gets its own register
// allocation (declared above kernel<N>; see kernel_should_noinline).
template<unsigned N, typename T, bool Forward>
ADM_NOINLINE void kernel_apply_boundary(const T* xre, const T* xim, std::size_t xstride,
                                        T* yre, T* yim) {
    kernel<N, T, Forward>::apply(xre, xim, xstride, yre, yim);
}

// Compile-time g^q mod P table for the Rader index permutation (q=0..L-1, L=P-1).
// Shared by rader_apply and rader_apply_batched; templated only on P/L to avoid
// duplicate instantiations for different T/V/Forward combinations.
template<unsigned P, unsigned L = P - 1>
consteval std::array<unsigned, L> rader_gpow_table() {
    constexpr unsigned g = ct_primitive_root(P);
    std::array<unsigned, L> t{};
    unsigned e = 1;
    for (unsigned q = 0; q < L; ++q) {
        t[q] = e;
        e = static_cast<unsigned>(static_cast<unsigned long long>(e) * g % P);
    }
    return t;
}

// Compile-time g^{-m} mod P table for the Rader output scatter (m=0..L-1, L=P-1).
// Shared by rader_apply and rader_apply_batched.
template<unsigned P, unsigned L = P - 1>
consteval std::array<unsigned, L> rader_ginvpow_table() {
    constexpr unsigned g = ct_primitive_root(P);
    const unsigned gi = ct_powmod(g, P - 2, P);
    std::array<unsigned, L> t{};
    unsigned e = 1;
    for (unsigned m = 0; m < L; ++m) {
        t[m] = e;
        e = static_cast<unsigned>(static_cast<unsigned long long>(e) * gi % P);
    }
    return t;
}

// Rader prime codelet: size-P DFT (P prime > 11) as a length-(P-1) cyclic
// convolution evaluated with forward+inverse kernel<P-1>. noinline so the two
// sub-transforms + the gather/scatter get an isolated register allocation (the
// chiplet's working set is the kernel<P-1> live set, not P's).
template<unsigned P, typename T, bool Forward>
ADM_NOINLINE void rader_apply(const T* xre, const T* xim, std::size_t xstride,
                              T* yre, T* yim) {
    constexpr unsigned L = P - 1;

    // Input gather index g^q mod p and output scatter index g^{-m} mod p.
    static constexpr std::array<unsigned, L> gpow = rader_gpow_table<P>();
    static constexpr std::array<unsigned, L> ginvpow = rader_ginvpow_table<P>();
    static constexpr std::array<T, L> Bre = make_rader_bhat<P, Forward, T, false>();
    static constexpr std::array<T, L> Bim = make_rader_bhat<P, Forward, T, true>();

    // DC output X[0] = sum_n x[n]; gather a_q = x[g^q] (with input stride).
    const T x0r = xre[0];
    const T x0i = xim[0];
    T sumr = x0r, sumi = x0i;
    T are[L], aim[L];
    for (unsigned q = 0; q < L; ++q) {
        const std::size_t idx = static_cast<std::size_t>(gpow[q]) * xstride;
        are[q] = xre[idx];
        aim[q] = xim[idx];
        sumr += are[q];
        sumi += aim[q];
    }

    // A = DFT_L(a); P = A .* Bhat; c = IDFT_L(P) = (1/L) * inverse-kernel<L>(P).
    T Are[L], Aim[L];
    kernel<L, T, true>::apply(are, aim, 1, Are, Aim);

    // Pointwise complex multiply P = A .* Bhat, SIMD-vectorized over k in the
    // planar split real/imag layout (same hand-rolled SoA + FMA as the
    // butterflies; xsimd::batch<complex> is slow), with a scalar tail
    // for the sub-batch remainder of L.
    T Pre[L], Pim[L];
    using batch = xsimd::batch<T>;
    constexpr std::size_t Wd = batch::size;
    std::size_t k = 0;
    for (; k + Wd <= L; k += Wd) {
        const batch ar = batch::load_unaligned(&Are[k]);
        const batch ai = batch::load_unaligned(&Aim[k]);
        const batch br = batch::load_unaligned(&Bre[k]);
        const batch bi = batch::load_unaligned(&Bim[k]);
        (ar * br - ai * bi).store_unaligned(&Pre[k]);
        (ar * bi + ai * br).store_unaligned(&Pim[k]);
    }
    for (; k < L; ++k) {
        Pre[k] = Are[k] * Bre[k] - Aim[k] * Bim[k];
        Pim[k] = Are[k] * Bim[k] + Aim[k] * Bre[k];
    }

    T cre[L], cim[L];
    kernel<L, T, false>::apply(Pre, Pim, 1, cre, cim);

    const T invL = T(1) / T(L);
    yre[0] = sumr;
    yim[0] = sumi;
    for (unsigned m = 0; m < L; ++m) {
        yre[ginvpow[m]] = x0r + cre[m] * invL;
        yim[ginvpow[m]] = x0i + cim[m] * invL;
    }
}

// Batched (W-lane) Rader codelet: rader_apply with T -> xsimd::batch<T> and the
// data-independent Bhat sequence broadcast across lanes. One independent size-P
// DFT per lane; the forward/inverse length-(P-1) transforms use kernel_batched.
template<unsigned P, typename T, bool Forward, typename V>
ADM_NOINLINE void rader_apply_batched(const V* xre, const V* xim, std::size_t xstride,
                                      V* yre, V* yim) {
    constexpr unsigned L = P - 1;

    static constexpr std::array<unsigned, L> gpow = rader_gpow_table<P>();
    static constexpr std::array<unsigned, L> ginvpow = rader_ginvpow_table<P>();
    static constexpr std::array<T, L> Bre = make_rader_bhat<P, Forward, T, false>();
    static constexpr std::array<T, L> Bim = make_rader_bhat<P, Forward, T, true>();

    const V x0r = xre[0];
    const V x0i = xim[0];
    V sumr = x0r, sumi = x0i;
    V are[L], aim[L];
    for (unsigned q = 0; q < L; ++q) {
        const std::size_t idx = static_cast<std::size_t>(gpow[q]) * xstride;
        are[q] = xre[idx];
        aim[q] = xim[idx];
        sumr += are[q];
        sumi += aim[q];
    }

    V Are[L], Aim[L];
    kernel_batched<L, T, true, V>::apply(are, aim, 1, Are, Aim);

    // Pointwise complex multiply P = A .* Bhat with Bhat broadcast across lanes.
    V Pre[L], Pim[L];
    for (unsigned k = 0; k < L; ++k) {
        const V br(Bre[k]);
        const V bi(Bim[k]);
        Pre[k] = Are[k] * br - Aim[k] * bi;
        Pim[k] = Are[k] * bi + Aim[k] * br;
    }

    V cre[L], cim[L];
    kernel_batched<L, T, false, V>::apply(Pre, Pim, 1, cre, cim);

    const T invL = T(1) / T(L);
    yre[0] = sumr;
    yim[0] = sumi;
    for (unsigned m = 0; m < L; ++m) {
        yre[ginvpow[m]] = x0r + cre[m] * invL;
        yim[ginvpow[m]] = x0i + cim[m] * invL;
    }
}

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"

