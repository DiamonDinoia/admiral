#pragma once

// ============================================================================
// DIF (Gentleman-Sande) passes: the vectorized SoA pass (dif_pass), the fused
// AoS-boundary passes (dif_pass_first/last), and their runtime-radix dispatch
// functors. Hot bodies are performance-critical (see butterfly.hpp).
// ============================================================================

#include <bit>       // std::bit_floor (small-ido sized-piece width)
#include <cassert>
#include <complex>
#include <cstddef>
#include <type_traits>  // std::integral_constant (transpose-tile offsets)
#include <utility>   // std::integer_sequence (small-ido allow-list)

#include <poet/poet.hpp>
#include <xsimd/xsimd.hpp>

#include "butterfly.hpp"      // dif_butterfly, dif_pass_unroll (brings fft_ct_math)
#include "simd_swizzle.hpp" // aos_deinterleave / aos_interleave
#include "macros.hpp"             // ADM_RESTRICT (included headers above undef it again)

namespace admiral {
namespace detail {

// Eligibility + allow-list for the small-ido sized-batch path.
//
// The pass below only helps when the a-vectorized loop fills zero SIMD lanes
// (ido < W) AND the radix runs scalar there. That's W==8 radix-5 today.
// Keep the predicate and the (IP,IDO) allow-list explicit and data-driven so the
// gate is findable and a future radix slots in by adding one specialization —
// without widening the instantiation set (unused (IP,IDO) bloat the TU and
// perturb other sizes' code layout).
template<typename T, int IP>
consteval bool small_ido_eligible() {
    constexpr std::size_t W = xsimd::batch<T>::size;
    // Width-parametric: the valley (ido<W) fast path helps wherever a radix pass
    // would otherwise fill <W lanes. One family is routed: radix-5 at W==8
    // (ido∈{5,7}). A radix-8 ido∈{2,4} arm existed for the pre-wide-radix
    // planner's pow2 valleys; after the DP admitted radices 16/32, no routes
    // into it remained, so it was deleted. Kept explicit rather than
    // "any (W,IP)" to bound TU/codegen bloat (unused (IP,IDO) instantiations
    // perturb other sizes' layout).
    return W == 8 && IP == 5;
}

// The ONLY place controlling which (IP,IDO) get a dif_pass_small_ido instantiation.
// Empty for radices we don't route -> the dispatch fold is empty -> dead-stripped.
template<int IP> struct small_ido_set      { using type = std::integer_sequence<int>; };
template<>       struct small_ido_set<5>   { using type = std::integer_sequence<int, 5, 7>; };
template<int IP> using small_ido_set_t = typename small_ido_set<IP>::type;

// Small-ido pass: sized-batch ISA mixing over the a dimension.
//
// When ido < W the standard a-vectorized pass fills zero SIMD lanes (the
// radix-5/7 butterfly runs fully scalar — e.g. the ido=5/7 passes of
// N=1260/1500/5040). Instead of padding a to W with masks, or packing b into
// the lanes (the retired lane-over-b transpose kernel), vectorize over a at
// EXACT width: one make_sized_batch_t<T,4> piece (ymm inside zmm code / xmm
// inside ymm code) covers a in [0,4), then pieces of 2 and 1 (scalar) cover
// the rest (f64 7 = 4+2+1; f32 5 = 4+1). Zero masks, zero transposes, plain
// contiguous loads/stores; the butterfly instantiates per piece width
// (dif_butterfly is V-generic, and xsimd fma/fms have scalar overloads).
// Register budget: one piece's live set is ~2*IP+6 width-PW registers — the
// compile-time piece recursion IS the register cut (each piece completes,
// stores, and releases before the next starts).

// Widest xsimd-provided piece width <= N, halving until make_sized_batch_t is
// non-void (the finufft simd.hpp min_simd_width idiom, inverted): e.g.
// make_sized_batch_t<float,2> is void, so f32 decomposes 7 = 4+1+1+1 while
// f64 gets 4+2+1. Width 1 (plain scalar) always works.
template<typename T, std::size_t N>
consteval std::size_t sized_piece_width() {
    if constexpr (N <= 1) return 1;
    else if constexpr (!std::is_void_v<xsimd::make_sized_batch_t<T, N>>) return N;
    else return sized_piece_width<T, N / 2>();
}

template<typename T, bool Forward, int IP, int IDO, std::size_t A0>
ADM_ALWAYS_INLINE void small_ido_piece(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                                       T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                                       std::size_t l1, std::size_t b,
                                       const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) {
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);
    constexpr std::size_t ido = static_cast<std::size_t>(IDO);
    constexpr std::size_t rem = ido - A0;
    if constexpr (rem > 0) {
        constexpr std::size_t PW = sized_piece_width<T, std::bit_floor(rem)>();
        using V = std::conditional_t<PW == 1, T, xsimd::make_sized_batch_t<T, PW>>;
        constexpr auto ld = [](const T* p) {
            if constexpr (PW == 1) { return *p; } else { return V::load_unaligned(p); }
        };
        constexpr auto st = [](T* p, V v) {
            if constexpr (PW == 1) { *p = v; } else { v.store_unaligned(p); }
        };
        V tr[IPu], ti_arr[IPu];
        poet::static_for<IPu>([&](auto J) {
            const std::size_t off = A0 + ido * (J + IPu * b);
            tr[J] = ld(ccre + off);
            ti_arr[J] = ld(ccim + off);
        });
        dif_butterfly<T, Forward, IP, V>(tr, ti_arr, [&](auto Kc, V sr, V si) {
            constexpr std::size_t k = Kc;
            const std::size_t off = A0 + ido * (b + l1 * k);
            if constexpr (k > 0u) {
                const V owr = ld(twre + (k - 1u) * ido + A0);
                const V owi = ld(twim + (k - 1u) * ido + A0);
                // xsimd::fms/fma have scalar overloads too (-> std::fma), so
                // the batch and PW==1 pieces share one expression.
                st(chre + off, xsimd::fms(owr, sr, owi * si));
                st(chim + off, xsimd::fma(owr, si, owi * sr));
            } else {
                st(chre + off, sr);
                st(chim + off, si);
            }
        });
        small_ido_piece<T, Forward, IP, IDO, A0 + PW>(ccre, ccim, chre, chim, l1, b, twre, twim);
    }
}

template<typename T, bool Forward, int IP, int IDO>
void dif_pass_small_ido(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                        T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                        std::size_t l1,
                        const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) {
    poet::dynamic_for<1, 1>(std::size_t{0}, l1, [&](std::size_t b) {
        small_ido_piece<T, Forward, IP, IDO, 0>(ccre, ccim, chre, chim, l1, b, twre, twim);
    });
}

// Compile-time dispatch over the allow-list with runtime fall-through: if `ido`
// matches an entry, run that specialization and report done; otherwise (e.g.
// ido 2/3/4/6 that pass the ido<W guard) the fold leaves done=false so the
// caller falls through to the generic a-vectorized loop. (A fold expression, not
// poet::dispatch — which throws on an unmatched ido — nor poet::static_for, which
// would instantiate the whole 2..7 range.)
template<typename T, bool Forward, int IP, int... IDOs>
ADM_ALWAYS_INLINE bool small_ido_try(std::integer_sequence<int, IDOs...>,
    std::size_t ido, const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
    T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
    std::size_t l1, const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) {
    bool done = false;
    ( (ido == std::size_t(IDOs)
        ? (dif_pass_small_ido<T, Forward, IP, IDOs>(ccre, ccim, chre, chim, l1, twre, twim), done = true)
        : false) || ... );
    return done;
}

// Single DIF (Gentleman-Sande) pass: radix-IP butterfly over cc, writing to ch.
//
// Array layout:
//   Input:  CC[a + ido*(j + IP*b)] for j in [0,IP) radix, b in [0,l1) groups, a in [0,ido)
//   Output: CH[a + ido*(b + l1*k)] for k in [0,IP) output, b in [0,l1) groups, a in [0,ido)
//
// Algorithm:
//   1. Gather IP inputs x_j from CC (no twiddle).
//   2. Apply radix-IP DFT matrix: out_k = sum_j W_IP^{sign*j*k} * x_j.
//   3. Apply output twiddles to out_k for k in [1,IP): out_k *= W_N^{k*l1*a}.
//      (k=0: trivial, W^0 = 1.)
//   4. Scatter out_k to CH.
//
// Twiddle table: twre/twim[(k-1)*ido + a] = W_N^{k*l1*a}, for k in [1,IP), a in [0,ido).
// Vectorizes over the contiguous `a` (ido) dimension.
template<typename T, bool Forward, int IP>
void dif_pass(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
              T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
              std::size_t l1, std::size_t ido,
              const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);

    // Small-ido sized-batch path (W==8, radix 5): when ido < W the a-loop
    // fills zero lanes and the radix-5 butterfly runs fully scalar. Route to
    // the exact-width ISA-mixing pass instead (compile-time IDO piece split).
    // Restricted to the (IP=5, ido∈{5,7}) cases the factorizer actually emits;
    // instantiating the unused odd radices/idos bloats the TU and perturbs
    // other sizes' code layout. Everything else falls through to the
    // a-vectorized loop.
    if constexpr (small_ido_eligible<T, IP>()) {
        if (ido > 1 && ido < W && l1 >= W
            && small_ido_try<T, Forward, IP>(small_ido_set_t<IP>{}, ido,
                                             ccre, ccim, chre, chim, l1, twre, twim))
            return;
    }

    // Column-major radix-4 path for 16-reg ISAs: output twiddles depend only on
    // (k, a), so with the column block outer and b inner they load once per
    // column instead of once per (b, a) butterfly — the a-inner loop otherwise
    // spills GPR bases for the twiddle addresses. Full-width columns only;
    // ragged/valley shapes fall through to the generic loop below.
    // L1-resident passes only: each column block re-walks the whole cc/ch
    // span (stride ido*IP), so past L1 the extra traffic swamps the saved
    // instructions. Radices 2/3/5 join r4 only at W >= 4 (SSE f64 excluded).
    if constexpr (IP >= 2 && IP <= 5 && poet::vector_register_count() <= 16
                  && (IP == 4 || W >= 4)) {
        if (ido >= W && ido % W == 0
            && l1 * IPu * ido * (4u * sizeof(T)) <= 32768u) {
            for (std::size_t aa = 0; aa < ido; aa += W) {
                batch owr[IPu - 1], owi[IPu - 1];
                poet::static_for<0, IP - 1>([&](auto Kc) {
                    constexpr std::size_t k = Kc;
                    owr[k] = batch::load_unaligned(twre + (k * ido + aa));
                    owi[k] = batch::load_unaligned(twim + (k * ido + aa));
                });
                for (std::size_t b = 0; b < l1; ++b) {
                    batch tr[IPu], ti_arr[IPu];
                    poet::static_for<0, IP>([&](auto Jc) {
                        constexpr std::size_t j = Jc;
                        tr[j] = batch::load_unaligned(ccre + (aa + ido * (j + IPu * b)));
                        ti_arr[j] = batch::load_unaligned(ccim + (aa + ido * (j + IPu * b)));
                    });
                    dif_butterfly<T, Forward, IP>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                        constexpr std::size_t k = Kc;
                        if constexpr (k > 0u) {
                            (owr[k - 1] * sr - owi[k - 1] * si)
                                .store_unaligned(chre + (aa + ido * (b + l1 * k)));
                            (owr[k - 1] * si + owi[k - 1] * sr)
                                .store_unaligned(chim + (aa + ido * (b + l1 * k)));
                        } else {
                            sr.store_unaligned(chre + (aa + ido * (b + l1 * k)));
                            si.store_unaligned(chim + (aa + ido * (b + l1 * k)));
                        }
                    });
                }
            }
            return;
        }
    }

    for (std::size_t b = 0; b < l1; ++b) {
        if (ido == 1) {
            // a = 0 only; output twiddles are all W^0 = 1 (trivial).
            T tr[IPu], ti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                tr[j] = ccre[j + IPu * b];
                ti[j] = ccim[j + IPu * b];
            }
            dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, T sr, T si) {
                constexpr std::size_t k = Kc;
                chre[b + l1 * k] = sr;
                chim[b + l1 * k] = si;
            });
        } else {
            // Vectorized over a in [0, ido): gather, butterfly, output-twiddle, scatter.
            // Issue U independent W-wide butterflies per iteration (U = registers /
            // peak-live; see dif_pass_unroll). U==1 keeps the proven single-batch
            // loop verbatim; U>1 only arises on wider register files (AVX-512).
            constexpr std::size_t U = dif_pass_unroll<IP>();
            // One W-wide butterfly at column `aa`. Force-inlined: the U==1 hot loop
            // must stay the proven single-batch body (an out-of-line shared copy
            // regresses every size). always_inline keeps the dedup without the call.
            auto do_batch = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
                auto emit_tw = [&](auto Kc, batch sr, batch si) {
                    constexpr std::size_t k = Kc;
                    if constexpr (k > 0u) {
                        const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                        const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                        // Plain complex multiply (NOT xsimd::fnma): on the avxvnni arch -march=native
                        // selects, fnma misses the fma3 dispatch and emits neg(vxorpd)+vfmadd; the plain
                        // form contracts to one vfnmadd231pd under -ffast-math (bit-identical rounding).
                        (owr * sr - owi * si).store_unaligned(chre + (aa + ido * (b + l1 * k)));
                        (owr * si + owi * sr).store_unaligned(chim + (aa + ido * (b + l1 * k)));
                    } else {
                        sr.store_unaligned(chre + (aa + ido * (b + l1 * k)));
                        si.store_unaligned(chim + (aa + ido * (b + l1 * k)));
                    }
                };
                batch tr[IPu], ti_arr[IPu];
                for (std::size_t j = 0; j < IPu; ++j) {
                    tr[j] = batch::load_unaligned(ccre + (aa + ido * (j + IPu * b)));
                    ti_arr[j] = batch::load_unaligned(ccim + (aa + ido * (j + IPu * b)));
                }
                dif_butterfly<T, Forward, IP>(tr, ti_arr, emit_tw);
            };
            // Half-butterfly at column `aa`: only the even (Odd=0) or odd (Odd=1)
            // DIF half of a wide pow2 radix. Loads the full input column but holds
            // just IP/2 combined batches + the H-point kernel's temps — the
            // register profile of the spill-free radix-8 pass. Used by the
            // two-sweep restage below (wants_reload radices only).
            auto do_half = [&](std::size_t aa, auto OddC) ADM_LAMBDA_ALWAYS_INLINE {
                constexpr bool Odd = decltype(OddC)::value;
                constexpr int H = IP / 2;
                constexpr std::size_t HU2 = static_cast<std::size_t>(H);
                auto emit_h = [&](auto Kc, batch sr, batch si) {
                    constexpr std::size_t k = Kc;
                    if constexpr (k > 0u) {
                        const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                        const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                        (owr * sr - owi * si).store_unaligned(chre + (aa + ido * (b + l1 * k)));
                        (owr * si + owi * sr).store_unaligned(chim + (aa + ido * (b + l1 * k)));
                    } else {
                        sr.store_unaligned(chre + (aa + ido * (b + l1 * k)));
                        si.store_unaligned(chim + (aa + ido * (b + l1 * k)));
                    }
                };
                batch hr[HU2], hi[HU2];
                poet::static_for<0, H>([&](auto Nc) {
                    constexpr std::size_t n = Nc;
                    const batch ar = batch::load_unaligned(ccre + (aa + ido * (n + IPu * b)));
                    const batch ai = batch::load_unaligned(ccim + (aa + ido * (n + IPu * b)));
                    const batch br = batch::load_unaligned(ccre + (aa + ido * (n + HU2 + IPu * b)));
                    const batch bi = batch::load_unaligned(ccim + (aa + ido * (n + HU2 + IPu * b)));
                    if constexpr (Odd) {
                        auto [fr2, fi2] = apply_stage_twiddle<T, Forward, IP, Nc.value, batch>(
                            ar - br, ai - bi);
                        hr[n] = fr2;
                        hi[n] = fi2;
                    } else {
                        hr[n] = ar + br;
                        hi[n] = ai + bi;
                    }
                });
                pow2_dif_butterfly<T, Forward, H, batch>(hr, hi, [&](auto Kc, batch yr, batch yi) {
                    emit_h(std::integral_constant<int, 2 * decltype(Kc)::value + (Odd ? 1 : 0)>{},
                           yr, yi);
                });
            };
            std::size_t a = 0;
            if constexpr (dif_butterfly_wants_reload<IP>) {
                // Two-sweep restage: even-half sweep over the vector columns,
                // then odd-half sweep. Separate physical loops — the compiler
                // cannot CSE them back into the 2*IP-live array kernel (a
                // single-loop variant canonicalizes to identical spilling codegen).
                // Pays a second read of the input columns (cache-hot) to eliminate
                // the spill traffic of the full-radix kernel. Any sub-W tail falls
                // through to the shared scalar loop below.
                auto sweep = [&](auto OddC) {
                    std::size_t ah = 0;
                    for (; ah + W <= ido; ah += W) do_half(ah, OddC);
                    if (ido >= W && (ido - ah) * 2 >= W) { do_half(ido - W, OddC); ah = ido; }
                    return ah;
                };
                sweep(std::integral_constant<bool, false>{});
                a = sweep(std::integral_constant<bool, true>{});
            } else if constexpr (U > 1) {
                for (; a + U * W <= ido; a += U * W) {
                    poet::static_for<0, static_cast<int>(U)>([&](auto UU) {
                        do_batch(a + UU * W);
                    });
                }
            }
            for (; a + W <= ido; a += W) do_batch(a);
            // Overlapping final full-width block: when the remaining tail is at
            // least W/2 wide and ido >= W, recompute the last W columns ending at
            // ido-1 instead of a scalar tail. Output column a depends only on input
            // column a, so the overlap recomputes bit-identical values (no mask, no
            // over-read past the ido-run). Gated to tail >= W/2 because the block
            // recomputes W-tail redundant columns: worth it for a fat tail (f64
            // radix-5 ido=7 / radix-4 ido=35 of N=1260/5040, tail 3 of W=4 -> only 1
            // redundant), a net loss for a thin one (f64 ido=5/25/125 of N=1500,
            // tail 1 -> 3 redundant), so a thin tail keeps the cheap scalar element.
            if (ido >= W && (ido - a) * 2 >= W) { do_batch(ido - W); a = ido; }
            // Scalar tail (only reached when ido < W — too narrow to overlap).
            for (; a < ido; ++a) {
                T tr_s[IPu], ti_s[IPu];
                for (std::size_t j = 0; j < IPu; ++j) {
                    tr_s[j] = ccre[a + ido * (j + IPu * b)];
                    ti_s[j] = ccim[a + ido * (j + IPu * b)];
                }
                dif_butterfly<T, Forward, IP>(tr_s, ti_s, [&](auto Kc, T sr, T si) {
                    constexpr std::size_t k = Kc;
                    if constexpr (k > 0u) {
                        const T owr = twre[(k - 1u) * ido + a];
                        const T owi = twim[(k - 1u) * ido + a];
                        chre[a + ido * (b + l1 * k)] = owr * sr - owi * si;
                        chim[a + ido * (b + l1 * k)] = owr * si + owi * sr;
                    } else {
                        chre[a + ido * (b + l1 * k)] = sr;
                        chim[a + ido * (b + l1 * k)] = si;
                    }
                });
            }
        }
    }
}

// ============================================================================
// Fused middle-pass pair: two adjacent SoA passes through an L1-resident tile.
//
// Motivation: at N >= 8192 the pass chain is L2-latency-bound — every pass is
// a full-array sweep and the radix-4-heavy DP chains do 6-8 of them. Fusing
// two adjacent middle passes halves those sweeps: pass p's outputs are staged
// in a small tile and consumed by pass p+1 while still cache-hot, so the
// arrays are read+written once per PAIR.
//
// Index algebra (why the tile is closed): with ido2 = ido/IP2 and
// l12 = l1*IP1, pass p+1's input CC'[a' + ido2*(j' + IP2*b')] aliases pass
// p's output CH[a + ido*(b + l1*k)] under a = a' + ido2*j', b' = b + l1*k.
// So for ONE pass-p group b and ONE a'-tile [a0, a0+Wa), the pass-p
// butterflies over the IP2 strided column ranges {a0 + ido2*j' + [0,Wa)}
// produce EXACTLY the inputs of the IP1 pass-p+1 groups b' = b + l1*k over
// a' in the tile — a closed working set of IP1*IP2*Wa complex values.
//
// The per-element operation order (butterfly, output twiddle, butterfly,
// output twiddle) is identical to the sequential pair, so results are
// bit-identical to the unfused chain.
//
// Caller contract (dif_driver): both passes are middle passes (dtw.sched marks
// pass p f2head), l1/ido are pass p's parameters, ido % (IP2*W) == 0 (the
// a'-tile loop is exact — pow2 chains), ptw is the pair's packed twiddle
// stream (layout documented at dif_twiddle_set::packed_pair). Writes CH in
// pass p+1's layout; the caller advances l1 by IP1*IP2 and flips ping ONCE.
//
// Both fused layers' twiddles arrive as ONE consumption-ordered stream walked
// by a single advancing pointer with compile-time load offsets. Four separate
// table pointers (~6 derived registers) would re-add GPR pressure to a loop
// that already keeps the 8 input/tile stream pointers live.
template<typename T, bool Forward, int IP1, int IP2>
void dif_pass_fused2(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                     T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                     std::size_t l1, std::size_t ido,
                     const T* ADM_RESTRICT ptw) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W  = batch::size;
    constexpr std::size_t P1 = static_cast<std::size_t>(IP1);
    constexpr std::size_t P2 = static_cast<std::size_t>(IP2);
    // L1-resident tile: 2 * P1*P2*WaMax * sizeof(T) <= 16 KB, rounded down to a
    // multiple of the SIMD width so every WaMax chunk is W-aligned at any width
    // (W=8 native, 16 on AVX-512, 4 on SSE, ...). The raw 8192/(P1*P2*sizeof)
    // was only W-aligned by luck at W=8; round explicitly to stay width-generic.
    constexpr std::size_t WaMax = (8192 / (P1 * P2 * sizeof(T)) / W) * W;
    static_assert(WaMax >= W, "fused2: L1 tile too small for this SIMD width");
    const std::size_t ido2 = ido / P2;
    const std::size_t l12  = l1 * P1;
    // Stride of ptw1 section: 2*(IP1-1) T-elements per vector-chunk position.
    constexpr std::size_t stride1 = 2u * (P1 - 1u);  // in units of individual T elements
    // Start of tw2 section in the packed buffer (T-element offset):
    //   tw1 section size = ido * 2*(IP1-1)
    const std::size_t tw2_off = ido * stride1;
    // Stride of ptw2 section: 2*(IP2-1) T-elements per vector-chunk position.
    constexpr std::size_t stride2 = 2u * (P2 - 1u);

    alignas(xsimd::batch<T>::arch_type::alignment()) T lbre[P1 * P2 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T lbim[P1 * P2 * WaMax];

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a0 = 0; a0 < ido2; a0 += WaMax) {
            const std::size_t Wa = ido2 - a0 < WaMax ? ido2 - a0 : WaMax;

            // ---- pass p: butterfly P2 strided subranges into L1-resident tile ----
            // ptw1_cur: one advancing pointer for all k's twiddles.
            // Initialized to the chunk corresponding to a0 + j2*ido2 (for j2=0 here;
            // the j2-loop below updates it at the start of each j2 iteration).
            for (std::size_t j2 = 0; j2 < P2; ++j2) {
                // ptw1_cur points to the interleaved twiddle block for chunk (a0+j2*ido2).
                // (a0 + j2*ido2) is a multiple of W (guaranteed by fuse gate).
                // stride1 T-elements per position → (a0+j2*ido2) * stride1 T-offset.
                const T* ptw1_cur = ptw + (a0 + j2 * ido2) * stride1;
                for (std::size_t t = 0; t < Wa; t += W) {
                    const std::size_t a = a0 + ido2 * j2 + t;
                    batch tr[P1], ti_arr[P1];
                    for (std::size_t j = 0; j < P1; ++j) {
                        tr[j]     = batch::load_unaligned(ccre + (a + ido * (j + P1 * b)));
                        ti_arr[j] = batch::load_unaligned(ccim + (a + ido * (j + P1 * b)));
                    }
                    dif_butterfly<T, Forward, IP1>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                        constexpr std::size_t k = Kc;
                        T* lr = lbre + ((k * P2 + j2) * WaMax + t);
                        T* li = lbim + ((k * P2 + j2) * WaMax + t);
                        if constexpr (k > 0u) {
                            // COMPILE-TIME byte offsets from ptw1_cur:
                            //   k=1: offsets 0 (re), W (im)
                            //   k=2: offsets 2W (re), 3W (im)
                            //   k=3: offsets 4W (re), 5W (im)
                            // No derived pointer; compiler uses immediate displacement.
                            const batch owr = batch::load_unaligned(ptw1_cur + (k - 1u) * 2u * W);
                            const batch owi = batch::load_unaligned(ptw1_cur + (k - 1u) * 2u * W + W);
                            // Plain complex multiply (xsimd::fnma is avxvnni-broken).
                            (owr * sr - owi * si).store_aligned(lr);
                            (owr * si + owi * sr).store_aligned(li);
                        } else {
                            sr.store_aligned(lr);
                            si.store_aligned(li);
                        }
                    });
                    // Advance ptw1_cur by one vector-chunk in the interleaved buffer.
                    ptw1_cur += stride1 * W;
                }
            }

            // ---- pass p+1: consume tile, write CH ----
            // ptw2_cur for tw2: one advancing pointer reset at each outer-k start.
            // tw2 twiddles don't depend on the outer k loop, so ptw2_start is shared.
            // tw2 section starts at ptw + tw2_off; chunk offset = a0 * stride2.
            const T* ptw2_start = ptw + tw2_off + a0 * stride2;
            for (std::size_t k = 0; k < P1; ++k) {
                const std::size_t bp = b + l1 * k;
                const T* ptw2_cur = ptw2_start;
                for (std::size_t t = 0; t < Wa; t += W) {
                    const std::size_t a2 = a0 + t;
                    batch tr[P2], ti_arr[P2];
                    for (std::size_t j2 = 0; j2 < P2; ++j2) {
                        tr[j2]     = batch::load_aligned(lbre + ((k * P2 + j2) * WaMax + t));
                        ti_arr[j2] = batch::load_aligned(lbim + ((k * P2 + j2) * WaMax + t));
                    }
                    dif_butterfly<T, Forward, IP2>(tr, ti_arr, [&](auto K2, batch sr, batch si) {
                        constexpr std::size_t k2 = K2;
                        if constexpr (k2 > 0u) {
                            // COMPILE-TIME byte offsets from ptw2_cur:
                            //   k2=1: offsets 0 (re), W (im)
                            //   k2=2: offsets 2W (re), 3W (im)  [for IP2=4]
                            const batch owr = batch::load_unaligned(ptw2_cur + (k2 - 1u) * 2u * W);
                            const batch owi = batch::load_unaligned(ptw2_cur + (k2 - 1u) * 2u * W + W);
                            (owr * sr - owi * si).store_unaligned(chre + (a2 + ido2 * (bp + l12 * k2)));
                            (owr * si + owi * sr).store_unaligned(chim + (a2 + ido2 * (bp + l12 * k2)));
                        } else {
                            sr.store_unaligned(chre + (a2 + ido2 * (bp + l12 * k2)));
                            si.store_unaligned(chim + (a2 + ido2 * (bp + l12 * k2)));
                        }
                    });
                    ptw2_cur += stride2 * W;
                }
            }
        }
    }
}

// ============================================================================
// Fused middle-pass triple: three adjacent SoA passes through L1-resident tiles.
//
// Extends dif_pass_fused2's closed-tile algebra one level deeper. Each outer
// (b, a0) iteration processes THREE consecutive middle passes entirely from
// L1, replacing three full-array sweeps with a single global read + write.
//
// Index algebra (why the tile is closed):
//   Let ido_0 = ido (pass p), ido_1 = ido_0/P2 (pass p+1), ido_2 = ido_0/(P2*P3) (pass p+2).
//   Fix tile offset a0 ∈ [0, ido_2) stepping by Wa (= ido_2 granularity).
//   Pass p reads at a_0 = a0 + ido_2*(j3 + P3*j2) + t for j2∈[0,P2), j3∈[0,P3), t∈[0,Wa).
//   Since ido_2*P3*P2 = ido_0, these P2*P3 strided sub-ranges cover exactly the
//   ido_0 positions that feed the (b, a0) tile of pass p+1 — a closed working set
//   of P1*P2*P3*Wa complex values.
//
// Two L1 tile buffers of equal shape:
//   tile1 (pass-p output):   layout (k1*P2*P3 + j2*P3 + j3)*WaMax + t
//   tile2 (pass-p+1 output): layout (k1*P2*P3 + k2*P3 + j3)*WaMax + t
//   Size per buffer: 2 planes × P1*P2*P3*WaMax*sizeof(T)
//   WaMax = 4096/(P1*P2*P3*sizeof(T))  →  each buffer = 8KB, total stack = 16KB.
//
// Sequential operation order (butterfly₁, twiddle₁, butterfly₂, twiddle₂,
// butterfly₃, twiddle₃) is preserved, so results are bit-identical to the
// unfused triple.
//
// Caller contract: three consecutive middle passes (pass p, p+1, p+2); l1/ido
// are pass p's params; ido % (P2*P3*W) == 0; tw2/tw3 are pass p+1/p+2 tables.
// Advances l1 by P1*P2*P3 and flips ping ONCE (writes to the dst buffer).
template<typename T, bool Forward, int IP1, int IP2, int IP3>
void dif_pass_fused3(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                     T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                     std::size_t l1, std::size_t ido,
                     const T* ADM_RESTRICT tw1re, const T* ADM_RESTRICT tw1im,
                     const T* ADM_RESTRICT tw2re, const T* ADM_RESTRICT tw2im,
                     const T* ADM_RESTRICT tw3re, const T* ADM_RESTRICT tw3im) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W  = batch::size;
    constexpr std::size_t P1 = static_cast<std::size_t>(IP1);
    constexpr std::size_t P2 = static_cast<std::size_t>(IP2);
    constexpr std::size_t P3 = static_cast<std::size_t>(IP3);
    // Two tiles, re+im planes each: total 4 * P1*P2*P3*WaMax*sizeof(T) = 16KB.
    // WaMax = 4096/(P1*P2*P3*sizeof(T)) so each tile pair (re+im) = 8KB.
    constexpr std::size_t WaMax = (4096 / (P1 * P2 * P3 * sizeof(T)) / W) * W;
    static_assert(WaMax >= W,
                  "fused3: WaMax must be >= W (tile too small for target)");

    const std::size_t ido2 = ido / P2;        // pass p+1's ido
    const std::size_t ido3 = ido / (P2 * P3); // pass p+2's ido = tile granularity
    const std::size_t l12  = l1 * P1;          // pass p+1's l1
    const std::size_t l123 = l1 * P1 * P2;     // pass p+2's l1

    // tile1: pass-p butterfly outputs indexed (k1, j2, j3, t).
    // tile2: pass-p+1 butterfly outputs indexed (k1, k2, j3, t).
    // Both use layout (outer*P2*P3 + mid*P3 + j3)*WaMax + t (j2→outer for tile1,
    // k2→mid for tile2; j3 in both).  Aligned accesses guaranteed: WaMax%W==0.
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile1re[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile1im[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile2re[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile2im[P1 * P2 * P3 * WaMax];

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a0 = 0; a0 < ido3; a0 += WaMax) {
            const std::size_t Wa = ido3 - a0 < WaMax ? ido3 - a0 : WaMax;

            // -- Stage 1: pass p → tile1 --
            // For j2∈[0,P2), j3∈[0,P3), t∈[0,Wa): read CC at ido_0 position
            // a = a0 + ido3*(j3 + P3*j2) + t, butterfly, twiddle, write tile1.
            for (std::size_t j2 = 0; j2 < P2; ++j2) {
                for (std::size_t j3 = 0; j3 < P3; ++j3) {
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a = a0 + ido3 * (j3 + P3 * j2) + t;
                        batch tr[P1], ti_arr[P1];
                        for (std::size_t j = 0; j < P1; ++j) {
                            tr[j]     = batch::load_unaligned(ccre + (a + ido * (j + P1 * b)));
                            ti_arr[j] = batch::load_unaligned(ccim + (a + ido * (j + P1 * b)));
                        }
                        dif_butterfly<T, Forward, IP1>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            constexpr std::size_t k = static_cast<std::size_t>(Kc);
                            T* lr = tile1re + ((k * P2 * P3 + j2 * P3 + j3) * WaMax + t);
                            T* li = tile1im + ((k * P2 * P3 + j2 * P3 + j3) * WaMax + t);
                            if constexpr (k > 0u) {
                                const batch owr = batch::load_unaligned(tw1re + ((k - 1u) * ido + a));
                                const batch owi = batch::load_unaligned(tw1im + ((k - 1u) * ido + a));
                                (owr * sr - owi * si).store_aligned(lr);
                                (owr * si + owi * sr).store_aligned(li);
                            } else {
                                sr.store_aligned(lr);
                                si.store_aligned(li);
                            }
                        });
                    }
                }
            }

            // -- Stage 2: tile1 → tile2 (pass p+1) --
            // For k1∈[0,P1), j3∈[0,P3), t∈[0,Wa): a_1 = a0 + ido3*j3 + t (ido_1 pos).
            // Read tile1 over j2; butterfly + twiddle(tw2); write tile2 over k2.
            for (std::size_t k1 = 0; k1 < P1; ++k1) {
                for (std::size_t j3 = 0; j3 < P3; ++j3) {
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a1 = a0 + ido3 * j3 + t;  // ido_1 coordinate
                        batch tr[P2], ti_arr[P2];
                        for (std::size_t j2 = 0; j2 < P2; ++j2) {
                            tr[j2]     = batch::load_aligned(tile1re + ((k1 * P2 * P3 + j2 * P3 + j3) * WaMax + t));
                            ti_arr[j2] = batch::load_aligned(tile1im + ((k1 * P2 * P3 + j2 * P3 + j3) * WaMax + t));
                        }
                        dif_butterfly<T, Forward, IP2>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            constexpr std::size_t k2 = static_cast<std::size_t>(Kc);
                            T* lr = tile2re + ((k1 * P2 * P3 + k2 * P3 + j3) * WaMax + t);
                            T* li = tile2im + ((k1 * P2 * P3 + k2 * P3 + j3) * WaMax + t);
                            if constexpr (k2 > 0u) {
                                const batch owr = batch::load_unaligned(tw2re + ((k2 - 1u) * ido2 + a1));
                                const batch owi = batch::load_unaligned(tw2im + ((k2 - 1u) * ido2 + a1));
                                (owr * sr - owi * si).store_aligned(lr);
                                (owr * si + owi * sr).store_aligned(li);
                            } else {
                                sr.store_aligned(lr);
                                si.store_aligned(li);
                            }
                        });
                    }
                }
            }

            // -- Stage 3: tile2 → CH (pass p+2) --
            // For k1∈[0,P1), k2∈[0,P2), t∈[0,Wa): a'' = a0+t (ido_2 pos),
            // bp2 = b + l1*k1 + l12*k2 (= pass-p+2's group index).
            // Read tile2 over j3; butterfly + twiddle(tw3); write CH.
            for (std::size_t k1 = 0; k1 < P1; ++k1) {
                for (std::size_t k2 = 0; k2 < P2; ++k2) {
                    const std::size_t bp2 = b + l1 * k1 + l12 * k2;
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a3 = a0 + t;   // ido_2 coordinate
                        batch tr[P3], ti_arr[P3];
                        for (std::size_t j3 = 0; j3 < P3; ++j3) {
                            tr[j3]     = batch::load_aligned(tile2re + ((k1 * P2 * P3 + k2 * P3 + j3) * WaMax + t));
                            ti_arr[j3] = batch::load_aligned(tile2im + ((k1 * P2 * P3 + k2 * P3 + j3) * WaMax + t));
                        }
                        dif_butterfly<T, Forward, IP3>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            constexpr std::size_t k3 = static_cast<std::size_t>(Kc);
                            if constexpr (k3 > 0u) {
                                const batch owr = batch::load_unaligned(tw3re + ((k3 - 1u) * ido3 + a3));
                                const batch owi = batch::load_unaligned(tw3im + ((k3 - 1u) * ido3 + a3));
                                (owr * sr - owi * si).store_unaligned(chre + (a3 + ido3 * (bp2 + l123 * k3)));
                                (owr * si + owi * sr).store_unaligned(chim + (a3 + ido3 * (bp2 + l123 * k3)));
                            } else {
                                sr.store_unaligned(chre + (a3 + ido3 * (bp2 + l123 * k3)));
                                si.store_unaligned(chim + (a3 + ido3 * (bp2 + l123 * k3)));
                            }
                        });
                    }
                }
            }
        }
    }
}

// ============================================================================
// Fused AoS-boundary DIF passes.
//
// dif_pass_first: first pass reads directly from AoS std::complex<T>* data
//   (gathering real/imag per element) and writes planar SoA output.
//   This eliminates the standalone de-interleave loop before the pass chain.
//
// dif_pass_last: last pass reads planar SoA input and writes directly to
//   AoS std::complex<T>* data (scattering per element).
//   This eliminates the standalone re-interleave loop after the pass chain.
//   (single-factor N == radix never reaches here — it routes to the codelet.)
//
// Both share the butterfly body with dif_pass; the only difference is
// whether the input gather and/or output scatter touch AoS or planar SoA.
// For the AoS gather/scatter we use element-wise scalar access (the gather/scatter
// is inherently scalar here, so no SIMD throughput is lost).
// ============================================================================

// First pass: reads from AoS std::complex<T>* data, writes to planar SoA.
// Layout: data[a + ido*(j + IP*b)] for inputs (j in [0,IP), b in [0,l1), a in [0,ido)).
//         chre/chim[a + ido*(b + l1*k)] for outputs.
// l1 == 1 always for the first pass (first factor), but written generically.
//
// Vectorization: for ido > 1, vectorizes over the contiguous `a` dimension with
// batch<T> + scalar tail. Inputs are gathered element-by-element from AoS into an
// aligned stack buffer then loaded into SIMD. For ido == 1 (single-factor N = IP),
// falls through to scalar since l1==1 means only one element total per butterfly.
template<typename T, bool Forward, int IP>
void dif_pass_first(const std::complex<T>* ADM_RESTRICT data,
                    T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                    std::size_t l1, std::size_t ido,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) {
    using batch_t = xsimd::batch<T>;
    constexpr std::size_t W = batch_t::size;
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);

    for (std::size_t b = 0; b < l1; ++b) {
        // Vectorized loop over a in blocks of W, U butterflies per iteration
        // (U = registers / peak-live; see dif_pass_unroll). U==1 keeps the proven
        // single-batch loop verbatim; U>1 only arises on wider register files.
        constexpr std::size_t U = dif_pass_unroll<IP>();
        // One W-wide butterfly at column `aa` (AoS gather -> SIMD -> SoA store).
        // Force-inlined (see dif_pass): the shared lambda must not go out-of-line.
        auto do_batch = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
            // Deinterleave W contiguous AoS complex per j into planar re/im batches.
            batch_t btr[IPu], bti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                const T* src = reinterpret_cast<const T*>(data + aa + ido * (j + IPu * b));
                aos_deinterleave<T>(src, btr[j], bti[j]);
            }
            dif_butterfly<T, Forward, IP>(btr, bti, [&](auto Kc, batch_t sr, batch_t si) {
                constexpr std::size_t k = Kc;
                if constexpr (k > 0u) {
                    const batch_t owr = batch_t::load_unaligned(twre + ((k - 1u) * ido + aa));
                    const batch_t owi = batch_t::load_unaligned(twim + ((k - 1u) * ido + aa));
                    (owr * sr - owi * si).store_unaligned(chre + (aa + ido * (b + l1 * k)));
                    (owr * si + owi * sr).store_unaligned(chim + (aa + ido * (b + l1 * k)));
                } else {
                    sr.store_unaligned(chre + (aa + ido * (b + l1 * k)));
                    si.store_unaligned(chim + (aa + ido * (b + l1 * k)));
                }
            });
        };
        std::size_t a = 0;
        if constexpr (U > 1) {
            for (; a + U * W <= ido; a += U * W) {
                poet::static_for<0, static_cast<int>(U)>([&](auto UU) {
                    do_batch(a + UU * W);
                });
            }
        }
        for (; a + W <= ido; a += W) do_batch(a);
        // Overlapping final full-width block (see dif_pass): recompute the last W
        // columns ending at ido-1 instead of a scalar tail when the tail is >= W/2.
        if (ido >= W && (ido - a) * 2 >= W) { do_batch(ido - W); a = ido; }
        // Scalar tail (only ido < W now — and the ido==1 single-element case).
        for (; a < ido; ++a) {
            T tr[IPu], ti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                const auto& c = data[a + ido * (j + IPu * b)];
                tr[j] = c.real();
                ti[j] = c.imag();
            }
            dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, T sr, T si) {
                constexpr std::size_t k = Kc;
                if constexpr (k > 0u) {
                    // ido == 1 means a == 0 and twiddle is W^0 = 1, still valid.
                    const T owr = (ido > 1) ? twre[(k - 1u) * ido + a] : T(1);
                    const T owi = (ido > 1) ? twim[(k - 1u) * ido + a] : T(0);
                    chre[a + ido * (b + l1 * k)] = owr * sr - owi * si;
                    chim[a + ido * (b + l1 * k)] = owr * si + owi * sr;
                } else {
                    chre[a + ido * (b + l1 * k)] = sr;
                    chim[a + ido * (b + l1 * k)] = si;
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Row-space DIF pre-levels for the wide-radix last pass (IP >= 2*W pow2).
//
// The lane-over-columns path below loads all IP arm-batches through W×W
// transposes and runs one 2*IP-live radix-IP butterfly: at IP=32 that is a
// guaranteed register-file blowout (258 stack stores / 252 reloads measured in
// the f32 r32 last pass — the store-bound share of the v4 mid-pow2 cells).
// Instead, compute the first log2(IP/W) DIF split levels in ROW space: for one
// b-row the IP arms are CONTIGUOUS, so each level is an elementwise add/sub of
// G=IP/W arm-batches with a compile-time-constant twiddle vector (arm index =
// batch*W + lane). Peak live is G+4 batches per row. Each resulting batch is
// one whole W-arm subgroup; a single W×W transpose per subgroup then feeds a
// spill-free batched DFT-W. Bit-identical operation order to pow2_dif_butterfly
// (even half -> outputs 2k, odd -> 2k+1, applied recursively).
// ---------------------------------------------------------------------------

// exp(sign*2*pi*i*(t*W+lane)/SubN) as a lane table for the row-space odd twiddle.
template<typename T, int SubN, int Tt, std::size_t W, bool Forward, bool Imag>
[[nodiscard]] consteval std::array<T, W> row_split_twiddle() {
    std::array<T, W> a{};
    for (std::size_t lane = 0; lane < W; ++lane) {
        const long long n = static_cast<long long>(Tt) * static_cast<long long>(W) +
                            static_cast<long long>(lane);
        const auto sc = ct_sincos_turns((Forward ? -1 : 1) * n, SubN);
        a[lane] = static_cast<T>(Imag ? sc.s : sc.c);
    }
    return a;
}

// In-place recursive row-space split: ar/ai hold SubN/W arm-batches of one
// b-row. After it returns, batch p holds the W-arm subgroup whose output
// interleave offset is bitrev(p) (see row_split_offset).
template<typename T, bool Forward, int SubN, typename V>
ADM_ALWAYS_INLINE void row_split_levels(V* ar, V* ai) {
    constexpr int g = SubN / static_cast<int>(V::size);
    if constexpr (g >= 2) {
        constexpr int h = g / 2;
        constexpr std::size_t W = V::size;
        poet::static_for<0, h>([&](auto Tc) {
            constexpr int t = Tc.value;
            alignas(V::arch_type::alignment()) static constexpr auto twr =
                row_split_twiddle<T, SubN, t, W, Forward, false>();
            alignas(V::arch_type::alignment()) static constexpr auto twi =
                row_split_twiddle<T, SubN, t, W, Forward, true>();
            const V er = ar[t] + ar[t + h];
            const V ei = ai[t] + ai[t + h];
            const V dr = ar[t] - ar[t + h];
            const V di = ai[t] - ai[t + h];
            const V wr = V::load_aligned(twr.data());
            const V wi = V::load_aligned(twi.data());
            ar[t] = er;
            ai[t] = ei;
            ar[t + h] = dr * wr - di * wi;
            ai[t + h] = dr * wi + di * wr;
        });
        row_split_levels<T, Forward, SubN / 2>(ar, ai);
        row_split_levels<T, Forward, SubN / 2>(ar + h, ai + h);
    }
}

// 1..W-1: the candidate partial-block row counts for the dif_pass_last tail
// (a poet::dispatch sequence; 0 rows never happens — l1 >= 1).
template<int... Is>
constexpr auto dif_tail_seq_shift(std::integer_sequence<int, Is...>)
    -> std::integer_sequence<int, (Is + 1)...>;
template<std::size_t W>
using dif_last_tail_seq =
    decltype(dif_tail_seq_shift(std::make_integer_sequence<int, static_cast<int>(W) - 1>{}));

// Output offset of subgroup p: DIF emits even halves at 2k, odd at 2k+1,
// recursively — the flags are p's bits MSB-first, so the offset is bitrev(p).
[[nodiscard]] consteval std::size_t row_split_offset(std::size_t p, std::size_t levels) {
    std::size_t off = 0;
    for (std::size_t l = 0; l < levels; ++l) {
        off = (off << 1) | (p & 1u);
        p >>= 1;
    }
    return off;
}

// Scalar rows [b, l1) of the dif_pass_last tail (0 < l1 % W < W/2 only).
// NOINLINE: outlined so its scalar arrays don't join the caller's register
// allocation — inlined, clang spills the hot store loop's address arithmetic
// at sizes that never even execute this tail.
template<typename T, bool Forward, int IP, bool Scale>
ADM_NOINLINE void dif_pass_last_scalar_rows(const T* ADM_RESTRICT ccre,
                                            const T* ADM_RESTRICT ccim,
                                            std::complex<T>* ADM_RESTRICT data,
                                            std::size_t l1, std::size_t b, T scale_val) {
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);
    for (; b < l1; ++b) {
        T tr[IPu], ti[IPu];
        for (std::size_t j = 0; j < IPu; ++j) {
            tr[j] = ccre[j + IPu * b];
            ti[j] = ccim[j + IPu * b];
        }
        dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, T sr, T si) {
            constexpr std::size_t k = Kc;
            if constexpr (Scale) {
                data[b + l1 * k] = std::complex<T>(sr * scale_val, si * scale_val);
            } else {
                data[b + l1 * k] = std::complex<T>(sr, si);
            }
        });
    }
}

// Width-adaptive batch for the last pass: at native W a small-IP last pass
// (2*IP <= W) fell into the scalar-gather branch, and when l1 < W the vector
// loop never ran at all (the tiny-N killer: N=60 f32 at W=16 has l1 <= 15).
// The narrowest available width >= IP restores the tiled (IP >= W') or masked
// (2*IP > W') vector path at full lane utilization; W' == W wherever
// bit_ceil(IP) >= W, so wide-IP passes and all of v2 are untouched.
template<typename T, int IP>
struct dif_last_batch {
    static constexpr std::size_t Wn = xsimd::batch<T>::size;
    static constexpr std::size_t Wfit = std::bit_ceil(static_cast<std::size_t>(IP));
    using sized_t = xsimd::make_sized_batch_t<T, (Wfit < Wn ? Wfit : Wn)>;
    using type = std::conditional_t<std::is_void_v<sized_t>, xsimd::batch<T>, sized_t>;
};

// One block of Rows (<= W) consecutive b-rows starting at `b` of the last
// pass (lane-over-columns: W consecutive b-values packed into SIMD lanes).
// Rows == W compiles mask-free (the plain full block). Rows < W is the
// partial block: rows >= Rows don't exist, so their loads are compile-time-
// guarded to zeros (dead lanes compute garbage harmlessly) and the AoS stores
// are prefix-masked — batch_bool_constant prefixes lower to plain moves,
// not vmaskmov.
// A free function template, NOT a lambda: a capturing lambda handed to
// poet::dispatch escapes, so clang materializes the closure in memory and the
// inlined main-loop copies reload ccre/ccim/data through it every iteration
// (28 hot-loop pointer reloads vs 4 in the pre-lambda code, v3 f64 last-pass
// asm, c7 rev D audit).
template<typename T, bool Forward, int IP, bool Scale, int Rows>
ADM_ALWAYS_INLINE void dif_pass_last_block(const T* ADM_RESTRICT ccre,
                                           const T* ADM_RESTRICT ccim,
                                           std::complex<T>* ADM_RESTRICT data,
                                           std::size_t l1, std::size_t b,
                                           [[maybe_unused]] T scale_val) {
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);
    using batch_t = typename dif_last_batch<T, IP>::type;
    constexpr std::size_t W = batch_t::size;
    constexpr std::size_t Rw = static_cast<std::size_t>(Rows);
    {
        // 32-reg ISAs only: on 16-reg ISAs the G+4-live row stage plus the
        // W-wide transpose tile spills, so the two-sweep reload path is used.
        constexpr bool row_split_path =
            (IPu & (IPu - 1)) == 0 && IPu >= 2 * W && dif_butterfly_wants_reload<IP> &&
            poet::vector_register_count() >= 32 && Rw == W;
        if constexpr (row_split_path) {
            // Row-space DIF pre-levels (see row_split_levels above): G+4 live
            // batches per row instead of the 2*IP-live full-radix butterfly.
            constexpr std::size_t G = IPu / W;                 // subgroups
            constexpr std::size_t L = std::bit_width(G) - 1u;  // split levels
            alignas(batch_t::arch_type::alignment()) T stg_re[G * W * W];
            alignas(batch_t::arch_type::alignment()) T stg_im[G * W * W];
            for (std::size_t bb = 0; bb < W; ++bb) {
                batch_t rr[G], ri[G];
                poet::static_for<0, static_cast<int>(G)>([&](auto Tc) {
                    constexpr std::size_t t = Tc;
                    rr[t] = batch_t::load_unaligned(ccre + (IPu * (b + bb) + t * W));
                    ri[t] = batch_t::load_unaligned(ccim + (IPu * (b + bb) + t * W));
                });
                row_split_levels<T, Forward, IP>(rr, ri);
                poet::static_for<0, static_cast<int>(G)>([&](auto Pc) {
                    constexpr std::size_t p = Pc;
                    rr[p].store_aligned(stg_re + (p * W + bb) * W);
                    ri[p].store_aligned(stg_im + (p * W + bb) * W);
                });
            }
            // Per subgroup: rows<->arms transpose, spill-free batched DFT-W over
            // the b-lanes, AoS store at output offset bitrev(p).
            poet::static_for<0, static_cast<int>(G)>([&](auto Pc) {
                constexpr std::size_t p = Pc;
                constexpr std::size_t K0 = row_split_offset(p, L);
                batch_t tr2[W], ti2[W];
                for (std::size_t s = 0; s < W; ++s) {
                    tr2[s] = batch_t::load_aligned(stg_re + (p * W + s) * W);
                    ti2[s] = batch_t::load_aligned(stg_im + (p * W + s) * W);
                }
                xsimd::transpose(tr2, tr2 + W);
                xsimd::transpose(ti2, ti2 + W);
                pow2_dif_butterfly<T, Forward, static_cast<int>(W), batch_t>(
                    tr2, ti2, [&](auto Kc, batch_t yr, batch_t yi) {
                        constexpr std::size_t k = Kc;
                        T* dst = reinterpret_cast<T*>(data + b + l1 * (k * G + K0));
                        if constexpr (Scale) {
                            const batch_t sv(scale_val);
                            aos_interleave<T>(dst, yr * sv, yi * sv);
                        } else {
                            aos_interleave<T>(dst, yr, yi);
                        }
                    });
            });
            return;
        }
        // Section-vectorized load: the W sub-transforms x[*,b0..b0+W) are W b-rows
        // of IP contiguous arms each; we want IP batches over the b-lanes. A scalar
        // stride-IP gather compiles to a p5-bound vmovhpd/vinsertf128/vperm2f128
        // pile (~70 cyc vs an ~18 cyc FMA kernel for IP=7); instead read each row
        // contiguously and transpose b<->arm with in-register WxW transposes.
        batch_t btr[IPu], bti[IPu];
        if constexpr (IPu >= W) {
            // Tiles cover arms [0,W),[W,2W),...; the final overlapping tile at IP-W
            // picks up the IP%W leftover arms. Every load reads arms in [off,off+W)
            // <= [0,IP) of a row, so it never reads past arm IP-1 (no over-read into
            // the next planar span / past N — the scratch only pads when N%256==0).
            auto load_tile = [&](auto OffC) ADM_LAMBDA_ALWAYS_INLINE {
                constexpr std::size_t off = OffC;
                batch_t rr[W], ri[W];
                poet::static_for<0, static_cast<int>(W)>([&](auto BB) {
                    constexpr std::size_t bb = BB;
                    if constexpr (bb < Rw) {
                        rr[bb] = batch_t::load_unaligned(ccre + (IPu * (b + bb) + off));
                        ri[bb] = batch_t::load_unaligned(ccim + (IPu * (b + bb) + off));
                    } else {
                        rr[bb] = batch_t(T(0));
                        ri[bb] = batch_t(T(0));
                    }
                });
                xsimd::transpose(rr, rr + W);
                xsimd::transpose(ri, ri + W);
                poet::static_for<0, static_cast<int>(W)>([&](auto Ac) {
                    constexpr std::size_t a = Ac;
                    if constexpr (off + a < IPu) { btr[off + a] = rr[a]; bti[off + a] = ri[a]; }
                });
            };
            poet::static_for<0, static_cast<int>(IPu / W)>([&](auto Tc) {
                load_tile(std::integral_constant<std::size_t,
                          Tc * W>{});
            });
            if constexpr (IPu % W != 0)
                load_tile(std::integral_constant<std::size_t, IPu - W>{});
        } else if constexpr (2u * IPu > W) {
            // W > IP and the IP-lane mask is past the half (wide enough to be
            // worth the WxW transpose below; contiguous masks load natively at
            // every ISA): masked-load W
            // b-rows of IP arms (lanes [IP,W) read 0, NO over-read past the row /
            // span — IP<W can't form an in-row overlap tile), one WxW transpose ->
            // IP batches over b. Kills the f32 IP=5/7 last-pass scalar gather.
            using arch = typename batch_t::arch_type;
            struct lane_lt_ip {
                static constexpr bool get(std::size_t i, std::size_t) { return i < IPu; }
            };
            constexpr auto mask = xsimd::make_batch_bool_constant<T, lane_lt_ip, arch>();
            batch_t rr[W], ri[W];
            poet::static_for<0, static_cast<int>(W)>([&](auto BB) {
                constexpr std::size_t bb = BB;
                if constexpr (bb < Rw) {
                    rr[bb] = batch_t::load(ccre + IPu * (b + bb), mask, xsimd::unaligned_mode{});
                    ri[bb] = batch_t::load(ccim + IPu * (b + bb), mask, xsimd::unaligned_mode{});
                } else {
                    rr[bb] = batch_t(T(0));
                    ri[bb] = batch_t(T(0));
                }
            });
            xsimd::transpose(rr, rr + W);
            xsimd::transpose(ri, ri + W);
            for (std::size_t j = 0; j < IPu; ++j) { btr[j] = rr[j]; bti[j] = ri[j]; }
        } else {
            // Very small IP (2*IP <= W, e.g. f64 IP=2, f32 IP<=4): a masked load
            // would leave more than half the lanes idle AND still need the WxW
            // transpose; the stride-IP gather into a transposed buffer skips both.
            alignas(batch_t::arch_type::alignment()) T buf_re[IPu][W];
            alignas(batch_t::arch_type::alignment()) T buf_im[IPu][W];
            for (std::size_t j = 0; j < IPu; ++j) {
                for (std::size_t lane = 0; lane < Rw; ++lane) {
                    buf_re[j][lane] = ccre[j + IPu * (b + lane)];
                    buf_im[j][lane] = ccim[j + IPu * (b + lane)];
                }
                // Rows < W partial block only: missing rows read as zeros.
                for (std::size_t lane = Rw; lane < W; ++lane) {
                    buf_re[j][lane] = T(0);
                    buf_im[j][lane] = T(0);
                }
            }
            for (std::size_t j = 0; j < IPu; ++j) {
                btr[j] = batch_t::load_aligned(buf_re[j]);
                bti[j] = batch_t::load_aligned(buf_im[j]);
            }
        }
        // Apply DFT matrix. No output twiddle (ido==1 → W^0 = 1).
        batch_t out_re[IPu], out_im[IPu];
        dif_butterfly<T, Forward, IP>(btr, bti, [&](auto Kc, batch_t sr, batch_t si) {
            constexpr std::size_t k = Kc;
            out_re[k] = sr;
            out_im[k] = si;
        });
        // Interleave to AoS output: data[b+lane + l1*k] is Rows contiguous
        // complex for each k — one (prefix-masked when Rows < W) zip pair per k
        // replaces the per-lane .get() scatter.
        // Scale=true: fold 1/N into the store (inverse transform); forward path
        // compiles to byte-identical code via if constexpr (no branch overhead).
        if constexpr (Scale) {
            const batch_t sv(scale_val);
            for (std::size_t k = 0; k < IPu; ++k)
                aos_interleave_prefix<Rw>(reinterpret_cast<T*>(data + b + l1 * k),
                                          out_re[k] * sv, out_im[k] * sv);
        } else {
            for (std::size_t k = 0; k < IPu; ++k)
                aos_interleave_prefix<Rw>(reinterpret_cast<T*>(data + b + l1 * k),
                                          out_re[k], out_im[k]);
        }
    }
}

// Tail chiplet dispatcher for dif_pass_last: a stateless functor taking all
// state as arguments (see dif_pass_last_block for why not a lambda).
template<typename T, bool Forward, int IP, bool Scale>
struct dif_last_tail_invoke {
    template<int Rows>
    void operator()(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                    std::complex<T>* ADM_RESTRICT data,
                    std::size_t l1, std::size_t b, T scale_val) const {
        dif_pass_last_block<T, Forward, IP, Scale, Rows>(ccre, ccim, data, l1, b, scale_val);
    }
};

// Last pass: reads from planar SoA, writes to AoS std::complex<T>* data.
// Layout: ccre/ccim[j + IP*b] for inputs, data[b + l1*k] for outputs. The
// last DIF pass always has ido == 1 (l1 == N/IP), so output twiddles are
// W^0 = 1 (no twiddle multiply, twre/twim unused). Block shape and lane
// packing: see dif_pass_last_block.
template<typename T, bool Forward, int IP, bool Scale = false>
void dif_pass_last(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                   std::complex<T>* ADM_RESTRICT data,
                   std::size_t l1, [[maybe_unused]] std::size_t ido,
                   [[maybe_unused]] const T* ADM_RESTRICT twre,
                   [[maybe_unused]] const T* ADM_RESTRICT twim,
                   [[maybe_unused]] T scale_val = T(1)) {
    using batch_t = typename dif_last_batch<T, IP>::type;
    constexpr std::size_t W = batch_t::size;
    assert(ido == 1);
    // Full-width blocks, then a remainder strategy picked once per call:
    //   l1 < W        -> one partial chiplet via poet::dispatch: the row count
    //                    is compile-time, so missing rows are zero-inited
    //                    registers and the AoS stores are batch_bool_constant
    //                    prefixes (plain-move lowering). Covers
    //                    the all-scalar tiny-N case.
    //   rem >= W/2    -> overlapped full-width block at l1-W: rows are
    //                    independent and the planar inputs never alias the
    //                    AoS output (ADM_RESTRICT), so recomputing rows
    //                    [l1-W, l1) re-stores identical values through the
    //                    already-hot loop body.
    //   0 < rem < W/2 -> scalar rows (NOINLINE, see dif_pass_last_scalar_rows):
    //                    for a thin remainder the scalar path is cheapest.
    std::size_t b = 0;
    for (; b + W <= l1; b += W)
        dif_pass_last_block<T, Forward, IP, Scale, static_cast<int>(W)>(ccre, ccim, data, l1, b,
                                                                        scale_val);
    if (b == l1) return;
    if (l1 < W) {
        poet::dispatch(dif_last_tail_invoke<T, Forward, IP, Scale>{},
                       poet::dispatch_param<dif_last_tail_seq<W>>{static_cast<int>(l1)},
                       ccre, ccim, data, l1, std::size_t{0}, scale_val);
        return;
    }
    if (2 * (l1 - b) >= W) {
        dif_pass_last_block<T, Forward, IP, Scale, static_cast<int>(W)>(ccre, ccim, data, l1,
                                                                        l1 - W, scale_val);
        return;
    }
    dif_pass_last_scalar_rows<T, Forward, IP, Scale>(ccre, ccim, data, l1, b, scale_val);
}

// Dispatch a single DIF pass with a runtime radix.
template<typename T, bool Forward>
struct dif_pass_invoke {
    template<int IP>
    void operator()(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                    T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                    std::size_t l1, std::size_t ido,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) const {
        dif_pass<T, Forward, IP>(ccre, ccim, chre, chim, l1, ido, twre, twim);
    }
};

// Dispatch functors for the fused AoS-boundary passes.

template<typename T, bool Forward>
struct dif_pass_first_invoke {
    template<int IP>
    void operator()(const std::complex<T>* ADM_RESTRICT data,
                    T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                    std::size_t l1, std::size_t ido,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) const {
        dif_pass_first<T, Forward, IP>(data, chre, chim, l1, ido, twre, twim);
    }
};

template<typename T, bool Forward, bool Scale = false>
struct dif_pass_last_invoke {
    T scale_val = T(1);
    template<int IP>
    void operator()(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                    std::complex<T>* ADM_RESTRICT data,
                    std::size_t l1, std::size_t ido,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) const {
        dif_pass_last<T, Forward, IP, Scale>(ccre, ccim, data, l1, ido, twre, twim, scale_val);
    }
};

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"

