#pragma once

// DIF (Gentleman-Sande) passes: vectorized SoA (dif_pass), fused AoS-boundary
// passes (dif_pass_first/last), and runtime-radix dispatch functors.

#include <algorithm>  // std::max / std::min (split-column tile width)
#include <bit>       // std::bit_floor (small-ido sized-piece width)
#include <cassert>
#include <complex>
#include <cstddef>
#include <type_traits>  // std::integral_constant (transpose-tile offsets)
#include <utility>   // std::integer_sequence (small-ido allow-list), std::cmp_less

#include <poet/poet.hpp>
#include "simd.hpp"

#include "butterfly.hpp"      // dif_butterfly, dif_pass_unroll (brings fft_ct_math)
#include "cache.hpp"          // kIpTileBytes (in-place tile block limit)
#include "codelet.hpp"       // rader_apply_batched (prime chiplet tile transform)
#include "simd_swizzle.hpp" // aos_deinterleave / aos_interleave
#include "macros.hpp"             // ADM_* (included headers above undef them again)

namespace admiral {
namespace detail {

// Small-ido pass: sized-batch codelets over the a dimension. dif_pass_impl's a-loop
// needs a + W <= ido, so at 1 < ido < W it never fires. Cover [0, ido) with
// exact-width pieces instead, widest first; no masks, contiguous loads/stores.
// dif_butterfly is V-generic and fma/fnma have FUSED scalar overloads, so one body
// serves every width down to PW==1; each piece stores and releases before the next.
// Only the width is a template argument: one instantiation per (radix, width).
// sized_piece_width / sized_piece_t are in simd_swizzle.hpp.

// One radix-IP butterfly over the PW contiguous a-columns starting at a. Carries the
// output map as (obase, kstride) rather than l1, like dif_pass_body: the in-place
// ragged tail reuses this piece and differs only in the store offset. CC/CH are
// template parameters so that caller can alias (restrict would lie).
template<typename T, std::size_t IP, std::size_t PW, typename CC, typename CH>
ADM_ALWAYS_INLINE void small_ido_piece(CC ccre, CC ccim, CH chre, CH chim,
                                       std::size_t ido, std::size_t b, std::size_t a,
                                       std::size_t obase, std::size_t kstride,
                                       const T* twre, const T* twim) {
    using V = sized_piece_t<T, PW>;
    constexpr auto ld = load_piece<T, PW>;
    constexpr auto st = store_piece<T, PW>;
    V tr[IP], ti_arr[IP];
    poet::static_for<IP>([&](auto J) {
        const std::size_t off = a + ido * (J + IP * b);
        tr[J] = ld(ccre + off);
        ti_arr[J] = ld(ccim + off);
    });
    dif_butterfly<T, IP, V>(tr, ti_arr, [&](const auto k, V sr, V si) {
        const std::size_t off = obase + a + kstride * k;
        if constexpr (k > 0u) {
            const V owr = ld(twre + (k - 1u) * ido + a);
            const V owi = ld(twim + (k - 1u) * ido + a);
            // piece_fnma / piece_fma, never the plain expression: this body runs at
            // PW == 1 and at PW == W, and the wrappers carry one association at both.
            st(chre + off, piece_fnma(owi, si, owr * sr));
            st(chim + off, piece_fma(owr, si, owi * sr));
        } else {
            st(chre + off, sr);
            st(chim + off, si);
        }
    });
}

// small_ido_piece with the (k,a)-twiddles already loaded. See dif_pass_small_ido:
// the twiddle row is b-invariant, so the valley pass loads it once per piece.
template<typename T, std::size_t IP, std::size_t PW, typename CC, typename CH>
ADM_ALWAYS_INLINE void small_ido_piece_tw(CC ccre, CC ccim, CH chre, CH chim,
                                          std::size_t ido, std::size_t b, std::size_t a,
                                          std::size_t obase, std::size_t kstride,
                                          const sized_piece_t<T, PW>* owr,
                                          const sized_piece_t<T, PW>* owi) {
    using V = sized_piece_t<T, PW>;
    V tr[IP], ti_arr[IP];
    poet::static_for<IP>([&](auto J) {
        const std::size_t off = a + ido * (J + IP * b);
        tr[J] = load_piece<T, PW>(ccre + off);
        ti_arr[J] = load_piece<T, PW>(ccim + off);
    });
    dif_butterfly<T, IP, V>(tr, ti_arr, [&](const auto k, V sr, V si) {
        const std::size_t off = obase + a + kstride * k;
        if constexpr (k > 0u) {
            store_piece<T, PW>(chre + off, piece_fnma(owi[k], si, owr[k] * sr));
            store_piece<T, PW>(chim + off, piece_fma(owr[k], si, owi[k] * sr));
        } else {
            store_piece<T, PW>(chre + off, sr);
            store_piece<T, PW>(chim + off, si);
        }
    });
}

// Cover [0, ido) with sized_cover (simd_swizzle.hpp), whose widest-first descent
// and backward-aligned overlap were tuned on this pass; the column driver's
// sub-batch column tail reuses the same mechanism.
template<typename T, std::size_t IP>
void dif_pass_small_ido(const T* ccre, const T* ccim,
                        T* chre, T* chim,
                        std::size_t l1, std::size_t ido,
                        const T* twre, const T* twim) {
    // ido < W, so the full width never fits; start one step down.
    constexpr std::size_t W0 = sized_piece_width<T, xsimd::batch<T>::size / 2>();
    // Piece-outer / block-inner with hoisted twiddles, strip-mined over b. The (k,a)
    // twiddle set is b-invariant, and at 1 < ido < W its 2*(IP-1) narrow loads per block
    // rival the 2*IP data loads: that reload is part of why the valley costs more than
    // the plateau. Only for IP <= 3: above that the 2*(IP-1) pinned twiddle batches spill
    // against the butterfly's live set, and large l1 is exactly where valley chains run.
    if constexpr (IP <= 3 && 2 * (IP - 1u) + 2 * IP + 4u <= poet::vector_register_count()) {
        constexpr std::size_t kL1 = 48u * 1024u;
        std::size_t bt = (kL1 / 2u) / (4u * IP * ido * sizeof(T));
        if (bt == 0) bt = 1;
        if (bt > l1) bt = l1;
        for (std::size_t b0 = 0; b0 < l1; b0 += bt) {
            const std::size_t bend = b0 + bt < l1 ? b0 + bt : l1;
            sized_cover<T, W0, true>(0, ido, [&](auto PWc, std::size_t a) {
                constexpr std::size_t PW = PWc.value;
                using V = sized_piece_t<T, PW>;
                V owr[IP], owi[IP];  // slots 1..IP-1 used; keeps k-indexing direct
                poet::static_for<1, IP>([&](const auto k) {
                    owr[k] = load_piece<T, PW>(twre + (k - 1u) * ido + a);
                    owi[k] = load_piece<T, PW>(twim + (k - 1u) * ido + a);
                });
                for (std::size_t b = b0; b < bend; ++b)
                    small_ido_piece_tw<T, IP, PW>(ccre, ccim, chre, chim, ido, b, a,
                                                  ido * b, ido * l1, owr, owi);
            });
        }
    } else {
        poet::dynamic_for<1, 1>(std::size_t{0}, l1, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
            sized_cover<T, W0, true>(0, ido, [&](auto PWc, std::size_t a) {
                small_ido_piece<T, IP, PWc.value>(ccre, ccim, chre, chim, ido, b, a,
                                                  ido * b, ido * l1, twre, twim);
            });
        });
    }
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
// InPlace=false: Stockham, CC -> CH, the two disjoint (hence restrict-qualified by the callers
// below). InPlace=true: CC and CH are the SAME buffer and every access stays inside block b, so
// the pointer types must NOT be restrict-qualified, hence CC/CH are template parameters rather
// than fixed signatures, which keeps one body serving both instead of a copy.
//
// The only difference is the output offset (obase/kstride): Stockham appends the digit
// HIGH (b + l1*k, self-sorting), in-place appends it LOW (k + IP*b), so the store
// lands on lines this pass just loaded. Twiddles are independent of b, so they are
// bit-identical under both maps.
template<typename T, std::size_t IP, bool InPlace, typename CC, typename CH>
void dif_pass_body(CC ccre, CC ccim, CH chre, CH chim,
              std::size_t l1, std::size_t ido,
              const T* twre, const T* twim,
                   // SoA element stride of the input / output buffer, elected in the driver:
                   //   1: two planar spans, re at base and im at base+gap: IP output
                   //        blocks x 2 planes = 2*IP concurrent store streams.
                   //   2: W-blocked planar, re and im of the SAME W-batch adjacent, so im
                   //        is base+W and every logical offset doubles: IP streams.
                   //        Legal only where every offset is W-aligned (ido % W == 0), which
                   //        is exactly when the masked/overlap/scalar tails, the only paths
                   //        with an unaligned `a`, cannot fire, so they always see 1.
                   std::size_t esi, std::size_t eso) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    // Loop-invariant, hoisted once: every SoA offset is linear in (a, ido), so scaling
    // BOTH by the element stride maps the whole address space.
    const std::size_t idi = ido * esi, idz = ido * eso;

    // 1 < ido < W: neither the a-loop (needs a + W <= ido) nor the overlap block
    // (needs ido >= W) can fire, so exact-width pieces cover any such ido. ido < 4
    // stays scalar, and InPlace must NOT come here: dif_pass_small_ido's cover takes
    // the backward-aligned OVERLAPPING piece, which would double-transform a column
    // in place.
    if constexpr (!InPlace) {
        if (ido >= 4 && ido < W) {
            dif_pass_small_ido<T, IP>(ccre, ccim, chre, chim, l1, ido, twre, twim);
            return;
        }
    }

    // Column-major path for 16-reg ISAs: twiddles depend only on (k,a), so
    // aa-outer/b-inner loads each twiddle once per column vs once per (b,a)
    // (a-inner otherwise spills GPR twiddle bases). Full-width columns only;
    // L1-resident only (aa-outer re-walks cc/ch span stride ido*IP per tile).
    // Radices 2/3/5 join only at W >= 4 (SSE f64 excluded).
    if constexpr (!InPlace && IP >= 2 && IP <= 5 && poet::vector_register_count() <= 16
                  && (IP == 4 || W >= 4)) {
        if (ido >= W && ido % W == 0
            && l1 * IP * ido * (4u * sizeof(T)) <= kIpTileBytes) {
            for (std::size_t aa = 0; aa < ido; aa += W) {
                batch owr[IP - 1], owi[IP - 1];
                poet::static_for<0, IP - 1>([&](const auto k) {
                    owr[k] = batch::load_unaligned(twre + (k * ido + aa));
                    owi[k] = batch::load_unaligned(twim + (k * ido + aa));
                });
                for (std::size_t b = 0; b < l1; ++b) {
                    batch tr[IP], ti_arr[IP];
                    poet::static_for<0, IP>([&](const auto j) {
                        tr[j] = batch::load_unaligned(ccre + (aa * esi + idi * (j + IP * b)));
                        ti_arr[j] = batch::load_unaligned(ccim + (aa * esi + idi * (j + IP * b)));
                    });
                    dif_butterfly<T, IP>(tr, ti_arr, [&](const auto k, batch sr, batch si) {
                        if constexpr (k > 0u) {
                            (owr[k - 1] * sr - owi[k - 1] * si)
                                .store_unaligned(chre + (aa * eso + idz * (b + l1 * k)));
                            (owr[k - 1] * si + owi[k - 1] * sr)
                                .store_unaligned(chim + (aa * eso + idz * (b + l1 * k)));
                        } else {
                            sr.store_unaligned(chre + (aa * eso + idz * (b + l1 * k)));
                            si.store_unaligned(chim + (aa * eso + idz * (b + l1 * k)));
                        }
                    });
                }
            }
            return;
        }
    }

    for (std::size_t b = 0; b < l1; ++b) {
        // Output offset, both maps as `obase + a + kstride*k`. Stockham appends k as the HIGH
        // digit (b + l1*k) and so self-sorts; in-place appends it as the LOW digit, keeping
        // every access inside block b's span so the store hits lines this iteration just
        // loaded. Plain hoisted scalars, NOT a lambda: gcc IPA-CP outlines the emit lambda
        // into .isra clones, where a captured helper degenerates into a closure-deref chain.
        const std::size_t obase   = InPlace ? idz * IP * b : idz * b;
        const std::size_t kstride = InPlace ? idz : idz * l1;
        // Column that output slot k occupies. Identity everywhere except the in-place wide
        // radix below, which keeps each half-butterfly inside the H columns it read and so
        // appends the digit as (odd, Kc) instead of k. Folded at compile time (k is always
        // an integral_constant here), so the other paths pay nothing for it.
        constexpr bool kSplitCols = InPlace && dif_butterfly_wants_reload<IP>;
        const auto ocol = [](std::size_t k) {
            if constexpr (kSplitCols) return (k & 1u) * (IP / 2u) + k / 2u;
            else                      return k;
        };
        // No ido==1 special case: every dispatch site in dif_driver.hpp leaves at
        // least one radix unconsumed, so ido >= 2 here.
        {
            // U independent W-wide butterflies per iteration (see dif_pass_unroll;
            // U>1 on AVX-512). U==1 keeps the single-batch loop verbatim.
            constexpr std::size_t U = dif_pass_unroll<IP>();
            // Force-inlined: out-of-line shared copy regresses every size.
            auto do_batch = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
                auto emit_tw = [&](const auto k, batch sr, batch si) {
                    const std::size_t off = obase + aa * eso + kstride * ocol(k);
                    if constexpr (k > 0u) {
                        const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                        const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                        // Plain multiply, not piece_fnma: this body is W-wide only, so no
                        // tail shares it and -ffast-math contracts it to one FMA anyway.
                        (owr * sr - owi * si).store_unaligned(chre + off);
                        (owr * si + owi * sr).store_unaligned(chim + off);
                    } else {
                        sr.store_unaligned(chre + off);
                        si.store_unaligned(chim + off);
                    }
                };
                batch tr[IP], ti_arr[IP];
                for (std::size_t j = 0; j < IP; ++j) {
                    tr[j] = batch::load_unaligned(ccre + (aa * esi + idi * (j + IP * b)));
                    ti_arr[j] = batch::load_unaligned(ccim + (aa * esi + idi * (j + IP * b)));
                }
                dif_butterfly<T, IP>(tr, ti_arr, emit_tw);
            };
            // Half-butterfly at `aa`: even (Odd=0) or odd (Odd=1) DIF half of a wide
            // pow2 radix. IP/2 combined batches live, the spill-free radix-8 profile.
            // Used by the two-sweep restage (wants_reload radices only).
            auto do_half = [&](std::size_t aa, auto ODD) ADM_LAMBDA_ALWAYS_INLINE {
                constexpr std::size_t H = IP / 2;
                // Both index maps hoisted out of the per-k / per-n bodies: written out,
                // `idi * (n + IP*b)` has a runtime right factor, so gcc emits a register
                // imul per n instead of a constant one off a hoisted base.
                const std::size_t obs = obase + aa * eso;
                const std::size_t ib = aa * esi + idi * (IP * b);
                auto emit_h = [&](const auto k, batch sr, batch si) {
                    const std::size_t off = obs + kstride * k;
                    if constexpr (k > 0u) {
                        const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                        const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                        (owr * sr - owi * si).store_unaligned(chre + off);
                        (owr * si + owi * sr).store_unaligned(chim + off);
                    } else {
                        sr.store_unaligned(chre + off);
                        si.store_unaligned(chim + off);
                    }
                };
                batch hr[H], hi[H];
                poet::static_for<0, H>([&](const auto n) {
                    const batch ar = batch::load_unaligned(ccre + (ib + idi * n));
                    const batch ai = batch::load_unaligned(ccim + (ib + idi * n));
                    const batch br = batch::load_unaligned(ccre + (ib + idi * (n + H)));
                    const batch bi = batch::load_unaligned(ccim + (ib + idi * (n + H)));
                    if constexpr (ODD) {
                        auto [fr2, fi2] = apply_stage_twiddle<T, IP, n, batch>(
                            ar - br, ai - bi);
                        hr[n] = fr2;
                        hi[n] = fi2;
                    } else {
                        hr[n] = ar + br;
                        hi[n] = ai + bi;
                    }
                });
                pow2_dif_butterfly<T, H, batch>(hr, hi, [&](auto Kc, batch yr, batch yi) {
                    emit_h(std::integral_constant<std::size_t, 2 * Kc + ODD>{}, yr, yi);
                });
            };
            std::size_t a = 0;
            // !InPlace: the restage's second sweep RE-READS input columns the first sweep would
            // already have overwritten in place (even sweep stores j-slots 0,2,4,...). In-place
            // takes the single-sweep full-radix path instead and pays its spills.
            if constexpr (dif_butterfly_wants_reload<IP> && !InPlace) {
                // Two-sweep restage: even-half then odd-half, as two separate physical
                // loops. The compiler CSEs a single loop back to 2*IP-live spilling.
                // The second (cache-hot) read removes the full-radix spill traffic.
                // Two halves per iteration: one deep-FMA-tree half per iteration
                // leaves the machine issue-starved.
                auto sweep = [&](auto ODD) {
                    std::size_t ah = 0;
                    for (; ah + 2 * W <= ido; ah += 2 * W) {
                        do_half(ah, ODD);
                        do_half(ah + W, ODD);
                    }
                    for (; ah + W <= ido; ah += W) do_half(ah, ODD);
                    if (ido >= W && (ido - ah) * 2 >= W) { do_half(ido - W, ODD); ah = ido; }
                    return ah;
                };
                sweep(std::integral_constant<bool, false>{});
                a = sweep(std::integral_constant<bool, true>{});
            } else if constexpr (kSplitCols) {
                // In-place wide radix, spill-free. The recursion's first level spills
                // to L1 on purpose (pow2_dif_butterfly would spill it to the stack):
                //   A: cc[n] = a+b, cc[n+H] = (a-b)*W_IP^n   (4 live batches)
                //   B: two H-point butterflies over the now-disjoint halves (3H/2 live)
                // The out-of-place restage is illegal here: its odd sweep re-reads
                // columns the even sweep overwrote. Costs 2*IP stores per block against
                // the Stockham form's IP, but each lands on a line just loaded (no RFO).
                // Each half writes back only the H columns it read (ocol above), since the
                // natural interleave clobbers {H,H+2,..} before the odd half reads them.
                // Appending (odd, Kc) is exactly what in-place passes of radix 2 then H
                // would append, so the plan records this pass as (2, H) in rowperm.
                constexpr std::size_t H = IP / 2;
                auto ip_combine = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
                    poet::static_for<0, H>([&](const auto n) {
                        const std::size_t o0 = aa * esi + idi * (n + IP * b);
                        const std::size_t o1 = aa * esi + idi * (n + H + IP * b);
                        const batch ar = batch::load_unaligned(ccre + o0);
                        const batch ai = batch::load_unaligned(ccim + o0);
                        const batch br = batch::load_unaligned(ccre + o1);
                        const batch bi = batch::load_unaligned(ccim + o1);
                        (ar + br).store_unaligned(chre + o0);
                        (ai + bi).store_unaligned(chim + o0);
                        const auto [fr, fi] =
                            apply_stage_twiddle<T, IP, n, batch>(ar - br, ai - bi);
                        fr.store_unaligned(chre + o1);
                        fi.store_unaligned(chim + o1);
                    });
                };
                auto ip_half = [&](std::size_t aa, auto ODD) ADM_LAMBDA_ALWAYS_INLINE {
                    constexpr std::size_t src = decltype(ODD)::value ? H : std::size_t{0};
                    batch hr[H], hi[H];
                    poet::static_for<0, H>([&](const auto n) {
                        hr[n] = batch::load_unaligned(ccre + (aa * esi + idi * (n + src + IP * b)));
                        hi[n] = batch::load_unaligned(ccim + (aa * esi + idi * (n + src + IP * b)));
                    });
                    pow2_dif_butterfly<T, H, batch>(hr, hi, [&](auto Kc, batch yr, batch yi) {
                        constexpr std::size_t k = 2u * Kc + decltype(ODD)::value;
                        const std::size_t off = obase + aa * eso + kstride * (src + Kc);
                        if constexpr (k > 0u) {
                            const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                            const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                            (owr * yr - owi * yi).store_unaligned(chre + off);
                            (owr * yi + owi * yr).store_unaligned(chim + off);
                        } else {
                            yr.store_unaligned(chre + off);
                            yi.store_unaligned(chim + off);
                        }
                    });
                };
                // Three sweeps, tiled over the columns: columns are independent, so
                // running a tile to completion caps the phase-A store / phase-B reload
                // block at kTileCols*IP*2*sizeof(T). Per-column phase fusion drives the
                // reload distance to zero and does not allocate. Full-width columns
                // only; the sub-W remainder runs the masked full-radix op below. No
                // overlap-block cover: re-doing columns is illegal in place.
                // Rounded DOWN to a whole number of W-wide strips, floored at one.
                constexpr std::size_t kTileCols =
                    std::max(W, (kIpTileBytes / (IP * 2u * sizeof(T))) / W * W);
                const std::size_t afull = ido - ido % W;
                for (std::size_t t = 0; t < afull; t += kTileCols) {
                    const std::size_t hi = std::min(t + kTileCols, afull);
                    for (std::size_t ah = t; ah < hi; ah += W) ip_combine(ah);
                    for (std::size_t ah = t; ah < hi; ah += W)
                        ip_half(ah, std::integral_constant<bool, false>{});
                    for (std::size_t ah = t; ah < hi; ah += W)
                        ip_half(ah, std::integral_constant<bool, true>{});
                }
                a = afull;
            } else if constexpr (U > 1) {
                for (; a + U * W <= ido; a += U * W) {
                    poet::static_for<0, U>([&](auto UU) {
                        do_batch(a + UU * W);
                    });
                }
            }
            for (; a + W <= ido; a += W) do_batch(a);
            if constexpr (InPlace) {
                // Tail policy, mirroring the Stockham overlap gate below but with a MASK
                // where in place cannot re-read: a fat tail not itself a piece width takes
                // ONE W-wide masked op, anything else one exact-width piece. Masking
                // unconditionally loses: it does full-W work for tails one exact piece
                // covers. kSplitCols takes the masked op for EVERY remainder: the descent
                // hard-codes the natural map and a pass may not use two different maps.
                if (a < ido) {
                    const std::size_t rem = ido - a;
                    if (kSplitCols || (2 * rem >= W && !((kPieceWidths<T> >> rem) & 1u))) {
                        // Masked lanes are exactly [a, ido): no over-read past the block, no
                        // double-transform. Written out rather than routed through do_batch
                        // with a Masked flag: the full-width loop above is the hottest code
                        // in the library and an all-true mask would emit masked forms there too.
                        constexpr auto um = xsimd::unaligned_mode{};
                        const xsimd::batch_bool<T> m = lane_prefix_mask<T>(rem);
                        batch tr[IP], ti_arr[IP];
                        for (std::size_t j = 0; j < IP; ++j) {
                            tr[j] = batch::load(ccre + (a * esi + idi * (j + IP * b)), m, um);
                            ti_arr[j] = batch::load(ccim + (a * esi + idi * (j + IP * b)), m, um);
                        }
                        dif_butterfly<T, IP>(
                            tr, ti_arr, [&](const auto k, batch sr, batch si) {
                                const std::size_t off = obase + a * eso + kstride * ocol(k);
                                if constexpr (k > 0u) {
                                    const batch owr =
                                        batch::load(twre + ((k - 1u) * ido + a), m, um);
                                    const batch owi =
                                        batch::load(twim + ((k - 1u) * ido + a), m, um);
                                    (owr * sr - owi * si).store(chre + off, m, um);
                                    (owr * si + owi * sr).store(chim + off, m, um);
                                } else {
                                    sr.store(chre + off, m, um);
                                    si.store(chim + off, m, um);
                                }
                            });
                    } else {
                        constexpr std::size_t W0 = sized_piece_width<T, W / 2>();
                        sized_cover<T, W0, false>(a, ido, [&](auto PWc, std::size_t aa) {
                            // Plain twiddle indexing: this tail runs only when
                            // ido % W != 0, which is when the table is never blocked.
                            small_ido_piece<T, IP, PWc.value>(
                                ccre, ccim, chre, chim, ido, b, aa, obase, kstride, twre, twim);
                        });
                    }
                }
            } else {
                // Overlap final block: tail >= W/2 -> redo the last W columns (output a
                // depends only on input a, so this is bit-identical and needs no mask).
                // Gated because a thin tail redoes more columns than it saves.
                if (ido >= W && (ido - a) * 2 >= W) { do_batch(ido - W); a = ido; }
                // Scalar tail: ido < W (too narrow to overlap).
                for (; a < ido; ++a) {
                    T tr_s[IP], ti_s[IP];
                    for (std::size_t j = 0; j < IP; ++j) {
                        tr_s[j] = ccre[a * esi + idi * (j + IP * b)];
                        ti_s[j] = ccim[a * esi + idi * (j + IP * b)];
                    }
                    dif_butterfly<T, IP>(tr_s, ti_s, [&](const auto k, T sr, T si) {
                        if constexpr (k > 0u) {
                            const T owr = twre[(k - 1u) * ido + a];
                            const T owi = twim[(k - 1u) * ido + a];
                            chre[obase + a * eso + kstride * k] = owr * sr - owi * si;
                            chim[obase + a * eso + kstride * k] = owr * si + owi * sr;
                        } else {
                            chre[obase + a * eso + kstride * k] = sr;
                            chim[obase + a * eso + kstride * k] = si;
                        }
                    });
                }
            }
        }
    }
}

// Stockham entry: CC and CH disjoint; decltype preserves __restrict__ through the template.
template<typename T, std::size_t IP>
ADM_ALWAYS_INLINE void dif_pass_impl(const T* ccre, const T* ccim,
                                     T* chre, T* chim,
                                     std::size_t l1, std::size_t ido,
                                     const T* twre, const T* twim,
                                     std::size_t esi, std::size_t eso) {
    // CC/CH given via decltype, NOT deduced: deduction drops the __restrict__ qualifier and
    // gcc then emits aliasing-versioned loop bodies.
    dif_pass_body<T, IP, false, decltype(ccre), decltype(chre)>(
        ccre, ccim, chre, chim, l1, ido, twre, twim, esi, eso);
}

// In-place (Gentleman-Sande) entry: ONE buffer. Every read and write stays inside
// block b's span, so the store hits lines the same iteration just loaded: no RFO
// on untouched lines, and the live footprint is 2N instead of 4N.
template<typename T, std::size_t IP, typename... A>
ADM_FLATTEN void dif_pass_ip_flat(A... a) { dif_pass_body<T, IP, true>(a...); }

// No flatten gate here, unlike dif_pass: on the in-place wants_reload path the
// kSplitCols phases are always-inlined lambdas that stay inside the register file,
// and flatten would re-spill them. dif_pass entry: flatten defeats GCC's IPA-CP
// outlining of the odd-radix emit lambda into a per-k .constprop clone (clang inlines
// it unaided); OFF for wants_reload radices, where the two-sweep restage owns the
// register budget.
template<typename T, std::size_t IP, typename... A>
ADM_FLATTEN void dif_pass_flat(A... a) { dif_pass_impl<T, IP>(a...); }

template<typename T, std::size_t IP>
void dif_pass(const T* ccre, const T* ccim,
              T* chre, T* chim,
              std::size_t l1, std::size_t ido,
              const T* twre, const T* twim, std::size_t esi, std::size_t eso) {
    if constexpr (dif_butterfly_wants_reload<IP>)
        dif_pass_impl<T, IP>(ccre, ccim, chre, chim, l1, ido, twre, twim, esi, eso);
    else
        dif_pass_flat<T, IP>(ccre, ccim, chre, chim, l1, ido, twre, twim, esi, eso);
}

// Runtime-prime middle pass (compile-time prime P in dif_generic_radices): a tile
// skeleton whose leaf is rader_apply_batched<P>: one size-P DFT per SIMD lane as a
// length-(P-1) cyclic convolution through kernel_batched<P-1>. The inner state stays
// in one a-tile's (register + L1 stack) footprint instead of paying Bluestein's
// traffic. Direction-free like every middle pass: the inverse rides swapped planes.
template<typename T, std::size_t P>
ADM_NOINLINE void dif_pass_prime_chip(const T* ccre, const T* ccim,
                                      T* chre, T* chim,
                                      std::size_t l1, std::size_t ido,
                                      const T* twre, const T* twim,
                                      std::size_t esi, std::size_t eso) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t idi = ido * esi, idz = ido * eso;
    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t aa = 0; aa < ido; aa += W) {
            const std::size_t rem = ido - aa;
            // Full-vs-masked split for the generic middle pass: the corner tile alone
            // masks, so the main case carries no mask object at all.
            auto tile = [&]<bool Full>(xsimd::batch_bool<T> m) ADM_LAMBDA_ALWAYS_INLINE {
                const auto um = xsimd::unaligned_mode{};
                batch xre[P], xim[P];
                for (unsigned j = 0; j < P; ++j) {
                    const T* xjr = ccre + (aa * esi + idi * (j + P * b));
                    const T* xji = ccim + (aa * esi + idi * (j + P * b));
                    if constexpr (Full) {
                        xre[j] = batch::load_unaligned(xjr);
                        xim[j] = batch::load_unaligned(xji);
                    } else {
                        xre[j] = batch::load(xjr, m, um);
                        xim[j] = batch::load(xji, m, um);
                    }
                }
                batch yre[P], yim[P];
                rader_apply_batched<P, T, batch>(xre, xim, 1, yre, yim);
                for (unsigned k = 0; k < P; ++k) {
                    const std::size_t off = aa * eso + idz * (b + l1 * k);
                    batch sr = yre[k], si = yim[k];
                    if (k > 0) {
                        const T* twr = twre + (std::size_t(k - 1u) * ido + aa);
                        const T* twi = twim + (std::size_t(k - 1u) * ido + aa);
                        if constexpr (Full) {
                            const batch owr = batch::load_unaligned(twr);
                            const batch owi = batch::load_unaligned(twi);
                            (owr * sr - owi * si).store_unaligned(chre + off);
                            (owr * si + owi * sr).store_unaligned(chim + off);
                        } else {
                            const batch owr = batch::load(twr, m, um);
                            const batch owi = batch::load(twi, m, um);
                            (owr * sr - owi * si).store(chre + off, m, um);
                            (owr * si + owi * sr).store(chim + off, m, um);
                        }
                    } else {
                        if constexpr (Full) {
                            sr.store_unaligned(chre + off);
                            si.store_unaligned(chim + off);
                        } else {
                            sr.store(chre + off, m, um);
                            si.store(chim + off, m, um);
                        }
                    }
                }
            };
            if (rem >= W) tile.template operator()<true>(xsimd::batch_bool<T>(true));
            else          tile.template operator()<false>(lane_prefix_mask<T>(rem));
        }
    }
}

//
// At large N the pass chain is L2-latency-bound. Fusing two adjacent passes halves
// the sweeps: pass p's output staged in an L1 tile, consumed by pass p+1 cache-hot.
//
// Tile closure: ido2=ido/P2, l12=l1*P1. Pass p+1's input CC'[a'+ido2*(j'+P2*b')]
// aliases pass p's output at a=a'+ido2*j', b'=b+l1*k. For ONE b and ONE a'-tile
// [a0,a0+Wa), pass p's butterflies produce exactly the inputs of P1 pass-p+1
// groups, a closed working set of P1*P2*Wa complex values.
//
// Operation order (butterfly, twiddle, butterfly, twiddle) preserved → bit-identical.
//
// Caller contract: both middle passes (dtw.sched marks p as f2head), ido%(P2*W)==0,
// ptw = packed twiddle stream (dif_twiddle_set::packed_pair). Advances l1 by P1*P2,
// flips ping ONCE. One advancing twiddle pointer: four tables would re-add derived
// GPRs to a loop already keeping eight input/tile pointers live.
template<typename T, std::size_t P1, std::size_t P2>
void dif_pass_fused2(const T* ccre, const T* ccim,
                     T* chre, T* chim,
                     std::size_t l1, std::size_t ido,
                     const T* ptw) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W  = batch::size;
    // L1 tile: 2 planes * P1*P2*WaMax*sizeof(T) <= kFusedTileBytes, WaMax rounded down
    // to a multiple of W. The raw quotient happens to be W-aligned only at W=8.
    constexpr std::size_t WaMax = (kFusedTileBytes / (2 * P1 * P2 * sizeof(T)) / W) * W;
    static_assert(WaMax >= W, "fused2: L1 tile too small for this SIMD width");
    const std::size_t ido2 = ido / P2;
    const std::size_t l12  = l1 * P1;
    constexpr std::size_t stride1 = 2u * (P1 - 1u);  // T-elements per vector-chunk in tw1
    // tw2 section starts at ido*stride1 (= tw1 section size).
    const std::size_t tw2_off = ido * stride1;
    constexpr std::size_t stride2 = 2u * (P2 - 1u);  // T-elements per vector-chunk in tw2

    alignas(xsimd::batch<T>::arch_type::alignment()) T lbre[P1 * P2 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T lbim[P1 * P2 * WaMax];

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a0 = 0; a0 < ido2; a0 += WaMax) {
            const std::size_t Wa = ido2 - a0 < WaMax ? ido2 - a0 : WaMax;

            // ---- pass p: butterfly P2 strided subranges into L1-resident tile ----
            for (std::size_t j2 = 0; j2 < P2; ++j2) {
                // ptw1_cur: interleaved twiddle block at (a0+j2*ido2)*stride1.
                // a0+j2*ido2 is W-aligned (fuse gate: ido%(P2*W)==0).
                const T* ptw1_cur = ptw + (a0 + j2 * ido2) * stride1;
                for (std::size_t t = 0; t < Wa; t += W) {
                    const std::size_t a = a0 + ido2 * j2 + t;
                    batch tr[P1], ti_arr[P1];
                    for (std::size_t j = 0; j < P1; ++j) {
                        tr[j]     = batch::load_unaligned(ccre + (a + ido * (j + P1 * b)));
                        ti_arr[j] = batch::load_unaligned(ccim + (a + ido * (j + P1 * b)));
                    }
                    dif_butterfly<T, P1>(tr, ti_arr, [&](const auto k, batch sr, batch si) {
                        T* lr = lbre + ((k * P2 + j2) * WaMax + t);
                        T* li = lbim + ((k * P2 + j2) * WaMax + t);
                        if constexpr (k > 0u) {
                            // Compile-time offsets from ptw1_cur: k: re=(k-1)*2W, im=(k-1)*2W+W.
                            const batch owr = batch::load_unaligned(ptw1_cur + (k - 1u) * 2u * W);
                            const batch owi = batch::load_unaligned(ptw1_cur + (k - 1u) * 2u * W + W);
                            // Plain multiply, W-wide only (see dif_pass).
                            (owr * sr - owi * si).store_aligned(lr);
                            (owr * si + owi * sr).store_aligned(li);
                        } else {
                            sr.store_aligned(lr);
                            si.store_aligned(li);
                        }
                    });
                    ptw1_cur += stride1 * W;  // advance one vector-chunk
                }
            }

            // ---- pass p+1: consume tile, write CH ----
            // tw2 twiddles don't depend on k; ptw2_start shared across k iterations.
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
                    dif_butterfly<T, P2>(tr, ti_arr, [&](const auto k2, batch sr, batch si) {
                        if constexpr (k2 > 0u) {
                            // Compile-time offsets: k2: re=(k2-1)*2W, im=(k2-1)*2W+W.
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
// Extends dif_pass_fused2 one level: each (b,a0) iteration processes three
// consecutive middle passes from L1, one global read + write.
//
// Tile closure: ido_0=ido, ido_1=ido_0/P2, ido_2=ido_0/(P2*P3); ido_2*P2*P3=ido_0,
// so the P2*P3 strided sub-ranges cover the ido_0 positions feeding the (b,a0) tile
// of pass p+1, a closed working set of P1*P2*P3*Wa complex values. Two L1 tiles:
//   tile1 layout: (k1*P2*P3 + j2*P3 + j3)*WaMax + t
//   tile2 layout: (k1*P2*P3 + k2*P3 + j3)*WaMax + t
//   WaMax = 4096/(P1*P2*P3*sizeof(T))
//
// Operation order (butterfly₁,twiddle₁,butterfly₂,twiddle₂,butterfly₃,twiddle₃)
// preserved → bit-identical to unfused triple.
//
// Caller contract: three consecutive middle passes; l1/ido are pass p's params;
// ido%(P2*P3*W)==0; tw2/tw3 are p+1/p+2 tables. Advances l1 by P1*P2*P3, flips ping ONCE.
template<typename T, std::size_t P1, std::size_t P2, std::size_t P3>
void dif_pass_fused3(const T* ccre, const T* ccim,
                     T* chre, T* chim,
                     std::size_t l1, std::size_t ido,
                     const T* tw1re, const T* tw1im,
                     const T* tw2re, const T* tw2im,
                     const T* tw3re, const T* tw3im) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W  = batch::size;
    // 4 planes × P1*P2*P3*WaMax*sizeof(T) = kFusedTileBytes.
    constexpr std::size_t WaMax = (kFusedTileBytes / (4 * P1 * P2 * P3 * sizeof(T)) / W) * W;
    static_assert(WaMax >= W,
                  "fused3: WaMax must be >= W (tile too small for target)");

    const std::size_t ido2 = ido / P2;        // pass p+1's ido
    const std::size_t ido3 = ido / (P2 * P3); // pass p+2's ido = tile granularity
    const std::size_t l12  = l1 * P1;          // pass p+1's l1
    const std::size_t l123 = l1 * P1 * P2;     // pass p+2's l1

    // tile1[k1,j2,j3,t], tile2[k1,k2,j3,t]: layout (outer*P2*P3+mid*P3+j3)*WaMax+t.
    // WaMax%W==0 → aligned accesses.
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile1re[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile1im[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile2re[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile2im[P1 * P2 * P3 * WaMax];

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a0 = 0; a0 < ido3; a0 += WaMax) {
            const std::size_t Wa = ido3 - a0 < WaMax ? ido3 - a0 : WaMax;

            // Stage 1: pass p → tile1. a = a0+ido3*(j3+P3*j2)+t.
            for (std::size_t j2 = 0; j2 < P2; ++j2) {
                for (std::size_t j3 = 0; j3 < P3; ++j3) {
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a = a0 + ido3 * (j3 + P3 * j2) + t;
                        batch tr[P1], ti_arr[P1];
                        for (std::size_t j = 0; j < P1; ++j) {
                            tr[j]     = batch::load_unaligned(ccre + (a + ido * (j + P1 * b)));
                            ti_arr[j] = batch::load_unaligned(ccim + (a + ido * (j + P1 * b)));
                        }
                        dif_butterfly<T, P1>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            T* lr = tile1re + ((Kc * P2 * P3 + j2 * P3 + j3) * WaMax + t);
                            T* li = tile1im + ((Kc * P2 * P3 + j2 * P3 + j3) * WaMax + t);
                            if constexpr (Kc > 0u) {
                                const batch owr = batch::load_unaligned(tw1re + ((Kc - 1u) * ido + a));
                                const batch owi = batch::load_unaligned(tw1im + ((Kc - 1u) * ido + a));
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

            // Stage 2: tile1 → tile2 (pass p+1). a_1 = a0+ido3*j3+t.
            for (std::size_t k1 = 0; k1 < P1; ++k1) {
                for (std::size_t j3 = 0; j3 < P3; ++j3) {
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a1 = a0 + ido3 * j3 + t;  // ido_1 coordinate
                        batch tr[P2], ti_arr[P2];
                        for (std::size_t j2 = 0; j2 < P2; ++j2) {
                            tr[j2]     = batch::load_aligned(tile1re + ((k1 * P2 * P3 + j2 * P3 + j3) * WaMax + t));
                            ti_arr[j2] = batch::load_aligned(tile1im + ((k1 * P2 * P3 + j2 * P3 + j3) * WaMax + t));
                        }
                        dif_butterfly<T, P2>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            T* lr = tile2re + ((k1 * P2 * P3 + Kc * P3 + j3) * WaMax + t);
                            T* li = tile2im + ((k1 * P2 * P3 + Kc * P3 + j3) * WaMax + t);
                            if constexpr (Kc > 0u) {
                                const batch owr = batch::load_unaligned(tw2re + ((Kc - 1u) * ido2 + a1));
                                const batch owi = batch::load_unaligned(tw2im + ((Kc - 1u) * ido2 + a1));
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

            // Stage 3: tile2 → CH (pass p+2). a3=a0+t, bp2=b+l1*k1+l12*k2.
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
                        dif_butterfly<T, P3>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            if constexpr (Kc > 0u) {
                                const batch owr = batch::load_unaligned(tw3re + ((Kc - 1u) * ido3 + a3));
                                const batch owi = batch::load_unaligned(tw3im + ((Kc - 1u) * ido3 + a3));
                                (owr * sr - owi * si).store_unaligned(chre + (a3 + ido3 * (bp2 + l123 * Kc)));
                                (owr * si + owi * sr).store_unaligned(chim + (a3 + ido3 * (bp2 + l123 * Kc)));
                            } else {
                                sr.store_unaligned(chre + (a3 + ido3 * (bp2 + l123 * Kc)));
                                si.store_unaligned(chim + (a3 + ido3 * (bp2 + l123 * Kc)));
                            }
                        });
                    }
                }
            }
        }
    }
}

// Fused AoS-boundary DIF passes: dif_pass_first reads AoS and writes planar SoA,
// dif_pass_last does the reverse, so neither the de-interleave nor the re-interleave
// loop exists separately. Both share dif_pass's butterfly body.

// First pass: AoS → planar SoA.
// Input: data[a+ido*(j+IP*b)]; output: chre/chim[a+ido*(b+l1*k)].
// l1==1 for the first factor (written generically). For ido>1: vectorizes over
// contiguous a; AoS inputs gathered per-element into SIMD. For ido==1: scalar.
// Split: the pass-0 row is FACTORED (twiddles.hpp, dif_twiddle_set::p0_block):
// twre/twim hold [C: (IP-1)*blk][A: (IP-1)*nb] and w[k][a1*blk+a0] = A[k][a1]*C[k][a0],
// trading one complex multiply per k for an L1-fitting table. blk is a power of two
// multiple of W, so a0 is a mask and a1 a shift, and no batch straddles a block --
// the overlap tail is the one exception.
template<typename T, bool Forward, std::size_t IP, bool Split = false>
void dif_pass_first_impl(const std::complex<T>* data,
                    T* chre, T* chim,
                    std::size_t l1, std::size_t ido,
                    const T* twre, const T* twim,
                    std::size_t eso, std::size_t blk = 0) {
    using batch_t = xsimd::batch<T>;
    constexpr std::size_t W = batch_t::size;
    const std::size_t idz = ido * eso;
    std::size_t bsh = 0, nb = 0;
    const T* are = nullptr;
    const T* aim = nullptr;
    if constexpr (Split) {
        bsh = static_cast<std::size_t>(std::countr_zero(blk));
        nb = (ido + blk - 1) >> bsh;
        are = twre + (IP - 1) * blk;
        aim = twim + (IP - 1) * blk;
    }

    for (std::size_t b = 0; b < l1; ++b) {
        // U W-wide butterflies per iteration (see dif_pass_unroll; U>1 on AVX-512).
        constexpr std::size_t U = dif_pass_unroll<IP>();
        // AoS gather → SIMD → SoA store. Force-inlined (out-of-line regresses).
        auto do_batch = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
            // Deinterleave W contiguous AoS complex per j into planar re/im batches.
            batch_t btr[IP], bti[IP];
            for (std::size_t j = 0; j < IP; ++j) {
                const T* src = reinterpret_cast<const T*>(data + aa + ido * (j + IP * b));
                auto [dr, di] = plane_refs<Forward>(btr[j], bti[j]);
                aos_deinterleave<T>(src, dr, di);
            }
            const std::size_t a0 = Split ? (aa & (blk - 1u)) : aa;
            const std::size_t a1 = Split ? (aa >> bsh) : 0u;
            dif_butterfly<T, IP>(btr, bti, [&](const auto k, batch_t sr, batch_t si) {
                if constexpr (k > 0u) {
                    batch_t owr, owi;
                    if constexpr (Split) {
                        const batch_t cr = batch_t::load_unaligned(twre + ((k - 1u) * blk + a0));
                        const batch_t ci = batch_t::load_unaligned(twim + ((k - 1u) * blk + a0));
                        const T ar = are[(k - 1u) * nb + a1], ai = aim[(k - 1u) * nb + a1];
                        owr = ar * cr - ai * ci;
                        owi = ar * ci + ai * cr;
                    } else {
                        owr = batch_t::load_unaligned(twre + ((k - 1u) * ido + aa));
                        owi = batch_t::load_unaligned(twim + ((k - 1u) * ido + aa));
                    }
                    (owr * sr - owi * si).store_unaligned(chre + (aa * eso + idz * (b + l1 * k)));
                    (owr * si + owi * sr).store_unaligned(chim + (aa * eso + idz * (b + l1 * k)));
                } else {
                    sr.store_unaligned(chre + (aa * eso + idz * (b + l1 * k)));
                    si.store_unaligned(chim + (aa * eso + idz * (b + l1 * k)));
                }
            });
        };
        std::size_t a = 0;
        if constexpr (U > 1) {
            for (; a + U * W <= ido; a += U * W) {
                poet::static_for<0, U>([&](auto UU) {
                    do_batch(a + UU * W);
                });
            }
        }
        for (; a + W <= ido; a += W) do_batch(a);
        // Overlap tail >= W/2: recompute the last W columns (see dif_pass). Never under
        // Split: ido-W is the only batch start that is not a multiple of W, and the
        // factored load C[(k-1)*blk + a0] is valid only while a0+W stays inside one
        // blk-wide block; past that the top lanes read the next k's block and the pass
        // writes garbage. The scalar tail indexes the factored row correctly at any column.
        if (!Split && ido >= W && (ido - a) * 2 >= W) { do_batch(ido - W); a = ido; }
        // Scalar tail: ido < W or ido==1.
        for (; a < ido; ++a) {
            T tr[IP], ti[IP];
            for (std::size_t j = 0; j < IP; ++j) {
                const auto& c = data[a + ido * (j + IP * b)];
                auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
                dr = c.real();
                di = c.imag();
            }
            dif_butterfly<T, IP>(tr, ti, [&](const auto k, T sr, T si) {
                if constexpr (k > 0u) {
                    // ido==1: a==0, twiddle W^0=1 → guard.
                    T owr = T(1), owi = T(0);
                    if constexpr (Split) {
                        const T cr = twre[(k - 1u) * blk + (a & (blk - 1u))];
                        const T ci = twim[(k - 1u) * blk + (a & (blk - 1u))];
                        const T ar = are[(k - 1u) * nb + (a >> bsh)];
                        const T ai = aim[(k - 1u) * nb + (a >> bsh)];
                        owr = ar * cr - ai * ci;
                        owi = ar * ci + ai * cr;
                    } else if (ido > 1) {
                        owr = twre[(k - 1u) * ido + a];
                        owi = twim[(k - 1u) * ido + a];
                    }
                    chre[a * eso + idz * (b + l1 * k)] = owr * sr - owi * si;
                    chim[a * eso + idz * (b + l1 * k)] = owr * si + owi * sr;
                } else {
                    chre[a * eso + idz * (b + l1 * k)] = sr;
                    chim[a * eso + idz * (b + l1 * k)] = si;
                }
            });
        }
    }
}

// Flatten gate, mirroring dif_pass: GCC IPA-CP outlines the odd-radix
// emit into per-k .isra clones inside the hot b-loop, which costs a call plus a spill per
// iteration; flatten forces full inlining (clang inlines unaided). Same wants_reload
// OFF-gate as dif_pass: there the two-sweep restage owns the register file.
template<typename T, bool Forward, std::size_t IP, bool Split, typename... A>
ADM_FLATTEN void dif_pass_first_flat(A... a) { dif_pass_first_impl<T, Forward, IP, Split>(a...); }

// blk != 0 selects the factored pass-0 row. The branch is once per call (the first pass
// runs at l1 == 1, one b-iteration), so it versions the whole kernel rather than sitting
// in the butterfly; the cost is one extra instantiation per radix.
template<typename T, bool Forward, std::size_t IP>
void dif_pass_first(const std::complex<T>* data,
                    T* chre, T* chim,
                    std::size_t l1, std::size_t ido,
                    const T* twre, const T* twim,
                    std::size_t eso, std::size_t blk) {
    const auto run = [&]<bool Split>() {
        if constexpr (dif_butterfly_wants_reload<IP>)
            dif_pass_first_impl<T, Forward, IP, Split>(data, chre, chim, l1, ido, twre, twim,
                                                       eso, blk);
        else
            dif_pass_first_flat<T, Forward, IP, Split>(data, chre, chim, l1, ido, twre, twim,
                                                       eso, blk);
    };
    if (blk != 0) run.template operator()<true>();
    else run.template operator()<false>();
}

// ---------------------------------------------------------------------------
// Row-space DIF pre-levels for the wide-radix last pass (IP >= 2*W pow2).
// Lane-over-columns would run a 2*IP-live butterfly and spill at IP >= 32. Instead
// compute log2(IP/W) split levels in ROW space: for one b-row the IP arms are
// contiguous, so each level is elementwise add/sub of G=IP/W batches with
// compile-time-constant twiddle (arm = batch*W + lane). Peak live: G+4. Each result
// batch is one W-arm subgroup; one W×W transpose feeds spill-free DFT-W. Operation
// order matches pow2_dif_butterfly → bit-identical.
// ---------------------------------------------------------------------------

// exp(sign*2*pi*i*(t*W+lane)/SubN) as a lane table for the row-space odd twiddle.
template<typename T, std::size_t SubN, std::size_t Tt, std::size_t W, bool Imag>
[[nodiscard]] consteval std::array<T, W> row_split_twiddle() {
    std::array<T, W> a{};
    for (std::size_t lane = 0; lane < W; ++lane) {
        const auto sc = ct_sincos_turns(/*conjugate=*/true, Tt * W + lane, SubN);
        a[lane] = static_cast<T>(Imag ? sc.s : sc.c);
    }
    return a;
}

// In-place recursive row split: ar/ai = SubN/W arm-batches of one b-row.
// After return, batch p = W-arm subgroup at output offset bitrev(p).
template<typename T, std::size_t SubN, typename V>
ADM_ALWAYS_INLINE void row_split_levels(V* ar, V* ai) {
    constexpr std::size_t g = SubN / V::size;
    if constexpr (g >= 2) {
        constexpr std::size_t h = g / 2;
        constexpr std::size_t W = V::size;
        poet::static_for<0, h>([&](const auto t) {
            alignas(V::arch_type::alignment()) static constexpr auto twr =
                row_split_twiddle<T, SubN, t, W, false>();
            alignas(V::arch_type::alignment()) static constexpr auto twi =
                row_split_twiddle<T, SubN, t, W, true>();
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
        row_split_levels<T, SubN / 2>(ar, ai);
        row_split_levels<T, SubN / 2>(ar + h, ai + h);
    }
}

// Row counts 1..W-1 for the dif_pass_last partial tail (poet::dispatch seq; 0 impossible).
template<std::size_t... Is>
constexpr auto dif_tail_seq_shift(std::index_sequence<Is...>) -> std::index_sequence<(Is + 1)...>;
template<std::size_t W>
using dif_last_tail_seq = decltype(dif_tail_seq_shift(std::make_index_sequence<W - 1>{}));

// Subgroup p output offset: DIF emits evens at 2k, odds at 2k+1 recursively → bitrev(p).
[[nodiscard]] consteval std::size_t row_split_offset(std::size_t p, std::size_t levels) {
    std::size_t off = 0;
    for (std::size_t l = 0; l < levels; ++l) {
        off = (off << 1) | (p & 1u);
        p >>= 1;
    }
    return off;
}

// Scalar rows [b, l1) of the dif_pass_last thin tail. NOINLINE so its scalar arrays do not
// join the caller's register allocation. Inlined, clang spills the hot store loop's
// address arithmetic at sizes that never execute this tail.
template<typename T, bool Forward, std::size_t IP>
ADM_COLD ADM_NOINLINE void dif_pass_last_scalar_rows(const T* ccre,
                                            const T* ccim,
                                            std::complex<T>* data,
                                            std::size_t l1, std::size_t b, T scale_val,
                                            const std::uint32_t* rowperm) {
    for (; b < l1; ++b) {
        const std::size_t rb = rowperm ? std::size_t(rowperm[b]) : b;
        T tr[IP], ti[IP];
        for (std::size_t j = 0; j < IP; ++j) {
            tr[j] = ccre[j + IP * rb];
            ti[j] = ccim[j + IP * rb];
        }
        dif_butterfly_terminal<T, IP>(tr, ti, [&](const auto k, T sr, T si) {
            const auto [xr, xi] = plane_vals<Forward>(sr * scale_val, si * scale_val);
            data[b + l1 * k] = std::complex<T>(xr, xi);
        });
    }
}

// Width-adaptive batch: at native W, small-IP (2*IP <= W) fell into scalar
// gather; l1 < W meant the vector loop never ran (e.g. N=60 f32 W=16, l1<=15).
// Narrowest width >= IP restores tiled (IP>=W') or masked (2*IP>W') path at
// full utilization. W'==W when bit_ceil(IP)>=W → wide-IP passes and v2 untouched.
template<typename T, std::size_t IP>
struct dif_last_batch {
    static constexpr std::size_t Wn = xsimd::batch<T>::size;
    static constexpr std::size_t Wfit = std::bit_ceil(IP);
    using sized_t = xsimd::make_sized_batch_t<T, std::min(Wfit, Wn)>;
    using type = std::conditional_t<std::is_void_v<sized_t>, xsimd::batch<T>, sized_t>;
};

// One block of Rows (<=W) b-rows: lane-over-columns (W b-values per lane).
// Rows==W: mask-free. Rows<W: missing-row loads guarded to zero at compile time;
// AoS stores prefix-masked (batch_bool_constant → plain moves, not vmaskmov).
// Free function with FLATTEN: gcc-14 outlines the AoS-interleave static_for body and
// clang the poet::dispatch capture otherwise, and both then reload the tile per call.
// Non-null rowperm: the leading in-place passes leave a digit reversal on b, so read
// row rowperm[b] while still WRITING b + l1*k. Only the gather BASE moves (each row is
// an IP-element contiguous run, so no gather instruction), and the RFO-carrying store
// side stays contiguous. A runtime pointer, not a template parameter: as a parameter
// it doubles the last-pass instantiation tree. The null test is loop-invariant but
// must be versioned by hand (see the row loop below).
template<typename T, bool Forward, std::size_t IP, std::size_t Rows>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_pass_last_block(const T* ccre,
                                           const T* ccim,
                                           std::complex<T>* data,
                                           std::size_t l1, std::size_t b,
                                           [[maybe_unused]] T scale_val,
                                           const std::uint32_t* rowperm) {
    using batch_t = typename dif_last_batch<T, IP>::type;
    constexpr std::size_t W = batch_t::size;
    const auto row = [&](std::size_t i) ADM_LAMBDA_ALWAYS_INLINE -> std::size_t {
        return rowperm ? std::size_t(rowperm[i]) : i;
    };
    {
        // 32-reg ISAs only: on 16-reg ISAs the G+4+transpose tile spills.
        constexpr bool row_split_path =
            std::has_single_bit(IP) && IP >= 2 * W && dif_butterfly_wants_reload<IP> &&
            poet::vector_register_count() >= 32 && Rows == W;
        if constexpr (row_split_path) {
            // Row-space pre-levels: G+4 live batches vs 2*IP-live full butterfly.
            constexpr std::size_t G = IP / W;                 // subgroups
            constexpr std::size_t L = std::bit_width(G) - 1u;  // split levels
            alignas(batch_t::arch_type::alignment()) T stg_re[G * W * W];
            alignas(batch_t::arch_type::alignment()) T stg_im[G * W * W];
            // Versioned by hand: letting gcc unswitch the rowperm != 0 test itself costs the
            // register allocation of rr/ri. Two explicit loops allocate as well as the old
            // Perm=false template parameter did.
            const auto stage = [&](std::size_t bb, std::size_t r) ADM_LAMBDA_ALWAYS_INLINE {
                batch_t rr[G], ri[G];
                poet::static_for<0, G>([&](const auto t) {
                    rr[t] = batch_t::load_unaligned(ccre + (IP * r + t * W));
                    ri[t] = batch_t::load_unaligned(ccim + (IP * r + t * W));
                });
                row_split_levels<T, IP>(rr, ri);
                poet::static_for<0, G>([&](const auto p) {
                    rr[p].store_aligned(stg_re + (p * W + bb) * W);
                    ri[p].store_aligned(stg_im + (p * W + bb) * W);
                });
            };
            if (rowperm)
                for (std::size_t bb = 0; bb < W; ++bb) stage(bb, std::size_t(rowperm[b + bb]));
            else
                for (std::size_t bb = 0; bb < W; ++bb) stage(bb, b + bb);
            // Per subgroup: transpose rows<->arms, DFT-W over b-lanes, store at bitrev(p).
            poet::static_for<0, G>([&](const auto p) {
                constexpr std::size_t K0 = row_split_offset(p, L);
                batch_t tr2[W], ti2[W];
                for (std::size_t s = 0; s < W; ++s) {
                    tr2[s] = batch_t::load_aligned(stg_re + (p * W + s) * W);
                    ti2[s] = batch_t::load_aligned(stg_im + (p * W + s) * W);
                }
                xsimd::transpose(tr2, tr2 + W);
                xsimd::transpose(ti2, ti2 + W);
                pow2_dif_butterfly<T, W, batch_t>(
                    tr2, ti2, [&](const auto k, batch_t yr, batch_t yi) {
                        T* dst = reinterpret_cast<T*>(data + b + l1 * (k * G + K0));
                        const batch_t sv(scale_val);
                        const auto [xr, xi] = plane_vals<Forward>(yr * sv, yi * sv);
                        aos_interleave<T>(dst, xr, xi);
                    });
            });
            return;
        }
        // Load IP batches over b-lanes. A scalar stride-IP gather is shuffle-port bound and
        // costs several times the butterfly itself, so read each row contiguously and
        // transpose b<->arm (WxW) instead.
        batch_t btr[IP], bti[IP];
        if constexpr (IP >= W) {
            // Tiles [0,W),[W,2W),...; final overlapping tile at IP-W covers IP%W leftovers.
            // No over-read past arm IP-1 (scratch pads N%256==0 only).
            auto load_tile = [&](const auto off) ADM_LAMBDA_ALWAYS_INLINE {
                batch_t rr[W], ri[W];
                poet::static_for<0, W>([&](const auto bb) {
                    if constexpr (std::cmp_less(bb.value, Rows)) {
                        rr[bb] = batch_t::load_unaligned(ccre + (IP * row(b + bb) + off));
                        ri[bb] = batch_t::load_unaligned(ccim + (IP * row(b + bb) + off));
                    } else {
                        rr[bb] = batch_t(T(0));
                        ri[bb] = batch_t(T(0));
                    }
                });
                xsimd::transpose(rr, rr + W);
                xsimd::transpose(ri, ri + W);
                poet::static_for<0, W>([&](const auto a) {
                    if constexpr (off + a < IP) { btr[off + a] = rr[a]; bti[off + a] = ri[a]; }
                });
            };
            poet::static_for<0, IP / W>([&](auto Tc) {
                load_tile(std::integral_constant<std::size_t,
                          Tc * W>{});
            });
            if constexpr (IP % W != 0)
                load_tile(std::integral_constant<std::size_t, IP - W>{});
        } else {
            // W > IP: masked-load W b-rows of IP arms (lanes [IP,W) -> 0, no over-read),
            // then a WxW transpose gives IP batches over b. Covers every IP < W, including
            // 2*IP <= W: wasting most of the lanes still beats the scalar stride-IP
            // gather that case would otherwise need.
            using arch = typename batch_t::arch_type;
            constexpr auto mask = xsimd::make_batch_bool_constant<T, lane_lt<IP>, arch>();
            batch_t rr[W], ri[W];
            poet::static_for<0, W>([&](const auto bb) {
                if constexpr (std::cmp_less(bb.value, Rows)) {
                    rr[bb] = batch_t::load(ccre + IP * row(b + bb), mask, xsimd::unaligned_mode{});
                    ri[bb] = batch_t::load(ccim + IP * row(b + bb), mask, xsimd::unaligned_mode{});
                } else {
                    rr[bb] = batch_t(T(0));
                    ri[bb] = batch_t(T(0));
                }
            });
            xsimd::transpose(rr, rr + W);
            xsimd::transpose(ri, ri + W);
            for (std::size_t j = 0; j < IP; ++j) { btr[j] = rr[j]; bti[j] = ri[j]; }
        }
        // No output twiddle (ido==1 → W^0=1).
        batch_t out_re[IP], out_im[IP];
        dif_butterfly_terminal<T, IP>(btr, bti, [&](const auto k, batch_t sr, batch_t si) {
            out_re[k] = sr;
            out_im[k] = si;
        });
        // Interleave to AoS: prefix-masked (Rows<W) zip pair per k.
        // scale_val folds 1/N into the store; it is 1 for the un-normalized direction.
        const batch_t sv(scale_val);
        for (std::size_t k = 0; k < IP; ++k) {
            const auto [xr, xi] = plane_vals<Forward>(out_re[k] * sv, out_im[k] * sv);
            aos_interleave_prefix<Rows>(reinterpret_cast<T*>(data + b + l1 * k), xr, xi);
        }
    }
}

// Tail chiplet dispatcher: stateless functor (see dif_pass_last_block re lambda closure).
template<typename T, bool Forward, std::size_t IP>
inline constexpr auto dif_last_tail_invoke = []<std::size_t Rows>(
        const T* ccre, const T* ccim,
        std::complex<T>* data,
        std::size_t l1, std::size_t b, T scale_val, const std::uint32_t* rowperm) {
    dif_pass_last_block<T, Forward, IP, Rows>(ccre, ccim, data, l1, b, scale_val, rowperm);
};

// Last pass: planar SoA → AoS. Inputs ccre/ccim[j+IP*b], outputs data[b+l1*k].
// ido==1 always (l1==N/IP), twiddles W^0=1 (twre/twim unused). See dif_pass_last_block.
template<typename T, bool Forward, std::size_t IP>
void dif_pass_last(const T* ccre, const T* ccim,
                   std::complex<T>* data,
                   std::size_t l1, [[maybe_unused]] std::size_t ido,
                   [[maybe_unused]] const T* twre,
                   [[maybe_unused]] const T* twim,
                   [[maybe_unused]] T scale_val = T(1),
                   const std::uint32_t* rowperm = nullptr) {
    using batch_t = typename dif_last_batch<T, IP>::type;
    constexpr std::size_t W = batch_t::size;
    assert(ido == 1);
    // Peel leading rows so the AoS output stores land cache-line aligned. The peel is
    // the same for every k because aos_store_align_peel requires l1 % LANE == 0, so
    // data + b + l1*k shares data's own offset. Remainder policy:
    //   l1 < W        → partial chiplet via poet::dispatch (compile-time row count).
    //   rem >= W/2    → overlapped full block at l1-W (rows independent, no alias).
    //   0 < rem < W/2 → scalar rows (NOINLINE, see dif_pass_last_scalar_rows).
    std::size_t b = 0;
    if (const std::size_t peel = aos_store_align_peel<T, W>(data, l1, l1);
        peel != 0 && l1 >= peel + W) {
        poet::dispatch(dif_last_tail_invoke<T, Forward, IP>,
                       poet::dispatch_param<dif_last_tail_seq<W>>{peel},
                       ccre, ccim, data, l1, std::size_t{0}, scale_val, rowperm);
        // Versioned by hand: one loop with a runtime start costs the unpeeled path
        // instructions in the bulk block, so the peel == 0 case keeps a literal 0.
        for (b = peel; b + W <= l1; b += W)
            dif_pass_last_block<T, Forward, IP, W>(ccre, ccim, data, l1, b, scale_val, rowperm);
    } else {
        for (; b + W <= l1; b += W)
            dif_pass_last_block<T, Forward, IP, W>(ccre, ccim, data, l1, b, scale_val, rowperm);
    }
    if (b == l1) return;
    if (l1 < W) {
        poet::dispatch(dif_last_tail_invoke<T, Forward, IP>,
                       poet::dispatch_param<dif_last_tail_seq<W>>{l1},
                       ccre, ccim, data, l1, std::size_t{0}, scale_val, rowperm);
        return;
    }
    if (2 * (l1 - b) >= W) {
        dif_pass_last_block<T, Forward, IP, W>(ccre, ccim, data, l1, l1 - W,
                                                     scale_val, rowperm);
        return;
    }
    dif_pass_last_scalar_rows<T, Forward, IP>(ccre, ccim, data, l1, b, scale_val, rowperm);
}

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"

