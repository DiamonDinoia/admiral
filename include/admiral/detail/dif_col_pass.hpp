#pragma once

// ============================================================================
// DIF (Gentleman-Sande) COLUMN passes: the batched-along-a-stride analogue of
// dif_passes.hpp. Used by the N-D row-column driver for every transform axis
// that is NOT the contiguous innermost one (the innermost axis reuses the 1D
// plan_impl path verbatim).
//
// For a transform along an outer axis of a row-major tensor, the `batch_count`
// elements that sit *after* the axis index in memory are contiguous, so the
// batch (column) index becomes the SIMD lane — the SoA-friendly shape this
// codebase favors. This is isomorphic to dif_passes.hpp with two changes:
//
//   * Vectorize over the contiguous column lane `c in [0,B)` (B = batch_count)
//     instead of the 1D `a` (ido) dimension.
//   * The output twiddle depends only on the axis position `a`, NOT on the
//     column, so it is a *broadcast* scalar (one tw[a] -> batch), not a
//     per-lane vector load.
//
// Working-buffer layout (planar SoA, length axis_extent * B):
//   element (logical axis position p, column c) lives at index  p * B + c.
//   Input  positions: p = a + ido*(j + IP*b)   (j radix, b group, a in [0,ido))
//   Output positions: p = a + ido*(b + l1*k)   (k butterfly output)
//
// The AoS boundary (first/last/fused) reads/writes std::complex<T>* through an
// `axis_stride` between consecutive axis positions (== B for a contiguous
// row-major slab); the column block is always contiguous, so the same
// aos_deinterleave / aos_interleave swizzle as the 1D fused passes applies.
//
// All radix math is reused unchanged via dif_butterfly<T,Fwd,IP> (V-generic).
// ============================================================================

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>

#include <poet/poet.hpp>
#include <xsimd/xsimd.hpp>

#include "butterfly.hpp"     // dif_butterfly
#include "simd_swizzle.hpp"  // aos_deinterleave / aos_interleave
#include "macros.hpp"        // ADM_RESTRICT

namespace admiral {
namespace detail {

// Number of leading columns to store one-at-a-time so the subsequent vectorized
// AoS writes land on 64-byte (cache-line) boundaries. The last column pass
// scatters W-complex AoS blocks to `data + p*axis_stride + c`; when `data` is only
// complex-aligned (std::vector gives 16B, not 64B) every wide store straddles two
// cache lines, and the split-store penalty dominates the whole ND f64 forward.
// A vmovupd to an already-aligned address does not split, so peeling the address
// into alignment — not switching to aligned store *instructions* — is the fix.
//
// Only valid when axis_stride is a whole number of cache lines (axis_stride % LANE
// == 0): then p*axis_stride is line-aligned for every output row p, so all rows
// share one peel derived from `data`'s own offset. Otherwise return 0.
template<typename T>
ADM_ALWAYS_INLINE std::size_t aos_store_align_peel(const std::complex<T>* data,
                                                   std::size_t axis_stride, std::size_t B) {
    constexpr std::size_t LANE = 64 / sizeof(std::complex<T>);  // complex values per cache line
    constexpr std::size_t W = xsimd::batch<T>::size;            // SIMD complex lanes per store
    // The head stores the peel window [0, peel) from a SINGLE W-wide block, so it
    // needs peel <= W. peel ranges over [0, LANE-1]; that bound holds for every
    // buffer offset only when W >= LANE. On narrower ISAs (SSE2 f64 W=2, SSE f32
    // W=4 vs LANE 4/8) a peel of LANE-1 would exceed W, and the single head block
    // would silently drop columns [W, peel) — leaving them with stale pre-pass
    // values. Disable the (perf-irrelevant on those baselines) cache-line peel there.
    if constexpr (LANE <= 1 || W < LANE) {
        return 0;
    } else {
        if (axis_stride % LANE != 0) return 0;
        const std::size_t off = (reinterpret_cast<std::uintptr_t>(data) / sizeof(std::complex<T>)) % LANE;
        const std::size_t peel = (LANE - off) % LANE;
        return peel < B ? peel : B;
    }
}

// Generic vectorized column DIF pass: planar SoA in -> planar SoA out.
// Vectorizes over the contiguous column lane c in [0,B); broadcast twiddle.
template<typename T, bool Forward, int IP>
void dif_col_pass(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                  T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                  std::size_t l1, std::size_t ido, std::size_t B,
                  const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a = 0; a < ido; ++a) {
            // Vectorized over the column lane c in blocks of W, scalar tail.
            std::size_t c = 0;
            for (; c + W <= B; c += W) {
                batch tr[IPu], ti[IPu];
                for (std::size_t j = 0; j < IPu; ++j) {
                    const std::size_t p = a + ido * (j + IPu * b);
                    tr[j] = batch::load_unaligned(ccre + p * B + c);
                    ti[j] = batch::load_unaligned(ccim + p * B + c);
                }
                dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, batch sr, batch si) {
                    constexpr std::size_t k = Kc;
                    const std::size_t p = a + ido * (b + l1 * k);
                    if constexpr (k > 0u) {
                        const batch owr(twre[(k - 1u) * ido + a]);  // broadcast scalar
                        const batch owi(twim[(k - 1u) * ido + a]);
                        (owr * sr - owi * si).store_unaligned(chre + p * B + c);
                        (owr * si + owi * sr).store_unaligned(chim + p * B + c);
                    } else {
                        sr.store_unaligned(chre + p * B + c);
                        si.store_unaligned(chim + p * B + c);
                    }
                });
            }
            for (; c < B; ++c) {
                T tr[IPu], ti[IPu];
                for (std::size_t j = 0; j < IPu; ++j) {
                    const std::size_t p = a + ido * (j + IPu * b);
                    tr[j] = ccre[p * B + c];
                    ti[j] = ccim[p * B + c];
                }
                dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, T sr, T si) {
                    constexpr std::size_t k = Kc;
                    const std::size_t p = a + ido * (b + l1 * k);
                    if constexpr (k > 0u) {
                        const T owr = twre[(k - 1u) * ido + a];
                        const T owi = twim[(k - 1u) * ido + a];
                        chre[p * B + c] = owr * sr - owi * si;
                        chim[p * B + c] = owr * si + owi * sr;
                    } else {
                        chre[p * B + c] = sr;
                        chim[p * B + c] = si;
                    }
                });
            }
        }
    }
}

// First pass: reads AoS std::complex<T>* (axis_stride between axis positions),
// writes planar SoA. l1 == 1 for the first pass; ido >= 2 (single-factor axes
// take the fused path), so the output twiddle is always present.
template<typename T, bool Forward, int IP>
void dif_col_pass_first(const std::complex<T>* ADM_RESTRICT data, std::size_t axis_stride,
                        T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                        std::size_t l1, std::size_t ido, std::size_t B,
                        const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a = 0; a < ido; ++a) {
            std::size_t c = 0;
            for (; c + W <= B; c += W) {
                batch tr[IPu], ti[IPu];
                for (std::size_t j = 0; j < IPu; ++j) {
                    const std::size_t p = a + ido * (j + IPu * b);
                    const T* src = reinterpret_cast<const T*>(data + p * axis_stride + c);
                    aos_deinterleave<T>(src, tr[j], ti[j]);
                }
                dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, batch sr, batch si) {
                    constexpr std::size_t k = Kc;
                    const std::size_t p = a + ido * (b + l1 * k);
                    if constexpr (k > 0u) {
                        const batch owr(twre[(k - 1u) * ido + a]);
                        const batch owi(twim[(k - 1u) * ido + a]);
                        (owr * sr - owi * si).store_unaligned(chre + p * B + c);
                        (owr * si + owi * sr).store_unaligned(chim + p * B + c);
                    } else {
                        sr.store_unaligned(chre + p * B + c);
                        si.store_unaligned(chim + p * B + c);
                    }
                });
            }
            for (; c < B; ++c) {
                T tr[IPu], ti[IPu];
                for (std::size_t j = 0; j < IPu; ++j) {
                    const auto& z = data[(a + ido * (j + IPu * b)) * axis_stride + c];
                    tr[j] = z.real();
                    ti[j] = z.imag();
                }
                dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, T sr, T si) {
                    constexpr std::size_t k = Kc;
                    const std::size_t p = a + ido * (b + l1 * k);
                    if constexpr (k > 0u) {
                        const T owr = twre[(k - 1u) * ido + a];
                        const T owi = twim[(k - 1u) * ido + a];
                        chre[p * B + c] = owr * sr - owi * si;
                        chim[p * B + c] = owr * si + owi * sr;
                    } else {
                        chre[p * B + c] = sr;
                        chim[p * B + c] = si;
                    }
                });
            }
        }
    }
}

// Last pass: reads planar SoA, writes AoS std::complex<T>* (axis_stride between
// axis positions). ido == 1 always for the last pass (l1 == axis_extent/IP), so
// the output twiddle is W^0 = 1 (twre/twim unused).
template<typename T, bool Forward, int IP, bool Scale = false>
void dif_col_pass_last(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                       std::complex<T>* ADM_RESTRICT data, std::size_t axis_stride,
                       std::size_t l1, [[maybe_unused]] std::size_t ido, std::size_t B,
                       [[maybe_unused]] const T* ADM_RESTRICT twre,
                       [[maybe_unused]] const T* ADM_RESTRICT twim,
                       [[maybe_unused]] T scale_val = T(1)) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);

    // Invariant: the last DIF pass always has ido == 1: no output twiddle.
    assert(ido == 1);

    // Peel leading columns to align the scattered AoS output stores to cache lines
    // (see aos_store_align_peel). Invariant across b (depends only on data, stride).
    const std::size_t peel = aos_store_align_peel<T>(data, axis_stride, B);

    for (std::size_t b = 0; b < l1; ++b) {
        // Compute the radix butterfly for the W columns starting at `c`, then store
        // only complex lanes [m0, m1) of the result (m0==0 && m1==W is the full
        // aligned bulk block). The head/tail sub-window stores touch each output
        // column exactly once — no re-stores — while every column stays on the
        // identical vector arithmetic (a scalar path would contract FMAs
        // differently and break nthreads=1-vs-N bit identity).
        const auto vec_block = [&](std::size_t c, std::size_t m0, std::size_t m1) {
            batch tr[IPu], ti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                const std::size_t p = j + IPu * b;
                tr[j] = batch::load_unaligned(ccre + p * B + c);
                ti[j] = batch::load_unaligned(ccim + p * B + c);
            }
            dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, batch sr, batch si) {
                constexpr std::size_t k = Kc;
                const std::size_t p = b + l1 * k;
                T* dst = reinterpret_cast<T*>(data + p * axis_stride + c);
                if constexpr (Scale) {
                    const batch sv(scale_val);
                    aos_interleave_window<T>(dst, sr * sv, si * sv, m0, m1);
                } else {
                    aos_interleave_window<T>(dst, sr, si, m0, m1);
                }
            });
        };
        const auto scalar_col = [&](std::size_t c) {
            T tr[IPu], ti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                const std::size_t p = j + IPu * b;
                tr[j] = ccre[p * B + c];
                ti[j] = ccim[p * B + c];
            }
            dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, T sr, T si) {
                constexpr std::size_t k = Kc;
                const std::size_t p = b + l1 * k;
                if constexpr (Scale) {
                    data[p * axis_stride + c] = std::complex<T>(sr * scale_val, si * scale_val);
                } else {
                    data[p * axis_stride + c] = std::complex<T>(sr, si);
                }
            });
        };
        // Peel the misaligned head [0,peel) so the bulk stores land cache-line
        // aligned, then a windowed tail block covers the ragged remainder. Both
        // sub-window stores are in-bounds (head block at 0, tail block at B-W) and
        // disjoint from the bulk, so each column is written exactly once.
        if (B >= W) {
            std::size_t c = 0;
            if (peel > 0) {                            // head: store lanes [0,peel)
                vec_block(0, 0, peel);
                c = peel;
            }
            for (; c + W <= B; c += W) vec_block(c, 0, W);   // aligned bulk, full block
            if (c < B) vec_block(B - W, c - (B - W), W);     // tail: store lanes [c-(B-W),W)
        } else {
            for (std::size_t c = 0; c < B; ++c) scalar_col(c);   // batch narrower than a vector
        }
    }
}

// Single-pass (fused first+last): reads and writes AoS. Reached when the axis
// length factors to a single radix, so l1 == 1 and ido == 1 (twiddle trivial).
// Invariant: ido == 1; the ido>1 twiddle branches are dead and have been removed.
template<typename T, bool Forward, int IP, bool Scale = false>
void dif_col_pass_fused(std::complex<T>* ADM_RESTRICT data, std::size_t axis_stride,
                        std::size_t l1, [[maybe_unused]] std::size_t ido, std::size_t B,
                        [[maybe_unused]] const T* ADM_RESTRICT twre,
                        [[maybe_unused]] const T* ADM_RESTRICT twim,
                        [[maybe_unused]] T scale_val = T(1)) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);

    // ido == 1 invariant: twiddles are W^0 = 1 (not needed).
    for (std::size_t b = 0; b < l1; ++b) {
        // a = 0 only (ido == 1).
        std::size_t c = 0;
        for (; c + W <= B; c += W) {
            batch tr[IPu], ti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                const std::size_t p = j + IPu * b;  // a==0, ido==1 → a + ido*(j+IP*b) = j+IP*b
                const T* src = reinterpret_cast<const T*>(data + p * axis_stride + c);
                aos_deinterleave<T>(src, tr[j], ti[j]);
            }
            dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, batch sr, batch si) {
                constexpr std::size_t k = Kc;
                const std::size_t p = b + l1 * k;  // a==0, ido==1 → a + ido*(b+l1*k) = b+l1*k
                if constexpr (Scale) {
                    const batch sv(scale_val);
                    aos_interleave<T>(reinterpret_cast<T*>(data + p * axis_stride + c), sr * sv, si * sv);
                } else {
                    aos_interleave<T>(reinterpret_cast<T*>(data + p * axis_stride + c), sr, si);
                }
            });
        }
        for (; c < B; ++c) {
            T tr[IPu], ti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                const auto& z = data[(j + IPu * b) * axis_stride + c];
                tr[j] = z.real();
                ti[j] = z.imag();
            }
            dif_butterfly<T, Forward, IP>(tr, ti, [&](auto Kc, T sr, T si) {
                constexpr std::size_t k = Kc;
                if constexpr (Scale) {
                    data[(b + l1 * k) * axis_stride + c] = std::complex<T>(sr * scale_val, si * scale_val);
                } else {
                    data[(b + l1 * k) * axis_stride + c] = std::complex<T>(sr, si);
                }
            });
        }
    }
}

// ----------------------------------------------------------------------------
// Runtime-radix dispatch functors (mirror dif_passes.hpp).
// ----------------------------------------------------------------------------

template<typename T, bool Forward>
struct dif_col_pass_invoke {
    template<int IP>
    void operator()(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                    T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) const {
        dif_col_pass<T, Forward, IP>(ccre, ccim, chre, chim, l1, ido, B, twre, twim);
    }
};

template<typename T, bool Forward>
struct dif_col_pass_first_invoke {
    template<int IP>
    void operator()(const std::complex<T>* ADM_RESTRICT data, std::size_t axis_stride,
                    T* ADM_RESTRICT chre, T* ADM_RESTRICT chim,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) const {
        dif_col_pass_first<T, Forward, IP>(data, axis_stride, chre, chim, l1, ido, B, twre, twim);
    }
};

template<typename T, bool Forward, bool Scale = false>
struct dif_col_pass_last_invoke {
    T scale_val = T(1);
    template<int IP>
    void operator()(const T* ADM_RESTRICT ccre, const T* ADM_RESTRICT ccim,
                    std::complex<T>* ADM_RESTRICT data, std::size_t axis_stride,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) const {
        dif_col_pass_last<T, Forward, IP, Scale>(ccre, ccim, data, axis_stride, l1, ido, B, twre, twim, scale_val);
    }
};

template<typename T, bool Forward, bool Scale = false>
struct dif_col_pass_fused_invoke {
    T scale_val = T(1);
    template<int IP>
    void operator()(std::complex<T>* ADM_RESTRICT data, std::size_t axis_stride,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* ADM_RESTRICT twre, const T* ADM_RESTRICT twim) const {
        dif_col_pass_fused<T, Forward, IP, Scale>(data, axis_stride, l1, ido, B, twre, twim, scale_val);
    }
};

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
