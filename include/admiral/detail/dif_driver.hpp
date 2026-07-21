#pragma once

// ============================================================================
// Iterative DIF (Gentleman-Sande) pass-chain driver.
//
// Computes a 7-smooth DFT iteratively (no recursion), with the layout:
//
//   Input:  CC[a + ido*(j + ip*b)]   j in [0,ip) radix, b in [0,l1) groups
//   Output: CH[a + ido*(b + l1*k)]   k in [0,ip) butterfly out, b in [0,l1) groups
//   a in [0,ido) contiguous; N = ip * l1 * ido.
//
// Pass recurrence: l1=1; for each factor ip (in factorization order):
//   ido = N/(l1*ip); run pass; l1 *= ip.
//
// Twiddles per pass: tw[(j-1)*ido + a] = W_N^{j*l1*a} for j in [1,ip), a in [0,ido).
// j=0 is trivial (W^0=1). After all passes the output is in natural order — no
// bit-reversal needed (Gentleman-Sande property).
//
// Ping-pong between two planar SoA buffers; the `a` (ido) dimension is
// vectorized with xsimd::batch<T> plus a scalar tail. ido==1 (last pass)
// scalarizes over the b dimension.
//
// All scratch and twiddle storage is owned externally (plan) and passed in;
// no allocation occurs in the hot path.
// ============================================================================

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include <poet/poet.hpp>

#include "codelet.hpp"     // kernel_batched (lane-packed terminal)
#include "dif_passes.hpp"  // dif_pass[_first/_last/_fused] + invokers
#include "math.hpp"        // codelet_dispatch (terminal base kernel)
#include "simd_swizzle.hpp" // aos_interleave (terminal AoS stores)
#include "twiddles.hpp"    // dif_twiddle_set, build_dif_twiddle_set

#include "macros.hpp"      // ADM_RESTRICT (SoA driver signature); undef at EOF

namespace admiral {
namespace detail {

// dif_radix_set (the integer_sequence routed through poet::dispatch below) is
// defined once in twiddles.hpp; see the comment there.

// ----------------------------------------------------------------------------
// Terminal base-codelet execution (the DP BASE_TERMINAL_RADIX branch).
//
// Terminal sizes with a lane-packed batched execution. pow2-only for now.
using dif_terminal_set = std::integer_sequence<int, 8, 16, 32, 64>;

// Scalar per-group fallback (also the < W-groups tail of the packed path).
template<typename T, bool Forward, bool Scale>
ADM_ALWAYS_INLINE void dif_terminal_scalar(const T* ADM_RESTRICT src_re,
                                           const T* ADM_RESTRICT src_im,
                                           std::complex<T>* ADM_RESTRICT data,
                                           std::size_t bn, std::size_t n_groups,
                                           std::size_t g0, T scale_val) {
    for (std::size_t g = g0; g < n_groups; ++g) {
        const std::size_t off = g * bn;
        alignas(xsimd::batch<T>::arch_type::alignment()) std::complex<T> gbuf[64];  // bn <= 64 (terminal limit)
        for (std::size_t i = 0; i < bn; ++i)
            gbuf[i] = {src_re[off + i], src_im[off + i]};
        codelet_dispatch<T, Forward>(gbuf, gbuf, bn);
        for (std::size_t i = 0; i < bn; ++i) {
            if constexpr (Scale)
                data[g + n_groups * i] = {gbuf[i].real() * scale_val,
                                          gbuf[i].imag() * scale_val};
            else
                data[g + n_groups * i] = gbuf[i];
        }
    }
}

// Lane-packed terminal: W consecutive groups ride one SIMD lane each through a
// batched size-BN codelet (kernel_batched — twiddles baked, broadcast across
// lanes). Groups are contiguous SoA slices, so the input is a stride-BN lane
// gather; the strided output X[g + n_groups*p] is CONTIGUOUS across the W
// lanes for each frequency p, so stores are plain aos_interleave — no scatter.
template<typename T, bool Forward, bool Scale>
struct dif_terminal_invoke {
    T scale_val;
    template<typename IC>
    void operator()(IC, const T* ADM_RESTRICT src_re, const T* ADM_RESTRICT src_im,
                    std::complex<T>* ADM_RESTRICT data, std::size_t n_groups) const {
        constexpr auto bn = static_cast<std::size_t>(IC::value);
        using V = xsimd::batch<T>;
        using IdxT = xsimd::as_integer_t<T>;
        constexpr std::size_t W = V::size;
        alignas(xsimd::batch<IdxT>::arch_type::alignment()) std::array<IdxT, W> idx_a{};
        for (std::size_t l = 0; l < W; ++l) idx_a[l] = static_cast<IdxT>(l * bn);
        const auto idx = xsimd::batch<IdxT>::load_aligned(idx_a.data());
        V xv[bn], iv[bn], sr[bn], si[bn];  // sr/si: sub-transform scratch only
        [[maybe_unused]] const V vs(scale_val);
        std::size_t g = 0;
        for (; g + W <= n_groups; g += W) {
            const T* base_re = src_re + g * bn;
            const T* base_im = src_im + g * bn;
            poet::static_for<0, static_cast<int>(bn)>([&](auto I) {
                xv[I] = V::gather(base_re + static_cast<std::size_t>(I), idx);
                iv[I] = V::gather(base_im + static_cast<std::size_t>(I), idx);
            });
            // Scale + AoS interleave fused into the kernel's last combine: each
            // output batch goes straight to data — no output-array round-trip.
            kernel_batched<static_cast<unsigned>(bn), T, Forward, V>::apply_sink(
                xv, iv, 1, sr, si, [&](std::size_t p, V outr, V outi) {
                    if constexpr (Scale) {
                        outr = outr * vs;
                        outi = outi * vs;
                    }
                    aos_interleave<T, V>(reinterpret_cast<T*>(data + g + n_groups * p),
                                         outr, outi);
                });
        }
        dif_terminal_scalar<T, Forward, Scale>(src_re, src_im, data, bn, n_groups,
                                               g, scale_val);
    }
};

// Fused-pair radices (IP1,IP2). Routed as a 2-D poet::dispatch over the pair, so
// the (4,4)/(4,8)/(8,4)/(8,8) combos map to dif_pass_fused2<...,IP1,IP2> without a
// hand-written if-ladder.
using dif_fused_pair_set = std::integer_sequence<int, 4, 8>;

// Two adjacent middle passes through the packed twiddle stream (dif_pass_fused2).
template<typename T, bool Forward>
struct dif_fused2_invoke {
    template<typename IC1, typename IC2>
    void operator()(IC1, IC2, const T* ADM_RESTRICT src_re, const T* ADM_RESTRICT src_im,
                    T* ADM_RESTRICT dst_re, T* ADM_RESTRICT dst_im,
                    std::size_t l1, std::size_t ido, const T* ADM_RESTRICT ptw) const {
        dif_pass_fused2<T, Forward, IC1::value, IC2::value>(
            src_re, src_im, dst_re, dst_im, l1, ido, ptw);
    }
};

// Execute the iterative DIF pass-chain with fused AoS boundary passes.
//
// The first pass reads directly from AoS `in[]` into the SoA ping buffer (cc0).
// Intermediate passes ping-pong between cc0 and cc1 (SoA).
// The last pass reads from the final SoA buffer and writes directly to AoS `out[]`.
//
// in == out is the in-place form; in != out (non-overlapping) is free
// out-of-place: the first pass fully consumes `in` into SoA scratch before any
// AoS store, so no input-preserving copy is ever needed on this route.
//
// cc0 and cc1 must each be >= N elements (re and im separately).
// Allocation-free. When Scale=false (default) the output is un-normalized; the
// caller is responsible for 1/N. When Scale=true the 1/N scale (scale_val) is
// folded directly into dif_pass_last's store loop — no separate sweep needed.
// Requires n_passes >= 2 (single-factor N routes to the codelet path, not here).
template<typename T, bool Forward, bool Scale = false>
void iterative_dif_execute_ws(const std::complex<T>* in, std::complex<T>* out,
                               std::size_t N,
                               T* cc0re, T* cc0im, T* cc1re, T* cc1im,
                               const dif_twiddle_set<T>& dtw,
                               [[maybe_unused]] T scale_val = T(1)) {
    if (N <= 1) return;

    const std::size_t n_passes = dtw.radices.size();

    // --- First pass: AoS -> SoA (cc0) ---
    {
        const unsigned ip = dtw.radices[0];
        const std::size_t ido = N / static_cast<std::size_t>(ip);  // l1==1 for first pass
        const T* twre = dtw.passes[0].first.data();
        const T* twim = dtw.passes[0].second.data();
        poet::dispatch(dif_pass_first_invoke<T, Forward>{},
                       poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                       in, cc0re, cc0im, std::size_t{1}, ido, twre, twim);
    }

    // Single-pass (N == ip), no terminal: dif_pass_first already applied the DFT
    // butterfly and wrote to SoA cc0.  Re-interleave cc0 back to AoS data.
    // When base_n != 0 the single pass is just the first factored stage; the
    // terminal codelet block below handles the remaining sub-transform.
    if (n_passes == 1 && dtw.base_n == 0) {
        for (std::size_t k = 0; k < N; ++k) {
            if constexpr (Scale) {
                out[k] = std::complex<T>(cc0re[k] * scale_val, cc0im[k] * scale_val);
            } else {
                out[k] = std::complex<T>(cc0re[k], cc0im[k]);
            }
        }
        return;
    }

    // --- Intermediate passes: SoA ping-pong ---
    // After the first pass, result is in cc0 (ping=true means input in cc0).
    std::size_t l1 = static_cast<std::size_t>(dtw.radices[0]);
    bool ping = false;  // first pass wrote to cc0; next pass reads cc0 (ping=false means src=cc0)

    for (std::size_t p = 1; p + 1 < n_passes; ++p) {
        const unsigned ip = dtw.radices[p];
        const std::size_t ido = N / (l1 * static_cast<std::size_t>(ip));
        const T* twre = dtw.passes[p].first.data();
        const T* twim = dtw.passes[p].second.data();

        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;
        T* dst_re = ping ? cc0re : cc1re;
        T* dst_im = ping ? cc0im : cc1im;

        // Fusion: fuse adjacent middle passes through L1-resident tiles,
        // replacing multiple full-array sweeps with one global read + write per
        // tile. The plan-time schedule dtw.sched (single source of truth,
        // dif_fusion_schedule in twiddles.hpp — gates, policy, and twiddle
        // representation documented there) decides per pass; this loop just
        // executes it. Register pressure for any r=8-containing pair:
        // 2*8+2 = 18 live batches, same as standalone dif_pass<8> (not additive).

        // Fused triple: three adjacent radix-4 middle passes in one tile sweep
        // (plain twiddle layers). Consumes THREE passes: one ping flip, l1 *= 64.
        if (dtw.sched[p] == dif_fuse::f3head) {
            dif_pass_fused3<T, Forward, 4, 4, 4>(
                src_re, src_im, dst_re, dst_im, l1, ido, twre, twim,
                dtw.passes[p + 1].first.data(), dtw.passes[p + 1].second.data(),
                dtw.passes[p + 2].first.data(), dtw.passes[p + 2].second.data());
            l1 *= 64u;  // P1 * P2 * P3 = 4 * 4 * 4
            ping = !ping;
            p += 2;  // skip passes p+1, p+2
            continue;
        }

        // Fused pair: two adjacent middle passes through the packed twiddle
        // stream (packed_pair[p], one advancing pointer — layout in
        // twiddles.hpp). Consumes TWO passes: one ping flip, l1 *= IP1*IP2.
        if (dtw.sched[p] == dif_fuse::f2head) {
            const unsigned r2 = dtw.radices[p + 1];
            const T* ptw = dtw.packed_pair[p].data();
            poet::dispatch(dif_fused2_invoke<T, Forward>{},
                           poet::dispatch_param<dif_fused_pair_set>{static_cast<int>(ip)},
                           poet::dispatch_param<dif_fused_pair_set>{static_cast<int>(r2)},
                           src_re, src_im, dst_re, dst_im, l1, ido, ptw);
            l1 *= static_cast<std::size_t>(ip) * r2;
            ping = !ping;
            ++p;  // skip pass p+1
            continue;
        }

        poet::dispatch(dif_pass_invoke<T, Forward>{},
                       poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                       src_re, src_im, dst_re, dst_im, l1, ido, twre, twim);

        l1 *= static_cast<std::size_t>(ip);
        ping = !ping;
    }

    // --- Terminal base-codelet branch (base_n != 0) ---
    //
    // SoA group layout at this point (verified from the DIF pass recurrence):
    //   After all factored passes, l1 == N/ido == N/base_n.  The pass output
    //   is CH[a + ido*(b + l1_prev*k)].  With the accumulation b2 = b + l1_prev*k
    //   ranging over [0, N/base_n), this simplifies to CH[a + base_n*b2].
    //   So group g = b2 occupies the contiguous slice [g*base_n, (g+1)*base_n)
    //   in src_re/src_im — no striding.
    //
    // When n_passes >= 2 the intermediate loop above covered [1, n_passes-1);
    // we run pass n_passes-1 here as a plain SoA->SoA pass (it is always
    // dif_fuse::plain — the fusion schedule only marks middle passes).
    // When n_passes == 1 the first pass already produced the grouped layout.
    if (dtw.base_n != 0) {
        if (n_passes >= 2) {
            const std::size_t p = n_passes - 1;
            const unsigned ip  = dtw.radices[p];
            const std::size_t ido = N / (l1 * static_cast<std::size_t>(ip));
            const T* twre  = dtw.passes[p].first.data();
            const T* twim  = dtw.passes[p].second.data();
            const T* src_re = ping ? cc1re : cc0re;
            const T* src_im = ping ? cc1im : cc0im;
            T*       dst_re = ping ? cc0re : cc1re;
            T*       dst_im = ping ? cc0im : cc1im;
            poet::dispatch(dif_pass_invoke<T, Forward>{},
                           poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                           src_re, src_im, dst_re, dst_im, l1, ido, twre, twim);
            l1 *= static_cast<std::size_t>(ip);
            ping = !ping;
        }
        // All passes done; src holds N/base_n groups of base_n elements.
        // SoA layout: group g at [g*bn, (g+1)*bn) — read contiguously.
        // Output mapping: DFT_{bn}(group g)[p] = X[g + n_groups*p].
        // In the DIF decomposition after k passes (l1 = N/bn = n_groups),
        // each group g contributes to strided output frequencies
        // {g, g+n_groups, g+2*n_groups, ...}, NOT to [g*bn, (g+1)*bn).
        // Writing contiguously here would produce the wrong DFT output.
        const std::size_t bn       = static_cast<std::size_t>(dtw.base_n);
        const std::size_t n_groups = N / bn;
        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;
        if (bn == 8 || bn == 16 || bn == 32 || bn == 64) {
            // dif_terminal_set members: lane-packed batched codelet.
            poet::dispatch(dif_terminal_invoke<T, Forward, Scale>{scale_val},
                           poet::dispatch_param<dif_terminal_set>{static_cast<int>(bn)},
                           src_re, src_im, out, n_groups);
        } else {
            dif_terminal_scalar<T, Forward, Scale>(src_re, src_im, out, bn,
                                                   n_groups, 0, scale_val);
        }
        return;
    }

    // --- Last pass: SoA -> AoS ---
    {
        const std::size_t p = n_passes - 1;
        const unsigned ip = dtw.radices[p];
        const std::size_t ido = N / (l1 * static_cast<std::size_t>(ip));
        const T* twre = dtw.passes[p].first.data();
        const T* twim = dtw.passes[p].second.data();

        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;

        poet::dispatch(dif_pass_last_invoke<T, Forward, Scale>{scale_val},
                       poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                       src_re, src_im, out, l1, ido, twre, twim);
    }
}

} // namespace detail
} // namespace admiral


#include "undef_macros.hpp"
