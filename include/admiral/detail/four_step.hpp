#pragma once

// ============================================================================
// Four-step (Cooley-Tukey) driver for N = N1*N2, both factors codelet catalog leaves
// (N1, N2 <= `kFourStepLeafMax`): two codelet passes + one twiddle multiply.
// UN-normalized; twiddles plan-owned.
//   input  n = n2*N1 + n1;  output k = k1*N2 + k2
//   inner: N1 size-N2 DFTs -> G[n1*N2+k2];  twiddle: G *= W_N^{n1*k2};
//   outer: N2 size-N1 DFTs over the G columns -> X[k1*N2+k2].
// `four_step_batched_ct` below takes some smooth sizes but needs N1%W==0 && N2%W==0.
// Ref: Bailey, "FFTs in external or hierarchical memory", J. Supercomput. 4
// (1990) 23. DOI 10.1007/BF00162341
// ============================================================================

#include <array>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include "codelet.hpp"       // `kernel_batched`, `xsimd::batch` (batched leaves)
#include "math.hpp"          // `codelet_dispatch`, `is_codelet_catalog`
#include "portable_trig.hpp" // `sincos_turns`

namespace admiral {
namespace detail {

struct four_step_split {
    std::size_t n1 = 0;
    std::size_t n2 = 0;
    [[nodiscard]] constexpr bool valid() const { return n1 != 0 && n2 != 0; }
};

// `kFourStepLeafMax` and the codelet cost tables live in `math.hpp`. The routing model
// and its offline fitter price every codelet-terminated form off that one table, and a
// header/fitter drift would be silent.

// Cost-optimal split N=N1*N2 with both factors in catalog; {0,0} if none exists.
[[nodiscard]] inline four_step_split choose_four_step_split(std::size_t N) {
    four_step_split best{};
    double best_cost = -1.0;
    for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
        if (N % n1 != 0) continue;
        const std::size_t n2 = N / n1;
        if (n1 > kFourStepLeafMax || n2 > kFourStepLeafMax) continue;
        if (!is_codelet_catalog(n1) || !is_codelet_catalog(n2)) continue;
        const double c = gate_four_step_cost(n1, n2);
        if (best_cost < 0.0 || c < best_cost) {
            best_cost = c;
            best = {n1, n2};
        }
    }
    return best;
}

// Stride penalty (cycles per element per saturated stride unit) and saturation length.
// f64 W=2 uses weight 0: no headroom there.
inline constexpr double kStridePenaltyF32 = 8.0;
inline constexpr double kStridePenaltyF64 = 4.0;  // W >= 4 only
inline constexpr std::size_t kStrideSatF32 = 12;
inline constexpr std::size_t kStrideSatF64 = 10;

// Execution-time split adds the stride penalty min(n1,L)+min(n2,L), which
// `four_step_cost` omits. The penalty is symmetric in (n1,n2), so no objective ranks the
// memory orders: the search stops at sqrt(N) and elects the smaller factor first.
template<typename T>
[[nodiscard]] inline four_step_split choose_four_step_split_exec(std::size_t N) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr double lam =
        sizeof(T) == 4 ? kStridePenaltyF32 : (W >= 4 ? kStridePenaltyF64 : 0.0);
    constexpr std::size_t L = sizeof(T) == 4 ? kStrideSatF32 : kStrideSatF64;
    if constexpr (lam == 0.0) {
        return choose_four_step_split(N);
    } else {
        four_step_split best{};
        double best_cost = -1.0;
        for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
            if (N % n1 != 0) continue;
            const std::size_t n2 = N / n1;
            if (n1 > kFourStepLeafMax || n2 > kFourStepLeafMax || !is_codelet_catalog(n1) || !is_codelet_catalog(n2)) continue;
            const double stride = double(std::min(n1, L) + std::min(n2, L));
            const double c = double(n1) * gate_leaf_cyc(n2) +
                             double(n2) * gate_leaf_cyc(n1) + lam * double(N) * stride;
            if (best_cost < 0.0 || c < best_cost) {
                best_cost = c;
                best = {n1, n2};
            }
        }
        return best;
    }
}

// Existence, not cost. The size guard is not redundant: a small composite has an
// admissible split too, but the four-step route exists only above the leaf ceiling.
[[nodiscard]] inline bool four_step_supported(std::size_t N) {
    return N > kFourStepLeafMax && choose_four_step_split(N).valid();
}

// Hand-fit zero-regression thresholds per (precision, W): wide f32 tightens hardest
// because Bluestein's pow2 leaves vectorize wide while strided leaves do not. One
// calibrated set with `gate_leaf_cyc_ref` and the Rader ratios.
inline constexpr double kFourStepGateF32Wide = 0.54;    // W >= 16
inline constexpr double kFourStepGateF32 = 0.74;        // W >= 8
inline constexpr double kFourStepGateF32Narrow = 0.885;
inline constexpr double kFourStepGateF64Wide = 0.95;    // W >= 8
inline constexpr double kFourStepGateF64Narrow = 1.10;

// If leaf cost < ratio * `bluestein_model_cost`, route four-step over Bluestein. The
// gate prices Bluestein's pad as the `bit_ceil` phantom behind the fitted ratios, not
// the smaller `bluestein_choose_pad` the engine runs (`math.hpp`).
template<typename T>
[[nodiscard]] inline bool four_step_beats_bluestein(std::size_t N) {
    const four_step_split sp = choose_four_step_split_exec<T>(N);
    if (!sp.valid()) return false;
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr double ratio =
        sizeof(T) == 4
            ? (W >= 16 ? kFourStepGateF32Wide : W >= 8 ? kFourStepGateF32 : kFourStepGateF32Narrow)
            : (W >= 8 ? kFourStepGateF64Wide : kFourStepGateF64Narrow);
    const double leaf = double(sp.n1) * gate_leaf_cyc(sp.n2) +
                        double(sp.n2) * gate_leaf_cyc(sp.n1);
    return leaf < ratio * bluestein_model_cost(N);
}

// Fill tw[idx(n1,k2)] = W_N^{n1*k2} with exact integer turn reduction; `idx` selects the
// layout (row-major for the scalar plan, V-contiguous for the batched one).
template<typename T, bool Forward, typename Index>
void fill_four_step_twiddles(std::size_t N1, std::size_t N2, std::complex<T>* tw, Index idx) {
    const std::size_t N = N1 * N2;
    for (std::size_t n1 = 0; n1 < N1; ++n1) {
        for (std::size_t k2 = 0; k2 < N2; ++k2) {
            const auto [sn, cs] = portable_trig::sincos_turns<Forward>(n1 * k2, N);
            tw[idx(n1, k2)] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
    }
}

// Plan-owned twiddle table W_N^{n1*k2}, laid out row-major [n1*N2 + k2].
template<typename T, bool Forward>
[[nodiscard]] inline std::vector<std::complex<T>>
build_four_step_twiddles(std::size_t N1, std::size_t N2) {
    std::vector<std::complex<T>> tw(N1 * N2);
    fill_four_step_twiddles<T, Forward>(
        N1, N2, tw.data(), [N2](std::size_t n1, std::size_t k2) { return n1 * N2 + k2; });
    return tw;
}

// DFT-N1*N2 on AoS `std::complex<T>`. If `in==out`, run in place. If `in!=out`, the
// call reads `in` fully before writing `out`. `G` is caller-owned scratch, length N1*N2.
// UN-normalized.
template<typename T, bool Forward>
void four_step_execute(const std::complex<T>* in, std::complex<T>* out,
                       std::size_t N1, std::size_t N2,
                       const std::complex<T>* tw, std::complex<T>* G) {
    std::complex<T> tmp[kFourStepLeafMax];  // one leaf's worth (the splitters bound N1,N2)

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
// Batched four-step: leaves run W-wide via `kernel_batched<N>` on planar data; scalar
// O(N) traffic is the entry deinterleave and exit reinterleave. Requires
// N1%W==0 && N2%W==0; guarded by `four_step_batched_supported`.
// Lane map: inner groups over n1, lane l scatters to G[(base+l)*N2+k2]; outer groups
// over k2 and stores contiguously.
// ============================================================================
// V-contiguous twiddles: entry [(g*N2+k2)*W+l] = W_N^{(g*W+l)*k2}; needs N1%W==0.
template<typename T, bool Forward>
[[nodiscard]] inline std::vector<std::complex<T>>
build_four_step_twiddles_v(std::size_t N1, std::size_t N2, std::size_t W) {
    std::vector<std::complex<T>> tw(N1 * N2);
    fill_four_step_twiddles<T, Forward>(
        N1, N2, tw.data(), [N2, W](std::size_t n1, std::size_t k2) {
            return ((n1 / W) * N2 + k2) * W + n1 % W;
        });
    return tw;
}

// Batched four-step on planar buffers (compile-time N1,N2).
// `Gre`/`Gim`: planar scratch; `twvre`/`twvim`: V-contiguous twiddles. `out` may alias `in`.
template<std::size_t N1, std::size_t N2, typename T, bool Forward, typename V = xsimd::batch<T>>
void four_step_batched_ct(const T* in_re, const T* in_im, T* out_re, T* out_im,
                          const T* twvre, const T* twvim, T* Gre, T* Gim) {
    constexpr std::size_t W = V::size;
    // if constexpr, not static_assert: dispatch instantiates every split for every W,
    // and an assert would break wide ISAs on narrow-tuned splits.
    if constexpr (N1 % W == 0 && N2 % W == 0) {
    // INNER: N1 size-N2 DFTs, W columns per group; twist + WxW register-transpose to G.
    for (std::size_t base = 0; base < N1; base += W) {
        V xv[N2], iv[N2], ov[N2], oi[N2];
        for (std::size_t n2 = 0; n2 < N2; ++n2) {
            xv[n2] = V::load_unaligned(in_re + n2 * N1 + base);
            iv[n2] = V::load_unaligned(in_im + n2 * N1 + base);
        }
        kernel_batched<N2, T, Forward, V>::apply(xv, iv, 1, ov, oi);
        const std::size_t g = base / W;
        // Twist in place: lane l (n1 = base+l) of ov[k2] *= W_N^{(base+l)*k2}.
        for (std::size_t k2 = 0; k2 < N2; ++k2) {
            const std::size_t tvi = (g * N2 + k2) * W;
            const V wr = V::load_unaligned(twvre + tvi);
            const V wi = V::load_unaligned(twvim + tvi);
            const V gr = ov[k2] * wr - oi[k2] * wi;
            const V gi = ov[k2] * wi + oi[k2] * wr;
            ov[k2] = gr;
            oi[k2] = gi;
        }
        // Transpose-scatter: WxW tile rows=k2, lanes=n1 -> rows=n1 contiguous in G.
        // vunpck/vperm, not vmaskmov/vgather.
        for (std::size_t k0 = 0; k0 < N2; k0 += W) {
            xsimd::transpose(ov + k0, ov + k0 + W);
            xsimd::transpose(oi + k0, oi + k0 + W);
            for (std::size_t a = 0; a < W; ++a) {
                ov[k0 + a].store_unaligned(Gre + (base + a) * N2 + k0);
                oi[k0 + a].store_unaligned(Gim + (base + a) * N2 + k0);
            }
        }
    }

    // OUTER: N2 size-N1 DFTs, W frequencies (k2) per group; contiguous in and out.
    for (std::size_t base2 = 0; base2 < N2; base2 += W) {
        V xv[N1], iv[N1], ov[N1], oi[N1];
        for (std::size_t n1 = 0; n1 < N1; ++n1) {
            xv[n1] = V::load_unaligned(Gre + n1 * N2 + base2);
            iv[n1] = V::load_unaligned(Gim + n1 * N2 + base2);
        }
        kernel_batched<N1, T, Forward, V>::apply(xv, iv, 1, ov, oi);
        for (std::size_t k1 = 0; k1 < N1; ++k1) {
            ov[k1].store_unaligned(out_re + k1 * N2 + base2);
            oi[k1].store_unaligned(out_im + k1 * N2 + base2);
        }
    }
    }  // if constexpr (N1 % W == 0 && N2 % W == 0)
}

// Batched four-step route (f32 W=8 only): wins where `iterative_dif` leaves the 8-wide
// register partly idle. f64 and the other widths fall back to `iterative_dif`.

// W=8-tuned splits; W=4 and W=16 stay off (the splits are not multiples of 16).
// {0,0} = not a batched-four-step size. Source for `fsb_split_for` and the dispatch.
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
    // Both leaves must be multiples of W, matching the guard in `four_step_batched_ct`.
    return sp.valid() && (sp.n1 % W == 0) && (sp.n2 % W == 0);
}

// Largest batched-four-step N (sizes the `execute()` stack scratch must hold).
inline constexpr std::size_t FSB_MAX_N = 768;

// Expand `fsb_splits` into one guarded `four_step_batched_ct` call per entry.
template<typename T, bool Forward, std::size_t... I>
inline void fsb_dispatch_pack(std::index_sequence<I...>, std::size_t N1, std::size_t N2,
        const T* ire, const T* iim, T* ore, T* oim,
        const T* twre, const T* twim, T* Gre, T* Gim) {
    (((N1 == fsb_splits[I].n1 && N2 == fsb_splits[I].n2)
          ? four_step_batched_ct<fsb_splits[I].n1, fsb_splits[I].n2, T, Forward>(
                ire, iim, ore, oim, twre, twim, Gre, Gim)
          : void()),
     ...);
}

// Runtime (N1,N2) -> compile-time `four_step_batched_ct` dispatch (f32 only; f64 body elided).
template<typename T, bool Forward>
inline void four_step_batched_dispatch([[maybe_unused]] std::size_t N1, [[maybe_unused]] std::size_t N2,
        [[maybe_unused]] const T* ire, [[maybe_unused]] const T* iim,
        [[maybe_unused]] T* ore, [[maybe_unused]] T* oim,
        [[maybe_unused]] const T* twre, [[maybe_unused]] const T* twim,
        [[maybe_unused]] T* Gre, [[maybe_unused]] T* Gim) {
    if constexpr (sizeof(T) == 4) {
        // Free helper (not lambda: capturing lambda materializes closure on stack).
        fsb_dispatch_pack<T, Forward>(std::make_index_sequence<fsb_splits.size()>{},
                                      N1, N2, ire, iim, ore, oim, twre, twim, Gre, Gim);
    }
}

// Plan state: split + V-contiguous planar twiddles (built once).
// `execute()`: deinterleave -> batched four-step -> reinterleave. UN-normalized.
template<typename T>
struct four_step_batched_plan {
    std::size_t n1 = 0, n2 = 0;
    bool is_forward = true;
    std::vector<T> twre, twim;  // V-contiguous planar twiddles for `is_forward`

    four_step_batched_plan(std::size_t N, bool fwd) : is_forward(fwd) {
        const four_step_split sp = fsb_split_for<T>(N);
        n1 = sp.n1;
        n2 = sp.n2;
        constexpr std::size_t W = xsimd::batch<T>::size;
        const auto tw = fwd ? build_four_step_twiddles_v<T, true>(n1, n2, W)
                            : build_four_step_twiddles_v<T, false>(n1, n2, W);
        twre.resize(N);
        twim.resize(N);
        for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
    }

    // If `in==out`, run in place. If `in!=out`, the call consumes `in` at the entry
    // deinterleave and writes `out` at the exit. Scratch stays a local array, not a
    // helper parameter. Behind plain `T*`, clang cannot prove no-alias and emits a
    // runtime alias check plus a scalar fallback deinterleave.
    void execute(const std::complex<T>* in, std::complex<T>* out) const {
        const std::size_t N = std::size_t(n1) * n2;
        // Every supported split has N <= `FSB_MAX_N`, so all four planes stay on the stack.
        alignas(xsimd::batch<T>::arch_type::alignment()) T are[FSB_MAX_N], aim[FSB_MAX_N], Gre[FSB_MAX_N], Gim[FSB_MAX_N];
        for (std::size_t i = 0; i < N; ++i) { are[i] = in[i].real(); aim[i] = in[i].imag(); }
        if (is_forward)
            four_step_batched_dispatch<T, true>(n1, n2, are, aim, are, aim, twre.data(), twim.data(), Gre, Gim);
        else
            four_step_batched_dispatch<T, false>(n1, n2, are, aim, are, aim, twre.data(), twim.data(), Gre, Gim);
        for (std::size_t i = 0; i < N; ++i) out[i] = std::complex<T>(are[i], aim[i]);
    }
};

} // namespace detail
} // namespace admiral
