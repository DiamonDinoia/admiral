#pragma once

// ============================================================================
// Vectorized Good-Thomas PFA kernel: N = N1*N2*N3 pairwise coprime, no inter-stage
// twiddles. UN-normalized; the inverse runs the forward kernel on conjugated data.
// Every index and extent is `std::size_t`, so nothing casts.
// Refs: Good, J. R. Stat. Soc. B 20 (1958) 361, DOI 10.1111/j.2517-6161.1958.tb00300.x;
// Thomas, Appl. Digital Computers (1963).
// ============================================================================

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>   // `std::gcd`

#include "simd.hpp"
#include "cxx_compat.hpp"  // `ADM_CONSTEVAL`, `detail::bit_floor`, `detail::has_single_bit`
#include <poet/poet.hpp>

#include "butterfly.hpp"  // `dif_butterfly` (symmetric odd-radix + recursive Cooley-Tukey DIF pow2)
#include "simd_swizzle.hpp"  // `aos_even_lane`, `aos_odd_lane` (shared de-interleave generators)
#include "macros.hpp"   // `ADM_ALWAYS_INLINE` (`butterfly.hpp` undefs it on the way out)

namespace admiral {
namespace detail {

[[nodiscard]] ADM_CONSTEVAL bool good_thomas_coprime(std::size_t a, std::size_t b) noexcept {
    return std::gcd(a, b) == 1;
}

// Ruritanian input map.
template<std::size_t N1, std::size_t N2, std::size_t N3>
[[nodiscard]] constexpr std::size_t good_thomas_in_idx(std::size_t n1, std::size_t n2,
                                                       std::size_t n3) noexcept {
    constexpr std::size_t N = N1*N2*N3;
    return (n1 * (N/N1) + n2 * (N/N2) + n3 * (N/N3)) % N;
}

// Gather mask builder. `xsimd::shuffle(a, b, mask)`: mask[i] in [0,W) selects a[mask[i]],
// in [W,2W) selects b[mask[i]-W]. `make_batch_constant` requires Arr.size()==W.

// Shuffle index width matches the lane width.
template<typename T>
using good_thomas_mask_u = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;

// S-input gather: a binary tree of two-input shuffles, one mask per internal node.

// Split rule, shared by every level of the tree so the gather and its masks agree.
constexpr std::size_t gt_split_left(std::size_t S) noexcept { return detail::bit_floor(S - 1); }

// One mask lane i for one tree node: a covers [lo,lo+L), b covers [lo+L,lo+L+R). A lane
// whose source lies outside the node is don't-care; an upper node overwrites the lane.
template<typename U>
[[nodiscard]] constexpr U gt_combine_lane(std::size_t s, std::size_t l,
                                          std::size_t lo, std::size_t L, std::size_t R,
                                          std::size_t i, std::size_t W) noexcept {
    const bool a_leaf = (L == 1), b_leaf = (R == 1);
    if (s >= lo && s < lo + L)           return static_cast<U>(a_leaf ? l : i);
    if (s >= lo + L && s < lo + L + R)   return static_cast<U>(W + (b_leaf ? l : i));
    return static_cast<U>(a_leaf ? i : W + i);
}

// Masks ride generator types keyed by integral NTTPs (class NTTPs are C++20-only); xsimd
// materializes each `batch_constant` from the generator. `Map::get(entry, ml)` returns
// {source batch, source lane} for one lane of one table slot.
template<std::size_t N1, std::size_t N2, std::size_t N3, std::size_t W, std::size_t BA>
struct gt_map_a {
    [[nodiscard]] static constexpr std::array<std::size_t, 2> get(std::size_t entry,
                                                                  std::size_t ml) noexcept {
        const std::size_t a = entry / BA, b = entry % BA;
        const std::size_t pos = b * W + ml;
        if (pos < N1 * N2) {
            const std::size_t nat = good_thomas_in_idx<N1, N2, N3>(pos / N2, pos % N2, a);
            return {nat / W, nat % W};
        }
        return {0, 0};
    }
};

template<std::size_t N1, std::size_t N2, std::size_t N3, std::size_t W,
         std::size_t BA, std::size_t BB>
struct gt_map_b {
    [[nodiscard]] static constexpr std::array<std::size_t, 2> get(std::size_t entry,
                                                                  std::size_t ml) noexcept {
        const std::size_t jp = entry / BB, b = entry % BB;
        const std::size_t pos = b * W + ml;
        if (pos < N1 * N3) {
            const std::size_t n1 = pos / N3, k3 = pos % N3;
            const std::size_t lane_in_arm = n1 * N2 + jp;
            return {k3 * BA + lane_in_arm / W, lane_in_arm % W};
        }
        return {0, 0};
    }
};

template<std::size_t N1, std::size_t N2, std::size_t N3, std::size_t W,
         std::size_t BB, std::size_t BC>
struct gt_map_c {
    [[nodiscard]] static constexpr std::array<std::size_t, 2> get(std::size_t entry,
                                                                  std::size_t ml) noexcept {
        const std::size_t k1 = entry / BC, b = entry % BC;
        const std::size_t pos = b * W + ml;
        if (pos < N2 * N3) {
            const std::size_t k2 = pos / N3, k3 = pos % N3;
            const std::size_t lane_in_arm = k1 * N3 + k3;
            return {k2 * BB + lane_in_arm / W, lane_in_arm % W};
        }
        return {0, 0};
    }
};

// Output permutation: the table slot is the output batch itself (no arm/batch split).
template<std::size_t N1, std::size_t N2, std::size_t N3, std::size_t W, std::size_t BC>
struct gt_map_out {
    [[nodiscard]] static constexpr std::array<std::size_t, 2> get(std::size_t zout,
                                                                  std::size_t ml) noexcept {
        constexpr std::size_t N = N1 * N2 * N3;
        const std::size_t k = zout * W + ml;
        if (k < N) {
            const std::size_t k1 = k % N1, k2 = k % N2, k3 = k % N3;
            const std::size_t pos_in_arm = k2 * N3 + k3;
            return {k1 * BC + pos_in_arm / W, pos_in_arm % W};
        }
        return {0, 0};
    }
};

// Generator for one node mask of one table entry; S == 1 is the leaf.
template<class Map, std::size_t W, class U, std::size_t Entry,
         std::size_t Lo, std::size_t S, std::size_t Nb>
struct gt_mask_gen {
    [[nodiscard]] static constexpr U get(std::size_t i, std::size_t) noexcept {
        const auto sl = Map::get(Entry, i);
        if constexpr (S == 1) {
            return static_cast<U>(sl[1]);
        } else {
            constexpr std::size_t L = gt_split_left(S), R = S - L;
            return gt_combine_lane<U>(sl[0], sl[1], Lo, L, R, i, W);
        }
    }
};

template<std::size_t Lo, std::size_t S, std::size_t Nb, class Map, std::size_t W, class U,
         std::size_t Entry, typename Batch, std::size_t A>
[[nodiscard]] ADM_ALWAYS_INLINE Batch gt_gather_sub(const std::array<Batch, A>& src) noexcept {
    if constexpr (S == 1) {
        return src[Lo];   // leaf: raw source, parent shuffles it
    } else {
        using Arch = typename Batch::arch_type;
        constexpr std::size_t L = gt_split_left(S), R = S - L;
        return xsimd::shuffle(
            gt_gather_sub<Lo, L, Nb + 1, Map, W, U, Entry>(src),
            gt_gather_sub<Lo + L, R, Nb + L, Map, W, U, Entry>(src),
            xsimd::make_batch_constant<U, gt_mask_gen<Map, W, U, Entry, Lo, S, Nb>, Arch>());
    }
}

template<std::size_t NumSrc, class Map, std::size_t W, class U, std::size_t Entry,
         typename Batch, std::size_t A>
[[nodiscard]] ADM_ALWAYS_INLINE Batch good_thomas_gather(const std::array<Batch, A>& src) noexcept {
    static_assert(NumSrc >= 1 && NumSrc <= A, "gather: NumSrc out of range");
    if constexpr (NumSrc == 1) {
        using Arch = typename Batch::arch_type;
        using Gen = gt_mask_gen<Map, W, U, Entry, 0, 1, 0>;
        return xsimd::swizzle(src[0], xsimd::make_batch_constant<U, Gen, Arch>());
    }
    return gt_gather_sub<0, NumSrc, 0, Map, W, U, Entry>(src);
}

// Radix butterfly on arm slots {k*B+Z}, shared by all three stages. Radix==1 no-op.
template<std::size_t Radix, std::size_t B, std::size_t Z, typename Batch, std::size_t S>
ADM_ALWAYS_INLINE void good_thomas_apply_dft(std::array<Batch, S>& xr,
                                             std::array<Batch, S>& xi) noexcept {
    if constexpr (Radix >= 2) {
        // forward-only: the PFA inverse conjugates outside, so the butterfly stays forward.
        using T = typename Batch::value_type;
        Batch tr[Radix], ti[Radix];
        poet::static_for<Radix>([&](auto K) { tr[K] = xr[K*B+Z]; ti[K] = xi[K*B+Z]; });
        dif_butterfly<T, Radix>(tr, ti, [&](auto K, Batch yr, Batch yi) {
            xr[K*B+Z] = yr;
            xi[K*B+Z] = yi;
        });
    }
}

// True iff: pairwise coprime, factors in {1,2,3,4,5,8,16}, and some stage boundary fits
// the register file. The register bounds over-count; `base_cost_model.hpp` routes finally.
template<typename T, std::size_t N1, std::size_t N2, std::size_t N3>
[[nodiscard]] ADM_CONSTEVAL bool good_thomas_eligible() noexcept {
    constexpr std::size_t N  = N1*N2*N3;
    constexpr std::size_t W  = xsimd::batch<T>::size;
    constexpr std::size_t S  = (N + W - 1) / W;          // ceil(N/W)
    constexpr std::size_t Mx = std::max({N1, N2, N3});

    if (!good_thomas_coprime(N1, N2) || !good_thomas_coprime(N1, N3) || !good_thomas_coprime(N2, N3))
        return false;

    // 1 is a no-op identity stage.
    auto supported = [](std::size_t f) {
        return f==1 || f==2 || f==3 || f==4 || f==5 || f==8 || f==16;
    };
    if (!supported(N1) || !supported(N2) || !supported(N3))
        return false;

    // (3a) Tightest stage boundary must fit the register file: a kernel over budget at
    // every boundary spills throughout, so the test is the minimum boundary.
    constexpr std::size_t BA = (N1*N2 + W - 1) / W;
    constexpr std::size_t BB = (N1*N3 + W - 1) / W;
    constexpr std::size_t BC = (N2*N3 + W - 1) / W;
    constexpr std::size_t min_boundary =
        4 + 2 * std::min({S + N3*BA, N3*BA + N2*BB, N2*BB + N1*BC, N1*BC + S});
    if (min_boundary > poet::vector_register_count()) return false;

    // (3b) register fit OR small-N. The bound over-counts peak live registers: the
    // butterfly callback reuses arm temps. W>=4 excludes SSE-f64 large-buffer kernels.
    if (2*S + 2*Mx + 4 <= poet::vector_register_count()) return true;
    // A pow2 largest factor emits each arm temp through the callback, so the bound
    // over-counts that case by Mx. The relaxed bound stays strict: exact register
    // saturation loses to `iterative_dif` because the CRT permutation cost grows with S.
    if (detail::has_single_bit(Mx) && 2*S + Mx + 4 < poet::vector_register_count()) return true;
    return N <= 32 && W >= 4;
}

// DFT-N on AoS `std::complex<T>`[N]. UN-normalized; the inverse negates Im before and
// after. If `in==out`, the call is safe: the kernel loads all of `in` before writing
// any of `out`, so one instantiation covers both.
template<typename T, std::size_t N1, std::size_t N2, std::size_t N3, bool Forward>
ADM_NOINLINE void good_thomas_execute(const std::complex<T>* in,
                                      std::complex<T>* out) noexcept {
    static_assert(good_thomas_coprime(N1,N2) && good_thomas_coprime(N1,N3) && good_thomas_coprime(N2,N3),
                  "good_thomas_execute: factors must be pairwise coprime");

    using Batch = xsimd::batch<T>;
    using Arch  = typename Batch::arch_type;
    static constexpr std::size_t N  = N1*N2*N3;
    static constexpr std::size_t W  = Batch::size;
    static constexpr std::size_t NS = (N + W - 1) / W;   // ceil(N/W): source batches
    static constexpr std::size_t BA = (N1*N2 + W - 1) / W;
    static constexpr std::size_t BB = (N1*N3 + W - 1) / W;
    static constexpr std::size_t BC = (N2*N3 + W - 1) / W;
    using U = good_thomas_mask_u<T>;

    // 1. AoS -> SoA, tail masked at 2N. No `ADM_RESTRICT`: step 1 loads all of `in`
    //    before step 6 writes, so `in==out` is safe and `ADM_RESTRICT` would be UB.
    std::array<Batch, NS> re_src, im_src;
    const T* ip = reinterpret_cast<const T*>(in);
    auto load_tail = [&](auto o_ic) {
        constexpr std::size_t O = std::decay_t<decltype(o_ic)>::value;
        if constexpr (O + W <= 2 * N) {
            return Batch::load_unaligned(ip + O);
        } else if constexpr (O >= 2 * N) {
            return Batch(T(0));
        } else {
            struct in_bounds {
                static constexpr bool get(std::size_t i, std::size_t) noexcept { return O + i < 2 * N; }
            };
            constexpr auto m = xsimd::make_batch_bool_constant<T, in_bounds, Arch>();
            return Batch::load(ip + O, m, xsimd::unaligned_mode{});
        }
    };
    poet::static_for<0, NS>([&](const auto s) {
        const Batch lo = load_tail(std::integral_constant<std::size_t, s * W * 2>{});
        const Batch hi = load_tail(std::integral_constant<std::size_t, s * W * 2 + W>{});
        re_src[s] = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<U, aos_even_lane, Arch>());
        im_src[s] = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<U, aos_odd_lane,  Arch>());
    });

    if constexpr (!Forward) {
        poet::static_for<0, NS>([&](const auto s) { im_src[s] = -im_src[s]; });
    }

    // 2. Stage A: DFT-N3 over the n3 axis.
    std::array<Batch, N3*BA> Ar, Ai;
    poet::static_for<0, N3*BA>([&](const auto JZ) {
        constexpr std::size_t jz = decltype(JZ)::value;
        Ar[JZ] = good_thomas_gather<NS, gt_map_a<N1, N2, N3, W, BA>, W, U, jz>(re_src);
        Ai[JZ] = good_thomas_gather<NS, gt_map_a<N1, N2, N3, W, BA>, W, U, jz>(im_src);
    });
    poet::static_for<0, BA>([&](const auto Z) {
        good_thomas_apply_dft<N3, BA, Z>(Ar, Ai);
    });

    // 3. Stage B: DFT-N2 over the n2 axis.
    std::array<Batch, N2*BB> Br, Bi;
    poet::static_for<0, N2*BB>([&](const auto JPZ) {
        constexpr std::size_t jpz = decltype(JPZ)::value;
        Br[JPZ] = good_thomas_gather<N3*BA, gt_map_b<N1, N2, N3, W, BA, BB>, W, U, jpz>(Ar);
        Bi[JPZ] = good_thomas_gather<N3*BA, gt_map_b<N1, N2, N3, W, BA, BB>, W, U, jpz>(Ai);
    });
    poet::static_for<0, BB>([&](const auto Z) {
        good_thomas_apply_dft<N2, BB, Z>(Br, Bi);
    });

    // 4. Stage C: DFT-N1 over the n1 axis.
    std::array<Batch, N1*BC> Cr, Ci;
    poet::static_for<0, N1*BC>([&](const auto K1Z) {
        constexpr std::size_t k1z = decltype(K1Z)::value;
        Cr[K1Z] = good_thomas_gather<N2*BB, gt_map_c<N1, N2, N3, W, BB, BC>, W, U, k1z>(Br);
        Ci[K1Z] = good_thomas_gather<N2*BB, gt_map_c<N1, N2, N3, W, BB, BC>, W, U, k1z>(Bi);
    });
    poet::static_for<0, BC>([&](const auto Z) {
        good_thomas_apply_dft<N1, BC, Z>(Cr, Ci);
    });

    // 5. Output gather: CRT permutation back to natural order.
    std::array<Batch, NS> Or, Oi;
    poet::static_for<0, NS>([&](const auto ZO) {
        constexpr std::size_t zo = decltype(ZO)::value;
        Or[ZO] = good_thomas_gather<N1*BC, gt_map_out<N1, N2, N3, W, BC>, W, U, zo>(Cr);
        Oi[ZO] = good_thomas_gather<N1*BC, gt_map_out<N1, N2, N3, W, BC>, W, U, zo>(Ci);
    });

    if constexpr (!Forward) {
        poet::static_for<0, NS>([&](const auto s) { Oi[s] = -Oi[s]; });
    }

    // 6. SoA -> AoS. Same aliasing as step 1.
    T* op = reinterpret_cast<T*>(out);
    auto store_tail = [&](auto o_ic, const Batch& v) {
        constexpr std::size_t O = std::decay_t<decltype(o_ic)>::value;
        if constexpr (O + W <= 2 * N) {
            v.store_unaligned(op + O);
        } else if constexpr (O < 2 * N) {
            struct in_bounds {
                static constexpr bool get(std::size_t i, std::size_t) noexcept { return O + i < 2 * N; }
            };
            constexpr auto m = xsimd::make_batch_bool_constant<T, in_bounds, Arch>();
            v.store(op + O, m, xsimd::unaligned_mode{});
        }  // O >= 2N: whole batch past the end, nothing to store
    };
    poet::static_for<0, NS>([&](const auto s) {
        store_tail(std::integral_constant<std::size_t, s * W * 2>{}, xsimd::zip_lo(Or[s], Oi[s]));
        store_tail(std::integral_constant<std::size_t, s * W * 2 + W>{}, xsimd::zip_hi(Or[s], Oi[s]));
    });
}

// PFA catalog: one `good_thomas_desc` per routed size; a size is a one-line diff here.
// `base_cost_model.hpp` owns per-cell precision routing.
template<std::size_t N1, std::size_t N2, std::size_t N3>
struct good_thomas_desc {
    static constexpr std::size_t n = N1 * N2 * N3;
    template<typename T>
    static constexpr bool admits = good_thomas_eligible<T, N1, N2, N3>();
    template<typename T, bool Forward>
    static void run(const std::complex<T>* in, std::complex<T>* out) noexcept {
        good_thomas_execute<T, N1, N2, N3, Forward>(in, out);
    }
};

template<class... Ds>
struct good_thomas_catalog_t {
    template<typename T>
    static constexpr bool available(std::size_t n) noexcept {
        return ((Ds::template admits<T> && n == Ds::n) || ...);
    }
    // `admits<T>` mirrors `available()`; ineligible kernels are never instantiated.
    template<typename T, bool Forward>
    static void run(const std::complex<T>* in, std::complex<T>* out, std::size_t n) noexcept {
        ([&] {
            if constexpr (Ds::template admits<T>) {
                if (n == Ds::n) Ds::template run<T, Forward>(in, out);
            }
        }(), ...);
    }
};

using good_thomas_catalog = good_thomas_catalog_t<
    good_thomas_desc<2, 1, 5>,         // 10
    good_thomas_desc<4, 3, 1>,         // 12
    good_thomas_desc<3, 1, 5>,         // 15
    good_thomas_desc<4, 1, 5>,         // 20
    good_thomas_desc<8, 3, 1>,         // 24
    good_thomas_desc<2, 3, 5>,         // 30
    good_thomas_desc<8, 1, 5>,         // 40
    good_thomas_desc<16, 3, 1>,        // 48
    good_thomas_desc<3, 4, 5>>;        // 60

// Trampoline: the extern templates keep the PFA tree out of routing TUs. Defined in
// `inst_gt_f.cpp` / `inst_gt_d.cpp`.
template<typename T, bool Forward>
void good_thomas_run(const std::complex<T>* in, std::complex<T>* out, std::size_t n) noexcept {
    good_thomas_catalog::run<T, Forward>(in, out, n);
}

extern template void good_thomas_run<float, true>(const std::complex<float>*, std::complex<float>*, std::size_t) noexcept;
extern template void good_thomas_run<float, false>(const std::complex<float>*, std::complex<float>*, std::size_t) noexcept;
extern template void good_thomas_run<double, true>(const std::complex<double>*, std::complex<double>*, std::size_t) noexcept;
extern template void good_thomas_run<double, false>(const std::complex<double>*, std::complex<double>*, std::size_t) noexcept;

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
