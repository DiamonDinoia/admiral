#pragma once

// ============================================================================
// Vectorized PFA (Good-Thomas) DFT kernel: N = N1*N2*N3 pairwise coprime, no inter-stage
// twiddle factors. Generic over T (f32/f64) and W = xsimd::batch<T>::size; eligibility
// predicate good_thomas_eligible<T,N1,N2,N3>() routes it.
//
// Index maps, derived at compile time from the factors:
//   in  (Ruritanian): in_idx(n1,n2,n3) = (n1*N/N1 + n2*N/N2 + n3*N/N3) % N
//   out (CRT):        k -> (k%N1, k%N2, k%N3), applied directly in the output
//                     permutation table, with no modular inverse needed.
// Butterflies run dif_butterfly; gathers are binary two-input shuffle trees
// (GatherMasks / good_thomas_gather). Inverse via conjugate identity:
// IDFT(x) = conj(FWD(conj(x))). UN-normalized; caller applies 1/N for inverse.
// Every extent, index and factor here is std::size_t: the natural type leaves nothing
// to cast.
//
// Ref: Good, "The interaction algorithm and practical Fourier analysis", J. R.
// Stat. Soc. B 20 (1958) 361, DOI 10.1111/j.2517-6161.1958.tb00300.x; Thomas,
// "Using a computer to solve problems in physics", Appl. Digital Computers (1963).
// ============================================================================

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>   // std::gcd

#include "simd.hpp"
#include "cxx_compat.hpp"  // ADM_CONSTEVAL, detail::bit_floor, detail::has_single_bit
#include <poet/poet.hpp>

#include "butterfly.hpp"  // dif_butterfly (symmetric odd-radix + recursive Cooley-Tukey DIF pow2)
#include "simd_swizzle.hpp"  // aos_even_lane, aos_odd_lane (shared de-interleave generators)
#include "macros.hpp"   // ADM_ALWAYS_INLINE (butterfly.hpp undefs it on the way out)

namespace admiral {
namespace detail {

[[nodiscard]] ADM_CONSTEVAL bool good_thomas_coprime(std::size_t a, std::size_t b) noexcept {
    return std::gcd(a, b) == 1;
}

// Natural input index from cube coordinates: Ruritanian map, coeff_i = N/Ni.
template<std::size_t N1, std::size_t N2, std::size_t N3>
[[nodiscard]] constexpr std::size_t good_thomas_in_idx(std::size_t n1, std::size_t n2,
                                                       std::size_t n3) noexcept {
    constexpr std::size_t N = N1*N2*N3;
    return (n1 * (N/N1) + n2 * (N/N2) + n3 * (N/N3)) % N;
}

// ============================================================================
// Gather mask builder (W-parametrized, GatherMasks as NTTP).
// make_batch_constant requires Arr.size()==W; masks are std::array<U,W>.
//
// xsimd::shuffle(a, b, mask): mask[i] in [0,W-1] → a[mask[i]], [W,2W-1] → b[mask[i]-W].
// ============================================================================

// Shuffle index type: uint32_t (f32/32-bit lanes), uint64_t (f64/64-bit lanes).
template<typename T>
using good_thomas_mask_u = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;

// S-input gather → one W-wide result: binary tree of S-1 two-input shuffles.
// Split: L=bit_floor(S-1) left, R=S-L right. Lone source = leaf; S==1 = single swizzle.
// One W-wide mask per internal node, pre-order.

// Split rule, shared by every level of the tree so the gather and its masks agree.
constexpr std::size_t gt_split_left(std::size_t S) noexcept { return detail::bit_floor(S - 1); }

// Shuffle mask for one tree node: a=[lo,lo+L), b=[lo+L,lo+L+R).
// Leaf contributes lane slane; subtree holds value at lane i.
// Don't-care lanes fold to b (subtree-a node) or identity (leaf), overwritten higher up.
// One lane of the node mask: source `s`/`l`, node covers [lo,lo+L+R), output lane i.
template<typename U>
[[nodiscard]] constexpr U gt_combine_lane(std::size_t s, std::size_t l,
                                          std::size_t lo, std::size_t L, std::size_t R,
                                          std::size_t i, std::size_t W) noexcept {
    const bool a_leaf = (L == 1), b_leaf = (R == 1);
    if (s >= lo && s < lo + L)           return static_cast<U>(a_leaf ? l : i);
    if (s >= lo + L && s < lo + L + R)   return static_cast<U>(W + (b_leaf ? l : i));
    return static_cast<U>(a_leaf ? i : W + i);
}

// Mask plumbing. No stage stores a mask table: a gather's masks ride generator TYPES
// (get(i, size)) keyed by integral NTTPs, the stage Map plus the (entry, lo, S, nb) node
// coordinates, and xsimd materializes each batch_constant straight from the generator.
// Map::get(entry, ml) returns {source batch, source lane} for one lane of one table slot;
// gt_combine_lane turns that into the shuffle index.
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

// Generator for one node mask of one table entry. S == 1 is the leaf-table case
// (good_thomas_make_masks stores the raw slane there); S >= 2 is one combine node.
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

// Apply radix-Radix butterfly to arm slots {k*B+Z : k in [0,Radix)} of SoA (re,im) buffers.
// Shared by all three PFA stages (A/B/C). Radix==1 is a no-op.
template<std::size_t Radix, std::size_t B, std::size_t Z, typename Batch, std::size_t S>
ADM_ALWAYS_INLINE void good_thomas_apply_dft(std::array<Batch, S>& xr,
                                             std::array<Batch, S>& xi) noexcept {
    if constexpr (Radix >= 2) {
        // Gather arm slots, run dif_butterfly (butterfly.hpp, forward-only), scatter back.
        // PFA inverse via conjugation, so this stays forward.
        using T = typename Batch::value_type;
        Batch tr[Radix], ti[Radix];
        poet::static_for<Radix>([&](auto K) { tr[K] = xr[K*B+Z]; ti[K] = xi[K*B+Z]; });
        dif_butterfly<T, Radix>(tr, ti, [&](auto K, Batch yr, Batch yi) {
            xr[K*B+Z] = yr;
            xi[K*B+Z] = yi;
        });
    }
}

// ============================================================================
// good_thomas_eligible<T, N1, N2, N3>()
//
// Returns true iff factors are pairwise coprime, in the butterfly set
// {1,2,3,4,5,8,16}, and some stage boundary fits the register file. The register
// bounds over-count peak live registers: base_cost_model.hpp is the final routing
// authority.
// ============================================================================
template<typename T, std::size_t N1, std::size_t N2, std::size_t N3>
[[nodiscard]] ADM_CONSTEVAL bool good_thomas_eligible() noexcept {
    constexpr std::size_t N  = N1*N2*N3;
    constexpr std::size_t W  = xsimd::batch<T>::size;
    constexpr std::size_t S  = (N + W - 1) / W;          // ceil(N/W)
    constexpr std::size_t Mx = std::max({N1, N2, N3});

    // (1) pairwise coprime
    if (!good_thomas_coprime(N1, N2) || !good_thomas_coprime(N1, N3) || !good_thomas_coprime(N2, N3))
        return false;

    // (2) factors in supported butterfly set (1 is a no-op identity stage)
    auto supported = [](std::size_t f) {
        return f==1 || f==2 || f==3 || f==4 || f==5 || f==8 || f==16;
    };
    if (!supported(N1) || !supported(N2) || !supported(N3))
        return false;

    // (3a) A gather loop holds the stage it reads plus the stage it fills, so the live
    // set at each stage boundary is one adjacent pair plus the butterfly temporaries. A
    // kernel over the register file at every boundary spills throughout; over it at only
    // some boundaries it still wins, so the test is the minimum.
    constexpr std::size_t BA = (N1*N2 + W - 1) / W;
    constexpr std::size_t BB = (N1*N3 + W - 1) / W;
    constexpr std::size_t BC = (N2*N3 + W - 1) / W;
    constexpr std::size_t min_boundary =
        4 + 2 * std::min({S + N3*BA, N3*BA + N2*BB, N2*BB + N1*BC, N1*BC + S});
    if (min_boundary > poet::vector_register_count()) return false;

    // (3b) register fit OR small-N. 2S+2Mx+4 over-counts peak live regs (the butterfly
    // reuses arm temps via callback), false-excluding a band of small sizes.
    // W>=4 excludes SSE-f64 (W=2) large-buffer instantiations.
    if (2*S + 2*Mx + 4 <= poet::vector_register_count()) return true;
    // A power-of-two largest factor runs pow2_dif_butterfly, which emits each arm temp
    // through the callback rather than holding 2*Mx live, so the bound above over-counts
    // that case by Mx. The relaxed bound is STRICT: the CRT permutation cost grows with
    // S while the arm arithmetic does not, so a form that exactly saturates the register
    // file loses to iterative_dif.
    if (detail::has_single_bit(Mx) && 2*S + Mx + 4 < poet::vector_register_count()) return true;
    return N <= 32 && W >= 4;
}

// good_thomas_execute<T, N1, N2, N3>(in, out, forward)
//
// DFT-N on AoS complex<T>[N], N = N1*N2*N3, pairwise coprime. UN-normalized.
// in==out: in-place. in!=out: reads in (preserved), writes out.
// Inverse: negate Im before and after the forward kernel.
// One instantiation covers both: the kernel loads all of `in` before it writes
// any of `out`, so aliasing needs no separate arm.
// ============================================================================
template<typename T, std::size_t N1, std::size_t N2, std::size_t N3, bool Forward>
ADM_NOINLINE void good_thomas_execute(const std::complex<T>* in,
                                      std::complex<T>* out) noexcept {
    static_assert(good_thomas_coprime(N1,N2) && good_thomas_coprime(N1,N3) && good_thomas_coprime(N2,N3),
                  "good_thomas_execute: factors must be pairwise coprime");

    using Batch = xsimd::batch<T>;
    using Arch  = typename Batch::arch_type;
    static constexpr std::size_t N  = N1*N2*N3;
    static constexpr std::size_t W  = Batch::size;
    static constexpr std::size_t NS = (N + W - 1) / W;   // ceil(N/W): source ZMMs
    static constexpr std::size_t BA = (N1*N2 + W - 1) / W;
    static constexpr std::size_t BB = (N1*N3 + W - 1) / W;
    static constexpr std::size_t BC = (N2*N3 + W - 1) / W;
    using U = good_thomas_mask_u<T>;

    // -------------------------------------------------------------------------
    // 1. AoS → SoA: load N complex<T> → NS ZMM pairs (re[s], im[s]).
    //    Direct unaligned load, tail masked at 2N. No ADM_RESTRICT and no staging
    //    buffer: step 1 loads every input before step 6 writes anything, so in==out
    //    is safe, and restrict would make that call UB.
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 2. Stage A: DFT-N3 over n3 axis.  N3 arms × BA ZMMs each.
    // -------------------------------------------------------------------------
    std::array<Batch, N3*BA> Ar, Ai;
    poet::static_for<0, N3*BA>([&](const auto JZ) {
        constexpr std::size_t jz = decltype(JZ)::value;
        Ar[JZ] = good_thomas_gather<NS, gt_map_a<N1, N2, N3, W, BA>, W, U, jz>(re_src);
        Ai[JZ] = good_thomas_gather<NS, gt_map_a<N1, N2, N3, W, BA>, W, U, jz>(im_src);
    });
    poet::static_for<0, BA>([&](const auto Z) {
        good_thomas_apply_dft<N3, BA, Z>(Ar, Ai);
    });

    // -------------------------------------------------------------------------
    // 3. Stage B: DFT-N2 over n2 axis.  N2 arms × BB ZMMs each.
    // -------------------------------------------------------------------------
    std::array<Batch, N2*BB> Br, Bi;
    poet::static_for<0, N2*BB>([&](const auto JPZ) {
        constexpr std::size_t jpz = decltype(JPZ)::value;
        Br[JPZ] = good_thomas_gather<N3*BA, gt_map_b<N1, N2, N3, W, BA, BB>, W, U, jpz>(Ar);
        Bi[JPZ] = good_thomas_gather<N3*BA, gt_map_b<N1, N2, N3, W, BA, BB>, W, U, jpz>(Ai);
    });
    poet::static_for<0, BB>([&](const auto Z) {
        good_thomas_apply_dft<N2, BB, Z>(Br, Bi);
    });

    // -------------------------------------------------------------------------
    // 4. Stage C: DFT-N1 over n1 axis.  N1 arms × BC ZMMs each.
    // -------------------------------------------------------------------------
    std::array<Batch, N1*BC> Cr, Ci;
    poet::static_for<0, N1*BC>([&](const auto K1Z) {
        constexpr std::size_t k1z = decltype(K1Z)::value;
        Cr[K1Z] = good_thomas_gather<N2*BB, gt_map_c<N1, N2, N3, W, BB, BC>, W, U, k1z>(Br);
        Ci[K1Z] = good_thomas_gather<N2*BB, gt_map_c<N1, N2, N3, W, BB, BC>, W, U, k1z>(Bi);
    });
    poet::static_for<0, BC>([&](const auto Z) {
        good_thomas_apply_dft<N1, BC, Z>(Cr, Ci);
    });

    // -------------------------------------------------------------------------
    // 5. Output gather: CRT permutation, N1*BC Stage-C ZMMs → NS natural-order.
    // -------------------------------------------------------------------------
    std::array<Batch, NS> Or, Oi;
    poet::static_for<0, NS>([&](const auto ZO) {
        constexpr std::size_t zo = decltype(ZO)::value;
        Or[ZO] = good_thomas_gather<N1*BC, gt_map_out<N1, N2, N3, W, BC>, W, U, zo>(Cr);
        Oi[ZO] = good_thomas_gather<N1*BC, gt_map_out<N1, N2, N3, W, BC>, W, U, zo>(Ci);
    });

    if constexpr (!Forward) {
        poet::static_for<0, NS>([&](const auto s) { Oi[s] = -Oi[s]; });
    }

    // -------------------------------------------------------------------------
    // 6. SoA → AoS: interleave re/im, store to out. Same aliasing as step 1.
    // -------------------------------------------------------------------------
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

// ============================================================================
// PFA catalog: one good_thomas_desc per routed size. This pack generates the
// eligibility predicate and the dispatch, so adding or removing a size is a one-line
// diff. base_cost_model.hpp owns per-cell precision routing.
// ============================================================================
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
    // admits<T> mirrors available(); ineligible kernels are never instantiated.
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

// Trampoline: one extern template per precision and direction keeps the whole
// PFA kernel tree out of every TU that only routes to it. Defined in
// inst_gt_f.cpp / inst_gt_d.cpp.
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
