#pragma once

// Flat interleaved-lane row codelet (ADM_ND_FLATROW): one whole row of N complex stays packed
// re/im in K = 2N/W registers, the DIF stages run between whole registers while the partner
// distance is >= one register, and the remaining intra-register stages are lifted onto a
// register-major layout by ONE small lane permutation per group, where every twiddle factor
// broadcast-folds and the output permutation resolves to whole-register store relabeling.
// The split-plane batched arms price one lane transposition per row batch instead: this arm
// trades a handful of in-flight twiddle registers for the entire gather/scatter transpose
// class and for the flat_leaf spill boundary (2N planes), at the cost of log2(C) lift
// shuffles per row. That is the inverse trade of the rejected packed prototype (pk_dif_run):
// its intra stages paid one shuffle per butterfly through XOR-lane exchanges plus selects and
// a separate bit-reversal tree, so its shuffle count went UP; here every stage below the
// register boundary is exactly the two-source lane merge the lift was built for.

#include <array>
#include <cstddef>

#include <poet/poet.hpp>

#include "cxx_compat.hpp"
#include "ct_math.hpp"
#include "simd.hpp"
#include "simd_swizzle.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

// Admission: powers of two covering whole batches with the lift group intact, sized by the
// register model. Live set is K = 2N/W data registers plus a working set of four (twiddle
// operands fold from memory as FMA sources; gcc needs two table loads and two temporaries in
// flight), priced against the same 3/4-file budget usable_vector_regs gives flat_leaf. The
// K <= 8 clause is kernel shape, not register price: the intra-store maps are derived for
// K in {4, 8} (see the static_asserts), and it is what keeps 128-bit ISAs with 32+ vector
// registers (arm_neon, ppc_vsx/altivec, mips_msa) from admitting f32 N = 32 at K = 16, a
// shape the register budget alone would wave through and the asserts would then hard-stop.
template<unsigned N, typename V>
inline constexpr bool kFlatRow =
    (N & (N - 1u)) == 0u && N >= 16u && N <= 32u && V::size >= 4 &&
    2u * N / V::size >= V::size / 2u && 2u * N / V::size <= 8u &&
    2u * N / V::size + 4u <= usable_vector_regs(poet::vector_register_count());

// Swap re/im inside every complex element.
struct fr_swap_pair {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return i ^ 1; }
};

// 128-bit half swap, the C == 2 intra butterfly partner exchange (W == 4 only).
struct fr_half_swap {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return i ^ 2; }
};

// Two-source complex-quarter merges for the lift at C == 4. In complex-slot units of W/2
// lanes, lo reads sources [A0|B0|A1|B1|...], hi the same two slots up.
template<std::size_t W>
struct fr_merge_lo {
    static constexpr std::size_t get(std::size_t i, std::size_t) {
        const std::size_t p = i / 2;
        return ((p & 1) != 0 ? W : 0) + 2 * (p / 2) + (i % 2);
    }
};
template<std::size_t W>
struct fr_merge_hi {
    static constexpr std::size_t get(std::size_t i, std::size_t) {
        const std::size_t p = i / 2;
        return ((p & 1) != 0 ? W : 0) + 2 * (p / 2 + 2) + (i % 2);
    }
};

// Stage twiddle for DIF sub-length 2M, diff-register I: complex lanes j = I*C + c get
// w_{2M}^j. The AoS multiply needs u = [wr, wr] per pair and v = [-wi, +wi].
template<unsigned M, unsigned I, bool Conj, typename T, std::size_t W>
[[nodiscard]] ADM_CONSTEVAL std::array<T, W> fr_tw(bool swapped) {
    constexpr std::size_t C = W / 2;
    std::array<T, W> a{};
    for (std::size_t c = 0; c < C; ++c) {
        const ct_sincos_t w = ct_sincos_turns(Conj, I * C + c, 2 * M);
        a[2 * c] = swapped ? -static_cast<T>(w.s) : static_cast<T>(w.c);
        a[2 * c + 1] = swapped ? static_cast<T>(w.s) : static_cast<T>(w.c);
    }
    return a;
}

// Broadcast twiddle w_Den^Num: every complex lane of the lifted cell shares one factor.
template<unsigned Num, unsigned Den, bool Conj, typename T, std::size_t W>
[[nodiscard]] ADM_CONSTEVAL std::array<T, W> fr_splat(bool swapped) {
    constexpr std::size_t C = W / 2;
    const ct_sincos_t w = ct_sincos_turns(Conj, Num, Den);
    std::array<T, W> a{};
    for (std::size_t c = 0; c < C; ++c) {
        a[2 * c] = swapped ? -static_cast<T>(w.s) : static_cast<T>(w.c);
        a[2 * c + 1] = swapped ? static_cast<T>(w.s) : static_cast<T>(w.c);
    }
    return a;
}

// out = d * u + swap_reim(d) * v, the AoS complex product against the table pair.
template<typename V>
[[nodiscard]] ADM_ALWAYS_INLINE V fr_cmul(V d, V u, V v) {
    using index = xsimd::as_unsigned_integer_t<typename V::value_type>;
    using arch = typename V::arch_type;
    const V sw = xsimd::swizzle(d, xsimd::make_batch_constant<index, fr_swap_pair, arch>());
    return piece_fma(sw, v, d * u);
}

// Inter-register DIF stages: while the partner distance M >= C complexes, butterfly partners
// sit M/C registers apart and every twiddle is a per-register AoS product.
template<unsigned N, unsigned M, bool Forward, typename T, typename V>
ADM_ALWAYS_INLINE void fr_stages(V* d) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t C = W / 2;
    constexpr std::size_t K = 2 * N / W;
    if constexpr (M >= C) {
        constexpr std::size_t S = M / C;
        poet::static_for<0, K / (2 * S)>([&](auto G) {
            poet::static_for<0, S>([&](auto I) {
                constexpr std::size_t a = G * 2 * S + I;
                static constexpr auto tu = fr_tw<M, I, Forward, T, W>(false);
                static constexpr auto tv = fr_tw<M, I, Forward, T, W>(true);
                const V x = d[a], y = d[a + S];
                d[a] = x + y;
                d[a + S] = fr_cmul(x - y, V::load_unaligned(tu.data()),
                                   V::load_unaligned(tv.data()));
            });
        });
        fr_stages<N, M / 2, Forward, T, V>(d);
    }
}

[[nodiscard]] ADM_CONSTEVAL std::size_t fr_rev2(std::size_t l) {
    return (l & 1u) * 2u + (l >> 1u);
}

// d<D>-point DIF over one register per element (post-lift), broadcast twiddles, natural
// output order in the register index up to the caller's store relabeling.
template<unsigned D, bool Forward, typename T, typename V>
ADM_ALWAYS_INLINE void fr_dif_regs(V* t) {
    if constexpr (D > 1) {
        constexpr unsigned M = D / 2;
        poet::static_for<0, M>([&](auto J) {
            constexpr unsigned j = J;
            const V x = t[j], y = t[j + M];
            t[j] = x + y;
            if constexpr (j == 0) {
                t[j + M] = x - y;
            } else {
                constexpr std::size_t W = V::size;
                static constexpr auto tu = fr_splat<J, D, Forward, T, W>(false);
                static constexpr auto tv = fr_splat<J, D, Forward, T, W>(true);
                t[j + M] = fr_cmul(x - y, V::load_unaligned(tu.data()),
                                   V::load_unaligned(tv.data()));
            }
        });
        fr_dif_regs<M, Forward, T, V>(t);
        fr_dif_regs<M, Forward, T, V>(t + M);
    }
}

// One whole row, packed re/im: 2N/W contiguous loads, the network, N complex back. The
// stores are relabeled, never shuffled: slot(G*rev2(l) + g) receives lifted cell (g, l), and
// at C == 2 the two output classes split by bit of rev_{log2 N}(2s) share one merge shape.
template<unsigned N, typename T, typename V, bool Forward>
ADM_ALWAYS_INLINE void flat_row_apply(const T* ADM_RESTRICT srcp, T* ADM_RESTRICT dstp, V f) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t C = W / 2;
    constexpr std::size_t K = 2 * N / W;
    using index = xsimd::as_unsigned_integer_t<T>;
    using arch = typename V::arch_type;

    V d[K];
    poet::static_for<0, K>([&](auto k) { d[k] = V::load_unaligned(srcp + k * W); });
    fr_stages<N, N / 2, Forward, T, V>(d);

    if constexpr (C == 2) {
        static_assert(K == 8, "the C == 2 ordering below is derived for K == 8");
        poet::static_for<0, K / 2>([&](auto a) {
            const V xa = d[a], xb = d[a + K / 2];
            const V sa =
                xsimd::swizzle(xa, xsimd::make_batch_constant<index, fr_half_swap, arch>());
            const V sb =
                xsimd::swizzle(xb, xsimd::make_batch_constant<index, fr_half_swap, arch>());
            const V aa = xa + sa, da = xa - sa;   // both lanes X0 ; X1 in lane 0
            const V ab = xb + sb, db = xb - sb;
            const V o0 = xsimd::shuffle(
                aa, ab, xsimd::make_batch_constant<index, fr_merge_lo<W>, arch>());
            const V o1 = xsimd::shuffle(
                da, db, xsimd::make_batch_constant<index, fr_merge_lo<W>, arch>());
            constexpr std::size_t s0 = fr_rev2(a);
            (o0 * f).store_unaligned(dstp + s0 * W);
            (o1 * f).store_unaligned(dstp + (s0 + K / 2) * W);
        });
    } else {
        static_assert(C == 4 && (K == 4 || K == 8), "lift shape derived for C == 4");
        constexpr std::size_t G = K / 4;
        poet::static_for<0, G>([&](auto g) {
            // The two-level lo/hi merge net permutes lanes by [0,2,1,3] on its own
            // (t0 = [qa.c0|qc.c0|qb.c0|qd.c0]), so sourcing in natural G-stride order is what
            // lands the rev2-folded lifted cell: T_l lane c = sub-dft G*rev2(c) + g.
            const V qa = d[g], qb = d[G + g], qc = d[2 * G + g], qd = d[3 * G + g];
            const V A = xsimd::shuffle(qa, qb, xsimd::make_batch_constant<index, fr_merge_lo<W>, arch>());
            const V B = xsimd::shuffle(qc, qd, xsimd::make_batch_constant<index, fr_merge_lo<W>, arch>());
            const V Ap = xsimd::shuffle(qa, qb, xsimd::make_batch_constant<index, fr_merge_hi<W>, arch>());
            const V Bp = xsimd::shuffle(qc, qd, xsimd::make_batch_constant<index, fr_merge_hi<W>, arch>());
            V t[4];
            t[0] = xsimd::shuffle(A, B, xsimd::make_batch_constant<index, fr_merge_lo<W>, arch>());
            t[1] = xsimd::shuffle(A, B, xsimd::make_batch_constant<index, fr_merge_hi<W>, arch>());
            t[2] = xsimd::shuffle(Ap, Bp, xsimd::make_batch_constant<index, fr_merge_lo<W>, arch>());
            t[3] = xsimd::shuffle(Ap, Bp, xsimd::make_batch_constant<index, fr_merge_hi<W>, arch>());
            fr_dif_regs<4, Forward, T, V>(t);
            poet::static_for<0, 4>([&](auto l) {
                (t[fr_rev2(l)] * f).store_unaligned(dstp + (G * l + g) * W);
            });
        });
    }
}

}
}

#include "undef_macros.hpp"
