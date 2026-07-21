#pragma once

// ============================================================================
// Vectorized PFA (Good-Thomas) DFT kernel, generic template.
// template<typename T, int N1, int N2, int N3>
//
// N = N1*N2*N3, pairwise coprime.  No inter-stage twiddle factors.
//
// CRT index maps (consteval-derived from factors):
//   in_coeff(i)  = N / Ni
//   out_coeff(i) = (N/Ni) * modinv(N/Ni, Ni) % N
//   in_idx (n1,n2,n3) = (n1*ic1 + n2*ic2 + n3*ic3) % N
//   out_idx(k1,k2,k3) = (k1*oc1 + k2*oc2 + k3*oc3) % N
//
// Generic over T (f32/f64) and W = xsimd::batch<T>::size.
// Eligibility predicate good_thomas_eligible<T,N1,N2,N3>() gates on:
//   - supported DFT factor set {2,3,4,5,8}
//   - register fit: 2*ceil(N/W) + 2*max(N1,N2,N3) + 4 <= vector_register_count()
//
// Butterfly set: DFT-2, DFT-3, DFT-4, DFT-5, DFT-8 (all lane-parallel batches).
//
// Gather: S source batches -> one W-wide result via a binary tree of S-1
// two-input shuffles (see GatherMasks / good_thomas_gather).
//
// Inverse via conjugate identity: IDFT(x) = conj(FWD(conj(x))).
// UN-normalized; caller applies 1/N for inverse.
// ============================================================================

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <xsimd/xsimd.hpp>
#include <poet/poet.hpp>

#include "butterfly.hpp"  // dif_butterfly (symmetric odd-radix + recursive split-radix pow2)
#include "macros.hpp"   // ADM_ALWAYS_INLINE (butterfly.hpp undefs it on the way out)

namespace admiral {
namespace detail {

// The PFA mask/batch arrays are sized by compile-time SIMD widths and factor
// products that are naturally `int` (W, NS, N3*BA, ...). std::array's extent is
// size_t, so convert once here rather than casting ~70 declarations. Signed
// index use into these arrays is made safe at the point of use via gt_at.
template<typename Elem, int Extent>
using gt_array = std::array<Elem, static_cast<std::size_t>(Extent)>;

// Always-false-but-dependent predicate for the unreachable else of an
// if-constexpr ladder (a plain `X != X` trips GCC 15's -Wtautological-compare).
template<auto> inline constexpr bool gt_dependent_false = false;

// ============================================================================
// Consteval CRT utilities
// ============================================================================

[[nodiscard]] consteval int good_thomas_gcd(int a, int b) noexcept {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

[[nodiscard]] consteval bool good_thomas_coprime(int a, int b) noexcept {
    return good_thomas_gcd(a, b) == 1;
}

// Extended Euclidean: returns x such that a*x ≡ 1 (mod m), 0 < a < m, gcd(a,m)=1.
[[nodiscard]] consteval int good_thomas_modinv(int a, int m) noexcept {
    int g = m, x = 0, y = 1;
    int ta = a;
    while (ta != 0) {
        int q  = g / ta;
        int tg = g - q*ta; g = ta; ta = tg;
        int tx = x - q*y;  x = y;  y = tx;
    }
    return (x % m + m) % m;
}

// Input CRT coefficient for axis i: coeff_in(i) = N/Ni.
template<int N1, int N2, int N3>
[[nodiscard]] consteval int good_thomas_in_coeff(int axis) noexcept {
    constexpr int N = N1*N2*N3;
    if (axis == 0) return N / N1;
    if (axis == 1) return N / N2;
    return N / N3;
}

// Output CRT coefficient for axis i: coeff_out(i) = (N/Ni) * modinv(N/Ni, Ni) % N.
template<int N1, int N2, int N3>
[[nodiscard]] consteval int good_thomas_out_coeff(int axis) noexcept {
    constexpr int N = N1*N2*N3;
    const int Ni   = (axis == 0) ? N1 : (axis == 1 ? N2 : N3);
    const int c    = N / Ni;
    return (c * good_thomas_modinv(c % Ni, Ni)) % N;
}

// Natural input index from cube coordinates (n1,n2,n3).
template<int N1, int N2, int N3>
[[nodiscard]] consteval int good_thomas_in_idx(int n1, int n2, int n3) noexcept {
    constexpr int N = N1*N2*N3;
    return (n1 * good_thomas_in_coeff<N1,N2,N3>(0) +
            n2 * good_thomas_in_coeff<N1,N2,N3>(1) +
            n3 * good_thomas_in_coeff<N1,N2,N3>(2)) % N;
}

// Natural output index from cube coordinates (k1,k2,k3).
template<int N1, int N2, int N3>
[[nodiscard]] consteval int good_thomas_out_idx(int k1, int k2, int k3) noexcept {
    constexpr int N = N1*N2*N3;
    return (k1 * good_thomas_out_coeff<N1,N2,N3>(0) +
            k2 * good_thomas_out_coeff<N1,N2,N3>(1) +
            k3 * good_thomas_out_coeff<N1,N2,N3>(2)) % N;
}

// ============================================================================
// Gather mask builder + applier (W-parametrized, structural GatherMasks for NTTP).
// make_batch_constant<Arr,Arch>() requires Arr.size()==W, so masks are gt_array<U,W>.
//
// Shuffle convention (xsimd::shuffle(a, b, mask)):
//   mask[i] in [0, W-1]   → a[mask[i]]
//   mask[i] in [W, 2W-1]  → b[mask[i] - W]
// ============================================================================

// Shuffle index type: uint32_t for f32 (32-bit lanes), uint64_t for f64 (64-bit lanes).
// xsimd::make_batch_constant<Arr, Arch>() requires Arr.size() == batch<U, Arch>::size.
template<typename T>
using good_thomas_mask_u = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;

// Subscript a std::array by a signed compile-time loop index (all provably
// non-negative and in-range here) without tripping -Wsign-conversion.
template<typename A>
constexpr auto& gt_at(A&& a, int i) noexcept { return a[static_cast<std::size_t>(i)]; }

// A gather of S source batches into one W-wide result is a binary tree of S-1
// two-input shuffles: xsimd::shuffle(a,b,mask) takes lane i from a[mask[i]] if
// mask[i]<W, else from b[mask[i]-W]. At every node the sources split
// L = bit_floor(S-1) left, R = S-L right; a lone source is a leaf passed
// straight to its parent shuffle, and a whole-transform S==1 is a single
// swizzle. One W-wide mask per internal node, stored in pre-order.
template<int S, int W, typename U>
struct GatherMasks {
    gt_array<gt_array<U, W>, (S > 1 ? S - 1 : 1)> node{};
};
template<int S, int W, typename U> using good_thomas_masks_t = GatherMasks<S, W, U>;

// Split rule, shared by the builder and the gather so their trees match.
constexpr int gt_split_left(int S) noexcept {
    return static_cast<int>(std::bit_floor(static_cast<unsigned>(S - 1)));
}

// One node's shuffle mask. Operand a = sources [lo,lo+L), operand b =
// [lo+L,lo+L+R). A raw single source (leaf) contributes lane slane; an
// already-gathered subtree holds its value at lane i. Don't-care lanes (sid
// outside this node) fold to the b side for a subtree-a node, to identity for
// the leaf pair — matching whatever overwrites them higher in the tree.
template<int W, typename U>
consteval gt_array<U, W> gt_combine_mask(gt_array<int, W> sid, gt_array<int, W> slane,
                                         int lo, int L, int R) noexcept {
    const bool a_leaf = (L == 1), b_leaf = (R == 1);
    gt_array<U, W> m{};
    for (int i = 0; i < W; ++i) {
        const int s = gt_at(sid, i), l = gt_at(slane, i);
        if (s >= lo && s < lo + L)              gt_at(m, i) = static_cast<U>(a_leaf ? l : i);
        else if (s >= lo + L && s < lo + L + R) gt_at(m, i) = static_cast<U>(W + (b_leaf ? l : i));
        else                                    gt_at(m, i) = static_cast<U>(a_leaf ? i : W + i);
    }
    return m;
}

// Pre-order fill of the node masks for sources [lo,lo+S), rooted at index nb.
// Left subtree occupies [nb+1,nb+L), right starts at nb+L.
template<int W, typename U>
consteval void gt_fill(auto& node,
                       gt_array<int, W> sid, gt_array<int, W> slane,
                       int lo, int S, int nb) noexcept {
    if (S < 2) return;
    const int L = gt_split_left(S), R = S - L;
    gt_at(node, nb) = gt_combine_mask<W, U>(sid, slane, lo, L, R);
    gt_fill<W, U>(node, sid, slane, lo, L, nb + 1);
    gt_fill<W, U>(node, sid, slane, lo + L, R, nb + L);
}

// gt_at(sid,i) in [0,S): source batch for lane i. gt_at(slane,i) in [0,W): lane
// within that source. Don't-care lanes: sid=0, slane=i (identity pass-through).
template<int S, int W, typename U>
consteval GatherMasks<S, W, U> good_thomas_make_masks(gt_array<int, W> sid,
                                                      gt_array<int, W> slane) noexcept {
    GatherMasks<S, W, U> m{};
    if constexpr (S == 1)   // single source: a swizzle by slane
        for (int i = 0; i < W; ++i) gt_at(gt_at(m.node, 0), i) = static_cast<U>(gt_at(slane, i));
    else
        gt_fill<W, U>(m.node, sid, slane, 0, S, 0);
    return m;
}

// ============================================================================
// Gather: apply the S-1 shuffle binary tree. Masks passed as NTTP M (a
// GatherMasks value); the Arch is deduced from Batch.
// ============================================================================
template<int lo, int S, int nb, auto M, typename Batch, std::size_t A>
[[nodiscard]] ADM_ALWAYS_INLINE Batch gt_gather_sub(const std::array<Batch, A>& src) noexcept {
    if constexpr (S == 1) {
        return src[static_cast<std::size_t>(lo)];   // leaf: raw source, parent shuffles it
    } else {
        using Arch = typename Batch::arch_type;
        constexpr int L = gt_split_left(S), R = S - L;
        return xsimd::shuffle(
            gt_gather_sub<lo, L, nb + 1, M>(src),
            gt_gather_sub<lo + L, R, nb + L, M>(src),
            xsimd::make_batch_constant<M.node[static_cast<std::size_t>(nb)], Arch>());
    }
}

// Dispatch gather by source count NumSrc, extracting from src[0..NumSrc).
// std::array<Batch,A> (NOT gt_array): gt_array's int->size_t static_cast makes
// its extent a non-deduced context, so A can't be deduced through the alias.
// (Regression fix b5afcff: swapping this to gt_array broke every W=16/W=8
// instantiation — the AVX-512 "good_thomas W=16 gap".)
template<int NumSrc, auto M, typename Batch, std::size_t A>
[[nodiscard]] ADM_ALWAYS_INLINE Batch good_thomas_gather(const std::array<Batch, A>& src) noexcept {
    static_assert(NumSrc >= 1 && NumSrc <= static_cast<int>(A), "gather: NumSrc out of range");
    if constexpr (NumSrc == 1) {
        using Arch = typename Batch::arch_type;
        return xsimd::swizzle(src[0], xsimd::make_batch_constant<M.node[0], Arch>());
    } else {
        return gt_gather_sub<0, NumSrc, 0, M>(src);
    }
}

// Apply the radix-Radix butterfly to arm slots {k*B+Z : k in [0,Radix)} of a
// SoA (re,im) register buffer. Shared by all three PFA stages (A/B/C): they
// differ only in radix, arm stride, and buffer, so the single gather → generic
// dif_butterfly → scatter lives here once instead of being copy-pasted per
// stage. Radix==1 is a no-op (identity axis). ADM_ALWAYS_INLINE — folds into
// the caller identically.
template<int Radix, int B, int Z, typename Batch, std::size_t S>
ADM_ALWAYS_INLINE void good_thomas_apply_dft(std::array<Batch, S>& xr,
                                             std::array<Batch, S>& xi) noexcept {
    if constexpr (Radix >= 2) {
        // Gather the Radix arm slots into a contiguous register buffer, run the
        // generic forward butterfly (butterfly.hpp: symmetric kernel for odd
        // radix, recursive split-radix for pow2 — same low-multiply arithmetic
        // as the old hand dft2/3/4/5/8, bench-validated there), scatter the
        // un-twiddled outputs back. PFA is forward-only (inverse via conjugation),
        // so Forward=true. always_inline folds the gather/scatter into renames.
        using T = typename Batch::value_type;
        Batch tr[Radix], ti[Radix];
        poet::static_for<Radix>([&](auto K) { tr[K] = xr[K*B+Z]; ti[K] = xi[K*B+Z]; });
        dif_butterfly<T, /*Forward=*/true, Radix>(tr, ti, [&](auto K, Batch yr, Batch yi) {
            xr[K*B+Z] = yr;
            xi[K*B+Z] = yi;
        });
    }
}

// ============================================================================
// good_thomas_eligible<T, N1, N2, N3>()
//
// Returns true iff:
//  1. All three factors are pairwise coprime.
//  2. All factors are in the supported butterfly set {2,3,4,5,8}.
//  3. Register fit: 2*ceil(N/W) + 2*max(N1,N2,N3) + 4 <= vector_register_count().
//     The +4 accounts for butterfly temporaries. This correctly excludes:
//       - AVX2 (16 regs) with W=8, N=60: 16 + 10 + 4 = 30 > 16
//       - AVX-512 (32 regs) with W=16, N=120 (8×3×5): 16 + 16 + 4 = 36 > 32
// ============================================================================
template<typename T, int N1, int N2, int N3>
[[nodiscard]] consteval bool good_thomas_eligible() noexcept {
    constexpr int N = N1*N2*N3;
    constexpr int W = static_cast<int>(xsimd::batch<T>::size);
    constexpr int S = (N + W - 1) / W;          // ceil(N/W)
    constexpr int Mx = (N1>N2 ? (N1>N3 ? N1 : N3) : (N2>N3 ? N2 : N3));  // max factor

    // (1) pairwise coprime
    if (!good_thomas_coprime(N1, N2) || !good_thomas_coprime(N1, N3) || !good_thomas_coprime(N2, N3))
        return false;

    // (2) factors in supported butterfly set (1 is a no-op identity stage)
    auto supported = [](int f) {
        return f==1 || f==2 || f==3 || f==4 || f==5 || f==8;
    };
    if (!supported(N1) || !supported(N2) || !supported(N3))
        return false;

    // (3) register fit, OR the small-N band measured to win on AVX2.
    // The static 2S+2Mx+4 estimate over-counts post-refactor peak live registers
    // (the butterfly emits via callback and reuses arm temps), so it false-excludes
    // {10,12,15,20,24,30}, which measurably beat codelet/dif on AVX2 (v3) by 6-52%.
    // W>=4 keeps the SSE-f64 (W=2) large-buffer instantiations out. The cost table
    // (base_cost_table.hpp) is the final routing authority; eligibility only decides
    // which kernels are instantiable and considered.
    constexpr std::size_t regs = poet::vector_register_count();
    if (static_cast<std::size_t>(2*S + 2*Mx + 4) <= regs) return true;
    return N <= 32 && W >= 4;
}

// ============================================================================
// Stage mask table generators (consteval functions returning std::array).
// ============================================================================


// Generic stage mask-table builder: ARMS arms × BATCHES batches, source = S ZMMs.
// For arm a, batch b, lane ml with position pos = b*W+ml < Limit, `map(pos, a)`
// yields {sid, slane} for that lane; out-of-range lanes stay don't-care (0,0).
// Table slot is a*BATCHES + b.
template<int S, int W, typename U, int ARMS, int BATCHES, int Limit, typename Map>
consteval gt_array<good_thomas_masks_t<S, W, U>, ARMS*BATCHES>
good_thomas_stage_masks_table(Map map) noexcept {
    gt_array<good_thomas_masks_t<S, W, U>, ARMS*BATCHES> tbl{};
    for (int a = 0; a < ARMS; ++a) {
        for (int b = 0; b < BATCHES; ++b) {
            gt_array<int, W> sid{}, slane{};
            for (int ml = 0; ml < W; ++ml) {
                const int pos = b*W + ml;
                if (pos < Limit) {
                    const auto [s, l] = map(pos, a);
                    gt_at(sid, ml)   = s;
                    gt_at(slane, ml) = l;
                }
            }
            gt_at(tbl, a*BATCHES + b) = good_thomas_make_masks<S, W, U>(sid, slane);
        }
    }
    return tbl;
}

// Stage A (DFT-N3 over n3): N3 arms × BA batches; source = S raw input ZMMs.
//   pos < N1*N2: cube(n1=pos/N2, n2=pos%N2, n3=j), nat = good_thomas_in_idx.
template<int N1, int N2, int N3, int W, int S, int BA, typename U>
consteval gt_array<good_thomas_masks_t<S, W, U>, N3*BA>
good_thomas_stage_a_masks_table() noexcept {
    return good_thomas_stage_masks_table<S, W, U, N3, BA, N1*N2>(
        [](int pos, int j) -> gt_array<int,2> {
            const int nat = good_thomas_in_idx<N1,N2,N3>(pos/N2, pos%N2, j);
            return {nat / W, nat % W};
        });
}

// Stage B (DFT-N2 over n2): N2 arms × BB batches; source = N3*BA Stage-A ZMMs.
//   pos < N1*N3: n1=pos/N3, k3=pos%N3; Stage-A arm=k3, lane_in_arm=n1*N2+jp;
//   global src batch = k3*BA + lane_in_arm/W.
template<int N1, int N2, int N3, int W, int BA, int BB, typename U>
consteval gt_array<good_thomas_masks_t<N3*BA, W, U>, N2*BB>
good_thomas_stage_b_masks_table() noexcept {
    return good_thomas_stage_masks_table<N3*BA, W, U, N2, BB, N1*N3>(
        [](int pos, int jp) -> gt_array<int,2> {
            const int n1 = pos / N3, k3 = pos % N3;
            const int lane_in_arm = n1*N2 + jp;
            return {k3*BA + lane_in_arm / W, lane_in_arm % W};
        });
}

// Stage C (DFT-N1 over n1): N1 arms × BC batches; source = N2*BB Stage-B ZMMs.
//   pos < N2*N3: k2=pos/N3, k3=pos%N3; Stage-B arm=k2, lane_in_arm=k1*N3+k3;
//   global src batch = k2*BB + lane_in_arm/W.
template<int N1, int N2, int N3, int W, int BB, int BC, typename U>
consteval gt_array<good_thomas_masks_t<N2*BB, W, U>, N1*BC>
good_thomas_stage_c_masks_table() noexcept {
    return good_thomas_stage_masks_table<N2*BB, W, U, N1, BC, N2*N3>(
        [](int pos, int k1) -> gt_array<int,2> {
            const int k2 = pos / N3, k3 = pos % N3;
            const int lane_in_arm = k1*N3 + k3;
            return {k2*BB + lane_in_arm / W, lane_in_arm % W};
        });
}

// Build output permutation mask table.
// Output (natural order): S_out = ceil(N/W) output ZMMs.
// Source: N1*BC Stage-C ZMMs indexed as (k1*BC + zc), k1∈[0,N1), zc∈[0,BC).
// Output ZMM zout, lane q: k = zout*W+q.
//   if k < N: k1=k%N1, k2=k%N2, k3=k%N3
//     Stage-C pos in arm = k2*N3+k3; src: arm=k1, zc=(k2*N3+k3)/W, lane=(k2*N3+k3)%W
//     Global src batch = k1*BC + zc.
template<int N1, int N2, int N3, int W, int BC, int S_out, typename U>
consteval gt_array<good_thomas_masks_t<N1*BC, W, U>, S_out>
good_thomas_output_masks_table() noexcept {
    constexpr int N    = N1*N2*N3;
    constexpr int Sout = N1*BC;
    gt_array<good_thomas_masks_t<Sout, W, U>, S_out> tbl{};
    for (int zout = 0; zout < S_out; ++zout) {
        gt_array<int, W> sid{}, slane{};
        for (int q = 0; q < W; ++q) {
            const int k = zout*W + q;
            if (k < N) {
                const int k1 = k % N1, k2 = k % N2, k3 = k % N3;
                const int pos_in_arm = k2*N3 + k3;
                gt_at(sid, q)   = k1*BC + pos_in_arm / W;
                gt_at(slane, q) = pos_in_arm % W;
            }
        }
        gt_at(tbl, zout) = good_thomas_make_masks<Sout, W, U>(sid, slane);
    }
    return tbl;
}

// ============================================================================
// good_thomas_execute<T, N1, N2, N3>(in, out, forward)
//
// DFT-N on AoS complex<T>[N].  N = N1*N2*N3, pairwise coprime.  in == out is
// in-place; in != out reads `in` (preserved) and writes `out`.
// forward=true: forward DFT.  forward=false: inverse (un-normalized).
// Inverse via conjugate identity: negate Im before and after the forward kernel.
//
// Aliasing is a compile-time parameter of the impl: the dispatch wrapper below
// branches once on in == out, so each instantiation carries a single entry/exit
// strategy and the unaliased one gets ADM_RESTRICT codegen throughout.
// ============================================================================
template<typename T, int N1, int N2, int N3, bool Aliased, bool Forward>
ADM_NOINLINE void good_thomas_execute_impl(const std::complex<T>* in,
                                           std::complex<T>* out) noexcept {
    static_assert(good_thomas_coprime(N1,N2) && good_thomas_coprime(N1,N3) && good_thomas_coprime(N2,N3),
                  "good_thomas_execute: factors must be pairwise coprime");

    using Batch = xsimd::batch<T>;
    using Arch  = typename Batch::arch_type;
    static constexpr int N  = N1*N2*N3;
    static constexpr int W  = static_cast<int>(Batch::size);
    static constexpr int NS = (N + W - 1) / W;   // ceil(N/W): source ZMMs
    static constexpr int BA = (N1*N2 + W - 1) / W;
    static constexpr int BB = (N1*N3 + W - 1) / W;
    static constexpr int BC = (N2*N3 + W - 1) / W;
    using U = good_thomas_mask_u<T>;

    // Mask tables (static constexpr — evaluated once per instantiation).
    // [[maybe_unused]]: degenerate factors (e.g. N2==1) collapse a stage so its
    // table is read only in a pruned constexpr index — no runtime use.
    [[maybe_unused]] static constexpr auto SA_tbl =
        good_thomas_stage_a_masks_table<N1, N2, N3, W, NS, BA, U>();
    [[maybe_unused]] static constexpr auto SB_tbl =
        good_thomas_stage_b_masks_table<N1, N2, N3, W, BA, BB, U>();
    [[maybe_unused]] static constexpr auto SC_tbl =
        good_thomas_stage_c_masks_table<N1, N2, N3, W, BB, BC, U>();
    [[maybe_unused]] static constexpr auto OUT_tbl =
        good_thomas_output_masks_table<N1, N2, N3, W, BC, NS, U>();

    // -------------------------------------------------------------------------
    // 1. AoS → SoA: load N complex<T> → NS ZMM pairs (re[s], im[s]).
    //    Entry/exit strategy follows the Aliased template parameter:
    //    - Aliased (in == out): stage through an aligned pad via memcpy.
    //      Back-to-back in-place transforms chain this call's exit stores into
    //      the next call's entry loads; store-to-load forwarding only works
    //      when the store decomposition exactly matches the load decomposition
    //      with no partial overlap — which the inlined memcpys on both sides
    //      provide. Masked-tail and overlapped-tail variants both stall 7-14%.
    //    - !Aliased: no store→load chain — load/store directly (unaligned,
    //      ADM_RESTRICT), the one W-op crossing scalar 2N compile-time masked.
    //      Skipping the staging copies wins 2-8%.
    // -------------------------------------------------------------------------
    gt_array<Batch, NS> re_src, im_src;
    struct even_t { static constexpr uint32_t get(uint32_t i, uint32_t) noexcept { return 2*i; } };
    struct odd_t  { static constexpr uint32_t get(uint32_t i, uint32_t) noexcept { return 2*i+1; } };
    if constexpr (Aliased) {
        alignas(xsimd::batch<T>::arch_type::alignment()) T ipad[static_cast<std::size_t>(NS) * W * 2];
        std::memcpy(ipad, in, N * sizeof(std::complex<T>));
        poet::static_for<0, NS>([&](auto s_ic) {
            constexpr int s = static_cast<int>(s_ic.value);
            const T* p = ipad + s * W * 2;
            const Batch lo = Batch::load_aligned(p);
            const Batch hi = Batch::load_aligned(p + W);
            re_src[s] = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<U, even_t, Arch>());
            im_src[s] = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<U, odd_t,  Arch>());
        });
    } else {
        const T* ADM_RESTRICT ip = reinterpret_cast<const T*>(in);
        auto load_tail = [&]<int O>(std::integral_constant<int, O>) {
            if constexpr (O + W <= 2 * N) {
                return Batch::load_unaligned(ip + O);
            } else if constexpr (O >= 2 * N) {
                return Batch(T(0));
            } else {
                struct in_bounds {
                    static constexpr bool get(std::size_t i, std::size_t) noexcept { return O + static_cast<int>(i) < 2 * N; }
                };
                constexpr auto m = xsimd::make_batch_bool_constant<T, in_bounds, Arch>();
                return Batch::load(ip + O, m, xsimd::unaligned_mode{});
            }
        };
        poet::static_for<0, NS>([&](auto s_ic) {
            constexpr int s = static_cast<int>(s_ic.value);
            const Batch lo = load_tail(std::integral_constant<int, s * W * 2>{});
            const Batch hi = load_tail(std::integral_constant<int, s * W * 2 + W>{});
            re_src[s] = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<U, even_t, Arch>());
            im_src[s] = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<U, odd_t,  Arch>());
        });
    }

    if constexpr (!Forward) {
        poet::static_for<0, NS>([&](auto s_ic) { im_src[s_ic.value] = -im_src[s_ic.value]; });
    }

    // -------------------------------------------------------------------------
    // 2. Stage A: DFT-N3 over n3 axis.  N3 arms × BA ZMMs each.
    // -------------------------------------------------------------------------
    gt_array<Batch, N3*BA> Ar, Ai;
    poet::static_for<0, N3*BA>([&](auto jz_ic) {
        constexpr int JZ = static_cast<int>(jz_ic.value);
        constexpr auto M = SA_tbl[JZ];
        Ar[JZ] = good_thomas_gather<NS, M>(re_src);
        Ai[JZ] = good_thomas_gather<NS, M>(im_src);
    });
    poet::static_for<0, BA>([&](auto z_ic) {
        constexpr int Z = static_cast<int>(z_ic.value);
        good_thomas_apply_dft<N3, BA, Z>(Ar, Ai);
    });

    // -------------------------------------------------------------------------
    // 3. Stage B: DFT-N2 over n2 axis.  N2 arms × BB ZMMs each.
    // -------------------------------------------------------------------------
    gt_array<Batch, N2*BB> Br, Bi;
    poet::static_for<0, N2*BB>([&](auto jpz_ic) {
        constexpr int JPZ = static_cast<int>(jpz_ic.value);
        constexpr auto M = SB_tbl[JPZ];
        Br[JPZ] = good_thomas_gather<N3*BA, M>(Ar);
        Bi[JPZ] = good_thomas_gather<N3*BA, M>(Ai);
    });
    poet::static_for<0, BB>([&](auto z_ic) {
        constexpr int Z = static_cast<int>(z_ic.value);
        good_thomas_apply_dft<N2, BB, Z>(Br, Bi);
    });

    // -------------------------------------------------------------------------
    // 4. Stage C: DFT-N1 over n1 axis.  N1 arms × BC ZMMs each.
    // -------------------------------------------------------------------------
    gt_array<Batch, N1*BC> Cr, Ci;
    poet::static_for<0, N1*BC>([&](auto k1z_ic) {
        constexpr int K1Z = static_cast<int>(k1z_ic.value);
        constexpr auto M = SC_tbl[K1Z];
        Cr[K1Z] = good_thomas_gather<N2*BB, M>(Br);
        Ci[K1Z] = good_thomas_gather<N2*BB, M>(Bi);
    });
    poet::static_for<0, BC>([&](auto z_ic) {
        constexpr int Z = static_cast<int>(z_ic.value);
        good_thomas_apply_dft<N1, BC, Z>(Cr, Ci);
    });

    // -------------------------------------------------------------------------
    // 5. Output gather: CRT permutation, N1*BC Stage-C ZMMs → NS natural-order.
    // -------------------------------------------------------------------------
    gt_array<Batch, NS> Or, Oi;
    poet::static_for<0, NS>([&](auto zo_ic) {
        constexpr int ZO = static_cast<int>(zo_ic.value);
        constexpr auto M = OUT_tbl[ZO];
        Or[ZO] = good_thomas_gather<N1*BC, M>(Cr);
        Oi[ZO] = good_thomas_gather<N1*BC, M>(Ci);
    });

    if constexpr (!Forward) {
        poet::static_for<0, NS>([&](auto s_ic) { Oi[s_ic.value] = -Oi[s_ic.value]; });
    }

    // -------------------------------------------------------------------------
    // 6. SoA → AoS: interleave re/im, store to `out`. Same aliasing dispatch
    //    as the entry (see the note there): staged pad + memcpy when in-place,
    //    direct unaligned stores with a compile-time-masked tail otherwise.
    // -------------------------------------------------------------------------
    if constexpr (Aliased) {
        alignas(xsimd::batch<T>::arch_type::alignment()) T opad[static_cast<std::size_t>(NS) * W * 2];
        poet::static_for<0, NS>([&](auto s_ic) {
            constexpr int s = static_cast<int>(s_ic.value);
            T* dst = opad + s * W * 2;
            xsimd::zip_lo(Or[s], Oi[s]).store_aligned(dst);
            xsimd::zip_hi(Or[s], Oi[s]).store_aligned(dst + W);
        });
        std::memcpy(out, opad, N * sizeof(std::complex<T>));
    } else {
        T* ADM_RESTRICT op = reinterpret_cast<T*>(out);
        auto store_tail = [&]<int O>(std::integral_constant<int, O>, const Batch& v) {
            if constexpr (O + W <= 2 * N) {
                v.store_unaligned(op + O);
            } else if constexpr (O < 2 * N) {
                struct in_bounds {
                    static constexpr bool get(std::size_t i, std::size_t) noexcept { return O + static_cast<int>(i) < 2 * N; }
                };
                constexpr auto m = xsimd::make_batch_bool_constant<T, in_bounds, Arch>();
                v.store(op + O, m, xsimd::unaligned_mode{});
            }  // O >= 2N: whole batch past the end — nothing to store
        };
        poet::static_for<0, NS>([&](auto s_ic) {
            constexpr int s = static_cast<int>(s_ic.value);
            store_tail(std::integral_constant<int, s * W * 2>{}, xsimd::zip_lo(Or[s], Oi[s]));
            store_tail(std::integral_constant<int, s * W * 2 + W>{}, xsimd::zip_hi(Or[s], Oi[s]));
        });
    }
}

// Aliasing dispatch: one branch, two specialized instantiations. OOP is the
// expected hot case (copy-free plan routes); in-place is the legacy span API.
template<typename T, int N1, int N2, int N3, bool Forward>
ADM_ALWAYS_INLINE void good_thomas_execute(const std::complex<T>* in,
                                           std::complex<T>* out) noexcept {
    if (static_cast<const void*>(in) == static_cast<const void*>(out)) [[unlikely]] {
        good_thomas_execute_impl<T, N1, N2, N3, /*Aliased=*/true, Forward>(in, out);
    } else {
        good_thomas_execute_impl<T, N1, N2, N3, /*Aliased=*/false, Forward>(in, out);
    }
}

// ============================================================================
// Routed PFA catalog — the single source of truth for which (N, factors,
// precision) cells route to good_thomas. One good_thomas_desc row per routed size; both the
// eligibility predicate and the execute dispatch are generated from the pack,
// so adding/removing a cell is a one-line diff here.
//
// Per-cell precision pruning is delegated to the generated cost table
// (base_cost_table.hpp): a cell routes good_thomas only where it measured cheapest
// for that (ISA, precision, N), so no per-desc precision mask is needed here.
// ============================================================================
template<int N1, int N2, int N3>
struct good_thomas_desc {
    static constexpr std::size_t n =
        std::size_t(N1) * std::size_t(N2) * std::size_t(N3);
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
    // Runs the matching kernel; the admits<T> constexpr gate mirrors
    // available() exactly, so ineligible kernels are never instantiated.
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
    good_thomas_desc<3, 4, 5>>;        // 60

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
