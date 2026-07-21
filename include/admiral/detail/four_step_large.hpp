#pragma once

// ============================================================================
// LARGE-N four-step (Bailey) route. The flat iterative_dif pass chain streams
// the WHOLE array to DRAM once per radix pass — ~log_radix(N) times — so at
// N beyond L2 it becomes DRAM-bandwidth-bound.
//
// The four-step factors N = N1*N2 with BOTH leaves cache-resident, so every leaf
// transform runs while its working set is in cache and the array is streamed only
// a handful of times. Same DIT algebra as four_step.hpp, but the leaves are FULL
// sub-FFTs (iterative_dif / col driver), not <=64 catalog codelets, so N1,N2 can
// be ~sqrt(N) (e.g. 4M = 2048*2048).
//
// DIT index map (n = n2*N1 + n1, k = k1*N2 + k2):
//   view x as row-major [N2][N1] (n1 contiguous):  x[n2*N1 + n1]
//   inner : for each n1, size-N2 DFT over n2 (strided, stride N1, batched over
//           the N1 contiguous columns) via col_dif_execute_ws
//             -> G0[k2*N1 + n1] = sum_{n2} x[n2*N1+n1] W_N2^{n2 k2}
//   twist : G[k2*N1+n1] = G0[k2*N1+n1] * W_N^{n1 k2}   (fused into the row below)
//   outer : for each k2, size-N1 DFT over the contiguous row  (iterative_dif)
//             -> X'[k1] = sum_{n1} G[k2*N1+n1] W_N1^{n1 k1} = X[k1*N2 + k2]
//   store : scatter X'[k1] to out[k1*N2 + k2]  (stride N2 -> the transpose)
//
// Reuses the ND row-column machinery (col_dif_execute_ws cache tiling + the
// iterative_dif row kernel), so no new butterfly/kernel code. Un-normalized
// forward; inverse folds 1/N2 into the col pass and 1/N1 into the row pass.
// ============================================================================

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

#include "dif_col_driver.hpp"  // col_dif_execute_ws
#include "dif_driver.hpp"      // iterative_dif_execute_ws
#include "math.hpp"            // is_codelet_supported
#include "portable_trig.hpp"   // sincos_turns
#include "scratch.hpp"         // soa_scratch
#include "simd_swizzle.hpp"    // aos_deinterleave, aos_interleave
#include "twiddles.hpp"        // dif_twiddle_set, build_dif_twiddle_set

namespace admiral {
namespace detail {

// Out-of-place complex transpose out[j*n2 + i] = W[i*n1 + j]. The naive strided
// scalar scatter writes one 16 B complex per 64 B line (partial-line RFO stores);
// the blocked WxW register transpose below turns each into W contiguous
// full-vector stores. Deinterleave W source rows into planar re/im, transpose
// both planes in registers (xsimd::transpose, same primitive as the DIF subgroup
// transpose in dif_passes.hpp), interleave back as W output rows. Width-parametric
// (W = native batch width); scalar remainder for the W-ragged right/bottom edges.
// Cache-blocked at B so a source tile stays L1-resident across the W output-row
// scatter.
constexpr std::size_t four_step_tblock = 32;  // transpose cache-block (complex)

// Vector WxW register transpose of the source ROW BAND [i_lo,i_hi) (both must be
// W-aligned) x all full cols [0, n1v), j-blocked at B for source-tile L1
// residency. out[j*n2+i] = Wm[i*n1+j] over that band. The band is transposed
// right after col_dif+row-FFT computes those rows (fused into execute's row
// loop), so `src` is read while still hot in L2 — killing the DRAM re-read of W
// that a separate full-array transpose pass suffered (W is 64 MB > L3 at 4M).
template<typename T>
void four_step_transpose_band(const std::complex<T>* Wm, std::complex<T>* out,
                              std::size_t n1, std::size_t n2,
                              std::size_t i_lo, std::size_t i_hi) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t B = four_step_tblock;
    static_assert(B % W == 0);
    const T* src = reinterpret_cast<const T*>(Wm);
    T* dst = reinterpret_cast<T*>(out);
    const std::size_t n1v = n1 & ~(W - 1);
    for (std::size_t j0 = 0; j0 < n1v; j0 += B) {
        const std::size_t jE = std::min(j0 + B, n1v);
        for (std::size_t i = i_lo; i < i_hi; i += W)
            for (std::size_t j = j0; j < jE; j += W) {
                batch re[W], im[W];
                for (std::size_t r = 0; r < W; ++r)
                    aos_deinterleave<T>(src + ((i + r) * n1 + j) * 2, re[r], im[r]);
                xsimd::transpose(re, re + W);   // re[c] lane r = Re W[i+r][j+c]
                xsimd::transpose(im, im + W);
                for (std::size_t c = 0; c < W; ++c)
                    aos_interleave<T>(dst + ((j + c) * n2 + i) * 2, re[c], im[c]);
            }
    }
}

// W-ragged remainder scatter (non-overlapping): right cols [n1v,n1) x all rows,
// then bottom rows [n2v,n2) x left cols [0,n1v). Done once after all bands.
template<typename T>
void four_step_transpose_remainder(const std::complex<T>* Wm, std::complex<T>* out,
                                   std::size_t n1, std::size_t n2) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t n2v = n2 & ~(W - 1);
    const std::size_t n1v = n1 & ~(W - 1);
    for (std::size_t i = 0; i < n2; ++i)
        for (std::size_t j = n1v; j < n1; ++j) out[j * n2 + i] = Wm[i * n1 + j];
    for (std::size_t i = n2v; i < n2; ++i)
        for (std::size_t j = 0; j < n1v; ++j) out[j * n2 + i] = Wm[i * n1 + j];
}

struct large_split {
    std::size_t n1 = 0;
    std::size_t n2 = 0;
    [[nodiscard]] constexpr bool valid() const { return n1 != 0 && n2 != 0; }
};

// A factor is a usable leaf iff a 1D iterative_dif can transform it: pow2 or any
// codelet-supported smooth length. (build_dif_twiddle_set requires this.)
[[nodiscard]] constexpr bool large_leaf_ok(std::size_t f) {
    return f > 1 && ((f & (f - 1)) == 0 || is_codelet_supported(f));
}

// Balanced split N = n1*n2, both leaves usable, as close to sqrt(N) as possible
// (a balanced split minimises max(leaf footprint) so both stay cache-resident).
// Returns {0,0} if no such split exists (e.g. a large prime -> caller keeps its
// existing route).
[[nodiscard]] constexpr large_split choose_large_split(std::size_t N) {
    large_split best{};
    std::size_t best_bal = 0;
    for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
        if (N % n1 != 0) continue;
        const std::size_t n2 = N / n1;
        if (!large_leaf_ok(n1) || !large_leaf_ok(n2)) continue;
        // n1 <= n2 by construction; the largest n1 is the most balanced.
        if (n1 > best_bal) { best_bal = n1; best = {n1, n2}; }
    }
    return best;
}

// Route gate: only for N whose flat pass chain is DRAM-bound. The crossover vs
// iterative_dif's transpose-free passes is measured; the caller passes the
// threshold (bytes). Requires a balanced usable split.
[[nodiscard]] constexpr bool four_step_large_supported(std::size_t N, std::size_t elem_bytes,
                                                       std::size_t threshold_bytes) {
    return N * elem_bytes > threshold_bytes && choose_large_split(N).valid();
}

// Plan-owned state: the two leaf twiddle sets, the four-step twist table, and a
// length-N working buffer. Mutable scratch => one plan instance is single-use at
// a time (same discipline as bluestein/rader plan-owned scratch).
template<typename T>
struct four_step_large_plan {
    std::size_t n1 = 0, n2 = 0;
    bool forward = true;
    dif_twiddle_set<T> dtw_n2;                 // col-form (strided inner pass)
    dif_twiddle_set<T> dtw_n1;                 // row (contiguous outer pass)
    // Split (two-table) twist: W_N^{q} = W_N^{qhi*M + qlo} = hitab[qhi]*lotab[qlo],
    // q = n1*k2 accumulated as an EXACT integer, M = 2^ceil(log2 sqrt(N)) a power of
    // two >= sqrt(N). One complex mul per element, machine precision (~3e-16, no
    // float drift), O(sqrt(N)) memory (128 KB even at 16M).
    std::vector<std::complex<T>> lotab;        // W_N^{j},   j in [0,M)
    std::vector<std::complex<T>> hitab;        // W_N^{h*M}, h in [0, (N-1)>>logM]
    std::size_t twist_M = 1, twist_logM = 0;
    // NB: no plan-owned working buffer. execute() is RE-ENTRANT (allocates its own
    // scratch call-locally) because the ND row driver shares one plan_impl across
    // worker threads and calls execute() concurrently on distinct rows
    // (nd_plan.hpp:150) — a mutable member would race there.

    four_step_large_plan(std::size_t N, bool fwd) : forward(fwd) {
        const large_split sp = choose_large_split(N);
        n1 = sp.n1;
        n2 = sp.n2;
        // Col form (fuse_packed=false) for the strided inner pass; plain row form
        // for the contiguous outer pass.
        dtw_n2 = fwd ? build_dif_twiddle_set<T, true>(n2, nullptr, /*fuse_packed=*/false)
                     : build_dif_twiddle_set<T, false>(n2, nullptr, /*fuse_packed=*/false);
        dtw_n1 = fwd ? build_dif_twiddle_set<T, true>(n1, nullptr)
                     : build_dif_twiddle_set<T, false>(n1, nullptr);
        // Split-twiddle tables for the four-step twist W_N^{n1 k2}. Splitting the
        // exponent q = n1*k2 = qhi*M + qlo into a high/low pair and taking one product
        // hitab[qhi]*lotab[qlo] keeps machine precision (each table entry is an exact
        // sincos, so error is one rounding, ~3e-16 — vs the ~5e-14 drift of an
        // incremental rotation) at O(sqrt(N)) memory. M is a power of two >= sqrt(N)
        // so qhi=q>>logM, qlo=q&(M-1) are a shift and a mask.
        twist_M = std::bit_ceil(static_cast<std::size_t>(std::sqrt(static_cast<double>(N))) + 1);
        twist_logM = static_cast<std::size_t>(std::countr_zero(twist_M));
        lotab.resize(twist_M);
        for (std::size_t j = 0; j < twist_M; ++j) {
            const auto [sn, cs] = fwd ? portable_trig::sincos_turns<true>(j, N)
                                      : portable_trig::sincos_turns<false>(j, N);
            lotab[j] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
        const std::size_t nhi = ((N - 1) >> twist_logM) + 1;
        hitab.resize(nhi);
        for (std::size_t h = 0; h < nhi; ++h) {
            const auto [sn, cs] = fwd ? portable_trig::sincos_turns<true>(h * twist_M, N)
                                      : portable_trig::sincos_turns<false>(h * twist_M, N);
            hitab[h] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
    }

    // in == out (in-place) or in != out; result written to out and scaled by
    // `fct`. The direction default (forward = 1, inverse = 1/N) keeps the tuned
    // split scale (col 1/n2, row 1/n1) byte-identical; any other fct is folded
    // entirely into the final row pass (col unscaled).
    void execute(const std::complex<T>* in, std::complex<T>* out, T fct) const {
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == (forward ? T(1) : T(1) / static_cast<T>(N)));
        // Per-call working buffer (no shared state: the shared-plan ND row driver
        // calls this concurrently across worker threads). make_unique_for_overwrite:
        // the col pass below overwrites all N elements before any read.
        auto Wbuf = std::make_unique_for_overwrite<std::complex<T>[]>(N);
        std::complex<T>* W = Wbuf.get();

        // --- inner: size-N2 strided DFTs, batched over the N1 contiguous columns,
        //     cache-tiled exactly like the ND column pass. W is [N2][N1]. The first
        //     col pass reads the const input `in` directly (first_src) and writes
        //     the result into W -- folding the input copy-in into the gather the
        //     first pass performs anyway (a standalone std::copy in->W would be a
        //     redundant N-elem read+write, ~9% of the transform). ---
        // four_step_large's col pass wants FAT tiles, unlike the ND column pass:
        // nd_col_block budgets the tile to L2 because there it competes with sibling
        // per-thread working sets, but this pass STREAMS DRAM and its SoA scratch
        // belongs in L3. The binding cost is per-tile overhead — reloading the
        // length-n2 twiddle set + poet::dispatch across the n1/Bt tiles — which
        // amortizes once each tile spans enough columns. The optimum is ~64 columns,
        // column-count-bound (overhead amortization), not scratch-size-bound. Cap so
        // the scratch (4 planar buffers of n2*Bt reals) stays within L3/3 at the
        // largest n2 (graceful shrink past ~64M); at many threads the fat tile stays
        // efficient (DRAM-bound, the traffic cut outweighs L3 contention). Not a
        // per-size override — a single amortization constant with a cache-derived ceiling.
        constexpr std::size_t four_step_col_cols = 64;
        constexpr std::size_t Wv = xsimd::batch<T>::size;
        const std::size_t scratch_cap = (cpu_cache().l3 / 3) / (4 * n2 * sizeof(T));
        std::size_t Bt = std::min({n1, four_step_col_cols, std::max(Wv, scratch_cap)});
        Bt -= Bt % Wv;
        if (Bt < Wv) Bt = std::min(n1, Wv);
        // Inverse col scale: 1/n2 only on the tuned default path; a custom fct is
        // applied wholly in the row pass, so col runs unscaled there.
        const T col_scale = (!forward && default_scale) ? T(1) / static_cast<T>(n2) : T(1);
        soa_scratch<T, 4> csc(n2 * Bt);
        for (std::size_t c0 = 0; c0 < n1; c0 += Bt) {
            const std::size_t bc = std::min(Bt, n1 - c0);
            if (forward)
                col_dif_execute_ws<T, true, false>(W + c0, n2, n1, bc,
                    csc.buf(0), csc.buf(1), csc.buf(2), csc.buf(3), dtw_n2, T(1), in + c0);
            else if (default_scale)
                col_dif_execute_ws<T, false, true>(W + c0, n2, n1, bc,
                    csc.buf(0), csc.buf(1), csc.buf(2), csc.buf(3), dtw_n2, col_scale, in + c0);
            else
                col_dif_execute_ws<T, false, false>(W + c0, n2, n1, bc,
                    csc.buf(0), csc.buf(1), csc.buf(2), csc.buf(3), dtw_n2, T(1), in + c0);
        }

        // --- twist + outer + transpose, FUSED per row-band: for each k2 row apply
        //     W_N^{n1 k2} then a size-N1 DFT IN PLACE over the contiguous row
        //     (iterative_dif buffers the row into SoA scratch first, so row==row is
        //     safe); W then holds X in [k2][k1] order. Each W-aligned band of
        //     `four_step_tblock` rows is transposed to out[k1*N2+k2]=W[k2*N1+k1]
        //     immediately after it is computed, while still hot in L2 — a separate
        //     transpose pass re-reads all of W from DRAM (64 MB > L3 at 4M). The
        //     W-ragged edge (right cols + bottom rows) is scattered once at the end
        //     from the fully-computed W. ---
        soa_scratch<T, 4> rsc(n1);
        // Row scale carries the output factor: the tuned 1/n1 on the default
        // inverse, or the full custom fct otherwise (col ran unscaled then).
        const T row_scale = default_scale ? (forward ? T(1) : T(1) / static_cast<T>(n1)) : fct;
        constexpr std::size_t RB = four_step_tblock;   // rows per band (multiple of Wv)
        const std::size_t n2v = n2 & ~(Wv - 1);
        // Fuse the band transpose into the row loop ONLY when W exceeds L3: then a
        // separate transpose pass re-reads all of W from DRAM. When W fits L3 (e.g.
        // 1M f64 = 16 MB < 45 MB L3) the separate pass already re-reads W from L3
        // for free, so fusing only adds L2 pressure. Cache-derived, not a size table.
        const bool fuse = N * sizeof(std::complex<T>) > cpu_cache().l3;
        for (std::size_t i0 = 0; i0 < n2; i0 += RB) {
            const std::size_t iEnd = std::min(i0 + RB, n2);
            for (std::size_t k2 = i0; k2 < iEnd; ++k2) {
                std::complex<T>* row = W + k2 * n1;
                // Split twist: q = n1*k2 exact, W_N^{n1 k2} = hitab[q>>logM]*lotab[q&(M-1)].
                // q increments by k2 per column (row 0 keeps q=0 -> twist 1).
                std::size_t q = 0;
                for (std::size_t nn1 = 0; nn1 < n1; ++nn1) {
                    row[nn1] *= hitab[q >> twist_logM] * lotab[q & (twist_M - 1)];
                    q += k2;
                }
                if (row_scale == T(1)) {
                    if (forward)
                        iterative_dif_execute_ws<T, true, false>(row, row, n1,
                            rsc.buf(0), rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw_n1);
                    else
                        iterative_dif_execute_ws<T, false, false>(row, row, n1,
                            rsc.buf(0), rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw_n1);
                } else if (forward) {
                    iterative_dif_execute_ws<T, true, true>(row, row, n1,
                        rsc.buf(0), rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw_n1, row_scale);
                } else {
                    iterative_dif_execute_ws<T, false, true>(row, row, n1,
                        rsc.buf(0), rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw_n1, row_scale);
                }
            }
            // Transpose this band's W-aligned rows [i0, min(iEnd,n2v)) (both bounds
            // W-aligned since RB is a multiple of Wv) while hot in L2.
            if (fuse) {
                const std::size_t iVEnd = std::min(iEnd, n2v);
                if (iVEnd > i0) four_step_transpose_band<T>(W, out, n1, n2, i0, iVEnd);
            }
        }
        // Non-fused (W fits L3): one transpose pass over all bands after the row
        // loop, re-reading W from L3.
        if (!fuse) four_step_transpose_band<T>(W, out, n1, n2, 0, n2v);
        four_step_transpose_remainder<T>(W, out, n1, n2);
    }
};

} // namespace detail
} // namespace admiral
