#pragma once

// ============================================================================
// LARGE-N six-step route: N = N1*N2 with both leaves cache-resident full sub-FFTs
// (iterative_dif row passes, not codelets), so N1,N2 ~ sqrt(N). No working buffer:
// execute() is re-entrant. Threading comes from the caller's pool pointer, null by
// construction for ND-shared sub-plans (built nthreads = 1), so a shared plan never
// nests a parallel_for.
//
// DIT index map (input viewed [n2][n1], n = i*n1 + nn1; output viewed [n1][n2],
// k = k1*n2 + k2), split-twist W_N^{nn1*k2} factored hitab*lotab:
//   P1 transpose in -> out : out[nn1][i]  = in[i*n1 + nn1]     ([n1][n2] view)
//   P2 size-n2 row DFTs, then twist W_N^{nn1*k2}. The twist is diagonal in k2,
//      the n2-DFT's OUTPUT coordinate, so it applies AFTER the DFT. Inverse default
//      folds 1/n2 into this pass.
//   P3 in-place transpose  ([n1][n2] -> [n2][n1]).
//      P2/P3 run as one band-fused sweep whenever n1 divides n2.
//   P4 size-n1 row DFTs. Inverse default folds 1/n1; a custom fct lands
//      entirely here, the same rule as the 1-D last pass.
//   P5 in-place transpose  ([n2][n1] -> [n1][n2]); out[k1*n2+k2] = X[k].
//      P4/P5 run as one band-fused sweep whenever n1 divides n2.
//
// Ref: Bailey, "FFTs in external or hierarchical memory", J. Supercomput. 4
// (1990) 23. DOI 10.1007/BF00162341
// ============================================================================

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "dif_driver.hpp"      // dif_dispatch (1-D engine)
#include "four_step.hpp"       // four_step_split
#include "math.hpp"            // is_codelet_supported
#include "portable_trig.hpp"   // sincos_turns
#include "scratch.hpp"         // soa_scratch
#include "simd_swizzle.hpp"    // aos_deinterleave, aos_interleave
#include "thread_pool.hpp"     // thread_pool (intra-transform threading)
#include "twiddles.hpp"        // dif_twiddle_set, build_dif_twiddle_set

namespace admiral {
namespace detail {

// Out-of-place complex transpose out[j*n2+i] = W[i*n1+j]: deinterleave W source rows to
// planar re/im, xsimd::transpose in registers, interleave back -- W contiguous full-vector
// stores per output row instead of a scalar scatter. Cache-blocked at B for L1 residency.
constexpr std::size_t four_step_tblock = 32;  // transpose cache-block (complex)

// Vector WxW transpose of source band [i_lo,i_hi) (W-aligned) x cols [0,n1v),
// j-blocked at B. out[j*n2+i] = Wm[i*ld+j] over the band, ld = padded row stride.
// Fused into execute's row loop so src is read L2-hot, avoiding a DRAM re-read.
template<typename T>
void four_step_transpose_band(const std::complex<T>* Wm, std::complex<T>* out,
                              std::size_t n1, std::size_t ld, std::size_t n2,
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
                    aos_deinterleave<T>(src + ((i + r) * ld + j) * 2, re[r], im[r]);
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
                                   std::size_t n1, std::size_t ld, std::size_t n2) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t n2v = n2 & ~(W - 1);
    const std::size_t n1v = n1 & ~(W - 1);
    for (std::size_t i = 0; i < n2; ++i)
        for (std::size_t j = n1v; j < n1; ++j) out[j * n2 + i] = Wm[i * ld + j];
    for (std::size_t i = n2v; i < n2; ++i)
        for (std::size_t j = 0; j < n1v; ++j) out[j * n2 + i] = Wm[i * ld + j];
}

// WxW tile transpose core: dst(r,c) = src(c,r) over the tile, i.e.
// dst[r*ld_d + c] = src[c*ld_s + r]. Loads all W rows before storing any, so
// src == dst (diagonal self-tile) is safe.
template<typename T>
void four_step_tile_transpose(const std::complex<T>* src, std::size_t ld_s,
                              std::complex<T>* dst, std::size_t ld_d) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const T* s = reinterpret_cast<const T*>(src);
    T* d = reinterpret_cast<T*>(dst);
    batch re[W], im[W];
    for (std::size_t r = 0; r < W; ++r)
        aos_deinterleave<T>(s + r * ld_s * 2, re[r], im[r]);
    xsimd::transpose(re, re + W);
    xsimd::transpose(im, im + W);
    for (std::size_t c = 0; c < W; ++c)
        aos_interleave<T>(d + c * ld_d * 2, re[c], im[c]);
}

// In-place transpose of the n x n leading-dim-ld matrix at m. Tile-pair swaps stage one
// side through an L1 stack tile; diagonal tiles self-transpose in registers; the ragged
// border (n % W) is scalar swaps. Tile rows are independent: tile (a,b) is touched only
// by row min(a,b)'s chunk.
template<typename T>
void four_step_square_transpose_inplace(std::complex<T>* m, std::size_t ld, std::size_t n,
                                        thread_pool* pool) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t nv = n & ~(W - 1);
    parallel_for(pool, nv / W, n * n, [&](std::size_t a0, std::size_t a1, std::size_t) {
        alignas(batch::arch_type::alignment()) std::complex<T> stage[W * W];
        for (std::size_t a = a0; a < a1; ++a) {
            const std::size_t i0 = a * W;
            four_step_tile_transpose<T>(m + i0 * ld + i0, ld, m + i0 * ld + i0, ld);
            for (std::size_t j0 = i0 + W; j0 < nv; j0 += W) {
                std::complex<T>* const hi = m + i0 * ld + j0;
                std::complex<T>* const lo = m + j0 * ld + i0;
                for (std::size_t r = 0; r < W; ++r) std::copy_n(hi + r * ld, W, stage + r * W);
                four_step_tile_transpose<T>(lo, ld, hi, ld);     // hi := lo^T
                four_step_tile_transpose<T>(stage, W, lo, ld);   // lo := hi_old^T
            }
        }
    });
    for (std::size_t j = nv; j < n; ++j)
        for (std::size_t i = 0; i < j; ++i) std::swap(m[i * ld + j], m[j * ld + i]);
}

// Grid transpose: buffer viewed as [rows][cols] grid of contiguous blk-complex
// blocks; transpose the grid, b = i*cols + q -> f(b) = (b % cols)*rows + b/cols.
// Cycles rotate with a one-block stash; moves are contiguous blk-runs.
template<typename T>
void four_step_block_grid_transpose(std::complex<T>* m, std::size_t rows, std::size_t cols,
                                    std::size_t blk, thread_pool* pool) {
    const std::size_t B = rows * cols;
    const auto fwd = [&](std::size_t b) { return (b % cols) * rows + b / cols; };
    const auto pred = [&](std::size_t b) { return (b % rows) * cols + b / rows; };
    // Rotation scratch must survive a full cycle, so collect orbit-min leaders
    // first (walk stops at the first smaller element, ~O(B log L) total).
    std::vector<std::size_t> leaders;
    for (std::size_t b = 1; b + 1 < B; ++b) {
        std::size_t q = fwd(b);
        while (q > b) q = fwd(q);
        if (q == b) leaders.push_back(b);
    }
    parallel_for(pool, leaders.size(), B * blk, [&](std::size_t c0, std::size_t c1, std::size_t) {
        const auto stash = make_aligned_buffer<std::complex<T>>(blk);
        for (std::size_t c = c0; c < c1; ++c) {
            const std::size_t start = leaders[c];
            std::copy_n(m + start * blk, blk, stash.get());
            std::size_t cur = start, nxt = pred(start);
            while (nxt != start) {
                std::copy_n(m + nxt * blk, blk, m + cur * blk);
                cur = nxt;
                nxt = pred(nxt);
            }
            std::copy_n(stash.get(), blk, m + cur * blk);
        }
    });
}

// Element-cycle fallback for C % R != 0 and R % C != 0 (never taken by a pow2
// split): f(p) = p*R mod (R*C-1) on 0 < p < R*C-1, endpoints fixed. Orbit-min
// leader, single-element stash. Correctness path, not a fast one.
template<typename T>
void four_step_transpose_cycles(std::complex<T>* m, std::size_t R, std::size_t C) {
    const std::size_t NM1 = R * C - 1;
    const auto fwd = [&](std::size_t p) { return p * R % NM1; };
    for (std::size_t p = 1; p < NM1; ++p) {
        std::size_t q = fwd(p);
        while (q > p) q = fwd(q);
        if (q != p) continue;  // not orbit-min (or not a fixed point that needs no move)
        const std::complex<T> first = m[p];
        std::complex<T> prev_val = first;
        std::size_t cur = p;
        do {
            const std::size_t nxt = fwd(cur);
            const std::complex<T> keep = m[nxt];
            m[nxt] = prev_val;
            prev_val = keep;
            cur = nxt;
        } while (cur != p);
    }
}

// In-place transpose of the R x C row-major matrix at m to C x R. Decomposes by
// shape: square tile swaps (R == C), or C = m*R as strided square transposes plus
// an [R][m] block-grid rotation (blocks of R), or R = m*C as contiguous square
// transposes plus an [m][C] block-grid rotation (blocks of C), else element
// cycles. All pow2 splits (square or 1:2) take the tiled paths.
template<typename T>
void four_step_transpose_inplace(std::complex<T>* m, std::size_t R, std::size_t C,
                                 thread_pool* pool) {
    if (R < 2 || C < 2) return;
    if (C == R) {
        four_step_square_transpose_inplace<T>(m, C, R, pool);
    } else if (C % R == 0) {
        const std::size_t mblk = C / R;
        for (std::size_t q = 0; q < mblk; ++q)
            four_step_square_transpose_inplace<T>(m + q * R, C, R, pool);
        four_step_block_grid_transpose<T>(m, R, mblk, R, pool);
    } else if (R % C == 0) {
        const std::size_t mblk = R / C;
        for (std::size_t q = 0; q < mblk; ++q)
            four_step_square_transpose_inplace<T>(m + q * C * C, C, C, pool);
        four_step_block_grid_transpose<T>(m, mblk, C, C, pool);
    } else {
        four_step_transpose_cycles<T>(m, R, C);
    }
}

// Step a of the fused sweeps' lower-triangular tile pass, on all m panels: diagonal
// self-tile plus pairs (a,b) for b < a. Writes tiles (a,b) and (b,a); keyed on the larger
// index, so concurrent steps never share a tile. Shuffles and copies only — no FP, so
// the serial arms calling this are bit-for-bit the threaded step.
template<typename T>
inline void four_step_fused_sweep_step(std::complex<T>* out, std::size_t ld,
                                       std::size_t n1, std::size_t m, std::size_t a,
                                       std::complex<T>* stage) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t i0 = a * W;
    for (std::size_t q = 0; q < m; ++q) {
        std::complex<T>* const M = out + q * n1;
        four_step_tile_transpose<T>(M + i0 * ld + i0, ld, M + i0 * ld + i0, ld);
        for (std::size_t j0 = 0; j0 < i0; j0 += W) {
            std::complex<T>* const hi = M + i0 * ld + j0;
            std::complex<T>* const lo = M + j0 * ld + i0;
            for (std::size_t r = 0; r < W; ++r)
                std::copy_n(hi + r * ld, W, stage + r * W);
            four_step_tile_transpose<T>(lo, ld, hi, ld);   // hi := lo^T
            four_step_tile_transpose<T>(stage, W, lo, ld); // lo := hi_old^T
        }
    }
}

// Contiguous tile-row bounds that equalize triangle area over `nparts` parts: step a
// costs a+1 pairs, so boundary c goes at ntiles*sqrt(c/nparts). Keyed on the chunk index
// (u0/chunk): part c covers [lo(c), lo(c+1)) and the parts partition [0,ntiles), incl.
// empty ranges.
[[nodiscard]] inline std::size_t four_step_sweep_lo(std::size_t c, std::size_t nparts,
                                                    std::size_t ntiles) {
    return static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(c) / static_cast<double>(nparts))
                     * static_cast<double>(ntiles)));
}

// Phase B of a threaded fused pass: the tile sweeps, after the DFT phase barrier. Only
// the first nparts chunks run (begin<end), so balance over nparts and key the range on
// the chunk index (u0/chunk), never on tid -- a skipped chunk would silently drop its
// tile rows. Shared by both fused passes below.
template<typename T>
inline void four_step_fused_sweep_phase(std::complex<T>* out, std::size_t ld,
                                        std::size_t n1, std::size_t m, std::size_t ntiles,
                                        thread_pool* pool) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t nt = pool->size();
    const std::size_t chunk = (ntiles + nt - 1) / nt;
    const std::size_t nparts = (ntiles + chunk - 1) / chunk;
    pool->parallel_for(ntiles, [&](std::size_t u0, std::size_t, std::size_t) {
        const std::size_t c = u0 / chunk;
        const std::size_t r0 = four_step_sweep_lo(c, nparts, ntiles);
        const std::size_t r1 = four_step_sweep_lo(c + 1, nparts, ntiles);
        alignas(batch::arch_type::alignment()) std::complex<T> stage[W * W];
        for (std::size_t a = r0; a < r1; ++a)
            four_step_fused_sweep_step(out, ld, n1, m, a, stage);
    });
}

// Fused P4+P5 for splits with n2 = m*n1: each fused step DFTs one band of [n2][n1] rows,
// then runs step a of the in-place transpose's tile sweep while the band is cache-hot.
// Bitwise identical to the unfused pair: the same W-tile swaps once each and the same
// per-row DFT op sequence; only the interleaving changes.
//
// Panels (m > 1 defer splits): P5 is m square n1 x n1 transposes at out + q*n1, ld = n2;
// panel q's local row r is [n2][n1] row q + r*m. One fused step DFTs the contiguous rows
// [m*a*W, m*(a+1)*W) and sweeps tile-row a on all m panels. m == 1 is the square case.
//
// Threaded runs split at a phase barrier: (A) all DFT bands on uniform chunks, then (B)
// the tile sweeps in area-equal contiguous ranges (four_step_sweep_lo). Sweep step a reads
// bands < a, so fusing the whole loop across the pool's static chunks would chain it into
// near-serial order. pool == nullptr runs the serial interleave verbatim below.
template<typename T>
void four_step_dft_transpose_fused(std::complex<T>* out, std::size_t n1, std::size_t n2,
                                   std::size_t m, bool is_forward,
                                   const dif_twiddle_set<T>& dtw, T row_scale,
                                   thread_pool* pool) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t ld = n2;              // panel matrix row stride (== n1 when square)
    const std::size_t ntiles = n1 / W;      // tile rows per panel; caller gates n1 % W == 0
    const std::size_t bandw = m * W;        // [n2][n1] rows DFT'd per fused step
    const std::size_t total = n1 * n2;
    if (will_thread(pool, ntiles, total)) {
        pool->parallel_for(ntiles, [&](std::size_t a0, std::size_t a1, std::size_t) {
            soa_scratch<T, 4> rsc(n1);
            for (std::size_t a = a0; a < a1; ++a) {
                const std::size_t k0 = a * bandw;
                for (std::size_t k = k0; k < k0 + bandw; ++k)
                    dif_dispatch<T>(is_forward, out + k * n1, out + k * n1, n1, rsc.buf(0),
                                    rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw, row_scale,
                                    rsc.stride());
            }
        });
        four_step_fused_sweep_phase<T>(out, ld, n1, m, ntiles, pool);
        return;
    }
    parallel_for(pool, ntiles, total, [&](std::size_t a0, std::size_t a1, std::size_t) {
        soa_scratch<T, 4> rsc(n1);
        alignas(batch::arch_type::alignment()) std::complex<T> stage[W * W];
        for (std::size_t a = a0; a < a1; ++a) {
            const std::size_t k0 = a * bandw;
            for (std::size_t k = k0; k < k0 + bandw; ++k)
                dif_dispatch<T>(is_forward, out + k * n1, out + k * n1, n1, rsc.buf(0),
                                rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw, row_scale,
                                rsc.stride());
            four_step_fused_sweep_step<T>(out, ld, n1, m, a, stage);
        }
    });
}

// One [n2] row: in-place DFT via dif_dispatch, then the split twiddle
// W_N^q = hitab[q>>logM] * lotab[q&(M-1)], q advancing by j per element.
template<typename T>
inline void four_step_row_dft_twist(std::complex<T>* row, std::size_t n2, std::size_t j,
                                    bool is_forward, const dif_twiddle_set<T>& dtw, T p2_scale,
                                    soa_scratch<T, 4>& rsc, const std::complex<T>* hitab,
                                    const std::complex<T>* lotab, std::size_t twist_M,
                                    std::size_t twist_logM) {
    dif_dispatch<T>(is_forward, row, row, n2, rsc.buf(0), rsc.buf(1), rsc.buf(2), rsc.buf(3),
                    dtw, p2_scale, rsc.stride());
    std::size_t q = 0;
    for (std::size_t k2 = 0; k2 < n2; ++k2) {
        row[k2] *= hitab[q >> twist_logM] * lotab[q & (twist_M - 1)];
        q += j;
    }
}

// Fused P2+P3: a band of W [n1][n2] rows gets its size-n2 DFT + split-twist, then step a
// of P3's tile sweep while the band is hot (panel row r == P2 row r directly). Same
// bitwise-identity argument and threaded two-dispatch split as
// four_step_dft_transpose_fused.
template<typename T>
void four_step_twist_dft_transpose_fused(std::complex<T>* out, std::size_t n1,
                                         std::size_t n2, std::size_t m, bool is_forward,
                                         const dif_twiddle_set<T>& dtw, T p2_scale,
                                         const std::complex<T>* hitab,
                                         const std::complex<T>* lotab, std::size_t twist_M,
                                         std::size_t twist_logM, thread_pool* pool) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t ld = n2;
    const std::size_t ntiles = n1 / W;
    const std::size_t total = n1 * n2;
    if (will_thread(pool, ntiles, total)) {
        pool->parallel_for(ntiles, [&](std::size_t a0, std::size_t a1, std::size_t) {
            soa_scratch<T, 4> rsc(n2);
            for (std::size_t a = a0; a < a1; ++a)
                for (std::size_t j = a * W; j < (a + 1) * W; ++j)
                    four_step_row_dft_twist<T>(out + j * n2, n2, j, is_forward, dtw, p2_scale,
                                               rsc, hitab, lotab, twist_M, twist_logM);
        });
        four_step_fused_sweep_phase<T>(out, ld, n1, m, ntiles, pool);
        return;
    }
    parallel_for(pool, ntiles, total, [&](std::size_t a0, std::size_t a1, std::size_t) {
        soa_scratch<T, 4> rsc(n2);
        alignas(batch::arch_type::alignment()) std::complex<T> stage[W * W];
        for (std::size_t a = a0; a < a1; ++a) {
            for (std::size_t j = a * W; j < (a + 1) * W; ++j)
                four_step_row_dft_twist<T>(out + j * n2, n2, j, is_forward, dtw, p2_scale,
                                           rsc, hitab, lotab, twist_M, twist_logM);
            four_step_fused_sweep_step<T>(out, ld, n1, m, a, stage);
        }
    });
}

using large_split = four_step_split;

// Usable leaf: pow2 or codelet-supported smooth length (build_dif_twiddle_set requirement).
[[nodiscard]] constexpr bool large_leaf_ok(std::size_t f) {
    return f > 1 && (std::has_single_bit(f) || is_codelet_supported(f));
}

// Balanced split N = n1*n2 (both usable), closest to sqrt(N); {0,0} if none exists.
// Orientation is NOT free: swapping n1/n2 to line-align the output stride loses. Keep
// n1 <= n2.
[[nodiscard]] constexpr large_split choose_large_split(std::size_t N) {
    large_split best{};
    for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
        if (N % n1 != 0) continue;
        const std::size_t n2 = N / n1;
        if (!large_leaf_ok(n1) || !large_leaf_ok(n2)) continue;
        best = {n1, n2};   // n1 ascends, so the last match is the most balanced
    }
    return best;
}

// Hand-fit admission byte lines for this route, shared with the callers that gate on
// them: a threshold and the quantity it was fitted against are ONE artifact -- change
// either and both are re-derived, together. Bluestein's inner six-step arm is admitted
// off the same f64 serial line, so it has to be one constant and not a matching literal.

// Threaded, both precisions: a BUDGET divided by nthreads, clamped below by a FLOOR.
// The crossover against a serial DIF chain falls as 1/nthreads, then bottoms out where
// the DIF stream stops being L2-resident per pass.
inline constexpr std::size_t kLargeRouteThreadedByteBudget = std::size_t{2} << 20;
inline constexpr std::size_t kLargeRouteThreadedFloorBytes = std::size_t{512} << 10;
inline constexpr std::size_t kLargeRouteSerialF64Bytes = std::size_t{12} << 20;
inline constexpr std::size_t kLargeRouteSerialF32Bytes = (std::size_t{16} << 20) - 1;

// Serial f32 admission is a WINDOW, not a half-line: the DIF chain wins again from an
// upper size on. f64 shows no upper crossover and threading removes it for both
// precisions, so this bound is f32-and-serial only.
inline constexpr std::size_t kLargeRouteSerialF32MaxBytes = std::size_t{32} << 20;

// Threaded admission line, one source for the gate and the test that pins it. The -1
// turns four_step_large_supported's strict compare into "admit at the line".
[[nodiscard]] constexpr std::size_t large_route_threaded_bytes(std::size_t nthreads) {
    return std::max(kLargeRouteThreadedFloorBytes, kLargeRouteThreadedByteBudget / nthreads) - 1;
}

// Route gate: N is DRAM-bound (> threshold_bytes) and has a balanced usable split.
[[nodiscard]] constexpr bool four_step_large_supported(std::size_t N, std::size_t elem_bytes,
                                                       std::size_t threshold_bytes) {
    return N * elem_bytes > threshold_bytes && choose_large_split(N).valid();
}

// Fused-legal split search: the most balanced leaf-ok split with n2 % n1 == 0 and
// n1 % W == 0. The balanced chooser does not know about the fused-shape requirement, so
// it can elect a split the fused sweeps cannot run. {0,0} when nothing legal exists.
template<typename T>
[[nodiscard]] constexpr large_split choose_fused_large_split(std::size_t N) {
    // W<=2: only balanced splits that also satisfy the fused-shape predicate, so that the
    // gate and the plan chooser agree.
    constexpr std::size_t W = xsimd::batch<T>::size;
    if constexpr (W <= 2) {
        const large_split sp = choose_large_split(N);
        if (sp.valid() && sp.n2 % sp.n1 == 0 && sp.n1 % W == 0) return sp;
        return {};
    }
    large_split best{};
    for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
        if (N % n1 != 0) continue;
        const std::size_t n2 = N / n1;
        if (n2 % n1 != 0 || n1 % W != 0) continue;
        if (!large_leaf_ok(n1) || !large_leaf_ok(n2)) continue;
        best = {n1, n2};   // n1 ascends: last match is the most balanced legal one
    }
    return best;
}

// The split shape the engine runs fast: n2 % n1 == 0 keeps the in-place transposes on the
// tiled/defer paths and enables the band-fused sweeps, whose own guard is n1 % W == 0 (so
// this predicate tracks W across ISAs). n2 % n1 != 0 puts both transposes on the serial
// element-cycle fallback (four_step_transpose_cycles), where even a scratch-buffer
// transpose loses to the DIF chain. Used by the public gate (plan.hpp large_route_admits)
// and the bluestein inner delegate (bluestein.hpp bluestein_inner_six_step_admits).
template<typename T>
[[nodiscard]] constexpr bool four_step_large_fused_shape(std::size_t N) {
    return choose_fused_large_split<T>(N).valid();
}

// Plan state: two leaf twiddle sets + split-twiddle tables.
// No working buffer: execute() moves data between in and out only (re-entrant).
template<typename T>
struct four_step_large_plan {
    std::size_t n1 = 0, n2 = 0;
    bool is_forward = true;
    dif_twiddle_set<T> dtw_n2;                 // row form (P2 contiguous rows)
    dif_twiddle_set<T> dtw_n1;                 // row form (P4 contiguous rows)
    // Split-twiddle: W_N^{q} = hitab[q>>logM] * lotab[q&(M-1)], q = nn1*k2 exact,
    // M = 2^ceil(log2 sqrt(N)). O(sqrt(N)) memory.
    std::vector<std::complex<T>> lotab;        // W_N^{j},   j in [0,M)
    std::vector<std::complex<T>> hitab;        // W_N^{h*M}, h in [0, (N-1)>>logM]
    std::size_t twist_M = 1, twist_logM = 0;

    four_step_large_plan(std::size_t N, bool fwd) : is_forward(fwd) {
        // Prefer the fused-legal split (the public gates only admit shapes that
        // have one); fall back to the balanced split for direct/forced builds
        // so the engine still runs on any leaf-ok factorization.
        large_split sp = choose_fused_large_split<T>(N);
        if (!sp.valid()) sp = choose_large_split(N);
        n1 = sp.n1;
        n2 = sp.n2;
        dtw_n2 = build_dif_twiddle_set<T>(n2, nullptr);
        dtw_n1 = build_dif_twiddle_set<T>(n1, nullptr);
        // Split-twiddle tables for W_N^{nn1 k2}: q = nn1*k2 = qhi*M + qlo,
        // W_N^q = hitab[qhi]*lotab[qlo]. Exact sincos per entry (one rounding), no
        // accumulated rotation error. M = 2^k >= sqrt(N).
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

    // in == out or in != out; result written to out, scaled by fct. Default scale
    // (forward=1, inverse=1/N): P2 folds 1/n2, P4 folds 1/n1. Custom fct goes
    // entirely into P4's row pass (P2 unscaled). pool == nullptr runs everything
    // inline; ND-shared sub-plans are built nthreads = 1, so their pool is null by
    // construction and nothing nests a parallel_for. No plan-owned buffer exists, so
    // execute() is re-entrant.
    void execute(const std::complex<T>* in, std::complex<T>* out, T fct,
                 thread_pool* pool = nullptr) const {
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == (is_forward ? T(1) : T(1) / static_cast<T>(N)));
        constexpr std::size_t Wv = xsimd::batch<T>::size;
        constexpr std::size_t RB = four_step_tblock;   // rows per band (multiple of Wv)

        // --- P1: transpose in -> out, out viewed [n1][n2]. Reuses the band
        //     transpose verbatim with ld == n1 (`in` has the natural stride). ---
        if (in == out) {
            four_step_transpose_inplace<T>(out, n2, n1, pool);
        } else {
            const std::size_t n2v = n2 & ~(Wv - 1);
            const std::size_t nbands = (n2 + RB - 1) / RB;
            parallel_for(pool, nbands, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
                const std::size_t i0 = std::min(b0 * RB, n2v);
                const std::size_t iE = std::min(b1 * RB, n2v);   // both Wv-aligned
                if (iE > i0) four_step_transpose_band<T>(in, out, n1, n1, n2, i0, iE);
            });
            four_step_transpose_remainder<T>(in, out, n1, n1, n2);
        }

        // --- P2: size-n2 DFT per row of out (viewed [n1][n2]), then the split
        //     twist W_N^{j*k2}, q increments by j per column; it lands after the
        //     DFT because it is diagonal in the DFT's output coordinate. ---
        // --- P3: [n1][n2] -> [n2][n1] in place. Wide splits (n2 = m*n1) defer the
        //     block-grid rotation: after the per-block square transposes every P4
        //     row is already contiguous, so the rotation composes into P5 below. ---
        const T p2_scale = (!is_forward && default_scale) ? T(1) / static_cast<T>(n2) : T(1);
        const bool defer_rot = (n2 % n1 == 0) && (n2 != n1);
        if (n2 % n1 == 0 && n1 % Wv == 0) {   // P2+P3 as one band-fused sweep
            four_step_twist_dft_transpose_fused<T>(out, n1, n2, n2 / n1, is_forward,
                                                   dtw_n2, p2_scale, hitab.data(),
                                                   lotab.data(), twist_M, twist_logM,
                                                   pool);
        } else {
            const std::size_t nb1 = (n1 + RB - 1) / RB;
            parallel_for(pool, nb1, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
                soa_scratch<T, 4> rsc(n2);
                for (std::size_t b = b0; b < b1; ++b) {
                    const std::size_t jE = std::min((b + 1) * RB, n1);
                    for (std::size_t j = b * RB; j < jE; ++j)
                        four_step_row_dft_twist<T>(out + j * n2, n2, j, is_forward, dtw_n2,
                                                   p2_scale, rsc, hitab.data(), lotab.data(),
                                                   twist_M, twist_logM);
                }
            });

            if (defer_rot) {
                for (std::size_t q = 0; q < n2 / n1; ++q)
                    four_step_square_transpose_inplace<T>(out + q * n1, n2, n1, pool);
            } else {
                four_step_transpose_inplace<T>(out, n1, n2, pool);
            }
        }

        // --- P4+P5: size-n1 row DFTs, then [n2][n1] -> [n1][n2]; out[k1*n2+k2]
        //     = X[k]. Square and defer splits (n2 = m*n1) fuse the two phases
        //     band-by-band, each DFT'd band transposed onward while cache-hot
        //     (four_step_dft_transpose_fused); the defer P5 is the same m strided
        //     square transposes as P3 (rotation composed away, see P3).
        //     Non-divisible splits keep the unfused phases below. ---
        const T row_scale = default_scale ? (is_forward ? T(1) : T(1) / static_cast<T>(n1)) : fct;
        if (n2 % n1 == 0 && n1 % Wv == 0) {   // P4+P5 as one band-fused sweep
            four_step_dft_transpose_fused<T>(out, n1, n2, n2 / n1, is_forward, dtw_n1,
                                             row_scale, pool);
        } else {
            const std::size_t nb2 = (n2 + RB - 1) / RB;
            parallel_for(pool, nb2, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
                soa_scratch<T, 4> rsc(n1);
                for (std::size_t b = b0; b < b1; ++b) {
                    const std::size_t kE = std::min((b + 1) * RB, n2);
                    for (std::size_t k = b * RB; k < kE; ++k)
                        dif_dispatch<T>(is_forward, out + k * n1, out + k * n1, n1, rsc.buf(0),
                                        rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw_n1, row_scale,
                                        rsc.stride());
                }
            });

            if (defer_rot) {
                for (std::size_t q = 0; q < n2 / n1; ++q)
                    four_step_square_transpose_inplace<T>(out + q * n1, n2, n1, pool);
            } else {
                four_step_transpose_inplace<T>(out, n2, n1, pool);
            }
        }
    }
};

} // namespace detail
} // namespace admiral
