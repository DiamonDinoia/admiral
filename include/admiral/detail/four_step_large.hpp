#pragma once

// ============================================================================
// LARGE-N six-step route: N = n1*n2 with both leaves cache-resident full sub-FFTs
// (`iterative_dif` row passes, not codelets), so n1,n2 ~ sqrt(N). No working buffer:
// `execute()` is re-entrant. ND-shared sub-plans are built nthreads = 1, so a shared
// plan never nests a `parallel_for`.
//
// DIT index map (input viewed [n2][n1], output viewed [n1][n2]), split-twist
// W_N^{nn1*k2} factored hitab*lotab:
//   P1 transpose in -> out.
//   P2 size-n2 row DFTs, then the twist. The twist is diagonal in k2, the n2-DFT's
//      OUTPUT coordinate, so it applies AFTER the DFT. Inverse default folds 1/n2 in.
//   P3 in-place transpose [n1][n2] -> [n2][n1].
//   P4 size-n1 row DFTs. Inverse default folds 1/n1; a custom fct lands entirely here.
//   P5 in-place transpose [n2][n1] -> [n1][n2]; out[k1*n2+k2] = X[k].
//   P2/P3 and P4/P5 run as one band-fused sweep whenever n1 divides n2.
//
// Ref: Bailey, "FFTs in external or hierarchical memory", J. Supercomput. 4
// (1990) 23. DOI 10.1007/BF00162341
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "cxx_compat.hpp"  // `detail::bit_ceil`, `detail::countr_zero`, `detail::has_single_bit`

#include "dif_driver.hpp"      // `dif_dispatch` (1-D engine)
#include "four_step.hpp"       // `four_step_split`
#include "math.hpp"            // `is_codelet_supported`
#include "portable_trig.hpp"   // `sincos_turns`
#include "scratch.hpp"         // `soa_scratch`
#include "simd_swizzle.hpp"    // `aos_deinterleave`, `aos_interleave`
#include "thread_pool.hpp"     // `thread_pool` (intra-transform threading)
#include "twiddles.hpp"        // `dif_twiddle_set`, `build_dif_twiddle_set`

namespace admiral {
namespace detail {

// Out-of-place complex transpose out[j*n2+i] = W[i*n1+j]: WxW register tiles over
// planar re/im give W contiguous full-vector stores per output row, not a scalar
// scatter. Cache-blocked at B for L1 residency.
constexpr std::size_t four_step_tblock = 32;  // transpose cache-block (complex)

// Vector WxW transpose of source band [`i_lo`,`i_hi`) (W-aligned) x cols [0,n1v),
// j-blocked at B: out[j*n2+i] = Wm[i*ld+j], ld = padded row stride. Fused into
// execute's row loop, which reads src L2-hot and skips a DRAM re-read.
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

// W-ragged remainder scatter, done once after all bands: right cols [n1v,n1) x all
// rows, then bottom rows [n2v,n2) x left cols [0,n1v).
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

// WxW tile transpose dst[r*`ld_d` + c] = src[c*`ld_s` + r]. All W rows load before any
// store, so src == dst (diagonal self-tile) is safe.
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

// In-place transpose of the n x n leading-dim-ld matrix at m: diagonal tiles
// self-transpose in registers, off-diagonal pairs swap through an L1 stack tile, and
// the ragged n % W border is scalar swaps. Tile rows are disjoint across a, which is
// what the `parallel_for` chunks over.
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
                four_step_tile_transpose<T>(stage, W, lo, ld);   // lo := `hi_old`^T
            }
        }
    });
    for (std::size_t j = nv; j < n; ++j)
        for (std::size_t i = 0; i < j; ++i) std::swap(m[i * ld + j], m[j * ld + i]);
}

// Grid transpose: the buffer viewed as a [rows][cols] grid of contiguous blk-complex
// blocks is transposed, b -> (b % cols)*rows + b/cols. Cycles rotate with a one-block
// stash; moves are contiguous blk-runs.
template<typename T>
void four_step_block_grid_transpose(std::complex<T>* m, std::size_t rows, std::size_t cols,
                                    std::size_t blk, thread_pool* pool) {
    const std::size_t B = rows * cols;
    const auto fwd = [&](std::size_t b) { return (b % cols) * rows + b / cols; };
    const auto pred = [&](std::size_t b) { return (b % rows) * cols + b / rows; };
    // Collect orbit-min leaders first: the rotation stash must survive a full cycle.
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

// Element-cycle fallback for mutually non-dividing R and C (no `pow2` split reaches
// it): f(p) = p*R mod (R*C-1) on 0 < p < R*C-1, endpoints fixed, orbit-min leaders,
// single-element stash. Correctness path, not a fast one.
template<typename T>
void four_step_transpose_cycles(std::complex<T>* m, std::size_t R, std::size_t C) {
    const std::size_t NM1 = R * C - 1;
    const auto fwd = [&](std::size_t p) { return p * R % NM1; };
    for (std::size_t p = 1; p < NM1; ++p) {
        std::size_t q = fwd(p);
        while (q > p) q = fwd(q);
        if (q != p) continue;  // not orbit-min, or a fixed point needing no move
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

// In-place transpose of the R x C row-major matrix at m to C x R, decomposed by
// shape: square tile swaps (R == C), or C = m*R / R = m*C as per-block square
// transposes plus a block-grid rotation, else element cycles. All `pow2` splits
// (square or 1:2) take the tiled paths.
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

// Step a of the fused sweeps' lower-triangular tile pass on all m panels: diagonal
// self-tile plus pairs (a,b) for b < a, keyed on the larger index, so concurrent
// steps never share a tile. Shuffles and copies only, no FP: the serial arms calling
// this are bit-for-bit the threaded step.
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
            four_step_tile_transpose<T>(stage, W, lo, ld); // lo := `hi_old`^T
        }
    }
}

// Contiguous tile-row bounds that equalize triangle area over nparts parts: step a
// costs a+1 pairs, so boundary c goes at ntiles*sqrt(c/nparts). Part c covers
// [lo(c), lo(c+1)); the parts partition [0,ntiles), including empty ranges.
[[nodiscard]] inline std::size_t four_step_sweep_lo(std::size_t c, std::size_t nparts,
                                                    std::size_t ntiles) {
    return static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(c) / static_cast<double>(nparts))
                     * static_cast<double>(ntiles)));
}

// Phase B of a threaded fused pass: the tile sweeps after the DFT phase barrier,
// shared by both fused passes below. Only the first nparts chunks run (begin<end), so
// balance over nparts and key the range on the chunk index (u0/chunk), never on tid:
// a skipped chunk would drop its tile rows with no error.
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

// Fused P4+P5 for splits with n2 = m*n1: each fused step DFTs one band of [n2][n1]
// rows, then runs step a of the in-place transpose's tile sweep while the band is
// cache-hot. Bitwise identical to the unfused pair: the same W-tile swaps and per-row
// DFT op sequence; only the interleaving changes.
//
// Panels (m > 1 defer splits): P5 is m square n1 x n1 transposes at out + q*n1, ld =
// n2. One fused step DFTs rows [m*a*W, m*(a+1)*W) and sweeps tile-row a on all m
// panels. Threaded runs split at a phase barrier, DFTs then area-equal sweeps:
// sweep step a reads bands < a, so fused static chunks would chain near-serially.
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

// One [n2] row: in-place DFT via `dif_dispatch`, then the split twiddle
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

// Fused P2+P3: a band of W [n1][n2] rows gets its size-n2 DFT + split-twist, then
// step a of P3's tile sweep while the band is hot. Same bitwise-identity argument and
// threaded phase split as `four_step_dft_transpose_fused`.
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

// Usable leaf: `pow2` or `codelet`-supported smooth length (`build_dif_twiddle_set` requirement).
[[nodiscard]] constexpr bool large_leaf_ok(std::size_t f) {
    return f > 1 && (detail::has_single_bit(f) || is_codelet_supported(f));
}

// Balanced split N = n1*n2 (both usable), closest to sqrt(N), n1 <= n2; {0,0} if
// none exists. Orientation is NOT free: swapping n1/n2 to line-align the output
// stride loses.
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
// them: a threshold and the quantity it was fitted against are ONE artifact, so change
// either and re-derive both. Bluestein's inner six-step arm is admitted off the same
// f64 serial line, so it is one constant, never a matching literal.

// Threaded, both precisions: BUDGET / nthreads, clamped below by FLOOR. The crossover
// against a serial DIF chain falls as 1/nthreads and bottoms out where the DIF stream
// stops being L2-resident per pass.
inline constexpr std::size_t kLargeRouteThreadedByteBudget = std::size_t{2} << 20;
inline constexpr std::size_t kLargeRouteThreadedFloorBytes = std::size_t{512} << 10;

// Serial admission, per precision: measured crossover against the serial DIF chain.
inline constexpr std::size_t kLargeRouteSerialF64Bytes = std::size_t{12} << 20;
inline constexpr std::size_t kLargeRouteSerialF32Bytes = (std::size_t{16} << 20) - 1;

// Serial f32 admission is a WINDOW, not a half-line: the DIF chain wins again above
// this size. f64 shows no upper crossover and threading removes it, so this bound is
// f32-serial only.
inline constexpr std::size_t kLargeRouteSerialF32MaxBytes = std::size_t{32} << 20;

// Threaded admission line, one source for the gate and the test that pins it. The -1
// turns `four_step_large_supported`'s strict compare into "admit at the line".
[[nodiscard]] constexpr std::size_t large_route_threaded_bytes(std::size_t nthreads) {
    return std::max(kLargeRouteThreadedFloorBytes, kLargeRouteThreadedByteBudget / nthreads) - 1;
}

// Route gate: N is DRAM-bound (> `threshold_bytes`) and has a balanced usable split.
[[nodiscard]] constexpr bool four_step_large_supported(std::size_t N, std::size_t elem_bytes,
                                                       std::size_t threshold_bytes) {
    return N * elem_bytes > threshold_bytes && choose_large_split(N).valid();
}

// Fused-legal split search: the most balanced leaf-ok split with n2 % n1 == 0 and
// n1 % W == 0, or {0,0}. The balanced chooser alone can elect a split the fused
// sweeps cannot run.
template<typename T>
[[nodiscard]] constexpr large_split choose_fused_large_split(std::size_t N) {
    // W<=2: keep only balanced splits that satisfy the fused-shape predicate, so the
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

// The split shape the engine runs fast: n2 % n1 == 0 keeps the transposes tiled and
// enables the band-fused sweeps, whose guard n1 % W == 0 tracks W across ISAs.
// n2 % n1 != 0 puts both transposes on the serial element-cycle fallback, which loses
// to the DIF chain. Used by plan.hpp `large_route_admits` and bluestein.hpp
// `bluestein_inner_six_step_admits`.
template<typename T>
[[nodiscard]] constexpr bool four_step_large_fused_shape(std::size_t N) {
    return choose_fused_large_split<T>(N).valid();
}

// WS-3 sweep gate by design (knob A/B on the three hosts, 2026-08-31; Measurer's
// knob tables fi/mt/out/knobab-*): threaded on AVX-512-class builds only — MT/ice
// wins everywhere (-20.6..-52.5%), MT/genoa 5/6 wins (worst -3.5% tie), MT/rome
// breaches the no-loss rule (+6.2% @ 2^20, +12.0% @ 2^23), ST everywhere loses up to
// +49%; the discriminator that matches the data is exactly (threaded, AVX-512
// class). The trait is compile-time (the bench trees build per-family
// BENCH_ARCHs, so the class is honest at compile time). The in-place engine stays
// reachable at (nthreads <= 1 || !avx512-class) for the A/Bs when the Measurer needs
// the old arm: those plans run it.
[[nodiscard]] inline constexpr bool fsl_ws_arch_class() { return XSIMD_WITH_AVX512F; }

// WS-3 G1+G3 sweeps for the large-N route (fi/ws3-profiler-r1.md; falsifier probe on
// SPR 2026-09-01: same-traffic contiguous copies in place of the strided transposes
// dropped the 2^24 f64 ST cell by 40%, and twist-off moved it by -3% only).
// Two contiguous-workspace sweeps replace the three in-place strided transposes;
// the data stays in the [n1][n2] view everywhere (as after today's P1+P2):
// ws[j*n2 + i] holds the twist-scaled size-n2 DFT of input column j, and
//   S1 = P1 gather through register WxW tiles + row DFT + split twist + contiguous ws store
//   S2 = ws column-tile gather + size-n1 row DFT (scale folded) + register-tile bursts
// Per row the DFT call, twist order and scale folds are the in-place engine's; the
// arithmetic per element is the same up to inline-context contraction under
// -ffast-math, and the sweep's own bits are invariant across in/out alignment (its
// arithmetic never sees a layout; the pre-WS-3 engine's bits do move with alignment).

// S1: per band of W rows (j0..j0+W): gather in's rows through register WxW tiles into
// W staged rows, run the n2 row DFT + split twist per staged row, store contiguously.
template<typename T>
void fsl_ws_s1(const std::complex<T>* in, std::complex<T>* ws, std::size_t n1,
               std::size_t n2, bool is_forward, const dif_twiddle_set<T>& dtw, T p2_scale,
               const std::complex<T>* hitab, const std::complex<T>* lotab,
               std::size_t twist_M, std::size_t twist_logM, thread_pool* pool) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t N = n1 * n2;
    const std::size_t n1v = n1 & ~(W - 1);
    const std::size_t n2v = n2 & ~(W - 1);
    // One unit per W-row band, plus the scalar tail band.
    const std::size_t nbands = n1v / W + (n1v < n1 ? 1 : 0);
    parallel_for(pool, nbands, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
        soa_scratch<T, 4> rsc(n2);
        const auto stage = make_aligned_buffer<std::complex<T>>(W * n2);
        for (std::size_t b = b0; b < b1; ++b) {
            const std::size_t j0 = b * W;
            if (j0 >= n1v) {
                // Row tail: band of < W rows, scalar gather (same per-row ops).
                for (std::size_t j = n1v; j < n1; ++j) {
                    for (std::size_t i = 0; i < n2; ++i) stage[i] = in[i * n1 + j];
                    four_step_row_dft_twist<T>(stage.get(), n2, j, is_forward, dtw, p2_scale,
                                               rsc, hitab, lotab, twist_M, twist_logM);
                    std::copy_n(stage.get(), n2, ws + j * n2);
                }
                continue;
            }
            for (std::size_t i0 = 0; i0 < n2v; i0 += W)
                // in rows i0..+W x cols j0..+W -> stage rows 0..W, cols i0..+W.
                four_step_tile_transpose<T>(in + i0 * n1 + j0, n1, stage.get() + i0, n2);
            for (std::size_t i = n2v; i < n2; ++i)   // column tail, scalar
                for (std::size_t k = 0; k < W; ++k) stage[k * n2 + i] = in[i * n1 + j0 + k];
            for (std::size_t k = 0; k < W; ++k) {
                four_step_row_dft_twist<T>(stage.get() + k * n2, n2, j0 + k, is_forward, dtw,
                                           p2_scale, rsc, hitab, lotab, twist_M, twist_logM);
                std::copy_n(stage.get() + k * n2, n2, ws + (j0 + k) * n2);
            }
        }
    });
}

// S2: per band of W ws columns k2 (b0..b0+W): gather ws columns through register WxW
// tiles into W staged rows, size-n1 DFT each (row scale folded), then burst the
// results into out by register tiles: out[(j1+r)*n2 + (b0+c)] = X[(j1+r)*n2 + b0+c].
template<typename T>
void fsl_ws_s2(const std::complex<T>* ws, std::complex<T>* out, std::size_t n1,
               std::size_t n2, bool is_forward, const dif_twiddle_set<T>& dtw, T row_scale,
               thread_pool* pool) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t N = n1 * n2;
    const std::size_t n1v = n1 & ~(W - 1);
    const std::size_t n2v = n2 & ~(W - 1);
    const std::size_t nbands = n2v / W + (n2v < n2 ? 1 : 0);
    parallel_for(pool, nbands, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
        soa_scratch<T, 4> rsc(n1);
        const auto stage = make_aligned_buffer<std::complex<T>>(W * n1);
        for (std::size_t b = b0; b < b1; ++b) {
            const std::size_t kb0 = b * W;
            if (kb0 >= n2v) {
                // Column-band tail: scalar gather/store, same per-row ops.
                for (std::size_t k2 = n2v; k2 < n2; ++k2) {
                    for (std::size_t j = 0; j < n1; ++j) stage[j] = ws[j * n2 + k2];
                    dif_dispatch<T>(is_forward, stage.get(), stage.get(), n1, rsc.buf(0),
                                    rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw, row_scale,
                                    rsc.stride());
                    for (std::size_t j = 0; j < n1; ++j) out[j * n2 + k2] = stage[j];
                }
                continue;
            }
            for (std::size_t j0 = 0; j0 < n1v; j0 += W)
                // ws rows j0..+W x cols kb0..+W -> stage rows 0..W (the band), cols j0..+W.
                four_step_tile_transpose<T>(ws + j0 * n2 + kb0, n2, stage.get() + j0, n1);
            for (std::size_t j = n1v; j < n1; ++j)   // row tail within the band, scalar
                for (std::size_t c = 0; c < W; ++c) stage[c * n1 + j] = ws[j * n2 + kb0 + c];
            for (std::size_t c = 0; c < W; ++c)
                dif_dispatch<T>(is_forward, stage.get() + c * n1, stage.get() + c * n1, n1,
                                rsc.buf(0), rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw,
                                row_scale, rsc.stride());
            for (std::size_t j1 = 0; j1 < n1v; j1 += W)
                // stage band rows x cols j1..+W -> out rows j1..+W x cols kb0..+W.
                four_step_tile_transpose<T>(stage.get() + j1, n1, out + j1 * n2 + kb0, n2);
            for (std::size_t j = n1v; j < n1; ++j)   // out row tail, scalar
                for (std::size_t c = 0; c < W; ++c) out[j * n2 + kb0 + c] = stage[c * n1 + j];
        }
    });
}

// Plan state: two leaf twiddle sets + split-twiddle tables. The WS-3 sweep keeps one
// contiguous workspace block per executing thread (sleef's `xn` scheme: the plan is
// const + re-entrant, the scratch is not); the in-place engine carries no buffer.
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
    // WS-3 sweep workspace, one N-element block per executing thread, lazily allocated
    // on first use and freed with the plan. The plan itself moves (it sits in the
    // route variant), so the map lives in a small holder the plan points to.
    struct fsl_ws_state {
        std::mutex mu;
        std::unordered_map<std::thread::id, aligned_buffer<std::complex<T>>> slots;
    };
    mutable std::shared_ptr<fsl_ws_state> ws_state_;

    four_step_large_plan(std::size_t N, bool fwd) : is_forward(fwd) {
        // Fused-legal split first (the public gates admit only shapes that have
        // one); balanced split as fallback, so a direct/forced build still runs.
        large_split sp = choose_fused_large_split<T>(N);
        if (!sp.valid()) sp = choose_large_split(N);
        n1 = sp.n1;
        n2 = sp.n2;
        dtw_n2 = build_dif_twiddle_set<T>(n2, nullptr);
        dtw_n1 = build_dif_twiddle_set<T>(n1, nullptr);
        // Exact sincos per entry (one rounding, no accumulated rotation error).
        // M = 2^k >= sqrt(N).
        twist_M = detail::bit_ceil(static_cast<std::size_t>(std::sqrt(static_cast<double>(N))) + 1);
        twist_logM = static_cast<std::size_t>(detail::countr_zero(twist_M));
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

    // Result written to out (in == out allowed), scaled by fct. Default scale
    // (forward 1, inverse 1/N): S1 folds 1/n2, S2 folds 1/n1; a custom fct goes
    // entirely into S2's row pass (S1 unscaled) — the same fold sites as the pre-WS-3
    // engine. pool == nullptr runs inline; the workspace map keeps `execute()`
    // re-entrant on the sweep. The env switch selects the sweeps.
    void execute(const std::complex<T>* in, std::complex<T>* out, T fct,
                 thread_pool* pool = nullptr) const {
        if (fsl_ws_arch_class() && pool != nullptr && pool->size() > 1) {
            execute_ws(in, out, fct, pool);
            return;
        }
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == (is_forward ? T(1) : T(1) / static_cast<T>(N)));
        constexpr std::size_t Wv = xsimd::batch<T>::size;
        constexpr std::size_t RB = four_step_tblock;   // rows per band (multiple of Wv)

        // P1: transpose in -> out, out viewed [n1][n2]; the band transpose with
        // ld == n1 (in has the natural stride).
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

        // P2: size-n2 DFT per row of out (viewed [n1][n2]), then the split twist,
        // which lands after the DFT because it is diagonal in the output coordinate.
        // P3: [n1][n2] -> [n2][n1] in place. Wide splits (n2 = m*n1) defer the
        // block-grid rotation: after the per-block square transposes every P4 row is
        // already contiguous, so the rotation composes into P5 below.
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

        // P4+P5: size-n1 row DFTs, then [n2][n1] -> [n1][n2]; out[k1*n2+k2] = X[k].
        // Square and defer splits (n2 = m*n1) fuse the phases band-by-band
        // (`four_step_dft_transpose_fused`); the defer P5 is the same m strided
        // square transposes as P3. Non-divisible splits keep the unfused phases.
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

    // The calling thread's N-element sweep workspace (sleef's `xn` scheme: lazily
    // allocated, cached on the plan keyed by thread id, freed with the plan).
    std::complex<T>* fsl_ws_buf() const {
        if (!ws_state_) ws_state_ = std::make_shared<fsl_ws_state>();
        const std::thread::id id = std::this_thread::get_id();
        std::lock_guard<std::mutex> lk(ws_state_->mu);
        auto& slot = ws_state_->slots[id];
        if (!slot) slot = make_aligned_buffer<std::complex<T>>(n1 * n2);
        return slot.get();
    }

    // The WS-3 pair. `in == out` is safe: every element of `in` is read by S1 before
    // S2 writes the first element of `out`.
    void execute_ws(const std::complex<T>* in, std::complex<T>* out, T fct,
                    thread_pool* pool = nullptr) const {
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == (is_forward ? T(1) : T(1) / static_cast<T>(N)));
        const T p2_scale = (!is_forward && default_scale) ? T(1) / static_cast<T>(n2) : T(1);
        const T row_scale = default_scale ? (is_forward ? T(1) : T(1) / static_cast<T>(n1)) : fct;
        std::complex<T>* const ws = fsl_ws_buf();
        fsl_ws_s1<T>(in, ws, n1, n2, is_forward, dtw_n2, p2_scale, hitab.data(),
                     lotab.data(), twist_M, twist_logM, pool);
        fsl_ws_s2<T>(ws, out, n1, n2, is_forward, dtw_n1, row_scale, pool);
    }
};

} // namespace detail
} // namespace admiral
