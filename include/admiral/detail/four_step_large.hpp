#pragma once

// Four-step FFT for out-of-cache N, with fused transpose sweeps. Bailey, FFTs in External or
// Hierarchical Memory, J. Supercomputing 4 (1990) 23.
//
// `execute()` holds no working buffer, so it is re-entrant. ND-shared sub-plans are built with
// nthreads = 1, so a shared plan never nests a `parallel_for`.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>

#include <vector>
#include "cache.hpp"
#include "cxx_compat.hpp"

#include "dif_driver.hpp"
#include "four_step.hpp"
#include "math.hpp"
#include "portable_trig.hpp"
#include "scratch.hpp"
#include "simd_swizzle.hpp"
#include "thread_pool.hpp"
#include "twiddles.hpp"

namespace admiral {
namespace detail {

constexpr std::size_t four_step_tblock = 32;

// Where the streaming transpose starts to pay, in multiples of L3. Read off a crossover on SPR
// (Xeon w5-3435X, 47.2 MB L3, W=8 f64, serial, out-of-place): the arm reads 1.007 at 32 MiB and
// 0.999 at 64 MiB against a 0.985-1.004 same-binary control, then 0.973, 0.946 and 0.914 at 128,
// 256 and 512 MiB. The sign flips between 1.36x and 2.71x of L3, and 2 is the geometric middle
// of that bracket. Re-derive it on any host class before trusting it there.
constexpr std::size_t kFourStepStreamL3Mult = 2;

// Stream: store the transposed columns non-temporally. Legal only when four_step_stream_ok
// says every destination address is arch-aligned, and profitable only past the byte line there.
template<typename T, bool Stream = false>
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
                xsimd::transpose(re, re + W);
                xsimd::transpose(im, im + W);
                for (std::size_t c = 0; c < W; ++c) {
                    T* p = dst + ((j + c) * n2 + i) * 2;
                    if constexpr (Stream) {
                        aos_interleave_stream<T>(p, re[c], im[c]);
                    } else {
                        aos_interleave<T>(p, re[c], im[c]);
                    }
                }
            }
    }
}

// The out-of-place transpose may stream its stores when both hold:
//
//   alignment  every store is one arch vector at dst + ((j + c) * n2 + i) * 2 reals, with i a
//              multiple of W. An arch-aligned base plus a row pitch that is a whole number of
//              arch vectors makes every such address arch-aligned.
//   profit     a stream store bypasses the cache, so it costs where the output is still read
//              back from L3 on the next pass and pays once it cannot be. The line is relative
//              to L3 and not absolute, because that is the quantity the sign tracks.
template<typename T>
[[nodiscard]] inline bool four_step_stream_ok(const std::complex<T>* out, std::size_t n2,
                                              std::size_t bytes) {
    constexpr std::size_t A = xsimd::batch<T>::arch_type::alignment();
    const auto base = reinterpret_cast<std::uintptr_t>(out);
    if (base % A != 0 || (n2 * 2 * sizeof(T)) % A != 0) return false;
    return bytes >= kFourStepStreamL3Mult * cpu_cache().l3;
}

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
                four_step_tile_transpose<T>(lo, ld, hi, ld);
                four_step_tile_transpose<T>(stage, W, lo, ld);
            }
        }
    });
    for (std::size_t j = nv; j < n; ++j)
        for (std::size_t i = 0; i < j; ++i) std::swap(m[i * ld + j], m[j * ld + i]);
}

template<typename T>
void four_step_block_grid_transpose(std::complex<T>* m, std::size_t rows, std::size_t cols,
                                    std::size_t blk, thread_pool* pool) {
    const std::size_t B = rows * cols;
    const auto fwd = [&](std::size_t b) { return (b % cols) * rows + b / cols; };
    const auto pred = [&](std::size_t b) { return (b % rows) * cols + b / rows; };
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

template<typename T>
void four_step_transpose_cycles(std::complex<T>* m, std::size_t R, std::size_t C) {
    const std::size_t NM1 = R * C - 1;
    const auto fwd = [&](std::size_t p) { return p * R % NM1; };
    for (std::size_t p = 1; p < NM1; ++p) {
        std::size_t q = fwd(p);
        while (q > p) q = fwd(q);
        if (q != p) continue;
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

// Tile rows per panel. `ld` is n2, a power of two whenever the length is, so the partner tile
// at M + j0 * ld + i0 advances W * ld per step and a whole row walk lands in the same cache
// sets. A B x B panel holds both tiles of every pair in L1 across the panel, which is the
// blocking four_step_transpose_band already has and this walk never got.
[[nodiscard]] inline constexpr std::size_t four_step_panel_tiles(std::size_t W) {
    return four_step_tblock >= W ? four_step_tblock / W : std::size_t{1};
}

// Transposes the tile rows [a0, a1) of every one of the `m` n1-wide blocks of `out`, pairing
// each with all tiles to its left. Every unordered pair is touched exactly once, so the panel
// order below moves the same data as a tile-at-a-time walk.
template<typename T>
inline void four_step_fused_sweep_range(std::complex<T>* out, std::size_t ld, std::size_t n1,
                                        std::size_t m, std::size_t a0, std::size_t a1,
                                        std::complex<T>* stage) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t B = four_step_tblock;
    constexpr std::size_t PT = four_step_panel_tiles(W);
    const auto swap_pair = [&](std::complex<T>* M, std::size_t i0, std::size_t j0) {
        std::complex<T>* const hi = M + i0 * ld + j0;
        std::complex<T>* const lo = M + j0 * ld + i0;
        for (std::size_t r = 0; r < W; ++r) std::copy_n(hi + r * ld, W, stage + r * W);
        four_step_tile_transpose<T>(lo, ld, hi, ld);
        four_step_tile_transpose<T>(stage, W, lo, ld);
    };
    for (std::size_t pa = a0; pa < a1; pa += PT) {
        const std::size_t paE = std::min(pa + PT, a1);
        const std::size_t jdiag = pa * W;
        for (std::size_t q = 0; q < m; ++q) {
            std::complex<T>* const M = out + q * n1;
            for (std::size_t jb = 0; jb < jdiag; jb += B) {
                const std::size_t jE = std::min(jb + B, jdiag);
                for (std::size_t a = pa; a < paE; ++a)
                    for (std::size_t j0 = jb; j0 < jE; j0 += W) swap_pair(M, a * W, j0);
            }
            // The panel's own diagonal block, where the column bound depends on the row.
            for (std::size_t a = pa; a < paE; ++a) {
                const std::size_t i0 = a * W;
                four_step_tile_transpose<T>(M + i0 * ld + i0, ld, M + i0 * ld + i0, ld);
                for (std::size_t j0 = jdiag; j0 < i0; j0 += W) swap_pair(M, i0, j0);
            }
        }
    }
}

[[nodiscard]] inline std::size_t four_step_sweep_lo(std::size_t c, std::size_t nparts,
                                                    std::size_t ntiles) {
    return static_cast<std::size_t>(
        std::llround(std::sqrt(static_cast<double>(c) / static_cast<double>(nparts))
                     * static_cast<double>(ntiles)));
}

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
        four_step_fused_sweep_range(out, ld, n1, m, r0, r1, stage);
    });
}

template<typename T>
void four_step_dft_transpose_fused(std::complex<T>* out, std::size_t n1, std::size_t n2,
                                   std::size_t m, bool is_forward,
                                   const dif_twiddle_set<T>& dtw, T row_scale,
                                   thread_pool* pool) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t ld = n2;
    const std::size_t ntiles = n1 / W;
    const std::size_t bandw = m * W;
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
        // A panel's sweeps run after the panel's own DFTs, and a sweep of tile row `a` reads
        // rows at most a * W + W - 1, which band floor(a / m) <= a already produced. So
        // deferring the sweeps by one panel keeps the dependency and buys the blocking.
        for (std::size_t pa = a0; pa < a1; pa += four_step_panel_tiles(W)) {
            const std::size_t paE = std::min(pa + four_step_panel_tiles(W), a1);
            for (std::size_t a = pa; a < paE; ++a) {
                const std::size_t k0 = a * bandw;
                for (std::size_t k = k0; k < k0 + bandw; ++k)
                    dif_dispatch<T>(is_forward, out + k * n1, out + k * n1, n1, rsc.buf(0),
                                    rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw, row_scale,
                                    rsc.stride());
            }
            four_step_fused_sweep_range<T>(out, ld, n1, m, pa, paE, stage);
        }
    });
}

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
        for (std::size_t pa = a0; pa < a1; pa += four_step_panel_tiles(W)) {
            const std::size_t paE = std::min(pa + four_step_panel_tiles(W), a1);
            for (std::size_t a = pa; a < paE; ++a)
                for (std::size_t j = a * W; j < (a + 1) * W; ++j)
                    four_step_row_dft_twist<T>(out + j * n2, n2, j, is_forward, dtw, p2_scale,
                                               rsc, hitab, lotab, twist_M, twist_logM);
            four_step_fused_sweep_range<T>(out, ld, n1, m, pa, paE, stage);
        }
    });
}

using large_split = four_step_split;

[[nodiscard]] constexpr bool large_leaf_ok(std::size_t f) {
    return f > 1 && (detail::has_single_bit(f) || is_codelet_supported(f));
}

[[nodiscard]] constexpr large_split choose_large_split(std::size_t N) {
    large_split best{};
    for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
        if (N % n1 != 0) continue;
        const std::size_t n2 = N / n1;
        if (!large_leaf_ok(n1) || !large_leaf_ok(n2)) continue;
        best = {n1, n2};
    }
    return best;
}

// --- threaded law (host-independent; WI-2c threaded grid, same sweep) ---
//
// At nt in {2..32} f64 and {4..8} f32 the crossover sits inside [2^14, 2^15] ELEMENTS for
// both precisions on every measured class, so the floor is element-keyed: the byte floor
// halves with the element size (values are the bracket geo-mids, floor-of-root). It holds
// flat to the knee (nt = 32 f64, nt = 8 f32; both brackets uniform across classes), then
// rises to the cap by the host's pool width P and min-pins there:
//   line(nt) = floor + (cap - floor) * min(nt - knee, span) / span, span = max(P, 2*knee) - knee
// P is a runtime quantity, not a host key, and the span guard (2*knee stand-in when
// P <= knee) is the only unmeasured regime -- there nt <= P <= knee means the rise never
// engages in auto mode, and explicit oversubscription interpolates to the cap by 2*knee.
// The f32 cap bracket [1, 2] MiB measured identical on all three classes; f64 measured
// [1, 2] MiB on rome and genoa against [2, 4] MiB on icelake, and the shared cap takes
// the lower geo-mid on a cost-asymmetry argument: mis-admission there costs at most
// 1.105x (icelake f64 nt=64, 128K elems) while mis-rejection would cost 1.34-1.76x
// (rome 0.744, genoa 0.568 at 2 MiB bytes). nt = 2 f32 sits above the plateau on every
// class (brackets [64K, 512K] elems) and pins the cap value, the union geo-mid.
// OFF rows the shape cannot hold, all race-recovered at effort::measure/automatic:
// genoa's non-monotone nt=8 f32 notch (four_step 0.65 at 32K, dif 1.756 at 64K, 1.183 at
// 128K, four_step 0.25 at 256K; no monotone law admits 32K without 64K), icelake's
// nt=32 f32 interpolation point (line 92681 elems vs the [128K, 256K] bracket, one mis-
// admitted rung at 1.35x), the icelake f64 cap compromise above, nt=2 f32's icelake
// rejection at 128K elems (1.20x) and rome admission at 256K (1.06x).
inline constexpr std::size_t kLargeRouteThreadFloorF64Bytes = 370727;  // [256, 512] KiB
inline constexpr std::size_t kLargeRouteThreadFloorF32Bytes = 185363;  // [128, 256] KiB
inline constexpr std::size_t kLargeRouteThreadKneeF64Nt = 32;
inline constexpr std::size_t kLargeRouteThreadKneeF32Nt = 8;
inline constexpr std::size_t kLargeRouteThreadCapBytes = 1482910;  // [1, 2] MiB

[[nodiscard]] constexpr std::size_t large_route_threaded_bytes(std::size_t elem_bytes,
                                                               std::size_t nthreads,
                                                               std::size_t pool_width) {
    const bool f64 = elem_bytes == 16;
    if (!f64 && nthreads == 2) return kLargeRouteThreadCapBytes;
    const std::size_t knee = f64 ? kLargeRouteThreadKneeF64Nt : kLargeRouteThreadKneeF32Nt;
    const std::size_t floor =
        f64 ? kLargeRouteThreadFloorF64Bytes : kLargeRouteThreadFloorF32Bytes;
    if (nthreads <= knee) return floor;
    const std::size_t span =
        (pool_width > 2 * knee ? pool_width : 2 * knee) - knee;
    return std::min(floor + (kLargeRouteThreadCapBytes - floor) * (nthreads - knee) / span,
                    kLargeRouteThreadCapBytes);
}

// --- serial lines: probed on the host, not keyed on it ---
//
// The WI-2c sweep (beat-standings runs/wi2c-{rome,icelake,genoa}-da86113, reduced in
// wi2c-large1d.md) puts the serial f64 crossover inside [8, 16] MiB on rome and icelake
// but [4, 8] MiB on genoa, and the f32 lower edge inside [8, 16] / [16, 32] / [64, 128]
// MiB respectively. No constant key reads that spread: rome and genoa carry the same
// 4 MiB of L3 per physical core and their f64 crossovers move in opposite directions
// from the shipped 12 MiB. So the serial line is MEASURED where it runs: a lazily
// engaged probe (plan_impl<T>::probe_large_route_serial) races forced dif and
// four_step_large serial plans on a short ladder around the prior and caches the
// crossover once per process and precision. The two constants below are the ladder's
// starting prior AND the whole answer when the probe is disabled
// (ADM_LARGE_ROUTE_PROBE=0) or throws: the pre-probe shipped values, so a host on which
// the probe cannot run keeps the answer it always had.
inline constexpr std::size_t kLargeRouteSerialF64Bytes = std::size_t{12} << 20;
inline constexpr std::size_t kLargeRouteSerialF32Bytes = (std::size_t{16} << 20) - 1;

// Ladders. Every measured class bracket is covered by two adjacent rungs, and pow2 gives
// both forced plans for free (large split and fused serial shape). f64 {4, 8, 16} MiB
// covers [4, 16] MiB brackets. f32 stops at {8, 16, 32} MiB on purpose: genoa's
// [64, 128] MiB bracket is reached by the all-dif answer (2x top) instead of a 64 MiB
// rung, which would cost ~350 ms alone for zero admittance difference at any measured
// rung (64 MiB comes out dif-side either way, 128 MiB admitted either way).
inline constexpr std::size_t kLargeRouteProbeLadderF64[] = {std::size_t{4} << 20,
                                                            std::size_t{8} << 20,
                                                            std::size_t{16} << 20};
inline constexpr std::size_t kLargeRouteProbeLadderF32[] = {std::size_t{8} << 20,
                                                            std::size_t{16} << 20,
                                                            std::size_t{32} << 20};
inline constexpr std::size_t kLargeRouteProbeLadderCount = 3;

// Side rule. A rung keeps the dif side unless four_step CLEARLY wins: a ratio within
// 1 +- kLargeRouteProbeTieTol is a tie, and a tie goes dif-side -- the sweep's own spread
// is 1.04-1.54% and its tie-adjacent serial cells read 0.996-1.034, so only a wider keep
// window would lose them. The probe re-samples once any rung whose first-pass ratio
// lands inside the wider kLargeRouteProbeAmbigTol band, min-accumulated: a single timed
// sample on a wandering-clock host mis-reads a true 0.88 as 0.96 one time in four
// (ccmlin075 under load, 2026-09-15), and the re-sample round pulls it back to the
// per-arm min.
inline constexpr double kLargeRouteProbeTieTol = 0.05;
inline constexpr double kLargeRouteProbeAmbigTol = 0.10;

[[nodiscard]] constexpr bool large_route_rung_dif_side(double fs_over_dif) {
    return fs_over_dif > 1.0 - kLargeRouteProbeTieTol;
}

// floor(sqrt(lo * hi)). Ladder rungs are <= 32 MiB as shipped, so lo * hi <= 2^50 and the
// division-form corrections below cannot wrap.
[[nodiscard]] constexpr std::size_t large_route_geo_mid(std::size_t lo, std::size_t hi) {
    if (lo > hi) { const std::size_t t = lo; lo = hi; hi = t; }
    const std::size_t x = lo * hi;
    std::size_t r = hi;  // >= sqrt(x) because lo <= hi; the iterates decrease monotonically
    while (r > x / r) r = (r + x / r) / 2;
    while (r + 1 <= x / (r + 1)) ++r;
    return r;
}

// Byte line from one walked ladder segment: ladder_bytes ascending, dif_side per visited
// rung, and the walk (probe_large_route_serial) guarantees the segment holds the sign
// change or reaches a ladder edge. Answers:
//   every rung dif-side     -> 2 * top        (dif holds the ladder: one octave above)
//   every rung four_step    -> bottom / 2     (four_step holds the ladder)
//   else                    -> geo-mid of the adjacent (dif-side, four_step-side) pair
[[nodiscard]] constexpr std::size_t
large_route_serial_from_ladder(const std::size_t* ladder_bytes, const bool* dif_side,
                               std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (dif_side[i]) continue;
        return i == 0 ? ladder_bytes[0] / 2
                      : large_route_geo_mid(ladder_bytes[i - 1], ladder_bytes[i]);
    }
    return 2 * ladder_bytes[count - 1];
}

// Below this byte count no probed line can ever admit (the probe answers at least
// ladder-bottom/2), so the router consults the probe only above it: small plans -- the
// digest cases, the unit sizes, CI's shape sweeps -- never pay or observe the probe.
[[nodiscard]] constexpr std::size_t large_route_serial_min_bytes(std::size_t elem_bytes) {
    return (elem_bytes == 16 ? kLargeRouteProbeLadderF64[0] : kLargeRouteProbeLadderF32[0]) /
           2;
}

[[nodiscard]] constexpr std::size_t large_route_serial_fallback(std::size_t elem_bytes) {
    return elem_bytes == 16 ? kLargeRouteSerialF64Bytes : kLargeRouteSerialF32Bytes;
}

// --- determinism seam (tests) ---
// The probed line is process state, so a test pin must never read it: the resolver
// consults an injected answer FIRST. Tests call set_large_route_serial_override before
// touching the large band and clear it (0) after; the probed-or-fallback answer then
// governs again. Single-threaded test use only -- the library itself only ever reads.
// These are declared, NOT defined inline: the storage must exist exactly once per
// process (src/large_route_probe.cpp), because a header-local static splits into two
// copies across the engine shared library and a -fvisibility=hidden test binary.
[[nodiscard]] std::size_t large_route_serial_override(std::size_t elem_bytes);
void set_large_route_serial_override(std::size_t elem_bytes, std::size_t bytes);
[[nodiscard]] bool large_route_probe_disabled();

// RAII pin for tests: sets the per-precision overrides passed (0 leaves a precision
// untouched) and clears exactly those on scope exit.
struct large_route_serial_override_scope {
    large_route_serial_override_scope(std::size_t f64_bytes, std::size_t f32_bytes)
        : f64_(f64_bytes != 0), f32_(f32_bytes != 0) {
        if (f64_) set_large_route_serial_override(16, f64_bytes);
        if (f32_) set_large_route_serial_override(8, f32_bytes);
    }
    ~large_route_serial_override_scope() {
        if (f64_) set_large_route_serial_override(16, 0);
        if (f32_) set_large_route_serial_override(8, 0);
    }
    large_route_serial_override_scope(const large_route_serial_override_scope&) = delete;
    large_route_serial_override_scope& operator=(const large_route_serial_override_scope&) =
        delete;

private:
    bool f64_, f32_;
};

[[nodiscard]] constexpr bool four_step_large_supported(std::size_t N, std::size_t elem_bytes,
                                                       std::size_t threshold_bytes) {
    return N * elem_bytes > threshold_bytes && choose_large_split(N).valid();
}

template<typename T>
[[nodiscard]] constexpr large_split choose_fused_large_split(std::size_t N) {
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
        best = {n1, n2};
    }
    return best;
}

template<typename T>
[[nodiscard]] constexpr bool four_step_large_fused_shape(std::size_t N) {
    return choose_fused_large_split<T>(N).valid();
}

[[nodiscard]] inline constexpr bool fsl_ws_engaged(const thread_pool* pool) {
    return pool != nullptr && pool->size() > 1;
}

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
    const std::size_t nbands = n1v / W + (n1v < n1 ? 1 : 0);
    parallel_for(pool, nbands, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
        soa_scratch<T, 4> rsc(n2);
        const auto stage = make_aligned_buffer<std::complex<T>>(W * n2);
        for (std::size_t b = b0; b < b1; ++b) {
            const std::size_t j0 = b * W;
            if (j0 >= n1v) {
                for (std::size_t j = n1v; j < n1; ++j) {
                    for (std::size_t i = 0; i < n2; ++i) stage[i] = in[i * n1 + j];
                    four_step_row_dft_twist<T>(stage.get(), n2, j, is_forward, dtw, p2_scale,
                                               rsc, hitab, lotab, twist_M, twist_logM);
                    std::copy_n(stage.get(), n2, ws + j * n2);
                }
                continue;
            }
            for (std::size_t i0 = 0; i0 < n2v; i0 += W)
                four_step_tile_transpose<T>(in + i0 * n1 + j0, n1, stage.get() + i0, n2);
            for (std::size_t i = n2v; i < n2; ++i)
                for (std::size_t k = 0; k < W; ++k) stage[k * n2 + i] = in[i * n1 + j0 + k];
            for (std::size_t k = 0; k < W; ++k) {
                four_step_row_dft_twist<T>(stage.get() + k * n2, n2, j0 + k, is_forward, dtw,
                                           p2_scale, rsc, hitab, lotab, twist_M, twist_logM);
                std::copy_n(stage.get() + k * n2, n2, ws + (j0 + k) * n2);
            }
        }
    });
}

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
                four_step_tile_transpose<T>(ws + j0 * n2 + kb0, n2, stage.get() + j0, n1);
            for (std::size_t j = n1v; j < n1; ++j)
                for (std::size_t c = 0; c < W; ++c) stage[c * n1 + j] = ws[j * n2 + kb0 + c];
            for (std::size_t c = 0; c < W; ++c)
                dif_dispatch<T>(is_forward, stage.get() + c * n1, stage.get() + c * n1, n1,
                                rsc.buf(0), rsc.buf(1), rsc.buf(2), rsc.buf(3), dtw,
                                row_scale, rsc.stride());
            for (std::size_t j1 = 0; j1 < n1v; j1 += W)
                four_step_tile_transpose<T>(stage.get() + j1, n1, out + j1 * n2 + kb0, n2);
            for (std::size_t j = n1v; j < n1; ++j)
                for (std::size_t c = 0; c < W; ++c) out[j * n2 + kb0 + c] = stage[c * n1 + j];
        }
    });
}

template<typename T>
struct four_step_large_plan {
    std::size_t n1 = 0, n2 = 0;
    bool is_forward = true;
    dif_twiddle_set<T> dtw_n2;
    dif_twiddle_set<T> dtw_n1;
    std::vector<std::complex<T>> lotab;
    std::vector<std::complex<T>> hitab;
    std::size_t twist_M = 1, twist_logM = 0;
    four_step_large_plan(std::size_t N, bool fwd) : is_forward(fwd) {
        large_split sp = choose_fused_large_split<T>(N);
        if (!sp.valid()) sp = choose_large_split(N);
        n1 = sp.n1;
        n2 = sp.n2;
        dtw_n2 = build_dif_twiddle_set<T>(n2, nullptr);
        dtw_n1 = build_dif_twiddle_set<T>(n1, nullptr);
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

    // Between the two passes the buffer is transposed back: defer_rot (n2 % n1 == 0,
    // n2 != n1, so n2 > n1) walks the square n1-side tiles, else one rectangular
    // transpose; the two call sites hand the rectangle's axes in opposite orders.
    void transpose_between_passes(std::complex<T>* out, std::size_t ta, std::size_t tb,
                                  bool defer_rot, thread_pool* pool) const {
        if (defer_rot) {
            for (std::size_t q = 0; q < n2 / n1; ++q)
                four_step_square_transpose_inplace<T>(out + q * n1, n2, n1, pool);
        } else {
            four_step_transpose_inplace<T>(out, ta, tb, pool);
        }
    }

    void execute(const std::complex<T>* in, std::complex<T>* out, T fct,
                 thread_pool* pool = nullptr) const {
        if (fsl_ws_engaged(pool)) {
            execute_ws(in, out, fct, pool);
            return;
        }
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == default_transform_fct<T>(is_forward, N));
        constexpr std::size_t Wv = xsimd::batch<T>::size;
        constexpr std::size_t RB = four_step_tblock;

        if (in == out) {
            four_step_transpose_inplace<T>(out, n2, n1, pool);
        } else {
            const std::size_t n2v = n2 & ~(Wv - 1);
            const std::size_t nbands = (n2 + RB - 1) / RB;
            const bool stream = four_step_stream_ok<T>(out, n2, N * sizeof(std::complex<T>));
            parallel_for(pool, nbands, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
                const std::size_t i0 = std::min(b0 * RB, n2v);
                const std::size_t iE = std::min(b1 * RB, n2v);
                if (iE <= i0) return;
                if (stream) {
                    four_step_transpose_band<T, true>(in, out, n1, n1, n2, i0, iE);
                    stream_store_fence();  // the next pass reads what this band just wrote
                } else {
                    four_step_transpose_band<T>(in, out, n1, n1, n2, i0, iE);
                }
            });
            four_step_transpose_remainder<T>(in, out, n1, n1, n2);
        }

        const T p2_scale = (!is_forward && default_scale) ? T(1) / static_cast<T>(n2) : T(1);
        const bool defer_rot = (n2 % n1 == 0) && (n2 != n1);
        if (n2 % n1 == 0 && n1 % Wv == 0) {
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

            transpose_between_passes(out, n1, n2, defer_rot, pool);
        }

        const T row_scale = default_scale ? (is_forward ? T(1) : T(1) / static_cast<T>(n1)) : fct;
        if (n2 % n1 == 0 && n1 % Wv == 0) {
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

            transpose_between_passes(out, n2, n1, defer_rot, pool);
        }
    }

    void execute_ws(const std::complex<T>* in, std::complex<T>* out, T fct,
                    thread_pool* pool = nullptr) const {
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == default_transform_fct<T>(is_forward, N));
        const T p2_scale = (!is_forward && default_scale) ? T(1) / static_cast<T>(n2) : T(1);
        const T row_scale = default_scale ? (is_forward ? T(1) : T(1) / static_cast<T>(n1)) : fct;
        auto ws_buf = make_aligned_buffer<std::complex<T>>(N);
        std::complex<T>* const ws = ws_buf.get();
        fsl_ws_s1<T>(in, ws, n1, n2, is_forward, dtw_n2, p2_scale, hitab.data(),
                     lotab.data(), twist_M, twist_logM, pool);
        fsl_ws_s2<T>(ws, out, n1, n2, is_forward, dtw_n1, row_scale, pool);
    }
};

}
}
