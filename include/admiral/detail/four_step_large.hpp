#pragma once

// Four-step FFT for out-of-cache N, with fused transpose sweeps. Bailey, FFTs in External or
// Hierarchical Memory, J. Supercomputing 4 (1990) 23.
//
// `execute()` holds no working buffer, so it is re-entrant. ND-shared sub-plans are built with
// nthreads = 1, so a shared plan never nests a `parallel_for`.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>

#include <vector>
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
                xsimd::transpose(re, re + W);
                xsimd::transpose(im, im + W);
                for (std::size_t c = 0; c < W; ++c)
                    aos_interleave<T>(dst + ((j + c) * n2 + i) * 2, re[c], im[c]);
            }
    }
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
            four_step_tile_transpose<T>(lo, ld, hi, ld);
            four_step_tile_transpose<T>(stage, W, lo, ld);
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
        for (std::size_t a = r0; a < r1; ++a)
            four_step_fused_sweep_step(out, ld, n1, m, a, stage);
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
        for (std::size_t a = a0; a < a1; ++a) {
            for (std::size_t j = a * W; j < (a + 1) * W; ++j)
                four_step_row_dft_twist<T>(out + j * n2, n2, j, is_forward, dtw, p2_scale,
                                           rsc, hitab, lotab, twist_M, twist_logM);
            four_step_fused_sweep_step<T>(out, ld, n1, m, a, stage);
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

inline constexpr std::size_t kLargeRouteThreadedByteBudget = std::size_t{2} << 20;
inline constexpr std::size_t kLargeRouteThreadedFloorBytes = std::size_t{512} << 10;

inline constexpr std::size_t kLargeRouteSerialF64Bytes = std::size_t{12} << 20;
inline constexpr std::size_t kLargeRouteSerialF32Bytes = (std::size_t{16} << 20) - 1;

inline constexpr std::size_t kLargeRouteSerialF32MaxBytes = std::size_t{32} << 20;

[[nodiscard]] constexpr std::size_t large_route_threaded_bytes(std::size_t nthreads) {
    return std::max(kLargeRouteThreadedFloorBytes, kLargeRouteThreadedByteBudget / nthreads) - 1;
}

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

    void execute(const std::complex<T>* in, std::complex<T>* out, T fct,
                 thread_pool* pool = nullptr) const {
        if (fsl_ws_engaged(pool)) {
            execute_ws(in, out, fct, pool);
            return;
        }
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == (is_forward ? T(1) : T(1) / static_cast<T>(N)));
        constexpr std::size_t Wv = xsimd::batch<T>::size;
        constexpr std::size_t RB = four_step_tblock;

        if (in == out) {
            four_step_transpose_inplace<T>(out, n2, n1, pool);
        } else {
            const std::size_t n2v = n2 & ~(Wv - 1);
            const std::size_t nbands = (n2 + RB - 1) / RB;
            parallel_for(pool, nbands, N, [&](std::size_t b0, std::size_t b1, std::size_t) {
                const std::size_t i0 = std::min(b0 * RB, n2v);
                const std::size_t iE = std::min(b1 * RB, n2v);
                if (iE > i0) four_step_transpose_band<T>(in, out, n1, n1, n2, i0, iE);
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

            if (defer_rot) {
                for (std::size_t q = 0; q < n2 / n1; ++q)
                    four_step_square_transpose_inplace<T>(out + q * n1, n2, n1, pool);
            } else {
                four_step_transpose_inplace<T>(out, n1, n2, pool);
            }
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

            if (defer_rot) {
                for (std::size_t q = 0; q < n2 / n1; ++q)
                    four_step_square_transpose_inplace<T>(out + q * n1, n2, n1, pool);
            } else {
                four_step_transpose_inplace<T>(out, n2, n1, pool);
            }
        }
    }

    void execute_ws(const std::complex<T>* in, std::complex<T>* out, T fct,
                    thread_pool* pool = nullptr) const {
        const std::size_t N = n1 * n2;
        const bool default_scale = (fct == (is_forward ? T(1) : T(1) / static_cast<T>(N)));
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
