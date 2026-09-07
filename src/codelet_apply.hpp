#pragma once

#include <complex>
#include <cstddef>
#include "admiral/detail/cxx_compat.hpp"

#ifndef ADM_ND_FLATROW
#define ADM_ND_FLATROW 0
#endif

#include "admiral/detail/codelet.hpp"
#include "admiral/detail/math.hpp"
#if ADM_ND_FLATROW
#include "admiral/detail/flat_row.hpp"
#endif
#include "admiral/detail/simd_swizzle.hpp"
#include "admiral/detail/macros.hpp"

namespace admiral {
namespace detail {

template<typename T, bool Forward>
struct aos_sink {
    std::complex<T>* out;
    template<typename V>
    inline void operator()(std::size_t p, V outr, V outi) const {
        const V im = Forward ? outi : -outi;
        if constexpr (std::is_same_v<V, T>) out[p] = std::complex<T>(outr, im);
        else aos_interleave<T, V>(reinterpret_cast<T*>(out + p), outr, im);
    }
};

template<unsigned N, typename T, bool Forward>
void codelet_apply(const std::complex<T>* in, std::complex<T>* out) {
    alignas(xsimd::batch<T>::arch_type::alignment()) T buf[4 * N];
    T* xre = buf;
    T* xim = buf + N;
    T* yre = buf + 2 * N;
    T* yim = buf + 3 * N;

    constexpr std::size_t W = xsimd::batch<T>::size;

    if constexpr (N < 64) {
        const T* s = reinterpret_cast<const T*>(in);
        for (std::size_t i = 0; i < N; ++i) {
            xre[i] = s[2 * i];
            if constexpr (Forward) xim[i] =  s[2 * i + 1];
            else                   xim[i] = -s[2 * i + 1];
        }
        kernel<N, T, true>::apply_sink(xre, xim, 1, yre, yim, aos_sink<T, Forward>{out});
        return;
    }

    const T* src = reinterpret_cast<const T*>(in);
    std::size_t i = 0;
    for (; i + W <= N; i += W) {
        xsimd::batch<T> re, im;
        aos_deinterleave(src + 2 * i, re, im);
        re.store_unaligned(xre + i);
        im.store_unaligned(xim + i);
    }
    poet::static_for<1, detail::bit_width(W)>([&](auto S) {
        constexpr std::size_t Wt = W >> S;
        using Vt = xsimd::make_sized_batch_t<T, Wt>;
        if constexpr (Wt >= 2 && !std::is_void_v<Vt>) {
            if (i + Wt <= N) {
                Vt re, im;
                aos_deinterleave<T, Vt>(src + 2 * i, re, im);
                re.store_unaligned(xre + i);
                im.store_unaligned(xim + i);
                i += Wt;
            }
        }
    });
    // Every arm above advances i only while i + w <= N, so i <= N holds here; gcc 13 at
    // x86-64-v2 loses that under -Waggressive-loop-optimizations and reads a wrap into the
    // scalar tail (false positive; the install job is -Werror). Counting the bounded copy out
    // from N - i keeps the analysis.
    const std::size_t tail = N - i;
    for (std::size_t j = 0; j < tail; ++j) {
        xre[i + j] = in[i + j].real();
        xim[i + j] = in[i + j].imag();
    }

    if constexpr (!Forward) for (std::size_t k = 0; k < N; ++k) xim[k] = -xim[k];
    kernel<N, T, true>::apply_sink(xre, xim, 1, yre, yim, aos_sink<T, Forward>{out});
}

// Direction is a pointer swap, not a template axis, wherever a codelet's gather, scale and scatter
// are direction-free: F_inv(x) = swap(F_fwd(swap(x))), so kernel_batched<N, T, false> is already
// the forward kernel over swapped re/im. Both leaves then share ONE body, the swap is four selects
// made once per call, and the catalog emits half the code for that family. The 1-D engines keep
// Forward compile-time, where it really does change the butterflies' signs.
template<unsigned N, typename T, typename V>
struct dir_swap {
    const V *a0, *a1;
    V *b0, *b1;
    ADM_ALWAYS_INLINE dir_swap(bool fwd, const V* x0, const V* x1, V* y0, V* y1)
        : a0(fwd ? x0 : x1), a1(fwd ? x1 : x0), b0(fwd ? y0 : y1), b1(fwd ? y1 : y0) {}
    ADM_ALWAYS_INLINE void apply() const {
        kernel_batched<N, T, true, V>::apply(a0, a1, 1, b0, b1);
    }
};

// One gather block: W lines by Cols columns, deinterleaved, transposed into the kernel's column
// arrays. Cols is compile-time because the mask is; j0 is a runtime argument so the full blocks can
// share one copy of this text instead of one per block.
template<std::size_t Cols, typename T, typename V>
ADM_ALWAYS_INLINE void many_gather_block(const T* ibase, std::size_t in_stride, std::size_t j0,
                                         V* re, V* im) {
    constexpr std::size_t W = V::size;
    V rb[W], ib[W];
    for (std::size_t l = 0; l < W; ++l)
        aos_deinterleave_masked<(2 * Cols > W), T>(ibase + l * 2 * in_stride + 2 * j0,
                                                   rb[l], ib[l], aos_ct_masks<Cols, T>{});
    xsimd::transpose(rb, rb + W);
    xsimd::transpose(ib, ib + W);
    poet::static_for<0, Cols>([&](auto J) {
        re[j0 + J] = rb[J];
        im[j0 + J] = ib[J];
    });
}

template<std::size_t Cols, typename T, typename V>
ADM_ALWAYS_INLINE void many_scatter_block(T* obase, std::size_t out_stride, std::size_t j0,
                                          const V* yr, const V* yi, V f) {
    constexpr std::size_t W = V::size;
    V tr[W], ti[W];
    poet::static_for<0, W>([&](auto J) {
        constexpr std::size_t j = J;
        const std::size_t k = (j < Cols) ? j0 + j : 0;
        tr[J] = yr[k] * f;
        ti[J] = yi[k] * f;
    });
    xsimd::transpose(tr, tr + W);
    xsimd::transpose(ti, ti + W);
    for (std::size_t l = 0; l < W; ++l)
        aos_interleave_prefix<Cols, T>(obase + l * 2 * out_stride + 2 * j0, tr[l], ti[l]);
}

// Transposing the RAW interleaved rows lands re and im on alternate output slots, so one W x W
// transpose covers W / 2 complex columns with no deinterleave at all: the split gather's per-row
// shuffle pair and one of its two plane transposes both disappear. Per W columns of an eight-row
// block at W = 8 that is 48 shuffles instead of 64, and the load and store counts are unchanged.
template<std::size_t Cols, bool Neg, typename T, typename V>
ADM_ALWAYS_INLINE void many_gather_x(const T* ibase, std::size_t in_stride, std::size_t j0,
                                     V* re, V* im) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t G = W / 2;
    constexpr std::size_t H = (Cols + G - 1) / G;
    static_assert(Cols >= 1 && Cols <= W);
    using arch = typename V::arch_type;
    V t[H][W];
    for (std::size_t l = 0; l < W; ++l) {
        const T* p = ibase + l * 2 * in_stride + 2 * j0;
        poet::static_for<0, H>([&](auto h) {
            constexpr std::size_t lanes = 2 * Cols - h * W < W ? 2 * Cols - h * W : W;
            if constexpr (lanes == W) t[h][l] = V::load_unaligned(p + h * W);
            else t[h][l] = V::load(p + h * W,
                                   xsimd::make_batch_bool_constant<T, lane_lt<lanes>, arch>(),
                                   xsimd::unaligned_mode{});
        });
    }
    poet::static_for<0, H>([&](auto h) { xsimd::transpose(t[h], t[h] + W); });
    poet::static_for<0, Cols>([&](auto K) {
        constexpr std::size_t h = K / G, k = 2 * (K % G);
        re[j0 + K] = t[h][k];
        im[j0 + K] = Neg ? -t[h][k + 1] : t[h][k + 1];
    });
}

template<std::size_t Cols, typename T, typename V>
ADM_ALWAYS_INLINE void many_scatter_x(T* obase, std::size_t out_stride, std::size_t j0,
                                      const V* yr, const V* yi, V fr, V fi) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t G = W / 2;
    constexpr std::size_t H = (Cols + G - 1) / G;
    static_assert(Cols >= 1 && Cols <= W);
    using arch = typename V::arch_type;
    V t[H][W];
    poet::static_for<0, H>([&](auto h) {
        // MSVC 19.44 counts the inner lambda's read of the captured h as a closure read and
        // refuses to fold it (P2280, C2131); a constexpr local of this lambda needs no capture.
        // MSVC then reports the fold as leaving hb unreferenced (C4189), so it is maybe_unused.
        [[maybe_unused]] constexpr std::size_t hb = h * G;
        poet::static_for<0, G>([&](auto K) {
            constexpr std::size_t j = hb + K;
            constexpr std::size_t k = j < Cols ? j : 0;
            t[h][2 * K] = yr[j0 + k] * fr;
            t[h][2 * K + 1] = yi[j0 + k] * fi;
        });
        xsimd::transpose(t[h], t[h] + W);
    });
    for (std::size_t l = 0; l < W; ++l) {
        T* p = obase + l * 2 * out_stride + 2 * j0;
        poet::static_for<0, H>([&](auto h) {
            constexpr std::size_t lanes = 2 * Cols - h * W < W ? 2 * Cols - h * W : W;
            if constexpr (lanes == W) t[h][l].store_unaligned(p + h * W);
            else t[h][l].store(p + h * W,
                               xsimd::make_batch_bool_constant<T, lane_lt<lanes>, arch>(),
                               xsimd::unaligned_mode{});
        });
    }
}

// A block iteration keeps 2W batches live: the W-line gather and the transpose it feeds. The
// register file divided by that is how many blocks fit, which over x86-64 through x86-64-v4 is
// 1 or 2 at float and 2 or 4 at double. Only N / W blocks are full; the N % W remainder keeps its
// own compile-time width and is emitted once.
template<typename V>
inline constexpr std::size_t kManyUnroll =
    poet::vector_register_count() >= 2 * V::size
        ? poet::vector_register_count() / (2 * V::size)
        : 1;

// The fused gather holds 2W batches live per block, so the whole codelet needs 2 * ceil(N / W) * W
// of them, which is the same 2N <= regs bound flat_leaf uses. Past it gcc runs out of vector
// registers and lowers xsimd::transpose through the stack: at N = 32, W = 8 the shuffles fall
// 386 -> 2 and the stack traffic rises 135 -> 366 memory operations, which costs 19% on 2d_32.
template<unsigned N, typename V>
inline constexpr bool kManyXpose =
    2 * V::size * ((N + V::size - 1) / V::size) <= poet::vector_register_count();

// The fused gather is one transpose per row group where the split gather priced two plane
// transposes plus a per-row deinterleave pair, so at N = 2/4 the block costs less than the
// scalar per-line fallback this exclusion was written for.
template<unsigned N, typename V>
inline constexpr bool kManyBlocked =
    kManyXpose<N, V> || (N != 2 && N != 4);

// ADM_NOINLINE is load-bearing: `fwd` arrives as a constant from each leaf wrapper, so a compiler
// free to inline this body would fold it and re-specialise, putting back the copy the merge cut.
template<unsigned N, typename T>
ADM_NOINLINE void codelet_many_body(const std::complex<T>* in, std::complex<T>* out,
                                    std::size_t nlines, std::size_t in_stride,
                                    std::size_t out_stride, T fct, bool fwd) {
    using V = xsimd::batch<T>;
    constexpr std::size_t W = V::size;
    constexpr std::size_t kFull = N / W;
    constexpr std::size_t kRem = N % W;
    constexpr std::size_t kUnroll = kManyUnroll<V>;

    std::size_t r = 0;
    if constexpr (kManyBlocked<N, V>) {
        const V f(fct);
        V re[N], im[N], yr[N], yi[N];
        const dir_swap<N, T, V> dir(fwd, re, im, yr, yi);
        for (; r + W <= nlines; r += W) {
            const T* ibase = reinterpret_cast<const T*>(in + r * in_stride);
            T* obase = reinterpret_cast<T*>(out + r * out_stride);

            if constexpr (kManyXpose<N, V>) {
                poet::dynamic_for<kUnroll>(kFull, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
                    many_gather_x<W, false, T, V>(ibase, in_stride, b * W, re, im);
                });
                if constexpr (kRem != 0)
                    many_gather_x<kRem, false, T, V>(ibase, in_stride, kFull * W, re, im);

                dir.apply();

                poet::dynamic_for<kUnroll>(kFull, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
                    many_scatter_x<W, T, V>(obase, out_stride, b * W, yr, yi, f, f);
                });
                if constexpr (kRem != 0)
                    many_scatter_x<kRem, T, V>(obase, out_stride, kFull * W, yr, yi, f, f);
            } else {
                poet::dynamic_for<kUnroll>(kFull, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
                    many_gather_block<W, T, V>(ibase, in_stride, b * W, re, im);
                });
                if constexpr (kRem != 0)
                    many_gather_block<kRem, T, V>(ibase, in_stride, kFull * W, re, im);

                dir.apply();

                poet::dynamic_for<kUnroll>(kFull, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
                    many_scatter_block<W, T, V>(obase, out_stride, b * W, yr, yi, f);
                });
                if constexpr (kRem != 0)
                    many_scatter_block<kRem, T, V>(obase, out_stride, kFull * W, yr, yi, f);
            }
        }
    }
    const bool unit = (fct == T(1));
    for (; r < nlines; ++r) {
        if (fwd) codelet_apply<N, T, true>(in + r * in_stride, out + r * out_stride);
        else     codelet_apply<N, T, false>(in + r * in_stride, out + r * out_stride);
        if (!unit) scale_inplace(out + r * out_stride, N, fct);
    }
}

// The static form of the same blocks: compile-time width, the direction back on the template head
// and folded into the sign of the imaginary gather rather than into a pointer swap.
template<unsigned N, typename T, bool Forward>
void codelet_many_static(const std::complex<T>* in, std::complex<T>* out,
                         std::size_t nlines, std::size_t in_stride,
                         std::size_t out_stride, T fct) {
    using V = xsimd::batch<T>;
    constexpr std::size_t W = V::size;
    constexpr std::size_t kBlocks = (N + W - 1) / W;

    std::size_t r = 0;
    if constexpr (kManyBlocked<N, V>) {
        const V fr(fct), fi(Forward ? fct : -fct);
        for (; r + W <= nlines; r += W) {
            const T* ibase = reinterpret_cast<const T*>(in + r * in_stride);
            T* obase = reinterpret_cast<T*>(out + r * out_stride);
            V re[N], im[N], yr[N], yi[N];

#if ADM_ND_FLATROW
            if constexpr (kFlatRow<N, V>) {
                const V f(fct);
                poet::static_for<0, W>([&](auto L) {
                    flat_row_apply<N, T, V, Forward>(
                        ibase + L * 2 * in_stride, obase + L * 2 * out_stride, f);
                });
            } else
#endif
            if constexpr (kManyXpose<N, V>) {
                poet::static_for<0, kBlocks>([&](auto B) {
                    constexpr std::size_t j0 = B * W;
                    constexpr std::size_t cols = (N - j0 < W) ? N - j0 : W;
                    many_gather_x<cols, !Forward, T, V>(ibase, in_stride, j0, re, im);
                });

                kernel_batched<N, T, true, V>::apply(re, im, 1, yr, yi);

                poet::static_for<0, kBlocks>([&](auto B) {
                    constexpr std::size_t j0 = B * W;
                    constexpr std::size_t cols = (N - j0 < W) ? N - j0 : W;
                    many_scatter_x<cols, T, V>(obase, out_stride, j0, yr, yi, fr, fi);
                });
            } else {
                poet::static_for<0, kBlocks>([&](auto B) {
                    constexpr std::size_t j0 = B * W;
                    constexpr std::size_t cols = (N - j0 < W) ? N - j0 : W;
                    V rb[W], ib[W];
                    for (std::size_t l = 0; l < W; ++l)
                        aos_deinterleave_masked<(2 * cols > W), T>(
                            ibase + l * 2 * in_stride + 2 * j0, rb[l], ib[l],
                            aos_ct_masks<cols, T>{});
                    xsimd::transpose(rb, rb + W);
                    xsimd::transpose(ib, ib + W);
                    poet::static_for<0, cols>([&](auto J) {
                        re[j0 + J] = rb[J];
                        im[j0 + J] = Forward ? ib[J] : -ib[J];
                    });
                });

                kernel_batched<N, T, true, V>::apply(re, im, 1, yr, yi);

                poet::static_for<0, kBlocks>([&](auto B) {
                    constexpr std::size_t j0 = B * W;
                    constexpr std::size_t cols = (N - j0 < W) ? N - j0 : W;
                    V tr[W], ti[W];
                    poet::static_for<0, W>([&](auto J) {
                        constexpr std::size_t j = J;
                        constexpr std::size_t k = (j < cols) ? j0 + j : 0;
                        tr[J] = yr[k] * fr;
                        ti[J] = yi[k] * fi;
                    });
                    xsimd::transpose(tr, tr + W);
                    xsimd::transpose(ti, ti + W);
                    for (std::size_t l = 0; l < W; ++l)
                        aos_interleave_prefix<cols, T>(obase + l * 2 * out_stride + 2 * j0,
                                                       tr[l], ti[l]);
                });
            }
        }
    }
    const bool unit = (fct == T(1));
    for (; r < nlines; ++r) {
        codelet_apply<N, T, Forward>(in + r * in_stride, out + r * out_stride);
        if (!unit) scale_inplace(out + r * out_stride, N, fct);
    }
}

template<unsigned N, typename T, bool Forward>
void codelet_apply_many_oop(const std::complex<T>* in, std::complex<T>* out,
                            std::size_t nlines, std::size_t in_stride,
                            std::size_t out_stride, T fct) {
    if constexpr (N / xsimd::batch<T>::size >= kManyRollMinBlocks)
        codelet_many_body<N, T>(in, out, nlines, in_stride, out_stride, fct, Forward);
    else
        codelet_many_static<N, T, Forward>(in, out, nlines, in_stride, out_stride, fct);
}


// In place is the out-of-place body with one buffer. Every block reads all W lines of its own
// column range before the kernel runs and writes only that same range, so no write can precede a
// read it feeds, and the residual loop already transforms line r onto itself. Forwarding costs one
// extra stride argument and removes a 54 KB near-duplicate of the body above from every catalog TU.
template<unsigned N, typename T, bool Forward>
void codelet_apply_many(std::complex<T>* data, std::size_t nlines, std::size_t stride, T fct) {
    codelet_apply_many_oop<N, T, Forward>(data, data, nlines, stride, stride, fct);
}


// The fewer-than-W-columns remainder, out of line and rolled. It runs once per call over fewer
// than W columns, so unrolling its 2N scalar gathers and scatters buys nothing and doubles the
// size of the function the hot loop lives in.
template<unsigned N, typename T>
ADM_NOINLINE void col_codelet_tail(const std::complex<T>* in, std::size_t in_inner,
                                   std::complex<T>* out, std::size_t out_inner,
                                   std::size_t c, std::size_t bc, T scale, bool fwd) {
    using V = xsimd::batch<T>;
    constexpr std::size_t W = V::size;
    alignas(xsimd::batch<T>::arch_type::alignment()) T sre[W];
    alignas(xsimd::batch<T>::arch_type::alignment()) T sim[W];
    V xre[N], xim[N], yre[N], yim[N];
    // Carry the direction on the data, not on the pointers: feed the forward kernel
    // (xre, gi*xim), gi = fwd ? 1 : -1, and unfold the output conjugation in the scale.
    const V gi(fwd ? T(1) : T(-1));
    const T si = fwd ? scale : -scale;
    for (std::size_t p = 0; p < N; ++p) {
        for (std::size_t l = 0; l < bc; ++l) {
            sre[l] = in[p * in_inner + c + l].real();
            sim[l] = in[p * in_inner + c + l].imag();
        }
        for (std::size_t l = bc; l < W; ++l) { sre[l] = T(0); sim[l] = T(0); }
        xre[p] = V::load_aligned(sre);
        xim[p] = gi * V::load_aligned(sim);
    }
    kernel_batched<N, T, true, V>::apply(xre, xim, 1, yre, yim);
    for (std::size_t p = 0; p < N; ++p) {
        yre[p].store_aligned(sre);
        yim[p].store_aligned(sim);
        for (std::size_t l = 0; l < bc; ++l)
            out[p * out_inner + c + l] = std::complex<T>(sre[l] * scale, sim[l] * si);
    }
}

// The column gather and scatter are element-wise and shuffle-bound (one vpermd per lane, all on
// the same port), so unrolling them buys no instruction-level parallelism; what it buys is loop
// overhead amortised, and what it costs is code. Block them at the register file divided by the
// four batches an iteration keeps live: that amortises the branch without spilling, and it is
// where poet's dynamic_for puts the boundary. N is a constant here, so the block count and the
// tail both resolve at compile time.
inline constexpr std::size_t kColUnroll =
    poet::vector_register_count() >= 4 ? poet::vector_register_count() / 4 : 1;

// ADM_NOINLINE is load-bearing: `fwd` arrives as a constant from each leaf wrapper, so a compiler
// free to inline this body would fold it and re-specialise, putting back the copy the merge cut.
template<unsigned N, typename T>
ADM_NOINLINE void col_codelet_body(const std::complex<T>* in, std::size_t in_inner,
                                   std::complex<T>* out, std::size_t out_inner,
                                   std::size_t ncols, T scale, bool fwd) {
    using V = xsimd::batch<T>;
    constexpr std::size_t W = V::size;
    const V sc(scale);
    V xre[N], xim[N], yre[N], yim[N];
    // Carry the direction on the data, not on four runtime-selected pointers: feed the
    // forward kernel (xre, gi*xim), gi = fwd ? 1 : -1, and unfold the output conjugation
    // in the scatter's scale (si = fwd ? scale : -scale). The plane arrays are then named
    // unconditionally, so SRA can promote them and kernel_batched<N>::apply can inline.
    const V gi(fwd ? T(1) : T(-1));
    const V si(fwd ? scale : -scale);
    std::size_t c = 0;
    for (; c + W <= ncols; c += W) {
        poet::dynamic_for<kColUnroll>(std::size_t(N), [&](std::size_t p) ADM_LAMBDA_ALWAYS_INLINE {
            aos_deinterleave(reinterpret_cast<const T*>(in + p * in_inner + c), xre[p], xim[p]);
            xim[p] *= gi;
        });
        kernel_batched<N, T, true, V>::apply(xre, xim, 1, yre, yim);
        poet::dynamic_for<kColUnroll>(std::size_t(N), [&](std::size_t p) ADM_LAMBDA_ALWAYS_INLINE {
            aos_interleave<T, V>(reinterpret_cast<T*>(out + p * out_inner + c),
                                 yre[p] * sc, yim[p] * si);
        });
    }
    if (c < ncols)
        col_codelet_tail<N, T>(in, in_inner, out, out_inner, c, ncols - c, scale, fwd);
}

template<unsigned N, typename T, bool Forward>
void col_codelet_apply(const std::complex<T>* in, std::size_t in_inner,
                       std::complex<T>* out, std::size_t out_inner, std::size_t ncols,
                       T scale) {
    col_codelet_body<N, T>(in, in_inner, out, out_inner, ncols, scale, Forward);
}

}
}

#include "admiral/detail/undef_macros.hpp"
