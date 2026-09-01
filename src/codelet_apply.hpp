#pragma once

#include <complex>
#include <cstddef>
#include "admiral/detail/cxx_compat.hpp"

#include "admiral/detail/codelet.hpp"
#include "admiral/detail/math.hpp"
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
    for (; i < N; ++i) {
        xre[i] = in[i].real();
        xim[i] = in[i].imag();
    }

    if constexpr (!Forward) for (std::size_t k = 0; k < N; ++k) xim[k] = -xim[k];
    kernel<N, T, true>::apply_sink(xre, xim, 1, yre, yim, aos_sink<T, Forward>{out});
}

template<unsigned N, typename T, bool Forward>
void codelet_apply_many(std::complex<T>* data, std::size_t nlines, std::size_t stride, T fct) {
    using V = xsimd::batch<T>;
    constexpr std::size_t W = V::size;
    constexpr std::size_t kBlocks = (N + W - 1) / W;

    std::size_t r = 0;
    if constexpr (N != 2 && N != 4) {
        const V fr(fct), fi(Forward ? fct : -fct);
        for (; r + W <= nlines; r += W) {
            T* base = reinterpret_cast<T*>(data + r * stride);
            V re[N], im[N], yr[N], yi[N];

            poet::static_for<0, kBlocks>([&](auto B) {
                constexpr std::size_t j0 = B * W;
                constexpr std::size_t cols = (N - j0 < W) ? N - j0 : W;
                V rb[W], ib[W];
                for (std::size_t l = 0; l < W; ++l)
                    aos_deinterleave_masked<(2 * cols > W), T>(
                        base + l * 2 * stride + 2 * j0, rb[l], ib[l], aos_ct_masks<cols, T>{});
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
                    aos_interleave_prefix<cols, T>(base + l * 2 * stride + 2 * j0, tr[l], ti[l]);
            });
        }
    }
    const bool unit = (fct == T(1));
    for (; r < nlines; ++r) {
        std::complex<T>* p = data + r * stride;
        codelet_apply<N, T, Forward>(p, p);
        if (!unit) scale_inplace(p, N, fct);
    }
}

template<unsigned N, typename T, bool Forward>
void codelet_apply_many_oop(const std::complex<T>* in, std::complex<T>* out,
                            std::size_t nlines, std::size_t in_stride,
                            std::size_t out_stride, T fct) {
    using V = xsimd::batch<T>;
    constexpr std::size_t W = V::size;
    constexpr std::size_t kBlocks = (N + W - 1) / W;

    std::size_t r = 0;
    if constexpr (N != 2 && N != 4) {
        const V fr(fct), fi(Forward ? fct : -fct);
        for (; r + W <= nlines; r += W) {
            const T* ibase = reinterpret_cast<const T*>(in + r * in_stride);
            T* obase = reinterpret_cast<T*>(out + r * out_stride);
            V re[N], im[N], yr[N], yi[N];

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
    const bool unit = (fct == T(1));
    for (; r < nlines; ++r) {
        codelet_apply<N, T, Forward>(in + r * in_stride, out + r * out_stride);
        if (!unit) scale_inplace(out + r * out_stride, N, fct);
    }
}

template<unsigned N, typename T, bool Forward>
void col_codelet_apply(const std::complex<T>* in, std::size_t in_inner,
                       std::complex<T>* out, std::size_t out_inner, std::size_t ncols,
                       T scale) {
    using V = xsimd::batch<T>;
    constexpr std::size_t W = V::size;
    const V sc(scale);
    std::size_t c = 0;
    for (; c + W <= ncols; c += W) {
        V xre[N], xim[N], yre[N], yim[N];
        poet::static_for<0, N>([&](auto P) ADM_LAMBDA_ALWAYS_INLINE {
            aos_deinterleave(reinterpret_cast<const T*>(in + P * in_inner + c),
                             xre[P], xim[P]);
        });
        kernel_batched<N, T, Forward>::apply(xre, xim, 1, yre, yim);
        poet::static_for<0, N>([&](auto P) ADM_LAMBDA_ALWAYS_INLINE {
            aos_interleave<T, V>(reinterpret_cast<T*>(out + P * out_inner + c),
                                 yre[P] * sc, yim[P] * sc);
        });
    }
    if (c < ncols) {
        const std::size_t bc = ncols - c;
        alignas(xsimd::batch<T>::arch_type::alignment()) T sre[W];
        alignas(xsimd::batch<T>::arch_type::alignment()) T sim[W];
        V xre[N], xim[N], yre[N], yim[N];
        poet::static_for<0, N>([&](auto P) ADM_LAMBDA_ALWAYS_INLINE {
            for (std::size_t l = 0; l < bc; ++l) {
                sre[l] = in[P * in_inner + c + l].real();
                sim[l] = in[P * in_inner + c + l].imag();
            }
            for (std::size_t l = bc; l < W; ++l) { sre[l] = T(0); sim[l] = T(0); }
            xre[P] = V::load_aligned(sre);
            xim[P] = V::load_aligned(sim);
        });
        kernel_batched<N, T, Forward>::apply(xre, xim, 1, yre, yim);
        poet::static_for<0, N>([&](auto P) ADM_LAMBDA_ALWAYS_INLINE {
            yre[P].store_aligned(sre);
            yim[P].store_aligned(sim);
            for (std::size_t l = 0; l < bc; ++l)
                out[P * out_inner + c + l] =
                    std::complex<T>(sre[l] * scale, sim[l] * scale);
        });
    }
}

}
}

#include "admiral/detail/undef_macros.hpp"
