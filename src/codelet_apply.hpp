#pragma once

// Private implementation header, included only by per-N TUs and codelet_dispatch.cpp.
// Defines codelet_apply<N,T,Forward>: deinterleave -> kernel<N>::apply -> reinterleave.
// in == out is in-place; in != out reads `in` (preserved) and writes `out`.
// Un-normalized (does NOT apply 1/N).  N <= CODELET_CATALOG_MAX is guaranteed by routing.

#include <bit>
#include <complex>
#include <cstddef>

#include "admiral/detail/codelet.hpp"
#include "admiral/detail/math.hpp"  // scale_inplace
#include "admiral/detail/simd_swizzle.hpp"

namespace admiral {
namespace detail {

// Output sink for kernel<N>::apply_sink: interleaves the emitted (re,im) pairs
// straight to AoS `out`, with no SoA output round-trip. The inverse's exit conjugation
// rides the sign here (see the conj-trick comment below); chunks arrive with j in
// ascending order, each output index emitted exactly once.
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
    // Exact-sized stack buffer: 4 planar buffers of N elements each (N is a
    // compile-time codelet size <= CODELET_CATALOG_MAX, so this is always stack).
    alignas(xsimd::batch<T>::arch_type::alignment()) T buf[4 * N];
    T* xre = buf;
    T* xim = buf + N;
    T* yre = buf + 2 * N;
    T* yim = buf + 3 * N;

    constexpr std::size_t W = xsimd::batch<T>::size;

    // Below N=64 the scalar boundary loops constant-fold/unroll better than the
    // width-descent. Keep the simple loops there.
    if constexpr (N < 64) {
        // Through the T* alias [complex.numbers.general], not std::complex::real/imag:
        // gcc 16 leaves those accessors out of line and calls one per element here.
        const T* s = reinterpret_cast<const T*>(in);
        for (std::size_t i = 0; i < N; ++i) {
            xre[i] = s[2 * i];
            if constexpr (Forward) xim[i] =  s[2 * i + 1];
            else                   xim[i] = -s[2 * i + 1];
        }
        kernel<N, T, true>::apply_sink(xre, xim, 1, yre, yim, aos_sink<T, Forward>{out});
        return;
    }

    // De-interleave AoS -> SoA: native-width shuffles, then a sized-batch width
    // descent (W/2, W/4, ..., 2) and at most one scalar residue per width gap.
    //
    // Coverage note: only N=64 and N=120 reach here (the catalog's members >=64).
    // 64 is a multiple of every W, so it skips the descent; 120 enters it only at
    // W=16 (f32/AVX-512), where 120 = 7*16 + 8. The scalar residue loops below
    // cannot run for either: the descent binary-decomposes any even residue
    // exactly, so a scalar remainder needs an odd N >= 64, which the catalog has
    // none of. They stay because they are what makes this correct if one is added.
    const T* src = reinterpret_cast<const T*>(in);
    std::size_t i = 0;
    for (; i + W <= N; i += W) {
        xsimd::batch<T> re, im;
        aos_deinterleave(src + 2 * i, re, im);
        re.store_unaligned(xre + i);
        im.store_unaligned(xim + i);
    }
    poet::static_for<1, std::bit_width(W)>([&](auto S) {
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

    // Inverse via the conjugate identity  inv(x) = conj(fwd(conj(x)))  so only
    // the forward kernel is instantiated (halves the codelet catalog's heavy
    // per-N instantiation). Entry conj negates the imaginary SoA lane; the exit
    // conj rides the sign in aos_sink.
    if constexpr (!Forward) for (std::size_t k = 0; k < N; ++k) xim[k] = -xim[k];
    kernel<N, T, true>::apply_sink(xre, xim, 1, yre, yim, aos_sink<T, Forward>{out});
}

// Lanes-as-lines twin of codelet_apply: `nlines` in-place lines at uniform `stride`,
// W lines per tile, one xsimd::transpose per W-wide column block in and out. The block
// loop is a static_for, so every block width (and therefore its AoS mask and its HiHalf
// arm) is a compile-time constant, and ceil(N/W) == 1 emits no extra code.
//
// Reads and writes exactly N complex per line: aos_deinterleave_masked never reads into
// the next line and aos_interleave_prefix never writes past it, so a tile needs no
// padding and the last full tile is not a special case.
//
// SCALED (unlike codelet_apply): `fct` folds into the output, replacing the run's second
// pass. Inverse rides the same forward kernel_batched via conj(fwd(conj(x))): the
// output conjugation is the sign of the imaginary scale, so only the Forward
// instantiation of the heavy batched kernel is ever compiled.
//
// N in {2,4} keeps the per-line loop: there kernel<N> is a register-resident pow2
// dif_butterfly with no twiddle table and no memory round-trip, cheaper than the tile's
// fixed 4*ceil(N/W) transposes.
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
                xsimd::transpose(rb, rb + W);  // rb[j].lane(l) = Re x_{j0+j} of line l
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
                    // Lanes past the block width land in columns the prefix store drops.
                    constexpr std::size_t j = J;  // J is signed; cols is not
                    constexpr std::size_t k = (j < cols) ? j0 + j : 0;
                    tr[J] = yr[k] * fr;
                    ti[J] = yi[k] * fi;
                });
                xsimd::transpose(tr, tr + W);  // tr[l].lane(j) = Re X_{j0+j} of line l
                xsimd::transpose(ti, ti + W);
                for (std::size_t l = 0; l < W; ++l)
                    aos_interleave_prefix<cols, T>(base + l * 2 * stride + 2 * j0, tr[l], ti[l]);
            });
        }
    }
    // Tail, and the whole run for N in {2,4} or nlines < W. Forward transforms carry
    // fct == 1, and at N == 4 an unconditional scale pass is the difference between
    // a loss and a win, so keep the same skip apply_scale uses.
    const bool unit = (fct == T(1));
    for (; r < nlines; ++r) {
        std::complex<T>* p = data + r * stride;
        codelet_apply<N, T, Forward>(p, p);
        if (!unit) scale_inplace(p, N, fct);
    }
}

} // namespace detail
} // namespace admiral
