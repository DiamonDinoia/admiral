#pragma once
// Lane-batched vecpass kernel for N = W * M, W = xsimd::batch<T>::size lanes.
// BENCH PROBE ONLY (--vpass): there is no plan-side route.
// See the note at the bottom of this file.
//
// Phase 1: contiguous load pack (AoS planar -> SoA V-planes, free).
// Phase 2: runtime-looped DIF multipass over size M in V-space. W independent
//          sub-transforms per SIMD lane; twiddles scalar-broadcast. Two ping-pong
//          V-plane pairs. No kernel_batched<M> instantiation (avoids the spill
//          pile that killed the f64 four-step).
// Phase 3: per-lane W_N^{l*k2} twist (fused into the load) + size-W cross-lane DFT.

#include <admiral/detail/butterfly.hpp>    // dif_butterfly<T,Forward,IP,V>
#include <admiral/detail/codelet.hpp>      // kernel_batched
#include <admiral/detail/twiddles.hpp>     // build_dif_factor_plan, dif_factor_plan
#include <admiral/detail/portable_trig.hpp>
#include <admiral/detail/simd_swizzle.hpp> // aos_interleave
#include <complex>
#include <memory>
#include <vector>

namespace vp {

// Per-pass metadata stored in the tables so the kernel needs no sincos/malloc.
struct pass_info {
    unsigned    ip;
    std::size_t l1;
    std::size_t ido;
    std::size_t tw_offset; // index into the flat tw_re/tw_im tables
};

// Read-only Phase-2 multipass tables: the DIF factor plan + flat per-pass W_M
// twiddle tables. Built once (direction-aware) via build<Forward>(M); cheap to
// hold plan-resident.
template<typename T>
struct multipass_tables {
    admiral::detail::dif_factor_plan fp{};
    std::vector<pass_info>       passes;
    std::vector<T>               tw_re, tw_im; // flat W_M^{j*l1*a} tables
    unsigned                     M = 0;

    template<bool Forward>
    void build(unsigned M_) {
        M = M_;
        fp = admiral::detail::build_dif_factor_plan<T>(static_cast<std::size_t>(M));
        passes.clear();
        tw_re.clear();
        tw_im.clear();

        std::size_t l1 = 1;
        for (std::size_t p = 0; p < fp.count; ++p) {
            const unsigned    ip  = fp.radices[p];
            const std::size_t ido = static_cast<std::size_t>(M) / (l1 * static_cast<std::size_t>(ip));
            const std::size_t tw_sz = static_cast<std::size_t>(ip - 1) * ido;

            pass_info pi{};
            pi.ip = ip;
            pi.l1 = l1;
            pi.ido = ido;
            pi.tw_offset = tw_re.size();
            tw_re.resize(tw_re.size() + tw_sz);
            tw_im.resize(tw_im.size() + tw_sz);
            for (unsigned j = 1; j < ip; ++j) {
                for (std::size_t a = 0; a < ido; ++a) {
                    const auto [sn, cs] = admiral::detail::portable_trig::sincos_turns<Forward>(
                        static_cast<std::size_t>(j) * l1 * a, static_cast<std::size_t>(M));
                    tw_re[pi.tw_offset + (j - 1u) * ido + a] = static_cast<T>(cs);
                    tw_im[pi.tw_offset + (j - 1u) * ido + a] = static_cast<T>(sn);
                }
            }

            passes.push_back(pi);
            l1 *= static_cast<std::size_t>(ip);
        }
    }
};

namespace detail {

// One DIF pass in V-space, ping-pong: src -> dst.
//   src: src_re/im[a + ido*(j + ip*b)]  j in [0,ip), b in [0,l1), a in [0,ido)
//   dst: dst_re/im[a + ido*(b + l1*k)]  k in [0,ip) butterfly out
// Twiddle: tw_re/im[(j-1)*ido + a] = W_M^{j*l1*a}, broadcast to V.
template<int IP, typename T, bool Forward, typename V>
inline __attribute__((always_inline)) void vpass_one(const V* src_re, const V* src_im,
               V* dst_re, V* dst_im,
               std::size_t l1, std::size_t ido,
               const T* tw_re, const T* tw_im) {
    using admiral::detail::dif_butterfly;
    constexpr std::size_t IPu = static_cast<std::size_t>(IP);

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a = 0; a < ido; ++a) {
            V tr[IPu], ti[IPu];
            for (std::size_t j = 0; j < IPu; ++j) {
                tr[j] = src_re[a + ido * (j + IPu * b)];
                ti[j] = src_im[a + ido * (j + IPu * b)];
            }
            dif_butterfly<T, Forward, IP, V>(tr, ti,
                [&](auto Kc, V sr, V si) {
                    constexpr std::size_t k = Kc;
                    if constexpr (k > 0u) {
                        const V owr(tw_re[(k - 1u) * ido + a]);
                        const V owi(tw_im[(k - 1u) * ido + a]);
                        dst_re[a + ido * (b + l1 * k)] = (owr * sr - owi * si);
                        dst_im[a + ido * (b + l1 * k)] = (owr * si + owi * sr);
                    } else {
                        dst_re[a + ido * (b + l1 * k)] = sr;
                        dst_im[a + ido * (b + l1 * k)] = si;
                    }
                });
        }
    }
}

// poet::dispatch invoker: turns the runtime radix into the compile-time IP of
// vpass_one, routed through the shared admiral::detail::dif_radix_set (twiddles.hpp).
template<typename T, bool Forward, typename V>
struct vpass_invoke {
    template<int IP>
    void operator()(const V* src_re, const V* src_im, V* dst_re, V* dst_im,
                    std::size_t l1, std::size_t ido,
                    const T* tw_re, const T* tw_im) const {
        vpass_one<IP, T, Forward, V>(src_re, src_im, dst_re, dst_im, l1, ido, tw_re, tw_im);
    }
};

template<typename T, bool Forward, typename V>
inline __attribute__((always_inline)) void vpass_dispatch(unsigned ip,
                    const V* src_re, const V* src_im,
                    V* dst_re, V* dst_im,
                    std::size_t l1, std::size_t ido,
                    const T* tw_re, const T* tw_im) {
    poet::dispatch(vpass_invoke<T, Forward, V>{},
                   poet::dispatch_param<admiral::detail::dif_radix_set>{static_cast<int>(ip)},
                   src_re, src_im, dst_re, dst_im, l1, ido, tw_re, tw_im);
}

} // namespace detail

// Run the Phase-2 batched DIF multipass over size M in V-space (ping-pong). Each
// SIMD lane is an INDEPENDENT, complete size-M complex transform (no four-step
// twist) — the batched primitive reused by the row-batched real-FFT (real_fft.hpp).
// Input in cur_*; returns the plane pair holding the natural-order size-M result
// (either cur_* or nxt_*, depending on the pass-count parity). UN-normalized.
template<typename T, bool Forward, typename V>
inline std::pair<const V*, const V*>
multipass_run(const multipass_tables<T>& tab,
              V* cur_re, V* cur_im, V* nxt_re, V* nxt_im) {
    const T* all_tw_re = tab.tw_re.data();
    const T* all_tw_im = tab.tw_im.data();
    bool use_nxt = false;
    for (const pass_info& pi : tab.passes) {
        const V* src_re = use_nxt ? nxt_re : cur_re;
        const V* src_im = use_nxt ? nxt_im : cur_im;
        V* dst_re       = use_nxt ? cur_re : nxt_re;
        V* dst_im       = use_nxt ? cur_im : nxt_im;
        detail::vpass_dispatch<T, Forward, V>(pi.ip, src_re, src_im, dst_re, dst_im,
                                              pi.l1, pi.ido,
                                              all_tw_re + pi.tw_offset, all_tw_im + pi.tw_offset);
        use_nxt = !use_nxt;
    }
    return { use_nxt ? nxt_re : cur_re, use_nxt ? nxt_im : cur_im };
}

// Lane-batched vecpass. in: planar (re,im) length N = W*M. out: AoS
// std::complex<T>* length N (same layout as the default plan).
// twN_re/twN_im: N scalars, twN[k2*W + l] = W_N^{+/- l*k2} for the Phase-3 twist
// (direction baked at build time, matching Forward). tab holds the precomputed
// Phase-2 W_M tables; cur_*/nxt_* are caller-owned ping-pong working V-planes
// (length M each, contents overwritten before read).
template<typename T, bool Forward, typename V = xsimd::batch<T>>
void vpass_forward(const T* re, const T* im, std::complex<T>* out,
                   const T* twN_re, const T* twN_im,
                   const multipass_tables<T>& tab,
                   V* cur_re, V* cur_im, V* nxt_re, V* nxt_im) {
    using admiral::detail::kernel_batched;
    constexpr unsigned W = static_cast<unsigned>(V::size);
    static_assert(W == 4 || W == 8 || W == 16, "combine specialized for W in {4,8,16}");

    const unsigned M = tab.M;

    // --- Phase 1: pack. d[n2] lanes = {x[W*n2 .. W*n2+W-1]} (contiguous load).
    for (unsigned n2 = 0; n2 < M; ++n2) {
        cur_re[n2] = V::load_unaligned(re + W * n2);
        cur_im[n2] = V::load_unaligned(im + W * n2);
    }

    // --- Phase 2: runtime DIF multipass over size M in V-space (zero sincos/malloc).
    const T* all_tw_re = tab.tw_re.data();
    const T* all_tw_im = tab.tw_im.data();

    bool use_nxt = false;
    for (const pass_info& pi : tab.passes) {
        const T* tw_re = all_tw_re + pi.tw_offset;
        const T* tw_im = all_tw_im + pi.tw_offset;
        const V* src_re = use_nxt ? nxt_re : cur_re;
        const V* src_im = use_nxt ? nxt_im : cur_im;
        V* dst_re       = use_nxt ? cur_re : nxt_re;
        V* dst_im       = use_nxt ? cur_im : nxt_im;
        detail::vpass_dispatch<T, Forward, V>(pi.ip, src_re, src_im, dst_re, dst_im,
                                              pi.l1, pi.ido, tw_re, tw_im);
        use_nxt = !use_nxt;
    }
    const V* Gre = use_nxt ? nxt_re : cur_re;
    const V* Gim = use_nxt ? nxt_im : cur_im;

    // --- Phase 3: fused W_N^{l*k2} twist + size-W cross-lane DFT.
    unsigned k2 = 0;
    for (; k2 + W <= M; k2 += W) {
        V r[W], i_[W];
        for (unsigned a = 0; a < W; ++a) {
            const V owr = V::load_unaligned(twN_re + (k2 + a) * W);
            const V owi = V::load_unaligned(twN_im + (k2 + a) * W);
            const V gr  = Gre[k2 + a];
            const V gi  = Gim[k2 + a];
            r[a]  = (owr * gr - owi * gi);
            i_[a] = (owr * gi + owi * gr);
        }
        xsimd::transpose(r, r + W);
        xsimd::transpose(i_, i_ + W);
        V o[W], oi[W];
        kernel_batched<W, T, Forward, V>::apply(r, i_, 1, o, oi);
        for (unsigned k1 = 0; k1 < W; ++k1) {
            admiral::detail::aos_interleave<T>(
                reinterpret_cast<T*>(out + k1 * M + k2), o[k1], oi[k1]);
        }
    }
    for (; k2 < M; ++k2) {              // scalar tail (M % W leftover)
        // ponytail: direct O(W^2) DFT for the <=W-1-element tail; ~1% of work.
        T gr[W], gi[W];
        for (unsigned l = 0; l < W; ++l) {
            const T owr = twN_re[k2 * W + l];
            const T owi = twN_im[k2 * W + l];
            const T g_r = Gre[k2].get(l);
            const T g_i = Gim[k2].get(l);
            gr[l] = g_r * owr - g_i * owi;
            gi[l] = g_r * owi + g_i * owr;
        }
        for (unsigned k1 = 0; k1 < W; ++k1) {
            T sr = T(0), si = T(0);
            for (unsigned l = 0; l < W; ++l) {
                const auto [sn, cs] = admiral::detail::portable_trig::sincos_turns<Forward>(
                    static_cast<unsigned long>(k1) * l, static_cast<unsigned long>(W));
                const T wr = static_cast<T>(cs), wi = static_cast<T>(sn);
                sr += gr[l] * wr - gi[l] * wi;
                si += gr[l] * wi + gi[l] * wr;
            }
            out[k1 * M + k2] = std::complex<T>(sr, si);
        }
    }
}

} // namespace vp

// The vp:: kernels above are benchmark-only (the --vpass probe); there is no
// plan-side route. See docs/codelet-optimization-frontier.md.
