#pragma once
// Batched DIF multipass over size M in V-space: W independent sub-transforms, one
// per SIMD lane, twiddles scalar-broadcast, two ping-pong V-plane pairs. No
// kernel_batched<M> instantiation. real_fft.hpp drives this as the r2c/c2r tile engine.
//
// This is phase 2 of the three-phase ducc0 cfftp_vecpass architecture for
// N = W*M. Phases 1 and 3 (the pack and the per-lane W_N^{l*k2} twist + size-W
// cross-lane DFT) have no plan-side route. The benchmark's --vpass probe exercises
// them.

#include <admiral/detail/butterfly.hpp>    // dif_butterfly<T,IP,V> (forward-only)
#include <admiral/detail/twiddles.hpp>     // build_dif_factor_plan, dif_factor_plan
#include <admiral/detail/portable_trig.hpp>
#include <vector>

#include "macros.hpp"

namespace vp {

// Per-pass metadata stored in the tables so the kernel needs no sincos/malloc.
struct pass_info {
    std::size_t ip;
    std::size_t l1;
    std::size_t ido;
    std::size_t tw_offset; // index into the flat tw_re/tw_im tables
};

// Read-only Phase-2 tables: DIF factor plan + flat per-pass W_M twiddles.
// DIRECTION-FREE: only the forward table lives here. The inverse multiplies the
// forward twiddle inside the swapped emit (see vpass_one): swap(w * swap(v)) is
// conj(w) * v, so the result is bitwise the inverse-table form with half the storage.
template<typename T>
struct multipass_tables {
    admiral::detail::dif_factor_plan fp{};
    std::vector<pass_info>       passes;
    std::vector<T>               tw_re, tw_im; // flat W_M^{j*l1*a} tables (forward)
    std::size_t                  M = 0;

    void build(std::size_t M_) {
        M = M_;
        fp = admiral::detail::build_dif_factor_plan<T>(M);
        passes.clear();
        tw_re.clear();
        tw_im.clear();

        std::size_t l1 = 1;
        for (std::size_t p = 0; p < fp.count; ++p) {
            const std::size_t ip  = fp.radices[p];
            const std::size_t ido = M / (l1 * ip);
            const std::size_t tw_sz = (ip - 1) * ido;

            pass_info pi{};
            pi.ip = ip;
            pi.l1 = l1;
            pi.ido = ido;
            pi.tw_offset = tw_re.size();
            tw_re.resize(tw_re.size() + tw_sz);
            tw_im.resize(tw_im.size() + tw_sz);
            for (std::size_t j = 1; j < ip; ++j) {
                for (std::size_t a = 0; a < ido; ++a) {
                    const auto [sn, cs] =
                        admiral::detail::portable_trig::sincos_turns<true>(j * l1 * a, M);
                    tw_re[pi.tw_offset + (j - 1) * ido + a] = static_cast<T>(cs);
                    tw_im[pi.tw_offset + (j - 1) * ido + a] = static_cast<T>(sn);
                }
            }

            passes.push_back(pi);
            l1 *= ip;
        }
    }
};

namespace detail {

// One DIF pass in V-space (ping-pong), radix IP.
//   src: src_re/im[a + ido*(j + ip*b)],  dst: dst_re/im[a + ido*(b + l1*k)]
//   Twiddle: tw_re/im[(j-1)*ido + a] = W_M^{j*l1*a}, broadcast to V.
//
// dif_butterfly is forward-only; the inverse runs it in swapped domain,
// swap(fwd(swap x)) == inv(x). The swap cannot move to the chain boundary
// here. That boundary is real data (im == 0), where exchanging planes would zero
// the real part. The twiddle multiply therefore moves INSIDE the swapped emit
// (swap(w*t) is the conjugated multiply), which is what makes one forward table
// serve both directions.
template<std::size_t IP, typename T, bool Forward, typename V>
ADM_ALWAYS_INLINE void vpass_one(const V* src_re, const V* src_im,
                                 V* dst_re, V* dst_im,
                                 std::size_t l1, std::size_t ido,
                                 const T* tw_re, const T* tw_im) {
    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a = 0; a < ido; ++a) {
            V tr[IP], ti[IP];
            for (std::size_t j = 0; j < IP; ++j) {
                tr[j] = src_re[a + ido * (j + IP * b)];
                ti[j] = src_im[a + ido * (j + IP * b)];
            }
            if constexpr (Forward) {
                admiral::detail::dif_butterfly<T, IP, V>(tr, ti,
                    [&](const auto k, V sr, V si) {
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
            } else {
                admiral::detail::dif_butterfly<T, IP, V>(ti, tr,
                    [&](const auto k, V sr, V si) {
                        if constexpr (k > 0u) {
                            const V owr(tw_re[(k - 1u) * ido + a]);
                            const V owi(tw_im[(k - 1u) * ido + a]);
                            dst_re[a + ido * (b + l1 * k)] = (owr * si + owi * sr);
                            dst_im[a + ido * (b + l1 * k)] = (owr * sr - owi * si);
                        } else {
                            dst_re[a + ido * (b + l1 * k)] = si;
                            dst_im[a + ido * (b + l1 * k)] = sr;
                        }
                    });
            }
        }
    }
}

// poet::dispatch adapter: maps runtime radix to compile-time IP of vpass_one.
// A struct, not a lambda: C++17 lambdas cannot take template parameters, and
// poet::dispatch calls f.template operator()<IP>(args...), which a member template
// serves the same way.
template<typename T, bool Forward, typename V>
struct vpass_invoke_t {
    template<std::size_t IP>
    void operator()(const V* src_re, const V* src_im, V* dst_re, V* dst_im,
                    std::size_t l1, std::size_t ido, const T* tw_re, const T* tw_im) const {
        vpass_one<IP, T, Forward, V>(src_re, src_im, dst_re, dst_im, l1, ido, tw_re, tw_im);
    }
};
template<typename T, bool Forward, typename V>
inline constexpr vpass_invoke_t<T, Forward, V> vpass_invoke{};

template<typename T, bool Forward, typename V>
ADM_ALWAYS_INLINE void vpass_dispatch(std::size_t ip,
                                      const V* src_re, const V* src_im,
                                      V* dst_re, V* dst_im,
                                      std::size_t l1, std::size_t ido,
                                      const T* tw_re, const T* tw_im) {
    poet::dispatch(vpass_invoke<T, Forward, V>,
                   poet::dispatch_param<admiral::detail::dif_radix_set>{ip},
                   src_re, src_im, dst_re, dst_im, l1, ido, tw_re, tw_im);
}

} // namespace detail

// Phase-2 batched DIF multipass over size M in V-space (ping-pong).
// Each SIMD lane is an independent size-M transform (no four-step twist).
// Input in cur_*; returns the plane pair with the natural-order result. UN-normalized.
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

} // namespace vp

#include "undef_macros.hpp"
