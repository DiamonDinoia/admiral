#pragma once

// ============================================================================
// Iterative DIF (Gentleman-Sande) pass-chain driver: mixed-radix, iterative,
// natural-order output. N = ip*l1*ido per pass; a in [0,ido) contiguous, vectorized
// with a scalar tail. Pass recurrence: l1=1; per factor ip: ido = N/(l1*ip); run pass;
// l1 *= ip. Twiddles per pass: tw[(j-1)*ido + a] = W_N^{j*l1*a}, j=0 trivial. Every
// pass here has ido >= 2 (ido==1 goes to `dif_pass_last`). Scratch and twiddles are
// externally owned; no hot-path allocation.
// Ref: Gentleman & Sande, AFIPS Fall JCC 29 (1966) 563. DOI 10.1145/1464291.1464352
// ============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <poet/poet.hpp>


#include "codelet.hpp"     // `kernel_batched` (lane-packed terminal)
#include "dif_passes.hpp"  // `dif_pass[_first/_last/_fused]` + invokers
#include "math.hpp"        // `codelet_dispatch` (terminal base kernel)
#include "scratch.hpp"     // `soa_scratch` (`dif_execute_in_place` buffers)
#include "simd_swizzle.hpp" // `aos_interleave` (terminal AoS stores)
#include "twiddles.hpp"    // `dif_twiddle_set`, `build_dif_twiddle_set`

#include "macros.hpp"      // `ADM_ALWAYS_INLINE` and `ADM_NOINLINE`; undef at EOF

namespace admiral {
namespace detail {

// `dif_radix_set` defined in `twiddles.hpp`; see comment there.

// Execute tape: the plan records the walk once as plain data; `execute()` is a flat loop.
// Every thunk has the one signature `dif_step::fn_t`. Thunks resolve tables by pass index
// into `rt.dtw`, never by stored pointer, so a copied `dif_twiddle_set` still transforms.
template<typename T>
struct dif_rt {
    const std::complex<T>* in;
    std::complex<T>* out;
    T scale;
    const dif_twiddle_set<T>* dtw;
};

// Element-stride decode of an `es` bit: bit b set means stride 2, clear means 1.
constexpr std::size_t es_stride(unsigned es, int bit) { return ((es >> bit) & 1u) + 1u; }

template<typename T, bool Forward, std::size_t IP>
void dif_tape_step_first(const T*, const T*, T* dr, T* di,
                         const dif_step<T>& s, const dif_rt<T>& rt) {
    const auto& tw = rt.dtw->passes[s.p];
    dif_pass_first<T, Forward, IP>(rt.in, dr, di, 1, s.ido, tw.first.data(), tw.second.data(),
                                   es_stride(s.es, 1),
                                   rt.dtw->p0_block);
}

template<typename T, std::size_t P, bool Chiplet>
void dif_tape_step_body(const T* sr, const T* si, T* dr, T* di,
                        const dif_step<T>& s, const dif_rt<T>& rt) {
    const auto& tw = rt.dtw->passes[s.p];
    if constexpr (Chiplet)
        dif_pass_prime_chip<T, P>(sr, si, dr, di, s.l1, s.ido, tw.first.data(),
                                  tw.second.data(), es_stride(s.es, 0), es_stride(s.es, 1));
    else
        dif_pass<T, P>(sr, si, dr, di, s.l1, s.ido, tw.first.data(), tw.second.data(),
                       es_stride(s.es, 0), es_stride(s.es, 1));
}

template<typename T, std::size_t IP>
void dif_tape_step_ip(const T*, const T*, T* dr, T* di,
                      const dif_step<T>& s, const dif_rt<T>& rt) {
    const auto& tw = rt.dtw->passes[s.p];
    const std::size_t es = es_stride(s.es, 1);
    dif_pass_ip_flat<T, IP>(dr, di, dr, di, s.l1, s.ido, tw.first.data(), tw.second.data(),
                            es, es);
}


template<typename T, std::size_t P1, std::size_t P2>
void dif_tape_step_f2(const T* sr, const T* si, T* dr, T* di,
                      const dif_step<T>& s, const dif_rt<T>& rt) {
    dif_pass_fused2<T, P1, P2>(sr, si, dr, di, s.l1, s.ido, rt.dtw->packed_pair[s.p].data());
}

template<typename T>
void dif_tape_step_f3(const T* sr, const T* si, T* dr, T* di,
                      const dif_step<T>& s, const dif_rt<T>& rt) {
    const auto& t0 = rt.dtw->passes[s.p];
    const auto& t1 = rt.dtw->passes[s.p + 1];
    const auto& t2 = rt.dtw->passes[s.p + 2];
    dif_pass_fused3<T, 4, 4, 4>(sr, si, dr, di, s.l1, s.ido,
                                t0.first.data(), t0.second.data(), t1.first.data(),
                                t1.second.data(), t2.first.data(), t2.second.data());
}

template<typename T, bool Forward, std::size_t IP>
void dif_tape_step_last(const T* sr, const T* si, T*, T*,
                        const dif_step<T>& s, const dif_rt<T>& rt) {
    const auto& tw = rt.dtw->passes[s.p];
    dif_pass_last<T, Forward, IP>(sr, si, rt.out, s.l1, 1, tw.first.data(), tw.second.data(),
                                  rt.scale, (s.es & 4u) ? rt.dtw->rowperm.data() : nullptr);
}

// Single-pass (N == ip, no terminal): re-interleave the SoA `cc0` back to AoS.
template<typename T, bool Forward>
void dif_tape_step_single(const T* sr, const T* si, T*, T*,
                          const dif_step<T>& s, const dif_rt<T>& rt) {
    for (std::size_t k = 0; k < s.n; ++k) {
        const auto [xr, xi] = plane_vals<Forward>(sr[k] * rt.scale, si[k] * rt.scale);
        rt.out[k] = std::complex<T>(xr, xi);
    }
}

// One struct per precision so one extern template covers every direction-free step
// family. Members are defined out of line deliberately: an explicit instantiation
// declaration does not suppress implicit instantiation of in-class (inline) members.
// Definitions: `src/inst_dif_thunks_{f,d}.cpp`.
// Warning: moving a step family across TUs moves codegen, because gcc budgets inlining
// per TU. Re-measure runtime after such a move.

// `poet::dispatch` makers, one stateless functor per step family.
template<typename T, bool Chiplet>
struct dif_thunk_body_maker {
    using fn_t = typename dif_step<T>::fn_t;
    template<std::size_t P>
    fn_t operator()() const noexcept { return &dif_tape_step_body<T, P, Chiplet>; }
};

template<typename T>
struct dif_thunk_ip_maker {
    using fn_t = typename dif_step<T>::fn_t;
    template<std::size_t IP>
    fn_t operator()() const noexcept { return &dif_tape_step_ip<T, IP>; }
};

template<typename T>
struct dif_thunk_f2_maker {
    using fn_t = typename dif_step<T>::fn_t;
    template<std::size_t P1, std::size_t P2>
    fn_t operator()() const noexcept { return &dif_tape_step_f2<T, P1, P2>; }
};

template<typename T, bool Forward>
struct dif_tape_fill_first {
    template<std::size_t IP>
    void operator()(dif_step<T>& s) const noexcept { s.fn = &dif_tape_step_first<T, Forward, IP>; }
};

template<typename T, bool Forward>
struct dif_tape_fill_last {
    template<std::size_t IP>
    void operator()(dif_step<T>& s) const noexcept { s.fn = &dif_tape_step_last<T, Forward, IP>; }
};

template<typename T>
struct dif_thunk {
    using fn_t = typename dif_step<T>::fn_t;
    static auto body(std::size_t ip) -> fn_t;
    static auto chiplet(std::size_t ip) -> fn_t;
    static auto in_place(std::size_t ip) -> fn_t;
    static auto fused2(std::size_t p1, std::size_t p2) -> fn_t;
    static auto fused3() -> fn_t;  // the one shape `fused3` is elected for, 4x4x4
};

template<typename T>
auto dif_thunk<T>::body(std::size_t ip) -> fn_t {
    return poet::dispatch(poet::throw_on_no_match, dif_thunk_body_maker<T, false>{},
                          poet::dispatch_param<dif_radix_set>{ip});
}

template<typename T>
auto dif_thunk<T>::chiplet(std::size_t ip) -> fn_t {
    return poet::dispatch(poet::throw_on_no_match, dif_thunk_body_maker<T, true>{},
                          poet::dispatch_param<dif_generic_radix_seq>{ip});
}

template<typename T>
auto dif_thunk<T>::in_place(std::size_t ip) -> fn_t {
    return poet::dispatch(poet::throw_on_no_match, dif_thunk_ip_maker<T>{},
                          poet::dispatch_param<dif_ip_radix_set>{ip});
}

template<typename T>
auto dif_thunk<T>::fused2(std::size_t p1, std::size_t p2) -> fn_t {
    return poet::dispatch(poet::throw_on_no_match, dif_thunk_f2_maker<T>{},
                          poet::dispatch_param<dif_fused_pair_set>{p1},
                          poet::dispatch_param<dif_fused_pair_set>{p2});
}

template<typename T>
auto dif_thunk<T>::fused3() -> fn_t {
    return &dif_tape_step_f3<T>;
}

extern template struct dif_thunk<float>;
extern template struct dif_thunk<double>;

// Record the walk once per element-stride variant: `blk` elects `es2` as if
// `soa_stride` >= N, `flat` uses stride 1. Step order and scalars match what the
// executed loop recomputes per call.
template<typename T, bool Forward>
void dif_build_tape(dif_twiddle_set<T>& dtw, std::size_t N) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t n_passes = dtw.radices.size();
    if (n_passes == 0) return;
    auto& tp = dtw.tape[Forward ? 0 : 1];

    for (unsigned variant = 0; variant < 2; ++variant) {
        std::vector<dif_step<T>>& tv = variant == 0 ? tp.blk : tp.flat;
        tv.reserve(n_passes + 1);

        // The driver's `es2` pipeline, verbatim, for the `blk` variant.
        std::uint64_t es2 = 0;
        if (variant == 0) {
            std::uint64_t blk = 0;
            std::size_t lb = 1;
            for (std::size_t p = 0; p < n_passes; ++p) {
                const std::size_t ip = dtw.radices[p], idop = N / (lb * ip);
                lb *= ip;
                if (p + 1 < n_passes && dtw.sched[p] == dif_fuse::plain && idop % W == 0)
                    blk |= std::uint64_t{1} << p;
            }
            // A buffer is blocked only if the pass that writes it and the pass that
            // reads it both index it that way.
            for (std::size_t p = 0; p + 1 < n_passes; ++p)
                if ((blk >> p & 1u) && (blk >> (p + 1) & 1u)) es2 |= std::uint64_t{1} << p;
            // An in-place pass rewrites the buffer it read: one buffer, so one stride.
            // Clearing both bits can expose a new mismatch one pass up, hence the fixpoint.
            for (bool changed = true; changed;) {
                changed = false;
                for (std::size_t p = 1; p < n_passes; ++p)
                    if ((dtw.ip_mask >> p & 1u) && ((es2 >> p ^ es2 >> (p - 1)) & 1u)) {
                        es2 &= ~((std::uint64_t{1} << p) | (std::uint64_t{1} << (p - 1)));
                        changed = true;
                    }
            }
        }
        const auto es_bit = [&](std::size_t p) { return static_cast<unsigned>(es2 >> p & 1u); };
        const auto b8 = [](bool b) { return static_cast<std::uint8_t>(b); };

        // --- First pass: AoS -> SoA (`cc0`) ---
        {
            dif_step<T> st{};
            st.dst = 0;
            st.dim = static_cast<std::uint8_t>(es_bit(0) ? 2 : 0);
            st.es = static_cast<std::uint8_t>(es_bit(0) << 1);
            st.ido = N / dtw.radices[0];
            poet::dispatch(poet::throw_on_no_match, dif_tape_fill_first<T, Forward>{},
                           poet::dispatch_param<dif_radix_set>{dtw.radices[0]}, st);
            tv.push_back(st);
        }

        // Single-pass (N == ip): re-interleave SoA `cc0` back to AoS.
        if (n_passes == 1) {
            dif_step<T> st{};
            st.fn = &dif_tape_step_single<T, Forward>;
            st.n = N;
            tv.push_back(st);
            continue;
        }

        // --- Intermediate passes: SoA ping-pong (first pass wrote `cc0`) ---
        std::size_t l1 = dtw.radices[0];
        bool ping = false;
        for (std::size_t p = 1; p + 1 < n_passes; ++p) {
            const std::size_t ip = dtw.radices[p];
            const std::size_t ido = N / (l1 * ip);
            dif_step<T> st{};
            st.p = p;
            st.l1 = l1;
            st.ido = ido;
            st.src = b8(ping);
            st.dst = b8(!ping);
            st.sim = es_bit(p - 1) ? static_cast<std::uint8_t>(2) : b8(ping);
            st.dim = es_bit(p) ? static_cast<std::uint8_t>(2) : b8(!ping);
            st.es = static_cast<std::uint8_t>(es_bit(p - 1) | es_bit(p) << 1);

            if (dif_is_generic_radix(ip)) {
                st.fn = dif_thunk<T>::chiplet(ip);
                st.n = ip;
                tv.push_back(st);
                l1 *= ip;
                ping = !ping;
                continue;
            }
            if (dtw.ip_mask >> p & 1u) {
                st.dst = st.src;  // one buffer: rewrite in place, no ping flip
                st.dim = st.sim;
                st.es = static_cast<std::uint8_t>(es_bit(p) << 1);
                st.fn = dif_thunk<T>::in_place(ip);
                tv.push_back(st);
                l1 *= ip;
                continue;
            }
            if (dtw.sched[p] == dif_fuse::f3head) {
                st.fn = dif_thunk<T>::fused3();
                tv.push_back(st);
                l1 *= 64u;  // P1 * P2 * P3 = 4 * 4 * 4
                ping = !ping;
                p += 2;
                continue;
            }
            if (dtw.sched[p] == dif_fuse::f2head) {
                st.fn = dif_thunk<T>::fused2(ip, dtw.radices[p + 1]);
                tv.push_back(st);
                l1 *= ip * dtw.radices[p + 1];
                ping = !ping;
                ++p;
                continue;
            }
            st.fn = dif_thunk<T>::body(ip);
            tv.push_back(st);
            l1 *= ip;
            ping = !ping;
        }

        // --- Last pass: SoA -> AoS (always reads a plain layout buffer) ---
        {
            const std::size_t p = n_passes - 1;
            dif_step<T> st{};
            st.p = p;
            st.l1 = l1;
            st.src = b8(ping);
            st.sim = b8(ping);
            st.es = dtw.rowperm.empty() ? std::uint8_t{0} : std::uint8_t{4};
            poet::dispatch(poet::throw_on_no_match, dif_tape_fill_last<T, Forward>{},
                           poet::dispatch_param<dif_radix_set>{dtw.radices[p]}, st);
            tv.push_back(st);
        }
    }
}

// Execute the DIF pass-chain: AoS in -> SoA ping-pong -> AoS out. If `in == out`, no
// staging copy is needed: the first pass fully drains `in` before any AoS store. `cc0`
// and `cc1` each hold >= N elements per plane; allocation-free. `scale_val` folds into
// `dif_pass_last`'s stores (1 for un-normalized). Requires `n_passes` >= 2; a
// single-factor N goes codelet.
//
// `soa_stride` >= N selects the W-blocked SoA tape (element stride 2 halves a radix-IP
// pass's 2*IP store streams to IP). `soa_stride` >= N declares that each re/im pair is
// one contiguous 2N span; only the allocation's owner can know that. The kernels never
// apply the value as an address stride. Four independent vectors must stay on the flat
// tape.
template<typename T, bool Forward>
void iterative_dif_execute_ws(const std::complex<T>* in, std::complex<T>* out,
                               std::size_t N,
                               T* cc0re, T* cc0im, T* cc1re, T* cc1im,
                               const dif_twiddle_set<T>& dtw,
                               T scale_val = T(1),
                               std::size_t soa_stride = 0) {
    if (N <= 1) return;
    constexpr std::size_t W = xsimd::batch<T>::size;
    const auto& tp = dtw.tape[Forward ? 0 : 1];
    const std::vector<dif_step<T>>& tv = soa_stride >= N ? tp.blk : tp.flat;
    const dif_rt<T> rt{in, out, scale_val, &dtw};
    for (const dif_step<T>& st : tv) {
        T* const dr = st.dst ? cc1re : cc0re;
        const T* const sr = st.src ? cc1re : cc0re;
        T* const di = st.dim == 2 ? dr + W : (st.dim ? cc1im : cc0im);
        const T* const si = st.sim == 2 ? sr + W : (st.sim ? cc1im : cc0im);
        st.fn(sr, si, dr, di, st, rt);
    }
}

// Instantiation boundary is the `<Forward>` leaf, one TU per direction:
// `src/inst_dif_{f,d}_{fwd,inv}.cpp`.
extern template void iterative_dif_execute_ws<float, true>(
    const std::complex<float>*, std::complex<float>*, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, std::size_t);
extern template void iterative_dif_execute_ws<float, false>(
    const std::complex<float>*, std::complex<float>*, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, std::size_t);
extern template void iterative_dif_execute_ws<double, true>(
    const std::complex<double>*, std::complex<double>*, std::size_t, double*, double*, double*,
    double*, const dif_twiddle_set<double>&, double, std::size_t);
extern template void iterative_dif_execute_ws<double, false>(
    const std::complex<double>*, std::complex<double>*, std::size_t, double*, double*, double*,
    double*, const dif_twiddle_set<double>&, double, std::size_t);
// The plan-time half of the boundary: every TU calls `dif_build_tape` (through
// `build_dif_twiddle_set`) but only the inst TUs instantiate `dif_build_tape`.
extern template void dif_build_tape<float, true>(dif_twiddle_set<float>&, std::size_t);
extern template void dif_build_tape<float, false>(dif_twiddle_set<float>&, std::size_t);
extern template void dif_build_tape<double, true>(dif_twiddle_set<double>&, std::size_t);
extern template void dif_build_tape<double, false>(dif_twiddle_set<double>&, std::size_t);

// `forward` -> `<Forward>` trampoline: both leaves are extern above, so the body is two
// calls and pulls in no pass tree. Mirrors `col_dif_dispatch` in `dif_col_driver.hpp`.
template<typename T>
void dif_dispatch(bool forward, const std::complex<T>* in, std::complex<T>* out, std::size_t N,
                  T* cc0re, T* cc0im, T* cc1re, T* cc1im, const dif_twiddle_set<T>& dtw,
                  T scale_val = T(1), std::size_t soa_stride = 0) {
    if (forward)
        iterative_dif_execute_ws<T, true>(in, out, N, cc0re, cc0im, cc1re, cc1im, dtw, scale_val,
                                          soa_stride);
    else
        iterative_dif_execute_ws<T, false>(in, out, N, cc0re, cc0im, cc1re, cc1im, dtw, scale_val,
                                           soa_stride);
}

// out doubles as the second ping-pong pair, halving the scratch to one pair. Four
// conditions, all necessary:
//   parity: the last pass must read buffer 0; pure flip-count parity, not arrangeable.
//   layout: each pair one contiguous 2N span; blocked passes address im as dr + W.
//   alignment: a performance guard only (every access is unaligned-tolerant); a pair off
//     the line boundary splits every W-block store across two lines.
//   residency: the smaller resident set wins only while the 4-plane scratch fits L2; the
//     aliased pair is stuck at stride N and loses `span_stride`'s anti-alias padding.
template<typename T>
[[nodiscard]] bool dif_out_aliasable(bool forward, const std::complex<T>* out, std::size_t N,
                                     const dif_twiddle_set<T>& tw) {
    // Read the executed tape, not a variant that happens to match. Warning: the
    // thresholds below come from one host's measurements; re-measure on another machine.
    const auto& tv = tw.tape[forward ? 0 : 1].blk;
    if (tv.size() < 2 || tv.front().dst != 0 || tv.back().src != 0) return false;
    if (4 * N * sizeof(T) > cpu_cache().l2) return false;
    return N * sizeof(T) % span_align<T> == 0 &&
           reinterpret_cast<std::uintptr_t>(out) % span_align<T> == 0;
}

// Allocate the SoA scratch and dispatch. Every route but `four_step_large` calls
// `dif_execute_in_place`; `four_step_large` owns per-thread scratch.
template<typename T>
void dif_execute_in_place(bool forward, const std::complex<T>* in, std::complex<T>* out,
                          std::size_t N, const dif_twiddle_set<T>& tw, T scale = T(1)) {
    if (dif_out_aliasable<T>(forward, out, N, tw)) {
        soa_scratch<T, 2> sc(N);
        T* const o = reinterpret_cast<T*>(out);
        dif_dispatch<T>(forward, in, out, N, sc.buf(0), sc.buf(1), o, o + N, tw, scale,
                        sc.stride());
        return;
    }
    soa_scratch<T, 4> sc(N);
    dif_dispatch<T>(forward, in, out, N, sc.buf(0), sc.buf(1), sc.buf(2), sc.buf(3), tw, scale,
                    sc.stride());
}

} // namespace detail
} // namespace admiral


#include "undef_macros.hpp"
