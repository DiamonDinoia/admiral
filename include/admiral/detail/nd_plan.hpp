#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include "cxx_compat.hpp"

#include <admiral/errors.hpp>

#include "simd.hpp"

#include "dif_col_driver.hpp"
#include "cache.hpp"
#include "math.hpp"
#include "plan.hpp"
#include "scratch.hpp"
#include "thread_pool.hpp"
#include "twiddles.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

[[nodiscard]] inline std::optional<std::size_t> extent_product(
    span<const std::size_t> shape) noexcept {
    std::size_t total = 1;
    for (const std::size_t e : shape) {
        if (e == 0 || total > std::numeric_limits<std::size_t>::max() / e) return std::nullopt;
        total *= e;
    }
    return total;
}

template<typename T>
struct nd_axis_state {
    std::size_t length = 0;
    bool dif = false;
    bool col_codelet = false;
    dif_twiddle_set<T> dtw;
    std::optional<plan_impl<T>> plan;
};

[[nodiscard]] inline dif_factor_plan build_radix4_plan(std::size_t n) {
    dif_factor_plan p;
    while (n % 4 == 0) { p.push(4); n /= 4; }
    if (n == 2) p.push(2);
    return p;
}

inline constexpr std::size_t kE2Len64MinL3PerCoreBytes = std::size_t{2} << 20;

[[nodiscard]] constexpr std::size_t e2_len_cap_by_l3(std::size_t l3_per_core_bytes) {
    return l3_per_core_bytes >= kE2Len64MinL3PerCoreBytes ? std::size_t{64} : std::size_t{32};
}

[[nodiscard]] inline std::size_t e2_len_cap() {
    const cache_bytes& cc = cpu_cache();
    if (cc.l3_cores == 0 || cc.l3 == 0) return 32;
    return e2_len_cap_by_l3(cc.l3 / cc.l3_cores);
}

template<typename T>
[[nodiscard]] inline nd_axis_state<T> make_nd_axis_state(std::size_t length, std::size_t inner,
                                                         bool is_forward, bool innermost,
                                                         std::size_t nthreads = 1,
                                                         admiral::effort eff =
                                                             admiral::effort::estimate) {
    nd_axis_state<T> st;
    st.length = length;
    if (length <= 1) {
        st.plan.emplace(length, is_forward, nthreads, nullptr, eff);
        return st;
    }
    if (!innermost && is_codelet_supported(length)) {
        st.dif = true;
        const std::size_t e2_cap = std::min(e2_len_cap(), kFourStepLeafMax);
        st.col_codelet = length >= 8 && length <= e2_cap &&
                         is_codelet_catalog(length) && inner <= 64;
        dif_factor_plan r4;
        const dif_factor_plan* ov = nullptr;
        if constexpr (sizeof(T) == 4) {
            constexpr std::size_t W = xsimd::batch<T>::size;
            const bool small_inner =
                (nd_col_block<T>(length, inner, nthreads, 1) / W) < 4
                || (inner % W) != 0;
            const bool pow2 = detail::has_single_bit(length);
            if (small_inner && pow2) { r4 = build_radix4_plan(length); ov = &r4; }
        }
        st.dtw = build_dif_twiddle_set<T>(length, ov, false);
    }
    st.plan.emplace(length, is_forward, nthreads, nullptr, eff);
    return st;
}

template<typename T, typename LineBase>
ADM_ALWAYS_INLINE void apply_lines_contiguous(std::complex<T>* data, std::size_t len,
                                              const nd_axis_state<T>& st, std::optional<T> fct,
                                              thread_pool* pool, std::size_t nrows,
                                              std::size_t total_elems, LineBase line_base,
                                              std::size_t row_stride = 0) {
    const exec_options<T> opts{fct};
    parallel_for(pool, nrows, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        if (row_stride) {
            st.plan->execute_many(data + line_base(b), e - b, row_stride, opts);
            return;
        }
        for (std::size_t r = b; r < e; ++r)
            st.plan->execute(span<std::complex<T>>(data + line_base(r), len), opts);
    });
}

enum class line_route : std::uint8_t {
    col_dif,
    transposed,
};

template<typename T>
[[nodiscard]] inline std::size_t transpose_group([[maybe_unused]] std::size_t len,
                                                 std::size_t run_len) {
    return std::min<std::size_t>(run_len, 2 * kCacheLine / sizeof(std::complex<T>));
}

template<typename T>
[[nodiscard]] inline line_route choose_line_route(const nd_axis_state<T>& st, std::size_t len,
                                                  std::size_t inner, std::size_t run_len,
                                                  std::size_t nthreads) {
    if (!st.dif) return line_route::transposed;
    if (nthreads <= 1 && col_budget_block<T>(len, 1) < 2 * xsimd::batch<T>::size)
        return line_route::transposed;
    if (2 * run_len <= xsimd::batch<T>::size
        && len * inner * sizeof(std::complex<T>) > col_cache_budget(nthreads))
        return line_route::transposed;
    return line_route::col_dif;
}

template<bool Gather, typename T>
void move_run(std::complex<T>* line, std::size_t inner, std::size_t len, std::size_t gw,
              std::complex<T>* buf) {
    for (std::size_t p = 0; p < len; ++p)
        for (std::size_t g = 0; g < gw; ++g) {
            if constexpr (Gather) buf[g * len + p] = line[p * inner + g];
            else line[p * inner + g] = buf[g * len + p];
        }
}

template<typename T, typename LineBase>
ADM_ALWAYS_INLINE void apply_lines_strided(std::complex<T>* data, std::size_t len,
                                           std::size_t inner, bool forward,
                                           const nd_axis_state<T>& st, std::optional<T> fct,
                                           thread_pool* pool, std::size_t nruns,
                                           std::size_t run_len, std::size_t total_elems,
                                           LineBase line_base) {
    const std::size_t nthreads = pool_size(pool);
    if (choose_line_route<T>(st, len, inner, run_len, nthreads) == line_route::col_dif) {
        const std::size_t Bt = nd_col_block<T>(len, run_len, nthreads, nruns);
        const std::size_t ntiles = (run_len + Bt - 1) / Bt;
        const std::size_t nunits = nruns * ntiles;
        const T scale = fct.value_or(forward ? T(1) : T(1) / static_cast<T>(len));
        parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
            std::size_t run = b / ntiles, tile = b % ntiles;
            auto* line = data + line_base(run);
            if (st.col_codelet) {
                for (std::size_t u = b; u < e; ++u) {
                    const std::size_t c0 = tile * Bt;
                    const std::size_t bc = std::min(Bt, run_len - c0);
                    col_codelet_dispatch<T>(forward, line + c0, inner, line + c0, inner, bc,
                                            len, scale);
                    if (++tile == ntiles) { tile = 0; line = data + line_base(++run); }
                }
                return;
            }
            soa_scratch<T, 4> sc(len * Bt);
            for (std::size_t u = b; u < e; ++u) {
                const std::size_t c0 = tile * Bt;
                const std::size_t bc = std::min(Bt, run_len - c0);
                col_dif_dispatch<T>(forward, line + c0, len, inner, bc,
                                    sc.buf(0), sc.buf(1), sc.buf(2), sc.buf(3), st.dtw, scale);
                if (++tile == ntiles) { tile = 0; line = data + line_base(++run); }
            }
        });
        return;
    }
    std::size_t group = transpose_group<T>(len, run_len);
    if (pool && nruns * ((run_len + group - 1) / group) < 2 * nthreads) {
        constexpr std::size_t kLine = kCacheLine / sizeof(std::complex<T>);
        const std::size_t target =
            ((run_len + 2 * nthreads - 1) / (2 * nthreads) + kLine - 1) / kLine * kLine;
        group = std::min(group, std::max(kLine, target));
    }
    const std::size_t ngroups = (run_len + group - 1) / group;
    const std::size_t nunits = nruns * ngroups;
    const exec_options<T> opts{fct};
    parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        soa_scratch<T, 1> scratch(2 * len * group);
        auto* const buf = reinterpret_cast<std::complex<T>*>(scratch.buf(0));
        for (std::size_t u = b; u < e; ++u) {
            const std::size_t c0 = (u % ngroups) * group;
            const std::size_t gw = std::min(group, run_len - c0);
            auto* const line = data + line_base(u / ngroups) + c0;
            move_run<true>(line, inner, len, gw, buf);
            st.plan->execute_many(buf, gw, len, opts);
            move_run<false>(line, inner, len, gw, buf);
        }
    });
}

template<typename T, typename SrcBase, typename DstBase>
ADM_ALWAYS_INLINE void
apply_lines_strided_oop(const std::complex<T>* src, std::size_t src_line,
                        std::size_t src_batch, std::complex<T>* dst,
                        std::size_t dst_line, std::size_t dst_batch, std::size_t len,
                        bool forward, const nd_axis_state<T>& st, std::optional<T> fct,
                        thread_pool* pool, std::size_t nruns, std::size_t run_len,
                        std::size_t total_elems, SrcBase src_base, DstBase dst_base) {
    const std::size_t nthreads = pool_size(pool);
    if (src_batch == 1 && dst_batch == 1 &&
        choose_line_route<T>(st, len, src_line, run_len, nthreads) ==
            line_route::col_dif) {
        const std::size_t Bt = nd_col_block<T>(len, run_len, nthreads, nruns);
        const std::size_t ntiles = (run_len + Bt - 1) / Bt;
        const std::size_t nunits = nruns * ntiles;
        const T scale = fct.value_or(forward ? T(1) : T(1) / static_cast<T>(len));
        parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
            std::size_t run = b / ntiles, tile = b % ntiles;
            const std::complex<T>* sline = src + src_base(run);
            std::complex<T>* dline = dst + dst_base(run);
            if (st.col_codelet) {
                for (std::size_t u = b; u < e; ++u) {
                    const std::size_t c0 = tile * Bt;
                    const std::size_t bc = std::min(Bt, run_len - c0);
                    col_codelet_dispatch<T>(forward, sline + c0, src_line, dline + c0,
                                            dst_line, bc, len, scale);
                    if (++tile == ntiles) {
                        tile = 0;
                        sline = src + src_base(++run);
                        dline = dst + dst_base(run);
                    }
                }
                return;
            }
            soa_scratch<T, 4> sc(len * Bt);
            for (std::size_t u = b; u < e; ++u) {
                const std::size_t c0 = tile * Bt;
                const std::size_t bc = std::min(Bt, run_len - c0);
                col_dif_dispatch<T>(forward, dline + c0, len, dst_line, bc, sc.buf(0),
                                    sc.buf(1), sc.buf(2), sc.buf(3), st.dtw, scale,
                                    sline + c0, src_line);
                if (++tile == ntiles) {
                    tile = 0;
                    sline = src + src_base(++run);
                    dline = dst + dst_base(run);
                }
            }
        });
        return;
    }
    std::size_t group = transpose_group<T>(len, run_len);
    if (pool && nruns * ((run_len + group - 1) / group) < 2 * nthreads) {
        constexpr std::size_t kLine = kCacheLine / sizeof(std::complex<T>);
        const std::size_t target =
            ((run_len + 2 * nthreads - 1) / (2 * nthreads) + kLine - 1) / kLine * kLine;
        group = std::min(group, std::max(kLine, target));
    }
    const std::size_t ngroups = (run_len + group - 1) / group;
    const std::size_t nunits = nruns * ngroups;
    const exec_options<T> opts{fct};
    parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        soa_scratch<T, 1> scratch(2 * len * group);
        auto* const buf = reinterpret_cast<std::complex<T>*>(scratch.buf(0));
        for (std::size_t u = b; u < e; ++u) {
            const std::size_t c0 = (u % ngroups) * group;
            const std::size_t gw = std::min(group, run_len - c0);
            const std::size_t r = u / ngroups;
            const std::complex<T>* const sline = src + src_base(r) + c0 * src_batch;
            std::complex<T>* const dline = dst + dst_base(r) + c0 * dst_batch;
            for (std::size_t p = 0; p < len; ++p)
                for (std::size_t g = 0; g < gw; ++g)
                    buf[g * len + p] = sline[p * src_line + g * src_batch];
            st.plan->execute_many(buf, gw, len, opts);
            for (std::size_t p = 0; p < len; ++p)
                for (std::size_t g = 0; g < gw; ++g)
                    dline[p * dst_line + g * dst_batch] = buf[g * len + p];
        }
    });
}

enum class band_form : std::uint8_t {
    packed,
    merged,
    split,
};

inline constexpr std::size_t kPackMinPasses = 5;

[[nodiscard]] constexpr band_form choose_band_form(bool dif, std::size_t n_passes,
                                                  std::size_t w0, std::size_t w1,
                                                  std::size_t simd_width) {
    if (w1 == 0) return band_form::split;
    if (dif && w0 + w1 <= simd_width && n_passes >= kPackMinPasses) return band_form::packed;
    if (w0 == w1) return band_form::merged;
    return band_form::split;
}

template<typename T, typename LineBases>
void apply_bands_strided_packed(std::complex<T>* data, std::size_t len, std::size_t inner,
                                bool forward, const nd_axis_state<T>& st, std::optional<T> fct,
                                thread_pool* pool, std::size_t nruns, std::size_t w0,
                                std::size_t w1, std::size_t total_elems, LineBases line_bases) {
    const std::size_t Bp = w0 + w1;
    const T scale = fct.value_or(forward ? T(1) : T(1) / static_cast<T>(len));
    const auto cp0 = real_run_copy<T>::make(2 * w0);
    const auto cp1 = real_run_copy<T>::make(2 * w1);
    parallel_for(pool, nruns, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        soa_scratch<T, 4> sc(len * Bp);
        soa_scratch<T, 1> slab_re(2 * len * Bp);
        auto* const slab = reinterpret_cast<std::complex<T>*>(slab_re.buf(0));
        for (std::size_t r = b; r < e; ++r) {
            const auto [o0, o1] = line_bases(r);
            auto* const l0 = reinterpret_cast<T*>(data + o0);
            auto* const l1 = reinterpret_cast<T*>(data + o1);
            auto* const sl = slab_re.buf(0);
            for (std::size_t p = 0; p < len; ++p) {
                cp0(l0 + 2 * p * inner, sl + 2 * p * Bp);
                cp1(l1 + 2 * p * inner, sl + 2 * p * Bp + 2 * w0);
            }
            col_dif_dispatch<T>(forward, slab, len, Bp, Bp, sc.buf(0), sc.buf(1),
                                sc.buf(2), sc.buf(3), st.dtw, scale);
            for (std::size_t p = 0; p < len; ++p) {
                cp0(sl + 2 * p * Bp, l0 + 2 * p * inner);
                cp1(sl + 2 * p * Bp + 2 * w0, l1 + 2 * p * inner);
            }
        }
    });
}

template<typename T>
void nd_apply_axis(std::complex<T>* data, std::size_t total, std::size_t len,
                   std::size_t inner, bool innermost, bool is_forward,
                   const nd_axis_state<T>& st, std::optional<T> axis_fct,
                   thread_pool* pool = nullptr) {
    if (len <= 1) return;
    const std::size_t outer = total / (len * inner);
    if (innermost)
        apply_lines_contiguous<T>(data, len, st, axis_fct, pool, outer, total,
                                  [len](std::size_t r) { return r * len; }, len);
    else
        apply_lines_strided<T>(data, len, inner, is_forward, st, axis_fct, pool, outer, inner,
                               total, [len, inner](std::size_t r) { return r * (len * inner); });
}

template<typename T>
class nd_runtime_plan {
    struct M {
        std::vector<std::size_t> shape;
        bool is_forward;
        std::size_t total;
        std::vector<nd_axis_state<T>> axes;
        std::unique_ptr<thread_pool> pool;
    } m;

public:
    nd_runtime_plan(span<const std::size_t> shape, bool is_forward,
                    std::size_t nthreads = 1,
                    admiral::effort eff = admiral::effort::estimate);
    void execute(std::complex<T>* data, const exec_options<T>& opts = {}) const;
    void execute(const std::complex<T>* src, std::complex<T>* dst,
                 const exec_options<T>& opts = {}) const;

    [[nodiscard]] std::size_t size() const noexcept { return m.total; }

private:
    ADM_NOINLINE void execute_nd(std::complex<T>* data, const exec_options<T>& opts) const;
    ADM_NOINLINE void execute_nd(const std::complex<T>* src, std::complex<T>* dst,
                                 const exec_options<T>& opts) const;

    ADM_NOINLINE ADM_COLD void trace(unsigned level, const char* how) const {
        dbg_print("rank=", m.shape.size(), m.is_forward ? " fwd " : " inv ", how, " total=",
                  m.total, m.pool ? " threaded" : " serial");
        if (level < dbg_shape) return;
        dbg_print_seq("  shape", m.shape);
        for (std::size_t d = 0; d < m.shape.size(); ++d) {
            const nd_axis_state<T>& ax = m.axes[d];
            dbg_print("  axis ", d, " len=", m.shape[d], " ",
                      ax.plan ? ax.plan->route_name() : "col_dif");
        }
    }

    struct scale_plan {
        bool custom;
        T fct;
        std::size_t scale_axis;
    };
    [[nodiscard]] scale_plan make_scale_plan(std::optional<T> fct) const {
        const T def = m.is_forward ? T(1) : T(1) / static_cast<T>(m.total);
        const T f = fct.value_or(def);
        std::size_t axis = m.shape.size();
        if (f != def)
            for (std::size_t di = 0; di < m.shape.size(); ++di) {
                const std::size_t d = m.shape.size() - 1 - di;
                if (m.shape[d] > 1) { axis = d; break; }
            }
        return {f != def, f, axis};
    }
    [[nodiscard]] std::optional<T> axis_fct(const scale_plan& sp, std::size_t d) const {
        if (!sp.custom) return std::nullopt;
        return d == sp.scale_axis ? sp.fct : T(1);
    }
};

template<typename T>
nd_runtime_plan<T>::nd_runtime_plan(span<const std::size_t> shape, bool is_forward,
                                    std::size_t nthreads, admiral::effort eff) {
    m.shape.assign(shape.begin(), shape.end());
    m.is_forward = is_forward;
    const auto total = extent_product(m.shape);
    if (!total) ADM_UNLIKELY throw size_error("Plan size must be greater than 0");
    m.total = *total;
    bool batch_threadable = false;
    for (const std::size_t d : m.shape) {
        if (d == 0) continue;
        const std::size_t units = m.total / d;
        batch_threadable |= units >= 2 && m.total >= kThreadMinElems && d > 1;
    }
    if (nthreads == 0 && batch_threadable) {
        std::size_t dispatches = 0;
        double work_cyc = 0.0;
        for (const std::size_t d : m.shape) {
            if (d <= 1) continue;
            ++dispatches;
            work_cyc += double(m.total / d) * line_work_cyc<T>(d);
        }
        const unsigned cls = m.shape.size() >= 3 ? 2 : 1;
        nthreads = resolve_nthreads(0, m.total, dispatches, work_cyc / core_cyc_per_ns(), cls);
    }
    m.axes.resize(m.shape.size());
    std::size_t inner = 1;
    for (std::size_t di = 0; di < m.shape.size(); ++di) {
        const std::size_t d = m.shape.size() - 1 - di;
        const std::size_t units = m.total / m.shape[d];
        const bool threads_above = units >= 2 && m.total >= kThreadMinElems;
        const std::size_t axis_threads = threads_above ? 1 : nthreads;
        m.axes[d] = make_nd_axis_state<T>(m.shape[d], inner, is_forward,
                                          d == m.shape.size() - 1, axis_threads,
                                          eff);
        inner *= m.shape[d];
    }
    if (nthreads > 1 && batch_threadable)
        m.pool = std::make_unique<thread_pool>(nthreads);
}

template<typename T>
void nd_runtime_plan<T>::execute(std::complex<T>* data, const exec_options<T>& opts) const {
    const std::size_t ndim = m.shape.size();
    if (ndim == 1) {
        const nd_axis_state<T>& ax0 = m.axes[0];
        if (ax0.plan) {
            const scale_plan sp = make_scale_plan(opts.fct);
            ax0.plan->execute(span<std::complex<T>>(data, m.total),
                              {sp.custom ? std::optional<T>(sp.fct) : std::nullopt, opts.debug});
            return;
        }
    }
    execute_nd(data, opts);
}

template<typename T>
void nd_runtime_plan<T>::execute_nd(std::complex<T>* data, const exec_options<T>& opts) const {
    const std::size_t ndim = m.shape.size();
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "in-place");
    const scale_plan sp = make_scale_plan(opts.fct);
    std::size_t inner = 1;
    for (std::size_t di = 0; di < ndim; ++di) {
        const std::size_t d = ndim - 1 - di;
        nd_apply_axis<T>(data, m.total, m.shape[d], inner,
                         d == ndim - 1, m.is_forward, m.axes[d],
                         axis_fct(sp, d), m.pool.get());
        inner *= m.shape[d];
    }
    if (sp.custom && sp.scale_axis == ndim) scale_inplace(data, m.total, sp.fct);
}

template<typename T>
void nd_runtime_plan<T>::execute(const std::complex<T>* src, std::complex<T>* dst,
                                 const exec_options<T>& opts) const {
    if (src == dst) { execute(dst, opts); return; }
    if (m.shape.empty()) {
        const scale_plan sp = make_scale_plan(opts.fct);
        *dst = sp.custom ? *src * sp.fct : *src;
        return;
    }
    if (m.shape.size() == 1) {
        const nd_axis_state<T>& ax0 = m.axes[0];
        if (ax0.plan) {
            const scale_plan sp = make_scale_plan(opts.fct);
            ax0.plan->execute(src, dst,
                              {sp.custom ? std::optional<T>(sp.fct) : std::nullopt, opts.debug});
            return;
        }
    }
    execute_nd(src, dst, opts);
}

template<typename T>
void nd_runtime_plan<T>::execute_nd(const std::complex<T>* src, std::complex<T>* dst,
                                    const exec_options<T>& opts) const {
    const std::size_t ndim = m.shape.size();
    const std::size_t len = m.shape[ndim - 1];
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "oop");
    const std::size_t rows = m.total / len;
    const nd_axis_state<T>& in_st = m.axes[ndim - 1];
    const scale_plan sp = make_scale_plan(opts.fct);
    const exec_options<T> row_opts{axis_fct(sp, ndim - 1)};
    parallel_for(m.pool.get(), rows, m.total, [&](std::size_t b, std::size_t e, std::size_t) {
        if (len <= 32 && is_codelet_catalog(len)) {
            const T fct = row_opts.fct.value_or(m.is_forward ? T(1) : T(1) / static_cast<T>(len));
            if (m.is_forward)
                codelet_dispatch_many_oop<T, true >(src + b * len, dst + b * len, e - b,
                                                    len, len, len, fct);
            else
                codelet_dispatch_many_oop<T, false>(src + b * len, dst + b * len, e - b,
                                                    len, len, len, fct);
            return;
        }
        for (std::size_t r = b; r < e; ++r)
            in_st.plan->execute(src + r * len, dst + r * len, row_opts);
    });
    std::size_t inner = len;
    for (std::size_t di = 1; di < ndim; ++di) {
        const std::size_t d = ndim - 1 - di;
        nd_apply_axis<T>(dst, m.total, m.shape[d], inner,
                         false, m.is_forward, m.axes[d],
                         axis_fct(sp, d), m.pool.get());
        inner *= m.shape[d];
    }
    if (sp.custom && sp.scale_axis == ndim) scale_inplace(dst, m.total, sp.fct);
}

extern template class nd_runtime_plan<float>;
extern template class nd_runtime_plan<double>;

}
}

#include "undef_macros.hpp"
