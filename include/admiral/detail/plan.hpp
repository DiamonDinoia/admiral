#pragma once

#include <admiral/admiral.hpp>

#include <admiral/detail/config.hpp>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include "cxx_compat.hpp"

#include "base_cost_model.hpp"
#include "bluestein.hpp"
#include "debug.hpp"
#include "dif_driver.hpp"
#include "four_step.hpp"
#include "four_step_large.hpp"
#include "math.hpp"
#include "good_thomas.hpp"
#include "rader.hpp"
#include "scratch.hpp"
#include "thread_pool.hpp"
#include "twiddles.hpp"

#include "macros.hpp"

namespace admiral {
namespace detail {

template<typename T>
struct exec_options {
    std::optional<T> fct = std::nullopt;
    unsigned debug = dbg_off;
};

template<typename T>
class plan_impl {
public:
    enum class route_kind {
        codelet,
        iterative_dif,
        four_step,
        four_step_batched,
        four_step_large,
        rader,
        bluestein,
        good_thomas
    };

    struct measured_choice {
        route_kind route;
        dif_factor_plan dif_chain;
    };

    static constexpr std::size_t kLargeDispatches = 5;

    static constexpr bool good_thomas_available(std::size_t n) noexcept {
        return good_thomas_catalog::available<T>(n);
    }

    static constexpr bool route_available(route_kind rk, std::size_t size) {
        switch (rk) {
        case route_kind::codelet:           return is_codelet_catalog(size);
        case route_kind::good_thomas:       return good_thomas_available(size);
        case route_kind::iterative_dif:     return dif_chain_supported(size);
        case route_kind::four_step:         return four_step_supported(size);
        case route_kind::four_step_batched: return four_step_batched_supported<T>(size);
        case route_kind::four_step_large:   return choose_large_split(size).valid();
        case route_kind::rader:             return rader_supported(size);
        case route_kind::bluestein:         return true;
        }
        return false;
    }

private:
    struct codelet_state {};
    struct good_thomas_state {};
    struct dif_state { dif_twiddle_set<T> tw; };
    struct four_step_state {
        four_step_split split;
        std::vector<std::complex<T>> tw;
    };
    using route_state = std::variant<codelet_state, good_thomas_state, dif_state,
                                     four_step_state, four_step_batched_plan<T>,
                                     four_step_large_plan<T>,
                                     rader_plan<T>, bluestein_plan<T>>;

    struct M {
        std::size_t size;
        bool is_forward;
        route_kind route;
        route_state st;
        std::unique_ptr<thread_pool> pool;
    } m;

    [[nodiscard]] static std::unique_ptr<thread_pool> make_route_pool(std::size_t nthreads,
                                                                      std::size_t size,
                                                                      route_kind rk) {
        return (nthreads > 1 && size > 1 && rk == route_kind::four_step_large)
                   ? std::make_unique<thread_pool>(nthreads)
                   : nullptr;
    }

public:

    plan_impl(std::size_t size, bool is_forward, route_kind forced, std::size_t nthreads = 1);

    plan_impl(std::size_t size, bool is_forward, std::size_t nthreads = 1,
              const dif_factor_plan* dif_override = nullptr,
              admiral::effort eff = admiral::effort::estimate);

private:
    struct routed_plan {
        measured_choice ch;
        std::size_t nthreads;
        const dif_factor_plan* dif_override;
    };

    plan_impl(std::size_t size, bool is_forward, routed_plan rp);
    plan_impl(std::size_t size, bool is_forward, std::size_t nthreads,
              const dif_factor_plan* dif_override, measured_choice choice)
        : plan_impl(size, is_forward, routed_plan{choice, nthreads, dif_override}) {}

    static double large_work_ns(std::size_t size) {
        const large_split sp = choose_large_split(size);
        const double cyc =
            sp.valid() ? kFourStepOverhead * (double(sp.n1) * line_work_cyc<T>(sp.n2) +
                                              double(sp.n2) * line_work_cyc<T>(sp.n1))
                       : line_work_cyc<T>(size);
        return cyc / core_cyc_per_ns();
    }

    static routed_plan route_plan(std::size_t size, bool is_forward, std::size_t nthreads,
                                  const dif_factor_plan* dif_override, admiral::effort eff) {
        if (size == 0) ADM_UNLIKELY return {measured_choice{}, 1, dif_override};
        if (dif_override) return {measured_choice{route_kind::iterative_dif, {}}, nthreads, dif_override};
        const auto elect = [&](std::size_t nt) {
            return eff != admiral::effort::estimate
                       ? measured_route(size, is_forward, nt)
                       : measured_choice{select_route(size, nt), {}};
        };
        if (nthreads != 0) return {elect(nthreads), nthreads, dif_override};
        const std::size_t P = resolve_nthreads(0);
        measured_choice route_est{select_route(size, P), {}};
        if (route_est.route != route_kind::four_step_large)
            return {route_est, 1, dif_override};
        const std::size_t nt = resolve_nthreads(0, size, kLargeDispatches, large_work_ns(size), 0);
        measured_choice ch = elect(nt);
        if (ch.route != route_kind::four_step_large)
            return {ch, 1, dif_override};
        return {ch, nt, dif_override};
    }

    void emplace_route_state(std::size_t size, bool is_forward, const dif_factor_plan* dif_override) {
        switch (m.route) {
        case route_kind::codelet:
            break;
        case route_kind::good_thomas:
            m.st.template emplace<good_thomas_state>();
            break;
        case route_kind::iterative_dif:
            m.st.template emplace<dif_state>(
                dif_state{build_dif_twiddle_set<T>(size, dif_override)});
            break;
        case route_kind::four_step: {
            const four_step_split split = choose_four_step_split_exec<T>(size);
            m.st.template emplace<four_step_state>(four_step_state{
                split, is_forward
                    ? build_four_step_twiddles<T, true>(split.n1, split.n2)
                    : build_four_step_twiddles<T, false>(split.n1, split.n2)});
            break;
        }
        case route_kind::four_step_batched:
            m.st.template emplace<four_step_batched_plan<T>>(size, is_forward);
            break;
        case route_kind::four_step_large:
            m.st.template emplace<four_step_large_plan<T>>(size, is_forward);
            break;
        case route_kind::rader:
            m.st.template emplace<rader_plan<T>>(size, is_forward);
            break;
        case route_kind::bluestein:
            m.st.template emplace<bluestein_plan<T>>(size, is_forward);
            break;
        }
    }

public:

    void execute(span<std::complex<T>> data, const exec_options<T>& opts = {}) const;

    void execute(const std::complex<T>* src, std::complex<T>* dst,
                 const exec_options<T>& opts = {}) const;

    void execute_many(std::complex<T>* data, std::size_t n, std::size_t stride,
                      const exec_options<T>& opts = {}) const;

private:
    ADM_NOINLINE ADM_COLD void trace(unsigned level, const char* how, T fct,
                                     std::size_t lines = 1, std::size_t stride = 0) const {
        dbg_print("N=", m.size, m.is_forward ? " fwd " : " inv ", route_name(), " ", how,
                  " fct=", static_cast<double>(fct), m.pool ? " threaded" : " serial");
        if (lines != 1) dbg_print("  lines=", lines, " stride=", stride);
        if (level < dbg_shape) return;
        std::visit([](const auto& st) {
            using S = std::decay_t<decltype(st)>;
            if constexpr (std::is_same_v<S, dif_state>) dbg_print_seq("  radices", st.tw.radices);
            else if constexpr (std::is_same_v<S, four_step_state>)
                dbg_print("  n1=", st.split.n1, " n2=", st.split.n2);
        }, m.st);
        if (level < dbg_cost) return;
        trace_ranking();
    }

    ADM_NOINLINE ADM_COLD void trace_ranking() const {
        const auto& r = base_route_ranking<T>(m.size);
        if (r.count == 0) {
            dbg_print("  cost model: N outside the fitted domain, gate ladder decided");
            return;
        }
        for (std::size_t i = 0; i < r.count; ++i) {
            const route_kind rk = form_to_route(r.form[i]);
            dbg_print("  model[", i, "] ", base_form_name(r.form[i]), " cyc=",
                      ct_exp(double(r.log_cyc[i])),
                      route_available(rk, m.size) ? " available" : " UNAVAILABLE");
        }
    }

    template<bool Forward>
    void execute_many_impl(std::complex<T>* data, std::size_t n, std::size_t stride, T fct) const {
        std::visit(
            [&](const auto& st) {
                for (std::size_t r = 0; r < n; ++r) {
                    std::complex<T>* p = data + r * stride;
                    if constexpr (std::is_same_v<std::decay_t<decltype(st)>,
                                                 four_step_large_plan<T>>)
                        execute_route<Forward>(st, p, p, fct, m.pool.get());
                    else
                        execute_route<Forward>(st, p, p, fct);
                }
            },
            m.st);
    }

    template<bool Forward>
    void execute_impl(const std::complex<T>* src, std::complex<T>* dst, T fct) const {
        std::visit([&](const auto& st) {
            if constexpr (std::is_same_v<std::decay_t<decltype(st)>, four_step_large_plan<T>>)
                execute_route<Forward>(st, src, dst, fct, m.pool.get());
            else
                execute_route<Forward>(st, src, dst, fct);
        }, m.st);
    }

public:

    [[nodiscard]] std::size_t size() const noexcept { return m.size; }
    [[nodiscard]] bool is_forward() const noexcept { return m.is_forward; }

    [[nodiscard]] const char* route_name() const noexcept {
        switch (m.route) {
        case route_kind::good_thomas:               return "good_thomas";
        case route_kind::codelet:           return "codelet";
        case route_kind::iterative_dif:     return "iterative_dif";
        case route_kind::four_step:         return "four_step";
        case route_kind::four_step_batched: return "four_step_batched";
        case route_kind::four_step_large:   return "four_step_large";
        case route_kind::rader:             return "rader";
        case route_kind::bluestein:         return "bluestein";
        }
        return "?";
    }
    [[nodiscard]] four_step_split four_step_split_used() const noexcept {
        const auto* fs = std::get_if<four_step_state>(&m.st);
        return fs ? fs->split : four_step_split{};
    }

private:
    static constexpr route_kind form_to_route(base_form f) {
        switch (f) {
        case base_form::codelet:           return route_kind::codelet;
        case base_form::iterative_dif:     return route_kind::iterative_dif;
        case base_form::good_thomas:       return route_kind::good_thomas;
        case base_form::four_step:         return route_kind::four_step;
        case base_form::four_step_batched: return route_kind::four_step_batched;
        case base_form::rader:             return route_kind::rader;
        case base_form::bluestein:         return route_kind::bluestein;
        }
        ADM_UNREACHABLE();
    }

    static measured_choice measure_route(std::size_t size, bool is_forward, std::size_t nthreads);

    static measured_choice measured_route(std::size_t size, [[maybe_unused]] bool is_forward,
                                          std::size_t nthreads) {
        if constexpr (adm_measure) return measure_route(size, is_forward, nthreads);
        else return measured_choice{select_route(size, nthreads), {}};
    }

    static constexpr std::size_t large_route_bytes(std::size_t nthreads) {
        if (nthreads > 1) return large_route_threaded_bytes(nthreads);
        if constexpr (sizeof(T) == 8) return kLargeRouteSerialF64Bytes;
        else return kLargeRouteSerialF32Bytes;
    }

    static constexpr bool large_route_admits(std::size_t size, std::size_t nthreads) {
        if (!four_step_large_supported(size, sizeof(std::complex<T>),
                                       large_route_bytes(nthreads)))
            return false;
        if (nthreads > 1) {
            const large_split sp = choose_large_split(size);
            return sp.n2 % sp.n1 == 0;
        }
        if constexpr (sizeof(T) == 4)
            if (size * sizeof(std::complex<T>) > kLargeRouteSerialF32MaxBytes) return false;
        return four_step_large_fused_shape<T>(size);
    }

    [[nodiscard]] static bool dif_chain_supported(std::size_t size) {
        if (detail::has_single_bit(size) || is_codelet_supported(size)) return true;
        return dif_chain_shape_ok(size, dif_elected_chain<T>(size));
    }

public:
    [[nodiscard]] static bool dif_chain_shape_ok(std::size_t size, const dif_factor_plan& p) {
        return detail::dif_chain_shape_ok<T>(size, p);
    }

private:
    static route_kind select_route(std::size_t size, std::size_t nthreads) {
        {
            const auto& ranking = base_route_ranking<T>(size);
            for (std::size_t i = 0; i < ranking.count; ++i) {
                const route_kind rk = form_to_route(ranking.form[i]);
                if (route_available(rk, size)) return rk;
            }
        }
        if (is_codelet_catalog(size)) return route_kind::codelet;
        if (four_step_batched_supported<T>(size)) return route_kind::four_step_batched;
        if (large_route_admits(size, nthreads)) return route_kind::four_step_large;
        if (dif_chain_supported(size))
            return route_kind::iterative_dif;
        if (four_step_supported(size) && four_step_beats_bluestein<T>(size))
            return route_kind::four_step;
        if (rader_supported(size) && rader_beats_bluestein<T>(size))
            return route_kind::rader;
        return route_kind::bluestein;
    }

    template<bool Forward>
    void execute_route(const good_thomas_state&, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        good_thomas_run<T, Forward>(in, out, m.size);
        apply_scale(out, fct);
    }

    template<bool Forward>
    void execute_route(const codelet_state&, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        codelet_dispatch<T, Forward>(in, out, m.size);
        apply_scale(out, fct);
    }

    template<bool Forward>
    void execute_route(const dif_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        dif_execute_in_place<T>(Forward, in, out, m.size, st.tw, fct);
    }

    template<bool Forward>
    void execute_route(const four_step_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        soa_scratch<T, 1> scratch(2 * m.size);
        four_step_execute<T, Forward>(in, out, st.split.n1, st.split.n2, st.tw.data(),
                                      reinterpret_cast<std::complex<T>*>(scratch.buf(0)));
        apply_scale(out, fct);
    }

    template<bool Forward>
    void execute_route(const four_step_batched_plan<T>& fsb, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        fsb.execute(in, out);
        apply_scale(out, fct);
    }

    template<bool Forward>
    void execute_route(const four_step_large_plan<T>& fsl, const std::complex<T>* in,
                       std::complex<T>* out, T fct, thread_pool* pool = nullptr) const {
        fsl.execute(in, out, fct, pool);
    }

    template<bool Forward>
    void execute_route(const rader_plan<T>& rp, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        rp.execute(in, out);
        apply_scale(out, fct);
    }

    template<bool Forward>
    void execute_route(const bluestein_plan<T>& bp, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        bp.execute(in, out, fct);
    }

    void apply_scale(std::complex<T>* out, T fct) const {
        if (fct != T(1)) scale_inplace(out, m.size, fct);
    }
};

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, route_kind forced,
                        std::size_t nthreads)
    : m{size, is_forward, forced, {},
        make_route_pool(nthreads, size, forced)}
{
    if (size == 0) ADM_UNLIKELY
        throw size_error("Plan size must be greater than 0");
    if (!route_available(forced, size))
        throw unsupported_error("force-route: kernel unavailable for this size/precision");
    emplace_route_state(size, is_forward, nullptr);
}

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, std::size_t nthreads,
                        const dif_factor_plan* dif_override, admiral::effort eff)
    : plan_impl(size, is_forward, route_plan(size, is_forward, nthreads, dif_override, eff)) {}

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, routed_plan rp)
    : m{size, is_forward,
        rp.ch.route,
        {},
        make_route_pool(rp.nthreads, size, rp.ch.route)}
{
    if (size == 0) ADM_UNLIKELY {
        throw size_error("Plan size must be greater than 0");
    }
    if (rp.dif_override && !route_available(route_kind::iterative_dif, size))
        throw unsupported_error("forced dif: size is not 11-smooth");
    const dif_factor_plan* chain =
        rp.dif_override != nullptr        ? rp.dif_override
        : rp.ch.dif_chain.count != 0 ? &rp.ch.dif_chain
                                     : nullptr;
    emplace_route_state(size, is_forward, chain);
}

inline constexpr std::size_t kMeasureCandidates = 8;
inline constexpr std::size_t kMeasureReps = 5;
inline constexpr std::size_t kMeasureMaxCandidates = 4;
inline constexpr std::chrono::nanoseconds::rep kMeasureMinNs = 50;
inline constexpr std::chrono::nanoseconds::rep kMeasureSampleNs = 4000;
[[nodiscard]] constexpr std::size_t measure_batch(std::chrono::nanoseconds::rep one_ns) {
    if (one_ns >= kMeasureSampleNs) return 1;
    return std::size_t(kMeasureSampleNs / (std::max)(one_ns, std::chrono::nanoseconds::rep{1})) + 1;
}
inline constexpr double kMeasureRejectRatio = 1.25;
inline constexpr double kMeasureInf = 1e300;

template<typename T>
typename plan_impl<T>::measured_choice
plan_impl<T>::measure_route(std::size_t size, bool is_forward, std::size_t nthreads) {
    const route_kind fallback = select_route(size, nthreads);
    using clock = std::chrono::steady_clock;

    measured_choice pick{fallback, {}};

    route_kind cands[kMeasureMaxCandidates];
    std::size_t nc = 0;
    const auto offer = [&](route_kind rk) {
        if (nc >= kMeasureMaxCandidates || !route_available(rk, size)) return;
        for (std::size_t i = 0; i < nc; ++i)
            if (cands[i] == rk) return;
        cands[nc++] = rk;
    };
    if (size >= BASE_MODEL_NMIN && size <= BASE_MODEL_NMAX) {
        offer(fallback);
        const auto& ranking = base_route_ranking<T>(size);
        for (std::size_t i = 0; i < ranking.count && nc + 1 < kMeasureMaxCandidates; ++i)
            offer(form_to_route(ranking.form[i]));
        if constexpr (sizeof(T) == 4)
            offer(route_kind::four_step_batched);
    }

    if (size > BASE_MODEL_NMAX && nthreads > 1) {
        const large_split sp = choose_large_split(size);
        if (sp.valid() && sp.n2 % sp.n1 == 0) {
            offer(fallback);
            offer(fallback == route_kind::four_step_large ? route_kind::iterative_dif
                                                         : route_kind::four_step_large);
        }
    }

    const dif_chain_list chain_cands =
        fallback == route_kind::iterative_dif ? dif_chain_candidates<T>(size) : dif_chain_list{};
    bool race_chains = false;
    for (std::size_t i = 0; i < chain_cands.count && !race_chains; ++i)
        race_chains = detail::dif_chain_shape_ok<T>(size, chain_cands[i]);

    if (nc < 2 && !race_chains) return pick;

    std::vector<std::complex<T>> in(size), out(size);
    for (std::size_t i = 0; i < size; ++i)
        in[i] = {T(0.5 * int(i % 7) - 1), T(0.25 * int(i % 5) - 0.5)};

    double unit = kMeasureInf;
    std::size_t raced = 0;
    const auto have_budget = [&] { return raced < kMeasureCandidates; };

    double best_ns = kMeasureInf;
    const auto time_plan = [&](plan_impl<T>& trial) {
        trial.execute(in.data(), out.data());
        trial.execute(in.data(), out.data());
        const std::size_t inner = [&] {
            const auto a = clock::now();
            trial.execute(in.data(), out.data());
            return measure_batch(std::chrono::nanoseconds(clock::now() - a).count());
        }();
        const double u = unit;
        double best = kMeasureInf;
        for (std::size_t r = 0; r < kMeasureReps; ++r) {
            const auto a = clock::now();
            for (std::size_t k = 0; k < inner; ++k) trial.execute(in.data(), out.data());
            const auto span = std::chrono::nanoseconds(clock::now() - a).count();
            best = (std::min)(best, double((std::max)(span, kMeasureMinNs)) / double(inner));
            unit = (std::min)(unit, best);
            if (r > 0 && best > u * kMeasureRejectRatio) break;
        }
        ++raced;
        return best;
    };

    for (std::size_t c = 0; c < nc && have_budget(); ++c) {
        plan_impl<T> trial(size, is_forward, cands[c], nthreads);
        const double ns = time_plan(trial);
        if (ns < best_ns) { best_ns = ns; pick.route = cands[c]; }
    }

    if (pick.route == route_kind::iterative_dif && race_chains) {
        raced = 0;
        double chain_best = kMeasureInf;
        const auto race = [&](const dif_factor_plan& chain) {
            if (!detail::dif_chain_shape_ok<T>(size, chain)) return;
            plan_impl<T> trial(size, is_forward, nthreads, nullptr,
                               measured_choice{route_kind::iterative_dif, chain});
            const double ns = time_plan(trial);
            if (ns < chain_best) {
                chain_best = ns;
                pick.dif_chain = chain;
            }
        };
        for (std::size_t i = 0; i < chain_cands.count && have_budget(); ++i)
            race(chain_cands[i]);
        raced = 0;
        dif_factor_plan perm = pick.dif_chain;
        for (std::size_t s = 1; s < perm.count && have_budget(); ++s) {
            std::rotate(perm.radices.begin(), perm.radices.begin() + 1,
                        perm.radices.begin() + perm.count);
            race(perm);
        }
        std::sort(perm.radices.begin(), perm.radices.begin() + perm.count);
        while (have_budget() &&
               std::next_permutation(perm.radices.begin(), perm.radices.begin() + perm.count))
            race(perm);
    }
    return pick;
}

template<typename T>
void plan_impl<T>::execute(span<std::complex<T>> data, const exec_options<T>& opts) const {
    if (data.size() != m.size) ADM_UNLIKELY {
        throw size_error("Data size doesn't match plan size");
    }
    execute(data.data(), data.data(), opts);
}

template<typename T>
void plan_impl<T>::execute(const std::complex<T>* src, std::complex<T>* dst,
                           const exec_options<T>& opts) const {
    const T fct = opts.fct.value_or(m.is_forward ? T(1) : T(1) / T(m.size));
    if (opts.debug >= dbg_route) ADM_UNLIKELY
        trace(opts.debug, src == dst ? "in-place" : "oop", fct);
    if (m.size == 1) ADM_UNLIKELY { *dst = *src * fct; return; }
    if (m.is_forward) execute_impl<true>(src, dst, fct);
    else              execute_impl<false>(src, dst, fct);
}

template<typename T>
void plan_impl<T>::execute_many(std::complex<T>* data, std::size_t n, std::size_t stride,
                                const exec_options<T>& opts) const {
    const T fct = opts.fct.value_or(m.is_forward ? T(1) : T(1) / T(m.size));
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "many", fct, n, stride);
    if (m.size == 1) ADM_UNLIKELY {
        for (std::size_t r = 0; r < n; ++r) data[r * stride] *= fct;
        return;
    }
    if (is_codelet_catalog(m.size)) {
        if (m.is_forward) codelet_dispatch_many<T, true >(data, n, stride, m.size, fct);
        else              codelet_dispatch_many<T, false>(data, n, stride, m.size, fct);
        return;
    }
    if (m.is_forward) execute_many_impl<true>(data, n, stride, fct);
    else              execute_many_impl<false>(data, n, stride, fct);
}

extern template class plan_impl<float>;
extern template class plan_impl<double>;

}
}

#include "undef_macros.hpp"
