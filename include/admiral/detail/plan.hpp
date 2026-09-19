#pragma once

#include <admiral/admiral.hpp>

#include <admiral/detail/config.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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
        ADM_UNREACHABLE();
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
#ifdef ADM_PIN_PLAN_ESTIMATE
        eff = admiral::effort::estimate;  // measurement knob: measure/automatic cannot race
#endif
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
            // Auto resolved to serial, so elect at that width. Returning the P-wide estimate
            // would downgrade effort::measure to effort::estimate on every non-large route.
            return {eff == admiral::effort::estimate ? route_est : elect(1), 1, dif_override};
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
        ADM_UNREACHABLE();
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

    template<typename TimePlan>
    static void race_dif_chains(std::size_t size, bool is_forward, std::size_t nthreads,
                                const dif_chain_list& chain_cands, measured_choice& pick,
                                TimePlan& time_plan, std::size_t& raced);

    static measured_choice measured_route(std::size_t size, [[maybe_unused]] bool is_forward,
                                          std::size_t nthreads) {
        if constexpr (adm_measure) return measure_route(size, is_forward, nthreads);
        else return measured_choice{select_route(size, nthreads), {}};
    }

    // The serial line is measured, not constant: large_route_serial_bytes resolves the
    // injected override (tests), then the cached probe answer, running the probe once per
    // process on first consultation. The probe makes this nt-variant and admits-variant
    // non-constexpr; route_available never consults the line, so no constant-evaluated
    // call site moves.
    static std::size_t large_route_serial_bytes() {
        constexpr std::size_t elem = sizeof(std::complex<T>);
        if (const std::size_t ov = detail::large_route_serial_override(elem)) return ov;
        static const std::size_t line =
            detail::large_route_probe_disabled()
                ? detail::large_route_serial_fallback(elem)
                : probe_large_route_serial();
        return line;
    }

    static std::size_t large_route_bytes(std::size_t nthreads) {
        if (nthreads > 1) {
            // The threaded law reads the pool width once, cached: a topology probe, not a
            // host key (thread_pool.hpp's own resolve statics work the same way).
            static const std::size_t pool = resolve_nthreads(0);
            return large_route_threaded_bytes(sizeof(std::complex<T>), nthreads, pool);
        }
        return large_route_serial_bytes();
    }

    static bool large_route_admits(std::size_t size, std::size_t nthreads) {
        constexpr std::size_t elem = sizeof(std::complex<T>);
        // Below the probe floor no line value admits, so never consult the probe there:
        // the digest (largest case 192 KiB, >= 3 octaves under the floor) and every small
        // plan stay probe-free by construction.
        if (nthreads <= 1 &&
            !four_step_large_supported(size, elem,
                                       detail::large_route_serial_min_bytes(elem)))
            return false;
        if (!four_step_large_supported(size, elem, large_route_bytes(nthreads)))
            return false;
        if (nthreads > 1) {
            const large_split sp = choose_large_split(size);
            return sp.n2 % sp.n1 == 0;
        }
        return four_step_large_fused_shape<T>(size);
    }

    // Runs the serial dif-vs-four_step_large ladder on this host; ~60-190 ms on the
    // reference classes, bounded by construction (3 rungs, <= 4 timed rounds per arm per
    // rung, one warmup each). Public as the probe's test entry point; the lazy path goes
    // through large_route_serial_bytes. Any build/allocation failure yields the fallback.
public:
    static std::size_t probe_large_route_serial();
private:

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
inline constexpr std::chrono::nanoseconds::rep kMeasureLongNs = 2'000'000;
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
    bool tail_race = false;  // set only by the size > BASE_MODEL_NMAX block below
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

    // Past the cost model's domain the kLargeRoute* lines are the only thing that elects.
    // The serial line below is itself measured on this host by the lazy probe
    // (large_route_serial_bytes), so the admission question the line cannot answer --
    // which side of the crossover TODAY's kernels sit on at this exact size -- is again
    // decision-shaped: the line only gates, and the race beyond it is the re-check. The
    // 32 MiB f32 cap that once closed a window here is deleted: the WI-2c sweep measured
    // four_step winning at every rung past it on every class. The shape predicate is the
    // one the elected plan would carry: fused serially, n1 | n2 threaded.
    if (size > BASE_MODEL_NMAX) {
        const large_split sp = choose_large_split(size);
        const bool shape_ok = nthreads > 1 ? sp.n2 % sp.n1 == 0
                                           : four_step_large_fused_shape<T>(size);
        // Serially the line GATES and the race RE-CHECKS (the selection loop's
        // kMeasureRejectRatio bar below — one noisy dif sample cannot flip the line).
        // Below the line the two routes are not in
        // contention, and racing there would only charge the DIF arm its per-call scratch faults:
        // at 2^16 f64 that reads it 3.7x slow and elects four_step_large where the chain wins by
        // 1.33x. The same probe gate as large_route_admits keeps racing (and the probe itself)
        // unreachable under the probe floor.
        const std::size_t bytes = size * sizeof(std::complex<T>);
        const bool in_band =
            nthreads > 1 ||
            (bytes > detail::large_route_serial_min_bytes(sizeof(std::complex<T>)) &&
             bytes > large_route_bytes(1));
        if (sp.valid() && shape_ok && in_band) {
            offer(fallback);
            offer(fallback == route_kind::four_step_large ? route_kind::iterative_dif
                                                         : route_kind::four_step_large);
            tail_race = true;
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
        const auto probe = [&] {
            const auto a = clock::now();
            trial.execute(in.data(), out.data());
            return std::chrono::nanoseconds(clock::now() - a).count();
        };
        // One execute past the cost model's domain costs milliseconds, so a warm-up plus five
        // reps is seconds of plan time per candidate at 2^24. Past kMeasureLongNs the probe IS
        // the sample: the routes there differ by more than their own spread. Below it the
        // executed sequence is unchanged, one warm-up and one probe before the rep loop.
        auto one_ns = probe();
        const bool cheap = one_ns < kMeasureLongNs;
        if (cheap) one_ns = probe();
        const std::size_t inner = measure_batch(one_ns);
        const double u = unit;
        double best = cheap ? kMeasureInf : double((std::max)(one_ns, kMeasureMinNs));
        unit = (std::min)(unit, best);
        for (std::size_t r = 0; cheap && r < kMeasureReps; ++r) {
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
        // The tail race re-checks the probed line; it does not re-open the election. The
        // challenger takes the line's answer only by kMeasureRejectRatio, the bar the rep
        // loop's early break already uses: a bare argmin flipped to the dif arm on one noisy
        // sample past the serial line, reading 7+ sweep cells at dif times where the forced
        // ladder had four_step 1.2-2.05x ahead (fi/wi3d large-1-D tail, 2026-09-15/16).
        const bool better = tail_race ? ns * kMeasureRejectRatio < best_ns : ns < best_ns;
        if (better) { best_ns = ns; pick.route = cands[c]; }
    }

    if (pick.route == route_kind::iterative_dif && race_chains)
        race_dif_chains(size, is_forward, nthreads, chain_cands, pick, time_plan, raced);
    return pick;
}

// The dif-chain race past the route race: candidates, rotations, permutations of
// pick.dif_chain, each timed through the same probe/budget protocol as the route race.
template<typename T>
template<typename TimePlan>
ADM_ALWAYS_INLINE void plan_impl<T>::race_dif_chains(std::size_t size, bool is_forward, std::size_t nthreads,
                                   const dif_chain_list& chain_cands, measured_choice& pick,
                                   TimePlan& time_plan, std::size_t& raced) {
    const auto have_budget = [&] { return raced < kMeasureCandidates; };
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
                    perm.radices.begin() + static_cast<std::ptrdiff_t>(perm.count));
        race(perm);
    }
    std::sort(perm.radices.begin(),
              perm.radices.begin() + static_cast<std::ptrdiff_t>(perm.count));
    while (have_budget() &&
           std::next_permutation(
               perm.radices.begin(),
               perm.radices.begin() + static_cast<std::ptrdiff_t>(perm.count)))
        race(perm);
}

// The serial-line probe. It walks the ladder from the rung nearest the prior (log2, ties
// lower): dif-side at the start walks UP, four_step-side walks DOWN, stopping at the
// first side flip or the ladder edge, so it visits two rungs on every measured class
// (three on an all-one-side host). Per rung: build both forced plans once, one untimed
// warmup per arm, then interleaved timed rounds with min per arm -- 3 rounds at or below
// 8 MiB (~12 ms per arm-pair), 1 above (single executes run 10-160 ms there) -- plus one
// re-sample round when the ratio lands inside 1 +- kLargeRouteProbeAmbigTol. Every arm
// failure (allocation, route construction) is answered with the fallback constants.
template<typename T>
std::size_t plan_impl<T>::probe_large_route_serial() {
    constexpr std::size_t elem = sizeof(std::complex<T>);
    // Not constexpr: MSVC's C2326 refuses a lambda capture of a constexpr pointer local.
    const std::size_t* const ladder =
        elem == 16 ? detail::kLargeRouteProbeLadderF64 : detail::kLargeRouteProbeLadderF32;
    constexpr std::size_t count = detail::kLargeRouteProbeLadderCount;
    const std::size_t top_n = ladder[count - 1] / elem;
    try {
        using clock = std::chrono::steady_clock;
        std::vector<std::complex<T>> buf(top_n);  // value-init pays the pages once, up front
        for (std::size_t i = 0; i < top_n; ++i)
            buf[i] = {T(0.5 * int(i % 7) - 1), T(0.25 * int(i % 5) - 0.5)};

        const auto sample = [&](std::size_t i) {
            const std::size_t n = ladder[i] / elem;
            plan_impl<T> dif(n, true, route_kind::iterative_dif, 1);
            plan_impl<T> fs(n, true, route_kind::four_step_large, 1);
            dif.execute(buf.data(), buf.data());  // one untimed warmup per arm
            fs.execute(buf.data(), buf.data());
            double bd = kMeasureInf, bf = kMeasureInf;
            const std::size_t base_rounds = ladder[i] <= (std::size_t{8} << 20) ? 3 : 1;
            double ratio = 1.0;
            for (std::size_t round = 0, total = base_rounds; round < total; ++round) {
                const auto a = clock::now();
                dif.execute(buf.data(), buf.data());
                const double d =
                    std::chrono::duration<double, std::nano>(clock::now() - a).count();
                const auto b = clock::now();
                fs.execute(buf.data(), buf.data());
                const double f =
                    std::chrono::duration<double, std::nano>(clock::now() - b).count();
                bd = (std::min)(bd, d);
                bf = (std::min)(bf, f);
                ratio = bf / bd;
                if (round + 1 == total && total == base_rounds &&
                    std::abs(ratio - 1.0) <= detail::kLargeRouteProbeAmbigTol)
                    ++total;  // ONE re-sample round, min-accumulated, then decide
            }
            return ratio;
        };
        // start: rung nearest the prior in log2, ties lower (strict < keeps the first)
        std::size_t start = 0;
        double best = 1e300;
        const double prior = double(detail::large_route_serial_fallback(elem));
        for (std::size_t i = 0; i < count; ++i) {
            const double d = std::abs(std::log2(double(ladder[i]) / prior));
            if (d < best) { best = d; start = i; }
        }
        bool dif_side[count] = {};
        const bool start_dif = detail::large_route_rung_dif_side(sample(start));
        dif_side[start] = start_dif;
        std::size_t lo = start, hi = start;
        if (start_dif) {
            for (std::size_t i = start + 1; i < count; ++i) {  // bracket [i-1, i] or top
                hi = i;
                dif_side[i] = detail::large_route_rung_dif_side(sample(i));
                if (!dif_side[i]) break;
            }
        } else {
            for (std::size_t i = start; i-- > 0;) {  // bracket [i, i+1] or bottom
                lo = i;
                dif_side[i] = detail::large_route_rung_dif_side(sample(i));
                if (dif_side[i]) break;
            }
        }
        return large_route_serial_from_ladder(ladder + lo, dif_side + lo, hi - lo + 1);
    } catch (...) {
        return detail::large_route_serial_fallback(elem);
    }
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
    const T fct = opts.fct.value_or(default_transform_fct<T>(m.is_forward, m.size));
    if (opts.debug >= dbg_route) ADM_UNLIKELY
        trace(opts.debug, src == dst ? "in-place" : "oop", fct);
    if (m.size == 1) ADM_UNLIKELY { *dst = *src * fct; return; }
    if (m.is_forward) execute_impl<true>(src, dst, fct);
    else              execute_impl<false>(src, dst, fct);
}

template<typename T>
void plan_impl<T>::execute_many(std::complex<T>* data, std::size_t n, std::size_t stride,
                                const exec_options<T>& opts) const {
    const T fct = opts.fct.value_or(default_transform_fct<T>(m.is_forward, m.size));
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
