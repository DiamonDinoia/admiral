#pragma once

// ============================================================================
// `plan_impl<T>` is the 1-D engine; every public entry point ends here.
// Construction picks one route for (T, n, direction, nthreads) from the fitted cost
// model in `base_cost_model.hpp`, not from a formula here, and builds only that
// route's tables. `execute()` dispatches with no branching on n. Route predicate
// names end with `_supported`. The build instantiates `plan_impl` once per
// precision, in `src/inst_plan_f.cpp` and `src/inst_plan_d.cpp`; every other TU
// sees only the extern declaration (compile-time measure; dispatch is always runtime).
// ============================================================================

#include <admiral/admiral.hpp>  // `effort`

#include <admiral/detail/config.hpp>  // `adm_measure`

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <variant>
#include <vector>
#include "cxx_compat.hpp"  // `ADM_UNLIKELY`, `ADM_UNREACHABLE`, `span`, `detail::has_single_bit`

#include "base_cost_model.hpp" // generated routing cost model, see `tools/fit_cost_model.cpp`
#include "bluestein.hpp"  // `bluestein_plan`
#include "debug.hpp"       // `dbg_print`, `dbg_off`/`dbg_route`/`dbg_shape` (`exec_options::debug`)
#include "dif_driver.hpp"   // `iterative_dif_execute_ws`, `dif_execute_in_place`
#include "four_step.hpp"    // `four_step_supported`, `choose_four_step_split`, `four_step_execute`
#include "four_step_large.hpp" // `four_step_large_plan`, `four_step_large_supported` (DRAM-bound)
#include "math.hpp"         // route predicates, `codelet_dispatch`
#include "good_thomas.hpp"  // `good_thomas_catalog` (routed Good-Thomas descriptor pack)
#include "rader.hpp"        // `rader_plan`, `rader_supported` (primes above the codelet catalog)
#include "scratch.hpp"      // `soa_scratch`
#include "thread_pool.hpp" // `thread_pool` (plan-owned intra-transform threading)
#include "twiddles.hpp"     // `dif_twiddle_set`, `build_dif_twiddle_set`, `dif_factor_plan`

// Include `macros.hpp` last, paired with `undef_macros.hpp` at the end of the file.
// A sibling header above must not re-include `macros.hpp` while this copy is defined.
#include "macros.hpp"       // `ADM_NOINLINE`, `ADM_COLD` (`trace()`)

namespace admiral {
namespace detail {

// Per-execute options. C++20 takes designated initializers; C++17 reads the
// aggregate positionally as `fct` then `debug`.
// `fct` == `std::nullopt`: forward 1, inverse 1/N. `debug` == 0: silent (see `debug.hpp`).
// The `nthreads` ctor argument fixes threading at plan creation; no per-call knob.
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
        four_step_large,   // DRAM-bound large N (`four_step_large.hpp`)
        rader,
        bluestein,
        good_thomas     // Good-Thomas PFA (eligibility: `good_thomas_catalog` in `good_thomas.hpp`)
    };

    // Result of an `effort::measure` election: the winning route plus, when the
    // chain-order race ran, the winning dif chain (`count` == 0: default order).
    struct measured_choice {
        route_kind route;
        dif_factor_plan dif_chain;
    };

    // Dispatches per execute on the four_step_large route: P1 band-transpose + the
    // P2/P3 twist-sweep pair + the P4/P5 pair (fi/mt t0-profiler-r1.md sect. b).
    static constexpr std::size_t kLargeDispatches = 5;

    static constexpr bool good_thomas_available(std::size_t n) noexcept {
        return good_thomas_catalog::available<T>(n);
    }

    // True iff `rk` is instantiable for (T, size): the predicate the force-route ctor
    // tests. Public because availability is ISA- and precision-dependent, so a caller
    // asks rather than catching the ctor's exception.
    static constexpr bool route_available(route_kind rk, std::size_t size) {
        switch (rk) {
        case route_kind::codelet:           return is_codelet_catalog(size);
        case route_kind::good_thomas:       return good_thomas_available(size);
        // 11-smooth, or anything the radix DP factors with static kernels plus
        // generic runtime-prime middle passes; see `dif_chain_supported`.
        case route_kind::iterative_dif:     return dif_chain_supported(size);
        case route_kind::four_step:         return four_step_supported(size);
        case route_kind::four_step_batched: return four_step_batched_supported<T>(size);
        // Instantiable whenever a balanced split exists; the size threshold in
        // `select_route` is policy, not availability (a forced route ignores it).
        case route_kind::four_step_large:   return choose_large_split(size).valid();
        case route_kind::rader:             return rader_supported(size);
        case route_kind::bluestein:         return true;
        }
        return false;
    }

private:
    // .rodata-table routes (`codelet`, `good_thomas`) carry an empty tag;
    // plan-object routes (`four_step_batched`, `rader`, `bluestein`) are the state.
    struct codelet_state {};
    struct good_thomas_state {};
    struct dif_state { dif_twiddle_set<T> tw; };
    struct four_step_state {
        four_step_split split;               // N = n1*n2, both <=64 catalog leaves
        std::vector<std::complex<T>> tw;     // plan-owned W_N^{n1 k2} twist table
    };
    using route_state = std::variant<codelet_state, good_thomas_state, dif_state,
                                     four_step_state, four_step_batched_plan<T>,
                                     four_step_large_plan<T>,
                                     rader_plan<T>, bluestein_plan<T>>;

    // All state in one aggregate `m`; the `size()`/`is_forward()` getters avoid the _ suffix.
    struct M {
        std::size_t size;
        bool is_forward;
        route_kind route;
        route_state st;
        // Plan-owned workers. `std::unique_ptr`, not `std::optional`: the workers capture
        // the pool address, so the pool must survive a plan move (`thread_pool` is not
        // movable). `get()` on a const plan yields a mutable `pool*`, which keeps
        // `execute()` const.
        std::unique_ptr<thread_pool> pool;
    } m;

    // Build the pool only for `four_step_large`: that route's executor is the only
    // reader of `m.pool`. On any other route a threaded plan would spawn parked
    // workers that nothing wakes.
    [[nodiscard]] static std::unique_ptr<thread_pool> make_route_pool(std::size_t nthreads,
                                                                      std::size_t size,
                                                                      route_kind rk) {
        return (nthreads > 1 && size > 1 && rk == route_kind::four_step_large)
                   ? std::make_unique<thread_pool>(nthreads)
                   : nullptr;
    }

public:
    // Out-of-line (extern-template in src/inst_plan_*.cpp): inlining pulls the
    // whole route tree into every consumer TU. Runtime dispatch is unchanged.

    // Test-only force-route ctor; unavailable routes throw unsupported_error. Always
    // serial: threaded routes are exercised through the nthreads ctor below.
    plan_impl(std::size_t size, bool is_forward, route_kind forced, std::size_t nthreads = 1);

    // `nthreads`: thread count of this plan; 0 = auto (the wake law, resolved with the
    // elected route). Only `four_step_large` executes its own passes in parallel, so
    // only that route choice reads `nthreads` (see `large_route_bytes`); `nthreads` > 1
    // also builds the plan-owned pool here. `eff`: `effort::measure` times the ranked
    // candidates on this machine.
    plan_impl(std::size_t size, bool is_forward, std::size_t nthreads = 1,
              const dif_factor_plan* dif_override = nullptr,
              admiral::effort eff = admiral::effort::estimate);

private:
    // Route plus the thread count it was elected at, bundled for the delegating ctor.
    struct routed_plan {
        measured_choice ch;
        std::size_t nthreads;
        const dif_factor_plan* dif_override;
    };

    // Route-resolved delegate: the pool gate reads the route, and the initializer
    // list of `m` cannot reference the route (clang -Werror,-Wuninitialized). `choice`
    // is by value: a won chain re-ordering must outlive `build_dif_twiddle_set` in the
    // body.
    plan_impl(std::size_t size, bool is_forward, routed_plan rp);
    plan_impl(std::size_t size, bool is_forward, std::size_t nthreads,
              const dif_factor_plan* dif_override, measured_choice choice)
        : plan_impl(size, is_forward, routed_plan{choice, nthreads, dif_override}) {}

    // Serial-work estimate (ns) for the four_step_large split: the kFourStepOverhead
    // form over the two leaf-priced legs, at the probed clock.
    static double large_work_ns(std::size_t size) {
        const large_split sp = choose_large_split(size);
        const double cyc =
            sp.valid() ? kFourStepOverhead * (double(sp.n1) * line_work_cyc<T>(sp.n2) +
                                              double(sp.n2) * line_work_cyc<T>(sp.n1))
                       : line_work_cyc<T>(size);
        return cyc / core_cyc_per_ns();
    }

    // Elect the route and the thread count together. An explicit count elects at that
    // count (unchanged path). 0 = auto: elect at the machine count, then price
    // threading by route. four_step_large pays the wake law's K dispatches; every
    // other route builds no pool and resolves serial.
    static routed_plan route_plan(std::size_t size, bool is_forward, std::size_t nthreads,
                                  const dif_factor_plan* dif_override, admiral::effort eff) {
        // Routing is undefined at 0, and this runs before the delegate's size guard:
        // `is_pentanomial(0)` divides 0 by 2 forever. Answer trivially and let the
        // delegate throw.
        if (size == 0) ADM_UNLIKELY return {measured_choice{}, 1, dif_override};
        if (dif_override) return {measured_choice{route_kind::iterative_dif, {}}, nthreads, dif_override};
        const auto elect = [&](std::size_t nt) {
            return eff != admiral::effort::estimate
                       ? measured_route(size, is_forward, nt)
                       : measured_choice{select_route(size, nt), {}};
        };
        if (nthreads != 0) return {elect(nthreads), nthreads, dif_override};
        const std::size_t P = resolve_nthreads(0);
        routed_plan rp{elect(P), P, dif_override};
        if (rp.ch.route != route_kind::four_step_large) {
            rp.nthreads = 1;
            return rp;
        }
        rp.nthreads = resolve_nthreads(0, size, kLargeDispatches, large_work_ns(size), 0);
        if (rp.nthreads == P) return rp;
        // The admission lines move with the count: re-elect once at the resolved count
        // and accept the outcome (a flip off four_step_large resolves serial).
        measured_choice rerun = elect(rp.nthreads);
        if (rerun.route != route_kind::four_step_large) {
            rp.ch = rerun;
            rp.nthreads = 1;
            return rp;
        }
        rp.ch = rerun;
        return rp;
    }

    // Emplace route state into `m.st` (`m.route` already set). Shared by both ctors.
    // `codelet`/`good_thomas` live in .rodata; `codelet_state` is the variant default.
    void emplace_route_state(std::size_t size, bool is_forward, const dif_factor_plan* dif_override) {
        switch (m.route) {
        case route_kind::codelet:
            break;  // codelet_state is the variant's default alternative
        case route_kind::good_thomas:
            m.st.template emplace<good_thomas_state>();
            break;
        // dif_state and four_step_state are aggregates: braced-construct-then-move
        // works around parenthesized aggregate init (P0960), which AppleClang 15
        // lacks; a move of two vectors at plan build costs nothing.
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

    // Execute in place; a threaded plan runs `four_step_large` on its plan-owned pool.
    // `opts.fct` scales the output (`std::nullopt` = direction default). Out-of-line.
    void execute(span<std::complex<T>> data, const exec_options<T>& opts = {}) const;

    // src == dst: in-place. src != dst: `src` preserved, no partial alias, and each
    // route folds the `src` read into its first pass (no staging copy). Out-of-line.
    void execute(const std::complex<T>* src, std::complex<T>* dst,
                 const exec_options<T>& opts = {}) const;

    // n in-place lines at uniform stride, route resolved once for the whole run. The
    // per-line path would pay an out-of-line call, a direction branch and a
    // `std::visit` before any arithmetic, several times the transform cost at small
    // size. Out-of-line.
    void execute_many(std::complex<T>* data, std::size_t n, std::size_t stride,
                      const exec_options<T>& opts = {}) const;

private:
    // exec_options::debug >= dbg_route. Out of line and cold: the call-site guard is
    // then the whole cost when tracing is off, and every argument is already live at
    // the guard.
    ADM_NOINLINE ADM_COLD void trace(unsigned level, const char* how, T fct,
                                     std::size_t lines = 1, std::size_t stride = 0) const {
        dbg_print("N=", m.size, m.is_forward ? " fwd " : " inv ", route_name(), " ", how,
                  " fct=", static_cast<double>(fct), m.pool ? " threaded" : " serial");
        if (lines != 1) dbg_print("  lines=", lines, " stride=", stride);
        if (level < dbg_shape) return;
        // The shape is whatever the elected route holds; a route whose name is
        // its whole shape prints nothing rather than a placeholder.
        std::visit([](const auto& st) {
            using S = std::decay_t<decltype(st)>;
            if constexpr (std::is_same_v<S, dif_state>) dbg_print_seq("  radices", st.tw.radices);
            else if constexpr (std::is_same_v<S, four_step_state>)
                dbg_print("  n1=", st.split.n1, " n2=", st.split.n2);
        }, m.st);
        if (level < dbg_cost) return;
        trace_ranking();
    }

    // The ranking select_route walked: each form's modelled cycles and buildability.
    // The elected route is the first line that is both cheap and available.
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

        // Run loop inside the `std::visit`, so the route is a compile-time type on
        // every line.
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
        // Only the DRAM-bound large four-step threads internally; the rest are
        // cache-resident, where per-pass barriers cost more than they save.
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

    // Route name for diagnostics/--decomp-report; not on any hot path.
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
    // Four-step split (n1,n2); {0,0} if the route is not `four_step`.
    [[nodiscard]] four_step_split four_step_split_used() const noexcept {
        const auto* fs = std::get_if<four_step_state>(&m.st);
        return fs ? fs->split : four_step_split{};
    }

private:
    // Inverse of the fitter's `FORM_ORDER` (`tools/fit_cost_model.cpp`).
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
        // Exhaustive: a new base_form must be mapped here, not silently routed.
        ADM_UNREACHABLE();
    }

    // `effort::measure`: time the model's top-ranked candidates on scratch, return the
    // measured winner. Stage 1 (base forms) stays in the fitted domain, the only place
    // a runtime A/B carries new information over the model. Stage 2 races the DP's
    // next-cheapest chains at any N. Plan-time only; threaded engines are excluded.
    static measured_choice measure_route(std::size_t size, bool is_forward, std::size_t nthreads);

    // Resolve a measuring effort: the timed winner when compiled in, the model pick
    // otherwise. Under -DADM_MEASURE=OFF the knob is accepted and inert.
    static measured_choice measured_route(std::size_t size, [[maybe_unused]] bool is_forward,
                                          std::size_t nthreads) {
        if constexpr (adm_measure) return measure_route(size, is_forward, nthreads);
        else return measured_choice{select_route(size, nthreads), {}};
    }

    // `four_step_large` admission size. The lines live in `four_step_large.hpp`, beside
    // the route the lines admit and the Bluestein arm that shares the f64 one.
    static constexpr std::size_t large_route_bytes(std::size_t nthreads) {
        // Threaded line: budget/nthreads clamped by a floor (see
        // large_route_threaded_bytes), so one crossover law covers every thread count.
        if (nthreads > 1) return large_route_threaded_bytes(nthreads);
        if constexpr (sizeof(T) == 8) return kLargeRouteSerialF64Bytes;
        else return kLargeRouteSerialF32Bytes;
    }

    // Bailey-split admission: byte line and split shape. n2 % n1 != 0 sends both
    // in-place transposes to the serial element-cycle fallback; the cliff survives
    // threading, so this clause binds every nthreads. Serial further requires
    // n1 % W == 0 (band fusion); under threading row DFTs and transposes take the pool.
    static constexpr bool large_route_admits(std::size_t size, std::size_t nthreads) {
        if (!four_step_large_supported(size, sizeof(std::complex<T>),
                                       large_route_bytes(nthreads)))
            return false;
        if (nthreads > 1) {
            const large_split sp = choose_large_split(size);
            return sp.n2 % sp.n1 == 0;
        }
        // Serial f32 also has an upper crossover (kLargeRouteSerialF32MaxBytes).
        if constexpr (sizeof(T) == 4)
            if (size * sizeof(std::complex<T>) > kLargeRouteSerialF32MaxBytes) return false;
        return four_step_large_fused_shape<T>(size);
    }


    // `iterative_dif` availability: 11-smooth, or any N the radix DP factors with
    // static kernels plus generic runtime-prime middle passes (`dif_generic_radices`).
    // The DP run is cheap against the plan build: twiddle generation dominates it.
    [[nodiscard]] static bool dif_chain_supported(std::size_t size) {
        if (detail::has_single_bit(size) || is_codelet_supported(size)) return true;
        // The executed chain is not always the DP's argmin: the shape vetoes condemn
        // some cheapest chains, and the election walks the candidate list past them.
        return dif_chain_shape_ok(size, dif_elected_chain<T>(size));
    }

public:
    // Public for the same reason route_available is: an explicitly supplied chain
    // (--factors, the chain sweep) needs it without the election. The rule lives by the DP.
    [[nodiscard]] static bool dif_chain_shape_ok(std::size_t size, const dif_factor_plan& p) {
        return detail::dif_chain_shape_ok<T>(size, p);
    }

private:
    // Route ladder, first match wins; availability is instantiability, not correctness.
    static route_kind select_route(std::size_t size, std::size_t nthreads) {
        // Modelled domain (2..512): take the model ranking's cheapest buildable route.
        // The gates below matter only above BASE_MODEL_NMAX, where no model exists.
        {
            const auto& ranking = base_route_ranking<T>(size);
            for (std::size_t i = 0; i < ranking.count; ++i) {
                const route_kind rk = form_to_route(ranking.form[i]);
                if (route_available(rk, size)) return rk;
            }
        }
        if (is_codelet_catalog(size)) return route_kind::codelet;
        // `four_step_batched`: f32 W==8 small-band only (`fsb_split_for`); not at W=16.
        // (`vecpass` survives in `vecpass.hpp` as the r2c tile engine; not routed here.)
        if (four_step_batched_supported<T>(size)) return route_kind::four_step_batched;
        // Large-N (Bailey): `iterative_dif` is DRAM-bound past L2 (~log_radix(N) full
        // reads). Factoring N=n1*n2 with cache-resident leaves streams only a handful
        // of times. Admission size is `large_route_bytes()`; runtime column tiling
        // uses the actual L2 size.
        if (large_route_admits(size, nthreads)) return route_kind::four_step_large;
        // No dif-vs-blue modelled gate: both sides' absolute scales misprice the real
        // routes. Structure decides: bookended-chain availability plus the DP's veto tier.
        if (dif_chain_supported(size))
            return route_kind::iterative_dif;
        // Non-11-smooth N=n1*n2 with both <=64 catalog leaves (143=11x13, 289=17x17,
        // 338=13x26): codelet composition instead of Bluestein, only when the model
        // says leaves win (169/209/221/247 keep Bluestein). Fires only past dif above.
        if (four_step_supported(size) && four_step_beats_bluestein<T>(size))
            return route_kind::four_step;
        // Primes p>64 (67, 71, 73 and onward): Rader turns the DFT into a
        // length-(p-1) cyclic
        // convolution via codelet/iterative_dif/four-step, cost-gated on beating
        // Bluestein; primes with expensive p-1 (79/83/127/191/251/283) keep Bluestein.
        if (rader_supported(size) && rader_beats_bluestein<T>(size))
            return route_kind::rader;
        return route_kind::bluestein;
    }

    // Per-route execute overloads (`std::visit` targets). Unnormalized; the 1/N folds
    // into the `dif` terminal pass or into a separate sweep on other routes.

    // Good-Thomas PFA via good_thomas_catalog. Forward compile-time;
    // inverse imaginary-negate folds away in the forward instantiation.
    template<bool Forward>
    void execute_route(const good_thomas_state&, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        good_thomas_run<T, Forward>(in, out, m.size);
        apply_scale(out, fct);
    }

    // Codelet transform via codelet_dispatch. Stack buffer; catalog sizes <=64.
    template<bool Forward>
    void execute_route(const codelet_state&, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        codelet_dispatch<T, Forward>(in, out, m.size);
        apply_scale(out, fct);
    }

    // Iterative DIF; first pass consumes in into SoA scratch (OOP copy-free).
    // fct != 1 folds the scale into dif_pass_last's store; fct == 1 runs unscaled.
    template<bool Forward>
    void execute_route(const dif_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        dif_execute_in_place<T>(Forward, in, out, m.size, st.tw, fct);
    }

    // Four-step (N = N1*N2, both <=64 catalog leaves). G is N-length scratch.
    template<bool Forward>
    void execute_route(const four_step_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        // Uninitialized: four_step_execute writes every G[n1*N2+k2] before the outer
        // pass reads any, so vector value-init would be an immediately-overwritten memset.
        soa_scratch<T, 1> scratch(2 * m.size);
        four_step_execute<T, Forward>(in, out, st.split.n1, st.split.n2, st.tw.data(),
                                      reinterpret_cast<std::complex<T>*>(scratch.buf(0)));
        apply_scale(out, fct);
    }

    // Batched four-step (f32 N1*N2): deinterleave -> batched -> reinterleave.
    // Direction baked at plan time (`fsb.is_forward`); the `Forward` template argument
    // goes unused here.
    template<bool Forward>
    void execute_route(const four_step_batched_plan<T>& fsb, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        fsb.execute(in, out);
        apply_scale(out, fct);
    }

    // Large-N four-step (DRAM-bound). `fct` folds into the final row pass; direction
    // is baked at plan time (the `Forward` template argument goes unused). The route
    // threads its passes on the plan-owned pool. The ND row driver shares sub-plans
    // built with `nthreads` = 1 across threads, so their `m.pool` is null and the
    // route runs inline.
    template<bool Forward>
    void execute_route(const four_step_large_plan<T>& fsl, const std::complex<T>* in,
                       std::complex<T>* out, T fct, thread_pool* pool = nullptr) const {
        fsl.execute(in, out, fct, pool);
    }

    // Rader prime route (isolated p>64 as cyclic convolution). Direction baked at plan time.
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

    // Scale the output by `fct`; no-op for `fct` == 1 (branch outside the hot loop).
    // `dif` and `four_step_large` fold the scale into their terminal store and bypass
    // this sweep.
    void apply_scale(std::complex<T>* out, T fct) const {
        if (fct != T(1)) scale_inplace(out, m.size, fct);
    }
};

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, route_kind forced,
                        std::size_t nthreads)
    // The force-route trial takes the same pool rule: without the pool it would time
    // the serial route under a threaded route's name.
    : m{size, is_forward, forced, {},
        make_route_pool(nthreads, size, forced)}
{
    if (size == 0) ADM_UNLIKELY
        throw size_error("Plan size must be greater than 0");
    // Force any instantiable route; reject unavailable (size, T) combinations.
    if (!route_available(forced, size))
        throw unsupported_error("force-route: kernel unavailable for this size/precision");
    emplace_route_state(size, is_forward, nullptr);
}

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, std::size_t nthreads,
                        const dif_factor_plan* dif_override, admiral::effort eff)
    // dif_override forces iterative_dif: select_route could otherwise ignore the
    // override and time a false A/B baseline. Only --factors/--factors-ab supplies it.
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
    // Every radix in dif_radix_set is 11-smooth, so no chain can represent a
    // non-smooth N. Forcing one would time a garbage A/B baseline. Reject loudly.
    if (rp.dif_override && !route_available(route_kind::iterative_dif, size))
        throw unsupported_error("forced dif: size is not 11-smooth");
    const dif_factor_plan* chain =
        rp.dif_override != nullptr        ? rp.dif_override
        : rp.ch.dif_chain.count != 0 ? &rp.ch.dif_chain
                                     : nullptr;
    emplace_route_state(size, is_forward, chain);
}

// Race counts are fixed per stage and not clock-denominated: a race is reproducible
// and machine-independent. No patience rule: candidate order ascends in model cost,
// not measured value, so early stopping would misread it as a measured ranking.
inline constexpr std::size_t kMeasureCandidates = 8;
// Samples per candidate: one sample cannot separate a slow timer round from a real loss.
inline constexpr std::size_t kMeasureReps = 5;
inline constexpr std::size_t kMeasureMaxCandidates = 4;
// Clock-resolution floor: a per-call time below this is the timer, not the plan.
inline constexpr std::chrono::nanoseconds::rep kMeasureMinNs = 50;
// Span each sample must cover: one small-N execution is shorter than the clock reads,
// so a sample of one elects a coin flip; batch executions clear the floor.
inline constexpr std::chrono::nanoseconds::rep kMeasureSampleNs = 4000;
    // Executions per sample: the fewest that span `kMeasureSampleNs` at one ns each.
[[nodiscard]] constexpr std::size_t measure_batch(std::chrono::nanoseconds::rep one_ns) {
    if (one_ns >= kMeasureSampleNs) return 1;
    return std::size_t(kMeasureSampleNs / (std::max)(one_ns, std::chrono::nanoseconds::rep{1})) + 1;
}
// Stop sampling a candidate once its best is this far behind the fastest execution seen.
inline constexpr double kMeasureRejectRatio = 1.25;
// Finite sentinel: -ffinite-math-only makes infinity UB.
inline constexpr double kMeasureInf = 1e300;

template<typename T>
typename plan_impl<T>::measured_choice
plan_impl<T>::measure_route(std::size_t size, bool is_forward, std::size_t nthreads) {
    const route_kind fallback = select_route(size, nthreads);
    using clock = std::chrono::steady_clock;

    // Nothing before the scratch allocation may do O(N) work: at big N the buffers
    // are O(N), and a `bluestein` or `four_step_large` gated call must not pay the
    // buffers.
    measured_choice pick{fallback, {}};

    // Stage 1 candidates (fitted domain): the model fallback first (a degenerate race
    // returns it unchanged), then distinct model candidates, plus the f32 batched route.
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
        // one slot held back for the f32 batched route offered below
        for (std::size_t i = 0; i < ranking.count && nc + 1 < kMeasureMaxCandidates; ++i)
            offer(form_to_route(ranking.form[i]));
        if constexpr (sizeof(T) == 4)
            offer(route_kind::four_step_batched);  // f32 band outside the fitted forms
    }

    // Above the fitted domain `large_route_bytes()` alone decides `four_step_large`,
    // so the race is the gate's only correction. The race must run both directions of
    // the gate: a rejection prices `four_step_large` against the fallback, an admission
    // prices the DIF chain against `four_step_large`. The split must be pool-safe
    // (n2 % n1) for the route to build.
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

    // Non-degenerate sample data: an all-zero or all-one input rides denormal and
    // constant-folding shortcuts no real signal ever takes.
    std::vector<std::complex<T>> in(size), out(size);
    for (std::size_t i = 0; i < size; ++i)
        in[i] = {T(0.5 * int(i % 7) - 1), T(0.25 * int(i % 5) - 0.5)};

    // unit is the fastest execution of any plan for this transform seen so far; it
    // bounds candidates by reject ratio, not the race budget.
    double unit = kMeasureInf;
    std::size_t raced = 0;
    const auto have_budget = [&] { return raced < kMeasureCandidates; };

    double best_ns = kMeasureInf;
    const auto time_plan = [&](plan_impl<T>& trial) {
        trial.execute(in.data(), out.data());  // warm + first-touch
        // Executions per sample, sized from one warm execution so the sampled interval
        // clears the clock's resolution.
        const std::size_t inner = [&] {
            const auto a = clock::now();
            trial.execute(in.data(), out.data());
            return measure_batch(std::chrono::nanoseconds(clock::now() - a).count());
        }();
        // unit before this candidate keeps the reject live in every race: the chain
        // race's incumbent starts at +inf, so an incumbent-relative reject never fires.
        const double u = unit;
        double best = kMeasureInf;
        for (std::size_t r = 0; r < kMeasureReps; ++r) {
            const auto a = clock::now();
            for (std::size_t k = 0; k < inner; ++k) trial.execute(in.data(), out.data());
            const auto span = std::chrono::nanoseconds(clock::now() - a).count();
            best = (std::min)(best, double((std::max)(span, kMeasureMinNs)) / double(inner));
            unit = (std::min)(unit, best);
            // A candidate a quarter behind the fastest seen cannot win; stop paying
            // reps. The reject needs two samples: one cannot separate a slow chain from
            // the cold twiddle table the candidate's own build just wrote.
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

    // Chain race: which multiset, then which ordering of the winner. Both are questions the model
    // cannot answer (orderings tie on the additive cost the DP minimizes).
    if (pick.route == route_kind::iterative_dif && race_chains) {
        // Each race gets the candidate count for itself: route and factorization are
        // different questions; a shared window would let routes consume it first.
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
        // Then the ordering of the winning multiset. The ordering walk decides the
        // result: rotations reach the cheap permutations, and `std::next_permutation`
        // covers the rest.
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
    // Direction default: forward=1, inverse=1/N.
    const T fct = opts.fct.value_or(m.is_forward ? T(1) : T(1) / T(m.size));
    if (opts.debug >= dbg_route) ADM_UNLIKELY
        trace(opts.debug, src == dst ? "in-place" : "oop", fct);
    if (m.size == 1) ADM_UNLIKELY { *dst = *src * fct; return; }  // ctor rejects size==0
    // Lift direction to compile-time once; hot path carries no runtime branch.
    if (m.is_forward) execute_impl<true>(src, dst, fct);
    else              execute_impl<false>(src, dst, fct);
}

template<typename T>
void plan_impl<T>::execute_many(std::complex<T>* data, std::size_t n, std::size_t stride,
                                const exec_options<T>& opts) const {
    const T fct = opts.fct.value_or(m.is_forward ? T(1) : T(1) / T(m.size));
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "many", fct, n, stride);
    if (m.size == 1) ADM_UNLIKELY {  // ctor rejects size==0
        for (std::size_t r = 0; r < n; ++r) data[r * stride] *= fct;
        return;
    }
    // Lanes-as-lines fills the vector units a per-line size-N codelet cannot fill: the
    // full-width loop is empty when a line has < W columns. Lanes-as-lines wins at
    // every catalog size. Consequence: `execute()` and `execute_many()` of one plan
    // can take different routes and agree only to roundoff. Nothing downstream may
    // assume bit identity between them.
    if (is_codelet_catalog(m.size)) {
        if (m.is_forward) codelet_dispatch_many<T, true >(data, n, stride, m.size, fct);
        else              codelet_dispatch_many<T, false>(data, n, stride, m.size, fct);
        return;
    }
    if (m.is_forward) execute_many_impl<true>(data, n, stride, fct);
    else              execute_many_impl<false>(data, n, stride, fct);
}

// Consumers reference these; src/inst_plan_*.cpp provides the explicit definition.
extern template class plan_impl<float>;
extern template class plan_impl<double>;

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
