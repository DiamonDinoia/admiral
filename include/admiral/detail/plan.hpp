#pragma once

// ============================================================================
// plan_impl<T>: the 1-D engine, and the one place that decides HOW to
// transform a given length. Every public entry point ends here.
// Construction picks a route for (T, n, direction, nthreads) and builds only
// that route's tables; execute() follows the choice with no branching on n.
//
// The routes:
//   codelet           n in the compiled catalog: one straight-line kernel<n>
//   iterative_dif     smooth n: mixed-radix decimation-in-frequency passes
//   four_step         n = n1*n2 that fits cache better split in two
//   four_step_batched the same split with both halves vectorised over lanes
//   four_step_large   n large enough that DRAM traffic, not arithmetic, decides
//   rader             prime n: a length-(n-1) cyclic convolution
//   bluestein         any n a chirp-z transform handles when nothing above fits
//   good_thomas       n = coprime factors: prime-factor algorithm, no twiddles
//
// The choice comes from a fitted cost model (base_cost_model.hpp), not from a
// formula here. Route predicates live next to the route: *_supported().
// plan_impl and nd_runtime_plan instantiate once per precision in
// src/inst_{plan,nd,real}_{f,d}.cpp; every other TU sees only the extern
// declaration (compile-time measure; dispatch is always runtime).
// ============================================================================

#include <admiral/admiral.hpp>  // effort

#include <admiral/detail/config.hpp>  // adm_measure

#include <algorithm>
#include <bit>
#include <chrono>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "base_cost_model.hpp" // generated routing cost model, see tools/fit_cost_model.cpp
#include "bluestein.hpp"  // bluestein_plan
#include "debug.hpp"        // dbg_print, dbg_off/dbg_route/dbg_shape (exec_options::debug)
#include "dif_driver.hpp"   // iterative_dif_execute_ws, dif_execute_in_place
#include "four_step.hpp"    // four_step_supported, choose_four_step_split, four_step_execute
#include "four_step_large.hpp" // four_step_large_plan, four_step_large_supported (DRAM-bound large N)
#include "math.hpp"         // route predicates, codelet_dispatch
#include "good_thomas.hpp"  // good_thomas_catalog (routed Good-Thomas descriptor pack)
#include "rader.hpp"        // rader_plan, rader_supported (primes above the codelet catalog)
#include "scratch.hpp"      // soa_scratch
#include "thread_pool.hpp" // thread_pool (plan-owned intra-transform threading)
#include "twiddles.hpp"     // dif_twiddle_set, build_dif_twiddle_set, dif_factor_plan

// Last of the includes, and paired with undef_macros.hpp at the end of the file: any
// sibling header above re-includes macros.hpp itself, which is an error while ours is
// still defined.
#include "macros.hpp"       // ADM_NOINLINE, ADM_COLD (trace())

namespace admiral {
namespace detail {

// Per-execute options. Designated-initializer friendly:
//   plan.execute(data, {.fct = T(0.5)}); // custom scale
//   plan.execute(data, {.debug = dbg_shape}); // trace route and shape to stderr
// fct == nullopt: forward=1, inverse=1/N.
// debug == 0: silent, and the only cost is the compare (see debug.hpp).
// Threading is NOT an option here: it is fixed at plan creation (the nthreads
// ctor argument), and the plan-owned pool is used internally at execute.
template<typename T>
struct exec_options {
    std::optional<T> fct = std::nullopt;
    unsigned debug = dbg_off;
};

// ============================================================================
// plan_impl: thin routing shell.
// select_route() picks one execution path at construction, precomputes its
// state into std::variant; execute() dispatches via std::visit.
// ============================================================================

template<typename T>
class plan_impl {
public:
    enum class route_kind {
        codelet,
        iterative_dif,
        four_step,
        four_step_batched,
        four_step_large,   // DRAM-bound large N (four_step_large.hpp)
        rader,
        bluestein,
        good_thomas     // Good-Thomas PFA (routed sizes + eligibility: good_thomas_catalog in good_thomas.hpp)
    };

    // Result of an effort::measure election: the winning route plus, when the
    // chain-order race ran, the winning dif chain (count == 0: default order).
    struct measured_choice {
        route_kind route;
        dif_factor_plan dif_chain;
    };

    // True iff good_thomas route is available for (T, n).
    // Generated from good_thomas_catalog (good_thomas.hpp).
    static constexpr bool good_thomas_available(std::size_t n) noexcept {
        return good_thomas_catalog::available<T>(n);
    }

    // True iff `rk` is instantiable for (T, size), the exact predicate the
    // force-route ctor tests. Public because availability is ISA- and
    // precision-dependent, so a caller must ask rather than catch the ctor's
    // exception. Pure predicate over compile-time catalogs; no state.
    static constexpr bool route_available(route_kind rk, std::size_t size) {
        switch (rk) {
        case route_kind::codelet:           return is_codelet_catalog(size);
        case route_kind::good_thomas:       return good_thomas_available(size);
        // 11-smooth, or anything the radix DP factors with the static kernels
        // plus generic runtime-prime middle passes (dif_generic_radices); see
        // dif_chain_supported.
        case route_kind::iterative_dif:     return dif_chain_supported(size);
        case route_kind::four_step:         return four_step_supported(size);
        case route_kind::four_step_batched: return four_step_batched_supported<T>(size);
        // Instantiable whenever a balanced split exists; the size threshold in
        // select_route is policy, not availability (a forced route ignores it).
        case route_kind::four_step_large:   return choose_large_split(size).valid();
        case route_kind::rader:             return rader_supported(size);
        case route_kind::bluestein:         return true;
        }
        return false;
    }

private:
    // .rodata-table routes (codelet, good_thomas) carry an empty tag;
    // plan-object routes (fsb, rader, bluestein) ARE the state.
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

    // All state in one aggregate `m`; size()/is_forward() getters avoid the _ suffix.
    struct M {
        std::size_t size;
        bool is_forward;
        route_kind route;
        route_state st;
        // Plan-owned workers, built at creation iff nthreads > 1 (see the ctor).
        // unique_ptr, not optional: the workers capture the pool address, so it
        // must survive a plan move; thread_pool holds a mutex and is not movable;
        // and get() on a const plan still yields a mutable pool*, keeping
        // execute() const.
        std::unique_ptr<thread_pool> pool;
    } m;

    // Pool iff the route is four_step_large: its executor is the only reader of m.pool,
    // so any other threaded plan would spawn parked workers it never wakes.
    [[nodiscard]] static std::unique_ptr<thread_pool> make_route_pool(std::size_t nthreads,
                                                                      std::size_t size,
                                                                      route_kind rk) {
        return (nthreads > 1 && size > 1 && rk == route_kind::four_step_large)
                   ? std::make_unique<thread_pool>(nthreads)
                   : nullptr;
    }

public:
    // Out-of-line (extern-template in src/inst_plan_*.cpp): inlining pulled the
    // whole route tree (~3.1 GiB peak per consumer TU). Runtime unchanged.

    // Test-only force-route ctor. Any route instantiable for (size, T) may be
    // forced; unavailable routes throw std::invalid_argument.
    // Always serial (no pool): threaded routes are exercised through the
    // ctor below, which mirrors the public nthreads-at-creation API.
    plan_impl(std::size_t size, bool is_forward, route_kind forced, std::size_t nthreads = 1);

    // nthreads: how many threads will drive this plan. Only four_step_large
    // executes its own passes in parallel, so it is the only route whose value
    // depends on it (see large_route_bytes); everything else ignores nthreads.
    // nthreads > 1 builds the plan-owned pool here; execute() then uses it
    // internally. There is no per-call threading knob.
    // eff: measure times the ranked candidates on this machine (module header).
    plan_impl(std::size_t size, bool is_forward, std::size_t nthreads = 1,
              const dif_factor_plan* dif_override = nullptr,
              admiral::effort eff = admiral::effort::estimate);

private:
    // Route-resolved delegate: the pool gate reads the route, which cannot be
    // referenced from m's own initializer list (clang -Werror,-Wuninitialized).
    // choice is deliberately by value: when the chain race won a re-ordering its
    // dif_chain member must outlive build_dif_twiddle_set in the body below.
    plan_impl(std::size_t size, bool is_forward, std::size_t nthreads,
              const dif_factor_plan* dif_override, measured_choice choice);

    // Emplace route state into m.st (m.route already set). Shared by both ctors.
    // codelet/good_thomas live in .rodata; codelet_state is the variant default.
    void emplace_route_state(std::size_t size, bool is_forward, const dif_factor_plan* dif_override) {
        switch (m.route) {
        case route_kind::codelet:
            break;  // codelet_state is the variant's default alternative
        case route_kind::good_thomas:
            m.st.template emplace<good_thomas_state>();
            break;
        // dif_state and four_step_state are aggregates, and variant::emplace
        // direct-initializes with parentheses. Braced-construct-then-move instead of
        // forwarding the members: parenthesized aggregate init (P0960) is C++20 but
        // AppleClang 15 does not implement it, and a move of two vectors at plan
        // build time costs nothing.
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

    // Execute in place. A plan built with nthreads > 1 threads four_step_large
    // internally on its plan-owned pool. opts.fct scales output (nullopt =
    // direction default). Out-of-line.
    void execute(std::span<std::complex<T>> data, const exec_options<T>& opts = {}) const;

    // src == dst: in-place. src != dst: reads src (preserved, no partial alias),
    // writes dst; no separate src->dst staging copy, since each route folds the read of
    // src into its first pass. Out-of-line.
    void execute(const std::complex<T>* src, std::complex<T>* dst,
                 const exec_options<T>& opts = {}) const;

    // `n` in-place lines at uniform `stride`, route resolved ONCE for the whole run.
    // The per-line path pays an out-of-line call, a direction branch and a std::visit
    // before any arithmetic, which at small `size` costs several times the transform
    // itself. Out-of-line.
    void execute_many(std::complex<T>* data, std::size_t n, std::size_t stride,
                      const exec_options<T>& opts = {}) const;

private:
    // exec_options::debug >= dbg_route. Out of line and cold so the guard at the call
    // site is the whole cost when tracing is off; every argument is already live there.
    ADM_NOINLINE ADM_COLD void trace(unsigned level, const char* how, T fct,
                                     std::size_t lines = 1, std::size_t stride = 0) const {
        dbg_print("N=", m.size, m.is_forward ? " fwd " : " inv ", route_name(), " ", how,
                  " fct=", static_cast<double>(fct), m.pool ? " threaded" : " serial");
        if (lines != 1) dbg_print("  lines=", lines, " stride=", stride);
        if (level < dbg_shape) return;
        // The shape is whatever the elected route actually holds; a route whose name is
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

    // The ranking select_route walked. Each form prints its modelled cycles and whether it
    // was buildable. The elected route is the first line that is both cheap and
    // available, so a surprising route reads off the table directly.
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

    // Run loop inside the visit, so the route is a compile-time type for every line.
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
    // Four-step split (n1,n2), or {0,0} if route is not four_step.
    [[nodiscard]] four_step_split four_step_split_used() const noexcept {
        const auto* fs = std::get_if<four_step_state>(&m.st);
        return fs ? fs->split : four_step_split{};
    }

private:
    // Inverse of the fitter's FORM_ORDER (tools/fit_cost_model.cpp).
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
        __builtin_unreachable();
    }

    // Measure-at-plan-time: time the model's top-ranked candidates on scratch
    // and return the measured winner. Stage 1 (base forms) stays in the fitted
    // domain, the only place a runtime A/B carries new information over the
    // fitted corrections; stage 2 races the DP's next-cheapest CHAINS, at any N.
    // Builds and discards one plan per candidate (plan-time only, never shipped).
    // Threaded engines are not stage-1 candidates (the ranked domain is serial).
    static measured_choice measure_route(std::size_t size, bool is_forward, std::size_t nthreads);

    // resolve a measuring effort: the timed winner when compiled in, the model
    // pick otherwise (knob accepted and inert under -DADM_MEASURE=OFF).
    static measured_choice measured_route(std::size_t size, [[maybe_unused]] bool is_forward,
                                          std::size_t nthreads) {
        if constexpr (adm_measure) return measure_route(size, is_forward, nthreads);
        else return measured_choice{select_route(size, nthreads), {}};
    }

    // four_step_large admission size; the lines themselves are in four_step_large.hpp,
    // beside the route they admit and the Bluestein arm that shares the f64 one.
    static constexpr std::size_t large_route_bytes(std::size_t nthreads) {
        // Threaded line is budget/nthreads clamped by a floor (see
        // large_route_threaded_bytes): one crossover law for every thread count, where a
        // fixed line fits one box and `nthreads >= 16` stranded desktop parts entirely.
        if (nthreads > 1) return large_route_threaded_bytes(nthreads);
        if constexpr (sizeof(T) == 8) return kLargeRouteSerialF64Bytes;
        else return kLargeRouteSerialF32Bytes;
    }

    // Bailey-split admission: byte line AND split shape. n2 % n1 != 0 sends both
    // in-place transposes to the element-cycle fallback, which is SERIAL -- the cliff
    // survives threading, so this clause binds every nthreads. Serial further requires
    // n1 % W == 0 (band fusion); threaded that clause would be WRONG, because the row
    // DFTs and transposes both take the pool.
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


    // iterative_dif availability: 11-smooth as before, or any N the radix DP can
    // factor with static kernels plus generic runtime-prime middle passes
    // (dif_generic_radices). The DP run is cheap against plan build (twiddle
    // generation dominates it).
    [[nodiscard]] static bool dif_chain_supported(std::size_t size) {
        if (std::has_single_bit(size) || is_codelet_supported(size)) return true;
        // The chain that would EXECUTE, which is not always the DP's argmin: the shape
        // vetoes (a big generic parked at ido < W) condemn some cheapest chains, and the
        // election walks the candidate list past them.
        return dif_chain_shape_ok(size, dif_elected_chain<T>(size));
    }

public:
    // Public for the same reason route_available is: it is the authoritative answer to
    // "will this build", and an explicitly supplied chain (--factors, the chain sweep)
    // needs it without going through the election. The rule itself lives next to the DP.
    [[nodiscard]] static bool dif_chain_shape_ok(std::size_t size, const dif_factor_plan& p) {
        return detail::dif_chain_shape_ok<T>(size, p);
    }

private:
    // True iff route rk has buildable state for (size, T) (instantiability, not
    // correctness). iterative_dif/bluestein build for any N.
    // Route ladder, first match wins.
    static route_kind select_route(std::size_t size, std::size_t nthreads) {
        // Modelled domain (2..512): walk the cost model's ranking and take the
        // cheapest route that is actually buildable, so an unavailable winner degrades
        // to the model's next choice. The gates below earn their keep only above
        // BASE_MODEL_NMAX, where there is no model at all.
        {
            const auto& ranking = base_route_ranking<T>(size);
            for (std::size_t i = 0; i < ranking.count; ++i) {
                const route_kind rk = form_to_route(ranking.form[i]);
                if (route_available(rk, size)) return rk;
            }
        }
        if (is_codelet_catalog(size)) return route_kind::codelet;
        // four_step_batched: f32 W==8 small-band only (fsb_split_for); not at W=16.
        // (vecpass survives in vecpass.hpp as the r2c tile engine; not routed here.)
        if (four_step_batched_supported<T>(size)) return route_kind::four_step_batched;
        // Large-N (Bailey): iterative_dif is DRAM-bound past L2 (~log_radix(N) full reads).
        // Factoring N=n1*n2 with cache-resident leaves streams only a handful of times.
        // Admission size is large_route_bytes(); runtime column tiling uses actual L2.
        if (large_route_admits(size, nthreads)) return route_kind::four_step_large;
        // No dif-vs-blue MODELLED comparison here: both sides' absolute scales misprice
        // the real routes, so gate terms pick losers. Structure decides instead:
        // availability (bookended chains only) plus the veto tier inside the DP.
        if (dif_chain_supported(size))
            return route_kind::iterative_dif;
        // Non-11-smooth N=n1*n2 with both <=64 catalog leaves (e.g. 143=11x13,
        // 289=17x17, 338=13x26): codelet composition instead of Bluestein.
        // Cost-gated: only when model says leaves beat Bluestein (169/209/221/247
        // keep Bluestein). Only fires on the Bluestein set (11-smooth dispatched above).
        if (four_step_supported(size) && four_step_beats_bluestein<T>(size))
            return route_kind::four_step;
        // Primes p>64 (67,71,73,...): Rader turns the DFT into a length-(p-1) cyclic
        // convolution via codelet/iterative_dif/four-step (no Bluestein recursion).
        // Gated on calibrated cost beating Bluestein; primes with expensive p-1
        // (79/83/127/191/251/283) keep Bluestein.
        if (rader_supported(size) && rader_beats_bluestein<T>(size))
            return route_kind::rader;
        return route_kind::bluestein;
    }

    // ------------------------------------------------------------------------
    // Per-route execute overloads (std::visit targets). UN-normalized; 1/N
    // folded into dif's last pass or a separate sweep on other routes.
    // ------------------------------------------------------------------------

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

    // Iterative DIF. Scratch: stack SBO for N<=SBO_MAX, heap otherwise.
    // First pass consumes `in` into SoA scratch -> OOP copy-free.
    // fct != 1: scale folded into dif_pass_last's store; fct == 1: unscaled.
    template<bool Forward>
    void execute_route(const dif_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        dif_execute_in_place<T>(Forward, in, out, m.size, st.tw, fct);
    }

    // Four-step (N = N1*N2, both <=64 catalog leaves). G is N-length scratch.
    template<bool Forward>
    void execute_route(const four_step_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        // Uninitialized: four_step_execute writes every G[n1*N2+k2] in its inner
        // pass before the outer pass reads any, so a std::vector's value-init is
        // a memset of the whole buffer that is immediately overwritten.
        soa_scratch<T, 1> scratch(2 * m.size);
        four_step_execute<T, Forward>(in, out, st.split.n1, st.split.n2, st.tw.data(),
                                      reinterpret_cast<std::complex<T>*>(scratch.buf(0)));
        apply_scale(out, fct);
    }

    // Batched four-step (f32 N1*N2): deinterleave -> batched -> reinterleave.
    // Direction baked at plan time (fsb.is_forward); Forward template arg unused here.
    template<bool Forward>
    void execute_route(const four_step_batched_plan<T>& fsb, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        fsb.execute(in, out);
        apply_scale(out, fct);
    }

    // Large-N four-step (DRAM-bound). fct folded into final row pass.
    // Direction baked at plan time (fsl.is_forward); Forward template arg unused here.
    // Threads its col/row passes on the plan-owned pool (m.pool at the call
    // sites). The ND row driver holds this contract by construction: the
    // sub-plans it shares across threads inside its parallel_for are built
    // with nthreads = 1, so their m.pool is null and the route runs inline.
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

    // Scale output by fct; no-op for fct==1 (branch outside hot loop).
    // dif and four_step_large fold scale into their terminal store; they bypass this.
    void apply_scale(std::complex<T>* out, T fct) const {
        if (fct != T(1)) scale_inplace(out, m.size, fct);
    }
};

// --- out-of-line plan_impl members ---

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, route_kind forced,
                        std::size_t nthreads)
    // The force-route trial takes the same pool rule: without the pool it would time
    // the SERIAL route under a threaded route's name.
    : m{.size = size, .is_forward = is_forward, .route = forced, .st = {},
        .pool = make_route_pool(nthreads, size, forced)}
{
    if (size == 0) [[unlikely]]
        throw std::invalid_argument("Plan size must be greater than 0");
    // Force any instantiable route; reject unavailable (size, T) combinations.
    if (!route_available(forced, size))
        throw std::invalid_argument("force-route: kernel unavailable for this size/precision");
    emplace_route_state(size, is_forward, nullptr);
}

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, std::size_t nthreads,
                        const dif_factor_plan* dif_override, admiral::effort eff)
    // dif_override forces iterative_dif: without this, select_route may pick
    // a different form and silently ignore the override, producing a false A/B baseline.
    // Only supplied by --factors/--factors-ab; production plans pass nullptr.
    // Route resolved once, here, so the delegate can gate the pool on it.
    : plan_impl(size, is_forward, nthreads, dif_override,
                [&] {
                    // This lambda runs BEFORE the delegate's size guard, and routing is
                    // undefined at 0: the chain DP hoists `dif_radix_admissible(0, 25)` out of
                    // its divisor loop (which is empty for 0), and is_pentanomial(0) divides 0
                    // by 2 forever. Answer trivially and let the delegate throw.
                    if (size == 0) [[unlikely]] return measured_choice{};
                    if (dif_override) return measured_choice{route_kind::iterative_dif, {}};
                    // effort::measure is kept for the FFTW mapping, not for a second budget.
                    if (eff != admiral::effort::estimate)
                        return measured_route(size, is_forward, nthreads);
                    return measured_choice{select_route(size, nthreads), {}};
                }()) {}

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, std::size_t nthreads,
                        const dif_factor_plan* dif_override, measured_choice choice)
    : m{.size = size, .is_forward = is_forward,
        .route = choice.route,
        .st = {},
        .pool = make_route_pool(nthreads, size, choice.route)}
{
    if (size == 0) [[unlikely]] {
        throw std::invalid_argument("Plan size must be greater than 0");
    }
    // Every radix in dif_radix_set is 11-smooth, so no chain can represent a
    // non-smooth N: forcing one would time a garbage A/B baseline. Reject loudly.
    if (dif_override && !route_available(route_kind::iterative_dif, size))
        throw std::invalid_argument("forced dif: size is not 11-smooth");
    const dif_factor_plan* chain =
        dif_override != nullptr      ? dif_override
        : choice.dif_chain.count != 0 ? &choice.dif_chain
                                      : nullptr;
    emplace_route_state(size, is_forward, chain);
}

// Race counts are fixed per stage, not clock-denominated: reproducible and machine-independent.
// Not a patience rule either: stopping after k non-improving candidates would read the
// candidate order as a ranking by measured value, and it is only ascending in MODEL cost.
inline constexpr std::size_t kMeasureCandidates = 8;
// Samples per candidate. A min of one rejects no outlier: an unlimited-budget race with a
// single sample elected a different chain every round at 8192/32768/25575.
inline constexpr std::size_t kMeasureReps = 5;
inline constexpr std::size_t kMeasureMaxCandidates = 4;
// Clock-resolution floor: a per-call time below this is the timer, not the plan.
inline constexpr std::chrono::nanoseconds::rep kMeasureMinNs = 50;
// Interval each sample must span. One small-N execution is shorter than the clock reads,
// so a sample of one times the clock and elects a coin flip; batch executions to clear it.
inline constexpr std::chrono::nanoseconds::rep kMeasureSampleNs = 4000;
// Executions per sample: the fewest whose span clears kMeasureSampleNs, given one
// execution costing one_ns. An execution that already spans it samples singly, which is
// every candidate above the small-N band.
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

    // Everything before the scratch allocation must be self-evident: at big N the
    // buffers are Θ(N) and a bluestein/fsl/gated call must not pay them.
    measured_choice pick{fallback, {}};

    // ---- stage 1 candidates (fitted domain): the model fallback FIRST -- a
    // degenerate race (clock failure, all-equal) returns it unchanged.
    // Top-2 distinct model candidates, plus the f32 batched route.
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

    // Above the fitted domain large_route_bytes() alone decides four_step_large, so the race is
    // the only correction to it. effort::estimate keeps the bare gate; the split must be
    // pool-safe (n2 % n1, as large_route_admits requires) for four_step_large to be buildable.

    // Both directions, or the race can only fix half the gate: a rejection needs four_step_large
    // priced against the fallback, an admission needs the DIF chain priced against it. One extra
    // plan build either way.
    if (size > BASE_MODEL_NMAX && nthreads > 1) {
        const large_split sp = choose_large_split(size);
        if (sp.valid() && sp.n2 % sp.n1 == 0) {
            offer(fallback);
            offer(fallback == route_kind::four_step_large ? route_kind::iterative_dif
                                                         : route_kind::four_step_large);
        }
    }

    // dif_chain_shape_ok is the same predicate route availability uses.
    const dif_chain_list chain_cands =
        fallback == route_kind::iterative_dif ? dif_chain_candidates<T>(size) : dif_chain_list{};
    bool race_chains = false;
    for (std::size_t i = 0; i < chain_cands.count && !race_chains; ++i)
        race_chains = detail::dif_chain_shape_ok<T>(size, chain_cands[i]);

    if (nc < 2 && !race_chains) return pick;

    // Live-ish data: an all-zero or all-one input rides denormal and
    // constant-folding shortcuts no real signal ever takes.
    std::vector<std::complex<T>> in(size), out(size);
    for (std::size_t i = 0; i < size; ++i)
        in[i] = {T(0.5 * int(i % 7) - 1), T(0.25 * int(i % 5) - 0.5)};

    // `unit` is the fastest execution of ANY plan for this transform seen so far. It is not a
    // budget, since the race is bounded by a candidate count. It is the yardstick the
    // reject ratio below measures a candidate against.
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
        // `unit` before this candidate: the fastest execution of ANY plan for this transform, so
        // the reject below is live in every race. Against the enclosing race's incumbent it would
        // be dead in the chain race, whose incumbent starts at +inf.
        const double u = unit;
        double best = kMeasureInf;
        for (std::size_t r = 0; r < kMeasureReps; ++r) {
            const auto a = clock::now();
            for (std::size_t k = 0; k < inner; ++k) trial.execute(in.data(), out.data());
            const auto span = std::chrono::nanoseconds(clock::now() - a).count();
            best = (std::min)(best, double((std::max)(span, kMeasureMinNs)) / double(inner));
            unit = (std::min)(unit, best);
            // A candidate a quarter behind the fastest plan seen cannot win, so stop
            // paying reps for it. TWO samples before that fires, never one: one sample
            // cannot separate a slow chain from the cold twiddle table the candidate's
            // own build just wrote.
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
        // Each race gets the candidate count for itself: which route and which
        // factorization are different questions; one shared window would let the route
        // candidates consume it before the chain race began.
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
        // Then WHICH ORDER of the winning multiset. The ordering walk is load-bearing: rotations
        // reach the cheap permutations; next_permutation covers the rest.
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
void plan_impl<T>::execute(std::span<std::complex<T>> data, const exec_options<T>& opts) const {
    if (data.size() != m.size) [[unlikely]] {
        throw std::invalid_argument("Data size doesn't match plan size");
    }
    execute(data.data(), data.data(), opts);
}

template<typename T>
void plan_impl<T>::execute(const std::complex<T>* src, std::complex<T>* dst,
                           const exec_options<T>& opts) const {
    // Direction default: forward=1, inverse=1/N.
    const T fct = opts.fct.value_or(m.is_forward ? T(1) : T(1) / T(m.size));
    if (opts.debug >= dbg_route) [[unlikely]]
        trace(opts.debug, src == dst ? "in-place" : "oop", fct);
    if (m.size == 1) [[unlikely]] { *dst = *src * fct; return; }  // ctor rejects size==0
    // Lift direction to compile-time once; hot path carries no runtime branch.
    if (m.is_forward) execute_impl<true>(src, dst, fct);
    else              execute_impl<false>(src, dst, fct);
}

template<typename T>
void plan_impl<T>::execute_many(std::complex<T>* data, std::size_t n, std::size_t stride,
                                const exec_options<T>& opts) const {
    const T fct = opts.fct.value_or(m.is_forward ? T(1) : T(1) / T(m.size));
    if (opts.debug >= dbg_route) [[unlikely]] trace(opts.debug, "many", fct, n, stride);
    if (m.size == 1) [[unlikely]] {  // ctor rejects size==0
        for (std::size_t r = 0; r < n; ++r) data[r * stride] *= fct;
        return;
    }
    // The variant's route was chosen for ONE line; a run is a different problem.
    // Lanes-as-lines fills the vector units that a single size-N codelet cannot
    // (radix_butterfly_v's full-width loop has zero iterations whenever a line has
    // < W columns), and it beats the per-line pick at every catalog size. Below W
    // lines the tile cannot fire and only the per-line dispatch is dropped.
    // Consequence: execute() and execute_many() of the same plan can take different
    // routes and agree only to roundoff; nothing downstream may assume bit identity.
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
