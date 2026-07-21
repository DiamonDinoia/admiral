#pragma once

#include <bit>
#include <complex>
#include <cstddef>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "base_cost_table.hpp" // generated winner table — see scripts/gen_base_cost_table.py
#include "bluestein.hpp"  // bluestein_plan
#include "dif_driver.hpp"   // iterative_dif_execute_ws, dif_twiddle_set
#include "four_step.hpp"    // four_step_supported, choose_four_step_split, four_step_execute
#include "four_step_large.hpp" // four_step_large_plan, four_step_large_supported (DRAM-bound large N)
#include "math.hpp"         // route predicates, codelet_dispatch
#include "good_thomas.hpp"  // good_thomas_catalog (routed Good-Thomas descriptor pack)
#include "rader.hpp"        // rader_plan, rader_supported (isolated primes >64)
#include "scratch.hpp"    // soa_scratch
#include "thread_pool.hpp" // thread_pool (opt-in intra-transform threading)
#include "twiddles.hpp"     // dif_twiddle_set, build_dif_twiddle_set, dif_factor_plan

namespace admiral {
namespace detail {

// Per-execute options for the transform engines, grouped so the execute()
// signatures stay stable as options accrue. Designated-initializer friendly:
//   plan.execute(data);                       // serial, direction-default scale
//   plan.execute(data, {.pool = p});          // threaded
//   plan.execute(data, {.fct = T(0.5)});      // custom output scale
// `pool` == nullptr is serial; `fct` == nullopt uses the direction default
// (forward = 1, inverse = 1/N), so a normalized round-trip is the identity.
template<typename T>
struct exec_options {
    thread_pool* pool = nullptr;
    std::optional<T> fct = std::nullopt;
};

// ============================================================================
// Plan Implementation (Pre-computed twiddle factors)
//
// A thin routing shell: select_route() picks one execution path at
// construction and precomputes only that path's state into a std::variant —
// exactly one alternative is engaged, heavy routes (bluestein, rader, fsb)
// are constructed in place, and execute() dispatches via std::visit onto the
// per-route execute_route overloads.
// ============================================================================

template<typename T>
class plan_impl {
public:
    enum class route_kind {
        codelet,
        iterative_dif,
        four_step,
        four_step_batched,
        four_step_large,   // DRAM-bound large N: cache-resident two-factor CT (four_step_large.hpp)
        rader,
        bluestein,
        good_thomas     // Good-Thomas PFA (routed sizes + eligibility: good_thomas_catalog in good_thomas.hpp)
    };

    // Returns true iff the good_thomas route can be constructed for this (T, n) pair.
    // Both this and the good_thomas execute_route are generated from the good_thomas_catalog
    // descriptor pack (good_thomas.hpp) — one row per routed (size, factors, precision).
    static constexpr bool good_thomas_available(std::size_t n) noexcept {
        return good_thomas_catalog::available<T>(n);
    }

private:
    // Per-route state. Routes whose tables are baked into .rodata carry an
    // empty tag; the plan-object routes (fsb, rader, bluestein) ARE the state.
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

    // All instance state lives in one internal aggregate `m` (no per-member _
    // suffix; the getters below keep the clean size()/is_forward() names).
    struct M {
        std::size_t size;
        bool is_forward;
        route_kind route;
        route_state st;
    } m;

public:
    // Defined out-of-line (below), instantiated once in c_api.cpp (see the
    // extern-template block at end of file): a consumer TU that used these
    // inline pulled the whole route tree (~3.1 GiB peak); the extern decl drops
    // it to the header-parse floor. Runtime is unchanged — the transform already
    // dispatches at runtime and was never inlined into callers.

    // Test-only: forced route for base_cost measurement and similar diagnostics.
    // Only codelet, good_thomas, and iterative_dif may be force-routed; all others throw
    // std::invalid_argument.  Does not alter default routing in any way.
    plan_impl(std::size_t size, bool is_forward, route_kind forced);

    plan_impl(std::size_t size, bool is_forward, const dif_factor_plan* dif_override = nullptr);

private:
    // Construct the selected route's state into the variant `m.st` (m.route is
    // already set). Shared by both ctors. codelet/good_thomas tables live in
    // .rodata — codelet carries no state (the variant's default alternative).
    void emplace_route_state(std::size_t size, bool is_forward, const dif_factor_plan* dif_override) {
        switch (m.route) {
        case route_kind::codelet:
            break;  // codelet_state is the variant's default alternative
        case route_kind::good_thomas:
            m.st.template emplace<good_thomas_state>();
            break;
        case route_kind::iterative_dif:
            m.st.template emplace<dif_state>(
                is_forward ? build_dif_twiddle_set<T, true>(size, dif_override)
                           : build_dif_twiddle_set<T, false>(size, dif_override));
            break;
        case route_kind::four_step: {
            const four_step_split split = choose_four_step_split(size);
            m.st.template emplace<four_step_state>(
                split, is_forward
                    ? build_four_step_twiddles<T, true>(split.n1, split.n2)
                    : build_four_step_twiddles<T, false>(split.n1, split.n2));
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

    // Execute in place. See exec_options: opts.pool (null == serial) threads the
    // iterative_dif engine for a large single transform; opts.fct scales the
    // output (nullopt == direction default). Out-of-line (below).
    void execute(std::span<std::complex<T>> data, exec_options<T> opts = {}) const;

    // src == dst is in-place; src != dst reads `src` (preserved, must not
    // partially alias `dst`) and writes `dst` — every route is copy-free
    // out-of-place (each gathers its input once at entry and scatters once at
    // exit). Out-of-line (below).
    void execute(const std::complex<T>* src, std::complex<T>* dst,
                 exec_options<T> opts = {}) const;

private:
    template<bool Forward>
    void execute_impl(const std::complex<T>* src, std::complex<T>* dst, T fct, thread_pool* pool) const {
        // Only iterative_dif threads a single transform; the pool reaches that one
        // overload, the rest keep a clean signature.
        std::visit([&](const auto& st) {
            if constexpr (std::is_same_v<std::decay_t<decltype(st)>, dif_state>)
                execute_route<Forward>(st, src, dst, fct, pool);
            else
                execute_route<Forward>(st, src, dst, fct);
        }, m.st);
    }

public:

    [[nodiscard]] std::size_t size() const noexcept { return m.size; }
    [[nodiscard]] bool is_forward() const noexcept { return m.is_forward; }

    // Name of the route select_route picked (diagnostics / --decomp-report only;
    // not on any hot path).
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
    // The chosen four-step split (n1,n2), or {0,0} when the route is not four_step.
    [[nodiscard]] four_step_split four_step_split_used() const noexcept {
        const auto* fs = std::get_if<four_step_state>(&m.st);
        return fs ? fs->split : four_step_split{};
    }

private:
    // Routing (first match wins):
    //   codelet       — N in the explicit compiled catalog.
    //   iterative_dif — pow2 / 11-smooth non-catalog sizes.
    //   four_step     — non-smooth composites (two <=64 catalog factors) where the
    //                   codelet composition beats Bluestein (cost-gated).
    //   rader         — isolated primes >64 (length-(p-1) convolution of kernels).
    //   bluestein     — everything else.
    // Returns true iff the measured winner for this (T, n) is iterative_dif.
    // Driven by the generated table (base_cost_table.hpp); receipts are
    // bench-results/base_cost_{v4,v3,v2}.txt, regenerated via
    // scripts/gen_base_cost_table.py.  Winner requires >=3% margin over the
    // runner-up; tied cells fall back to the original incumbent routing.
    static constexpr bool dif_beats_codelet(std::size_t n) {
        const auto entry = base_cost_for<T>(n);
        return entry.cyc >= 0.f && entry.form == base_form::iterative_dif;
    }

    // Route priority ladder (first match wins). Each predicate is a measured/
    // structural fact; see the per-route notes in dif_beats_codelet, the *_supported
    // helpers, and four_step.hpp for the calibration behind each.
    static constexpr route_kind select_route(std::size_t size) {
        // PFA: Good-Thomas zero-twiddle kernel.  The measured winner is read
        // from the generated table (base_cost_table.hpp); good_thomas_available() is
        // the mandatory eligibility gate that prevents routing to an
        // uninstantiated kernel (only v4 ISAs have good_thomas rows; the generator
        // script asserts that v3/v2 tables contain no good_thomas cells).
        if (size <= 64) {
            const auto entry = base_cost_for<T>(size);
            if (entry.form == base_form::good_thomas && good_thomas_available(size))
                return route_kind::good_thomas;
        }
        if (dif_beats_codelet(size))  return route_kind::iterative_dif;
        if (is_codelet_catalog(size)) return route_kind::codelet;
        // four_step_batched: f32 W==8 small-band only (fsb_split_for); the wide-radix
        // DP does not support it at W=16.
        // (vecpass, the other lane-batched route, is not routed here at any (T, W);
        // the vp:: kernels survive in vecpass.hpp as the r2c tile engine
        // (real_fft.hpp) and --vpass probe.)
        if constexpr (sizeof(T) == 4) {
            if (four_step_batched_supported<T>(size)) return route_kind::four_step_batched;
        }
        // Large-N four-step (Bailey): once the array no longer fits cache, the flat
        // iterative_dif pass chain is DRAM-bandwidth-bound — it read+writes the whole
        // array ~log_radix(N) times. Factoring N = n1*n2 with both leaves
        // cache-resident + one blocked transpose streams it only a handful of times.
        // F64-ONLY: the crossover is a precision fact, not a per-size override —
        // f32 reaches ~2x further per byte in cache, so its iterative_dif stays ahead
        // until far larger N. Same precision-gating shape as four_step_batched (f32-only).
        // 12 MiB is a coarse ROUTING constant; the exact machine cache still drives the
        // runtime column tiling.
        if constexpr (sizeof(T) == 8) {
            if (four_step_large_supported(size, sizeof(std::complex<T>), std::size_t{12} << 20))
                return route_kind::four_step_large;
        }
        // NB: a four-step split into two <=64 codelet leaves is slower here than
        // the iterative DIF chain (four_step.hpp gathers each leaf scalar-wise, no
        // cofactor SIMD batching); keep dif for pow2 / codelet-factorable smooth N.
        if (std::has_single_bit(size) || is_codelet_supported(size)) return route_kind::iterative_dif;
        // Non-11-smooth composites whose two factors are both <=64 catalog codelets
        // (e.g. 143=11x13, 289=17x17, 338=13x26) can be executed as a COMPOSITION of
        // two spill-free codelet leaves (four-step Cooley-Tukey) instead of Bluestein.
        // Cost-gated: only when the calibrated model says the codelet composition
        // beats Bluestein (cheap leaves and/or a Bluestein pad-size jump) — so no size
        // regresses (169/209/221/247 keep Bluestein). For the 11-smooth sizes above,
        // iterative_dif is already selected above, so this only ever fires on the
        // Bluestein set.
        if (four_step_supported(size) && four_step_beats_bluestein(size))
            return route_kind::four_step;
        // Isolated primes p>64 (67,71,73,79,83,89,97,...) — Rader's algorithm turns
        // the prime DFT into a length-(p-1) cyclic convolution run by the codelet
        // kernel paths (composition of kernels over the composite p-1), avoiding
        // Bluestein's chirp-z zero-padding. Gated on the inner p-1 transform being
        // codelet/iterative_dif/four-step (never recursing into Bluestein) AND the
        // calibrated recursive cost model saying Rader beats Bluestein, so primes
        // with a dear p-1 transform (79/83/127/191/251/283) correctly keep
        // Bluestein.
        if (rader_supported(size) && rader_beats_bluestein(size))
            return route_kind::rader;
        return route_kind::bluestein;
    }

    // ------------------------------------------------------------------------
    // Per-route execute overloads — std::visit target set. Every route computes
    // the UN-normalized transform in the planned direction; inverse applies 1/N
    // (folded into the last pass on the dif route, a separate sweep elsewhere).
    // ------------------------------------------------------------------------

    // PFA Good-Thomas kernel via the good_thomas_catalog descriptor pack. Per-row
    // admits<T> gates mirror good_thomas_available exactly, so ineligible kernels are
    // never instantiated. Forward is compile-time, so the inverse's per-element
    // imaginary-negate sweeps fold away in the forward instantiation.
    template<bool Forward>
    void execute_route(const good_thomas_state&, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        good_thomas_catalog::run<T, Forward>(in, out, m.size);
        apply_scale(out, fct);
    }

    // Compile-time codelet transform via the compiled codelet object library
    // (codelet_dispatch). Scratch is the codelet's own fixed stack buffer
    // (catalog sizes <= 64).
    template<bool Forward>
    void execute_route(const codelet_state&, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        codelet_dispatch<T, Forward>(in, out, m.size);
        apply_scale(out, fct);
    }

    // Iterative DIF transform. Scratch is allocated per execute: alloc-free
    // stack SBO for N <= SBO_MAX, otherwise a heap buffer via std::vector.
    // The first DIF pass consumes `in` into SoA scratch, so in != out is
    // copy-free. When fct != 1 the scale is folded into dif_pass_last's store
    // loop (Scale=true), so there is no separate scaling sweep; fct == 1 takes
    // the unscaled instantiation (the default forward path, byte-identical).
    template<bool Forward>
    void execute_route(const dif_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct, thread_pool* /*pool*/ = nullptr) const {
        soa_scratch<T, 4> sc(m.size);
        T* b0 = sc.buf(0); T* b1 = sc.buf(1); T* b2 = sc.buf(2); T* b3 = sc.buf(3);
        if (fct == T(1)) {
            iterative_dif_execute_ws<T, Forward, false>(in, out, m.size, b0, b1, b2, b3, st.tw);
        } else {
            iterative_dif_execute_ws<T, Forward, true>(in, out, m.size, b0, b1, b2, b3, st.tw, fct);
        }
    }

    // Four-step (N = N1*N2, both <=64 catalog leaves). G is N-length scratch.
    template<bool Forward>
    void execute_route(const four_step_state& st, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        std::vector<std::complex<T>> G(m.size);
        four_step_execute<T, Forward>(in, out, st.split.n1, st.split.n2,
                                      st.tw.data(), G.data());
        apply_scale(out, fct);
    }

    // Batched four-step (f32 N1*N2 route). Drop-in on interleaved complex
    // (deinterleave -> batched four-step -> reinterleave).
    template<bool Forward>
    void execute_route(const four_step_batched_plan<T>& fsb, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        fsb.execute(in, out);
        apply_scale(out, fct);
    }

    // Large-N four-step (DRAM-bound sizes). The plan folds `fct` into its final
    // row pass (no separate scaling sweep).
    template<bool Forward>
    void execute_route(const four_step_large_plan<T>& fsl, const std::complex<T>* in,
                       std::complex<T>* out, T fct) const {
        fsl.execute(in, out, fct);
    }

    // Rader prime route (isolated prime p>64 as a length-(p-1) cyclic
    // convolution / composition of codelet kernels). Direction baked at plan time.
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

    // Multiply the transform output by `fct` (a no-op for fct == 1, which is the
    // default forward factor — the branch is outside any hot loop). Routes that
    // fold the scale into their terminal store (dif, four_step_large) bypass this.
    void apply_scale(std::complex<T>* out, T fct) const {
        if (fct != T(1))
            for (std::size_t i = 0; i < m.size; ++i) out[i] *= fct;
    }
};

// --- out-of-line plan_impl members (see the note on the ctor declarations) ---

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, route_kind forced)
    : m{.size = size, .is_forward = is_forward, .route = forced, .st = {}}
{
    if (size == 0) [[unlikely]]
        throw std::invalid_argument("Plan size must be greater than 0");
    // Only codelet/good_thomas/iterative_dif may be force-routed; validate here,
    // then emplace via the shared switch (iterative_dif uses the default DP
    // factorisation, same as the nullptr path).
    switch (forced) {
    case route_kind::codelet:
        if (!is_codelet_catalog(size))
            throw std::invalid_argument("codelet force-route: size not in catalog");
        break;
    case route_kind::good_thomas:
        if (!good_thomas_available(size))
            throw std::invalid_argument("good_thomas force-route: unavailable for this size/precision");
        break;
    case route_kind::iterative_dif:
        break;
    default:
        throw std::invalid_argument(
            "force-route: only codelet/good_thomas/iterative_dif are supported");
    }
    emplace_route_state(size, is_forward, nullptr);
}

template<typename T>
plan_impl<T>::plan_impl(std::size_t size, bool is_forward, const dif_factor_plan* dif_override)
    // An explicit DIF factorization is only meaningful for the iterative_dif
    // engine, so honor it by forcing that route. Without this, select_route can
    // independently land on four_step/codelet/bluestein and SILENTLY ignore the
    // override — the bench would then report a number labelled with a
    // factorization it never ran (a false A/B baseline). dif_override is supplied
    // only by the benchmark's --factors / --factors-ab paths; production plans
    // pass nullptr and route normally.
    : m{.size = size, .is_forward = is_forward,
        .route = dif_override ? route_kind::iterative_dif : select_route(size),
        .st = {}}
{
    if (size == 0) [[unlikely]] {
        throw std::invalid_argument("Plan size must be greater than 0");
    }
    emplace_route_state(size, is_forward, dif_override);
}

template<typename T>
void plan_impl<T>::execute(std::span<std::complex<T>> data, exec_options<T> opts) const {
    if (data.size() != m.size) [[unlikely]] {
        throw std::invalid_argument("Data size doesn't match plan size");
    }
    execute(data.data(), data.data(), opts);
}

template<typename T>
void plan_impl<T>::execute(const std::complex<T>* src, std::complex<T>* dst,
                           exec_options<T> opts) const {
    // Direction default: forward is unscaled, inverse divides by N.
    const T fct = opts.fct.value_or(m.is_forward ? T(1) : T(1) / T(m.size));
    if (m.size == 1) [[unlikely]] { *dst = *src * fct; return; }  // ctor rejects size==0
    // Lift the direction to a compile-time template once, here — every route
    // below is Forward-templated, so the hot path carries no runtime direction
    // branch (both instantiations already exist: admiral::plan builds fwd+inv).
    if (m.is_forward) execute_impl<true>(src, dst, fct, opts.pool);
    else              execute_impl<false>(src, dst, fct, opts.pool);
}

#ifndef ADM_INSTANTIATE_ENGINE
extern template class plan_impl<float>;
extern template class plan_impl<double>;
#endif

} // namespace detail
} // namespace admiral
