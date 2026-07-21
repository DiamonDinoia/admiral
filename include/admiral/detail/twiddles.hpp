#pragma once

// ============================================================================
// Precomputed inter-stage twiddle tables for the iterative DIF driver.
//
// Built once per plan and reused across every execute(); the hot pass loops do
// only indexed reads, no runtime trigonometry. Twiddles are generated with the
// exact integer turn reduction in portable_trig::sincos_turns.
// ============================================================================

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <poet/poet.hpp>
#include <xsimd/xsimd.hpp>

#include "ct_math.hpp"    // smallest_radix
#include "portable_trig.hpp"  // sincos_turns

namespace admiral {
namespace detail {

struct dif_factor_plan {
    static constexpr std::size_t max_passes = 32;

    std::array<unsigned, max_passes> radices{};
    std::size_t count = 0;
    // Non-zero when the DP elected a base codelet terminal of this size.
    // The factored pass chain in radices[] reduces N to base_n; the remaining
    // sub-transform of size base_n is handled by codelet_dispatch.  Zero means
    // the chain factors all the way to n=1 (default behaviour, unchanged).
    unsigned base_n = 0;

    constexpr void push(unsigned radix) {
        if (count < max_passes) {
            radices[count++] = radix;
        }
    }

    [[nodiscard]] constexpr unsigned operator[](std::size_t index) const {
        return radices[index];
    }
};

// Single source of truth for the radix set the iterative DIF passes are
// instantiated for. An explicit value list (not a contiguous range) so only the
// radices the factorizer actually emits get compiled: {2,3,4,5,7} for 7-smooth,
// 8 for pow2 cofactors (preferred by build_dif_factor_plan — fewer, wider
// passes), and 11 for 11-smooth composites (e.g. 121 = 11^2). Skips the
// never-emitted 6/9/10. dif_radix_set drives poet::dispatch in dif_driver/
// dif_col_driver/vecpass; dif_candidate_radices is the runtime-iterable form
// used by the cost-model factorizer below — both derive from the same sequence.
// Wide register files (32 vector regs = AVX-512) additionally admit the fused
// pow2 radices 16/32: one radix-16 pass replaces two radix-4 full-array sweeps,
// and the recursive split-scope pow2_dif_butterfly keeps peak-live bounded well
// under 32. On 16-reg ISAs radix-16 register-starves (see docs/pow2-fftw-codelet-frontier.md),
// so the admission is register-count-parametric.
inline constexpr bool dif_wide_radices = poet::vector_register_count() >= 32;
// 9/15 (merged 3·3 and 3·5 passes through the generic odd symmetric kernel)
// are wide-only: one r9 pass replaces two r3 full-array sweeps; at 16 regs their live
// set (radix+9) register-starves like r16 does.
using dif_radix_set = std::conditional_t<
    dif_wide_radices,
    std::integer_sequence<int, 2, 3, 4, 5, 7, 8, 11, 16, 32, 9, 15>,
    std::integer_sequence<int, 2, 3, 4, 5, 7, 8, 11>>;

template<int... Rs>
constexpr auto radix_seq_to_array(std::integer_sequence<int, Rs...>) {
    return std::array<unsigned, sizeof...(Rs)>{static_cast<unsigned>(Rs)...};
}
// DP factorizer candidates: same set as dif_radix_set (single source of truth,
// register-parametric — 16-reg ISAs still exclude 16/32). The DP must be able to
// end chains in 16/32: at W=16 every last-pass radix except 11/16/32 takes the
// small-IP gather branch, making trailing-2 chains structurally slower.
inline constexpr auto dif_candidate_radices = radix_seq_to_array(dif_radix_set{});

// ============================================================================
// Measured per-pass cost model.
//
// ONE measured table, no analytic multipliers: cyc/elem per (radix, regime)
// for each (precision, SIMD width) — width is compile-time, so this is per
// ISA level. Regimes:
//   vec    ido >= W   (a-vectorized full-lane pass; also the first pass)
//   valley 1 < ido < W (partial lanes / lane-over-b path)
//   last   ido == 1   (lane-over-b, trivial twiddles)
// Register starvation is captured in the table values, not a 2r+10 formula.
// Missing valley cells (a width where 1<ido<W is empty or unprobed) fall back
// to the vec value.
// ============================================================================

struct dif_cost_row { double vec, valley, last; };

// Analytical fallback for (precision, W, registers) keys with NO measured
// table below (non-x86 widths: SVE, RVV, AVX10/256, ...). Physically-derived
// terms, constants fitted on the six measured x86 tables (log-rms ~±35% —
// coarse in absolute terms but preserves the structural rankings that are
// compile-time knowable: register-spill walls, lane waste, stream counts):
//   traffic  4*sizeof(T) bytes/elem through ~12.6 B/cyc L2-class bandwidth,
//            degraded by concurrent read streams beyond ~4 (the radix)
//   compute  flops/elem / (0.77 * 4W flops/cyc)  [2 FMA ports assumed]
//   spill    1.5 cyc per (peak-live - registers) batch, amortized over W;
//            peak-live: flat pow2 kernel 2r+6, recursive split-scope >8
//            bounded ~22-26, odd kernels r+9
//   valley   3x vec when the radix has no small-ido fast path, 1.25x when it
//            does (see small_ido_set); last ~= one traffic trip, half spill.
template<typename T>
[[nodiscard]] constexpr dif_cost_row dif_analytical_cost(unsigned radix) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr std::size_t regs = poet::vector_register_count();
    constexpr double bytes_pe = 4.0 * sizeof(T);
    const bool pow2 = (radix & (radix - 1u)) == 0u;
    const double flops = radix == 2 ? 5 : radix == 3 ? 8 : radix == 4 ? 8.5
                       : radix == 5 ? 12 : radix == 7 ? 16 : radix == 8 ? 12.5
                       : radix == 9 ? 19 : radix == 11 ? 22 : radix == 15 ? 34
                       : radix == 16 ? 17 : 21.5;
    const double liv = !pow2 ? radix + 9.0
                     : radix <= 8 ? 2.0 * radix + 6.0
                     : radix == 16 ? 24.0 : 26.0;
    const double traffic = bytes_pe / 12.6
                         * (1.0 + 0.056 * (radix > 4 ? radix - 4.0 : 0.0));
    const double comp = flops / (0.77 * 4.0 * static_cast<double>(W));
    const double spill = 1.5 * (liv > static_cast<double>(regs) ? liv - static_cast<double>(regs) : 0.0)
                       / static_cast<double>(W);
    const double vec = (comp > traffic ? comp : traffic) + spill;
    const bool haspath = (radix == 8 && W >= 4) || (radix == 5 && W == 8);
    const double vall = vec * (haspath ? 1.25 : 3.0);
    const double lastc = (comp > bytes_pe / 12.6 ? comp : bytes_pe / 12.6) + 0.5 * spill;
    return {vec, vall, lastc};
}

template<typename T>
[[nodiscard]] constexpr dif_cost_row dif_measured_cost(unsigned radix) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr std::size_t regs = poet::vector_register_count();
    // radix -> table index: 2,3,4,5,7,8,11,16,32,9,15 (9/15 appended to keep
    // the original nine columns' indices stable)
    [[maybe_unused]] const std::size_t i = radix == 2 ? 0 : radix == 3 ? 1 : radix == 4 ? 2
                        : radix == 5 ? 3 : radix == 7 ? 4 : radix == 8 ? 5
                        : radix == 11 ? 6 : radix == 16 ? 7 : radix == 32 ? 8
                        : radix == 9 ? 9 : 10;
    if constexpr (sizeof(T) == 8 && W == 8 && regs == 32) {   // f64 AVX-512
        constexpr dif_cost_row t[11] = {
            {2.85, 3.47, 2.89}, {2.90, 3.19, 1.98}, {3.03, 3.24, 2.21},
            {3.21, 3.00, 1.80}, {2.64, 3.88, 1.71}, {3.79, 4.11, 1.85},
            {3.85, 6.16, 3.28}, {4.16, 20.6, 2.92}, {5.60, 23.4, 3.66},
            {4.00, 7.70, 4.80}, {5.20, 22.2, 3.77}};
        return t[i];
    } else if constexpr (sizeof(T) == 8 && W == 4 && regs == 16) { // f64 AVX2 (no valley probe: vec fallback)
        // 9/15 rows are sentinels: not in the 16-reg radix set, never emitted.
        constexpr dif_cost_row t[11] = {
            {2.23, 2.23, 2.28}, {2.10, 2.10, 2.08}, {2.62, 2.62, 2.39},
            {2.53, 2.53, 2.38}, {3.50, 3.50, 3.04}, {6.10, 6.10, 2.70},
            {4.19, 4.19, 4.26}, {6.01, 6.01, 4.66}, {7.00, 7.00, 5.84},
            {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
        return t[i];
    } else if constexpr (sizeof(T) == 8 && W == 2 && regs == 16) { // f64 SSE (W=2: no valley exists)
        constexpr dif_cost_row t[11] = {
            {2.76, 2.76, 2.43}, {2.18, 2.18, 2.16}, {2.84, 2.84, 2.34},
            {2.73, 2.73, 2.22}, {3.26, 3.26, 3.11}, {9.99, 9.99, 3.84},
            {4.83, 4.83, 4.51}, {10.2, 10.2, 7.77}, {11.2, 11.2, 8.70},
            {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
        return t[i];
    } else if constexpr (sizeof(T) == 4 && W == 16 && regs == 32) { // f32 AVX-512
        constexpr dif_cost_row t[11] = {
            {1.34, 3.62, 1.13}, {1.20, 4.53, 1.23}, {1.48, 4.09, 3.69},
            {1.48, 1.30, 3.20}, {1.31, 4.80, 3.05}, {2.02, 2.00, 1.40},
            {1.60, 7.37, 1.37}, {1.98, 19.4, 1.68}, {2.21, 21.1, 1.63},
            {1.74, 5.73, 2.35}, {2.35, 15.8, 1.73}};
        return t[i];
    } else if constexpr (sizeof(T) == 4 && W == 8 && regs == 16) { // f32 AVX2
        constexpr dif_cost_row t[11] = {
            {1.10, 1.79, 1.68}, {1.07, 3.61, 1.68}, {1.33, 3.96, 1.69},
            {1.17, 2.65, 1.27}, {1.52, 5.91, 1.50}, {3.00, 3.23, 1.51},
            {2.05, 9.22, 1.81}, {2.98, 20.1, 2.31}, {3.39, 22.5, 2.47},
            {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
        return t[i];
    } else if constexpr (sizeof(T) == 4 && W == 4 && regs == 16) { // f32 SSE (no valley probe: vec fallback)
        constexpr dif_cost_row t[11] = {
            {1.42, 1.42, 1.26}, {1.20, 1.20, 1.18}, {1.55, 1.55, 1.06},
            {1.42, 1.42, 1.38}, {1.72, 1.72, 1.84}, {2.60, 2.60, 1.15},
            {2.38, 2.38, 2.46}, {5.07, 5.07, 3.79}, {5.78, 5.78, 3.86},
            {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
        return t[i];
    } else {  // uncovered (W, regs): SVE, RVV, AVX10/256, ... -> analytical model
        return dif_analytical_cost<T>(radix);
    }
}

// Footprint multiplier for STREAMING passes (big ido: many concurrent
// ido-strided planes). Per-side working set = N * 2 * sizeof(T) bytes.
// Three anchor sizes (128KB ref / 1MB / 4MB), log-interpolated between them,
// clamped outside. The multiplier applies only when ido >= 512; mid passes
// (small ido, big l1) are footprint-flat.
// Higher radix opens more concurrent streams (r32: 32 streams, prefetcher/TLB
// overrun), so the 4MB anchor is larger for r32 than for r<=8.
[[nodiscard]] constexpr double dif_footprint_mult(unsigned radix, std::size_t ido,
                                                  std::size_t side_bytes) {
    if (ido < 512) return 1.0;
    const double m1 = radix >= 32 ? 2.05 : radix >= 16 ? 1.57 : 1.42;  // at 1MB
    const double m4 = radix >= 32 ? 2.33 : 1.55;                       // at 4MB
    constexpr double kRef = 128.0 * 1024.0, kMid = 1024.0 * 1024.0, kBig = 4.0 * kMid;
    const double b = static_cast<double>(side_bytes);
    if (b <= kRef) return 1.0;
    // ct-friendly log2 via repeated halving is overkill: a rational proxy
    // t = (b-lo)/(hi-lo) ranks identically within each segment.
    if (b <= kMid) return 1.0 + (m1 - 1.0) * (b - kRef) / (kMid - kRef);
    if (b <= kBig) return m1 + (m4 - m1) * (b - kMid) / (kBig - kMid);
    return m4;
}

// Terminal base-codelet cost in cyc/elem, for use as a DP seed (see
// build_dif_factor_plan).  Returns a NEGATIVE value for all k to signal
// "no terminal installed" — the cost table is intentionally empty until
// in-chain --terminal-ab receipts are collected.  Never populate from
// isolated microbenches; the isolated-vs-in-chain trap documented in
// dif_measured_cost applies equally here.
template<typename T>
[[nodiscard]] constexpr double dif_terminal_cost([[maybe_unused]] std::size_t k) noexcept {
#ifdef ADM_TC64_PROBE
    // Probe-only hook (never defined in production builds): candidate k=64
    // f64 terminal cost for the DP flip-window search.
    if (k == 64 && sizeof(T) == 8) return ADM_TC64_PROBE;
#endif
    return -1.0;  // no terminals installed; costs to be added from --terminal-ab receipts
}

// Per-stage cost in cyc/elem * N (comparable across chains of the same N).
// n = remaining transform length entering this stage.
template<typename T>
[[nodiscard]] constexpr double dif_stage_cost(std::size_t N, std::size_t n, unsigned radix) {
    const std::size_t ido = n / radix;
    constexpr std::size_t W = xsimd::batch<T>::size;
    const dif_cost_row row = dif_measured_cost<T>(radix);
    const double per_elem = (ido == 1) ? row.last
                          : (ido < W)  ? row.valley
                                       : row.vec * dif_footprint_mult(radix, ido, N * 2u * sizeof(T));
    // Lane-tail surcharge: a partial last vector block (ido % W) costs a small
    // fixed extra; keeps composite sizes with ragged idos honestly ranked.
    const std::size_t tail = (ido > 1) ? (ido % W) : ((N / n) % W);
    const double tail_mult = tail ? 1.0 + 0.025 * static_cast<double>(tail) / static_cast<double>(W)
                                  : 1.0;
    // Ordering epsilon for 16-reg ISAs: equal-cost permutations of the same
    // radix multiset (no regime difference) are unrankable by an additive
    // model. Sum(radix*n) is minimized by ascending radix order (placing
    // cheaper low-pressure radices on the larger-ido first passes).
    // The weight is ~9 orders below real cost terms, so it ONLY decides
    // exact ties. 32-reg ISAs use the DP tie-break instead.
    const double order_eps = poet::vector_register_count() < 32
        ? 1e-9 * static_cast<double>(radix) * static_cast<double>(n) : 0.0;
    return per_elem * tail_mult * static_cast<double>(N) + order_eps;
}

// Post-DP reorder for f64 odd-radix composites. The DP ranks plans on additive
// per-pass cost, which sorts SMALL radices first and so clusters >=2 expensive
// odd radices (5/7/11) at the small-ido tail — exactly the 1<ido<W scalar valley
// where odd radices spill. Moving the expensive odds to the extremes (one at the
// largest ido = first pass, one at ido=1 = last pass, with 3s/pow2 in the middle)
// reduces total valley time. f32 and radix-8 sizes are scoped out (the radix-8
// ido=1 pass dominates). A per-pass DP cost term can't encode this (it's
// cross-pass); a reorder of the final chain can. Sizes where a hand-tuned order
// beats this generic reorder keep an explicit measured_dif_factor_plan entry
// (checked first).
template<typename T>
[[nodiscard]] constexpr dif_factor_plan reorder_odd_radices_to_extremes(const dif_factor_plan& in) {
    if constexpr (sizeof(T) != 8) {
        return in;
    } else {
        unsigned exotic[dif_factor_plan::max_passes];
        std::size_t n_exotic = 0, n_threes = 0;
        unsigned pow2[dif_factor_plan::max_passes];
        std::size_t n_pow2 = 0;
        bool has8 = false;
        for (std::size_t i = 0; i < in.count; ++i) {
            const unsigned r = in[i];
            if (r == 5 || r == 7 || r == 11) exotic[n_exotic++] = r;
            else if (r == 3) ++n_threes;
            else { if (r == 8) has8 = true; pow2[n_pow2++] = r; }
        }
        if (has8 || n_exotic < 2) return in;            // rule scope: f64, no radix-8, >=2 expensive odds
        // sort exotic and pow2 ascending (small multisets, insertion sort keeps it constexpr)
        for (std::size_t i = 1; i < n_exotic; ++i)
            for (std::size_t j = i; j > 0 && exotic[j-1] > exotic[j]; --j)
                { unsigned t = exotic[j]; exotic[j] = exotic[j-1]; exotic[j-1] = t; }
        for (std::size_t i = 1; i < n_pow2; ++i)
            for (std::size_t j = i; j > 0 && pow2[j-1] > pow2[j]; --j)
                { unsigned t = pow2[j]; pow2[j] = pow2[j-1]; pow2[j-1] = t; }
        // first = smallest expensive odd (5 before 7); last = largest remaining; middle = rest odds, 3s, pow2 asc.
        dif_factor_plan out;
        out.push(exotic[0]);
        const unsigned last = exotic[n_exotic - 1];
        for (std::size_t i = 1; i + 1 < n_exotic; ++i) out.push(exotic[i]);
        for (std::size_t i = 0; i < n_threes; ++i) out.push(3u);
        for (std::size_t i = 0; i < n_pow2; ++i) out.push(pow2[i]);
        out.push(last);
        return out;
    }
}

// Per-pass fusion role in the row driver's (iterative_dif_execute_ws) walk.
enum class dif_fuse : std::uint8_t { plain = 0, f2head, f2tail, f3head, f3tail };

// Fusion only for N >= kDifFuseMinN (L2-latency-bound band, see dif_driver.hpp).
// 16-reg ISAs admit 4096: their pow2 chains at 4096 are all full-array sweeps
// of narrow radices, so sweep count dominates and fusion helps.
// f64 additionally admits 2048 (working set already 32KB, sweeps dominate);
// f32 2048 fits L1 per pass so fusion's tile staging overhead dominates —
// kept at 4096.
template<typename T>
inline constexpr std::size_t kDifFuseMinN =
    poet::vector_register_count() <= 16 ? (sizeof(T) == 8 ? 2048 : 4096) : 8192;

// Plan-time fusion schedule for the row driver: one entry per pass. This is
// the SINGLE source of truth for which middle passes fuse — the driver executes
// exactly this schedule (no runtime gate re-checks) and build_dif_twiddle_set
// chooses each pass's twiddle representation from it (packed_pair[p] non-empty
// iff f2head; on a row-form set the plain tables of f2head/f2tail passes are
// dropped, so plain and packed are never stored twice).
//
// Gates (fusion policy, see dif_driver.hpp): fused3 (three adjacent r4, one
// tile sweep) applies only to f64 8192/16384; above 256KB working set its
// tile staging costs more than the saved sweep, so fused2 pairs take over.
// Alignment gates guarantee exact W-tiling at the inner (ido2) level.
// Array-form core (constexpr: also runs inside the consteval pow2 plan table).
template<typename T>
constexpr void dif_fusion_schedule_into(std::size_t N, const unsigned* radices,
                                        std::size_t n, dif_fuse* sched) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    for (std::size_t i = 0; i < n; ++i) sched[i] = dif_fuse::plain;
    if (N < kDifFuseMinN<T> || n < 3) return;
    const bool fused3_ok = sizeof(T) == 8 && N <= 16384u;
    std::size_t l1 = radices[0];  // first pass (p=0) never fuses
    for (std::size_t p = 1; p + 1 < n; ++p) {
        const unsigned ip = radices[p];
        const std::size_t ido = N / (l1 * ip);
        if (fused3_ok && p + 3 < n && ip == 4u && radices[p + 1] == 4u
            && radices[p + 2] == 4u && ido % (16u * W) == 0u) {
            sched[p] = dif_fuse::f3head;
            sched[p + 1] = sched[p + 2] = dif_fuse::f3tail;
            l1 *= 64u;
            p += 2;
            continue;
        }
        if (p + 2 < n) {
            const unsigned r2 = radices[p + 1];
            const bool pair_ok = (ip == 4u && r2 == 4u && ido % (4u * W) == 0u)
                              || (ip == 4u && r2 == 8u && ido % (8u * W) == 0u)
                              || (ip == 8u && r2 == 4u && ido % (4u * W) == 0u)
                              || (ip == 8u && r2 == 8u && ido % (8u * W) == 0u);
            if (pair_ok) {
                sched[p] = dif_fuse::f2head;
                sched[p + 1] = dif_fuse::f2tail;
                l1 *= static_cast<std::size_t>(ip) * r2;
                p += 1;
                continue;
            }
        }
        l1 *= ip;
    }
}

template<typename T>
[[nodiscard]] inline std::vector<dif_fuse>
dif_fusion_schedule(std::size_t N, const std::vector<unsigned>& radices) {
    std::vector<dif_fuse> sched(radices.size());
    dif_fusion_schedule_into<T>(N, radices.data(), radices.size(), sched.data());
    return sched;
}

// Fused-group discount for the whole-chain pow2 scoring below. A fused2/fused3
// group runs its member passes through one L1-tiled sweep, so a member costs a
// FRACTION of its isolated full-sweep cost. The fraction is footprint-class
// dependent (per-side SoA working set = N * 2 * sizeof(T)).
// Above ~512KB per side the pass working set (read+write = 2 sides) no longer
// fits the 2MB L2, so the tile's saved round-trips are L3-hits, not DRAM, and
// fusion buys little (g -> 1).
[[nodiscard]] constexpr double dif_fuse_discount(std::size_t side_bytes) {
    return side_bytes <= 512u * 1024u ? 0.40 : 0.90;
}

// Whole-chain pow2 selection for the fusion band (N >= kDifFuseMinN): the
// fused2/fused3 gates are a cross-pass effect the additive DP cannot see, so
// exhaustively score every ordered composition of lg2 N over the admitted
// radix set (plan-time only, at most a few hundred chains):
//     cost = sum dif_stage_cost, with fused-group members discounted.
// dif_fusion_schedule supplies the driver's REAL gates. Below the fusion band
// no pass fuses, so the additive DP stays authoritative.
template<typename T>
[[nodiscard]] constexpr dif_factor_plan enumerate_pow2_dif_plan(std::size_t N) {
    unsigned chain[dif_factor_plan::max_passes]{};
    dif_factor_plan best;
    // Finite sentinel, not infinity: -ffast-math makes infinity UB.
    constexpr double kUnset = 1e300;
    double best_cost = kUnset;
    const double fuse_g = dif_fuse_discount(N * 2u * sizeof(T));
    // Score the prefix chain[0..len) with the driver's real fusion gates, plus
    // an optional terminal base sweep (base > 1; the terminal never fuses).
    const auto score = [&](std::size_t len, std::size_t base, double base_cost) {
        dif_fuse sched[dif_factor_plan::max_passes]{};
        dif_fusion_schedule_into<T>(N, chain, len, sched);
        double cost = base_cost;
        std::size_t n = N;
        for (std::size_t i = 0; i < len; ++i) {
            const double pc = dif_stage_cost<T>(N, n, chain[i]);
            cost += (sched[i] == dif_fuse::plain) ? pc : fuse_g * pc;
            n /= chain[i];
        }
        if (cost < best_cost) {
            best_cost = cost;
            best = {};
            for (std::size_t i = 0; i < len; ++i) best.push(chain[i]);
            best.base_n = static_cast<unsigned>(base > 1 ? base : 0);
        }
    };
    auto rec = [&](auto&& self, std::size_t rem, std::size_t len) -> void {
        if (rem == 1) {
            score(len, 1, 0.0);
            return;
        }
        // Terminal edge: end the chain here with a base-rem lane-packed codelet
        // sweep (dif_terminal_cost is cyc/elem over the whole array, like
        // dif_stage_cost). Negative cost = terminal not installed for rem.
        if (rem <= 64 && len > 0) {
            const double tc = dif_terminal_cost<T>(rem);
            if (tc >= 0.0)
                score(len, rem, tc * static_cast<double>(N));
        }
        // Wide-register ISAs also enumerate 16/32 (see dif_wide_radices).
        constexpr unsigned cands[] = {4u, 8u, 16u, 32u};
        constexpr std::size_t n_cands = dif_wide_radices ? 4u : 2u;
        for (std::size_t ci = 0; ci < n_cands; ++ci) {
            const unsigned r = cands[ci];
            if (rem % r == 0) {
                chain[len] = r;
                self(self, rem / r, len + 1);
            }
        }
    };
    rec(rec, N, 0);
    return best;
}

template<typename T>
[[nodiscard]] inline dif_factor_plan build_dif_factor_plan(std::size_t N) {
    // The enumerator is constexpr-capable, but plans are built once at runtime
    // (microseconds, dwarfed by twiddle generation) — eagerly materializing a
    // consteval plan table costs ~0.7s of constant evaluation per TU for zero
    // runtime gain, so it deliberately does not exist.
    if (N >= kDifFuseMinN<T> && (N & (N - 1u)) == 0u) {
        return enumerate_pow2_dif_plan<T>(N);
    }

    // Finite sentinel, not infinity: -ffast-math makes infinity UB.
    constexpr double kUnreachable = 1e300;
    // Sentinel radix for a terminal base-codelet edge in the DP.  Distinguished
    // from any real radix (max 32) so backtracking can recognise it and stop.
    constexpr unsigned BASE_TERMINAL_RADIX = 0xFFu;

    struct entry {
        double cost = kUnreachable;
        unsigned radix = 0;
    };

    std::vector<entry> dp(N + 1);
    dp[1].cost = 0.0;

    // Seed terminal entries: for each k in [2, min(N,64)] that divides N and
    // has a non-negative terminal cost, dp[k] is seeded as a codelet base.
    // With all dif_terminal_cost<T> values negative (the default) this loop
    // is a no-op and nothing changes downstream.
    for (std::size_t k = 2; k <= std::min(N, std::size_t{64}); ++k) {
        if (N % k != 0) continue;
        const double tc = dif_terminal_cost<T>(k);
        if (tc < 0.0) continue;
        dp[k].cost = tc * static_cast<double>(N);
        dp[k].radix = BASE_TERMINAL_RADIX;
    }

    for (std::size_t n = 2; n <= N; ++n) {
        for (unsigned radix : dif_candidate_radices) {
            if (n % radix != 0) continue;
            const std::size_t next = n / radix;
            if (dp[next].cost >= kUnreachable) continue;
            const double cost = dif_stage_cost<T>(N, n, radix) + dp[next].cost;
            const bool better = cost < dp[n].cost;
            const bool tie = cost == dp[n].cost && radix > dp[n].radix;
            if (better || tie) {
                dp[n].cost = cost;
                dp[n].radix = radix;
            }
        }
    }

    dif_factor_plan plan;
    for (std::size_t n = N; n > 1;) {
        const unsigned radix = dp[n].radix;
        if (radix == BASE_TERMINAL_RADIX) {
            // Terminal reached: record the base size and stop factoring.
            plan.base_n = static_cast<unsigned>(n);
            break;
        }
        if (radix == 0 || n % radix != 0) {
            plan = {};
            while (n > 1) {
                const unsigned fallback = smallest_radix(static_cast<unsigned>(n));
                plan.push(fallback);
                n /= fallback;
            }
            return plan;
        }
        plan.push(radix);
        n /= radix;
    }
    // Reorder only applies to the pure-radix (no terminal) case; terminal plans
    // must not be reordered since the pass chain no longer factors to 1.
    if (plan.base_n != 0) return plan;
    return reorder_odd_radices_to_extremes<T>(plan);
}

// Measured plan overrides: sizes where the DP plan order is suboptimal.
// Empty (count==0) => no override, use build_dif_factor_plan.
//
// Why these can't live in the DP cost model: every entry is a re-ordering and/or
// re-partition of the SAME prime content the DP factors — the DP picks reasonable
// factors but the wrong pass ORDER, and the optimal order is NOT a function of
// (radix, ido). The proof it's genuinely cross-stage: 5040 wants radix-7 LAST
// (ido=1) while 1260/2520 want radix-5 FIRST — no monotone ido rule yields both.
// So these stay an explicit table rather than a DP term.
//
// Maintenance rule: re-audit on any dif_stage_cost change and DELETE any entry
// that merely ties its DP plan.
template<typename T, bool Forward>
[[nodiscard]] constexpr dif_factor_plan measured_dif_factor_plan(std::size_t N) {
    dif_factor_plan plan;
    constexpr std::size_t W = xsimd::batch<T>::size;
    if constexpr (sizeof(T) == 8) {
        if (N == 1260) { for (unsigned r : {5u, 3u, 3u, 7u, 4u})     plan.push(r); return plan; }
        // f64 W=8: same multiset as the DP [15,8], pure ORDER effect. The
        // additive model prices r15 first (vec + r8.last) under r8 first
        // (vec + r15.last), but at N=120 the r15 vec pass has exactly ONE
        // vector column (ido = W = 8) and its calibrated vec cost doesn't
        // amortize there. Encoding that as an ido==W surcharge re-ranks large-N
        // r15 chains, so it stays an explicit override.
        if (N == 120 && W == 8) { for (unsigned r : {8u, 15u}) plan.push(r); return plan; }
        // f64 W=8 (AVX-512): r16 at f64 W=8 holds 2*16 = 32 live batches —
        // exactly the ZMM file — so r16 passes spill while r8 stays
        // register-clean; but raising r16.vec to encode this re-ranks other
        // validated chains, so this stays an explicit override.
        if (N == 4096 && W == 8) { for (unsigned r : {8u, 8u, 8u, 8u}) plan.push(r); return plan; }
        // f64 W=8: [16,16]+codelet-64 terminal. Stays an explicit override
        // rather than a dif_terminal_cost cell: one size doesn't license the
        // DP to elect terminals everywhere.
        if (N == 16384 && W == 8) {
            for (unsigned r : {16u, 16u}) plan.push(r);
            plan.base_n = 64;
            return plan;
        }
        // 16384 at W<=4 uses the 8-8 tail while 32768 uses mostly-4s —
        // opposite r8 preference at two adjacent sizes, provably outside any
        // constant (r8.vec, r8.last) fit.
        if (N == 16384 && W <= 4) { for (unsigned r : {4u, 4u, 4u, 4u, 8u, 8u}) plan.push(r); return plan; }
    }
    if constexpr (sizeof(T) == 4) {
        // Common failure mode of the DP: it appends a cheap radix-2/valley tail
        // or buries an expensive odd radix mid-chain — cross-pass placement the
        // additive model can't rank.
        // W-gated: at wide-radix ISAs 2520 ties the DP chain — ties die; W<=8
        // keeps the explicit override.
        if (N == 2520 && !dif_wide_radices) { for (unsigned r : {5u, 3u, 3u, 7u, 8u}) plan.push(r); return plan; }
        if (N == 5292) { for (unsigned r : {7u, 3u, 3u, 3u, 4u, 7u}) plan.push(r); return plan; }
        if (N == 7560 && W >= 16) { for (unsigned r : {5u, 7u, 3u, 3u, 3u, 8u}) plan.push(r); return plan; }
        // W-gated: at wide-radix ISAs the DP elects wider chains; at W<=8
        // (no 16/32 candidates) these override the narrow DP.
        if (N == 2048 && !dif_wide_radices) { for (unsigned r : {8u, 8u, 4u, 8u}) plan.push(r); return plan; }
        if (N == 4096 && !dif_wide_radices) { for (unsigned r : {8u, 8u, 8u, 8u}) plan.push(r); return plan; }
        // W=4 (SSE): 16384 uses all-4 while 65536 uses 8-8 tails — the
        // sandwiched all-4 preference at the 128KB-side size is outside any
        // constant r8 fit (same pattern as the f64 16384 note above).
        if (N == 16384 && W == 4) { for (unsigned r : {4u, 4u, 4u, 4u, 4u, 4u, 4u}) plan.push(r); return plan; }
        if (N == 65536 && W == 4) { for (unsigned r : {4u, 4u, 4u, 4u, 4u, 8u, 8u}) plan.push(r); return plan; }
        // W<=8: the enumerator promotes one r8 ahead of the ido>=512 footprint
        // gate, but the late 8-8-8 tail is structurally faster at this size.
        // Lowering the gate to fix this generically re-ranks other chains, so
        // this stays an explicit override.
        if (N == 32768 && W <= 8) { for (unsigned r : {4u, 4u, 4u, 8u, 8u, 8u}) plan.push(r); return plan; }
    }
    return plan;
}

// Precomputed twiddle tables for the iterative DIF driver.
// One entry per pass (one per factor in the factorization chain).
// For pass with radix ip and l1 groups:
//   twiddles[pass][(j-1)*ido + i] = W_N^{j * l1 * i}, for j in [1,ip), i in [0,ido).
template<typename T>
struct dif_twiddle_set {
    // Each pass: pair of (re, im) flat vectors, length (ip-1)*ido.
    // On a row-form set (fuse_packed=true), f2head/f2tail passes have EMPTY
    // plain tables — their twiddles live only in packed_pair[head].
    std::vector<std::pair<std::vector<T>, std::vector<T>>> passes;
    // Radix for each pass (in factorization order).
    std::vector<unsigned> radices;
    // Fusion schedule the row driver executes (all-plain on a col-form set).
    std::vector<dif_fuse> sched;
    // Non-zero when a base codelet terminal is active.  The pass chain in
    // radices[] reduces N to base_n; codelet_dispatch handles the rest.
    // Zero means the chain factors to n=1 (default, no terminal).
    // Twiddle tables are built only for the factored passes (radices[]) —
    // never for the terminal base itself, which carries its own baked twiddles.
    unsigned base_n = 0;
    // Packed twiddle stream for dif_pass_fused2; non-empty iff
    // sched[p] == f2head. Both fused layers in ONE buffer, stored in exact
    // consumption order so the kernel walks a single advancing pointer:
    //   tw1 section: for each vector chunk ac in [0, ido/W):
    //     for k=1..IP1-1: W values tw1re[(k-1)*ido+ac*W..], W values tw1im[...]
    //   tw2 section: for each chunk ac2 in [0, ido2/W):
    //     for k2=1..IP2-1: W values tw2re[(k2-1)*ido2+ac2*W..], W values tw2im[...]
    // Kernel access: ptw_cur = ptw + (a0+j2*ido2)*2*(IP1-1)   [T-unit offset]
    //   k=m re/im at compile-time offsets (m-1)*2*W / (m-1)*2*W+W;
    //   advance ptw_cur += 2*(IP1-1)*W per t-step.
    // ONE base register, zero per-k address arithmetic.
    std::vector<std::vector<T>> packed_pair;
};

// fuse_packed selects the representation for fused2-eligible pass pairs:
//   true  (row form): packed_pair[head] built, plain tables of the pair dropped.
//           For plans executed by iterative_dif_execute_ws (1-D, Bluestein, Rader).
//   false (col form): plain tables for every pass, empty sched/packed_pair.
//           For plans executed by col_dif_execute_ws (N-D strided axes), which
//           reads passes[p] for every pass and never fuses.
template<typename T, bool Forward>
[[nodiscard]] dif_twiddle_set<T> build_dif_twiddle_set(std::size_t N,
                                         const dif_factor_plan* override_plan = nullptr,
                                         bool fuse_packed = true) {
    dif_twiddle_set<T> s;
    const dif_factor_plan measured = override_plan ? dif_factor_plan{}
                                                   : measured_dif_factor_plan<T, Forward>(N);
    const dif_factor_plan planned = override_plan ? *override_plan
                                  : measured.count != 0 ? measured
                                                        : build_dif_factor_plan<T>(N);

    // Propagate terminal size.  When base_n != 0 the twiddle loop below
    // naturally stops at planned.count (the passes BEFORE the terminal) so
    // no twiddle table is ever built for the terminal base sub-problem.
    s.base_n = planned.base_n;

    std::size_t l1 = 1;
    for (std::size_t pass = 0; pass < planned.count; ++pass) {
        const unsigned ip = planned[pass];
        const std::size_t ido = N / (l1 * static_cast<std::size_t>(ip));

        s.radices.push_back(ip);

        // Build twiddles: for j in [1,ip), i in [0,ido):
        //   w = W_N^{j * l1 * i} = exp(sign * 2*pi*i * (j*l1*i) / N)
        const std::size_t sz = static_cast<std::size_t>(ip - 1) * ido;
        std::vector<T> re(sz), im(sz);
        for (unsigned j = 1; j < ip; ++j) {
            for (std::size_t i = 0; i < ido; ++i) {
                const auto [sn, cs] = portable_trig::sincos_turns<Forward>(j * l1 * i, N);
                re[(j - 1) * ido + i] = static_cast<T>(cs);
                im[(j - 1) * ido + i] = static_cast<T>(sn);
            }
        }
        s.passes.emplace_back(std::move(re), std::move(im));

        l1 *= static_cast<std::size_t>(ip);
    }

    // Row form: repack each fused2 pair's twiddles into one consumption-order
    // stream (layout documented at dif_twiddle_set::packed_pair) and drop the
    // pair's plain tables — exactly one representation is ever stored, so the
    // twiddle footprint is unchanged vs an unfused set. fused3 passes keep
    // plain tables (dif_pass_fused3 reads the three layers separately).
    s.packed_pair.resize(s.passes.size());  // all empty by default
    s.sched = fuse_packed ? dif_fusion_schedule<T>(N, s.radices)
                          : std::vector<dif_fuse>(s.radices.size(), dif_fuse::plain);
    {
        constexpr std::size_t W = xsimd::batch<T>::size;
        std::size_t lp = 1;
        for (std::size_t p = 0; p < s.radices.size(); ++p) {
            const unsigned ip1 = s.radices[p];
            const std::size_t ido_p = N / (lp * static_cast<std::size_t>(ip1));
            if (s.sched[p] == dif_fuse::f2head) {
                const unsigned ip2 = s.radices[p + 1u];
                const std::size_t ido2_p = ido_p / static_cast<std::size_t>(ip2);
                // Same total size as the two plain tables:
                //   tw1: (ip1-1)*2*ido_p   tw2: (ip2-1)*2*ido2_p
                const std::size_t sz1 = static_cast<std::size_t>(ip1 - 1u) * 2u * ido_p;
                const std::size_t sz2 = static_cast<std::size_t>(ip2 - 1u) * 2u * ido2_p;
                std::vector<T> packed(sz1 + sz2);
                const auto& tw1 = s.passes[p];
                const auto& tw2 = s.passes[p + 1u];
                std::size_t off = 0;
                // tw1 section, interleaved by vector-chunk (consumption order).
                // Invariant: ido_p % W == 0 (guaranteed by the schedule's gate).
                const std::size_t n_chunks1 = ido_p / W;
                for (std::size_t ac = 0; ac < n_chunks1; ++ac) {
                    for (unsigned k = 1u; k < ip1; ++k) {
                        const std::size_t base = static_cast<std::size_t>(k - 1u) * ido_p + ac * W;
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw1.first[base + i];
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw1.second[base + i];
                    }
                }
                // tw2 section, interleaved by vector-chunk.
                // Invariant: ido2_p % W == 0 (schedule gate ensures ido_p % (ip2*W) == 0).
                const std::size_t n_chunks2 = ido2_p / W;
                for (std::size_t ac = 0; ac < n_chunks2; ++ac) {
                    for (unsigned k2 = 1u; k2 < ip2; ++k2) {
                        const std::size_t base2 = static_cast<std::size_t>(k2 - 1u) * ido2_p + ac * W;
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw2.first[base2 + i];
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw2.second[base2 + i];
                    }
                }
                s.packed_pair[p] = std::move(packed);
                // Drop the plain copies — the row driver never reads them.
                s.passes[p] = {};
                s.passes[p + 1u] = {};
            }
            lp *= static_cast<std::size_t>(ip1);
        }
    }

    return s;
}

} // namespace detail
} // namespace admiral

