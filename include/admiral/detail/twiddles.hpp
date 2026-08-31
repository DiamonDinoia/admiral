#pragma once

// Inter-stage twiddle tables for the iterative DIF driver: built once per plan, so
// hot loops do indexed reads and no runtime trig. Exact integer turn reduction.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <admiral/errors.hpp>  // `unsupported_error`

#include <poet/poet.hpp>
#include "simd.hpp"

#include "cache.hpp"      // `cpu_cache` (pass-0 twiddle residency gate)
#include "ct_math.hpp"    // `smallest_radix`
#include "cxx_compat.hpp"  // span, detail::bit_ceil, detail::has_single_bit, detail::const_find
#include "math.hpp"
#include "twiddle_row.hpp"    // `geom_twiddle_row` (W-wide twiddle row from an exact seed)
#include "portable_trig.hpp"  // `sincos_turns`

namespace admiral {
namespace detail {

struct dif_factor_plan {
    static constexpr std::size_t max_passes = 32;

    std::array<std::size_t, max_passes> radices{};
    std::size_t count = 0;

    constexpr dif_factor_plan() = default;
    // Literal chains (tests, --factors, the chain sweep); count is the list length.
    constexpr dif_factor_plan(std::initializer_list<std::size_t> rs) {
        for (const std::size_t r : rs) push(r);
    }


    constexpr void push(std::size_t radix) {
        if (count < max_passes) {
            radices[count++] = radix;
        }
    }

    [[nodiscard]] constexpr std::size_t operator[](std::size_t index) const {
        return radices[index];
    }
};

// Radix set for iterative DIF passes; only radices the factorizer emits get compiled.
// Drives poet::dispatch in the drivers; `dif_candidate_radices` is the DP's iterable form.
inline constexpr bool dif_wide_radices = poet::vector_register_count() >= 32;
// Wide-only: the 16/32 and merged 9/15/25/10 live sets starve a 16-reg register file.
// r25 keeps 5-power sizes valley-free. r20 stays out: PFA(5,4) holds 40 live batches
// on 32 zmm, so it spills. Readmit r20 only with a spill-free kernel, never with a
// cost row.
using dif_radix_set = std::conditional_t<
    dif_wide_radices,
    std::integer_sequence<std::size_t, 2, 3, 4, 5, 7, 8, 11, 16, 32, 9, 15, 25, 10>,
    std::integer_sequence<std::size_t, 2, 3, 4, 5, 7, 8, 11>>;

template<std::size_t... Rs>
constexpr auto radix_seq_to_array(std::integer_sequence<std::size_t, Rs...>) {
    return std::array{Rs...};
}
// DP factorizer candidates: same set as `dif_radix_set`. `dif_last_batch` narrows the
// last pass to `bit_ceil`(IP) lanes, so 2*IP <= W holds for f32 IP=2 alone.
inline constexpr auto dif_candidate_radices = radix_seq_to_array(dif_radix_set{});

// Runtime primes above the static radix set, admitted as MIDDLE passes only
// (`dif_pass_prime_chip`); the leaf is the batched-Rader chiplet. Admission pool and
// tape dispatch set in one sequence.
using dif_generic_radix_seq =
    std::integer_sequence<std::size_t, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
                          61, 67, 71, 73, 79, 83, 89, 97>;
inline constexpr auto dif_generic_radices = radix_seq_to_array(dif_generic_radix_seq{});
[[nodiscard]] constexpr bool dif_is_generic_radix(std::size_t r) {
    return detail::const_find(dif_generic_radices.begin(), dif_generic_radices.end(), r)
        != dif_generic_radices.end();
}

// In-place passes accept the full radix set; `butterfly_wants_reload` radices take the
// kSplitCols path, so nothing here is unreachable.
using dif_ip_radix_set = dif_radix_set;

// Per-pass cost models (`dif_stage_cost` dispatches on the key): the analytic surface
// for AVX-512, else this measured cyc/elem tape per (radix, regime) per (precision, W).
// Regimes: vec (ido >= W), valley (1 < ido < W), last (ido == 1); a missing valley
// cell falls back to vec.

struct dif_cost_row { double vec, valley, last; };

// Analytical fallback for keys with no measured table (SVE, RVV, AVX10/256 and other
// unmeasured ISAs):
//   traffic  4*sizeof(T) B/elem at the fitted L2 bandwidth, +5.6% per read stream past 4
//   compute  flops/elem / (0.77 * 4W flops/cyc)  [2 FMA ports]
//   spill    1.5 cyc per (peak-live - registers) batch / W; peak-live is flat `pow2`
//            2r+6, recursive split-scope ~22-26, odd r+9
//   valley   1.25x vec (lane fill, not a scalar fallback); last ~= one traffic
//            trip plus half the spill
inline constexpr double kL2BwBytesPerCyc = 12.6;

template<typename T>
[[nodiscard]] constexpr dif_cost_row dif_analytical_cost(std::size_t radix) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr std::size_t regs = poet::vector_register_count();
    constexpr double bytes_pe = 4.0 * sizeof(T);
    const bool pow2 = detail::has_single_bit(radix);
    const double flops = radix == 2 ? 5 : radix == 3 ? 8 : radix == 4 ? 8.5
                       : radix == 5 ? 12 : radix == 7 ? 16 : radix == 8 ? 12.5
                       : radix == 9 ? 19 : radix == 11 ? 22 : radix == 15 ? 34
                       : radix == 16 ? 17 : radix == 25 ? 24 : 21.5;
    const double r = static_cast<double>(radix);
    const double liv = !pow2 ? r + 9.0
                     : radix <= 8 ? 2.0 * r + 6.0
                     : radix == 16 ? 24.0 : 26.0;
    const double bw = bytes_pe / kL2BwBytesPerCyc;
    const double traffic = bw * (1.0 + 0.056 * (radix > 4 ? r - 4.0 : 0.0));
    const double comp = flops / (0.77 * 4.0 * static_cast<double>(W));
    const double spill = 1.5 * (liv > static_cast<double>(regs) ? liv - static_cast<double>(regs) : 0.0)
                       / static_cast<double>(W);
    const double vec = (comp > traffic ? comp : traffic) + spill;
    // `dif_pass_small_ido` gives every radix exact lane fill, so the valley costs one
    // factor for all. `kValleyPenalty` excludes ido < 4 outright.
    const double vall = vec * 1.25;
    const double lastc = (comp > bw ? comp : bw) + 0.5 * spill;
    return {vec, vall, lastc};
}

// Column order of every `dif_cost_row` table; the 16-reg set is a prefix, so the first
// seven columns match both ISAs. r25 is NOT measured: priced above the two r5 passes it
// replaces so the DP never prefers it. Never price r25 from its isolated measurement:
// that is the isolated-vs-in-chain trap.
inline constexpr std::array<std::size_t, 12> dif_cost_radices{2, 3, 4, 5, 7, 8, 11, 16, 32, 9, 15, 25};

[[nodiscard]] constexpr std::size_t dif_cost_index(std::size_t radix) {
    const auto* const first = dif_cost_radices.begin();
    return static_cast<std::size_t>(const_find(first, dif_cost_radices.end(), radix) - first);
}

// Every candidate is measured or priced from its coprime factors (`dif_measured_cost`),
// so an out-of-array candidate is priced, not an error.
static_assert(dif_cost_index(25) < dif_cost_radices.size(),
              "dif_cost_radices lost a measured radix");

// Measured DIF cost per (sizeof(T), W, vector registers). Class-scope static for one
// .rodata copy: a function-local constexpr array cannot be static before C++23.
// An unmeasured key has no t, hence 99.0 rows where 9/15/25 are absent at 16 regs.
template<std::size_t Bytes, std::size_t W, std::size_t Regs>
struct dif_cost_table {};   // unmeasured key -> analytical model

template<> struct dif_cost_table<8, 4, 16> {   // f64 AVX2 (no valley probe: vec fallback)
    static constexpr dif_cost_row t[12] = {
        {2.23, 2.23, 2.28}, {2.10, 2.10, 2.08}, {2.62, 2.62, 2.39},
        {2.53, 2.53, 2.38}, {3.50, 3.50, 3.04}, {6.10, 6.10, 2.70},
        {4.19, 4.19, 4.26}, {6.01, 6.01, 4.66}, {7.00, 7.00, 5.84},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};
template<> struct dif_cost_table<8, 2, 16> {   // f64 SSE (W=2: no valley exists)
    static constexpr dif_cost_row t[12] = {
        {2.76, 2.76, 2.43}, {2.18, 2.18, 2.16}, {2.84, 2.84, 2.34},
        {2.73, 2.73, 2.22}, {3.26, 3.26, 3.11}, {9.99, 9.99, 3.84},
        {4.83, 4.83, 4.51}, {10.2, 10.2, 7.77}, {11.2, 11.2, 8.70},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};
template<> struct dif_cost_table<4, 8, 16> {   // f32 AVX2
    static constexpr dif_cost_row t[12] = {
        {1.10, 1.79, 1.68}, {1.07, 3.61, 1.68}, {1.33, 3.96, 1.69},
        {1.17, 2.65, 1.27}, {1.52, 5.91, 1.50}, {3.00, 3.23, 1.51},
        {2.05, 9.22, 1.81}, {2.98, 20.1, 2.31}, {3.39, 22.5, 2.47},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};
template<> struct dif_cost_table<4, 4, 16> {   // f32 SSE (no valley probe: vec fallback)
    static constexpr dif_cost_row t[12] = {
        {1.42, 1.42, 1.26}, {1.20, 1.20, 1.18}, {1.55, 1.55, 1.06},
        {1.42, 1.42, 1.38}, {1.72, 1.72, 1.84}, {2.60, 2.60, 1.15},
        {2.38, 2.38, 2.46}, {5.07, 5.07, 3.79}, {5.78, 5.78, 3.86},
        {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}, {99.0, 99.0, 99.0}};
};

#if !ADM_CXX20
// Member detections for the C++17 arms of the two requires-expressions below.
template<typename, typename = void>
struct dif_cost_table_has_t : std::false_type {};
template<typename X>
struct dif_cost_table_has_t<X, std::void_t<decltype(X::t)>> : std::true_type {};
template<typename, typename = void>
struct dif_surface_has_c : std::false_type {};
template<typename X>
struct dif_surface_has_c<X, std::void_t<decltype(X::c)>> : std::true_type {};
#endif

// A merged coprime radix (10 = 5*2) has no measured column: price it as the SUM of the
// two passes it replaces; the merge saves only the ping-pong sweep between them.
template<typename T>
[[nodiscard]] constexpr dif_cost_row dif_measured_cost(std::size_t radix) {
    using table = dif_cost_table<sizeof(T), xsimd::batch<T>::size, poet::vector_register_count()>;
#if ADM_CXX20
    if constexpr (requires { table::t; }) {
#else
    if constexpr (dif_cost_table_has_t<table>::value) {
#endif
        if (dif_cost_index(radix) < dif_cost_radices.size()) return table::t[dif_cost_index(radix)];
        // Neither measured nor coprime-splittable (a bare prime): the analytical model
        // is all that is left. The guard also stops `coprime_split`(0) recursing forever.
        if (const auto s = coprime_split(radix); s.first != 0) {
            const dif_cost_row a = dif_measured_cost<T>(s.first), b = dif_measured_cost<T>(s.second);
            return {a.vec + b.vec, a.valley + b.valley, a.last + b.last};
        }
    }
    return dif_analytical_cost<T>(radix);   // SVE, RVV, AVX10/256 and other unmeasured ISAs
}

// Analytic pass cost per precision, keyed like `dif_cost_table`; unkeyed ISAs fall
// through to the tape model. Form (span = N for every pass of a chain):
//   ido == 1:  eps + lambda*A_r/span                      terminal, plus the level term
//   ido > 1:   p0 + p1*sqrt(A_r)                          plateau, L1-resident
//              + kappa/ido            for 1 < ido < W     per-block fixed cost / lanes
//              + m2*[48KB < B <= 2MB]                     L1 wall
//              + (m3 + m3a*sqrt(A_r))*[B > 2MB]           L2 exit + radix-scaled part
//              + H/span                                   fixed pass-cycles / span
//   B = (4*span + 2*(radix-1)*ido)*sizeof(T)  (data planes + twiddle streams)
//   A_r = full-pipeline static arith (fma+vmul+vadd+vsub); sqrt(A_r) reproduces
//         primes vs `pow2` vs merged with no per-radix constants.

template<std::size_t Bytes, std::size_t W, std::size_t Regs>
struct dif_surface {};   // unmeasured key -> the tape model stays authoritative

// The helpers below read arith[`dif_cost_index`(radix)] unchecked: `dif_stage_cost` must
// filter radix into `dif_cost_radices` first.

template<> struct dif_surface<8, 8, 32> {   // f64 AVX-512
    //        p0      p1      kappa   m2     m3     m3a     H      lambda  eps
    static constexpr std::array<double, 9> c{
        0.4132, 0.0185, 7.4518, 1.871, 2.246, 0.0365, 29.3, 0.0324, 0.818};
    static constexpr std::array<double, 12> arith{206, 568, 570, 875, 1465, 1512, 3131, 4610, 10375,
                                                  2246, 4698, 7654};
};
template<> struct dif_surface<4, 16, 32> {  // f32 AVX-512
    static constexpr std::array<double, 9> c{
        0.1582, 0.0087, 5.5862, 0.992, 0.441, 0.0272, 24.8, 0.0293, 0.440};
    static constexpr std::array<double, 12> arith{246, 602, 620, 1053, 1764, 1637, 3485, 4812, 10525,
                                                  2444, 5484, 8225};
};

// Surface alias for T on this build. A key is ANALYTIC when it carries fitted coefficients.
template<typename T>
using dif_surface_t = dif_surface<sizeof(T), xsimd::batch<T>::size, poet::vector_register_count()>;
#if ADM_CXX20
template<typename T>
inline constexpr bool dif_surface_is_analytic = requires { dif_surface_t<T>::c; };
#else
template<typename T>
inline constexpr bool dif_surface_is_analytic = dif_surface_has_c<dif_surface_t<T>>::value;
#endif

// 4 planes of span (SoA in/out re/im) + the (radix-1) twiddle streams of ido;
// ido == 1 carries no twiddles (the fit's convention).
template<typename T>
[[nodiscard]] constexpr std::size_t dif_pass_footprint_bytes(std::size_t span,
                                                             std::size_t radix,
                                                             std::size_t ido) {
    return (4u * span + (ido > 1u ? 2u * (radix - 1u) * ido : 0u)) * sizeof(T);
}

// L1-resident per-point cost of a full-width pass. p1*sqrt(A_r) puts r25 just above
// the two r5 passes it replaces; `dif_radix_admissible` prevents the r25 collapse.
template<typename T>
[[nodiscard]] constexpr double cyc_per_pt_l1_resident(std::size_t radix) {
    using S = dif_surface_t<T>;
    return S::c[0] + S::c[1] * std::sqrt(S::arith[dif_cost_index(radix)]);
}

// Small-ido underfill: a part-width pass pays its fixed per-block work over ido lanes.
// kappa is radix-free in the fit; it prices the per-block twiddle reload.
template<typename T>
[[nodiscard]] constexpr double valley_underfill_cyc(std::size_t ido) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (ido <= 1 || ido >= W) return 0.0;
    return dif_surface<sizeof(T), xsimd::batch<T>::size,
                       poet::vector_register_count()>::c[2]
           / static_cast<double>(ido);
}

// Cache breaks the surface was fitted at. These belong to the fit, not to the running
// machine: substituting `cpu_cache()` here reprices the surface against other silicon.
inline constexpr std::size_t kSurfaceL1dBytes = 48u * 1024u;
inline constexpr std::size_t kSurfaceL2Bytes  = 2u * 1024u * 1024u;

// Additive cpe steps at the cache breaks: radix-flat to L2, then a sqrt(A_r) part for
// in-table radices (merged and generic radices pay the flat charge only). The idx
// bound below is load-bearing.
template<typename T>
[[nodiscard]] constexpr double footprint_cache_level_cpe(std::size_t radix,
                                                         std::size_t bytes) {
    if (bytes <= kSurfaceL1dBytes) return 0.0;
    if (bytes <= kSurfaceL2Bytes) return dif_surface_t<T>::c[3];
    const std::size_t idx = dif_cost_index(radix);
    return dif_surface_t<T>::c[4]
         + (idx < dif_cost_radices.size()
                ? dif_surface_t<T>::c[5] * std::sqrt(dif_surface_t<T>::arith[idx])
                : 0.0);
}

// Fixed per-pass cycles (setup, pipeline fill) amortized over the span.
template<typename T>
[[nodiscard]] constexpr double amortized_launch_cpe(std::size_t span) {
    return dif_surface<sizeof(T), xsimd::batch<T>::size,
                       poet::vector_register_count()>::c[6]
           / static_cast<double>(span);
}

// Terminal (ido == 1) pass: sublinear in span. The level charge is NOT here.
// `dif_stage_cost` adds it to the terminal too, because `dif_pass_last` streams at big spans.
template<typename T>
[[nodiscard]] constexpr double terminal_pass_cpe(std::size_t radix, std::size_t span) {
    using S = dif_surface_t<T>;
    return S::c[8] + S::c[7] * S::arith[dif_cost_index(radix)] / static_cast<double>(span);
}


// 128 KiB residency boundary: largest side the pass still runs L2-resident (both routes).
inline constexpr std::size_t kResidentBytes = 128u * 1024u;

// Streaming-pass footprint multiplier: fitted anchors at 128KB/1MB/4MB, interpolated
// and clamped, active at ido >= 512. A higher radix opens more concurrent streams.
[[nodiscard]] constexpr double dif_footprint_mult(std::size_t radix, std::size_t ido,
                                                  std::size_t side_bytes) {
    if (ido < 512) return 1.0;
    const double m1 = radix >= 32 ? 2.05 : radix >= 16 ? 1.57 : 1.42;  // at 1MB
    const double m4 = radix >= 32 ? 2.33 : 1.55;                       // at 4MB
    constexpr double kRef = double(kResidentBytes), kMid = 1024.0 * 1024.0, kBig = 4.0 * kMid;
    const double b = static_cast<double>(side_bytes);
    if (b <= kRef) return 1.0;
    // Rational proxy t=(b-lo)/(hi-lo) ranks identically within each segment; avoids log2.
    if (b <= kMid) return 1.0 + (m1 - 1.0) * (b - kRef) / (kMid - kRef);
    if (b <= kBig) return m1 + (m4 - m1) * (b - kMid) / (kBig - kMid);
    return m4;
}

// Placement admissibility is structural, not calibration: the valley is a hard tier
// the model cannot outvote, the interior-kernel surcharge a finite multiplier it can.

// 1 < ido < W never fills the vector width and is never the cheaper placement: a
// hard veto tier, not a price. Repricing it as graded lane fill loses.
inline constexpr double kValleyPenalty = 1e9;

// Cost of an interior pass over the terminal one: a multiplier, not a tier, graded by
// register overflow (a `pow2` radix-r butterfly holds 2r live batches). No coprime-odd
// arm: it would fight the `pow2` arm for the terminal slot.
[[nodiscard]] constexpr double dif_interior_kernel_mult(std::size_t radix) {
    constexpr double regs = static_cast<double>(poet::vector_register_count());
    if (detail::has_single_bit(radix) && radix >= 4) {
        const double live = 2.0 * static_cast<double>(radix);
        return live > regs ? 1.0 + 0.5 * (live - regs) / regs : 1.0;
    }
    return 1.0;
}

// A multiplier the model can outvote; only the valley stays a hard tier.
template<typename T>
[[nodiscard]] constexpr double dif_placement_mult(std::size_t N, std::size_t ido,
                                                  std::size_t radix) {
    // Past kResidentBytes a pass streams a full memory sweep; halving the pass count
    // with a wide radix then beats the worse interior kernel it forces.
    if (ido == 1 || 2 * sizeof(T) * N > kResidentBytes) return 1.0;
    return dif_interior_kernel_mult(radix);
}

// N = 2^a * 5^b: chains with no competing odd radix, where the valley lane floor and
// r25 admission are safe.
[[nodiscard]] constexpr bool is_pentanomial(std::size_t N) {
    while (N % 2u == 0u) N /= 2u;
    while (N % 5u == 0u) N /= 5u;
    return N == 1u;
}

// Gate r25 to pentanomial N: elsewhere r25 (merging 5*5) displaces the r15 chain and
// forces the surviving 3s into a worse arrangement (450: 2-3-5-15 -> 9-2-25).
[[nodiscard]] constexpr bool dif_radix_admissible(std::size_t N, std::size_t radix) {
    return radix != 25 || is_pentanomial(N);
}

// Applies to the col driver too: pass count dominates there, and this veto merges
// 5*5 into r25. A radix >= kValleyWideRadix pays a double tier: its per-block twiddle
// reload scales with the radix while only ido lanes are live.
inline constexpr std::size_t kValleyWideRadix = 15;

template<typename T>
[[nodiscard]] constexpr double dif_valley_penalty(std::size_t ido, std::size_t radix) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (ido <= 1 || ido >= W) return 0.0;
    return radix >= kValleyWideRadix ? 2.0 * kValleyPenalty : kValleyPenalty;
}

// Above this radix a generic prime with a part-width tail loses to Bluestein. The
// route veto (`dif_chain_supported`) and the election price ask this one predicate and
// must not drift apart. Widening the bound vetoes valley passes that still beat
// Bluestein. `test_factor_plan.cpp` pins the band from both sides.
inline constexpr std::size_t kGenericStarvedTailMinRadix = 47;

template<typename T>
[[nodiscard]] constexpr bool dif_generic_tail_starved(std::size_t g, std::size_t ido) {
    return g >= kGenericStarvedTailMinRadix && ido < xsimd::batch<T>::size;
}

// Cost of one generic middle pass, cyc/elem: the batched-Rader chiplet's law:
//   `chiplet_cpe` ~ kConvCyc * L * sumf(L) / (p*W),  L = p-1, sumf = prime factors of L.
// A starved tail (`dif_generic_tail_starved`) is a structural veto, not a term here.
inline constexpr double kConvCyc = 2.891;

template<typename T>
[[nodiscard]] constexpr double generic_prime_cost(std::size_t g) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    std::size_t L = g - 1, sumf = 0, m = L;
    for (std::size_t d = 2; d * d <= m; ++d)
        while (m % d == 0) { sumf += d; m /= d; }
    if (m > 1) sumf += m;
    return kConvCyc * static_cast<double>(L * sumf)
           / (static_cast<double>(g) * static_cast<double>(W));
}
template<typename T>
[[nodiscard]] constexpr double dif_generic_stage_cost(std::size_t N, std::size_t n,
                                                      std::size_t g) {
    const std::size_t ido = n / g;
    // Veto shapes are priced out of election, one tier above the valley. Without the
    // tier, the DP elects chains the engine cannot run, and the route falls to
    // `bluestein`.
    if (dif_generic_tail_starved<T>(g, ido))
        return 2.0 * kValleyPenalty * static_cast<double>(N);
    // Lane-mask waste, both arms: ceil(ido/W)*W tile-lanes run for ido real columns.
    constexpr std::size_t W = xsimd::batch<T>::size;
    const double mask_waste = static_cast<double>((ido + W - 1u) / W * W)
                              / static_cast<double>(ido);
    if constexpr (dif_surface_is_analytic<T>) {
        // Same terms as `dif_stage_cost`; ido == 1 never happens (boundary passes are static).
        const std::size_t B = dif_pass_footprint_bytes<T>(N, g, ido);
        const double per_elem = generic_prime_cost<T>(g) * mask_waste
                                + footprint_cache_level_cpe<T>(g, B)
                                + amortized_launch_cpe<T>(N);
        return (per_elem + dif_valley_penalty<T>(ido, g)) * static_cast<double>(N);
    } else {
        // Unmeasured ISA: the earlier form, with the same lane-mask waste.
        return (generic_prime_cost<T>(g) * mask_waste
                  * dif_footprint_mult(g, ido, N * 2u * sizeof(T))
                + dif_valley_penalty<T>(ido, g))
               * static_cast<double>(N);
    }
}

// Tie-break weight far below real cost: an additive model cannot rank equal-cost
// permutations, but pass order matters on hardware. The `estimate` path needs it (no
// race absorbs ordering there). 32-reg ISAs nudge the first pass only.
[[nodiscard]] constexpr double dif_order_eps(std::size_t N, std::size_t n, std::size_t radix) {
    if (poet::vector_register_count() >= 32 && n != N) return 0.0;
    return 1e-9 * static_cast<double>(radix) * static_cast<double>(n);
}

// Tape-model stage cost from the measured per-ISA rows; serves ISAs without an
// analytic surface and radices with no surface entry.
template<typename T>
[[nodiscard]] constexpr double dif_stage_cost_tape(std::size_t N, std::size_t n,
                                                   std::size_t radix) {
    const std::size_t ido = n / radix;
    constexpr std::size_t W = xsimd::batch<T>::size;
    const dif_cost_row row = dif_measured_cost<T>(radix);
    // Some measured entries read valley < vec (a probe artifact); floor at vec*(W/ido)
    // capped at 2, pentanomial N only, so a narrow radix is not routed into the valley.
    const bool pentanomial = is_pentanomial(N);
    constexpr double kValleyLaneCap = 2.0;
    const double lane_mult = static_cast<double>(W) / static_cast<double>(ido);
    const double valley = (!pentanomial || row.valley >= row.vec)
                              ? row.valley
                              : row.vec * (lane_mult < kValleyLaneCap ? lane_mult : kValleyLaneCap);
    const double per_elem = (ido == 1) ? row.last
                          : (ido < W)  ? valley
                                       : row.vec * dif_footprint_mult(radix, ido, N * 2u * sizeof(T));
    // Lane-tail surcharge, fitted at 0.025: below the ordering weights it competes with.
    const std::size_t tail = (ido > 1) ? (ido % W) : ((N / n) % W);
    const double tail_mult = tail ? 1.0 + 0.025 * static_cast<double>(tail) / static_cast<double>(W)
                                  : 1.0;
    return (per_elem * tail_mult * dif_placement_mult<T>(N, ido, radix)
            + dif_valley_penalty<T>(ido, radix))
               * static_cast<double>(N) + dif_order_eps(N, n, radix);
}

// Per-stage cost cyc/elem * N (comparable across chains of the same N).
// n = remaining transform length entering this stage. Every return adds `dif_order_eps`.
template<typename T>
[[nodiscard]] constexpr double dif_stage_cost(std::size_t N, std::size_t n, std::size_t radix) {
    if constexpr (dif_surface_is_analytic<T>) {
        // An out-of-table coprime radix (10 = 5*2) is the SUM of its factor plateaus at
        // the real ido = n/radix. Never recurse `dif_stage_cost` on the factors: wrong
        // idos, double launch charge, the valley veto never fires. 15 is measured and
        // must NOT split into 5+3.
        if (dif_cost_index(radix) >= dif_cost_radices.size()) {
            const auto s = coprime_split(radix);
            if (s.first == 0)
                return dif_stage_cost_tape<T>(N, n, radix);  // no surface entry (bare prime)
            const std::size_t ido = n / radix;
            const std::size_t B = dif_pass_footprint_bytes<T>(N, radix, ido);
            // Coprime radices are never `pow2`, so the interior surcharge is 1.0 here.
            const double kernel =
                ido == 1 ? terminal_pass_cpe<T>(s.first, N) + terminal_pass_cpe<T>(s.second, N)
                         : (cyc_per_pt_l1_resident<T>(s.first)
                            + cyc_per_pt_l1_resident<T>(s.second)
                            + valley_underfill_cyc<T>(ido));
            const double per_elem = kernel + footprint_cache_level_cpe<T>(radix, B)
                                  + (ido == 1 ? 0.0 : amortized_launch_cpe<T>(N));
            return (per_elem + dif_valley_penalty<T>(ido, radix)) * static_cast<double>(N)
                 + dif_order_eps(N, n, radix);
        }
        const std::size_t ido = n / radix;
        const std::size_t B = dif_pass_footprint_bytes<T>(N, radix, ido);
        // Charge the interior surcharge only while kernel-bound (B <= L2): past L2 the
        // pass is memory-saturated and charging it mis-ranks measured winners.
        const double kernel =
            ido == 1 ? terminal_pass_cpe<T>(radix, N)
                     : (cyc_per_pt_l1_resident<T>(radix) + valley_underfill_cyc<T>(ido))
                           * (B <= kSurfaceL2Bytes ? dif_interior_kernel_mult(radix) : 1.0);
        const double per_elem =
            kernel + footprint_cache_level_cpe<T>(radix, B)
            + (ido == 1 ? 0.0 : amortized_launch_cpe<T>(N));
        // No lane-tail surcharge or lane floor here: both sit below the surface's fit error.
        return (per_elem + dif_valley_penalty<T>(ido, radix)) * static_cast<double>(N)
             + dif_order_eps(N, n, radix);
    } else {
        return dif_stage_cost_tape<T>(N, n, radix);
    }
}

// Modeled cycles of running chain p at N: the DP's own per-pass prices summed over
// the chain that executes (reordered, raced or `pow2`-enumerated).
template<typename T>
[[nodiscard]] inline double dif_chain_cost(std::size_t N, const dif_factor_plan& p) {
    double      total = 0.0;
    std::size_t n     = N;
    for (std::size_t i = 0; i < p.count; ++i) {
        total += dif_is_generic_radix(p[i]) ? dif_generic_stage_cost<T>(N, n, p[i])
                                           : dif_stage_cost<T>(N, n, p[i]);
        n /= p[i];
    }
    return total;
}

// Per-pass fusion role in the row driver's (`iterative_dif_execute_ws`) walk.
enum class dif_fuse : std::uint8_t { plain = 0, f2head, f2tail, f3head, f3tail };

// Radices `dif_pass_fused2` is instantiated for; dispatch set and schedule must agree or
// a fused pair reaches a kernel that cannot run it. A pair with 25 breaks WaMax >= W.
using dif_fused_pair_set = std::integer_sequence<std::size_t, 4, 5, 8>;

[[nodiscard]] constexpr bool dif_fusable_radix(std::size_t r) noexcept {
    return in_seq(dif_fused_pair_set{}, r);
}

// Minimum N for middle-pass fusion in the row driver: sweep count dominates a `pow2`
// chain only from here; below, tile staging dominates.
template<typename T>
inline constexpr std::size_t kDifFuseMinN =
    poet::vector_register_count() <= 16 ? (sizeof(T) == 8 ? 2048 : 4096) : 8192;

// Plan-time fusion schedule, one entry per pass; the row driver executes exactly this.
// `fused3` is f64-only, capped where tile staging costs more than the sweep it saves.
inline constexpr std::size_t kDifFused3MaxNF64 = 256u * 1024u / (2 * sizeof(double));

template<typename T>
constexpr void dif_fusion_schedule_into(std::size_t N, const std::size_t* radices,
                                        std::size_t n, dif_fuse* sched) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    for (std::size_t i = 0; i < n; ++i) sched[i] = dif_fuse::plain;
    if (N < kDifFuseMinN<T> || n < 3) return;
    const bool fused3_ok = sizeof(T) == 8 && N <= kDifFused3MaxNF64;
    std::size_t l1 = radices[0];  // first pass (p=0) never fuses
    for (std::size_t p = 1; p + 1 < n; ++p) {
        const std::size_t ip = radices[p];
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
            const std::size_t r2 = radices[p + 1];
            // ido % (r2*W) == 0 is the kernel's alignment contract: it gives
            // ido2 = ido/r2 a whole number of vector chunks.
            if (dif_fusable_radix(ip) && dif_fusable_radix(r2) && ido % (r2 * W) == 0u) {
                sched[p] = dif_fuse::f2head;
                sched[p + 1] = dif_fuse::f2tail;
                l1 *= ip * r2;
                p += 1;
                continue;
            }
        }
        l1 *= ip;
    }
}

template<typename T>
[[nodiscard]] inline std::vector<dif_fuse>
dif_fusion_schedule(std::size_t N, const std::vector<std::size_t>& radices) {
    std::vector<dif_fuse> sched(radices.size());
    dif_fusion_schedule_into<T>(N, radices.data(), radices.size(), sched.data());
    return sched;
}

// Fused-group discount: fused members share one L1-tiled sweep, so each costs a
// fraction of its isolated sweep. Past the cap the saved round-trips are L3 hits, not
// DRAM, so the discount is exactly 1.0 there. Never fit it from the `fuse_packed` on/off
// ratio: that ratio holds the chain fixed.
inline constexpr std::size_t kDifFuseDiscountMaxNF64 = 16384;
inline constexpr std::size_t kDifFuseDiscountMaxNF32 = 131072;
inline constexpr double kDifFuseDiscountFactor = 0.40;

template<typename T>
[[nodiscard]] constexpr double dif_fuse_discount(std::size_t N) {
    const std::size_t cap = sizeof(T) == 8 ? kDifFuseDiscountMaxNF64 : kDifFuseDiscountMaxNF32;
    return N <= cap ? kDifFuseDiscountFactor : 1.0;
}

// The DP's K cheapest chains, ascending; a plan-time race settles among them.
inline constexpr std::size_t kDifCandidates = 16;
// One slot per distinct multiset (see `offer`), so beam == candidates. The DP is
// O(divisors * radices * beam^2), so the beam width is a plan-time cost.
inline constexpr std::size_t kDifBeam = kDifCandidates;

// True iff two chains carry the same radix multiset. Chains that differ only in order
// tie on the additive cost the DP minimizes, so the multiset is the equivalence the
// model cannot see through.
[[nodiscard]] constexpr bool dif_same_multiset(const dif_factor_plan& a,
                                               const dif_factor_plan& b) {
    if (a.count != b.count) return false;
    auto x = a.radices, y = b.radices;
    std::sort(x.begin(), x.begin() + a.count);
    std::sort(y.begin(), y.begin() + b.count);
    return x == y;
}

// At most ONE chain per multiset: a plain top-K is dominated by reorderings of one
// factorization, so the race would sample orderings instead of factorizations. The
// slots buy factorization coverage; the race measures ordering itself.
struct dif_chain_list {
    std::array<dif_factor_plan, kDifCandidates> chain{};
    std::size_t count = 0;
    // Slots this list keeps; `estimate` consumes only the argmin, so it says 1.
    // A wider list multiplies DP time.
    std::size_t cap = kDifCandidates;

    // For an ascending producer (the DP): first offer of a multiset is its cheapest.
    constexpr void push(const dif_factor_plan& p) {
        if (count < cap && find(p) == count) chain[count++] = p;
    }
    // Cost-ordered insert, for a producer that enumerates in no particular order (the
    // `pow2` enumerator). The cheapest ordering of a multiset must win explicitly here.
    constexpr void offer(const dif_factor_plan& p, double c) {
        if (const std::size_t j = find(p); j < count) {
            if (cost[j] <= c) return;
            for (std::size_t m = j + 1; m < count; ++m) {
                chain[m - 1] = chain[m];
                cost[m - 1] = cost[m];
            }
            --count;
        }
        std::size_t k = count;
        while (k > 0 && c < cost[k - 1]) --k;
        if (k >= cap) return;
        for (std::size_t j = (std::min)(count, cap - 1); j > k; --j) {
            chain[j] = chain[j - 1];
            cost[j] = cost[j - 1];
        }
        chain[k] = p;
        cost[k] = c;
        count = (std::min)(count + 1, cap);
    }
    [[nodiscard]] constexpr const dif_factor_plan& operator[](std::size_t i) const {
        return chain[i];
    }

private:
    [[nodiscard]] constexpr std::size_t find(const dif_factor_plan& p) const {
        std::size_t j = 0;
        while (j < count && !dif_same_multiset(chain[j], p)) ++j;
        return j;
    }
    std::array<double, kDifCandidates> cost{};
};

// Whole-chain `pow2` selection in the fusion band (N >= kDifFuseMinN): the fusion gates
// are cross-pass, so the additive DP cannot see them. Scores every ordered composition
// of log2(N): cost = sum `dif_stage_cost`, fused members discounted by `dif_fuse_discount`.
// Gates come from `dif_fusion_schedule`.
template<typename T>
[[nodiscard]] constexpr dif_chain_list enumerate_pow2_dif_chains(std::size_t N,
                                                                std::size_t want = kDifCandidates) {
    std::size_t chain[dif_factor_plan::max_passes]{};
    dif_chain_list out;
    out.cap = (std::max)(std::size_t{1}, (std::min)(want, kDifCandidates));
    const double fuse_g = dif_fuse_discount<T>(N);
    // Score chain[0..len) with real fusion gates.
    const auto score = [&](std::size_t len) {
        dif_fuse sched[dif_factor_plan::max_passes]{};
        dif_fusion_schedule_into<T>(N, chain, len, sched);
        double cost = 0.0;
        std::size_t n = N;
        for (std::size_t i = 0; i < len; ++i) {
            const double pc = dif_stage_cost<T>(N, n, chain[i]);
            cost += (sched[i] == dif_fuse::plain) ? pc : fuse_g * pc;
            n /= chain[i];
        }
        dif_factor_plan p;
        for (std::size_t i = 0; i < len; ++i) p.push(chain[i]);
        out.offer(p, cost);
    };
    auto rec = [&](auto&& self, std::size_t rem, std::size_t len) -> void {
        if (rem == 1) {
            score(len);
            return;
        }
        // Wide-register ISAs enumerate 16/32 too (see `dif_wide_radices`).
        constexpr std::size_t cands[] = {4, 8, 16, 32};
        constexpr std::size_t n_cands = dif_wide_radices ? 4u : 2u;
        for (std::size_t ci = 0; ci < n_cands; ++ci) {
            const std::size_t r = cands[ci];
            if (rem % r == 0) {
                chain[len] = r;
                self(self, rem / r, len + 1);
            }
        }
    };
    rec(rec, N, 0);
    return out;
}
template<typename T>
[[nodiscard]] constexpr dif_factor_plan enumerate_pow2_dif_plan(std::size_t N) {
    return enumerate_pow2_dif_chains<T>(N, 1)[0];
}

// Can the engine RUN this exact chain for this size? The election, the route predicate
// and an explicitly supplied chain (--factors, the chain sweep) all ask this question.
// The predicate lives beside the DP.
template<typename T>
[[nodiscard]] inline bool dif_chain_shape_ok(std::size_t N, const dif_factor_plan& p) {
    if (p.count == 0) return false;
    std::size_t prefix = 1;  // product of radices before the current pass
    for (std::size_t i = 0; i < p.count; ++i) {
        const std::size_t r = p[i];
        if (!in_seq(dif_radix_set{}, r)) {
            // Generic passes exist as middle passes only: the first pass is AoS->SoA
            // and the last is SoA->AoS, both static-kernel shaped. The no-chain fallback
            // can emit them (bare 13 -> [13], 26 -> [2,13]); reject them here.
            if (!dif_is_generic_radix(r)) return false;
            if (i == 0 || i + 1 == p.count) return false;
            // Lane fill gates the veto; the half-empty-tile case is still excluded.
            if (dif_generic_tail_starved<T>(r, N / (prefix * r))) return false;
        }
        prefix *= r;
    }
    return prefix == N;
}

// Divisors of n, ascending. The radix-chain DP's state space is exactly this set,
// and d(n) <= 240 below 10^6.
[[nodiscard]] inline std::vector<std::size_t> ascending_divisors(std::size_t n) {
    std::vector<std::size_t> d{1};
    for (std::size_t p = 2; p * p <= n; ++p) {
        if (n % p != 0) continue;
        const std::size_t known = d.size();
        for (std::size_t q = p; n % p == 0; q *= p) {
            n /= p;
            for (std::size_t i = 0; i < known; ++i) d.push_back(d[i] * q);
        }
    }
    if (n > 1) {  // the one prime factor above sqrt, if any
        const std::size_t known = d.size();
        for (std::size_t i = 0; i < known; ++i) d.push_back(d[i] * n);
    }
    std::sort(d.begin(), d.end());
    return d;
}


// want candidates; the beam only has to be deeper than the race is wide. want = 1
// (`estimate`, and every twiddle build behind it) runs the 1-best DP.
template<typename T>
[[nodiscard]] inline dif_chain_list dif_chain_candidates(std::size_t N,
                                                         std::size_t want = kDifCandidates) {
    const std::size_t beam = want <= 1 ? 1 : kDifBeam;
    // `Constexpr`-capable but built at runtime: materializing the plan table `consteval`
    // would cost constant evaluation per TU for zero runtime gain.
    if (N >= kDifFuseMinN<T> && (N & (N - 1u)) == 0u) return enumerate_pow2_dif_chains<T>(N, want);

    // Finite sentinel: -ffast-math makes infinity UB.
    constexpr double kUnreachable = 1e300;
    struct entry {
        double cost = kUnreachable;
        std::size_t radix = 0;
        std::size_t next = 0;  // which of the successor state's beam entries this continues
        std::size_t key = 0;   // radix MULTISET filter, order-independent
    };

    // Summed, so every permutation of a chain keys the same. Mixed first because raw radices
    // sum to collisions. The sum is only a filter: every key hit is confirmed against the
    // actual multiset.
    const auto rmix = [](std::size_t r) {
        std::size_t x = r * 0x9E3779B97F4A7C15ull;
        x ^= x >> 31;
        return x * 0xBF58476D1CE4E5B9ull;
    };

    // The DP runs over the divisors of N, not [1, N]: n = 1 is the only base case and
    // every transition divides, so no other state is reachable.
    const std::vector<std::size_t> divisors = ascending_divisors(N);
    const auto state = [&divisors](std::size_t n) {
        const auto* const first = divisors.data();
        const auto* const at = std::lower_bound(first, first + divisors.size(), n);
        return static_cast<std::size_t>(at - first);
    };

    // divisors[0] == 1 (cost 0) stays the base case, not a stored entry: writing dp[0]
    // trips -Wnull-dereference under gcc-14 -Werror, which cannot prove this vector is
    // non-null. Rows size to beam, not kDifBeam: the want=1 DP is what `estimate` pays.
    std::vector<entry> dp(divisors.size() * beam);
    const auto row = [&dp, beam](std::size_t i) { return span<entry>(&dp[i * beam], beam); };

    // Ascending insert; exact ties prefer the larger radix, the 1-best DP's own tie
    // rule. The SAME predicate picks a multiset's representative below.
    const auto before = [](const entry& a, const entry& b) {
        return a.cost < b.cost || (a.cost == b.cost && a.radix > b.radix);
    };
    // The multiset an entry stands for, sorted: walk its back-pointers exactly as rebuild does.
    const auto sorted_chain = [&row, &state](std::size_t n, entry e) {
        std::array<std::size_t, dif_factor_plan::max_passes> rs{};
        std::size_t c = 0;
        // Capped like `dif_factor_plan::push`; a raw DP link chain is unbounded by construction.
        while (n > 1 && c < rs.size()) {
            rs[c++] = e.radix;
            n /= e.radix;
            if (n > 1) e = row(state(n))[e.next];
        }
        std::sort(rs.begin(), rs.begin() + c);
        return std::pair{rs, c};
    };
    const auto offer = [beam, &before, &sorted_chain](std::size_t n, span<entry> r,
                                                      const entry e) {
        // One slot per multiset, keeping its cheapest ordering: without this,
        // permutations of one factorization crowd out the rest. Skipped at beam == 1.
        for (std::size_t m = 0; beam > 1 && m < beam && r[m].cost < kUnreachable; ++m) {
            if (r[m].key != e.key || sorted_chain(n, r[m]) != sorted_chain(n, e)) continue;
            if (!before(e, r[m])) return;
            for (std::size_t j = m + 1; j < beam; ++j) r[j - 1] = r[j];
            r[beam - 1] = entry{};
            break;
        }
        std::size_t k = beam;
        while (k > 0 && before(e, r[k - 1])) --k;
        if (k >= beam) return;
        for (std::size_t j = beam - 1; j > k; --j) r[j] = r[j - 1];
        r[k] = e;
    };

    // Admissibility depends on N alone, so both filters hoist out of the DP loop.
    std::array<std::size_t, dif_candidate_radices.size()> radices{};
    std::size_t nrad = 0;
    for (const std::size_t r : dif_candidate_radices)
        if (N % r == 0 && dif_radix_admissible(N, r)) radices[nrad++] = r;
    std::array<std::size_t, dif_generic_radices.size()> generics{};
    std::size_t ngen = 0;
    for (const std::size_t g : dif_generic_radices)
        if (N % g == 0) generics[ngen++] = g;

    for (std::size_t i = 1; i < divisors.size(); ++i) {
        const std::size_t n = divisors[i];
        const auto expand = [&](const std::size_t radix, const double stage) {
            const std::size_t rest = n / radix;  // a smaller divisor: already final
            if (rest == 1) return offer(n, row(i), {stage, radix, 0, rmix(radix)});
            const span<const entry> sub = row(state(rest));
            // Rows are cost-ascending, so past the beam's worst entry every later
            // combination prunes. Strict >: offer's tie rule can take an equal cost.
            for (std::size_t j = 0; j < beam && sub[j].cost < kUnreachable; ++j) {
                if (stage + sub[j].cost > row(i)[beam - 1].cost) break;
                offer(n, row(i), {stage + sub[j].cost, radix, j, sub[j].key + rmix(radix)});
            }
        };
        for (std::size_t k = 0; k < nrad; ++k)
            if (n % radices[k] == 0) expand(radices[k], dif_stage_cost<T>(N, n, radices[k]));
        // Generic primes enter as middle passes only: never leading (n == N is the
        // static first pass) and never the sole pass (n / g == 1).
        if (n != N)
            for (std::size_t k = 0; k < ngen; ++k)
                if (n % generics[k] == 0 && n / generics[k] != 1)
                    expand(generics[k], dif_generic_stage_cost<T>(N, n, generics[k]));
    }

    const auto rebuild = [&](std::size_t k) {
        dif_factor_plan p;
        for (std::size_t n = N, idx = k; n > 1;) {
            const entry e = row(state(n))[idx];
            if (e.cost >= kUnreachable || e.radix == 0 || n % e.radix != 0)
                return dif_factor_plan{};
            p.push(e.radix);
            n /= e.radix;
            idx = e.next;
        }
        return p;
    };

    dif_chain_list out;
    out.cap = (std::max)(std::size_t{1}, (std::min)(want, kDifCandidates));
    for (std::size_t k = 0; k < beam && out.count < out.cap; ++k) {
        const dif_factor_plan p = rebuild(k);
        if (p.count == 0) break;  // the beam is contiguous: no entry k implies none after
        out.push(p);
    }
    // No DP chain (a prime outside `dif_radix_set`): return the trivial factorization,
    // which callers read as "not batchable" and fall back to scalar.
    if (out.count == 0) {
        dif_factor_plan p;
        for (std::size_t n = N; n > 1;) {
            const std::size_t fallback = smallest_radix(n);
            p.push(fallback);
            n /= fallback;
        }
        out.push(p);
    }
    return out;
}


// The chain that executes: the cheapest candidate the engine can run. The shape vetoes
// are chain properties, not N properties, so the next candidate rescues a vetoed
// argmin. Route availability and twiddle construction go through this one call.
template<typename T>
[[nodiscard]] inline dif_factor_plan dif_elected_chain(std::size_t N) {
    // The argmin runs at all but a handful of sizes. Asking for one candidate runs the
    // 1-best DP. Pay for the wide beam only where the vetoes apply.
    const dif_factor_plan best = dif_chain_candidates<T>(N, 1)[0];
    if (dif_chain_shape_ok<T>(N, best)) return best;
    const dif_chain_list c = dif_chain_candidates<T>(N);
    for (std::size_t i = 0; i < c.count; ++i)
        if (dif_chain_shape_ok<T>(N, c[i])) return c[i];
    // Nothing runnable: return the argmin, the trivial "not batchable" factorization.
    // Callers that must not build it ask `dif_chain_shape_ok` first.
    return best;
}

template<typename T>
[[nodiscard]] inline dif_factor_plan build_dif_factor_plan(std::size_t N) {
    return dif_chain_candidates<T>(N, 1)[0];
}

// Precomputed twiddle tables for the iterative DIF driver, one entry per pass:
//   twiddles[pass][(j-1)*ido + i] = W_N^{j * l1 * i}, j in [1,ip), i in [0,ido).
// One `dif_step` is one pass call, resolved at plan time. src/dst/sim/dim select the
// re/im plane (0 = cc0, 1 = cc1; 2 = re+W, W-blocked). es bits 0/1: element stride 2;
// bit 2: last pass reads `rowperm`. Boundary steps read rt.in/rt.out.
template<typename T>
struct dif_rt;

template<typename T>
struct dif_step {
    using fn_t = void (*)(const T*, const T*, T*, T*, const dif_step&, const dif_rt<T>&);
    fn_t fn = nullptr;
    std::size_t p = 0;    // pass index (twiddle/aux tables)
    std::size_t l1 = 0;   // outer (done) prefix product at this pass
    std::size_t ido = 0;  // inner dimension at this pass
    std::size_t n = 0;    // generic radix / terminal `n_groups` / single N
    std::uint8_t src = 0, dst = 0, sim = 0, dim = 0, es = 0;
};

// blk bakes the W-blocked SoA election (executes when `soa_stride` >= N);
// flat bakes element stride 1 everywhere (four independent planes).
template<typename T>
struct dif_tape_pair {
    std::vector<dif_step<T>> blk, flat;
};

template<typename T>
struct dif_twiddle_set {
    // One (re, im) pair per pass, length (ip-1)*ido. On a row-form set the fused
    // pair's plain tables stay EMPTY; their twiddles live in `packed_pair`[head].
    std::vector<std::pair<std::vector<T>, std::vector<T>>> passes;
    // Radix for each pass (in factorization order).
    std::vector<std::size_t> radices;
    // Fusion schedule the row driver executes (all-plain on a col-form set).
    std::vector<dif_fuse> sched;
    // Packed twiddle stream for `dif_pass_fused2`; non-empty iff sched[p]==f2head. Both
    // fused layers in one buffer, consumption order: per vector chunk, W re then W im
    // values for each k in [1, ip); the kernel reads them at compile-time offsets.
    std::vector<std::vector<T>> packed_pair;
    // Bit p set = pass p runs in place; 0 = all-Stockham. Reads and twiddles are
    // store-map-invariant, so only the block order differs; `rowperm` un-permutes the
    // last pass's gather base, empty iff `ip_mask` == 0.
    std::uint32_t ip_mask = 0;
    std::vector<std::uint32_t> rowperm;
    // Non-zero: passes[0] holds the FACTORED pass-0 row (see the p0 block below), block
    // width B, layout [C: (ip-1)*B][A: (ip-1)*nb], w[j][a1*B+a0] = A[j][a1]*C[j][a0].
    // Zero: passes[0] is the flat (ip-1)*ido row. Row form only.
    std::size_t p0_block = 0;
    // Execute tape for the row driver, [0] = forward, [1] = inverse. Holds no
    // pointers into this set's storage (thunks resolve tables by index), so copies of
    // the owning plan stay valid. Empty on col-form sets (`fuse_packed` = false).
    dif_tape_pair<T> tape[2];
};

// Defined in `dif_driver.hpp` next to the kernels the tape resolves. Instantiated
// once per (T, Forward) leaf in `src/inst_dif_*`, so the tape's thunks ride the same
// instantiation boundary as the driver.
template<typename T, bool Forward>
void dif_build_tape(dif_twiddle_set<T>& s, std::size_t N);

// `fuse_packed` selects the twiddle representation for `fused2`-eligible pairs:
//   true  (row form): `packed_pair`[head] built, the pair's plain tables dropped.
//           `iterative_dif_execute_ws` (1-D, Bluestein, Rader).
//   false (col form): plain tables every pass, empty sched/`packed_pair`.
//           `col_dif_execute_ws` (N-D strided axes); never fuses.
// The tables are always the forward ones: a complex multiply by w on the inverse's
// swapped planes (`plane_refs` in `simd_swizzle.hpp`) is multiplication by conj(w).
template<typename T>
[[nodiscard]] dif_twiddle_set<T> build_dif_twiddle_set(std::size_t N,
                                         const dif_factor_plan* override_plan = nullptr,
                                         bool fuse_packed = true) {
    dif_twiddle_set<T> s;
    // `dif_elected_chain`, not the DP's bare argmin: route availability elected through the
    // same call, so the tables are always built for the chain the router said would run.
    const dif_factor_plan planned = override_plan ? *override_plan : dif_elected_chain<T>(N);
    // Same for a forced chain's radices: every pass dispatches over `dif_radix_set`, which is
    // register-parametric, so a radix outside it would SKIP the pass instead of running it.
    // Only a forced chain can be wrong here: the DP draws from the set.
    if (override_plan)
        for (std::size_t p = 0; p < planned.count; ++p) {
            if (!in_seq(dif_radix_set{}, planned[p]) && !dif_is_generic_radix(planned[p]))
                throw unsupported_error("forced dif: radix is not in dif_radix_set");
            // Generic passes only exist as middle passes (AoS/last shapes are static):
            // a boundary slot would throw from the dispatch at execute instead.
            if (dif_is_generic_radix(planned[p]) &&
                (p == 0 || p + 1 == planned.count))
                throw unsupported_error("forced dif: generic radix must be a middle pass");
        }

    std::size_t l1 = 1;
    for (std::size_t pass = 0; pass < planned.count; ++pass) {
        const std::size_t ip = planned[pass];
        const std::size_t ido = N / (l1 * ip);

        s.radices.push_back(ip);

        // Each j is one geometric sequence in i, so a SIMD prefix product of complex
        // multiplies builds the row (`twiddle_row.hpp`), no per-entry sincos.
        const std::size_t sz = (ip - 1) * ido;
        std::vector<T> re(sz), im(sz);
        for (std::size_t j = 1; j < ip; ++j)
            geom_twiddle_row<T, /*Forward=*/true>(j * l1, N, ido,
                                                  re.data() + (j - 1) * ido,
                                                  im.data() + (j - 1) * ido);
        s.passes.emplace_back(std::move(re), std::move(im));

        l1 *= ip;
    }

    // Pass 0 runs at l1 == 1, so its row streams once with zero reuse. Factor it: with
    // a = a1*B + a0,
    //   W_N^{j*a} = W_N^{j*B*a1} * W_N^{j*a0} = A[j][a1] * C[j][a0],
    // and (ip-1)*(B+nb) entries with B ~ sqrt(ido) replace (ip-1)*ido. B is a
    // power-of-two multiple of W, so no W-batch straddles a block. Row form only:
    // `col_dif_execute_ws` reads passes[0] flat.
    if (fuse_packed && !s.radices.empty()) {
        constexpr std::size_t W = xsimd::batch<T>::size;
        const std::size_t ip0 = s.radices[0], ido0 = N / ip0;
        const std::size_t flat = 2 * (ip0 - 1) * ido0 * sizeof(T);
        // Residency is decided against L2, not L1: while in, out and scratch fit L2,
        // the flat row stays resident and factoring only adds a multiply per element.
        if (8 * N * sizeof(T) + flat > cpu_cache().l2) {
            std::size_t bw = W;
            while (bw * bw < ido0) bw *= 2;
            const std::size_t nb = (ido0 + bw - 1) / bw;
            std::vector<T> re((ip0 - 1) * (bw + nb)), im(re.size());
            for (std::size_t j = 1; j < ip0; ++j) {
                geom_twiddle_row<T, /*Forward=*/true>(j, N, bw,
                                                      re.data() + (j - 1) * bw,
                                                      im.data() + (j - 1) * bw);
                geom_twiddle_row<T, /*Forward=*/true>(j * bw, N, nb,
                                                      re.data() + (ip0 - 1) * bw + (j - 1) * nb,
                                                      im.data() + (ip0 - 1) * bw + (j - 1) * nb);
            }
            s.passes[0] = {std::move(re), std::move(im)};
            s.p0_block = bw;
        }
    }

    // Row form: repack each `fused2` pair into one consumption-order stream and drop its
    // plain tables; the footprint is unchanged. `fused3` keeps plain tables.
    s.packed_pair.resize(s.passes.size());  // all empty by default
    s.sched = fuse_packed ? dif_fusion_schedule<T>(N, s.radices)
                          : std::vector<dif_fuse>(s.radices.size(), dif_fuse::plain);
    {
        constexpr std::size_t W = xsimd::batch<T>::size;
        std::size_t lp = 1;
        for (std::size_t p = 0; p < s.radices.size(); ++p) {
            const std::size_t ip1 = s.radices[p];
            const std::size_t ido_p = N / (lp * ip1);
            if (s.sched[p] == dif_fuse::f2head) {
                const std::size_t ip2 = s.radices[p + 1u];
                const std::size_t ido2_p = ido_p / ip2;
                // Same total size as the two plain tables:
                //   tw1: (ip1-1)*2*`ido_p`   tw2: (ip2-1)*2*`ido2_p`
                const std::size_t sz1 = (ip1 - 1u) * 2u * ido_p;
                const std::size_t sz2 = (ip2 - 1u) * 2u * ido2_p;
                std::vector<T> packed(sz1 + sz2);
                const auto& tw1 = s.passes[p];
                const auto& tw2 = s.passes[p + 1u];
                std::size_t off = 0;
                // tw1 section, interleaved by vector-chunk (consumption order).
                // Invariant: `ido_p` % W == 0 (schedule gate).
                const std::size_t n_chunks1 = ido_p / W;
                for (std::size_t ac = 0; ac < n_chunks1; ++ac) {
                    for (std::size_t k = 1; k < ip1; ++k) {
                        const std::size_t base = (k - 1) * ido_p + ac * W;
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw1.first[base + i];
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw1.second[base + i];
                    }
                }
                // tw2 section, interleaved by vector-chunk.
                // Invariant: `ido2_p` % W == 0 (schedule gate: `ido_p` % (ip2*W) == 0).
                const std::size_t n_chunks2 = ido2_p / W;
                for (std::size_t ac = 0; ac < n_chunks2; ++ac) {
                    for (std::size_t k2 = 1; k2 < ip2; ++k2) {
                        const std::size_t base2 = (k2 - 1) * ido2_p + ac * W;
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw2.first[base2 + i];
                        for (std::size_t i = 0; i < W; ++i) packed[off++] = tw2.second[base2 + i];
                    }
                }
                s.packed_pair[p] = std::move(packed);
                // Drop plain copies: the row driver never reads them.
                s.passes[p] = {};
                s.passes[p + 1u] = {};
            }
            lp *= ip1;
        }
    }

    // In-place election. An `ip_mask` pass stores at col + IP*b instead of b + l1*k: the
    // store lands on just-loaded lines (4N*T traffic, not 6N*T), but the digit appends
    // LOW and permutes the block word; `rowperm` undoes it in the last pass's gather
    // base only. Preconditions, each otherwise silent corruption: row form, and a
    // plain-scheduled pass (`fused2`/3 hard-code the Stockham store). Plain radices
    // elect early (the gather window grows with the prefix), split-column ones late.
    if (fuse_packed && s.radices.size() >= 3) {
        // The plain-group gate is the last pass's gather window L*`r_last`*2*sizeof(T),
        // not a pass count: a fixed count degrades on chains a count grid misses.
        constexpr std::size_t kGatherWindowBytes = 256u * 1024u;
        const std::size_t r_last = s.radices.back();
        std::size_t L = s.radices[0];
        std::size_t ip_prefix = 0;
        for (std::size_t p = 1; p + 1 < s.radices.size(); ++p) {
            if (s.sched[p] != dif_fuse::plain) break;
            // A `wants_reload` radix cannot go in place here: the odd sweep re-reads
            // columns the even sweep overwrote. The split-column kernel below can.
            if (dif_is_generic_radix(s.radices[p])) break;  // Stockham only: no in-place kernel
            if (butterfly_wants_reload(s.radices[p], poet::vector_register_count())) break;
            // No ido gate: `ido >= W` is the right line only while the in-place tail is
            // scalar. With an exact-width cover it blocks passes that win.
            const std::size_t Ln = L * s.radices[p];
            if (Ln * r_last * 2u * sizeof(T) > kGatherWindowBytes) break;
            L = Ln;
            ip_prefix = p;
        }
        // Bit 0 is never set: pass 0 is AoS->SoA, both maps store the same address at
        // l1 == 1. `rowperm` must apply the split-column map to every in-place
        // `wants_reload` bit, else an r16-led chain permutes its first digit wrongly.
        for (std::size_t p = 1; p <= ip_prefix; ++p) s.ip_mask |= 1u << p;

        // The split-column kernel (`dif_passes.hpp`, kSplitCols) does take a `wants_reload`
        // radix in place: phase A stores a+b and (a-b)*W^n, phase B butterflies the
        // disjoint halves, nothing spills. Gates: not the last intermediate pass (its
        // digit tops the block word; appending LOW permutes the full range), and the
        // phase-A/B tile ido*IP*2*sizeof(T) <= 32 KB.
        for (std::size_t p = 1, l1p = s.radices[0]; p + 2 < s.radices.size();
             l1p *= s.radices[p], ++p) {
            const std::size_t ip = s.radices[p];
            // The two groups are disjoint by construction; no already-set test.
            if (s.sched[p] == dif_fuse::plain &&
                butterfly_wants_reload(ip, poet::vector_register_count()) &&
                (N / (l1p * ip)) * ip * 2u * sizeof(T) <= kIpTileBytes)
                s.ip_mask |= 1u << p;
        }
    }
    if (s.ip_mask != 0) {
        // `rowperm[logical]` = physical. Per pass the digit appends HIGH (Stockham,
        // b + l1*k), LOW (in place, col + r*b), or the split column (k&1)*(r/2) +
        // k/2 for `wants_reload`. NOT the (2, r/2) pair expansion: that form gives
        // odd*(r/2) + Kc; the kernel needs 2*Kc + odd. Materialized flat over
        // `l1_last`
        // rows: sequential reads, no runtime div/mod by a possibly-odd L.
        const std::size_t K = s.radices.size() - 1;  // every pass but the last
        const std::size_t l1_last = N / s.radices.back();
        s.rowperm.resize(l1_last);
        std::size_t k[dif_factor_plan::max_passes]{};
        for (std::size_t n = 0; n < l1_last; ++n) {
            std::size_t lg = 0, ph = 0, w = 1;
            for (std::size_t i = 0; i < K; ++i) {
                const std::size_t r = s.radices[i];
                lg += w * k[i];
                if (s.ip_mask >> i & 1u)
                    ph = r * ph + (butterfly_wants_reload(r, poet::vector_register_count())
                                       ? (k[i] & 1u) * (r / 2u) + k[i] / 2u
                                       : k[i]);
                else
                    ph += w * k[i];
                w *= r;
            }
            s.rowperm[lg] = static_cast<std::uint32_t>(ph);
            for (std::size_t i = 0; i < K; ++i)
                if (++k[i] < s.radices[i]) break;
                else k[i] = 0;
        }
    }

    // Record the row driver's walk once, per direction and per es2 variant.
    if (fuse_packed && !s.radices.empty()) {
        dif_build_tape<T, true>(s, N);
        dif_build_tape<T, false>(s, N);
    }

    return s;
}

} // namespace detail
} // namespace admiral

