// Granule-admission A/B runner (WI-0c-ii). scripts/validate.sh's granule-admission arm
// builds this binary twice per ISA (ADM_GRANULE_ADMIT ON/OFF) and interleaves invocations
// on one pinned core; the arm owns the verdict, this binary only measures and prints TSV.
//
// Two modes:
//   default:     public-engine cells over the admitted set ({12,24}^2 x {f32,f64} plus
//                {16}^2 f32, and the WI-1a fast3d cube set {4,4,4}/{8,8,8} x {f32,f64}).
//                One out-of-place plan forward call engages both codelet choke points
//                where the knob lives on the square cells (rows through
//                codelet_apply_many_oop, cols through col_codelet_apply) and the fast3d
//                seat on the cube cells, so the same call sequence in the ON and OFF
//                binaries A/Bs exactly the code the admission toggles. Cube cells carry
//                an engagement assert at registration (`uses_fast3d` on the detail
//                plan over the same shape/terms), compiled out with the OFF knob.
//   --candidate: leaf-level transfer cells for the built-but-unadmitted N in
//                {9,10,11,13,14,15}. The engine cannot route there (admission gates the
//                choke points, and widening the predicates is out of scope), so the
//                granule drivers are called DIRECTLY -- the same reach test_granule.cpp
//                uses -- against the library's own SoA leaves for the same N as the
//                baseline, both arms inside one binary.
//
// Protocol per cell: calibrate the burst length to ~30 us, 2 discarded warmup reps,
// R reps of reseed-from-seed + burst, min-of-R per-op ns, per-rep series optional,
// accumulated sink printed on every row (anti-DCE).

#include <admiral/admiral.hpp>

#include <admiral/detail/nd_plan.hpp>

#include "granule_apply.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// The SoA baselines the candidate cells race. Forward-declared so the bench TU links the
// library's own explicit instantiations (one per catalog N) instead of growing its own --
// the baseline bits are then exactly what the engine ships for that N.
namespace admiral::detail {
template<unsigned N, typename T, bool Forward>
void codelet_apply_many_oop(const std::complex<T>*, std::complex<T>*, std::size_t,
                            std::size_t, std::size_t, T);
template<unsigned N, typename T, bool Forward>
void col_codelet_apply(const std::complex<T>*, std::size_t, std::complex<T>*, std::size_t,
                       std::size_t, T);
}  // namespace admiral::detail

namespace {

using clk = std::chrono::steady_clock;
using admiral::detail::codelet_apply_many_oop;
using admiral::detail::col_codelet_apply;
using admiral::detail::granule_apply_rows_oop;
using admiral::detail::granule_col_apply;

constexpr unsigned kSeed = 7;

// Admitted repertoire (N, precision bits, rank): the granule_rows_admit_v /
// granule_cols_admit_v squares plus the granule_cube_admit_v cubes. Kept static so the
// arm can assert every expected cell shows up in the output -- an empty result set is a
// bug, never a quiet pass.
constexpr unsigned kAdmitted[][3] = {{12, 32, 2}, {12, 64, 2}, {16, 32, 2}, {24, 32, 2},
                                     {24, 64, 2}, {4, 32, 3},  {4, 64, 3},  {8, 32, 3},
                                     {8, 64, 3},  {9, 64, 2},  {10, 32, 2}, {10, 64, 2},
                                     {11, 64, 2}, {13, 32, 2}, {13, 64, 2}, {14, 32, 2},
                                     {14, 64, 2}, {15, 32, 2}, {15, 64, 2}};
constexpr unsigned kCandidates[] = {9, 10, 11, 13, 14, 15};

struct Cfg {
    int reps = 9;
    long burst = 0;    // 0: calibrate per cell
    long ramp_ms = 0;  // frequency-ramp spin before each measured cell
    bool series = false;
    bool candidate = false;
    std::string tag = "run";
    std::string cells;  // comma list of N restricting the repertoire; empty = all
    std::string prec;   // "f32" | "f64" | "" (both)
};

bool cell_selected(const Cfg& cfg, unsigned n, int bits) {
    if (!cfg.prec.empty() && ((bits == 32) != (cfg.prec == "f32"))) return false;
    if (cfg.cells.empty()) return true;
    std::size_t pos = 0;
    while (pos <= cfg.cells.size()) {
        const std::size_t comma = cfg.cells.find(',', pos);
        const std::string tok =
            cfg.cells.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!tok.empty() && static_cast<unsigned>(std::stoul(tok)) == n) return true;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return false;
}

template<typename T>
void fill(std::complex<T>* p, std::size_t n, unsigned s) {
    unsigned x = s * 2654435761u + 12345u;
    for (std::size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        const T a = static_cast<T>((x >> 9) & 0xffffu) / static_cast<T>(65536.0) - T(0.5);
        x = x * 1664525u + 1013904223u;
        const T b = static_cast<T>((x >> 9) & 0xffffu) / static_cast<T>(65536.0) - T(0.5);
        p[i] = std::complex<T>(a, b);
    }
}

template<typename T>
std::complex<T>* aligned_buf(std::size_t n) {
    return static_cast<std::complex<T>*>(
        std::aligned_alloc(64, ((n * sizeof(std::complex<T>) + 63) / 64) * 64));
}

// Per-op ns of one burst of k body() calls back to back. Templated on the body so the
// timed loop carries no std::function dispatch: at ~50 ns cells that call overhead would
// dilute both arms toward 1 and eat verdict margin.
template<typename Body>
double burst_ns(Body&& body, long k) {
    const auto t0 = clk::now();
    for (long i = 0; i < k; ++i) body();
    const auto t1 = clk::now();
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
           static_cast<double>(k);
}

template<typename Body>
long calibrate_burst(long requested, Body&& body) {
    if (requested > 0) return requested;
    const double probe = burst_ns(body, 4);
    if (!(probe > 0.0)) return 1;  // paranoia: a zero-resolution probe, treat as instant
    // ~2 ms bursts (ci_perf_ab.py's 20-100 ms/cell discipline, divided across the reps):
    // long enough for intel_pstate to settle at the AVX license bin -- 30-150 us targets
    // measured a +-10-50% on/ctl floor under powersave here, 2 ms measures ~5%.
    const double k = 2000000.0 / probe;
    return std::max(1L, std::min(65536L, static_cast<long>(std::ceil(k))));
}

// intel_pstate in powersave idles the pinned core near base clock and ramps only under
// load; one bench invocation is far too short to ramp by itself, so without a spin the
// per-invocation min lands on a random clock state (observed: +-50% on the on/ctl
// identity floor). A fixed busy spin before each measured cell puts the core at boost
// before the first timed burst. The volatile accumulator keeps the spin live.
void ramp_clock(long ms) {
    if (ms <= 0) return;
    const auto end = clk::now() + std::chrono::milliseconds(ms);
    volatile double x = 1.0;
    while (clk::now() < end)
        for (int i = 0; i < 2048; ++i) x = x * 1.00000011921 + 1e-15;
}

// The shared cell protocol: per rep, reseed src from the pristine seed, time a burst of
// out-of-place calls src -> dst (the seed copy stays the live input, values bounded at
// every burst length), accumulate the sink from dst, keep the min per-op ns across reps.
struct Measured {
    double ns_min;
    long burst;
    double sink;
    std::vector<double> per_rep;
};

template<typename T, typename Body>
Measured measure(const Cfg& cfg, const std::complex<T>* seed, std::complex<T>* src,
                 std::complex<T>* dst, std::size_t n, Body&& body) {
    ramp_clock(cfg.ramp_ms);
    const long k = calibrate_burst(cfg.burst, body);
    Measured m{std::numeric_limits<double>::max(), k, 0.0, {}};
    for (int rep = -2; rep < cfg.reps; ++rep) {  // negative reps: discarded warmup
        std::memcpy(src, seed, n * sizeof(std::complex<T>));
        const double rep_ns = burst_ns(body, k);
        asm volatile("" ::"r"(src), "r"(dst) : "memory");
        if (rep < 0) continue;
        m.ns_min = std::min(m.ns_min, rep_ns);
        if (cfg.series) m.per_rep.push_back(rep_ns);
        for (std::size_t i = 0; i < n; ++i)
            m.sink += static_cast<double>(dst[i].real()) - static_cast<double>(dst[i].imag());
    }
    return m;
}

void print_row(const char* cell, const char* prec, const char* mode, const char* arm,
               const Cfg& cfg, const Measured& m) {
    std::printf(
        "ab\tcell=%s\tprec=%s\tmode=%s\tarm=%s\ttag=%s\tns_min=%.2f\treps=%d\tburst=%ld"
        "\tsink=%.6e",
        cell, prec, mode, arm, cfg.tag.c_str(), m.ns_min, cfg.reps, m.burst, m.sink);
    if (cfg.series) {
        std::printf("\tseries=");
        for (std::size_t i = 0; i < m.per_rep.size(); ++i)
            std::printf("%s%.2f", i ? "," : "", m.per_rep[i]);
    }
    std::printf("\n");
}

template<typename T>
void run_plan_cell(unsigned n, unsigned rank, const Cfg& cfg, const char* prec) {
    std::size_t ntot = 1;
    std::vector<std::size_t> shape(rank, n);
    char cell[24];
    char* cp = cell;
    for (unsigned d = 0; d < rank; ++d) {
        ntot *= n;
        cp += std::snprintf(cp, sizeof(cell) - static_cast<std::size_t>(cp - cell),
                            d ? "x%u" : "%u", n);
    }
    std::complex<T>* seed = aligned_buf<T>(ntot);
    std::complex<T>* src = aligned_buf<T>(ntot);
    std::complex<T>* dst = aligned_buf<T>(ntot);
    fill(seed, ntot, kSeed);
    std::memcpy(src, seed, ntot * sizeof(std::complex<T>));
    admiral::options o;
    o.nthreads = 1;
    o.eff = admiral::effort::estimate;
    const admiral::plan<T> p(admiral::span<const std::size_t>(shape.data(), shape.size()), o);
#ifndef ADM_GRANULE_ADMIT_OFF
    // Engagement, asserted not assumed on the cube cells: the same shape and terms the
    // timed public plan routes on must take the fast3d seat in an ON build. The accessor
    // is not on the public surface, so the probe is the detail plan; the OFF knob compiles
    // the assert out with the seat.
    if (rank == 3 && admiral::detail::kGranuleFmaddsub) {
        const admiral::detail::nd_runtime_plan<T> probe(
            admiral::span<const std::size_t>(shape.data(), shape.size()), true, 1);
        if (!probe.uses_fast3d()) {
            std::fprintf(stderr, "bench_granule_ab: cell %s plan did not take the fast3d "
                                 "seat\n", cell);
            std::exit(3);
        }
    }
#endif
    const auto body = [&] { p.forward(src, dst); };
    print_row(cell, prec, "plan", "lib", cfg, measure<T>(cfg, seed, src, dst, ntot, body));
    std::free(seed);
    std::free(src);
    std::free(dst);
}

// One leaf axis of one candidate cell: both arms share the buffers and the protocol, so
// the granule driver and the engine's SoA leaf differ only in the code under test.
// Granule/Soa take (src, dst) and issue one leaf call of N lines/columns of N points.
template<unsigned N, typename T, typename Granule, typename Soa>
void run_leaf_axis(const Cfg& cfg, const char* cell, const char* prec, const char* axis,
                   Granule&& granule, Soa&& soa) {
    constexpr std::size_t n2 = static_cast<std::size_t>(N) * N;
    std::complex<T>* seed = aligned_buf<T>(n2);
    std::complex<T>* src = aligned_buf<T>(n2);
    std::complex<T>* dst = aligned_buf<T>(n2);
    fill(seed, n2, kSeed);
    const auto g = [&] { granule(src, dst); };
    const auto s = [&] { soa(src, dst); };
    print_row(cell, prec, axis, "granule", cfg, measure<T>(cfg, seed, src, dst, n2, g));
    print_row(cell, prec, axis, "soa", cfg, measure<T>(cfg, seed, src, dst, n2, s));
    std::free(seed);
    std::free(src);
    std::free(dst);
}

template<unsigned N, typename T>
void run_leaf_cell(const Cfg& cfg, const char* prec) {
    // The granule family compiles out below fmaddsub ISAs (kGranuleFmaddsub); the arm
    // only builds v3/v4, but keep the compile-time guard the test uses.
    if constexpr (!admiral::detail::kGranuleFmaddsub) {
        std::printf("ab\tcell=%ux%u\tprec=%s\tmode=leaf\tskipped=kGranuleFmaddsub\n", N, N,
                    prec);
    } else {
        char cell[16];
        std::snprintf(cell, sizeof(cell), "%ux%u", N, N);
        run_leaf_axis<N, T>(
            cfg, cell, prec, "leaf-rows",
            [](const std::complex<T>* in, std::complex<T>* out) {
                granule_apply_rows_oop<N, T, true>(in, out, N, N, N, T(1));
            },
            [](const std::complex<T>* in, std::complex<T>* out) {
                codelet_apply_many_oop<N, T, true>(in, out, N, N, N, T(1));
            });
        run_leaf_axis<N, T>(
            cfg, cell, prec, "leaf-cols",
            [](const std::complex<T>* in, std::complex<T>* out) {
                granule_col_apply<N, T, true>(in, N, out, N, N, T(1));
            },
            [](const std::complex<T>* in, std::complex<T>* out) {
                col_codelet_apply<N, T, true>(in, N, out, N, N, T(1));
            });
    }
}

template<typename T>
void run_candidates(const Cfg& cfg, const char* prec) {
    for (unsigned n : kCandidates) {
        if (!cell_selected(cfg, n, sizeof(T) == 4 ? 32 : 64)) continue;
        switch (n) {
#define ADM_LEAF_CASE(NN) \
    case NN: run_leaf_cell<NN, T>(cfg, prec); break
            ADM_LEAF_CASE(9);
            ADM_LEAF_CASE(10);
            ADM_LEAF_CASE(11);
            ADM_LEAF_CASE(13);
            ADM_LEAF_CASE(14);
            ADM_LEAF_CASE(15);
#undef ADM_LEAF_CASE
            default: break;
        }
    }
}

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s [--candidate] [--reps R] [--burst K] [--tag S] [--series]\n"
                 "          [--cell N[,N...]] [--prec f32|f64] [--ramp-ms N]\n",
                 prog);
    std::exit(2);
}

std::string need_arg(int& i, int argc, char** argv, const char* prog) {
    if (++i >= argc) usage(prog);
    return argv[i];
}

}  // namespace

int main(int argc, char** argv) {
    Cfg cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--candidate") cfg.candidate = true;
        else if (a == "--reps") cfg.reps = std::atoi(need_arg(i, argc, argv, argv[0]).c_str());
        else if (a == "--burst") cfg.burst = std::atol(need_arg(i, argc, argv, argv[0]).c_str());
        else if (a == "--ramp-ms") cfg.ramp_ms = std::atol(need_arg(i, argc, argv, argv[0]).c_str());
        else if (a == "--tag") cfg.tag = need_arg(i, argc, argv, argv[0]);
        else if (a == "--series") cfg.series = true;
        else if (a == "--cell") cfg.cells = need_arg(i, argc, argv, argv[0]);
        else if (a == "--prec") cfg.prec = need_arg(i, argc, argv, argv[0]);
        else usage(argv[0]);
    }
    if (cfg.reps < 1) usage(argv[0]);
    if (!cfg.prec.empty() && cfg.prec != "f32" && cfg.prec != "f64") usage(argv[0]);

    if (cfg.candidate) {
        if (cfg.prec.empty() || cfg.prec == "f32") run_candidates<float>(cfg, "f32");
        if (cfg.prec.empty() || cfg.prec == "f64") run_candidates<double>(cfg, "f64");
        return 0;
    }
    for (const auto& cell : kAdmitted) {
        const unsigned n = cell[0];
        const int bits = static_cast<int>(cell[1]);
        const unsigned rank = cell[2];
        if (!cell_selected(cfg, n, bits)) continue;
        if (bits == 32) run_plan_cell<float>(n, rank, cfg, "f32");
        else run_plan_cell<double>(n, rank, cfg, "f64");
    }
    return 0;
}
