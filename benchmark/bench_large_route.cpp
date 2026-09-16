// Forced-route 1-D driver (WI-2c tooling): executes the serial iterative-DIF chain or
// four_step_large at sizes where the shipped picker elects the other, so the hand-fit
// kLargeRoute* lines in include/admiral/detail/four_step_large.hpp can be re-derived per host
// class by A/B over the byte range instead of nudged. No in-tree constant is changed: the
// forced route rides the same detail::plan_impl forced-route constructor
// test/transforms/test_route_forced.cpp uses, and plan construction stays at
// effort::estimate with an explicit nthreads (the effort::measure race is bistable at 1-D
// 2^15..2^18 on 2P nodes).
//
// Interface (a peer fft_bench lane packages cluster jobs against exactly this):
//   bench_large_route --route={auto|dif|four_step} --prec={f32|f64} [--nthreads=N]
//                     [--bytes=LIST | --n=LIST] [--reps=R] [--ramp-ms=M] [--huge={0,1}]
//                     [--out=TSV]
//
// Protocol per cell: ramp-clock spin (--ramp-ms), one uncounted warm execute, then min of
// --reps wall-clock out-of-place executes; 64 B aligned buffers, one deterministic LCG fill.
// The calling thread is pinned to the first --nthreads physical cores (one per core, and the
// pool threads inherit the mask). --huge=1 adds MADV_HUGEPAGE on the data buffers. stderr
// carries the host header (cache sizes, the kFourStepStreamL3Mult * L3 streaming threshold,
// the pin) and the numeric self-check verdicts; stdout (or --out) carries one TSV row/cell.

#include <admiral/admiral.hpp>

#include <admiral/detail/cache.hpp>
#include <admiral/detail/four_step_large.hpp>
#include <admiral/detail/plan.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

// Baked by benchmark/CMakeLists.txt at configure time; the literals here are the
// never-configured fallbacks.
#ifndef ADM_BENCH_SHA
#define ADM_BENCH_SHA "unknown"
#endif
#ifndef ADM_BENCH_COMPILER
#define ADM_BENCH_COMPILER "unknown"
#endif
#ifndef ADM_BENCH_ARCH
#define ADM_BENCH_ARCH "unknown"
#endif
#ifndef ADM_BENCH_CXXSTD
#define ADM_BENCH_CXXSTD "unknown"
#endif

namespace {

using clk = std::chrono::steady_clock;
using admiral::detail::plan_impl;

constexpr unsigned kSeed = 7;
constexpr std::size_t kAlign = 64;

enum class route_sel { auto_route, dif, four_step };

struct Cfg {
    route_sel route = route_sel::auto_route;
    bool route_set = false;
    std::string prec;
    std::size_t nthreads = 0;  // 0/absent: auto resolution (route=auto), serial (forced)
    int reps = 5;
    long ramp_ms = 0;
    bool huge = false;
    std::string sizes_n;      // comma list of element counts
    std::string sizes_bytes;  // comma list of N * sizeof(complex<T>)
    std::string out;          // TSV path; empty = stdout
};

[[noreturn]] void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s --route={auto|dif|four_step} --prec={f32|f64} [--nthreads=N]\n"
                 "          [--bytes=LIST | --n=LIST] [--reps=R] [--ramp-ms=M] [--huge={0,1}]\n"
                 "          [--out=TSV]\n"
                 "  route:   auto = shipped picker (TSV reports what it elects), dif = force the\n"
                 "          serial iterative-DIF chain at any size, four_step = force"
                 " four_step_large\n"
                 "  nthreads: explicit plan width, bypassing resolve_nthreads; 0 or absent =\n"
                 "          auto resolution for route=auto, serial for the forced routes\n"
                 "  bytes:   N * sizeof(complex<T>) comma list (must divide evenly); n: element\n"
                 "          counts; exactly one of the two, at least one value\n"
                 "  reps:    timed executes per cell, min reported (default 5); one uncounted\n"
                 "          warm execute always runs first\n"
                 "  ramp-ms: busy-spin per cell before measuring, to reach the boost clock\n"
                 "  huge:    1 = MADV_HUGEPAGE the data buffers; results are bitwise unchanged\n"
                 "  out:     TSV output path (default stdout); stderr gets the host header and\n"
                 "          the self-check verdicts\n"
                 "Plans use effort::estimate semantics only (effort::measure is never used: its\n"
                 "race is bistable at 1-D 2^15..2^18 on 2P nodes). Every invocation first runs\n"
                 "the numeric self-check and exits nonzero when any arm fails:\n"
                 "  (a) forced-dif vs forced-four_step agree at band sizes, rel tol 1e-11 f64 /\n"
                 "      1e-5 f32 (1e-11 is below f32 eps; 1e-5 ~= 80 eps at the sizes checked)\n"
                 "  (b) a small-N case matches a direct O(N^2) long double reference\n"
                 "  (c) huge=0 vs huge=1 outputs are bitwise identical (memcmp)\n",
                 prog);
    std::exit(2);
}

bool parse_kv(const std::string& a, const char* key, std::string& val, int& i, int argc,
              char** argv) {
    const std::string k = std::string("--") + key;
    if (a == k) {
        if (++i >= argc) usage(argv[0]);
        val = argv[i];
        return true;
    }
    if (a.compare(0, k.size() + 1, k + "=") == 0) {
        val = a.substr(k.size() + 1);
        return true;
    }
    return false;
}

std::vector<std::size_t> parse_list(const std::string& s) {
    std::vector<std::size_t> out;
    std::size_t pos = 0;
    while (pos < s.size()) {
        const std::size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos) return {};
        out.push_back(static_cast<std::size_t>(std::stoull(tok)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

struct AlignFree {
    void operator()(void* p) const { std::free(p); }
};
template<typename T>
using buf_ptr = std::unique_ptr<T[], AlignFree>;

template<typename T>
buf_ptr<T> alloc_buf(std::size_t n, [[maybe_unused]] bool huge) {
    // madvise() needs a page-aligned base, so the huge arm aligns the whole buffer to the
    // THP granule: everything THP can actually back then is backed, and 64 B alignment is a
    // weaker condition that still holds.
    constexpr std::size_t kHugeAlign = std::size_t{2} << 20;
    const std::size_t align = huge ? kHugeAlign : kAlign;
    const std::size_t bytes = ((n * sizeof(T) + align - 1) / align) * align;
    void* p = std::aligned_alloc(align, bytes);
    if (p == nullptr) {
        std::fprintf(stderr, "bench_large_route: allocation of %zu bytes failed\n", bytes);
        std::exit(4);
    }
#if defined(__linux__)
    if (huge && madvise(p, bytes, MADV_HUGEPAGE) != 0)
        std::fprintf(stderr, "note: madvise(MADV_HUGEPAGE): %s\n", std::strerror(errno));
#endif
    return buf_ptr<T>(static_cast<T*>(p));
}

template<typename T>
void fill(std::complex<T>* p, std::size_t n) {
    unsigned x = kSeed * 2654435761u + 12345u;
    for (std::size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        const T a = static_cast<T>((x >> 9) & 0xffffu) / static_cast<T>(65536.0) - T(0.5);
        x = x * 1664525u + 1013904223u;
        const T b = static_cast<T>((x >> 9) & 0xffffu) / static_cast<T>(65536.0) - T(0.5);
        p[i] = std::complex<T>(a, b);
    }
}

// One allowed cpu per physical core: the lowest-numbered allowed sibling, so a caller-side
// taskset composes (the enumeration never leaves the current affinity mask).
std::vector<unsigned> anchor_cpus() {
    std::vector<unsigned> out;
#if defined(__linux__)
    cpu_set_t aff;
    if (sched_getaffinity(0, sizeof(aff), &aff) != 0) return out;
    std::vector<unsigned> anchors;
    for (unsigned c = 0; c < static_cast<unsigned>(CPU_SETSIZE); ++c) {
        if (!CPU_ISSET(c, &aff)) continue;
        char path[96];
        std::snprintf(path, sizeof path,
                      "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list", c);
        unsigned first = c;
        if (std::FILE* f = std::fopen(path, "re")) {
            if (std::fscanf(f, "%u", &first) != 1) first = c;
            std::fclose(f);
        }
        if (std::find(anchors.begin(), anchors.end(), first) == anchors.end()) {
            anchors.push_back(first);
            out.push_back(c);
        }
    }
#endif
    return out;
}

bool pin_to([[maybe_unused]] const std::vector<unsigned>& anchors, [[maybe_unused]] std::size_t k) {
#if defined(__linux__)
    if (anchors.empty()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    const std::size_t n = std::min(k, anchors.size());
    for (std::size_t i = 0; i < n; ++i) CPU_SET(anchors[i], &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    return false;
#endif
}

void ramp_clock(long ms) {
    if (ms <= 0) return;
    const auto end = clk::now() + std::chrono::milliseconds(ms);
    volatile double x = 1.0;
    while (clk::now() < end)
        for (int i = 0; i < 2048; ++i) x = x * 1.00000011921 + 1e-15;
}

// Order-sensitive bit digest of the output (FNV-1a over the raw bytes): the guard column.
// The self-check's pointwise comparisons carry the numeric claim; this exists so a sweep can
// confirm two arms produced the same bits without shipping the arrays.
std::uint64_t guard_fnv(const void* p, std::size_t bytes) {
    const auto* b = static_cast<const unsigned char*>(p);
    std::uint64_t h = 14695981039346656037ull;
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

template<typename T>
double rel_diff(const std::complex<T>* a, const std::complex<T>* b, std::size_t n) {
    long double num = 0.0L, den = 0.0L;
    for (std::size_t i = 0; i < n; ++i) {
        const long double dr = static_cast<long double>(a[i].real()) - b[i].real();
        const long double di = static_cast<long double>(a[i].imag()) - b[i].imag();
        const long double br = b[i].real(), bi = b[i].imag();
        num = std::max(num, dr * dr + di * di);
        den = std::max(den, br * br + bi * bi);
    }
    if (den <= 0.0L) return num > 0.0L ? std::numeric_limits<double>::infinity() : 0.0;
    return static_cast<double>(std::sqrt(num / den));
}

template<typename T>
void ref_dft(const std::complex<T>* in, std::complex<long double>* out, std::size_t n) {
    constexpr long double kTwoPi = 6.28318530717958647692528676655900576L;
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<long double> acc(0.0L, 0.0L);
        for (std::size_t j = 0; j < n; ++j) {
            const long double ang = -kTwoPi * static_cast<long double>((k * j) % n) /
                                    static_cast<long double>(n);
            acc += std::complex<long double>(in[j].real(), in[j].imag()) *
                   std::complex<long double>(std::cos(ang), std::sin(ang));
        }
        out[k] = acc;
    }
}

template<typename T>
using P = plan_impl<T>;

template<typename T>
typename P<T>::route_kind forced_kind(route_sel r) {
    switch (r) {
    case route_sel::dif:       return P<T>::route_kind::iterative_dif;
    case route_sel::four_step: return P<T>::route_kind::four_step_large;
    case route_sel::auto_route: break;
    }
    std::fprintf(stderr, "bench_large_route: internal error (forced_kind of auto)\n");
    std::exit(2);
}

template<typename T>
const char* prec_name() {
    return sizeof(T) == 4 ? "f32" : "f64";
}

// ~1e-11: both routes are exact algorithms with different rounding. 1e-11 is below f32 eps,
// so f32 carries 1e-5 instead (~80 eps at the largest checked size); the contract's intent
// is a pointwise agreement floor, not a literal universal constant.
template<typename T>
constexpr double rel_tol() {
    return sizeof(T) == 4 ? 1e-5 : 1e-11;
}

// Self-check (a) sizes, spanning the shipped serial band per precision: f64 is gated on the
// 12 MiB line with no upper edge; f32 on the (16 MiB - 1, 32 MiB] window, and its upper size
// sits PAST the cap so the exact-math claim is guarded where auto has already fallen back
// to the chain.
template<typename T>
std::vector<std::size_t> band_check_sizes() {
    return sizeof(T) == 4 ? std::vector<std::size_t>{std::size_t{1} << 21,
                                                     std::size_t{1} << 23}
                          : std::vector<std::size_t>{std::size_t{1} << 20,
                                                     std::size_t{1} << 22};
}

template<typename T>
bool self_check() {
    bool ok = true;
    const auto report = [&](const char* check, std::size_t n, double rel, bool pass) {
        std::fprintf(stderr, "self-check %s prec=%s n=%zu rel=%.3e tol=%.1e %s\n", check,
                     prec_name<T>(), n, rel, rel_tol<T>(), pass ? "PASS" : "FAIL");
        ok = ok && pass;
    };
    {
        // (a) forced-dif vs forced-four_step, same seeded input, serial. Both sizes are
        // powers of two, so both forced routes can always be constructed here; a
        // construction failure is a check defect, not a quiet skip.
        for (const std::size_t n : band_check_sizes<T>()) {
            auto src = alloc_buf<std::complex<T>>(n, false);
            auto out_d = alloc_buf<std::complex<T>>(n, false);
            auto out_f = alloc_buf<std::complex<T>>(n, false);
            fill(src.get(), n);
            P<T> pd(n, true, forced_kind<T>(route_sel::dif), 1);
            P<T> pf(n, true, forced_kind<T>(route_sel::four_step), 1);
            pd.execute(src.get(), out_d.get());
            pf.execute(src.get(), out_f.get());
            const double rel = rel_diff(out_d.get(), out_f.get(), n);
            report("route-agree", n, rel, rel <= rel_tol<T>());
        }
    }
    {
        // (b) small-N vs a direct O(N^2) long double reference: both forced routes + auto.
        for (const std::size_t n : {std::size_t{64}, std::size_t{96}}) {
            auto src = alloc_buf<std::complex<T>>(n, false);
            auto got = alloc_buf<std::complex<T>>(n, false);
            auto refb = alloc_buf<std::complex<long double>>(n, false);
            fill(src.get(), n);
            ref_dft(src.get(), refb.get(), n);
            const auto vs_ref = [&](const P<T>& pl) {
                pl.execute(src.get(), got.get());
                long double num = 0.0L, den = 0.0L;
                for (std::size_t i = 0; i < n; ++i) {
                    const long double dr =
                        static_cast<long double>(got[i].real()) - refb[i].real();
                    const long double di =
                        static_cast<long double>(got[i].imag()) - refb[i].imag();
                    num = std::max(num, dr * dr + di * di);
                    den = std::max(den, std::norm(refb[i]));
                }
                return static_cast<double>(std::sqrt(num / den));
            };
            P<T> pd(n, true, forced_kind<T>(route_sel::dif), 1);
            P<T> pf(n, true, forced_kind<T>(route_sel::four_step), 1);
            P<T> pa(n, true, 1, nullptr, admiral::effort::estimate);
            const double worst = std::max(std::max(vs_ref(pd), vs_ref(pf)), vs_ref(pa));
            report("ref-small", n, worst, worst <= rel_tol<T>());
        }
    }
    {
        // (c) --huge 0 vs 1 bitwise identical, at a buffer size THP can actually serve.
        const std::size_t n = std::size_t{1} << (sizeof(T) == 4 ? 19 : 18);  // 4 MiB
        auto src = alloc_buf<std::complex<T>>(n, false);
        fill(src.get(), n);
#if defined(__linux__)
        auto out0 = alloc_buf<std::complex<T>>(n, false);
        auto out1 = alloc_buf<std::complex<T>>(n, true);
        P<T> pl(n, true, 1, nullptr, admiral::effort::estimate);
        pl.execute(src.get(), out0.get());
        pl.execute(src.get(), out1.get());
        const bool same =
            std::memcmp(out0.get(), out1.get(), n * sizeof(std::complex<T>)) == 0;
        std::fprintf(stderr, "self-check huge-bitwise prec=%s n=%zu memcmp %s\n",
                     prec_name<T>(), n, same ? "PASS" : "FAIL");
        ok = ok && same;
#else
        std::fprintf(stderr, "self-check huge-bitwise prec=%s n=%zu SKIP (no madvise)\n",
                     prec_name<T>(), n);
#endif
    }
    return ok;
}

template<typename T>
void print_header(const Cfg& cfg, const std::vector<unsigned>& anchors, std::size_t pinned,
                  bool pin_ok) {
    const auto& c = admiral::detail::cpu_cache();
#if defined(__linux__)
    char host[256] = "?";
    if (gethostname(host, sizeof host - 1) == 0) {
        host[sizeof(host) - 1] = '\0';
        std::fprintf(stderr, "# host: %s\n", host);
    }
    struct utsname un {};
    if (uname(&un) == 0)
        std::fprintf(stderr, "# uname: %s %s %s\n", un.sysname, un.release, un.machine);
#endif
    std::fprintf(stderr, "# source-sha: %s\n", ADM_BENCH_SHA);
    std::fprintf(stderr, "# build: compiler=%s target-arch=%s c++%s isa-macros=%s\n",
                 ADM_BENCH_COMPILER, ADM_BENCH_ARCH, ADM_BENCH_CXXSTD,
#if defined(__AVX512F__)
                 "avx512f"
#elif defined(__AVX2__)
                 "avx2"
#elif defined(__ARM_NEON)
                 "neon"
#else
                 "scalar"
#endif
    );
    std::fprintf(stderr,
                 "# cache: L1d=%zu L2=%zu L3=%zu (l3_cores=%zu logical, l3_phys_cores=%zu"
                 " physical, L3/physical-core=%zu)\n",
                 c.l1d, c.l2, c.l3, c.l3_cores, c.l3_phys_cores,
                 c.l3_phys_cores != 0 ? c.l3 / c.l3_phys_cores : std::size_t{0});
    // The line four_step_stream_ok consumes, printed so a re-derivation reads what the
    // engine reads: bytes >= kFourStepStreamL3Mult * L3, plus the two arch-alignment terms
    // (dst base and the n2 row pitch); the per-cell stream_arm_elected column evaluates the
    // same call against the live dst pointer.
    std::fprintf(stderr,
                 "# four_step stream line: bytes >= kFourStepStreamL3Mult(=%zu) * L3 = %zu"
                 " bytes [four_step_stream_ok also requires dst and the n2 row pitch"
                 " arch-aligned]\n",
                 admiral::detail::kFourStepStreamL3Mult,
                 admiral::detail::kFourStepStreamL3Mult * c.l3);
    std::string pin;
    if (pin_ok) {
        const std::size_t np = std::min(pinned, anchors.size());
        for (std::size_t i = 0; i < np; ++i) {
            pin += std::to_string(anchors[i]);
            if (i + 1 < np) pin += ",";
        }
    } else {
        pin = "unavailable (not pinned)";
    }
    std::fprintf(stderr, "# pin: %s\n", pin.c_str());
    if (cfg.huge) std::fprintf(stderr, "# huge: MADV_HUGEPAGE requested on data buffers\n");
}

const char* route_arg_name(route_sel r) {
    switch (r) {
    case route_sel::auto_route: return "auto";
    case route_sel::dif:        return "dif";
    case route_sel::four_step:  return "four_step";
    }
    return "?";
}

template<typename T>
int run_cells(const Cfg& cfg, const std::vector<std::size_t>& sizes, std::FILE* out) {
    std::fprintf(out, "prec\tn\tbytes\tnthreads\troute_forced\troute_elected\t"
                      "stream_arm_elected\treps\tmin_ns\tguard\n");
    for (const std::size_t n : sizes) {
        // The elected route, read off a plan the shipped picker builds under estimate
        // semantics with the same nthreads argument the cell carries. The auto cells time
        // this very plan; the forced cells scrap its state before their plan allocates.
        std::optional<P<T>> elect_plan{std::in_place, n, true, cfg.nthreads, nullptr,
                                       admiral::effort::estimate};
        const std::string elected = elect_plan->route_name();
        std::optional<P<T>> timed;
        try {
            if (cfg.route == route_sel::auto_route) {
                timed = std::move(elect_plan);
            } else {
                elect_plan.reset();
                timed.emplace(n, true, forced_kind<T>(cfg.route),
                              cfg.nthreads != 0 ? cfg.nthreads : 1);
            }
        } catch (const admiral::unsupported_error& e) {
            std::fprintf(stderr,
                         "bench_large_route: --route=%s unavailable at n=%zu prec=%s: %s\n",
                         route_arg_name(cfg.route), n, prec_name<T>(), e.what());
            return 4;
        }
        auto src = alloc_buf<std::complex<T>>(n, cfg.huge);
        auto dst = alloc_buf<std::complex<T>>(n, cfg.huge);
        fill(src.get(), n);
        ramp_clock(cfg.ramp_ms);
        timed->execute(src.get(), dst.get());  // warm: pages, pool wake, never counted
        long long best = std::numeric_limits<long long>::max();
        for (int r = 0; r < cfg.reps; ++r) {
            const auto t0 = clk::now();
            timed->execute(src.get(), dst.get());
            const auto t1 = clk::now();
            best = std::min(best,
                            static_cast<long long>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                                    .count()));
        }
        // What the engine would stream at this cell: the same split the four_step_large_plan
        // ctor picks, the same four_step_stream_ok call, the live dst pointer.
        const char* stream = "na";
        std::string elected_name = timed->route_name();
        if (elected_name == "four_step_large") {
            admiral::detail::large_split sp = admiral::detail::choose_fused_large_split<T>(n);
            if (!sp.valid()) sp = admiral::detail::choose_large_split(n);
            stream = admiral::detail::four_step_stream_ok<T>(
                         dst.get(), sp.n2, n * sizeof(std::complex<T>))
                         ? "1"
                         : "0";
        }
        const std::size_t nt_out =
            cfg.route == route_sel::auto_route ? cfg.nthreads
                                               : (cfg.nthreads != 0 ? cfg.nthreads : 1);
        std::fprintf(out, "%s\t%zu\t%zu\t%zu\t%s\t%s\t%s\t%d\t%lld\t%016llx\n",
                     prec_name<T>(), n, n * sizeof(std::complex<T>), nt_out,
                     route_arg_name(cfg.route), elected.c_str(), stream, cfg.reps, best,
                     static_cast<unsigned long long>(
                         guard_fnv(dst.get(), n * sizeof(std::complex<T>))));
        std::fflush(out);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Cfg cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        std::string v;
        if (a == "--help" || a == "-h") usage(argv[0]);
        else if (parse_kv(a, "route", v, i, argc, argv)) {
            if (v == "auto") cfg.route = route_sel::auto_route;
            else if (v == "dif") cfg.route = route_sel::dif;
            else if (v == "four_step") cfg.route = route_sel::four_step;
            else usage(argv[0]);
            cfg.route_set = true;
        } else if (parse_kv(a, "prec", v, i, argc, argv)) {
            cfg.prec = v;
        } else if (parse_kv(a, "nthreads", v, i, argc, argv)) {
            cfg.nthreads = static_cast<std::size_t>(std::stoull(v));
        } else if (parse_kv(a, "n", v, i, argc, argv)) {
            cfg.sizes_n = v;
        } else if (parse_kv(a, "bytes", v, i, argc, argv)) {
            cfg.sizes_bytes = v;
        } else if (parse_kv(a, "reps", v, i, argc, argv)) {
            cfg.reps = std::atoi(v.c_str());
        } else if (parse_kv(a, "ramp-ms", v, i, argc, argv)) {
            cfg.ramp_ms = std::atol(v.c_str());
        } else if (parse_kv(a, "huge", v, i, argc, argv)) {
            if (v == "0") cfg.huge = false;
            else if (v == "1") cfg.huge = true;
            else usage(argv[0]);
        } else if (parse_kv(a, "out", v, i, argc, argv)) {
            cfg.out = v;
        } else {
            usage(argv[0]);
        }
    }
    if (!cfg.route_set || (cfg.prec != "f32" && cfg.prec != "f64")) usage(argv[0]);
    if (!cfg.sizes_n.empty() && !cfg.sizes_bytes.empty()) usage(argv[0]);
    if (cfg.reps < 1 || cfg.nthreads > 65536) usage(argv[0]);

    const std::size_t elem = cfg.prec == "f32" ? sizeof(std::complex<float>)
                                               : sizeof(std::complex<double>);
    std::vector<std::size_t> sizes;
    if (!cfg.sizes_n.empty()) sizes = parse_list(cfg.sizes_n);
    if (!cfg.sizes_bytes.empty()) {
        for (const std::size_t b : parse_list(cfg.sizes_bytes)) {
            if (b % elem != 0 || b / elem == 0) {
                std::fprintf(stderr, "bench_large_route: --bytes value %zu is not a positive"
                                     " multiple of %zu\n", b, elem);
                return 2;
            }
            sizes.push_back(b / elem);
        }
    }
    if (sizes.empty()) usage(argv[0]);
    for (const std::size_t n : sizes) {
        if (n == 0) {
            std::fprintf(stderr, "bench_large_route: size 0 is not transformable\n");
            return 2;
        }
    }

    std::FILE* out = stdout;
    if (!cfg.out.empty()) {
        out = std::fopen(cfg.out.c_str(), "w");
        if (out == nullptr) {
            std::fprintf(stderr, "bench_large_route: cannot open %s: %s\n", cfg.out.c_str(),
                         std::strerror(errno));
            return 2;
        }
    }

    const std::vector<unsigned> anchors = anchor_cpus();
    // route=auto with nthreads=0 resolves its own width, so it gets every anchor; an
    // explicit width (or a forced serial plan at width 1) gets that many anchors, never two
    // contexts on one core.
    const std::size_t pinned = cfg.nthreads != 0 ? cfg.nthreads : anchors.size();
    const bool pin_ok = pin_to(anchors, pinned);

    int rc;
    if (cfg.prec == "f32") {
        print_header<float>(cfg, anchors, pinned, pin_ok);
        rc = self_check<float>() ? run_cells<float>(cfg, sizes, out) : 3;
    } else {
        print_header<double>(cfg, anchors, pinned, pin_ok);
        rc = self_check<double>() ? run_cells<double>(cfg, sizes, out) : 3;
    }
    if (out != stdout) std::fclose(out);
    return rc;
}
