// Positional bit-identity digest: FNV-1a over the raw output bytes of every case in
// test/digest/case_list.hpp. Two builds must agree with the committed goldens line by
// line. `--ulp <case>` is the positive control: it perturbs the first output element of
// exactly one case by one ulp, so that case's digest -- and only that one -- must move.
// Self-checks (case_list.hpp states the rule they enforce): the emitted case count must
// equal kCaseCount, and case kUlpControlCase must still be named kUlpControlName.
#include <admiral/admiral.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "case_list.hpp"

static std::uint64_t fnv(const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static std::uint64_t rng_state = 88172645463325252ull;
static double rnd() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return static_cast<double>(rng_state >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

static std::size_t g_case = 0, g_ulp = ~std::size_t(0), g_dump = ~std::size_t(0);
static const char* g_dumpfile = nullptr;
static std::string g_control_name;

template<typename T>
static void emit(const char* tag, const std::string& name,
                 const std::complex<T>* out, std::size_t n) {
    std::vector<std::complex<T>> v(out, out + n);
    if (g_case == g_ulp && n) v[0] = std::complex<T>(std::nextafter(v[0].real(),
                                     std::numeric_limits<T>::max()), v[0].imag());
    if (g_case == digest_cases::kUlpControlCase) g_control_name = std::string(tag) + " " + name;
    if (g_case == g_dump && g_dumpfile) {
        std::FILE* f = std::fopen(g_dumpfile, "wb");
        if (f) { std::fwrite(v.data(), sizeof(std::complex<T>), n, f); std::fclose(f); }
    }
    std::printf("%4zu %s %-28s %016llx\n", g_case, tag, name.c_str(),
                static_cast<unsigned long long>(fnv(v.data(), n * sizeof(std::complex<T>))));
    ++g_case;
}

template<typename T>
static void fill(std::vector<std::complex<T>>& v) {
    rng_state = 88172645463325252ull;
    for (auto& z : v) z = std::complex<T>(static_cast<T>(rnd()), static_cast<T>(rnd()));
}

template<typename T>
static void do_plan(const char* tag, const std::vector<std::size_t>& shape, bool fwd) {
    std::size_t n = 1; for (auto s : shape) n *= s;
    std::vector<std::complex<T>> v(n); fill(v);
    admiral::options o; o.nthreads = 1; o.eff = admiral::effort::estimate;
    admiral::plan<T> p(admiral::span<const std::size_t>(shape.data(), shape.size()), o);
    if (fwd) p.forward(v.data()); else p.inverse(v.data());
    std::string nm; for (auto s : shape) nm += std::to_string(s) + "x";
    nm += fwd ? "F" : "I";
    emit<T>(tag, nm, v.data(), n);
}

template<typename T>
static void do_strides(const char* tag, const digest_cases::StridesCase& c, bool fwd) {
    const std::size_t need = (c.batch - 1) * c.ibatch + (c.len - 1) * c.istride + 1;
    const std::size_t needo = (c.batch - 1) * c.obatch + (c.len - 1) * c.ostride + 1;
    std::vector<std::complex<T>> in(need), out(needo, std::complex<T>(0, 0));
    fill(in);
    admiral::options o; o.nthreads = 1; o.eff = admiral::effort::estimate;
    admiral::strides_plan<T> p(c.len, c.batch, c.istride, c.ibatch, c.ostride, c.obatch, o);
    if (fwd) p.forward(in.data(), out.data()); else p.inverse(in.data(), out.data());
    emit<T>(tag, "str" + std::to_string(c.len) + "_" + std::to_string(c.istride) + "_" +
            std::to_string(c.ostride) + (fwd ? "F" : "I"), out.data(), needo);
}

template<typename T>
static void sweep(const char* tag) {
    using namespace digest_cases;
    for (std::size_t n = kPlan1dFirst; n <= kPlan1dLast; ++n)
        for (bool f : {true, false}) do_plan<T>(tag, {n}, f);
    for (const auto& sh : kNdShapes) {
        const std::vector<std::size_t> shape(sh.dim, sh.dim + sh.ndim);
        for (bool f : {true, false}) do_plan<T>(tag, shape, f);
    }
    for (const auto& sc : kStridesCases)
        for (bool f : {true, false}) do_strides<T>(tag, sc, f);
}

// Trailing family (case_list.hpp's kCubeShapes): runs after BOTH precision blocks, so
// every committed id keeps its number when the family grows.
template<typename T>
static void sweep_tail(const char* tag) {
    for (const auto& sh : digest_cases::kCubeShapes) {
        const std::vector<std::size_t> shape(sh.dim, sh.dim + sh.ndim);
        for (bool f : {true, false}) do_plan<T>(tag, shape, f);
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        char* end = nullptr;
        if (std::strcmp(argv[i], "--ulp") == 0 && i + 1 < argc) {
            const long v = std::strtol(argv[++i], &end, 10);
            if (end && *end == '\0' && v >= 0) g_ulp = static_cast<std::size_t>(v);
        } else if (std::strcmp(argv[i], "--dump") == 0 && i + 2 < argc) {
            const long v = std::strtol(argv[++i], &end, 10);
            if (end && *end == '\0' && v >= 0) {
                g_dump = static_cast<std::size_t>(v); g_dumpfile = argv[++i];
            }
        }
    }
    sweep<float>("f32");
    sweep<double>("f64");
    sweep_tail<float>("f32");
    sweep_tail<double>("f64");
    if (g_case != digest_cases::kCaseCount) {
        std::fprintf(stderr,
            "digest: emitted %zu cases, kCaseCount is %zu: the case list changed without a "
            "kCaseCount bump and regoldening (test/digest/case_list.hpp)\n",
            g_case, digest_cases::kCaseCount);
        return 2;
    }
    if (g_control_name != digest_cases::kUlpControlName) {
        std::fprintf(stderr,
            "digest: case %zu is \"%s\", expected \"%s\": the case list renumbered, so the "
            "--ulp %zu control no longer targets its case (test/digest/case_list.hpp)\n",
            digest_cases::kUlpControlCase, g_control_name.c_str(), digest_cases::kUlpControlName,
            digest_cases::kUlpControlCase);
        return 3;
    }
    return 0;
}
