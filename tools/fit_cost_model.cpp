// fit_cost_model regenerates include/admiral/detail/base_cost_model.hpp from
// measured sweeps. C++ standard library and one project leaf header (math.hpp)
// only. No Python, no other dependency.
//
// One sparse linear model per form: every receipt in --data pools into the same
// coefficients, so an unswept build routes off them too. Receipts are
// self-describing (arch, compiler, major, W, regs, precision).
//
// Run:  cmake -B b -DADM_FIT_COST_MODEL=ON, targets admiral_cost_sweep /
//       admiral_cost_model; or directly:
//           fit_cost_model [--data DIR] [--alpha A] [--out F]
// Input: base_cost_*.txt BASECOST receipts (see --data).

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// The measured codelet cost table and the engine's own leaf/four-step cost, read from
// the one place they are written down instead of copied here (see CostModel.cmake).
#include <admiral/detail/math.hpp>

namespace {

// The structural features come from math.hpp too, so the emitted header scores a
// plan with the same definitions these coefficients were fitted with.
using admiral::detail::balanced_split;
using admiral::detail::bluestein_choose_pad;
using admiral::detail::chain_work;
using admiral::detail::four_step_cost;
using admiral::detail::leaf_cost_cyc;
using admiral::detail::lpf_nfac;

// Receipts written before BASECOST-ENV existed: those files carry no
// width/register/version fields and nothing is inferable; new sweeps are
// self-describing.
const std::map<std::string, std::tuple<std::string, std::string, int,
                                       std::map<std::string, std::pair<int, int>>>>
    LEGACY_ENV = {
        {"base_cost_gcc_v4.txt",   {"x86_64", "gcc", 14, {{"f32", {16, 32}}, {"f64", {8, 32}}}}},
        {"base_cost_gcc_v3.txt",   {"x86_64", "gcc", 14, {{"f32", {8, 16}}, {"f64", {4, 16}}}}},
        {"base_cost_gcc_v2.txt",   {"x86_64", "gcc", 14, {{"f32", {4, 16}}, {"f64", {2, 16}}}}},
        {"base_cost_clang_v4.txt", {"x86_64", "clang", 18, {{"f32", {16, 32}}, {"f64", {8, 32}}}}},
        {"base_cost_clang_v3.txt", {"x86_64", "clang", 18, {{"f32", {8, 16}}, {"f64", {4, 16}}}}},
        {"base_cost_clang_v2.txt", {"x86_64", "clang", 18, {{"f32", {4, 16}}, {"f64", {2, 16}}}}},
};

constexpr std::size_t NMIN = 2, NMAX = 512;  // measured domain this model reproduces
const std::array<const char*, 7> FORM_ORDER = {"codelet", "iterative_dif", "good_thomas",
                                               "four_step", "four_step_batched", "rader",
                                               "bluestein"};

// Every form's feature vector is zero-padded to NF; the last three slots are
// always log2(W), log(bytes) and the clang indicator. Widest vector: four_step's ten.
constexpr std::size_t NF = 13;

double lg(double x) { return std::log2(std::max(x, 1e-12)); }
double vecs(std::size_t n, std::size_t w) { return double((n + w - 1) / w); }
double waste(std::size_t n, std::size_t w) {
    return (vecs(n, w) * double(w) - double(n)) / double(w);
}

// One form's own features, in feature-table order. The count is the contract with
// form_body()/feature_names(); emit() refuses to write a header that breaks it.
// szt is sizeof(T): the leaf tables are per-precision and W does not imply one,
// since f64 W=8 and f32 W=8 are both real targets.
std::vector<double> feat_own(const std::string& form, std::size_t n, std::size_t w,
                             std::size_t regs, std::size_t szt, bool& ok) {
    ok = true;
    std::vector<double> v;
    const auto [lpf, nfac] = lpf_nfac(n);
    const auto leaf = [szt](std::size_t k) {
        return szt == 4 ? leaf_cost_cyc<float>(k) : leaf_cost_cyc<double>(k);
    };
    const auto fs_cost = [szt](std::size_t a, std::size_t b) {
        return szt == 4 ? four_step_cost<float>(a, b) : four_step_cost<double>(a, b);
    };
    if (form == "codelet") {
        // The measured table, not a polynomial in n: a prime leaf is rader_apply<P>
        // over kernel<P-1>, several times costlier than a composite neighbour. No
        // function of (n, lpf, nfac) expresses that, since lpf(n) == n for every prime,
        // and the gap differs per precision, so the table is per-precision too.
        v = {lg(double(n)), lg(leaf(n)), vecs(n, w), waste(n, w),
             double(nfac), lg(double(lpf))};
    } else if (form == "iterative_dif") {
        const double cw = chain_work(n, w, regs);
        v = {lg(double(n)), cw, cw * lg(double(n)), double(nfac),
             lg(double(lpf)), waste(lpf, w)};
    } else if (form == "four_step" || form == "four_step_batched" || form == "good_thomas") {
        // balanced_split maximizes n1 <= sqrt(n), so its n2 is the SMALLEST any split
        // achieves: whenever four_step is supported at all both factors are in the
        // measured table, and four_step_cost is the engine's own leaf model.
        const auto [n1, n2] = balanced_split(n);
        v = {lg(double(n)),      vecs(n1, w) * double(n2) / double(n1),
             double(n2) * vecs(n1, w),    double(n1) * vecs(n2, w),
             lg(double(n1)),     lg(double(n2)),
             waste(n1, w),       waste(n2, w),
             double(n) / double(w), lg(fs_cost(n1, n2))};
    } else if (form == "bluestein") {
        // bluestein.hpp delegates to bluestein_choose_pad: {2,3,5,7}-smooth with v3<=4,
        // v7<=2, else bit_ceil. Every one of those factors is an admissible radix, so cw is
        // always a real chain here and needs no no_chain indicator (rader's L does).
        const std::size_t m = bluestein_choose_pad(n);
        const double cw = chain_work(m, w, regs);
        v = {lg(double(n)), lg(double(m)), double(m) / double(n), cw,
             cw * lg(double(m)), vecs(m, w), waste(m, w),
             double(lpf_nfac(m)[1])};
    } else if (form == "rader") {
        // rader.hpp runs a length-(p-1) CYCLIC convolution, no chirp-z zero padding, so the
        // discriminator is how well p-1 itself factors (282 = 2*3*47 vs 256).
        const std::size_t L = n > 2 ? n - 1 : 1;
        const double cw = chain_work(L, w, regs);
        const auto [lpf_l, nfac_l] = lpf_nfac(L);  // {1, 0} at L==1, which is n==2
        v = {lg(double(n)), lg(double(L)), cw, cw * lg(double(L)), vecs(L, w),
             waste(L, w), double(nfac_l), lg(double(lpf_l)),
             cw == 0.0 && L > 1 ? 1.0 : 0.0};
    } else {
        ok = false;
    }
    return v;
}

std::array<double, NF> feat(const std::string& form, std::size_t n, std::size_t w,
                            std::size_t regs, double byt, const std::string& cc, bool& ok) {
    const std::vector<double> own = feat_own(form, n, w, regs, std::size_t(byt / 2), ok);
    std::array<double, NF> v{};
    std::copy(own.begin(), own.end(), v.begin());
    v[NF - 3] = lg(double(w));
    v[NF - 2] = std::log(byt);
    v[NF - 1] = cc == "clang" ? 1.0 : 0.0;
    return v;
}

// ---------------------------------------------------------------------------
// Receipts: key = (arch, compiler, major, W, regs, prec, uarch); value n ->
// form -> cyc. uarch is "generic" for receipts predating the field.
// ---------------------------------------------------------------------------
using Key = std::tuple<std::string, std::string, int, int, int, std::string,
                       std::string>;
using Table = std::map<Key, std::map<std::size_t, std::map<std::string, double>>>;

[[noreturn]] void die(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

Table load(const std::string& data_dir) {
    // Two BASECOST-ENV field orders exist on file: uarch last, and uarch between
    // major and prec. Accept both; old receipts are never re-emitted.
    const std::regex env_re(R"(BASECOST-ENV arch=(\S+) compiler=(\S+) major=(\d+) )"
                            R"(prec=(\w+) w=(\d+) regs=(\d+)(?: uarch=(\S+))?)");
    const std::regex env_re_mid(R"(BASECOST-ENV arch=(\S+) compiler=(\S+) major=(\d+) )"
                                R"(uarch=(\S+) prec=(\w+) w=(\d+) regs=(\d+))");
    const std::regex cost_re(R"(BASECOST size=\s*(\d+) prec=(\w+) form=(\w+) cyc=\s*([\d.]+))");
    Table T;
    std::vector<std::string> files;
    for (const auto& e : std::filesystem::directory_iterator(data_dir)) {
        const std::string fn = e.path().filename().string();
        if (fn.starts_with("base_cost_") && fn.ends_with(".txt"))
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) die("no base_cost_*.txt receipts in " + data_dir);
    for (const auto& path : files) {
        const std::string fn = std::filesystem::path(path).filename().string();
        std::map<std::string, Key> env;  // prec -> key
        if (const auto it = LEGACY_ENV.find(fn); it != LEGACY_ENV.end()) {
            const auto& [arch, cc, major, per_prec] = it->second;
            for (const auto& [p, wr] : per_prec)
                env[p] = {arch, cc, major, wr.first, wr.second, p, "generic"};
        }
        std::ifstream fh(path);
        std::string line;
        while (std::getline(fh, line)) {
            std::smatch m;
            if (std::regex_search(line, m, env_re)) {
                env[m[4]] = {m[1], m[2], std::stoi(m[3]), std::stoi(m[5]), std::stoi(m[6]),
                             m[4], m[7].matched ? m[7].str() : "generic"};
                continue;
            }
            if (std::regex_search(line, m, env_re_mid)) {
                env[m[5]] = {m[1], m[2], std::stoi(m[3]), std::stoi(m[6]), std::stoi(m[7]),
                             m[5], m[4]};
                continue;
            }
            if (!std::regex_search(line, m, cost_re)) continue;
            const std::size_t n = std::stoull(m[1]);
            const std::string prec = m[2];
            if (n < NMIN || n > NMAX) continue;
            if (!env.count(prec))
                die(fn + ": BASECOST line for prec=" + prec +
                    " with no BASECOST-ENV (re-sweep, or add the file to LEGACY_ENV)");
            T[env[prec]][n][m[3]] = std::stod(m[4]);
        }
    }
    return T;
}

// ---------------------------------------------------------------------------
// Lasso on standardized features via cyclic coordinate descent, returned as
// raw-space (w, b): objective (1/2n)||y - Xw - b||^2 + alpha*||w||_1.
// ---------------------------------------------------------------------------
std::pair<std::vector<double>, double>
fit_form(const std::vector<std::array<double, NF>>& X, const std::vector<double>& y,
         double alpha) {
    const std::size_t n = X.size();
    std::array<double, NF> mu{}, sd{};
    for (const auto& x : X)
        for (std::size_t j = 0; j < NF; ++j) mu[j] += x[j];
    for (double& m : mu) m /= double(n);
    for (const auto& x : X)
        for (std::size_t j = 0; j < NF; ++j) sd[j] += (x[j] - mu[j]) * (x[j] - mu[j]);
    for (double& v : sd) {
        v = std::sqrt(v / double(n));
        if (v == 0.0) v = 1.0;
    }
    std::vector<std::array<double, NF>> Z(X.begin(), X.end());
    for (auto& z : Z)
        for (std::size_t j = 0; j < NF; ++j) z[j] = (z[j] - mu[j]) / sd[j];

    const double b0 = std::accumulate(y.begin(), y.end(), 0.0) / double(n);
    std::vector<double> r(y.begin(), y.end());
    for (double& v : r) v -= b0;

    std::array<double, NF> z{}, colnorm{};
    for (std::size_t j = 0; j < NF; ++j) {
        for (const auto& zi : Z) colnorm[j] += zi[j] * zi[j];
        colnorm[j] /= double(n);
    }
    for (int iter = 0; iter < 200000; ++iter) {
        double max_change = 0.0;
        for (std::size_t j = 0; j < NF; ++j) {
            if (colnorm[j] == 0.0) continue;
            double rho = colnorm[j] * z[j];
            for (std::size_t i = 0; i < n; ++i) rho += Z[i][j] * r[i] / double(n);
            const double z_new =
                std::copysign(std::max(std::fabs(rho) - alpha, 0.0), rho) / colnorm[j];
            const double dz = z_new - z[j];
            if (dz == 0.0) continue;
            for (std::size_t i = 0; i < n; ++i) r[i] -= Z[i][j] * dz;
            z[j] = z_new;
            max_change = std::max(max_change, std::fabs(dz));
        }
        if (max_change < 1e-13) break;
    }
    // z holds the standardized coefficients; w_i = z_i/s_i, b = b0 - sum z_i mu_i / s_i.
    std::vector<double> w(NF);
    double b = b0;
    for (std::size_t j = 0; j < NF; ++j) {
        w[j] = z[j] / sd[j];
        b -= z[j] * mu[j] / sd[j];
    }
    return {w, b};
}

double score(const std::array<double, NF>& x, const std::vector<double>& w, double b) {
    double acc = b;
    for (std::size_t i = 0; i < NF; ++i) acc += w[i] * x[i];
    return acc;
}

std::string build_name(const Key& k) {
    char buf[80];
    const char* name_tail = std::get<6>(k) == "generic" ? "" : " uarch=";
    if (std::get<1>(k).empty())
        std::snprintf(buf, sizeof buf, "%s any compiler %s W=%2d regs=%d%s%s",
                      std::get<0>(k).c_str(), std::get<5>(k).c_str(), std::get<3>(k),
                      std::get<4>(k), name_tail,
                      std::get<6>(k) == "generic" ? "" : std::get<6>(k).c_str());
    else
        std::snprintf(buf, sizeof buf, "%s %s-%d %s W=%2d regs=%d%s%s",
                      std::get<0>(k).c_str(), std::get<1>(k).c_str(), std::get<2>(k),
                      std::get<5>(k).c_str(), std::get<3>(k), std::get<4>(k), name_tail,
                      std::get<6>(k) == "generic" ? "" : std::get<6>(k).c_str());
    return buf;
}

// Shortest round-trip double literal, like python's repr(float).
std::string cxxd(double v) {
    if (!std::isfinite(v)) die("fit produced a non-finite coefficient");
    if (v == 0.0) v = 0.0;  // canonicalize -0.0 so refits stay diff-stable
    char buf[32];
    const auto res = std::to_chars(buf, buf + sizeof buf, v);
    std::string s(buf, res.ptr);
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";  // stay a double literal
    return s;
}

// C++ for each form's own features, in feature-table order. `env` is the
// width/precision environment, n the transform length.
struct FormBody {
    const char* pre[2];
    std::vector<const char*> exprs;
};
const FormBody& form_body(const std::string& form) {
    static const FormBody codelet = {
        {"const auto [lpf, nfac] = lpf_nfac(n);", nullptr},
        {"ct_log2(double(n))", "ct_log2(leaf_cost_cyc<T>(n))", "e::vec(n)", "e::waste(n)",
         "double(nfac)", "ct_log2(double(lpf))"}};
    static const FormBody iterative_dif = {
        {"const auto [lpf, nfac] = lpf_nfac(n);", "const double cw = chain_work<T>(n);"},
        {"ct_log2(double(n))", "cw", "cw * ct_log2(double(n))", "double(nfac)",
         "ct_log2(double(lpf))", "e::waste(lpf)"}};
    static const FormBody four_step = {
        {"const auto [n1, n2] = balanced_split(n);", nullptr},
        {"ct_log2(double(n))", "e::vec(n1) * double(n2) / double(n1)",
         "double(n2) * e::vec(n1)", "double(n1) * e::vec(n2)", "ct_log2(double(n1))",
         "ct_log2(double(n2))", "e::waste(n1)", "e::waste(n2)", "double(n) / e::W",
         "ct_log2(four_step_cost<T>(n1, n2))"}};
    static const FormBody bluestein = {
        {"const std::size_t m = bluestein_choose_pad(n);",
         "const double cw = chain_work<T>(m);"},
        {"ct_log2(double(n))", "ct_log2(double(m))", "double(m) / double(n)", "cw",
         "cw * ct_log2(double(m))", "e::vec(m)", "e::waste(m)",
         "double(lpf_nfac(m)[1])"}};
    static const FormBody rader = {
        {"const std::size_t L = n > 2 ? n - 1 : 1;", "const double cw = chain_work<T>(L);"},
        {"ct_log2(double(n))", "ct_log2(double(L))", "cw", "cw * ct_log2(double(L))",
         "e::vec(L)", "e::waste(L)", "double(lpf_nfac(L)[1])",
         "ct_log2(double(lpf_nfac(L)[0]))", "cw == 0.0 && L > 1 ? 1.0 : 0.0"}};
    if (form == "codelet") return codelet;
    if (form == "iterative_dif") return iterative_dif;
    if (form == "bluestein") return bluestein;
    if (form == "rader") return rader;
    return four_step;  // four_step, four_step_batched, good_thomas
}

const std::vector<const char*>& feature_names(const std::string& form) {
    static const std::vector<const char*> codelet = {
        "log2(n)", "log2(leaf_cyc)", "vec", "lane_waste", "n_factors", "log2(lpf)"};
    static const std::vector<const char*> iterative_dif = {
        "log2(n)", "chainwork", "chainwork*log2(n)", "n_factors", "log2(lpf)",
        "lane_waste(lpf)"};
    static const std::vector<const char*> four_step = {
        "log2(n)", "n2*vec(n1)/n1", "n2*vec(n1)", "n1*vec(n2)", "log2(n1)", "log2(n2)",
        "lane_waste(n1)", "lane_waste(n2)", "n/W", "log2(four_step_cost)"};
    static const std::vector<const char*> bluestein = {
        "log2(n)", "log2(M)", "M/n", "chainwork(M)", "chainwork(M)*log2(M)", "vec(M)",
        "lane_waste(M)", "n_factors(M)"};
    static const std::vector<const char*> rader = {
        "log2(n)", "log2(L)", "chainwork(L)", "chainwork(L)*log2(L)", "vec(L)",
        "lane_waste(L)", "n_factors(L)", "log2(lpf(L))", "no_chain(L)"};
    if (form == "codelet") return codelet;
    if (form == "iterative_dif") return iterative_dif;
    if (form == "bluestein") return bluestein;
    if (form == "rader") return rader;
    return four_step;
}

struct Fitted {
    std::vector<double> w;
    double b;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
}

// Per build: out-of-fold median, p90 and mean route regret, then in-sample max.
using Summary = std::map<Key, std::tuple<double, double, double, double>>;

void emit(std::ostream& os, const std::vector<std::string>& present,
          const std::map<std::string, Fitted>& coef, const Summary& summary,
          std::size_t nz) {
    // The emitted features() must compute exactly what feat_own() fitted. Checked,
    // not assumed. Probed at a small prime and at one above the leaf table, where
    // balanced_split(509) = {1, 509} and the rader/bluestein preludes must stay in
    // range.
    for (const std::string f : FORM_ORDER) {
        for (const std::size_t probe : {std::size_t{13}, std::size_t{509}}) {
            bool ok = false;
            const std::size_t nown = feat_own(f, probe, 8, 32, 8, ok).size();
            if (form_body(f).exprs.size() != nown || feature_names(f).size() != nown)
                die("feature count mismatch for form " + f + " at n=" + std::to_string(probe) +
                    ": feat_own=" + std::to_string(nown) + " exprs=" +
                    std::to_string(form_body(f).exprs.size()) + " names=" +
                    std::to_string(feature_names(f).size()));
            if (nown + 3 > NF) die("NF too small for form " + f);
        }
    }
    // bluestein carries no no_chain indicator (rader does): its pad is
    // {2,3,5,7}-smooth or a power of two, all admissible radices, so chain_work is
    // never the sentinel. The narrow 7-radix set is checked since the wide one
    // contains it; chain_work clamps "no chain" to 0.0, unreachable for a real pad.
    for (std::size_t n = NMIN; n <= NMAX; ++n) {
        const std::size_t pad = bluestein_choose_pad(n);
        if (chain_work(pad, 8, 16) == 0.0)
            die("bluestein pad " + std::to_string(pad) + " (n=" + std::to_string(n) +
                ") has no admissible radix chain: restore the no_chain feature");
    }
    os << "#pragma once\n"
          "\n"
          "// GENERATED by tools/fit_cost_model.cpp. Do not edit by hand.\n"
          "// Regenerate: cmake -DADM_FIT_COST_MODEL=ON, target admiral_cost_model.\n"
          "//\n"
          "// Routing cost model: one sparse linear model per form over structural\n"
          "// features of N, predicting log(cycles). The route is argmin over forms, and\n"
          "// exp is monotonic, so ranking needs no exponentiation.\n"
          "//\n"
          "// The coefficients are shared by every build: they describe the algorithm\n"
          "// (radix content, lane waste, padding, footprint), so even a target this tool\n"
          "// never swept routes on them, and nothing here is keyed to a machine. One\n"
          "// slot is keyed to the COMPILER, which a machine key would not reach: gcc and\n"
          "// clang price four_step against iterative_dif 1.25x apart on one host at one\n"
          "// kernel vintage, and they move that pair in OPPOSITE directions at equal width.\n"
          "// An unswept compiler takes the gcc branch. Every\n"
          "// form is priced off what the engine actually runs: the measured PER-PRECISION\n"
          "// leaf table (math.hpp) for codelet-terminated forms, the elected pad for\n"
          "// Bluestein. A cell the coefficients still misprice is recovered by measuring at\n"
          "// plan time (effort::automatic), not by a table that only covers swept machines.\n"
          "//\n";
    os << "// " << nz << " coefficients. Route regret vs the exhaustive " << NMIN << ".."
       << NMAX << " sweep,\n"
       << "// in-sample max | out-of-fold:\n";
    for (const auto& [k, s] : summary) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "max %4.1f%%  |  median %4.1f%% p90 %5.1f%% mean %5.1f%%",
                      100 * std::get<3>(s), 100 * std::get<0>(s), 100 * std::get<1>(s),
                      100 * std::get<2>(s));
        os << "//   " << build_name(k) << ": " << buf << "\n";
    }
    os << "//\n"
          "\n"
          "#include <array>\n"
          "#include <cstddef>\n"
          "#include <cstdint>\n"
          "#include <tuple>\n"
          "#include <type_traits>\n"
          "#include <utility>\n"
          "\n"
          "#include <admiral/detail/build_id.hpp>\n"   // build_width, build_vector_regs, build_compiler
          "#include <admiral/detail/math.hpp>\n"       // ct_log2, the measured leaf table
          "\n"
          "namespace admiral::detail {\n"
          "\n"
          "enum class base_form : std::uint8_t { codelet, iterative_dif, good_thomas,\n"
          "                                      four_step, four_step_batched, rader, bluestein };\n"
          "\n"
          "struct base_cost_entry {\n"
          "    float cyc;   // modelled cycles (< 0 outside the modelled domain)\n"
          "    base_form form;\n"
          "};\n"
          "\n"
          "inline constexpr std::size_t BASE_MODEL_NMIN = 2;\n"
          "inline constexpr std::size_t BASE_MODEL_NMAX = 512;\n"
          "\n"
          "namespace base_model {\n"
          "\n"
          "// Width/precision environment: the only target properties the features read.\n"
          "template<typename T>\n"
          "struct env {\n"
          "    static constexpr std::size_t Wi = build_width<T>;\n"
          "    static constexpr double W = double(Wi);\n"
          "    static constexpr double bytes = 2.0 * double(sizeof(T));\n"
          "    static constexpr double vec(std::size_t m) { return double((m + Wi - 1) / Wi); }\n"
          "    static constexpr double waste(std::size_t m) { return (vec(m) * W - double(m)) / W; }\n"
          "};\n"
          "\n"
          "// lpf_nfac, balanced_split and chain_work are NOT defined here: they live in\n"
          "// math.hpp, and tools/fit_cost_model.cpp fits the coefficients below by calling\n"
          "// those same definitions. A copy in this file would be a copy the lasso can\n"
          "// silently disagree with, which is how a partial-chain clamp once mispriced\n"
          "// 1355 of 2047 lengths. Only the <T> binding of W and the register count is\n"
          "// local, because the fitter pools targets and cannot bind either.\n"
          "template<typename T>\n"
          "[[nodiscard]] constexpr double chain_work(std::size_t n) {\n"
          "    return admiral::detail::chain_work(n, env<T>::Wi, build_vector_regs);\n"
          "}\n"
          "\n"
          "// Every form also carries log2(W), log(bytes per element pair) and a clang\n"
          "// indicator. The two compilers price four_step against iterative_dif 1.25x\n"
          "// apart on one host at one kernel vintage, and that pair is the dominant route\n"
          "// below 512; an unswept compiler takes the gcc branch.\n"
          "template<typename T, std::size_t K>\n"
          "[[nodiscard]] constexpr std::array<double, K + 3> with_tail(const std::array<double, K>& a) {\n"
          "    std::array<double, K + 3> f{};\n"
          "    for (std::size_t i = 0; i < K; ++i) f[i] = a[i];\n"
          "    f[K]     = ct_log2(env<T>::W);\n"
          "    f[K + 1] = ct_log2(env<T>::bytes) / ct_log2(2.7182818284590452354);\n"
          "    f[K + 2] = build_compiler == \"clang\" ? 1.0 : 0.0;\n"
          "    return f;\n"
          "}\n"
          "\n"
          "// One class per form, each priced from ITS OWN sub-problem: bluestein and\n"
          "// rader from the padded length M, four_step from its split, iterative_dif from\n"
          "// its radix chain. The coefficient array is sized to that form's feature\n"
          "// vector, so a mismatch is a compile error rather than a silent misread.\n";
    for (const auto& form : present) {
        const auto& [w, b] = coef.at(form);
        const FormBody& body = form_body(form);
        const std::size_t keep = body.exprs.size();
        os << "\nstruct " << form << " {\n"
           << "    static constexpr base_form tag = base_form::" << form << ";\n"
           << "    static constexpr double bias = " << cxxd(b) << ";\n"
           << "    // ";
        for (const char* nm : feature_names(form)) os << nm << ", ";
        os << "log2(W), log(bytes), is_clang\n";
        os << "    static constexpr std::array<double, " << keep + 3 << "> w = {\n";
        std::string line = "       ";
        const auto push = [&](double v) {
            const std::string s = cxxd(v);
            if (line.size() + s.size() + 2 > 96) {
                os << line << "\n";
                line = "       ";
            }
            line += " " + s + ",";
        };
        for (std::size_t i = 0; i < keep; ++i) push(w[i]);
        push(w[NF - 3]);
        push(w[NF - 2]);
        push(w[NF - 1]);
        os << line << "\n    };\n"
           << "    template<typename T>\n"
           << "    [[nodiscard]] static constexpr std::array<double, " << keep + 3
           << "> features(std::size_t n) {\n"
           << "        using e = env<T>;\n";
        for (const char* pre : body.pre)
            if (pre) os << "        " << pre << "\n";
        os << "        return with_tail<T>(std::array<double, " << keep << ">{\n";
        for (const char* e : body.exprs) os << "            " << e << ",\n";
        os << "        });\n    }\n};\n";
    }
    os << "\nusing forms = std::tuple<";
    for (std::size_t i = 0; i < present.size(); ++i) os << (i ? ", " : "") << present[i];
    os << ">;\n"
          "inline constexpr std::size_t NFORM = std::tuple_size_v<forms>;\n"
          "\n"
          "template<typename M, typename T>\n"
          "[[nodiscard]] constexpr double score(std::size_t n) {\n"
          "    const auto x = M::template features<T>(n);\n"
          "    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(x)>> ==\n"
          "                  std::tuple_size_v<std::remove_cvref_t<decltype(M::w)>>,\n"
          "                  \"feature/coefficient count mismatch\");\n"
          "    double acc = M::bias;\n"
          "    for (std::size_t i = 0; i < x.size(); ++i) acc += M::w[i] * x[i];\n"
          "    return acc;\n"
          "}\n"
          "\n"
          "template<typename T, std::size_t... I>\n"
          "[[nodiscard]] constexpr auto score_all(std::size_t n, std::index_sequence<I...>) {\n"
          "    return std::array<double, sizeof...(I)>{score<std::tuple_element_t<I, forms>, T>(n)...};\n"
          "}\n"
          "\n"
          "template<std::size_t... I>\n"
          "[[nodiscard]] constexpr auto tags(std::index_sequence<I...>) {\n"
          "    return std::array<base_form, sizeof...(I)>{std::tuple_element_t<I, forms>::tag...};\n"
          "}\n"
          "\n"
          "}  // namespace base_model\n"
          "\n"
          "// Ranked routes for n, cheapest modelled first. The caller walks the ranking\n"
          "// and takes the first buildable route, so an unavailable winner degrades to\n"
          "// the model's next choice rather than falling out of the model entirely.\n"
          "struct base_model_ranking {\n"
          "    std::array<base_form, base_model::NFORM> form{};\n"
          "    // form[k]'s score, in LOG cycles. Kept unexponentiated: only best_cyc is ever\n"
          "    // needed in cycles, and one std::exp per form would cost the ranking more than\n"
          "    // it sorts. dbg_cost tracing exponentiates on its own cold path.\n"
          "    std::array<float, base_model::NFORM> log_cyc{};\n"
          "    std::size_t count = 0;\n"
          "    float best_cyc = -1.f;\n"
          "};\n"
          "\n"
          "// Form names for dbg_cost tracing.\n"
          "[[nodiscard]] constexpr const char* base_form_name(base_form f) noexcept {\n"
          "    switch (f) {\n"
          "    case base_form::codelet:           return \"codelet\";\n"
          "    case base_form::iterative_dif:     return \"iterative_dif\";\n"
          "    case base_form::good_thomas:       return \"good_thomas\";\n"
          "    case base_form::four_step:         return \"four_step\";\n"
          "    case base_form::four_step_batched: return \"four_step_batched\";\n"
          "    case base_form::rader:             return \"rader\";\n"
          "    case base_form::bluestein:         return \"bluestein\";\n"
          "    }\n"
          "    return \"?\";\n"
          "}\n"
          "\n"
          "// Scored with ct_log2/ct_exp's series rather than libm, so routing is bitwise\n"
          "// identical across libm versions. Only the table below calls it.\n"
          "template<typename T>\n"
          "[[nodiscard]] constexpr base_model_ranking score_ranking(std::size_t n) {\n"
          "    using namespace base_model;\n"
          "    base_model_ranking out;\n"
          "    if (n < BASE_MODEL_NMIN || n > BASE_MODEL_NMAX) return out;\n"
          "\n"
          "    constexpr auto seq = std::make_index_sequence<NFORM>{};\n"
          "    const auto z = score_all<T>(n, seq);\n"
          "    constexpr auto tag = tags(seq);\n"
          "\n"
          "    // selection sort on log-cost: NFORM is small, and this must stay constexpr.\n"
          "    std::array<bool, NFORM> used{};\n"
          "    for (std::size_t k = 0; k < NFORM; ++k) {\n"
          "        std::size_t bi = NFORM;\n"
          "        for (std::size_t f = 0; f < NFORM; ++f)\n"
          "            if (!used[f] && (bi == NFORM || z[f] < z[bi])) bi = f;\n"
          "        used[bi] = true;\n"
          "        out.form[k] = tag[bi];\n"
          "        out.log_cyc[k] = float(z[bi]);\n"
          "        if (k == 0) out.best_cyc = float(ct_exp(z[bi]));\n"
          "    }\n"
          "    out.count = NFORM;\n"
          "    return out;\n"
          "}\n"
          "\n"
          "// Ranked routes for n, cheapest modelled first; count == 0 outside the domain.\n"
          "//\n"
          "// Filled on first use, not by the compiler: as a constexpr table this exceeded\n"
          "// clang's default -fconstexpr-steps and cost seconds of constant evaluation in\n"
          "// every TU that routes. Slots below NMIN stay value-initialised, so an\n"
          "// out-of-domain n reads slot 0 and reports count == 0 without a second branch.\n"
          "template<typename T>\n"
          "[[nodiscard]] inline const base_model_ranking& base_route_ranking(std::size_t n) {\n"
          "    static const std::array<base_model_ranking, BASE_MODEL_NMAX + 1> table = [] {\n"
          "        std::array<base_model_ranking, BASE_MODEL_NMAX + 1> t{};\n"
          "        for (std::size_t i = BASE_MODEL_NMIN; i <= BASE_MODEL_NMAX; ++i) t[i] = "
          "score_ranking<T>(i);\n"
          "        return t;\n"
          "    }();\n"
          "    return table[n <= BASE_MODEL_NMAX ? n : 0];\n"
          "}\n"
          "\n"
          "// Best modelled route for n; cyc < 0 outside the modelled domain.\n"
          "// Retained for --decomp-report; routing uses base_route_ranking.\n"
          "template<typename T>\n"
          "[[nodiscard]] inline base_cost_entry base_cost_for(std::size_t n) noexcept {\n"
          "    const auto& r = base_route_ranking<T>(n);\n"
          "    if (r.count == 0) return {-1.f, base_form::codelet};\n"
          "    return {r.best_cyc, r.form[0]};\n"
          "}\n"
          "\n"
          "}  // namespace admiral::detail\n";
}

struct Args {
    std::string data = "bench-results";
    std::string out = "include/admiral/detail/base_cost_model.hpp";
    // Fewest coefficients that still rank routes; tighter alphas buy coefficients
    // without moving regret.
    double alpha = 0.05;
};

int run(const Args& args) {
    if (!std::filesystem::is_directory(args.data))
        die("no sweep data at " + args.data + " (pass --data)");
    const Table T = load(args.data);
    std::size_t nsizes = 0, nreceipts = 0;
    for (const auto& [k, d] : T) {
        nsizes += d.size();
        for (const auto& [n, fs] : d) nreceipts += fs.size();
    }
    std::cout << "loaded " << T.size() << " builds, " << nsizes << " sizes, " << nreceipts
              << " (n,form) receipts\n";
    // Refuse to write a header fit from nothing: the default --out is the tracked
    // routing table, so an unparseable sweep must fail loudly, not "succeed" empty.
    if (T.empty() || nreceipts < 40)
        die("not enough usable receipts (" + std::to_string(nreceipts) +
            ") in " + args.data + " -- nothing written");

    // pooled fit: one model per form, slopes shared across every variant and both
    // compilers; only the is_clang slot is not shared.
    struct Pool {
        std::vector<std::array<double, NF>> X;
        std::vector<double> y;
        std::vector<std::size_t> g;
    };
    std::map<std::string, Pool> pool;
    for (const auto& [key, d] : T) {
        [[maybe_unused]] const auto& [arch, cc, major, w, regs, prec, uarch] = key;
        const double byt = (prec == "f32" ? 4 : 8) * 2;
        for (const auto& [n, fs] : d)
            for (const auto& [form, cyc] : fs) {
                bool ok;
                const auto x = feat(form, n, std::size_t(w), std::size_t(regs), byt, cc, ok);
                if (!ok) continue;
                pool[form].X.push_back(x);
                pool[form].y.push_back(std::log(cyc));
                pool[form].g.push_back(n);
            }
    }

    std::map<std::string, Fitted> coef;
    std::map<std::string, std::vector<double>> oof;
    for (auto& [form, p] : pool) {
        if (p.X.size() < 40) continue;
        coef[form] = [&] {
            const auto [w, b] = fit_form(p.X, p.y, args.alpha);
            return Fitted{w, b};
        }();
        // 5-fold out-of-fold predictions over distinct n (deterministic rank%5
        // assignment): the honest estimate for unswept sizes.
        std::vector<double> pred(p.y.size(), 0.0);
        std::vector<std::size_t> groups(p.g.begin(), p.g.end());
        std::sort(groups.begin(), groups.end());
        groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
        for (std::size_t fold = 0; fold < 5; ++fold) {
            Pool tr;
            std::vector<std::size_t> te_idx;
            for (std::size_t i = 0; i < p.X.size(); ++i) {
                const auto rank = std::size_t(
                    std::lower_bound(groups.begin(), groups.end(), p.g[i]) - groups.begin());
                if (rank % 5 == fold)
                    te_idx.push_back(i);
                else {
                    tr.X.push_back(p.X[i]);
                    tr.y.push_back(p.y[i]);
                }
            }
            const auto [w_, b_] = fit_form(tr.X, tr.y, args.alpha);
            for (std::size_t i : te_idx) pred[i] = score(p.X[i], w_, b_);
        }
        oof[form] = std::move(pred);
    }

    std::size_t nz = 0;
    for (const auto& [form, c] : coef)
        nz += std::size_t(std::count_if(c.w.begin(), c.w.end(),
                                        [](double v) { return v != 0.0; })) + 1;
    std::set<std::string> ccs;
    for (const auto& [k, d] : T) ccs.insert(std::get<1>(k));
    std::size_t per_cc = 0;
    for (const auto& [k, d] : T)
        if (std::get<1>(k) == *ccs.begin()) per_cc += d.size();
    std::cout << "fitted " << coef.size() << " forms, " << nz
              << " nonzero coefficients total, shared across " << ccs.size()
              << " compilers (the table it replaces was " << per_cc << " floats + "
              << per_cc << " enums, single-compiler)\n";

    // ---- score the shipped predictor: in-sample and out-of-fold route regret
    // against the measured optimum, per build. Reported, not corrected: what the
    // coefficients cannot rank is what effort::automatic measures at plan time.
    std::map<std::string, std::size_t> idx;
    std::map<std::pair<Key, std::size_t>, std::map<std::string, double>> order, oof_order;
    for (const auto& [key, d] : T) {
        [[maybe_unused]] const auto& [arch, cc, major, w, regs, prec, uarch] = key;
        const double byt = (prec == "f32" ? 4 : 8) * 2;
        for (const auto& [n, fs] : d)
            for (const auto& [form, cyc] : fs) {
                bool ok;
                const auto x = feat(form, n, std::size_t(w), std::size_t(regs), byt, cc, ok);
                if (!ok || !coef.count(form)) continue;
                const auto& c = coef[form];
                order[{key, n}][form] = score(x, c.w, c.b);
                oof_order[{key, n}][form] = oof[form][idx[form]++];
            }
    }

    Summary summary;
    for (const auto& [k, d] : T) {
        std::vector<double> reg, oreg;
        for (const auto& [n, fs] : d) {
            const auto o_it = order.find({k, n});
            if (o_it == order.end()) continue;
            std::map<std::string, double> p;
            for (const auto& [f, v] : o_it->second)
                if (fs.count(f)) p[f] = v;
            if (p.size() < 2) continue;
            const auto argmin = [](const std::map<std::string, double>& m) {
                return std::min_element(m.begin(), m.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                })->first;
            };
            const auto best = argmin(fs);
            const double r = fs.at(argmin(p)) / fs.at(best) - 1.0;
            const auto oo = oof_order.find({k, n});
            if (oo != oof_order.end()) {
                std::map<std::string, double> po;
                for (const auto& [f, v] : oo->second)
                    if (fs.count(f)) po[f] = v;
                oreg.push_back(fs.at(argmin(po)) / fs.at(best) - 1.0);
            }
            reg.push_back(r);
        }
        std::sort(oreg.begin(), oreg.end());
        const double mx = reg.empty() ? 0.0 : *std::max_element(reg.begin(), reg.end());
        const double p90 = oreg.empty() ? 0.0 : oreg[std::size_t(0.9 * double(oreg.size()))];
        summary[k] = {median(oreg), p90, mean(oreg), mx};
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "regret mean %4.1f%% max %5.1f%% of %3zu | OOF mean %4.1f%% p90 %5.1f%%",
                      100 * mean(reg), 100 * mx, reg.size(), 100 * mean(oreg), 100 * p90);
        std::cout << "  " << build_name(k) << ": " << buf << "\n";
    }

    std::vector<std::string> present;
    for (const char* f : FORM_ORDER)
        if (coef.count(f)) present.push_back(f);

    std::ofstream fh(args.out);
    if (!fh) die("cannot open " + args.out + " for writing");
    emit(fh, present, coef, summary, nz);
    fh.close();
    std::cout << "wrote " << args.out << " (" << nz << " coefficients, no per-build rows)\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char* name) -> std::string {
            if (++i >= argc) die(std::string(name) + " expects a value");
            return argv[i];
        };
        if (a == "--data")
            args.data = next("--data");
        else if (a == "--out")
            args.out = next("--out");
        else if (a == "--alpha")
            args.alpha = std::stod(next("--alpha"));
        else
            die("unknown argument: " + a);
    }
    return run(args);
}
