
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

#include <admiral/detail/math.hpp>

namespace {

using admiral::detail::balanced_split;
using admiral::detail::bluestein_choose_pad;
using admiral::detail::chain_work;
using admiral::detail::four_step_cost;
using admiral::detail::leaf_cost_cyc;
using admiral::detail::lpf_nfac;

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

constexpr std::size_t NMIN = 2, NMAX = 512;
const std::array<const char*, 7> FORM_ORDER = {"codelet", "iterative_dif", "good_thomas",
                                               "four_step", "four_step_batched", "rader",
                                               "bluestein"};

constexpr std::size_t NF = 13;

double lg(double x) { return std::log2(std::max(x, 1e-12)); }
double vecs(std::size_t n, std::size_t w) { return double((n + w - 1) / w); }
double waste(std::size_t n, std::size_t w) {
    return (vecs(n, w) * double(w) - double(n)) / double(w);
}

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
        v = {lg(double(n)), lg(leaf(n)), vecs(n, w), waste(n, w),
             double(nfac), lg(double(lpf))};
    } else if (form == "iterative_dif") {
        const double cw = chain_work(n, w, regs);
        v = {lg(double(n)), cw, cw * lg(double(n)), double(nfac),
             lg(double(lpf)), waste(lpf, w)};
    } else if (form == "four_step" || form == "four_step_batched" || form == "good_thomas") {
        const auto [n1, n2] = balanced_split(n);
        v = {lg(double(n)),      vecs(n1, w) * double(n2) / double(n1),
             double(n2) * vecs(n1, w),    double(n1) * vecs(n2, w),
             lg(double(n1)),     lg(double(n2)),
             waste(n1, w),       waste(n2, w),
             double(n) / double(w), lg(fs_cost(n1, n2))};
    } else if (form == "bluestein") {
        const std::size_t m = bluestein_choose_pad(n);
        const double cw = chain_work(m, w, regs);
        v = {lg(double(n)), lg(double(m)), double(m) / double(n), cw,
             cw * lg(double(m)), vecs(m, w), waste(m, w),
             double(lpf_nfac(m)[1])};
    } else if (form == "rader") {
        const std::size_t L = n > 2 ? n - 1 : 1;
        const double cw = chain_work(L, w, regs);
        const auto [lpf_l, nfac_l] = lpf_nfac(L);
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

using Key = std::tuple<std::string, std::string, int, int, int, std::string,
                       std::string>;
using Table = std::map<Key, std::map<std::size_t, std::map<std::string, double>>>;

[[noreturn]] void die(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

Table load(const std::string& data_dir) {
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
        std::map<std::string, Key> env;
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

std::string cxxd(double v) {
    if (!std::isfinite(v)) die("fit produced a non-finite coefficient");
    if (v == 0.0) v = 0.0;
    char buf[32];
    const auto res = std::to_chars(buf, buf + sizeof buf, v);
    std::string s(buf, res.ptr);
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return s;
}

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
    return four_step;
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

using Summary = std::map<Key, std::tuple<double, double, double, double>>;

const char* form_doc(const std::string& f) {
    if (f == "codelet") return "// One straight-line kernel of length n, priced on the measured leaf table.\n";
    if (f == "iterative_dif") return "// The DIF pass chain, priced on chain_work: the cheapest radix path over n's divisors.\n";
    if (f == "good_thomas") return "// The prime factor split at balanced_split(n), priced on both sub-transform lengths.\n";
    if (f == "four_step") return "// The N = n1*n2 four-step split, priced on the two leaf lengths and the transpose.\n";
    if (f == "four_step_batched") return "// four_step with the column pass batched, priced on the same two leaf lengths.\n";
    if (f == "rader") return "// Prime n as a length-(n-1) cyclic convolution, priced on the DIF chain over n-1.\n";
    if (f == "bluestein") return "// Chirp convolution at the smooth pad from bluestein_choose_pad, priced on that pad.\n";
    return "";
}

void emit(std::ostream& os, const std::vector<std::string>& present,
          const std::map<std::string, Fitted>& coef) {
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
          "\n"
          "#include <array>\n"
          "#include <cstddef>\n"
          "#include <cstdint>\n"
          "#include <tuple>\n"
          "#include <type_traits>\n"
          "#include <utility>\n"
          "\n"
          "#include <admiral/detail/build_id.hpp>\n"
          "#include <admiral/detail/math.hpp>\n"
          "#include \"cxx_compat.hpp\"\n"
          "\n"
          "namespace admiral::detail {\n"
          "\n"
          "enum class base_form : std::uint8_t { codelet, iterative_dif, good_thomas,\n"
          "                                      four_step, four_step_batched, rader, bluestein };\n"
          "\n"
          "struct base_cost_entry {\n"
          "    float cyc;\n"
          "    base_form form;\n"
          "};\n"
          "\n"
          "inline constexpr std::size_t BASE_MODEL_NMIN = 2;\n"
          "inline constexpr std::size_t BASE_MODEL_NMAX = 512;\n"
          "\n"
          "namespace base_model {\n"
          "\n"
          "// Build constants: Wi lanes per batch, bytes per complex element, vec and waste in lanes.\n"
          "template<typename T>\n"
          "struct env {\n"
          "    static constexpr std::size_t Wi = build_width<T>;\n"
          "    static constexpr double W = double(Wi);\n"
          "    static constexpr double bytes = 2.0 * double(sizeof(T));\n"
          "    static constexpr double vec(std::size_t m) { return double((m + Wi - 1) / Wi); }\n"
          "    static constexpr double waste(std::size_t m) { return (vec(m) * W - double(m)) / W; }\n"
          "};\n"
          "\n"
          "template<typename T>\n"
          "[[nodiscard]] constexpr double chain_work(std::size_t n) {\n"
          "    return admiral::detail::chain_work(n, env<T>::Wi, build_vector_regs);\n"
          "}\n"
          "\n"
          "// Appends the three features every form shares: log2(lanes), ln(bytes per element), clang flag.\n"
          "template<typename T, std::size_t K>\n"
          "[[nodiscard]] constexpr std::array<double, K + 3> with_tail(const std::array<double, K>& a) {\n"
          "    std::array<double, K + 3> f{};\n"
          "    for (std::size_t i = 0; i < K; ++i) f[i] = a[i];\n"
          "    f[K]     = ct_log2(env<T>::W);\n"
          "    f[K + 1] = ct_log2(env<T>::bytes) / ct_log2(2.7182818284590452354);\n"
          "    f[K + 2] = build_compiler == \"clang\" ? 1.0 : 0.0;\n"
          "    return f;\n"
          "}\n";
    for (const auto& form : present) {
        const auto& [w, b] = coef.at(form);
        const FormBody& body = form_body(form);
        const std::size_t keep = body.exprs.size();
        os << "\n" << form_doc(form) << "struct " << form << " {\n"
           << "    static constexpr base_form tag = base_form::" << form << ";\n"
           << "    static constexpr double bias = " << cxxd(b) << ";\n";
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
          "// Predicted ln(cycles) for form M at length n: bias + w . features.\n"
          "template<typename M, typename T>\n"
          "[[nodiscard]] constexpr double score(std::size_t n) {\n"
          "    const auto x = M::template features<T>(n);\n"
          "    static_assert(std::tuple_size_v<admiral::detail::remove_cvref_t<decltype(x)>> ==\n"
          "                  std::tuple_size_v<admiral::detail::remove_cvref_t<decltype(M::w)>>,\n"
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
          "}\n"
          "\n"
          "struct base_model_ranking {\n"
          "    std::array<base_form, base_model::NFORM> form{};\n"
          "    std::array<float, base_model::NFORM> log_cyc{};\n"
          "    std::size_t count = 0;\n"
          "    float best_cyc = -1.f;\n"
          "};\n"
          "\n"
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
          "// Every form scored at n, cheapest first. best_cyc is exp() of the winner, so it is in cycles.\n"
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
          "// score_ranking memoized for 2..512 in a function-local static table.\n"
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
          "// The cheapest form at n and its predicted cycles; cyc is -1 outside 2..512.\n"
          "template<typename T>\n"
          "[[nodiscard]] inline base_cost_entry base_cost_for(std::size_t n) noexcept {\n"
          "    const auto& r = base_route_ranking<T>(n);\n"
          "    if (r.count == 0) return {-1.f, base_form::codelet};\n"
          "    return {r.best_cyc, r.form[0]};\n"
          "}\n"
          "\n"
          "}\n";
}

struct Args {
    std::string data = "bench-results";
    std::string out = "include/admiral/detail/base_cost_model.hpp";
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
    if (T.empty() || nreceipts < 40)
        die("not enough usable receipts (" + std::to_string(nreceipts) +
            ") in " + args.data + ", nothing written");

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
    emit(fh, present, coef);
    fh.close();
    std::cout << "wrote " << args.out << " (" << nz << " coefficients, no per-build rows)\n";
    return 0;
}

}

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
