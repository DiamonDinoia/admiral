// Wave-A DAG census pilot (math-clarity campaign D4c): trace the shipped dif_butterfly /
// dif_butterfly_terminal / kernel_batched text through the symbolic V of
// test/admiral-dag and REQUIRE the op census against the shipped closed forms. The
// oracle never re-decides contraction: it reads kFusedFma from the shipped header and
// expands the count the same way the text does. The poison twin builds this file with
// -DADM_DAG_TEST_POISON, which biases the oracle helper by +1; the census REQUIREs then
// fail, proving the comparator is non-vacuous (ctest WILL_FAIL).
//
// ORDER CONTRACT: sym_dag.hpp must precede every admiral include in this TU -- the
// namespace-xsimd fma/fnma/fms shims it declares must be visible at the kernel templates'
// definition points (see that header's top comment; probe negative control
// /tmp/adm-dag-probe/probe_order_neg.cpp). The admiral includes come after it.

#include "admiral-dag/sym_dag.hpp"

#include <admiral/detail/butterfly.hpp>
#include <admiral/detail/codelet.hpp>
#include <admiral/detail/ct_math.hpp>
#include <admiral/detail/granule_codelet.hpp>
#include <admiral/detail/simd_swizzle.hpp>
#include <admiral/detail/twiddles.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace dag = admiral_dag;
using namespace admiral::detail;

namespace {

// The dif_butterfly / dif_butterfly_terminal arm ladder, classified at compile time from
// the shipped constexprs (odd_ct_split, coprime_split, has_single_bit). Census rows are
// labelled by this, never hand-chosen.
enum class dif_arm_kind { radix_sym, ct, pfa, pow2, generic };

const char* arm_name(dif_arm_kind a) noexcept {
    switch (a) {
        case dif_arm_kind::radix_sym: return "radix_sym";
        case dif_arm_kind::ct: return "ct";
        case dif_arm_kind::pfa: return "pfa";
        case dif_arm_kind::pow2: return "pow2";
        case dif_arm_kind::generic: return "generic";
    }
    return "?";  // unreachable: the switch is exhaustive over dif_arm_kind
}

constexpr dif_arm_kind dif_arm(std::size_t n) noexcept {
    if (odd_ct_split(n).first != 0) return dif_arm_kind::ct;
    if (n % 2 == 1 && n >= 3) return dif_arm_kind::radix_sym;
    if (n % 2 == 0 && coprime_split(n).first != 0) return dif_arm_kind::pfa;
    if (n >= 4 && has_single_bit(n)) return dif_arm_kind::pow2;
    return dif_arm_kind::generic;
}

constexpr dif_arm_kind terminal_arm(std::size_t n) noexcept {
    if (n % 2 == 1 && coprime_split(n).first != 0) return dif_arm_kind::pfa;
    return dif_arm(n);
}

// ---------------------------------------------------------------------------
// Oracles. The closed forms count each piece_* call as ONE op, matching the text as
// shipped under kFusedFma; at ISA levels without an FMA unit the same text issues
// mul+add, so the oracle expands the same way. Contraction is never re-decided here.
// ---------------------------------------------------------------------------

constexpr std::size_t sym_ops_oracle(std::size_t n) noexcept {
    const std::size_t h = (n - 1) / 2;
    const std::size_t base = kFusedFma ? sym_dft_ops(n) : 8 * h * h + 10 * h;
#ifdef ADM_DAG_TEST_POISON
    // Comparator control: a +1 oracle must turn the odd-arm census REQUIRE below red.
    return base + 1;
#else
    return base;
#endif
}

constexpr std::size_t ct_ops_oracle(std::size_t n1, std::size_t n2) noexcept {
    if (kFusedFma) return ct_dft_ops(n1, n2);  // shipped closed form
    return n2 * sym_ops_oracle(n1) + n1 * sym_ops_oracle(n2) + 6 * (n1 - 1) * (n2 - 1);
}

constexpr std::size_t pfa_ops_oracle(std::size_t n1, std::size_t n2) noexcept {
    return n2 * sym_ops_oracle(n1) + n1 * sym_ops_oracle(n2);
}

// ---------------------------------------------------------------------------
// Fold-set construction. ct_sincos_turns is consteval at C++20, so each (Num, Den, Conj)
// fold instantiates at compile time with the shipped fold spelling (ct_real_t<T>, then a
// static_cast per value) and the bits are memcpy'd at runtime.
// ---------------------------------------------------------------------------

template <typename T>
constexpr char prec_char() {
    return sizeof(T) == 4 ? 'f' : 'd';
}

template <typename T, std::size_t Num, std::size_t Den, bool Conj>
constexpr std::array<T, 2> fold_vals() {
    constexpr auto w = ct_sincos_turns<ct_real_t<T>>(Conj, Num, Den);
    return {static_cast<T>(w.c), static_cast<T>(w.s)};
}

template <typename T, std::size_t Num, std::size_t Den, bool Conj>
void push_turn(std::vector<dag::turn_entry>& v) {
    constexpr std::array<T, 2> cs = fold_vals<T, Num, Den, Conj>();
    v.push_back(dag::turn_entry{static_cast<std::uint32_t>(Num),
                                static_cast<std::uint32_t>(Den), Conj, dag::bits_of(cs[0]),
                                dag::bits_of(cs[1])});
}

template <typename T, std::size_t Den, std::size_t... Num>
void append_den(std::vector<dag::turn_entry>& v, std::index_sequence<Num...>) {
    ((push_turn<T, Num, Den, false>(v), push_turn<T, Num, Den, true>(v)), ...);
}

template <std::size_t... Dens>
struct dens_list {};

template <std::size_t... Dens>
inline constexpr dens_list<Dens...> dens{};

template <typename T, std::size_t... Dens>
std::vector<dag::turn_entry> fold_set(dens_list<Dens...>) {
    std::vector<dag::turn_entry> v;
    (append_den<T, Dens>(v, std::make_index_sequence<Dens>{}), ...);
    return v;
}

// ---------------------------------------------------------------------------
// Butterfly tracer.
// ---------------------------------------------------------------------------

struct dif_row {
    std::uint64_t ops = 0;
    std::uint64_t created = 0;
    std::uint64_t dups = 0;
    std::uint32_t emitted = 0;
    std::array<std::uint64_t, dag::kOpCount> hist{};
};

template <typename T, std::size_t IP, bool Terminal, std::size_t... Dens>
dif_row trace_butterfly(const char* family, dens_list<Dens...> ds) {
    using S = dag::sym<T>;
    dag::pool<T> pool;
    dag::sym_ctx<T> ctx;
    ctx.p = &pool;
    S::bind(&ctx);
    S xr[IP], xi[IP];
    for (std::size_t i = 0; i < IP; ++i) {
        xr[i] = S::input(dag::plane::re, dag::layout::soa, static_cast<std::uint32_t>(i),
                         dag::provenance::aligned);
        xi[i] = S::input(dag::plane::im, dag::layout::soa, static_cast<std::uint32_t>(i),
                         dag::provenance::aligned);
    }
    std::uint32_t emitted = 0;
    const auto sink = [&]([[maybe_unused]] auto k, [[maybe_unused]] S re,
                          [[maybe_unused]] S im) { ++emitted; };
    if constexpr (Terminal) {
        dif_butterfly_terminal<T, IP, S>(xr, xi, sink);
    } else {
        dif_butterfly<T, IP, S>(xr, xi, sink);
    }
    S::bind(nullptr);
    pool.tag_turns(fold_set<T>(ds));
    dag::registry().push_back(
        dag::rung{family, arm_name(Terminal ? terminal_arm(IP) : dif_arm(IP)),
                  dag::provenance::aligned, prec_char<T>(), static_cast<std::uint32_t>(IP),
                  static_cast<std::uint32_t>(S::size), 1u, pool.op_count()});
    return dif_row{pool.op_count(), pool.created(), pool.intern_dups(), emitted,
                   pool.op_hist()};
}

void print_row(const char* tag, std::size_t n, const char* arm, const dif_row& r,
               long oracle) {
    const auto h = [&](dag::op o) {
        return static_cast<unsigned long long>(r.hist[static_cast<std::size_t>(o)]);
    };
    const double cse =
        r.created != 0 ? static_cast<double>(r.dups) / static_cast<double>(r.created) : 0.0;
    std::printf(
        "%-8s n=%-3zu arm=%-9s ops=%-5llu oracle=%-5ld emits=%-3u add=%llu sub=%llu neg=%llu "
        "mul=%llu fma=%llu fnma=%llu cse=%.3f\n",
        tag, n, arm, static_cast<unsigned long long>(r.ops), oracle, r.emitted, h(dag::op::add),
        h(dag::op::sub), h(dag::op::neg), h(dag::op::mul), h(dag::op::fma), h(dag::op::fnma),
        cse);
}

// ---------------------------------------------------------------------------
// kernel_batched tracer. One case = one pool + its input plumbing + captured outputs.
// ---------------------------------------------------------------------------

template <typename T, unsigned N>
struct kb_case {
    dag::pool<T> pool;
    dag::input_map<T> map;
    std::vector<T> raw_re;
    std::vector<T> raw_im;
    std::array<dag::sym<T>, N> xr;
    std::array<dag::sym<T>, N> xi;
    std::array<dag::sym<T>, N> yr;
    std::array<dag::sym<T>, N> yi;
    std::uint64_t unmapped = 0;
    std::uint64_t ops = 0;
};

template <typename T, unsigned N, bool Forward, std::size_t... Dens>
void kb_run(kb_case<T, N>& kc, dag::provenance prov, dens_list<Dens...> ds) {
    using S = dag::sym<T>;
    dag::sym_ctx<T> ctx;
    ctx.p = &kc.pool;
    ctx.m = &kc.map;
    S::bind(&ctx);
    if (prov == dag::provenance::aligned) {
        for (std::size_t i = 0; i < N; ++i) {
            kc.xr[i] = S::input(dag::plane::re, dag::layout::soa,
                                static_cast<std::uint32_t>(i), dag::provenance::aligned);
            kc.xi[i] = S::input(dag::plane::im, dag::layout::soa,
                                static_cast<std::uint32_t>(i), dag::provenance::aligned);
        }
    } else {
        kc.raw_re.assign(N, T(0));
        kc.raw_im.assign(N, T(0));
        for (std::size_t i = 0; i < N; ++i) {
            kc.map.emplace(kc.raw_re.data() + i,
                           dag::input_slot{dag::plane::re, dag::layout::soa,
                                           static_cast<std::uint32_t>(i), prov});
            kc.map.emplace(kc.raw_im.data() + i,
                           dag::input_slot{dag::plane::im, dag::layout::soa,
                                           static_cast<std::uint32_t>(i), prov});
        }
        if (prov == dag::provenance::unaligned) {
            for (std::size_t i = 0; i < N; ++i) {
                kc.xr[i] = S::load_unaligned(kc.raw_re.data() + i);
                kc.xi[i] = S::load_unaligned(kc.raw_im.data() + i);
            }
        } else {
            const xsimd::batch_bool<T> mask(true);
            for (std::size_t i = 0; i < N; ++i) {
                kc.xr[i] = S::load(kc.raw_re.data() + i, mask, xsimd::unaligned_mode{});
                kc.xi[i] = S::load(kc.raw_im.data() + i, mask, xsimd::unaligned_mode{});
            }
        }
    }
    kernel_batched<N, T, Forward, S>::apply(kc.xr.data(), kc.xi.data(), 1, kc.yr.data(),
                                            kc.yi.data());
    S::bind(nullptr);
    kc.unmapped = ctx.unmapped_loads;
    kc.pool.tag_turns(fold_set<T>(ds));
    kc.ops = kc.pool.op_count();
    dag::registry().push_back(dag::rung{"kernel_batched", Forward ? "fwd" : "inv", prov,
                                        prec_char<T>(), N, static_cast<std::uint32_t>(S::size),
                                        1u, kc.ops});
}

// Direction normalization proof: the inverse specialisation swaps the re/im POINTERS, so
// the inverse DAG under the plane-relabel view must be the forward DAG. inv.yi[k] holds
// the text's re-plane output and inv.yr[k] its im-plane output.
template <typename T, unsigned N>
void require_direction_equal(const kb_case<T, N>& fwd, const kb_case<T, N>& inv) {
    REQUIRE(fwd.ops == inv.ops);
    {
        dag::canonizer cz;
        const auto fc = cz.run(fwd.pool, dag::level::l1, dag::canon_view{});
        const auto ic = cz.run(inv.pool, dag::level::l1, dag::canon_view{true, false});
        for (std::size_t k = 0; k < N; ++k) {
            INFO("L1 canon mismatch at k=" << k);
            CHECK(fc[fwd.yr[k].id()] == ic[inv.yi[k].id()]);
            CHECK(fc[fwd.yi[k].id()] == ic[inv.yr[k].id()]);
        }
    }
    {
        dag::canonizer cz;
        const auto fc = cz.run(fwd.pool, dag::level::l2, dag::canon_view{});
        const auto ic = cz.run(inv.pool, dag::level::l2, dag::canon_view{true, false});
        for (std::size_t k = 0; k < N; ++k) {
            INFO("L2 canon mismatch at k=" << k);
            CHECK(fc[fwd.yr[k].id()] == ic[inv.yi[k].id()]);
            CHECK(fc[fwd.yi[k].id()] == ic[inv.yr[k].id()]);
        }
    }
}

// Same-role comparison for the provenance arms (all forward): one view each, no relabel.
template <typename T, unsigned N>
void require_provenance_equal(const kb_case<T, N>& a, const kb_case<T, N>& b) {
    REQUIRE(a.ops == b.ops);
    {
        dag::canonizer cz;
        const auto ca = cz.run(a.pool, dag::level::l1, dag::canon_view{});
        const auto cb = cz.run(b.pool, dag::level::l1, dag::canon_view{});
        for (std::size_t k = 0; k < N; ++k) {
            INFO("L1 provenance canon mismatch at k=" << k);
            CHECK(ca[a.yr[k].id()] == cb[b.yr[k].id()]);
            CHECK(ca[a.yi[k].id()] == cb[b.yi[k].id()]);
        }
    }
    {
        dag::canonizer cz;
        const auto ca = cz.run(a.pool, dag::level::l2, dag::canon_view{});
        const auto cb = cz.run(b.pool, dag::level::l2, dag::canon_view{});
        for (std::size_t k = 0; k < N; ++k) {
            INFO("L2 provenance canon mismatch at k=" << k);
            CHECK(ca[a.yr[k].id()] == cb[b.yr[k].id()]);
            CHECK(ca[a.yi[k].id()] == cb[b.yi[k].id()]);
        }
    }
}

// ---------------------------------------------------------------------------
// Granule dialect tracers. Inputs are AoS granule leaves (one register per granule
// position); the constant pools resolve through ctx.read_through to real rodata values,
// so a uniform lane pool collapses to a broadcast constant and the alternating {-s,+s}
// pools keep one full-lane-hash cplain.
// ---------------------------------------------------------------------------

struct gran_stats {
    std::uint64_t fp = 0;
    std::uint64_t add = 0;
    std::uint64_t sub = 0;
    std::uint64_t neg = 0;
    std::uint64_t mul = 0;
    std::uint64_t fma = 0;
    std::uint64_t swz = 0;
    std::uint64_t shfl = 0;
    std::uint64_t flip = 0;
    std::uint64_t lifts = 0;
    std::uint64_t sunk = 0;
    std::uint64_t unmapped = 0;
};

template <typename T>
gran_stats stats_of(const dag::pool<T>& pool, const dag::sym_ctx<T>& ctx) {
    gran_stats s;
    const auto h = [&](dag::op o) { return pool.op_hist()[static_cast<std::size_t>(o)]; };
    s.fp = pool.op_count();
    s.add = h(dag::op::add);
    s.sub = h(dag::op::sub);
    s.neg = h(dag::op::neg);
    s.mul = h(dag::op::mul);
    s.fma = h(dag::op::fma);
    s.swz = h(dag::op::swz);
    s.shfl = h(dag::op::shfl);
    s.flip = h(dag::op::flip);
    s.unmapped = ctx.unmapped_loads;
    return s;
}

template <typename T, std::size_t IP, bool Fwd, std::size_t... Dens>
gran_stats trace_granule_sym(const char* family, dens_list<Dens...> ds) {
    using S = dag::sym<T>;
    dag::pool<T> pool;
    dag::sym_ctx<T> ctx;
    ctx.p = &pool;
    ctx.read_through = true;
    S::bind(&ctx);
    S x[IP], y[IP];
    for (std::size_t i = 0; i < IP; ++i)
        x[i] = S::input(dag::plane::re, dag::layout::aos, static_cast<std::uint32_t>(i),
                        dag::provenance::aligned);
    granule_radix_sym<T, IP, S, Fwd>(x, y);
    S::bind(nullptr);
    pool.tag_turns(fold_set<T>(ds));
    gran_stats s = stats_of(pool, ctx);
    s.lifts = ctx.lift_next;
    s.sunk = ctx.sunk;
    dag::registry().push_back(dag::rung{family, Fwd ? "fwd" : "inv",
                                        dag::provenance::aligned, prec_char<T>(),
                                        static_cast<std::uint32_t>(IP),
                                        static_cast<std::uint32_t>(S::size), 1u, s.fp});
    return s;
}

void print_gran(const char* tag, std::size_t n, const char* arm, const gran_stats& s) {
    std::printf("%-8s n=%-3zu arm=%-9s fp=%llu add=%llu sub=%llu neg=%llu mul=%llu "
                "fma=%llu swz=%llu shfl=%llu flip=%llu lifts=%llu sunk=%llu unmapped=%llu\n",
                tag, n, arm, static_cast<unsigned long long>(s.fp),
                static_cast<unsigned long long>(s.add), static_cast<unsigned long long>(s.sub),
                static_cast<unsigned long long>(s.neg), static_cast<unsigned long long>(s.mul),
                static_cast<unsigned long long>(s.fma), static_cast<unsigned long long>(s.swz),
                static_cast<unsigned long long>(s.shfl),
                static_cast<unsigned long long>(s.flip),
                static_cast<unsigned long long>(s.lifts),
                static_cast<unsigned long long>(s.sunk),
                static_cast<unsigned long long>(s.unmapped));
}

const char* root_form_name(root_form f) noexcept {
    switch (f) {
        case root_form::one: return "one";
        case root_form::neg_one: return "neg_one";
        case root_form::neg_i: return "neg_i";
        case root_form::pos_i: return "pos_i";
        case root_form::diag: return "diag";
        case root_form::anti_diag: return "anti_diag";
        case root_form::generic: return "generic";
    }
    return "?";  // unreachable: the switch is exhaustive over root_form
}

// Expected hash of the granule_flip integer mask at one parity, recomputed from the
// shipped construction rule (top bit on lanes Parity, Parity+2, ...).
template <typename T, std::size_t W>
std::uint64_t candidate_flip_mask(std::size_t parity) {
    using U = xsimd::as_unsigned_integer_t<T>;
    constexpr unsigned top = sizeof(U) * 8 - 1;
    std::array<U, W> m{};
    for (std::size_t i = parity; i < W; i += 2) m[i] = static_cast<U>(U(1) << top);
    return dag::fnv1a64(m.data(), W * sizeof(U));
}

struct stage_row {
    std::size_t n = 0;
    std::size_t ip = 0;
    bool fwd = false;
    root_form form = root_form::generic;
    gran_stats s;
    std::uint64_t flip_mask = 0;
};

template <typename T, std::size_t IP, std::size_t N, bool Fwd>
stage_row trace_stage() {
    using S = dag::sym<T>;
    dag::pool<T> pool;
    dag::sym_ctx<T> ctx;
    ctx.p = &pool;
    ctx.read_through = true;
    S::bind(&ctx);
    const S u = S::input(dag::plane::re, dag::layout::aos, 0, dag::provenance::aligned);
    [[maybe_unused]] const S v = granule_stage_twiddle<T, IP, N, S, Fwd>(u);
    S::bind(nullptr);
    pool.tag_turns(fold_set<T>(dens<IP>));
    stage_row r;
    r.n = N;
    r.ip = IP;
    r.fwd = Fwd;
    r.form = ct_root<N, IP, Fwd>();
    r.s = stats_of(pool, ctx);
    for (const dag::node& nd : pool.nodes())
        if (nd.o == dag::op::flip) r.flip_mask = nd.bits;
    dag::registry().push_back(dag::rung{"granule_stage_twiddle", root_form_name(r.form),
                                        dag::provenance::aligned, prec_char<T>(),
                                        static_cast<std::uint32_t>(IP),
                                        static_cast<std::uint32_t>(S::size), 1u, r.s.fp});
    return r;
}

template <typename T, std::size_t IP, std::size_t... Ns>
void trace_stage_den(std::vector<stage_row>& out, std::index_sequence<Ns...>) {
    ((out.push_back(trace_stage<T, IP, Ns, true>()),
      out.push_back(trace_stage<T, IP, Ns, false>())),
     ...);
}

// Per-row net census of one granule row tile. f32 exercises the real-batch transpose +
// lift seam; f64 keeps the shuffle net inside the dialect.
template <typename T, unsigned N, bool Forward>
struct tile_case {
    dag::pool<T> pool;
    gran_stats s;
};

template <typename T, unsigned N, bool Forward>
void trace_row_tile(tile_case<T, N, Forward>& tc) {
    using S = dag::sym<T>;
    constexpr std::size_t W = S::size;
    constexpr std::size_t GL = W / 2;
    dag::sym_ctx<T> ctx;
    ctx.p = &tc.pool;
    ctx.read_through = true;
    dag::input_map<T> imap;
    const std::size_t in_stride = N;
    std::vector<T> in(2 * GL * in_stride, T(0));
    std::vector<T> out(2 * GL * in_stride, T(0));
    constexpr std::size_t DB = sizeof(T) == 4 ? 1 : 2;
    if (sizeof(T) == 8) {
        // The tile reads B::load_unaligned(in + l * 2 * in_stride + qb * DB * GL); those
        // are input granule blocks, registered by address (f32 keeps its loads in real
        // double batches, so it never reaches the map).
        for (std::size_t l = 0; l < GL; ++l)
            for (std::size_t qb = 0; qb < (N + GL - 1) / GL; ++qb)
                imap.emplace(in.data() + l * 2 * in_stride + qb * DB * GL,
                             dag::input_slot{dag::plane::re, dag::layout::aos,
                                             static_cast<std::uint32_t>(l * N + qb * GL),
                                             dag::provenance::unaligned});
        ctx.m = &imap;
    }
    S::bind(&ctx);
    granule_row_tile<N, T, Forward, false, S>(in.data(), out.data(), in_stride, in_stride,
                                              T(1));
    S::bind(nullptr);
    tc.pool.tag_turns(fold_set<T>(dens<3, 4, N>));
    tc.s = stats_of(tc.pool, ctx);
    tc.s.lifts = ctx.lift_next;
    tc.s.sunk = ctx.sunk;
    dag::registry().push_back(dag::rung{"granule_row_tile", Forward ? "fwd" : "inv",
                                        dag::provenance::aligned, prec_char<T>(), N,
                                        static_cast<std::uint32_t>(S::size), 1u, tc.s.fp});
}

// ---------------------------------------------------------------------------
// radix_butterfly_ct chunk ladder (codelet.hpp): one full-width rung (nfull trips), the
// sized-batch cover of rem's set bits, and the scalar tail. The ladder arithmetic and the
// per-rung existence test are read off the shipped constexprs (batch size, bit_width,
// make_sized_batch_t, min_sized_tail_width); the enumeration's coverage REQUIREs below
// pin it against the shipped scheduler.
// ---------------------------------------------------------------------------

template <typename T, std::size_t Wt>
constexpr bool sized_exists() {
    return Wt >= 2 && !std::is_void_v<xsimd::make_sized_batch_t<T, Wt>>;
}

struct rung_row {
    std::uint32_t width;
    std::uint32_t trips;
    std::uint64_t j0;
};

const char* rung_name(std::uint32_t w) noexcept {
    switch (w) {
        case 16: return "w16";
        case 8: return "w8";
        case 4: return "w4";
        case 2: return "w2";
        case 1: return "s1";
        default: return "w?";
    }
}

template <typename T, unsigned R, unsigned N>
struct chunk_ladder {
    static constexpr std::size_t M = N / R;
    static constexpr std::size_t W = xsimd::batch<T>::size;
    static constexpr std::size_t nfull = M / W;
    static constexpr std::size_t rem = M - nfull * W;
    static constexpr std::size_t minw = min_sized_tail_width<T>();
    static constexpr std::size_t nscal = rem & (minw - 1);
    static constexpr std::size_t jscal = nfull * W + (rem & ~(minw - 1));

    static std::vector<rung_row> rungs() {
        std::vector<rung_row> out;
        if constexpr (nfull > 0)
            out.push_back(rung_row{static_cast<std::uint32_t>(W),
                                   static_cast<std::uint32_t>(nfull), 0});
        rungs_sized(out, std::make_index_sequence<static_cast<std::size_t>(bit_width(W)) -
                                                   1>{});
        if constexpr (nscal > 0)
            out.push_back(rung_row{1u, static_cast<std::uint32_t>(nscal), jscal});
        return out;
    }

private:
    template <std::size_t... S>
    static void rungs_sized(std::vector<rung_row>& out, std::index_sequence<S...>) {
        (push_sized_rung<S>(out), ...);
    }
    template <std::size_t S>
    static void push_sized_rung(std::vector<rung_row>& out) {
        constexpr std::size_t Wt = W >> (S + 1);
        if constexpr (sized_exists<T, Wt>() && (rem & Wt) != 0)
            out.push_back(rung_row{static_cast<std::uint32_t>(Wt), 1u,
                                   nfull * W + (rem & ~(2 * Wt - 1))});
    }
};

// The stage-twiddle op census per shipped root form (apply_stage_twiddle arms, scalar and
// batched alike: zero, two, one, or six raw ops).
constexpr std::size_t stage_arm_ops(root_form f) noexcept {
    switch (f) {
        case root_form::one: return 0;
        case root_form::neg_one: return 2;
        case root_form::neg_i: return 1;
        case root_form::pos_i: return 1;
        case root_form::diag: return 4;
        case root_form::anti_diag: return 4;
        case root_form::generic: return 6;
    }
    return 0;  // unreachable: the switch is exhaustive over root_form
}

constexpr std::size_t scalar_twiddle_ops(std::size_t R, std::size_t N, std::size_t J) noexcept {
    std::size_t t = 0;
    for (std::size_t q = 1; q < R; ++q) t += stage_arm_ops(ct_root_form((q * J) % N, N, true));
    return t;
}

// One radix_butterfly_ct (R, N) tree, traced chunk by chunk. Inputs are double-entry:
// real T buffers + the pointer map for the batched chunks' V::load_unaligned, sym arrays
// with the same (plane, index) tags for the scalar chunk's direct slot reads; the static
// twiddle tables resolve through read-through and tag by bits at den {R, N}.
template <typename T, unsigned R, unsigned N, std::size_t... Dens>
struct rb_case {
    static constexpr std::size_t M = chunk_ladder<T, R, N>::M;
    dag::pool<T> pool;
    dag::input_map<T> map;
    std::vector<T> fv_re = std::vector<T>(N, T(0));
    std::vector<T> fv_im = std::vector<T>(N, T(0));
    std::array<dag::sym<T>, N> sv_re;
    std::array<dag::sym<T>, N> sv_im;
    std::array<dag::sym<T>, N> o_re;
    std::array<dag::sym<T>, N> o_im;
    std::vector<char> covered = std::vector<char>(N, 0);
    std::size_t cur_width = 1;  // lanes the chunk in flight covers per emit
    std::size_t tagged = 0;
};

struct chunk_tick {
    std::uint32_t width;
    std::uint64_t j;
    std::uint64_t ops;  // fp ops this chunk added
};

template <typename T, unsigned R, unsigned N, std::size_t... Dens>
struct rb_trace {
    std::vector<chunk_tick> ticks;
    rb_case<T, R, N, Dens...> c;
};

template <typename T, unsigned R, unsigned N, std::size_t J, std::size_t... Dens,
          typename Sink, std::size_t... I>
void scalar_tail(rb_trace<T, R, N, Dens...>& t, const Sink& sink, std::index_sequence<I...>) {
    using S = dag::sym<T>;
    (([&] {
         constexpr std::size_t Jj = J + I;
         const std::uint64_t before = t.c.pool.op_count();
         bfly_chunk_scalar_ct<R, N, Jj, S>(t.c.sv_re.data(), t.c.sv_im.data(), sink);
         t.ticks.push_back(chunk_tick{1u, Jj, t.c.pool.op_count() - before});
     }()),
     ...);
}

template <typename T, unsigned R, unsigned N, std::size_t... Dens>
void rb_run(rb_trace<T, R, N, Dens...>& t, dens_list<Dens...> ds) {
    using S = dag::sym<T>;
    using L = chunk_ladder<T, R, N>;
    constexpr std::size_t M = L::M;
    constexpr auto twre = make_twiddle_table<N, R, T, false>();
    constexpr auto twim = make_twiddle_table<N, R, T, true>();
    rb_case<T, R, N, Dens...>& c = t.c;
    dag::sym_ctx<T> ctx;
    ctx.p = &c.pool;
    ctx.m = &c.map;
    ctx.read_through = true;
    S::bind(&ctx);
    for (std::size_t p = 0; p < N; ++p) {
        c.map.emplace(c.fv_re.data() + p,
                      dag::input_slot{dag::plane::re, dag::layout::soa,
                                      static_cast<std::uint32_t>(p), dag::provenance::unaligned});
        c.map.emplace(c.fv_im.data() + p,
                      dag::input_slot{dag::plane::im, dag::layout::soa,
                                      static_cast<std::uint32_t>(p), dag::provenance::unaligned});
        c.sv_re[p] = S::input(dag::plane::re, dag::layout::soa, static_cast<std::uint32_t>(p),
                              dag::provenance::aligned);
        c.sv_im[p] = S::input(dag::plane::im, dag::layout::soa, static_cast<std::uint32_t>(p),
                              dag::provenance::aligned);
    }
    const auto sink = [&](std::size_t p, S outr, S outi) {
        // One emit writes cur_width lanes at p .. p + cur_width (the store side of
        // bfly_chunk). Overlap or a hole at the tail both fail the partition check below.
        for (std::size_t q = p; q < p + c.cur_width; ++q) {
            REQUIRE(q < N);
            REQUIRE(c.covered[q] == 0);
            c.covered[q] = 1;
        }
        c.o_re[p] = outr;
        c.o_im[p] = outi;
    };
    // Full-width rung: nfull trips of the same chunk, traced every trip.
    if constexpr (L::nfull > 0) {
        c.cur_width = L::W;
        for (std::size_t b = 0; b < L::nfull; ++b) {
            const std::uint64_t before = c.pool.op_count();
            bfly_chunk<R, T, S>(twre.data(), twim.data(), c.fv_re.data(), c.fv_im.data(), M,
                                b * L::W, sink);
            t.ticks.push_back(chunk_tick{static_cast<std::uint32_t>(L::W), b * L::W,
                                         c.pool.op_count() - before});
        }
    }
    // Sized-batch rungs, the shipped cover rule over rem's set bits.
    for (const rung_row& rr : L::rungs()) {
        if (rr.width == L::W || rr.width == 1) continue;  // full and scalar handled aside
        c.cur_width = rr.width;
        const std::uint64_t before = c.pool.op_count();
        bfly_chunk<R, T, S>(twre.data(), twim.data(), c.fv_re.data(), c.fv_im.data(), M,
                            rr.j0, sink);
        t.ticks.push_back(chunk_tick{rr.width, rr.j0, c.pool.op_count() - before});
    }
    c.cur_width = 1;
    // Scalar tail, one compile-time chunk per leftover column (bfly_chunk_scalar_ct needs
    // J as a constant). The chunk runs T = sym<T>: its constants fold at ct_real_t<T>,
    // which is double for both precisions, so the tagged bits match the batched rungs'.
    if constexpr (L::nscal > 0)
        scalar_tail<T, R, N, L::jscal>(t, sink, std::make_index_sequence<L::nscal>{});
    S::bind(nullptr);
    c.tagged = c.pool.tag_turns(fold_set<T>(ds));
    // The (q*M+j) read map and (L*M+j) write map partition [0, N) exactly.
    for (std::size_t p = 0; p < N; ++p)
        REQUIRE(c.covered[p] != 0);
    REQUIRE(ctx.unmapped_loads == 0u);
}

}  // namespace

TEST_CASE("dag census: dif_butterfly odd arm matches sym_dft_ops", "[dag][census]") {
    dag::sym<float>::reset_counters();
    // The odd arm is odd_ct_split == {0,0}: pure sym radices AND distinct-prime odd
    // composites (15, 21, 33, 35, 45) file here, not under ct.
    static_assert(dif_arm(3) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(5) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(7) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(9) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(11) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(13) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(15) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(21) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(33) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(35) == dif_arm_kind::radix_sym);
    static_assert(dif_arm(45) == dif_arm_kind::radix_sym);

    const auto check = [](std::size_t n, const dif_row& r) {
        const long oracle = static_cast<long>(sym_ops_oracle(n));
        print_row("dif odd", n, "radix_sym", r, oracle);
        CHECK(r.emitted == n);
        REQUIRE(r.ops == static_cast<std::uint64_t>(oracle));
    };
    check(3, trace_butterfly<float, 3, false>("dif_butterfly", dens<3>));
    check(5, trace_butterfly<float, 5, false>("dif_butterfly", dens<5>));
    check(7, trace_butterfly<float, 7, false>("dif_butterfly", dens<7>));
    check(9, trace_butterfly<float, 9, false>("dif_butterfly", dens<9>));
    check(11, trace_butterfly<float, 11, false>("dif_butterfly", dens<11>));
    check(13, trace_butterfly<float, 13, false>("dif_butterfly", dens<13>));
    check(15, trace_butterfly<float, 15, false>("dif_butterfly", dens<15>));
    check(21, trace_butterfly<float, 21, false>("dif_butterfly", dens<21>));
    check(33, trace_butterfly<float, 33, false>("dif_butterfly", dens<33>));
    check(35, trace_butterfly<float, 35, false>("dif_butterfly", dens<35>));
    check(45, trace_butterfly<float, 45, false>("dif_butterfly", dens<45>));
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
}

TEST_CASE("dag census: dif_butterfly ct arm matches ct_dft_ops", "[dag][census]") {
    dag::sym<float>::reset_counters();
    // Only pure odd prime powers reach this arm; the splits are the shipped constexpr's
    // own answers.
    static_assert(odd_ct_split(25) == std::pair<std::size_t, std::size_t>{5, 5});
    static_assert(odd_ct_split(27) == std::pair<std::size_t, std::size_t>{3, 9});
    static_assert(odd_ct_split(49) == std::pair<std::size_t, std::size_t>{7, 7});
    static_assert(dif_arm(25) == dif_arm_kind::ct);
    static_assert(dif_arm(27) == dif_arm_kind::ct);
    static_assert(dif_arm(49) == dif_arm_kind::ct);

    const auto check = [](std::size_t n, std::size_t n1, std::size_t n2, const dif_row& r) {
        const long oracle = static_cast<long>(ct_ops_oracle(n1, n2));
        print_row("dif ct", n, "ct", r, oracle);
        CHECK(r.emitted == n);
        REQUIRE(r.ops == static_cast<std::uint64_t>(oracle));
    };
    check(25, 5, 5, trace_butterfly<float, 25, false>("dif_butterfly", dens<5, 25>));
    check(27, 3, 9, trace_butterfly<float, 27, false>("dif_butterfly", dens<3, 9, 27>));
    check(49, 7, 7, trace_butterfly<float, 49, false>("dif_butterfly", dens<7, 49>));
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
}

TEST_CASE("dag census: dif_butterfly_terminal odd PFA matches the split oracles",
          "[dag][census]") {
    dag::sym<float>::reset_counters();
    static_assert(terminal_arm(15) == dif_arm_kind::pfa);
    static_assert(terminal_arm(21) == dif_arm_kind::pfa);
    static_assert(terminal_arm(35) == dif_arm_kind::pfa);
    static_assert(coprime_split(15) == std::pair<std::size_t, std::size_t>{5, 3});
    static_assert(coprime_split(21) == std::pair<std::size_t, std::size_t>{7, 3});
    static_assert(coprime_split(35) == std::pair<std::size_t, std::size_t>{7, 5});

    const auto check = [](std::size_t n, std::size_t n1, std::size_t n2, const dif_row& r) {
        const long oracle = static_cast<long>(pfa_ops_oracle(n1, n2));
        print_row("term pfa", n, "pfa", r, oracle);
        CHECK(r.emitted == n);
        REQUIRE(r.ops == static_cast<std::uint64_t>(oracle));
    };
    check(15, 5, 3, trace_butterfly<float, 15, true>("dif_terminal", dens<3, 5>));
    check(21, 7, 3, trace_butterfly<float, 21, true>("dif_terminal", dens<3, 7>));
    check(35, 7, 5, trace_butterfly<float, 35, true>("dif_terminal", dens<5, 7>));
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
}

TEST_CASE("dag census: kernel_batched direction normalization", "[dag][census]") {
    dag::sym<float>::reset_counters();
    dag::sym<double>::reset_counters();
    // kernel_batched::<N, T, false> swaps the re/im pointers and reruns the forward text,
    // so the two DAGs must agree after a plane relabel. Node counts must also agree.
    {
        kb_case<float, 15> f, i;
        kb_run<float, 15, true>(f, dag::provenance::aligned, dens<3, 5, 15>);
        kb_run<float, 15, false>(i, dag::provenance::aligned, dens<3, 5, 15>);
        require_direction_equal(f, i);
    }
    {
        kb_case<double, 15> f, i;
        kb_run<double, 15, true>(f, dag::provenance::aligned, dens<3, 5, 15>);
        kb_run<double, 15, false>(i, dag::provenance::aligned, dens<3, 5, 15>);
        require_direction_equal(f, i);
    }
    {
        kb_case<float, 21> f, i;
        kb_run<float, 21, true>(f, dag::provenance::aligned, dens<3, 7, 21>);
        kb_run<float, 21, false>(i, dag::provenance::aligned, dens<3, 7, 21>);
        require_direction_equal(f, i);
    }
    {
        kb_case<double, 21> f, i;
        kb_run<double, 21, true>(f, dag::provenance::aligned, dens<3, 7, 21>);
        kb_run<double, 21, false>(i, dag::provenance::aligned, dens<3, 7, 21>);
        require_direction_equal(f, i);
    }
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
    REQUIRE(dag::sym<double>::uninit_reads() == 0u);
}

TEST_CASE("dag census: kernel_batched provenance arms are canon-equal", "[dag][census]") {
    dag::sym<float>::reset_counters();
    dag::sym<double>::reset_counters();
    // The three input flavors the cofactor arm can hand kernel_batched: direct aligned
    // arrays, unaligned loads, masked loads. Provenance never enters canonical identity.
    {
        kb_case<float, 15> a, u, m;
        kb_run<float, 15, true>(a, dag::provenance::aligned, dens<3, 5, 15>);
        kb_run<float, 15, true>(u, dag::provenance::unaligned, dens<3, 5, 15>);
        kb_run<float, 15, true>(m, dag::provenance::masked, dens<3, 5, 15>);
        REQUIRE(u.unmapped == 0u);
        REQUIRE(m.unmapped == 0u);
        require_provenance_equal(a, u);
        require_provenance_equal(a, m);
    }
    {
        kb_case<double, 15> a, u, m;
        kb_run<double, 15, true>(a, dag::provenance::aligned, dens<3, 5, 15>);
        kb_run<double, 15, true>(u, dag::provenance::unaligned, dens<3, 5, 15>);
        kb_run<double, 15, true>(m, dag::provenance::masked, dens<3, 5, 15>);
        REQUIRE(u.unmapped == 0u);
        REQUIRE(m.unmapped == 0u);
        require_provenance_equal(a, u);
        require_provenance_equal(a, m);
    }
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
    REQUIRE(dag::sym<double>::uninit_reads() == 0u);
}

TEST_CASE("dag census: granule_radix_sym halves the SoA plane-op count", "[dag][census]") {
    dag::sym<float>::reset_counters();
    dag::sym<double>::reset_counters();
    // The granule dialect replaces each SoA re/im op pair with one granule op: the
    // radix-3 census halves 14 to 7. Tracing devices make_batch_constant over
    // sym::arch_type = xsimd::default_arch (the real arch): the swizzle net folds by
    // construction, which the swz count below witnesses.
    const gran_stats g3f = trace_granule_sym<float, 3, true>("granule_radix_sym", dens<3>);
    print_gran("gran sym", 3, "fwd f32", g3f);
    REQUIRE(g3f.fp == 7u);
    REQUIRE(g3f.swz == 1u);
    REQUIRE(g3f.flip == 0u);
    REQUIRE(g3f.unmapped == 0u);
    if constexpr (kFusedFma) {
        // The halving fact is stated against the fused SoA text; at kFusedFma the SoA
        // census of radix 3 is 14 = sym_ops_oracle(3). (At ISA levels without an FMA unit
        // the SoA text expands to 18 while the granule text still records its fma.)
        REQUIRE(2 * g3f.fp == sym_ops_oracle(3));
    }
    const gran_stats g3d = trace_granule_sym<double, 3, true>("granule_radix_sym", dens<3>);
    print_gran("gran sym", 3, "fwd f64", g3d);
    CHECK(g3d.fp == g3f.fp);
    const gran_stats g3di = trace_granule_sym<double, 3, false>("granule_radix_sym", dens<3>);
    print_gran("gran sym", 3, "inv f64", g3di);
    // Direction folds into the constants in this dialect, so the op census is invariant.
    CHECK(g3di.fp == g3d.fp);
    CHECK(g3di.flip == g3d.flip);
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
    REQUIRE(dag::sym<double>::uninit_reads() == 0u);
}

TEST_CASE("dag census: granule_stage_twiddle arms follow the ct_root descriptor",
          "[dag][census]") {
    dag::sym<double>::reset_counters();
    // Lattice over (N, IP, Fwd); the expected arm is the shipped ct_root<N,IP,Fwd>()
    // answer, the measured arm is the op signature the trace recorded. diag folds into
    // the generic arm structurally (pool {-c,+c} == {-s,+s} at c == s): one 3-node arm.
    std::vector<stage_row> rows;
    trace_stage_den<double, 3>(rows, std::make_index_sequence<3>{});
    trace_stage_den<double, 4>(rows, std::make_index_sequence<4>{});
    trace_stage_den<double, 5>(rows, std::make_index_sequence<5>{});
    trace_stage_den<double, 8>(rows, std::make_index_sequence<8>{});
    trace_stage_den<double, 9>(rows, std::make_index_sequence<9>{});
    std::uint64_t bad = 0;
    for (const stage_row& r : rows) {
        bool ok = false;
        switch (r.form) {
            case root_form::one:
                ok = r.s.fp == 0 && r.s.swz == 0 && r.s.flip == 0;
                break;
            case root_form::neg_one:
                ok = r.s.fp == 1 && r.s.neg == 1 && r.s.swz == 0 && r.s.flip == 0;
                break;
            case root_form::neg_i:
                ok = r.s.fp == 0 && r.s.swz == 1 && r.s.flip == 1 &&
                     r.flip_mask == candidate_flip_mask<double, dag::sym<double>::size>(1);
                break;
            case root_form::pos_i:
                ok = r.s.fp == 0 && r.s.swz == 1 && r.s.flip == 1 &&
                     r.flip_mask == candidate_flip_mask<double, dag::sym<double>::size>(0);
                break;
            case root_form::diag:
            case root_form::anti_diag:
            case root_form::generic:
                ok = r.s.fp == 2 && r.s.swz == 1 && r.s.mul == 1 && r.s.fma == 1 &&
                     r.s.flip == 0;
                break;
        }
        if (!ok) {
            ++bad;
            INFO("stage arm mismatch (N,IP,Fwd)=(" << r.n << ", " << r.ip << ", " << r.fwd
                    << ") form=" << root_form_name(r.form) << " fp=" << r.s.fp
                    << " neg=" << r.s.neg << " swz=" << r.s.swz << " flip=" << r.s.flip
                    << " mul=" << r.s.mul << " fma=" << r.s.fma);
        }
    }
    REQUIRE(bad == 0u);
    // Every shipped form saw at least one lattice point on this sweep.
    const auto seen = [&](root_form f) {
        std::uint64_t n = 0;
        for (const stage_row& r : rows) n += r.form == f;
        return n;
    };
    CHECK(seen(root_form::one) > 0u);
    CHECK(seen(root_form::neg_one) > 0u);
    CHECK(seen(root_form::neg_i) > 0u);
    CHECK(seen(root_form::pos_i) > 0u);
    CHECK(seen(root_form::diag) > 0u);
    CHECK(seen(root_form::anti_diag) > 0u);
    CHECK(seen(root_form::generic) > 0u);
    std::printf("stage ladder: %zu rows over IP in {3,4,5,8,9} x Fwd; one=%llu neg1=%llu "
                "negi=%llu posi=%llu diag=%llu anti=%llu generic=%llu\n",
                rows.size(),
                static_cast<unsigned long long>(seen(root_form::one)),
                static_cast<unsigned long long>(seen(root_form::neg_one)),
                static_cast<unsigned long long>(seen(root_form::neg_i)),
                static_cast<unsigned long long>(seen(root_form::pos_i)),
                static_cast<unsigned long long>(seen(root_form::diag)),
                static_cast<unsigned long long>(seen(root_form::anti_diag)),
                static_cast<unsigned long long>(seen(root_form::generic)));
    REQUIRE(dag::sym<double>::uninit_reads() == 0u);
}

TEST_CASE("dag census: granule row-tile nets and the f32 lift seam", "[dag][report]") {
    dag::sym<float>::reset_counters();
    dag::sym<double>::reset_counters();
    if constexpr (dag::sym<double>::size >= 4) {
        tile_case<double, 12, true> t;
        trace_row_tile(t);
        print_gran("gran tile", 12, "fwd f64", t.s);
        REQUIRE(t.s.unmapped == 0u);
        CHECK(t.s.fp > 0u);
        CHECK(t.s.flip > 0u);   // granule_pfa(4,3)'s pow2 side carries neg_i arms
        CHECK(t.s.shfl > 0u);   // f64 128-bit-granule two-source shuffle net
    } else {
        // Below one granule per f64 register the shipped tile asserts GL >= 2; the trace
        // respects the same boundary instead of instantiating an illegal tile.
        std::printf("gran tile f64 skipped: GL < 2 at this arch\n");
    }
    {
        tile_case<float, 12, true> t;
        trace_row_tile(t);
        print_gran("gran tile", 12, "fwd f32", t.s);
        REQUIRE(t.s.unmapped == 0u);
        CHECK(t.s.fp > 0u);
        CHECK(t.s.flip > 0u);
        // Full granule block plus the masked terminal block: N = 12 registers in and out
        // through the real-batch seam (lifts in, downcast sinks out).
        CHECK(t.s.lifts == 12u);
        CHECK(t.s.sunk == 12u);
    }
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
    REQUIRE(dag::sym<double>::uninit_reads() == 0u);
}

// One (R, N) ladder: run it, REQUIRE the rungs to cover M, the (q*M+j) <-> (L*M+j) maps
// to partition [0, N), the per-chunk census to match the same oracles the dif census
// uses (where a closed form applies), and every batched chunk of the tree to share one
// census (the chunk text is width-invariant). Prints the ladder structure.
template <typename T, unsigned R, unsigned N, std::size_t... Dens>
void check_ladder(const char* tag, dens_list<Dens...> ds) {
    static_assert(R >= 2 && N % R == 0, "ladder traces a real radix split");
    rb_trace<T, R, N, Dens...> t;
    rb_run(t, ds);
    using L = chunk_ladder<T, R, N>;
    std::size_t cover = 0;
    for (const rung_row& rr : L::rungs()) cover += static_cast<std::size_t>(rr.width) * rr.trips;
    REQUIRE(cover == L::M);
    const char* fam = "radix_butterfly_ct";
    std::uint64_t batch_ref = 0;
    bool have_batch = false;
    for (const rung_row& rr : L::rungs()) {
        std::uint64_t rung_ops = 0;
        for (const chunk_tick& tk : t.ticks)
            if (tk.width == rr.width) rung_ops += tk.ops;
        dag::registry().push_back(dag::rung{fam, rr.width == L::W ? "full" : rung_name(rr.width),
                                            dag::provenance::unaligned, prec_char<T>(), N,
                                            rr.width, rr.trips, rung_ops});
    }
    for (const chunk_tick& tk : t.ticks) {
        if (tk.width > 1) {
            if (!have_batch) {
                batch_ref = tk.ops;
                have_batch = true;
            } else {
                INFO("batched chunk census differs at j=" << tk.j << " w=" << tk.width);
                REQUIRE(tk.ops == batch_ref);
            }
            if (dif_arm(R) == dif_arm_kind::radix_sym)
                REQUIRE(tk.ops == 6 * (R - 1) + sym_ops_oracle(R));
        } else if (dif_arm(R) == dif_arm_kind::radix_sym) {
            REQUIRE(tk.ops == sym_ops_oracle(R) + scalar_twiddle_ops(R, N, tk.j));
        }
    }
    std::printf("%s N=%-3u r=%u M=%-3zu W=%-2zu nfull=%-2zu rungs=", tag, N, R, L::M, L::W,
                L::nfull);
    for (const rung_row& rr : L::rungs())
        std::printf("[%s x%u j%llu]", rung_name(rr.width), rr.trips,
                    static_cast<unsigned long long>(rr.j0));
    std::printf(" tw-cols=%zu tagged=%zu\n", (R - 1) * L::M, t.c.tagged);
}

TEST_CASE("dag census: radix_butterfly_ct chunk ladder", "[dag][census]") {
    dag::sym<float>::reset_counters();
    dag::sym<double>::reset_counters();
    static_assert(codelet_radix(15) == 3);
    static_assert(codelet_radix(45) == 3);
    static_assert(codelet_radix(100) == 4);
    static_assert(codelet_radix(405) == 3);
    static_assert(codelet_radix(125) == 5);
    check_ladder<float, 3, 15>("ladder f32", dens<3, 15>);
    check_ladder<float, 3, 45>("ladder f32", dens<3, 45>);
    check_ladder<float, 4, 100>("ladder f32", dens<2, 4, 100>);
    check_ladder<float, 3, 405>("ladder f32", dens<3, 405>);
    check_ladder<double, 3, 15>("ladder f64", dens<3, 15>);
    check_ladder<double, 3, 45>("ladder f64", dens<3, 45>);
    check_ladder<double, 5, 125>("ladder f64", dens<5, 125>);
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
    REQUIRE(dag::sym<double>::uninit_reads() == 0u);
}

TEST_CASE("dag census: report rows (even PFA, pow2, generic-arm routing, registry)",
          "[dag][report]") {
    dag::sym<float>::reset_counters();
    // Even PFA and pow2 carry no closed-form oracle in the tree: printed, not pinned.
    // dif_arm(2) is generic per the shipped ladder (pow2 needs IP >= 4).
    static_assert(dif_arm(6) == dif_arm_kind::pfa);
    static_assert(dif_arm(10) == dif_arm_kind::pfa);
    static_assert(dif_arm(12) == dif_arm_kind::pfa);
    static_assert(dif_arm(14) == dif_arm_kind::pfa);
    static_assert(dif_arm(22) == dif_arm_kind::pfa);
    static_assert(dif_arm(2) == dif_arm_kind::generic);
    static_assert(dif_arm(4) == dif_arm_kind::pow2);
    static_assert(dif_arm(8) == dif_arm_kind::pow2);
    static_assert(dif_arm(16) == dif_arm_kind::pow2);

    print_row("dif pfa", 6, "pfa", trace_butterfly<float, 6, false>("dif_butterfly", dens<2, 3>),
              -1);
    print_row("dif pfa", 10, "pfa",
              trace_butterfly<float, 10, false>("dif_butterfly", dens<2, 5>), -1);
    print_row("dif pfa", 12, "pfa",
              trace_butterfly<float, 12, false>("dif_butterfly", dens<3, 4>), -1);
    print_row("dif pfa", 14, "pfa",
              trace_butterfly<float, 14, false>("dif_butterfly", dens<2, 7>), -1);
    print_row("dif pfa", 22, "pfa",
              trace_butterfly<float, 22, false>("dif_butterfly", dens<2, 11>), -1);
    print_row("dif gen", 2, "generic",
              trace_butterfly<float, 2, false>("dif_butterfly", dens<2>), -1);
    print_row("dif pow2", 4, "pow2", trace_butterfly<float, 4, false>("dif_butterfly", dens<4>),
              -1);
    print_row("dif pow2", 8, "pow2", trace_butterfly<float, 8, false>("dif_butterfly", dens<8>),
              -1);

    // Routing census over the shipped dif radix sets: the O(n^2) generic arm of
    // dif_butterfly (butterfly.hpp else arm) is address-taken only where the ladder says
    // generic. Contrary to the plan's earlier wording it is reachable -- at exactly
    // IP=2, which the radix-2 dispatch elects; every other shipped radix closes on
    // ct/radix_sym/pfa/pow2. Traced nowhere by design.
    std::vector<std::size_t> generic_radices;
    for (const std::size_t r : dif_candidate_radices)
        if (dif_arm(r) == dif_arm_kind::generic) generic_radices.push_back(r);
    for (const std::size_t r : dif_generic_radices)
        if (dif_arm(r) == dif_arm_kind::generic) generic_radices.push_back(r);
    REQUIRE(generic_radices == std::vector<std::size_t>{2});
    std::printf("routing: dif_butterfly's generic O(n^2) arm is reached by exactly one shipped"
                " dif radix set entry (IP=2); every other entry closes on ct/radix_sym/pfa/pow2"
                " (printed, never traced)\n");

    std::printf("kFusedFma=%d (the oracle above expands with it, as the text does)\n",
                kFusedFma ? 1 : 0);
    std::printf("%-15s %-9s %-9s %-4s %-5s %-4s %-5s %s\n", "family", "arm", "prov", "prec",
                "radix", "wd", "trips", "nodes");
    for (const dag::rung& r : dag::registry())
        std::printf("%-15s %-9s %-9s %-4c %-5u %-4u %-5u %llu\n", r.family, r.arm,
                    dag::provenance_name(r.prov), r.prec, r.radix, r.width, r.trips,
                    static_cast<unsigned long long>(r.nodes));
    REQUIRE(dag::sym<float>::uninit_reads() == 0u);
}
