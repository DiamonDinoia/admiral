
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/codelet_max.hpp>
#include <admiral/detail/ct_math.hpp>
#include <admiral/detail/twiddles.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

using namespace admiral::detail;

// The octant map at den 8 (num = k gives oct = k, rem = 0): the folded pair at those roots
// is the exact constant the map names, so these pin both sides of the classification. They
// are excluded from the poison twin: the swap makes them fail at compile time, and the
// twin's asserted failure is the lattice REQUIRE at runtime.
#ifndef ADM_CT_ROOT_POISON
static_assert(ct_root<0, 8, false>() == root_form::one);
static_assert(ct_root<1, 8, false>() == root_form::diag);
static_assert(ct_root<2, 8, false>() == root_form::pos_i);
static_assert(ct_root<3, 8, false>() == root_form::anti_diag);
static_assert(ct_root<4, 8, false>() == root_form::neg_one);
static_assert(ct_root<5, 8, false>() == root_form::diag);
static_assert(ct_root<6, 8, false>() == root_form::neg_i);
static_assert(ct_root<7, 8, false>() == root_form::anti_diag);
static_assert(ct_root<1, 8, true>() == root_form::anti_diag);
static_assert(ct_root<2, 8, true>() == root_form::neg_i);
static_assert(ct_root<1, 3, false>() == root_form::generic);
#endif

namespace {

// Lattice bound: every stage-twiddle denominator the lattice sweeps is <= this. The census
// below ties the shipped denominators to the bound, so growing the shipped census past it
// breaks THIS build until the lattice follows.
inline constexpr std::size_t kLatticeDenMax = 512;

// Shipped stage-twiddle denominators, taken from the tree's own constants: the codelet
// catalog (CODELET_CATALOG_SIZES covers the kernels' butterfly stages, and the granule
// cubes' N in {4, 8} sit inside it) and the dif radix sets in twiddles.hpp (dif_candidate
// radixes drive the pass stage twiddles, dif_generic the prime radixes; lane_stage_twiddle
// shares those denominators).
constexpr bool is_shipped_den(std::size_t d) {
    for (const std::size_t n : CODELET_CATALOG_SIZES)
        if (n == d) return true;
    for (const std::size_t n : dif_candidate_radices)
        if (n == d) return true;
    for (const std::size_t n : dif_generic_radices)
        if (n == d) return true;
    return false;
}

constexpr std::size_t max_shipped_den() {
    std::size_t m = 0;
    for (const std::size_t n : CODELET_CATALOG_SIZES) m = n > m ? n : m;
    for (const std::size_t n : dif_candidate_radices) m = n > m ? n : m;
    for (const std::size_t n : dif_generic_radices) m = n > m ? n : m;
    return m;
}

static_assert(max_shipped_den() <= kLatticeDenMax,
              "shipped stage-twiddle denominators must stay inside the ct_root lattice");

struct root_lattice_bad {
    std::size_t den = 0, num = 0;
    bool conj = false;
};

// Sweep policy per precision: double carries the full lattice (every Num < Den); float and
// long double full-sweep the shipped denominators, every den divisible by 8 (all octant
// grids) and everything <= 64, and spot elsewhere. Keeps the constexpr-fold cost of this TU
// on gcc inside a normal test build (times recorded in the wave receipt).
template<typename F>
constexpr bool lattice_swept(std::size_t den, std::size_t num) {
    if (std::is_same_v<F, double>) return true;
    if (den <= 64 || den % 8 == 0 || is_shipped_den(den)) return true;
    return num == 0 || num == 1 || num == den / 4 || num == den / 2 || num == den - 1;
}

// gcc caps one constant evaluation at -fconstexpr-ops-limit; a full-precision 1..512
// sweep busts it, so the lattice folds in 64-den chunks and the chunks merge.
inline constexpr std::size_t kLatticeChunk = 64;
static_assert(kLatticeDenMax % kLatticeChunk == 0);

// First lattice point in dens (Chunk*kLatticeChunk, (Chunk+1)*kLatticeChunk] where the
// integer (oct, rem) classification disagrees with the folded (s, c) pair's form through
// the arms' own FP conditions; {0, 0, false} when none does.
// Immediate, not constexpr: cl does not escalate a constexpr caller of ct_sincos_turns (P2564).
template<typename F, std::size_t Chunk>
ADM_CONSTEVAL root_lattice_bad first_bad_chunk() {
    for (std::size_t den = Chunk * kLatticeChunk + 1; den <= (Chunk + 1) * kLatticeChunk;
         ++den)
        for (std::size_t num = 0; num < den; ++num)
            for (const bool conj : {false, true})
                if (lattice_swept<F>(den, num) &&
                    ct_root_form(num, den, conj) !=
                        fold_form(ct_sincos_turns<F>(conj, num, den)))
                    return {den, num, conj};
    return {};
}

// One variable per chunk: each gets its own constexpr-ops budget.
template<typename F, std::size_t Chunk>
inline constexpr root_lattice_bad bad_chunk_v = first_bad_chunk<F, Chunk>();

template<typename F, std::size_t... Ch>
constexpr root_lattice_bad first_bad_root(std::index_sequence<Ch...>) {
    for (const root_lattice_bad b : {bad_chunk_v<F, Ch>...})
        if (b.den != 0) return b;
    return {};
}

template<typename F>
constexpr root_lattice_bad first_bad_root() {
    return first_bad_root<F>(
        std::make_index_sequence<kLatticeDenMax / kLatticeChunk>{});
}

inline constexpr root_lattice_bad bad_f = first_bad_root<float>();
inline constexpr root_lattice_bad bad_d = first_bad_root<double>();
inline constexpr root_lattice_bad bad_ld = first_bad_root<long double>();

} // namespace

TEST_CASE("ct_root lattice agrees with the sincos fold at float precision",
          "[ct_math][ct_root]") {
    INFO("first bad (den, num, conj) = " << bad_f.den << ", " << bad_f.num << ", "
                                         << bad_f.conj);
    REQUIRE(bad_f.den == 0u);
}

TEST_CASE("ct_root lattice agrees with the sincos fold at double precision",
          "[ct_math][ct_root]") {
    INFO("first bad (den, num, conj) = " << bad_d.den << ", " << bad_d.num << ", "
                                         << bad_d.conj);
    REQUIRE(bad_d.den == 0u);
}

TEST_CASE("ct_root lattice agrees with the sincos fold at long double precision",
          "[ct_math][ct_root][longdouble]") {
    INFO("first bad (den, num, conj) = " << bad_ld.den << ", " << bad_ld.num << ", "
                                         << bad_ld.conj);
    REQUIRE(bad_ld.den == 0u);
}
