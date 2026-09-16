// AoS-granule leaf dialect (include/admiral/detail/granule_codelet.hpp bodies and tiles,
// src/granule_apply.hpp drivers). The drivers are called DIRECTLY, so every built N is
// exercised regardless of the admission table: the table itself is asserted separately,
// and the poison twin binary builds this same source with -DADM_GRANULE_TEST_POISON
// (one lane-flipped radix-3 pool entry, plus lane-flipped stage-twiddle pools at the
// 8-point stage, which the cube case below is what exercises) and must fail, ctest
// WILL_FAIL side.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include "utils/reference.hpp"

#include <admiral/detail/granule_codelet.hpp>

#include "granule_apply.hpp"
#include "granule_cube.hpp"

#include <cstring>
#include <complex>
#include <vector>

using namespace admiral::detail;

namespace {

// End-of-buffer guard zone: the masked terminal block must not write past a line's
// granules, and the proof is a live guard where an unmasked store would land.
constexpr std::size_t kGuard = 2 * 16;

// Rows: nlines lines of N points, lane ladders + masked granule block + residual.
template<unsigned N, typename T, bool Forward>
void check_granule_rows(std::size_t nlines, std::size_t stride, T fct, unsigned seed) {
    CAPTURE(N, nlines, stride, fct, seed, Forward);
    const std::complex<T> guard(T(-9), T(7));
    const auto x = make_input<T>(nlines * stride, seed);

    std::vector<std::complex<T>> want(nlines * stride + kGuard, guard);
    for (std::size_t r = 0; r < nlines; ++r) {
        const std::vector<std::complex<T>> line(x.data() + r * stride,
                                                x.data() + r * stride + N);
        const auto ref = reference_dft<T>(line, Forward);
        for (std::size_t k = 0; k < N; ++k) want[r * stride + k] = ref[k] * fct;
    }

    std::vector<std::complex<T>> oop(nlines * stride + kGuard, guard);
    if constexpr (kGranuleFmaddsub) {
    granule_apply_rows_oop<N, T, Forward>(x.data(), oop.data(), nlines, stride, stride, fct);
    require_close(oop, want, fft_tol<T>());

    // In place. Guard everywhere except the used cells, so the whole-buffer comparison
    // covers over-write: the padding past N (and past the last line) must come back
    // untouched.
    std::vector<std::complex<T>> ip(nlines * stride + kGuard, guard);
    for (std::size_t r = 0; r < nlines; ++r)
        std::copy_n(x.data() + r * stride, N, ip.data() + r * stride);
    granule_apply_rows_oop<N, T, Forward>(ip.data(), ip.data(), nlines, stride, stride, fct);
    require_close(ip, want, fft_tol<T>());
    }
}

// Cols: transform N points down each of ncols columns; leftover columns go to the
// existing col body, and columns past ncols are the input's in the in-place case.
template<unsigned N, typename T, bool Forward>
void check_granule_cols(std::size_t ncols, T scale, bool in_place, unsigned seed) {
    CAPTURE(N, ncols, scale, in_place, seed, Forward);
    const std::complex<T> guard(T(5), T(-3));
    const std::size_t in_inner = ncols + 2;
    const std::size_t out_inner = in_place ? in_inner : ncols + 5;
    const auto x = make_input<T>(N * in_inner, seed);

    std::vector<std::complex<T>> want(N * out_inner + kGuard, guard);
    for (std::size_t c = 0; c < ncols; ++c) {
        std::vector<std::complex<T>> col(N);
        for (std::size_t p = 0; p < N; ++p) col[p] = x[p * in_inner + c];
        const auto ref = reference_dft<T>(col, Forward);
        for (std::size_t p = 0; p < N; ++p) want[p * out_inner + c] = ref[p] * scale;
    }

    std::vector<std::complex<T>> out(N * out_inner + kGuard, guard);
    if (in_place) {
        for (std::size_t i = 0; i < N * in_inner; ++i) out[i] = x[i];
        for (std::size_t p = 0; p < N; ++p)
            for (std::size_t c = ncols; c < out_inner; ++c)
                want[p * out_inner + c] = x[p * in_inner + c];
    }
    if constexpr (kGranuleFmaddsub) {
        const std::complex<T>* src = in_place ? out.data() : x.data();
        granule_col_apply<N, T, Forward>(src, in_inner, out.data(), out_inner, ncols, scale);
        require_close(out, want, fft_tol<T>());
    }
}

template<unsigned N>
void check_granule_n() {
    using Ts = float;
    const std::size_t Gf = xsimd::batch<float>::size / 2;
    const std::size_t Gd = xsimd::batch<double>::size / 2;
    // One shape per ladder position at seed 7, plus the seed/design sweep at GL + 1.
    for (const unsigned seed : {7u, 99u, 4242u}) {
        check_granule_rows<N, float, true>(Gf + 1, N + 3, 1.0f, seed);
        check_granule_rows<N, float, false>(Gf + 1, N + 3, 0.5f, seed);
        check_granule_rows<N, double, true>(Gd + 1, N + 3, 1.0, seed);
        check_granule_rows<N, double, false>(Gd + 1, N + 3, 0.5, seed);
        check_granule_cols<N, float, true>(Gf + 1, 1.0f, false, seed);
        check_granule_cols<N, float, false>(Gf + 1, 0.25f, true, seed);
        check_granule_cols<N, double, true>(Gd + 1, 1.0, false, seed);
        check_granule_cols<N, double, false>(Gd + 1, 0.25, true, seed);
    }
    for (const std::size_t nl :
         {std::size_t{1}, Gf - 1, Gf, Gf + 1, 2 * Gf - 1, 2 * Gf, 3 * Gf + 1})
        for (const std::size_t stride : {std::size_t{N}, std::size_t{N + 3}})
            for (const float fct : {1.0f, 0.5f}) {
                check_granule_rows<N, Ts, true>(nl, stride, fct, 4242u);
                check_granule_rows<N, Ts, false>(nl, stride, fct, 4242u);
            }
    for (const std::size_t nc :
         {std::size_t{1}, Gd - 1, Gd, Gd + 1, 2 * Gd - 1, 2 * Gd, 3 * Gd + 1})
        for (const double scale : {1.0, 0.25})
            for (const bool ip : {false, true}) {
                check_granule_cols<N, double, true>(nc, scale, ip, 4242u);
                check_granule_cols<N, double, false>(nc, scale, ip, 4242u);
            }
}

// Bits must not move with the data's alignment class: every granule load is the same
// granularity in every tile, so no arm ever forks on buffer alignment. The check is a
// byte compare across offsets 0..15 through the public rows/cols drivers, at the
// committed (N, prec) set.
template<unsigned N, typename T, bool Forward>
void check_granule_align() {
    // Dense buffers (stride == N): the memcmp must cover only transform output; pad
    // over-write is the ladder sweep's job above, not this byte-identity case's.
    if constexpr (granule_rows_admit_v<N, T>) {
        const std::size_t nlines = xsimd::batch<T>::size + 3;
        const std::size_t stride = N;
        const auto x = make_input<T>(nlines * stride, 0xA11Cu);
        std::vector<std::complex<T>> ref(nlines * stride);
        granule_apply_rows_oop<N, T, Forward>(x.data(), ref.data(), nlines, stride, stride,
                                              T(0.5));
        for (std::size_t off = 1; off < 16; ++off) {
            std::vector<std::complex<T>> ib(nlines * stride + 16, std::complex<T>(9, 9));
            std::vector<std::complex<T>> ob(nlines * stride + 16, std::complex<T>(9, 9));
            std::copy(x.begin(), x.end(), ib.begin() + static_cast<std::ptrdiff_t>(off));
            granule_apply_rows_oop<N, T, Forward>(ib.data() + off, ob.data() + off, nlines,
                                                  stride, stride, T(0.5));
            REQUIRE(std::memcmp(ob.data() + off, ref.data(),
                                nlines * stride * sizeof(std::complex<T>)) == 0);
        }
    }
    if constexpr (granule_cols_admit_v<N, T>) {
        const std::size_t ncols = xsimd::batch<T>::size + 3;
        const std::size_t inner = ncols;
        const auto x = make_input<T>(N * inner, 0xA11Du);
        std::vector<std::complex<T>> ref(N * inner);
        granule_col_apply<N, T, Forward>(x.data(), inner, ref.data(), inner, ncols, T(0.25));
        for (std::size_t off = 1; off < 16; ++off) {
            std::vector<std::complex<T>> ib(N * inner + 16, std::complex<T>(9, 9));
            std::vector<std::complex<T>> ob(N * inner + 16, std::complex<T>(9, 9));
            std::copy(x.begin(), x.end(), ib.begin() + static_cast<std::ptrdiff_t>(off));
            granule_col_apply<N, T, Forward>(ib.data() + off, inner, ob.data() + off, inner,
                                             ncols, T(0.25));
            REQUIRE(std::memcmp(ob.data() + off, ref.data(),
                                N * inner * sizeof(std::complex<T>)) == 0);
        }
    }
}

// Reduced-width ladder with data in the two quadrants the seed/design sweep runs only
// at GL + 1: f64 rows and f32 cols. Same rung set, strides, fct/scale, in-place arms
// and guard sentinels as the two ladders above, at all three seeds. The reduced-width
// rungs put the f64 2-line tile (the V::size == 4 transpose arm, the plain tp_pair
// network) through direct-DFT compares: it is the sized-batch rung at v4 widths and
// the full tile at v3, where GL is 2.
template<unsigned N>
void check_granule_reduced() {
    const std::size_t Gd = xsimd::batch<double>::size / 2;
    const std::size_t Gf = xsimd::batch<float>::size / 2;
    for (const unsigned seed : {7u, 99u, 4242u}) {
        for (const std::size_t nl :
             {std::size_t{1}, Gd - 1, Gd, Gd + 1, 2 * Gd - 1, 2 * Gd, 3 * Gd + 1})
            for (const std::size_t stride : {std::size_t{N}, std::size_t{N + 3}})
                for (const double fct : {1.0, 0.5}) {
                    check_granule_rows<N, double, true>(nl, stride, fct, seed);
                    check_granule_rows<N, double, false>(nl, stride, fct, seed);
                }
        for (const std::size_t nc :
             {std::size_t{1}, Gf - 1, Gf, Gf + 1, 2 * Gf - 1, 2 * Gf, 3 * Gf + 1})
            for (const float scale : {1.0f, 0.25f})
                for (const bool ip : {false, true}) {
                    check_granule_cols<N, float, true>(nc, scale, ip, seed);
                    check_granule_cols<N, float, false>(nc, scale, ip, seed);
                }
    }
}

// Cubes: the three-pass cube driver against the separable 3-D reference, OOP and in
// place. The driver folds the whole fct into its first pass; mirroring means scaling
// the reference once. This is the case that makes the poison twin observe the extended
// seam: 8^3's stage-twiddle pools are its only FP constants, and 4^3 has none (rotations
// are integer top-bit masks), so 4^3's control IS this reference compare.
template<unsigned N, typename T, bool Forward>
void check_granule_cube(T fct, unsigned seed) {
    CAPTURE(N, fct, seed, Forward);
    const std::size_t n3 = N * N * N;
    const auto x = make_input<T>(n3, seed);
    const auto ref0 = reference_cube3d<T, long double>(x, N, Forward);
    std::vector<std::complex<long double>> ref(n3);
    for (std::size_t i = 0; i < n3; ++i) ref[i] = ref0[i] * static_cast<long double>(fct);

    if constexpr (kGranuleFmaddsub) {
        std::vector<std::complex<T>> oop(n3);
        granule_cube_apply<N, T, Forward>(x.data(), oop.data(), fct);
        require_close_pointwise(oop, ref);

        auto ip = x;
        granule_cube_apply<N, T, Forward>(ip.data(), ip.data(), fct);
        require_close_pointwise(ip, ref);
    }
}

// Cube twin of the rows/cols alignment case: three through-memory passes of unaligned
// loads only, so bits must not move with the data's alignment class.
template<unsigned N, typename T, bool Forward>
void check_granule_cube_align() {
    if constexpr (granule_cube_admit_v<N, T>) {
        const std::size_t n3 = N * N * N;
        const auto x = make_input<T>(n3, 0xC0BEu);
        std::vector<std::complex<T>> ref(n3);
        granule_cube_apply<N, T, Forward>(x.data(), ref.data(), T(1));
        for (std::size_t off = 1; off < 16; ++off) {
            std::vector<std::complex<T>> ib(n3 + 16, std::complex<T>(9, 9));
            std::vector<std::complex<T>> ob(n3 + 16, std::complex<T>(9, 9));
            std::copy(x.begin(), x.end(), ib.begin() + static_cast<std::ptrdiff_t>(off));
            granule_cube_apply<N, T, Forward>(ib.data() + off, ob.data() + off, T(1));
            REQUIRE(std::memcmp(ob.data() + off, ref.data(),
                                n3 * sizeof(std::complex<T>)) == 0);
        }
    }
}

}  // namespace

TEST_CASE("granule rows/cols match reference DFT over ladders, strides, fct, seeds",
          "[codelet][granule]") {
    if constexpr (!kGranuleFmaddsub) {
        // Below AVX2/FMA3 the entire dialect is if constexpr-dead; there is nothing to
        // compare. Named skip, not a quiet pass (the native build asserts non-empty
        // admission in the catalog case below).
        SKIP("granule dialect compiled out at this ISA");
    }
    check_granule_n<9>();
    check_granule_n<10>();
    check_granule_n<11>();
    check_granule_n<12>();
    check_granule_n<13>();
    check_granule_n<14>();
    check_granule_n<15>();
    // f64 at N = 16 is built but admission-off (measured loss); built-set coverage stands.
    check_granule_n<16>();
    // The drivers' residual names codelet_apply<N> / col_codelet_body<N>, which exist
    // only for catalog N; a trimmed catalog (the sanitizer cap at 16) has no 24.
    if constexpr (is_codelet_catalog(24)) check_granule_n<24>();
}

TEST_CASE("granule f64-rows / f32-cols reduced-width rungs match reference DFT",
          "[codelet][granule]") {
    if constexpr (!kGranuleFmaddsub) {
        SKIP("granule dialect compiled out at this ISA");
    }
    check_granule_reduced<9>();
    check_granule_reduced<10>();
    check_granule_reduced<11>();
    check_granule_reduced<12>();
    check_granule_reduced<13>();
    check_granule_reduced<14>();
    check_granule_reduced<15>();
    check_granule_reduced<16>();
    if constexpr (is_codelet_catalog(24)) check_granule_reduced<24>();
}

// The admission truth tables, evaluated over the whole generated catalog: each predicate
// answers true exactly at its body's measured set, so a catalog change or a flipped bit is
// loud at compile time in both precisions. ISA-probe-relative, so they hold in every build.
// Under ADM_GRANULE_ADMIT=OFF the ON-state folds would assert against a table the knob
// already zeroed, so the knob sides are compiled separately: ON keeps the exact table,
// OFF asserts the table is empty (the fold cannot pass unless every entry is false).
#ifndef ADM_GRANULE_ADMIT_OFF
namespace {
// The hardcoded admitted sets, one per body: every flip edits exactly one of these
// right-hand sides, and the fold then proves the live table equals it across the whole
// catalog, in both precisions.
template<typename T>
constexpr bool granule_rows_rhs(std::size_t n) {
    return kGranuleFmaddsub &&
           (n == 12 || n == 24 || (n == 16 && std::is_same_v<T, float>) ||
            (n == 10 && std::is_same_v<T, float>) ||
            (n == 10 && std::is_same_v<T, double>) ||
            (n == 14 && std::is_same_v<T, double>) ||
            (n == 14 && std::is_same_v<T, float>) ||
            (n == 15 && std::is_same_v<T, float>) ||
            (n == 15 && std::is_same_v<T, double>) ||
            (n == 13 && std::is_same_v<T, double>) ||
            (n == 9 && std::is_same_v<T, double>) ||
            (n == 13 && std::is_same_v<T, float>) ||
            (n == 11 && std::is_same_v<T, double>));
}
template<typename T>
constexpr bool granule_cols_rhs(std::size_t n) {
    return kGranuleFmaddsub &&
           (n == 12 || n == 24 || (n == 16 && std::is_same_v<T, float>) ||
            (n == 10 && std::is_same_v<T, float>) ||
            (n == 10 && std::is_same_v<T, double>) ||
            (n == 14 && std::is_same_v<T, double>) ||
            (n == 14 && std::is_same_v<T, float>) ||
            (n == 11 && std::is_same_v<T, double>) ||
            (n == 15 && std::is_same_v<T, double>) ||
            (n == 15 && std::is_same_v<T, float>) ||
            (n == 9 && std::is_same_v<T, double>));
}
template<typename T, std::size_t... Is>
constexpr bool granule_admit_table_ok(std::index_sequence<Is...>) {
    return ((granule_rows_admit_v<static_cast<unsigned>(CODELET_CATALOG_SIZES[Is]), T> ==
             granule_rows_rhs<T>(CODELET_CATALOG_SIZES[Is])) &&
            ...) &&
           ((granule_cols_admit_v<static_cast<unsigned>(CODELET_CATALOG_SIZES[Is]), T> ==
             granule_cols_rhs<T>(CODELET_CATALOG_SIZES[Is])) &&
            ...);
}
}
static_assert(granule_admit_table_ok<float>(
                  std::make_index_sequence<CODELET_CATALOG_SIZES.size()>{}));
static_assert(granule_admit_table_ok<double>(
                  std::make_index_sequence<CODELET_CATALOG_SIZES.size()>{}));

// Truth table on the boundary cells that pull the design's load: 16 f64 OFF (measured
// loss), N < 9 OFF (flat arms already win), N > 24 OFF (register-cliff work pending).
// Inside 9..15 the admitted set is the per-N sweep's output and the folds above pin it;
// the named negatives below pin the cells still waiting on their own flip.
// ISA-probe-relative positives, so the table holds in every build flavor.
static_assert(granule_rows_admit_v<12, float> == kGranuleFmaddsub);
static_assert(granule_rows_admit_v<12, double> == kGranuleFmaddsub);
static_assert(granule_cols_admit_v<16, float> == kGranuleFmaddsub);
static_assert(!granule_rows_admit_v<16, double>);
static_assert(!granule_cols_admit_v<16, double>);
static_assert(!granule_rows_admit_v<8, float>);
static_assert(!granule_rows_admit_v<9, float>);
static_assert(!granule_rows_admit_v<20, float>);
static_assert(!granule_cols_admit_v<25, double>);
static_assert(!granule_cols_admit_v<64, float>);
static_assert(!granule_rows_admit_v<12, long double>);   // scalar backend boundary

// The cube table is the decoupled one: 4 and 8 live ONLY there (the rows/cols negatives
// for them above stay), and their admission set matches the seat's measured set.
static_assert(granule_cube_admit_v<4, float> == kGranuleFmaddsub);
static_assert(granule_cube_admit_v<4, double> == kGranuleFmaddsub);
static_assert(granule_cube_admit_v<8, float> == kGranuleFmaddsub);
static_assert(granule_cube_admit_v<8, double> == kGranuleFmaddsub);
static_assert(!granule_cube_admit_v<12, float>);
static_assert(!granule_cube_admit_v<16, float>);
static_assert(!granule_cube_admit_v<2, double>);
static_assert(!granule_cube_admit_v<4, long double>);
static_assert(!granule_rows_admit_v<4, float>);
static_assert(!granule_rows_admit_v<4, double>);
static_assert(!granule_cols_admit_v<8, float>);
#else
namespace {
template<typename T, std::size_t... Is>
constexpr bool granule_admit_table_empty(std::index_sequence<Is...>) {
    return ((!granule_rows_admit_v<static_cast<unsigned>(CODELET_CATALOG_SIZES[Is]), T>) &&
            ...);
}
}
static_assert(granule_admit_table_empty<float>(
                  std::make_index_sequence<CODELET_CATALOG_SIZES.size()>{}));
static_assert(granule_admit_table_empty<double>(
                  std::make_index_sequence<CODELET_CATALOG_SIZES.size()>{}));
static_assert(!granule_cols_admit_v<16, float>);   // the cols alias follows rows to false
static_assert(!granule_cube_admit_v<4, float>);    // the knob zeroes the cube table too
static_assert(!granule_cube_admit_v<8, double>);
#endif

TEST_CASE("granule catalog admission table is exact, and nonempty at the native build",
          "[codelet][granule]") {
    // The fold asserts above are the table. This case is the runtime face: under the
    // dialect probe the native build must admit SOMETHING (a native build that admits
    // nothing is a bug shaped here), and at v2 the family is if constexpr-dead. With
    // the admission knob OFF, empty is the asserted state on every ISA.
    const bool any = granule_rows_admit_v<12, float> || granule_rows_admit_v<24, double> ||
                     granule_cols_admit_v<16, float>;
#ifndef ADM_GRANULE_ADMIT_OFF
    CHECK(any == kGranuleFmaddsub);
#else
    CHECK(!any);
#endif
}

TEST_CASE("granule bits do not depend on data alignment class", "[codelet][granule]") {
    if constexpr (!kGranuleFmaddsub)
        SKIP("granule dialect compiled out at this ISA");
#ifdef ADM_GRANULE_ADMIT_OFF
    // No admitted (N, prec) exists for this case to exercise; the OFF static asserts
    // above are its evidence. Name the skip rather than pass on zero work.
    SKIP("granule admission compiled out (ADM_GRANULE_ADMIT=OFF)");
#endif
    // One pair per (N, prec) whose rows flip landed; each call exercises whichever
    // body the same commit admitted (the other arm goes live at its own flip).
    check_granule_align<9, double, true>();
    check_granule_align<9, double, false>();
    check_granule_align<10, float, true>();
    check_granule_align<10, float, false>();
    check_granule_align<10, double, true>();
    check_granule_align<10, double, false>();
    check_granule_align<11, double, true>();
    check_granule_align<11, double, false>();
    check_granule_align<12, float, true>();
    check_granule_align<12, float, false>();
    check_granule_align<12, double, true>();
    check_granule_align<12, double, false>();
    check_granule_align<13, float, true>();
    check_granule_align<13, float, false>();
    check_granule_align<13, double, true>();
    check_granule_align<13, double, false>();
    check_granule_align<14, float, true>();
    check_granule_align<14, float, false>();
    check_granule_align<14, double, true>();
    check_granule_align<14, double, false>();
    check_granule_align<15, float, true>();
    check_granule_align<15, float, false>();
    check_granule_align<15, double, true>();
    check_granule_align<15, double, false>();
    check_granule_align<16, float, true>();
    check_granule_align<16, float, false>();
    // A trimmed catalog would leave the N = 24 residual undefined; the sanitizer cap is 16.
    if constexpr (is_codelet_catalog(24)) {
        check_granule_align<24, float, true>();
        check_granule_align<24, float, false>();
        check_granule_align<24, double, true>();
        check_granule_align<24, double, false>();
    }
}

TEST_CASE("granule cubes 4^3/8^3 match the separable 3-D reference", "[codelet][granule]") {
    if constexpr (!kGranuleFmaddsub) {
        SKIP("granule dialect compiled out at this ISA");
    }
    for (const unsigned seed : {7u, 99u, 4242u}) {
        // Unit fct forward (the unscaled arm) and the plan's own default 1/N^3 inverse.
        check_granule_cube<4, float, true>(1.0f, seed);
        check_granule_cube<4, double, true>(1.0, seed);
        check_granule_cube<8, float, true>(1.0f, seed);
        check_granule_cube<8, double, true>(1.0, seed);
        check_granule_cube<4, float, false>(1.0f / 64.0f, seed);
        check_granule_cube<4, double, false>(1.0 / 64.0, seed);
        check_granule_cube<8, float, false>(1.0f / 512.0f, seed);
        check_granule_cube<8, double, false>(1.0 / 512.0, seed);
    }
    // The scaled arm in both directions (a custom fct is what exercises it at the seat).
    check_granule_cube<4, float, true>(0.5f, 4242u);
    check_granule_cube<4, double, true>(0.5, 4242u);
    check_granule_cube<8, float, true>(0.5f, 4242u);
    check_granule_cube<8, double, true>(0.5, 4242u);
    check_granule_cube<4, float, false>(0.25f, 4242u);
    check_granule_cube<4, double, false>(0.25, 4242u);
    check_granule_cube<8, float, false>(0.25f, 4242u);
    check_granule_cube<8, double, false>(0.25, 4242u);
}

TEST_CASE("granule cube bits do not depend on data alignment class", "[codelet][granule]") {
    if constexpr (!kGranuleFmaddsub)
        SKIP("granule dialect compiled out at this ISA");
#ifdef ADM_GRANULE_ADMIT_OFF
    SKIP("granule admission compiled out (ADM_GRANULE_ADMIT=OFF)");
#endif
    check_granule_cube_align<4, float, true>();
    check_granule_cube_align<4, float, false>();
    check_granule_cube_align<4, double, true>();
    check_granule_cube_align<4, double, false>();
    check_granule_cube_align<8, float, true>();
    check_granule_cube_align<8, float, false>();
    check_granule_cube_align<8, double, true>();
    check_granule_cube_align<8, double, false>();
}
