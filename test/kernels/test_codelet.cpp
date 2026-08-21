// Correctness tests for the generic metaprogrammed codelet kernel<N, T, Forward>.
// Each case checks one instantiation against a direct O(N^2) reference DFT, the
// ground truth the generic Cooley-Tukey recursion must reproduce.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "utils/reference.hpp"

#include <admiral/detail/codelet.hpp>
#include <admiral/detail/four_step.hpp>

#include <cmath>
#include <complex>
#include <vector>

using namespace admiral::detail;

// --- Cofactor-SIMD gate: an even cofactor is eligible from one batch tile up, no cap ---
// Written in tiles, not in literal M, because the batch width follows the ISA. Wc+2 and
// 3*Wc are both even and both at least one tile wide, so both are eligible; Wc-2 is even
// but narrower than a tile, and no other clause admits it.
static_assert(cofactor_simd_profitable<double, 4>(cofactor_batch_width<double, 4>() + 2));
static_assert(cofactor_simd_profitable<double, 4>(3 * cofactor_batch_width<double, 4>()));
static_assert(!cofactor_simd_profitable<double, 4>(cofactor_batch_width<double, 4>() - 2));
static_assert(cofactor_simd_profitable<float, 8>(cofactor_batch_width<float, 8>() + 2));
static_assert(cofactor_simd_profitable<float, 8>(3 * cofactor_batch_width<float, 8>()));
static_assert(!cofactor_simd_profitable<float, 8>(cofactor_batch_width<float, 8>() - 2));
// The six above all take the Wc == R exit, so they never reach the width gate. R=3 gives
// Wc > R for both precisions, which routes an even_m cofactor through a masked load. Which
// masked-load arm is live follows the ISA, so tie the assertion to Wc rather than to a
// literal. A 16-byte Wc is cheap and admits any even M, a 32-byte one first has to clear
// R*M >= kMaskedLoad256MinWork.
constexpr bool kMasked256F64 = cofactor_batch_width<double, 3>() * sizeof(double) > 16;
static_assert(kMasked256F64 == !cofactor_simd_profitable<double, 3>(8));  // 3*8 < 27
static_assert(cofactor_simd_profitable<double, 3>(10));  // 3*10 >= 27, the smallest even M
static_assert(cofactor_simd_profitable<float, 3>(cofactor_batch_width<float, 3>()));

namespace {

template<unsigned N, bool Forward, typename T>
void check_one() {
    const auto x = make_input<T>(N);
    std::vector<T> xre(N), xim(N), yre(N), yim(N);
    for (std::size_t i = 0; i < N; ++i) {
        xre[i] = x[i].real();
        xim[i] = x[i].imag();
    }

    kernel<N, T, Forward>::apply(xre.data(), xim.data(), 1, yre.data(), yim.data());

    const auto ref = reference_dft<T>(x, Forward);

    std::vector<std::complex<T>> got(N);
    for (std::size_t k = 0; k < N; ++k) got[k] = {yre[k], yim[k]};
    require_close(got, ref, fft_tol<T>());
}

template<unsigned N>
void check_size() {
    check_one<N, true, double>();
    check_one<N, false, double>();
    check_one<N, true, float>();
    check_one<N, false, float>();
}

} // namespace

TEST_CASE("codelet kernel<N> matches reference DFT", "[codelet]") {
    check_size<1>();
    check_size<2>();
    check_size<3>();
    check_size<4>();
    check_size<5>();
    check_size<6>();
    check_size<7>();
    check_size<8>();
    check_size<9>();
    check_size<10>();
    check_size<12>();
    check_size<15>();
    check_size<16>();
    check_size<24>();
    check_size<30>();
    check_size<32>();
    check_size<36>();
    check_size<48>();
    check_size<52>();  // cofactor-SIMD batched path (4 x flat radix-13) for double
    check_size<60>();
    check_size<64>();
    check_size<120>();
    check_size<128>();
}

TEST_CASE("codelet kernel<N> small primes (direct DFT path)", "[codelet]") {
    check_size<11>();
    check_size<13>();
}

// A non-pow2 leaf takes dif_butterfly_terminal instead of a cofactor combine, and the
// check_size cases above only cover that branch while the bound admits the size. Assert
// the bound, so widening or narrowing it cannot drop the coverage in silence. A pow2 leaf
// uses the whole file and every other leaf the usable part, so the two split at 32.
TEST_CASE("a small non-pow2 codelet takes the flat leaf", "[codelet]") {
    if constexpr (poet::vector_register_count() == 32) {
        CHECK(kernel_batched<11, double, true>::flat_leaf);        // 22 <= 24 usable
        CHECK_FALSE(kernel_batched<13, double, true>::flat_leaf);  // 26 > 24
        CHECK(kernel_batched<16, double, true>::flat_leaf);        // pow2: 32 <= 32
    }
    CHECK_FALSE(kernel_batched<CODELET_CATALOG_MAX, double, true>::flat_leaf);
}

// The scalar leaf competes with the cofactor-SIMD path rather than with a combine, so
// M > r decides it. The check_size cases cover both outcomes but not which one ran;
// assert the bound so a widening cannot silently take 10 away from the cofactor path.
TEST_CASE("the scalar flat leaf yields to a larger cofactor", "[codelet]") {
    if constexpr (poet::vector_register_count() == 32) {
        CHECK(kernel<11, double, true>::flat_leaf);        // prime, M == 1
        CHECK(kernel<12, double, true>::flat_leaf);        // 4x3: M <= r
        CHECK_FALSE(kernel<10, double, true>::flat_leaf);  // 2x5: M > r, cofactor batches 5
    }
    CHECK_FALSE(kernel<16, double, true>::flat_leaf);  // pow2 above the 8 cap, every ISA
    CHECK_FALSE(kernel<CODELET_CATALOG_MAX, double, true>::flat_leaf);
}

TEST_CASE("codelet kernel<N> prime composites (cofactor-SIMD batched path)", "[codelet]") {
    check_size<26>();  // 2*13
    check_size<34>();  // 2*17
    check_size<38>();  // 2*19
    check_size<39>();  // 3*13
    check_size<46>();  // 2*23
    check_size<51>();  // 3*17
    check_size<57>();  // 3*19
    check_size<58>();  // 2*29
    check_size<62>();  // 2*31
}

// Batched four-step (N = N1*N2, both <= 64 catalog, both multiples of W): the
// composition-of-kernels path for N > 64. Validate the index/twiddle math against
// the O(N^2) reference DFT on planar buffers.
template<unsigned N1, unsigned N2, typename T>
void check_four_step() {
    constexpr unsigned N = N1 * N2;
    const auto x = make_input<T>(N);
    std::vector<T> in_re(N), in_im(N), out_re(N), out_im(N), Gre(N), Gim(N);
    for (std::size_t i = 0; i < N; ++i) { in_re[i] = x[i].real(); in_im[i] = x[i].imag(); }

    auto run = [&](bool fwd) {
        constexpr unsigned W = static_cast<unsigned>(xsimd::batch<T>::size);
        if (fwd) {
            const auto tw = admiral::detail::build_four_step_twiddles_v<T, true>(N1, N2, W);
            std::vector<T> twre(N), twim(N);
            for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
            admiral::detail::four_step_batched_ct<N1, N2, T, true>(
                in_re.data(), in_im.data(), out_re.data(), out_im.data(),
                twre.data(), twim.data(), Gre.data(), Gim.data());
        } else {
            const auto tw = admiral::detail::build_four_step_twiddles_v<T, false>(N1, N2, W);
            std::vector<T> twre(N), twim(N);
            for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
            admiral::detail::four_step_batched_ct<N1, N2, T, false>(
                in_re.data(), in_im.data(), out_re.data(), out_im.data(),
                twre.data(), twim.data(), Gre.data(), Gim.data());
        }
    };

    for (bool fwd : {true, false}) {
        run(fwd);
        const auto ref = reference_dft<T>(x, fwd);
        std::vector<std::complex<T>> got(N);
        for (std::size_t k = 0; k < N; ++k) got[k] = {out_re[k], out_im[k]};
        require_close(got, ref, fft_tol<T>());
    }

    // Reference-free roundtrip: forward then inverse must recover x (up to 1/N).
    // No large-angle reference involved, so this is a tight check on the math.
    run(true);
    std::vector<T> f_re = out_re, f_im = out_im;
    for (std::size_t i = 0; i < N; ++i) { in_re[i] = f_re[i]; in_im[i] = f_im[i]; }
    run(false);
    std::vector<std::complex<T>> back(N);
    for (std::size_t i = 0; i < N; ++i) back[i] = {out_re[i] / T(N), out_im[i] / T(N)};
    require_close(back, x, fft_tol<T>());
}

TEST_CASE("batched four-step matches reference DFT (composition of kernels, N>64)", "[codelet][fourstep]") {
    check_four_step<16, 16, double>();   // 256
    check_four_step<32, 32, double>();   // 1024
    check_four_step<64, 64, double>();   // 4096
    check_four_step<32, 64, double>();   // 2048 (unequal factors)
    check_four_step<16, 16, float>();    // 256
    check_four_step<32, 32, float>();    // 1024
    check_four_step<32, 64, float>();    // 2048
}

// kernel<N>::apply_sink: the sink emits every output index exactly once (mixed
// scalar T and sized-batch V chunks). got starts poisoned, so a
// skipped or double-emitted index surfaces; values go against reference_dft and
// apply()'s own output within tolerance (the two paths reassociate differently
// under fast-math, so bitwise equality is not a property of the interface).
TEST_CASE("codelet kernel<N>::apply_sink matches apply and reference DFT", "[codelet]") {
    auto check = [](auto Nc, auto forward, auto tag) {
        constexpr unsigned N = decltype(Nc)::value;
        constexpr bool Forward = decltype(forward)::value;
        using T = decltype(tag);
        const auto x = make_input<T>(N);
        std::vector<T> xre(N), xim(N), scratch_re(N), scratch_im(N);
        std::vector<T> ref_re(N), ref_im(N), got_re(N, T(1e30)), got_im(N, T(1e30));
        for (std::size_t i = 0; i < N; ++i) { xre[i] = x[i].real(); xim[i] = x[i].imag(); }

        kernel<N, T, Forward>::apply(xre.data(), xim.data(), 1,
                                     ref_re.data(), ref_im.data());
        kernel<N, T, Forward>::apply_sink(
            xre.data(), xim.data(), 1, scratch_re.data(), scratch_im.data(),
            [&](std::size_t p, auto outr, auto outi) {
                using V = decltype(outr);
                if constexpr (std::is_same_v<V, T>) {
                    got_re[p] = outr;
                    got_im[p] = outi;
                } else {
                    outr.store_unaligned(got_re.data() + p);
                    outi.store_unaligned(got_im.data() + p);
                }
            });

        std::vector<std::complex<T>> got(N), via_apply(N);
        for (std::size_t k = 0; k < N; ++k) {
            got[k] = {got_re[k], got_im[k]};
            via_apply[k] = {ref_re[k], ref_im[k]};
        }
        require_close(got, reference_dft<T>(x, Forward), fft_tol<T>());
        require_close(got, via_apply, fft_tol<T>());
    };
    auto both = [&](auto Nc) {
        check(Nc, std::true_type{}, double{});
        check(Nc, std::false_type{}, double{});
        check(Nc, std::true_type{}, float{});
        check(Nc, std::false_type{}, float{});
    };
    both(std::integral_constant<unsigned, 2>{});
    both(std::integral_constant<unsigned, 6>{});
    both(std::integral_constant<unsigned, 12>{});  // cofactor tile-less (M=3 < Wc)
    both(std::integral_constant<unsigned, 13>{});  // Rader forward-through
    both(std::integral_constant<unsigned, 15>{});  // odd composite recursion
    both(std::integral_constant<unsigned, 16>{});  // cofactor single tile
    both(std::integral_constant<unsigned, 20>{});  // cofactor tile + scalar residue
    both(std::integral_constant<unsigned, 25>{});
    both(std::integral_constant<unsigned, 39>{});  // cofactor via batched Rader
    both(std::integral_constant<unsigned, 52>{});
    both(std::integral_constant<unsigned, 64>{});  // cofactor multi-tile
    both(std::integral_constant<unsigned, 128>{});
}

TEST_CASE("codelet forward then inverse is identity (up to 1/N)", "[codelet]") {
    auto roundtrip = [](auto Nc) {
        constexpr unsigned N = decltype(Nc)::value;
        using T = double;
        const auto x = make_input<T>(N);
        std::vector<T> xre(N), xim(N), fre(N), fim(N), gre(N), gim(N);
        for (std::size_t i = 0; i < N; ++i) { xre[i] = x[i].real(); xim[i] = x[i].imag(); }

        kernel<N, T, true>::apply(xre.data(), xim.data(), 1, fre.data(), fim.data());
        kernel<N, T, false>::apply(fre.data(), fim.data(), 1, gre.data(), gim.data());

        std::vector<std::complex<T>> back(N);
        for (std::size_t i = 0; i < N; ++i) back[i] = {gre[i] / T(N), gim[i] / T(N)};
        require_close(back, x, fft_tol<T>());
    };
    roundtrip(std::integral_constant<unsigned, 12>{});
    roundtrip(std::integral_constant<unsigned, 60>{});
    roundtrip(std::integral_constant<unsigned, 64>{});
}
