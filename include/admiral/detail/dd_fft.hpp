#pragma once

// Long double transforms on pairs of double SIMD lanes, for hosts whose long double is the x87
// 80-bit format (64 significand bits) and whose vector unit has FMA.
//
// All values of one transform sit on one grid: v = (hi + lo) * 2^-s, with hi an integer and
// |hi| <= 2^50. A sum adds hi and lo apart, and the hi sum is exact. A product with a constant
// |c| <= 1, c = ch + cl, rounds ch * xh to the grid with the 1.5 * 2^52 shift of Rump, Ogita and
// Oishi (SIAM J. Sci. Comput. 31 (2008) 189) and carries the remainder into lo, the common
// exponent split of Ozaki et al. (Numer. Algorithms 59 (2012) 95). hi + lo then holds about
// 100 bits against the largest value, past the 64 bits of the format.
//
// Reassociation folds the 1.5 * 2^52 shift and the constant split to nothing, so the TU that
// instantiates this engine must not build with fast math (src/inst_api_ld.cpp asserts it).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <poet/poet.hpp>

#include "butterfly.hpp"
#include "cxx_compat.hpp"
#include "scalar_fft.hpp"
#include "scratch.hpp"
#include "simd_swizzle.hpp"
#include "thread_pool.hpp"

#include "macros.hpp"

namespace admiral {
namespace detail {

inline constexpr bool kDdLongDouble = std::numeric_limits<long double>::digits == 64 &&
                                      sizeof(long double) == 16 && kFusedFma;

// Dependent on T, so only a TU that instantiates the engine checks it.
template<typename T>
inline constexpr bool kDdStrictFp = !XSIMD_REASSOCIATIVE_MATH;

using dd_batch = xsimd::batch<double>;
inline constexpr std::size_t kDdW = dd_batch::size;
// Doubles of one panel element: re.hi, re.lo, im.hi, im.lo, one batch each.
inline constexpr std::size_t kDdElem = 4 * kDdW;

struct gdd {
    dd_batch hi, lo;
    gdd() = default;
    ADM_ALWAYS_INLINE gdd(dd_batch h, dd_batch l) : hi(h), lo(l) {}
    ADM_ALWAYS_INLINE explicit gdd(ct_dd c) : hi(c.hi), lo(c.lo) {}
};

ADM_ALWAYS_INLINE gdd operator+(const gdd& a, const gdd& b) { return {a.hi + b.hi, a.lo + b.lo}; }
ADM_ALWAYS_INLINE gdd operator-(const gdd& a, const gdd& b) { return {a.hi - b.hi, a.lo - b.lo}; }
ADM_ALWAYS_INLINE gdd operator-(const gdd& a) { return {-a.hi, -a.lo}; }

// c is the constant, x the grid value. |ch * xh| < 2^51 puts ch * xh + 1.5 * 2^52 in
// [2^52, 2^53), where the ulp is 1, so p is ch * xh rounded to an integer.
ADM_ALWAYS_INLINE gdd operator*(const gdd& c, const gdd& x) {
    const dd_batch shift(0x1.8p52);
    const dd_batch p = xsimd::fma(c.hi, x.hi, shift) - shift;
    const dd_batch e = xsimd::fms(c.hi, x.hi, p);
    return {p, xsimd::fma(c.hi, x.lo, xsimd::fma(c.lo, x.hi, e))};
}

ADM_ALWAYS_INLINE gdd piece_fma(const gdd& a, const gdd& b, const gdd& c) { return a * b + c; }
ADM_ALWAYS_INLINE gdd piece_fnma(const gdd& a, const gdd& b, const gdd& c) { return c - a * b; }

ADM_ALWAYS_INLINE void dd_load(const double* p, gdd& re, gdd& im) {
    re = {dd_batch::load_aligned(p), dd_batch::load_aligned(p + kDdW)};
    im = {dd_batch::load_aligned(p + 2 * kDdW), dd_batch::load_aligned(p + 3 * kDdW)};
}

ADM_ALWAYS_INLINE void dd_store(double* p, const gdd& re, const gdd& im) {
    re.hi.store_aligned(p);
    re.lo.store_aligned(p + kDdW);
    im.hi.store_aligned(p + 2 * kDdW);
    im.lo.store_aligned(p + 3 * kDdW);
}

ADM_ALWAYS_INLINE void dd_cmul(const gdd& wr, const gdd& wi, gdd& yr, gdd& yi) {
    const gdd r = wr * yr - wi * yi;
    yi = wr * yi + wi * yr;
    yr = r;
}

// A complex constant split over two doubles per part.
struct dd_const {
    double rh, rl, ih, il;
};

inline dd_const dd_split(const ct_sincos_v<ct_dd>& w) { return {w.c.hi, w.c.lo, w.s.hi, w.s.lo}; }

inline dd_const dd_conj(dd_const w) { return {w.rh, w.rl, -w.ih, -w.il}; }

// w_n^e = exp(-2 pi i e / n) for every e from 2 sqrt(n) series runs: w^e = w^(B (e / B)) w^(e % B).
class dd_roots {
public:
    explicit dd_roots(std::size_t n) : n_(n) {
        while (b_ * b_ < n_) ++b_;
        for (std::size_t e = 0; e < n_; e += b_)
            coarse_.push_back(sincos_turns_eval<ct_dd>(true, e, n_));
        for (std::size_t e = 0; e < b_; ++e) fine_.push_back(sincos_turns_eval<ct_dd>(true, e, n_));
    }

    [[nodiscard]] dd_const operator()(std::size_t e) const {
        e %= n_;
        const ct_sincos_v<ct_dd>& a = coarse_[e / b_];
        const ct_sincos_v<ct_dd>& b = fine_[e % b_];
        return dd_split({a.c * b.s + a.s * b.c, a.c * b.c - a.s * b.s});
    }

private:
    std::size_t n_, b_ = 1;
    std::vector<ct_sincos_v<ct_dd>> coarse_, fine_;
};

ADM_ALWAYS_INLINE void dd_twiddle(const dd_const& w, gdd& yr, gdd& yi) {
    dd_cmul({dd_batch(w.rh), dd_batch(w.rl)}, {dd_batch(w.ih), dd_batch(w.il)}, yr, yi);
}

// The grid of one transform: hi + lo = v * 2^s. emax is the largest biased x87 exponent of the
// input, so every |v| < 2^(emax - 16382), and 4 * growth * max|v| * 2^s <= 2^50 bounds every
// value of the transform. The output sits on the grid so = s - down and leaves multiplied by
// f / div = (fh + fl) * 2^q, to about 2^-104: a long double 1 / div would round the scale first.
struct dd_grid {
    dd_grid(unsigned emax, std::size_t growth, long double f, int down = 0, std::size_t div = 1)
        : fct(f / static_cast<long double>(div)) {
        s = 48 - bit_width(growth - 1) - (static_cast<int>(std::max(emax, 1u)) - 16382);
        so = s - down;
        int q = 0;
        const long double m = std::frexp(f, &q);
        const double mh = static_cast<double>(m);
        const ct_dd t =
            ct_dd(mh, static_cast<double>(m - static_cast<long double>(mh))) / ct_dd(div);
        fh = t.hi;
        fl = t.lo;
        out_bias = 16383 - 1023 + q - so;
    }
    int s, so, out_bias;
    double fh, fl;
    long double fct;
};

inline constexpr unsigned kLdExpInfNan = 0x7FFF;

// Keeps the exponent of each long double read as four 32-bit words (the third holds sign and
// exponent, the fourth is padding).
struct ld_exp_lane {
    static constexpr std::uint32_t get(std::size_t i, std::size_t) {
        return i % 4 == 2 ? 0x7FFF : 0;
    }
};

inline unsigned ld_max_exponent(const std::complex<long double>* x, std::size_t n) {
    using u32 = xsimd::batch<std::uint32_t>;
    constexpr std::size_t per = u32::size / 4;
    const auto* w = reinterpret_cast<const std::uint32_t*>(x);
    const u32 mask = xsimd::make_batch_constant<std::uint32_t, ld_exp_lane, u32::arch_type>();
    u32 acc(0);
    std::size_t i = 0;
    for (; i + per <= 2 * n; i += per) acc = xsimd::max(acc, u32::load_unaligned(w + 4 * i) & mask);
    // The fewer than per values left; the lanes past them read zero.
    const auto tail = lane_prefix_mask<std::uint32_t>(4 * (2 * n - i));
    acc = xsimd::max(acc, u32::load(w + 4 * i, tail, xsimd::unaligned_mode{}) & mask);
    return xsimd::reduce_max(acc);
}

using dd_u64 = xsimd::batch<std::uint64_t>;
using dd_i64 = xsimd::batch<std::int64_t>;

// Batch c of the 64-bit words of the first `valid` complex long doubles, four words each.
ADM_ALWAYS_INLINE xsimd::batch_bool<std::uint64_t> ld_word_mask(std::size_t valid,
                                                                std::size_t c) {
    return lane_prefix_mask<std::uint64_t>(4 * valid > c * kDdW ? 4 * valid - c * kDdW : 0);
}

// The x87 fields of the first `valid` of kDdW complex long doubles, the other lanes zero:
// significand words m and sign/exponent words e.
ADM_ALWAYS_INLINE void ld_fields_load(const std::complex<long double>* p, std::size_t valid,
                                      dd_u64& mr, dd_u64& er, dd_u64& mi, dd_u64& ei) {
    using arch = dd_u64::arch_type;
    const auto* w = reinterpret_cast<const std::uint64_t*>(p);
    const auto even = xsimd::make_batch_constant<std::uint64_t, aos_even_lane, arch>();
    const auto odd = xsimd::make_batch_constant<std::uint64_t, aos_odd_lane, arch>();
    dd_u64 b[4];
    if (valid == kDdW) {
        for (std::size_t c = 0; c < 4; ++c) b[c] = dd_u64::load_unaligned(w + c * kDdW);
    } else {
        for (std::size_t c = 0; c < 4; ++c)
            b[c] = dd_u64::load(w + c * kDdW, ld_word_mask(valid, c), xsimd::unaligned_mode{});
    }
    const dd_u64 m01 = xsimd::shuffle(b[0], b[1], even), e01 = xsimd::shuffle(b[0], b[1], odd);
    const dd_u64 m23 = xsimd::shuffle(b[2], b[3], even), e23 = xsimd::shuffle(b[2], b[3], odd);
    mr = xsimd::shuffle(m01, m23, even);
    mi = xsimd::shuffle(m01, m23, odd);
    er = xsimd::shuffle(e01, e23, even);
    ei = xsimd::shuffle(e01, e23, odd);
}

// Writes the first `valid` of kDdW complex long doubles only. A full batch keeps the plain
// store: a ymm vpmaskmovq store is 18 or 19 uops on Zen+ to Zen 4 (uops.info), 3 on Intel.
ADM_ALWAYS_INLINE void ld_fields_store(std::complex<long double>* p, std::size_t valid,
                                       dd_u64 mr, dd_u64 er, dd_u64 mi, dd_u64 ei) {
    auto* w = reinterpret_cast<std::uint64_t*>(p);
    const dd_u64 mlo = xsimd::zip_lo(mr, mi), mhi = xsimd::zip_hi(mr, mi);
    const dd_u64 elo = xsimd::zip_lo(er, ei), ehi = xsimd::zip_hi(er, ei);
    const dd_u64 b[4] = {xsimd::zip_lo(mlo, elo), xsimd::zip_hi(mlo, elo), xsimd::zip_lo(mhi, ehi),
                         xsimd::zip_hi(mhi, ehi)};
    if (valid == kDdW) {
        for (std::size_t c = 0; c < 4; ++c) b[c].store_unaligned(w + c * kDdW);
    } else {
        for (std::size_t c = 0; c < 4; ++c)
            b[c].store(w + c * kDdW, ld_word_mask(valid, c), xsimd::unaligned_mode{});
    }
}

// hi + lo = v * 2^s, v = m * 2^(max(E, 1) - 16446). The two halves of m convert exactly through
// the 2^52 bit pattern. A lane under 2^-958 on the grid flushes to zero.
ADM_ALWAYS_INLINE void dd_from_fields(dd_u64 m, dd_u64 e, int s, dd_batch& hi, dd_batch& lo) {
    const dd_u64 two52_bits(0x4330000000000000);
    const dd_batch two52(0x1p52), shift(0x1.8p52);
    const dd_batch mh = xsimd::bitwise_cast<double>((m >> 12) | two52_bits) - two52;
    const dd_batch ml = xsimd::bitwise_cast<double>((m & dd_u64(0xFFF)) | two52_bits) - two52;
    const dd_i64 ex = xsimd::max(xsimd::bitwise_cast<std::int64_t>(e & dd_u64(0x7FFF)), dd_i64(1)) +
                      dd_i64(std::int64_t{s} - 16446 + 1023);
    const dd_batch unit = xsimd::bitwise_cast<double>(xsimd::max(ex, dd_i64(0)) << 52);
    const dd_batch a = mh * (unit * dd_batch(0x1p12)), b = ml * unit;
    const dd_batch h = (a + shift) - shift;
    const dd_batch sign = xsimd::bitwise_cast<double>((e & dd_u64(0x8000)) << 48);
    hi = h ^ sign;
    lo = ((a - h) + b) ^ sign;
}

// The x87 fields of (hi + lo) * (fh + fl) * 2^(q - s), rounded once to 64 bits. False when a lane
// leaves the normal x87 range; the caller then converts that batch one lane at a time.
ADM_ALWAYS_INLINE bool dd_to_fields(dd_batch hi, dd_batch lo, const dd_grid& g, dd_u64& m,
                                    dd_u64& e) {
    const dd_batch fh(g.fh), fl(g.fl), shift(0x1.8p52);
    const dd_batch sum = hi + lo, bv = sum - hi;
    const dd_batch err = (hi - (sum - bv)) + (lo - bv);
    const dd_batch p = sum * fh;
    const dd_batch pe = xsimd::fma(sum, fl, xsimd::fma(err, fh, xsimd::fms(sum, fh, p)));
    const dd_batch r = p + pe, re = pe - (r - p);
    const dd_u64 rb = xsimd::bitwise_cast<std::uint64_t>(r);
    const dd_u64 er = (rb >> 52) & dd_u64(0x7FF);
    const dd_u64 m53 = (rb & dd_u64(0xFFFFFFFFFFFFF)) | dd_u64(std::uint64_t{1} << 52);
    // -sign(r) * 2^(64 - e_r), e_r the exponent of r: (4158 - sign:e_r) mod 4096 is 2110 - e_r
    // with the sign bit of r inverted. The fnma then puts re * sign(r), the residual of the
    // magnitude, in units of half an x87 ulp of r. Exact for er >= 64; below that the lane
    // flushes to zero.
    const dd_batch unit = xsimd::bitwise_cast<double>((dd_u64(4158) - (rb >> 52)) << 52);
    const dd_i64 shift_bits = xsimd::bitwise_cast<std::int64_t>(shift);
    const dd_i64 d64 = xsimd::bitwise_cast<std::int64_t>(xsimd::fnma(re, unit, shift)) - shift_bits;
    const dd_i64 d63 =
        xsimd::bitwise_cast<std::int64_t>(xsimd::fnma(re, unit * dd_batch(0.5), shift)) -
        shift_bits;
    // r a power of two and re negative: the value sits one binade lower, at twice the resolution.
    const auto borrow = xsimd::batch_bool_cast<std::uint64_t>(d64 < dd_i64(0)) &&
                        (m53 == dd_u64(std::uint64_t{1} << 52));
    const dd_u64 mant =
        xsimd::select(borrow, xsimd::bitwise_cast<std::uint64_t>(d64),
                      (m53 << 11) + xsimd::bitwise_cast<std::uint64_t>(d63));
    const dd_i64 ex =
        xsimd::bitwise_cast<std::int64_t>(er) + dd_i64(g.out_bias) -
        xsimd::select(xsimd::batch_bool_cast<std::int64_t>(borrow), dd_i64(1), dd_i64(0));
    const auto zero = er < dd_u64(64);
    m = xsimd::select(zero, dd_u64(0), mant);
    e = xsimd::select(zero, dd_u64(0), xsimd::bitwise_cast<std::uint64_t>(ex)) | ((rb >> 63) << 15);
    const auto bad = (ex < dd_i64(1)) || (ex > dd_i64(32766)) ||
                     xsimd::batch_bool_cast<std::int64_t>(er == dd_u64(0x7FF));
    return xsimd::none(bad && !xsimd::batch_bool_cast<std::int64_t>(zero));
}

// One panel element from the first `valid` of kDdW complex long doubles, the other lanes zero.
// The inverse swaps re and im on the way in and out, so the engine only runs forward.
template<bool forward>
ADM_ALWAYS_INLINE void dd_load_ld(const std::complex<long double>* p, std::size_t valid, int s,
                                  double* dst) {
    dd_u64 mr, er, mi, ei;
    ld_fields_load(p, valid, mr, er, mi, ei);
    if constexpr (!forward) {
        std::swap(mr, mi);
        std::swap(er, ei);
    }
    gdd re, im;
    dd_from_fields(mr, er, s, re.hi, re.lo);
    dd_from_fields(mi, ei, s, im.hi, im.lo);
    dd_store(dst, re, im);
}

// The panel element src to the first `valid` of kDdW complex long doubles at p.
template<bool forward>
ADM_ALWAYS_INLINE void dd_store_ld(const double* src, const dd_grid& g,
                                   std::complex<long double>* p, std::size_t valid) {
    gdd re, im;
    dd_load(src, re, im);
    if constexpr (!forward) std::swap(re, im);
    dd_u64 mr, er, mi, ei;
    const bool fast = dd_to_fields(re.hi, re.lo, g, mr, er);
    if (dd_to_fields(im.hi, im.lo, g, mi, ei) && fast) {
        ld_fields_store(p, valid, mr, er, mi, ei);
        return;
    }
    // One rounding of v = (h + l) * (fh + fl) * 2^e, v to about 2^-104 in double-double: to 64
    // bits for a normal result, to the subnormal grid 2^-16445 otherwise. There the residual t
    // of the 64-bit sum decides a tie.
    const int e = g.out_bias - 16383 + 1023;
    const auto slow = [&g, e](double h, double l) -> long double {
        // A comparison, not isfinite: clang rejects isfinite where -ffinite-math-only parses this.
        if (!(std::fabs(g.fct) <= std::numeric_limits<long double>::max()))
            return (static_cast<long double>(h) + static_cast<long double>(l)) * g.fct;
        const ct_dd v = dd_two_sum(h, l) * ct_dd(g.fh, g.fl);
        const long double a = std::ldexp(static_cast<long double>(v.hi), e + 16445),
                          b = std::ldexp(static_cast<long double>(v.lo), e + 16445);
        const long double s = a + b;
        if (!(std::fabs(s) < 0x1p63L))
            return std::ldexp(static_cast<long double>(v.hi) + static_cast<long double>(v.lo), e);
        const long double t = (a - s) + b, i = std::nearbyint(s), f = s - i;
        const bool away = std::fabs(f) == 0.5L && t != 0 && std::signbit(t) == std::signbit(f);
        return std::ldexp(away ? i + 2 * f : i, -16445);
    };
    const std::size_t r = forward ? 0 : 2, i = forward ? 2 : 0;
    for (std::size_t l = 0; l < valid; ++l)
        p[l] = {slow(src[r * kDdW + l], src[(r + 1) * kDdW + l]),
                slow(src[i * kDdW + l], src[(i + 1) * kDdW + l])};
}

// Forward DFT of length n over the kDdW lanes of a panel, the DIF recursion of scalar_c2c at
// V = gdd. Every twiddle is lane-uniform. The output is digit-reversed: natural index k sits
// at element pos(k).
template<typename T>
class dd_lines {
public:
    explicit dd_lines(std::size_t n) : n_(n) {
        static_assert(kDdStrictFp<T>, "the dd engine needs -fno-fast-math");
        const dd_roots w(n_);
        std::size_t m = n_;
        while (m > 1) {
            const std::size_t p = factor(m);
            if (p == 0) break;
            level_state ls{p, m, std::vector<dd_const>(m)};
            for (std::size_t j = 0; j < m; ++j) ls.tw[j] = w(j * (n_ / m));
            levels_.push_back(std::move(ls));
            m /= p;
        }
        leaf_ = m;
        if (leaf_ > kDirectMax) {
            blue_ = std::make_unique<blue_state>(leaf_);
        } else if (leaf_ > 1) {
            dft_tab_.resize(leaf_);
            for (std::size_t j = 0; j < leaf_; ++j) dft_tab_[j] = dd_conj(w(j * (n_ / leaf_)));
        }
        build_perm();
    }

    [[nodiscard]] std::size_t size() const { return n_; }
    [[nodiscard]] std::size_t pos(std::size_t k) const { return pos_of_[k]; }
    [[nodiscard]] std::size_t work_doubles() const { return blue_ ? 2 * blue_->pad * kDdElem : 0; }

    // Vector operations of one panel run, the split cost of the four-step driver.
    [[nodiscard]] static double cost(std::size_t n) {
        double c = 0;
        std::size_t m = n;
        for (std::size_t p; m > 1 && (p = factor(m)) != 0; m /= p)
            c += double(n) * std::log2(double(p));
        if (m > kDirectMax) {
            const double pad = double(blue_pad(m));
            c += double(n / m) * (2 * pad * std::log2(pad) + 6 * pad);
        } else if (m > 1) {
            c += double(n) * double(m) / 2;
        }
        return c;
    }

    void run(double* x, double* work) const {
        if (levels_.empty()) leaf(x, work);
        else dft(x, 0, 0, work);
    }

private:
    static std::size_t factor(std::size_t m) {
        for (const std::size_t p : {8u, 3u, 5u, 7u, 4u, 2u})
            if (m % p == 0) return p;
        return 0;
    }

    static std::size_t blue_pad(std::size_t n) {
        for (std::size_t cand = 2 * n - 1;; ++cand) {
            std::size_t v = cand;
            for (const std::size_t p : {4u, 3u, 5u, 7u, 2u})
                while (v % p == 0) v /= p;
            if (v == 1) return cand;
        }
    }

    void leaf(double* x, double* work) const {
        if (blue_) blue_->run(x, work);
        else if (leaf_ > 1) direct(x);
    }

    template<std::size_t P>
    void combine(double* x, std::size_t off, std::size_t level) const {
        const level_state& lv = levels_[level];
        const std::size_t u = lv.m / P;
        const dd_const* tw = lv.tw.data();
        double* base = x + off * kDdElem;
        for (std::size_t q = 0; q < u; ++q) {
            gdd ar[P], ai[P];
            for (std::size_t i = 0; i < P; ++i) dd_load(base + (q + i * u) * kDdElem, ar[i], ai[i]);
            sub_dft<ct_dd, P>(ar, ai, [&](auto sc, gdd yr, gdd yi) {
                if constexpr (sc != 0) dd_twiddle(tw[q * sc], yr, yi);
                dd_store(base + (q + sc * u) * kDdElem, yr, yi);
            });
        }
    }

    using radix_set = std::integer_sequence<std::size_t, 2, 3, 4, 5, 7, 8>;

    struct combine_invoke_t {
        template<std::size_t P>
        void operator()(const dd_lines* self, double* x, std::size_t off, std::size_t level) const {
            self->template combine<P>(x, off, level);
        }
    };

    void dft(double* x, std::size_t off, std::size_t level, double* work) const {
        const level_state& lv = levels_[level];
        poet::dispatch(poet::throw_on_no_match, combine_invoke_t{},
                       poet::dispatch_param<radix_set>{lv.p}, this, x, off, level);
        const std::size_t u = lv.m / lv.p;
        if (u == 1) return;
        for (std::size_t s = 0; s < lv.p; ++s) {
            if (level + 1 == levels_.size()) leaf(x + (off + s * u) * kDdElem, work);
            else dft(x, off + s * u, level + 1, work);
        }
    }

    // Singleton's symmetric DFT of a prime leaf, as radix_sym_dft with runtime constants.
    void direct(double* x) const {
        const std::size_t P = leaf_, H = (P - 1) / 2;
        gdd ar[kDirectMax / 2 + 1], ai[kDirectMax / 2 + 1];
        gdd dr[kDirectMax / 2 + 1], di[kDirectMax / 2 + 1];
        gdd x0r, x0i;
        dd_load(x, x0r, x0i);
        gdd sr = x0r, si = x0i;
        for (std::size_t m = 1; m <= H; ++m) {
            gdd pr, pi, qr, qi;
            dd_load(x + m * kDdElem, pr, pi);
            dd_load(x + (P - m) * kDdElem, qr, qi);
            ar[m] = pr + qr;
            ai[m] = pi + qi;
            dr[m] = pr - qr;
            di[m] = pi - qi;
            sr = sr + ar[m];
            si = si + ai[m];
        }
        dd_store(x, sr, si);
        for (std::size_t k = 1; k <= H; ++k) {
            gdd pr = x0r, pi = x0i, qr(ct_dd(0)), qi(ct_dd(0));
            for (std::size_t m = 1; m <= H; ++m) {
                const dd_const& w = dft_tab_[(m * k) % P];
                const gdd c{dd_batch(w.rh), dd_batch(w.rl)}, s{dd_batch(w.ih), dd_batch(w.il)};
                pr = c * ar[m] + pr;
                pi = c * ai[m] + pi;
                qr = s * di[m] + qr;
                qi = s * dr[m] + qi;
            }
            dd_store(x + k * kDdElem, pr + qr, pi - qi);
            dd_store(x + (P - k) * kDdElem, pr - qr, pi + qi);
        }
    }

    void build_perm() {
        pos_of_.resize(n_);
        for (std::size_t j = 0; j < n_; ++j) {
            std::size_t residual = j, stride_k = 1, natural = 0;
            for (const level_state& lv : levels_) {
                const std::size_t u = lv.m / lv.p;
                const std::size_t s = residual / u;
                residual -= s * u;
                natural += s * stride_k;
                stride_k *= lv.p;
            }
            natural += residual * stride_k;
            pos_of_[natural] = j;
        }
    }

    // Bluestein's chirp-z leaf. The inverse transform is swap(forward(swap)). Both runs stay on
    // the grid of the line, see dd_c2c::build_blue.
    struct blue_state {
        explicit blue_state(std::size_t len) : m(len), pad(blue_pad(len)), inner(pad) {
            const dd_roots w(2 * m);
            chirp.resize(m);
            for (std::size_t k = 0; k < m; ++k) chirp[k] = w(k * k);
            // FFT(b), b the conjugate chirp wrapped around pad, by this engine on a grid that
            // holds |b| <= 1 grown by pad. |FFT(b)| <= 2m - 1 <= pad, so every product constant
            // bmul is at most 1.
            const int sb = 48 - bit_width(pad - 1);
            auto panel = make_aligned_buffer<double>(pad * kDdElem);
            const auto put = [sb](double* p, double h, double l) {
                const double a = std::ldexp(h, sb), r = std::nearbyint(a);
                std::fill_n(p, kDdW, r);
                std::fill_n(p + kDdW, kDdW, (a - r) + std::ldexp(l, sb));
            };
            std::fill_n(panel.get(), pad * kDdElem, 0.0);
            for (std::size_t k = 0; k < m; ++k)
                for (const std::size_t j : {k, (pad - k) % pad}) {
                    put(panel.get() + j * kDdElem, chirp[k].rh, chirp[k].rl);
                    put(panel.get() + j * kDdElem + 2 * kDdW, -chirp[k].ih, -chirp[k].il);
                }
            inner.run(panel.get(), nullptr);
            bmul.resize(pad);
            const ct_dd scale = ct_dd(std::ldexp(1.0, -sb)) / ct_dd(pad);
            for (std::size_t k = 0; k < pad; ++k) {
                const double* e = panel.get() + inner.pos(k) * kDdElem;
                const ct_dd re = dd_two_sum(e[0], e[kDdW]) * scale;
                const ct_dd im = dd_two_sum(e[2 * kDdW], e[3 * kDdW]) * scale;
                bmul[k] = {re.hi, re.lo, im.hi, im.lo};
            }
        }

        void run(double* x, double* work) const {
            double* a = work;
            double* z = work + pad * kDdElem;
            for (std::size_t j = 0; j < m; ++j) {
                gdd re, im;
                dd_load(x + j * kDdElem, re, im);
                dd_twiddle(chirp[j], re, im);
                dd_store(a + j * kDdElem, re, im);
            }
            std::fill(a + m * kDdElem, a + pad * kDdElem, 0.0);
            inner.run(a, nullptr);
            for (std::size_t j = 0; j < pad; ++j) {
                gdd re, im;
                dd_load(a + inner.pos(j) * kDdElem, re, im);
                dd_twiddle(bmul[j], re, im);
                dd_store(z + j * kDdElem, im, re);
            }
            inner.run(z, nullptr);
            for (std::size_t j = 0; j < m; ++j) {
                gdd re, im;
                dd_load(z + inner.pos(j) * kDdElem, im, re);
                dd_twiddle(chirp[j], re, im);
                dd_store(x + j * kDdElem, re, im);
            }
        }

        std::size_t m, pad;
        dd_lines inner;
        std::vector<dd_const> chirp, bmul;
    };

    struct level_state {
        std::size_t p, m;
        std::vector<dd_const> tw;
    };

    std::size_t n_, leaf_ = 1;
    std::vector<level_state> levels_;
    std::vector<std::size_t> pos_of_;
    std::vector<dd_const> dft_tab_;
    std::unique_ptr<blue_state> blue_;
};

// Four-step over panels: n = n1 * n2, input j1 * n2 + j2, output k1 + n1 * k2. Pass A runs
// n1-point columns over kDdW lanes j2, twists by w_n^(k1 j2) and transposes kDdW x kDdW blocks
// into panels of kDdW rows k1. Pass B runs each n2-point panel in place. The caller moves the
// data: src(j, lanes, dst) writes the panel element of inputs j .. j + lanes - 1 with the other
// lanes zero, and sink(k, lanes, src) takes the panel element of outputs k .. k + lanes - 1.
template<typename T>
class dd_four_step {
public:
    struct split_t {
        std::size_t n1;
        double cost;
    };

    // blocked: n1 and n2 are multiples of kDdW, so the lanes of every element are one aligned
    // block of kDdW consecutive indices.
    dd_four_step(std::size_t n, bool blocked)
        : n_(n), n1_(split(n, blocked).n1), n2_(n / n1_), nb1_((n1_ + kDdW - 1) / kDdW),
          nb2_((n2_ + kDdW - 1) / kDdW), a_(n1_), b_(n2_) {
        tw_ = make_aligned_buffer<double>(nb2_ * n1_ * kDdElem);
        const dd_roots root(n_);
        for (std::size_t jb = 0; jb < nb2_; ++jb)
            for (std::size_t k1 = 0; k1 < n1_; ++k1) {
                double* t = tw_.get() + (jb * n1_ + k1) * kDdElem;
                for (std::size_t l = 0; l < kDdW; ++l) {
                    const std::size_t j2 = jb * kDdW + l;
                    const dd_const w = root(j2 < n2_ ? k1 * j2 : 0);
                    t[l] = w.rh;
                    t[kDdW + l] = w.rl;
                    t[2 * kDdW + l] = w.ih;
                    t[3 * kDdW + l] = w.il;
                }
            }
    }

    // Panel lines above this length leave L2: 4096 * kDdElem doubles is 1 MiB at kDdW = 8.
    static constexpr std::size_t kLineCap = 4096;

    static split_t split(std::size_t n, bool blocked) {
        split_t best{0, 0};
        bool best_over = true;
        for (std::size_t d = 1; d * d <= n; ++d) {
            if (n % d) continue;
            for (const std::size_t n1 : {d, n / d}) {
                const std::size_t n2 = n / n1;
                if (blocked && (n1 % kDdW != 0 || n2 % kDdW != 0)) continue;
                const bool over = std::max(n1, n2) > kLineCap;
                const double c = double((n2 + kDdW - 1) / kDdW) * dd_lines<T>::cost(n1) +
                                 double((n1 + kDdW - 1) / kDdW) * dd_lines<T>::cost(n2);
                if (best.n1 == 0 ? true
                    : over != best_over ? !over
                    : c != best.cost    ? c < best.cost
                                        : std::max(n1, n2) < std::max(best.n1, n / best.n1)) {
                    best = {n1, c};
                    best_over = over;
                }
            }
        }
        return best;
    }

    template<typename Src, typename Sink>
    void run(const Src& src, const Sink& sink, thread_pool* pool) const {
        if (n1_ >= kDdW && n2_ >= kDdW) {
            run_panels(src, sink, pool);
            return;
        }
        // A side of fewer than kDdW lanes would pass the caller one partial batch per element.
        // The staging buffer holds all n elements in blocks of kDdW consecutive indices, and
        // the caller moves full blocks. That side has one block of panels, so it runs serially
        // either way.
        const std::size_t nb = (n_ + kDdW - 1) / kDdW;
        soa_scratch<double, 1> st_sc(nb * kDdElem);
        double* st = st_sc.buf(0);
        // The caller converts whole batches: lanes past n must hold zero, not stack garbage that
        // sends the batch down the per-lane path.
        std::fill_n(st + (nb - 1) * kDdElem, kDdElem, 0.0);
        const auto at =[st](std::size_t e, std::size_t c) -> double& {
            return st[e / kDdW * kDdElem + c * kDdW + e % kDdW];
        };
        const auto blocks = [&](const auto& f) {
            for (std::size_t b = 0; b < nb; ++b)
                f(b * kDdW, std::min(kDdW, n_ - b * kDdW), st + b * kDdElem);
        };
        if (n2_ < kDdW) blocks(src);
        run_panels(
            [&](std::size_t j, std::size_t lanes, double* dst) {
                if (n2_ >= kDdW) {
                    src(j, lanes, dst);
                    return;
                }
                for (std::size_t c = 0; c < 4; ++c)
                    for (std::size_t l = 0; l < kDdW; ++l)
                        dst[c * kDdW + l] = l < lanes ? at(j + l, c) : 0.0;
            },
            [&](std::size_t k, std::size_t lanes, double* p) {
                if (n1_ >= kDdW) {
                    sink(k, lanes, p);
                    return;
                }
                for (std::size_t c = 0; c < 4; ++c)
                    for (std::size_t l = 0; l < lanes; ++l) at(k + l, c) = p[c * kDdW + l];
            },
            pool);
        if (n1_ < kDdW) blocks(sink);
    }

private:
    template<typename Src, typename Sink>
    void run_panels(const Src& src, const Sink& sink, thread_pool* pool) const {
        soa_scratch<double, 1> at_sc(nb1_ * n2_ * kDdElem);
        double* at = at_sc.buf(0);
        parallel_for(pool, nb2_, n_, [&](std::size_t begin, std::size_t end, std::size_t) {
            soa_scratch<double, 1> sc(n1_ * kDdElem + a_.work_doubles());
            for (std::size_t jb = begin; jb < end; ++jb) pass_a(src, at, jb, sc.buf(0));
        });
        parallel_for(pool, nb1_, n_, [&](std::size_t begin, std::size_t end, std::size_t) {
            soa_scratch<double, 1> sc(b_.work_doubles());
            for (std::size_t kb = begin; kb < end; ++kb) pass_b(sink, at, kb, sc.buf(0));
        });
    }

    template<typename Src>
    void pass_a(const Src& src, double* at, std::size_t jb, double* buf) const {
        const std::size_t j0 = jb * kDdW, lanes = std::min(kDdW, n2_ - j0);
        for (std::size_t j1 = 0; j1 < n1_; ++j1) src(j1 * n2_ + j0, lanes, buf + j1 * kDdElem);
        a_.run(buf, buf + n1_ * kDdElem);
        const double* tw = tw_.get() + jb * n1_ * kDdElem;
        alignas(dd_batch::arch_type::alignment()) double blk[kDdW * kDdElem];
        for (std::size_t kb = 0; kb < nb1_; ++kb) {
            for (std::size_t r = 0; r < kDdW; ++r) {
                const std::size_t k1 = kb * kDdW + r;
                gdd re{dd_batch(0.0), dd_batch(0.0)}, im = re;
                if (k1 < n1_) {
                    gdd wr, wi;
                    dd_load(buf + a_.pos(k1) * kDdElem, re, im);
                    dd_load(tw + k1 * kDdElem, wr, wi);
                    dd_cmul(wr, wi, re, im);
                }
                dd_store(blk + r * kDdElem, re, im);
            }
            double* dst = at + (kb * n2_ + j0) * kDdElem;
            for (std::size_t c = 0; c < 4; ++c) {
                dd_batch m[kDdW];
                for (std::size_t r = 0; r < kDdW; ++r)
                    m[r] = dd_batch::load_aligned(blk + r * kDdElem + c * kDdW);
                xsimd::transpose(m, m + kDdW);
                for (std::size_t l = 0; l < lanes; ++l)
                    m[l].store_aligned(dst + l * kDdElem + c * kDdW);
            }
        }
    }

    template<typename Sink>
    void pass_b(const Sink& sink, double* at, std::size_t kb, double* work) const {
        const std::size_t k0 = kb * kDdW, lanes = std::min(kDdW, n1_ - k0);
        double* panel = at + kb * n2_ * kDdElem;
        b_.run(panel, work);
        for (std::size_t k2 = 0; k2 < n2_; ++k2)
            sink(k0 + n1_ * k2, lanes, panel + b_.pos(k2) * kDdElem);
    }

    std::size_t n_, n1_, n2_, nb1_, nb2_;
    dd_lines<T> a_, b_;
    aligned_buffer<double> tw_;
};

// Multiplies a panel element by the lane-varying constants of one table element.
ADM_ALWAYS_INLINE void dd_cmul_table(const double* w, double* x) {
    gdd wr, wi, re, im;
    dd_load(w, wr, wi);
    dd_load(x, re, im);
    dd_cmul(wr, wi, re, im);
    dd_store(x, re, im);
}

// The shortest c2c the dd engine runs. Below it the x87 engine is faster: the panels fill few
// lanes, and the conversions and transposes cost more than the transform. An odd r2c of 9 or 11
// is slower than x87 too but keeps the dd engine for its 0.5 ulp.
inline constexpr std::size_t kDdMinLen = 8;

// 1-D long double c2c. The four-step runs n directly, or, when its cost model prices that higher,
// runs Bluestein's chirp-z over a blocked four-step of length pad >= 2n - 1: a prime n gives
// the direct four-step one lane of kDdW. The interface is scalar_c2c's.
template<typename T>
class dd_c2c {
public:
    explicit dd_c2c(std::size_t n, std::size_t = 1)
        : n_(n), pad_(blue_pad(n)), fs_(pad_ != 0 ? pad_ : n_, pad_ != 0) {
        if (n_ < kDdMinLen) x87_ = std::make_unique<scalar_c2c<T>>(n_);
        if (pad_ != 0) build_blue();
    }

    [[nodiscard]] std::size_t size() const { return n_; }

    template<bool forward>
    void execute(std::complex<T>* x, T fct, std::size_t tid, thread_pool* pool = nullptr) const {
        static_assert(kDdStrictFp<T>, "the dd engine needs -fno-fast-math");
        if (x87_) {
            x87_->template execute<forward>(x, fct, tid);
            return;
        }
        // The default inverse scale is 1 / n rounded; the grid folds in 1 / n exactly instead,
        // as four_step_large does with its default scale.
        if (!forward && fct == default_transform_fct<T>(false, n_))
            execute_dd<forward>(x, T(1), n_, pool);
        else
            execute_dd<forward>(x, fct, 1, pool);
    }

    // The dd path at scale fct / div. ADM_NOINLINE keeps its 64-byte stack realignment and
    // register saves off the x87 branch of execute: inlined, they cost 59 instructions a call
    // below kDdMinLen.
    template<bool forward>
    ADM_NOINLINE void execute_dd(std::complex<T>* x, T fct, std::size_t div,
                                 thread_pool* pool) const {
        const unsigned emax = ld_max_exponent(x, n_);
        if (emax == kLdExpInfNan) ADM_UNLIKELY {
            const T nan = std::numeric_limits<T>::quiet_NaN();
            std::fill_n(x, n_, std::complex<T>(nan, nan));
            return;
        }
        const dd_grid g(emax, n_, fct, 0, div);
        transform(
            [&](std::size_t j, std::size_t valid, double* dst) {
                dd_load_ld<forward>(x + j, valid, g.s, dst);
            },
            [&](std::size_t k, std::size_t valid, const double* src) {
                dd_store_ld<forward>(src, g, x + k, valid);
            },
            pool);
    }

    // The forward transform between caller boundaries on one grid: load(j, valid, dst) writes
    // the panel element of inputs j .. j + valid - 1 with the other lanes zero, and
    // store(k, valid, src) takes the panel element of outputs k .. k + valid - 1 on the same
    // grid.
    template<typename Load, typename Store>
    void transform(const Load& load, const Store& store, thread_pool* pool) const {
        if (pad_ == 0) {
            fs_.run(load, store, pool);
            return;
        }
        // a = x * chirp, z = swap(FFT(a) * bmul), x = chirp * swap(FFT(z)): the second run is the
        // inverse transform, and bmul carries its 1 / pad.
        soa_scratch<double, 1> z_sc(pad_ * kDdElem / kDdW);
        double* z = z_sc.buf(0);
        fs_.run(
            [&](std::size_t j, std::size_t, double* dst) {
                if (j >= n_) {
                    std::fill_n(dst, kDdElem, 0.0);
                    return;
                }
                load(j, std::min(kDdW, n_ - j), dst);
                dd_cmul_table(chirp_.get() + j / kDdW * kDdElem, dst);
            },
            [&](std::size_t k, std::size_t, double* src) {
                dd_cmul_table(bmul_.get() + k / kDdW * kDdElem, src);
                gdd re, im;
                dd_load(src, re, im);
                dd_store(z + k / kDdW * kDdElem, im, re);
            },
            pool);
        fs_.run(
            [&](std::size_t j, std::size_t, double* dst) {
                std::copy_n(z + j / kDdW * kDdElem, kDdElem, dst);
            },
            [&](std::size_t k, std::size_t, double* src) {
                if (k >= n_) return;
                gdd re, im;
                dd_load(src, im, re);
                dd_store(src, re, im);
                dd_cmul_table(chirp_.get() + k / kDdW * kDdElem, src);
                store(k, std::min(kDdW, n_ - k), src);
            },
            pool);
    }

    // The Bluestein length: the smallest blocked 7-smooth pad >= 2n - 1, or 0 when the direct
    // four-step costs less.
    static std::size_t blue_pad(std::size_t n) {
        if (n < 2) return 0;
        constexpr std::size_t blk = kDdW * kDdW;
        std::size_t pad = (2 * n - 1 + blk - 1) / blk * blk;
        for (;; pad += blk) {
            std::size_t v = pad / blk;
            for (const std::size_t p : {2u, 3u, 5u, 7u})
                while (v % p == 0) v /= p;
            if (v == 1) break;
        }
        const double blue = 2 * dd_four_step<T>::split(pad, true).cost +
                            kBlueOps * double(2 * n + pad) / double(kDdW);
        return blue < dd_four_step<T>::split(n, false).cost ? pad : 0;
    }

private:
    // Vector operations of the chirp and spectrum products per element, against the units of
    // dd_lines::cost.
    static constexpr double kBlueOps = 2;

    // chirp_j = w_2n^(j^2) for j < n, and bmul = FFT(b) / pad for b the conjugate chirp wrapped
    // around pad. FFT(b) runs on the grid sb that holds |b| <= 1 grown by pad. |FFT(b)| <= 2n - 1
    // < pad, so every product constant is below 1. The inverse run stays on the input grid with
    // no shift: its outputs are the convolution of a with |b| = 1, at most n max |x|, the
    // growth the grid holds. A DIF stage value is an average of the outputs it feeds, and a
    // 7-smooth pad keeps every butterfly sum to at most 7 of them.
    void build_blue() {
        const std::size_t nb = (n_ + kDdW - 1) / kDdW, elems = pad_ * kDdElem / kDdW;
        chirp_ = make_aligned_buffer<double>(nb * kDdElem);
        std::fill_n(chirp_.get(), nb * kDdElem, 0.0);
        auto b = make_aligned_buffer<double>(elems);
        std::fill_n(b.get(), elems, 0.0);
        const int sb = 48 - bit_width(pad_ - 1);
        const auto put = [sb](double* p, double h, double l) {
            const double a = std::ldexp(h, sb), r = std::nearbyint(a);
            p[0] = r;
            p[kDdW] = (a - r) + std::ldexp(l, sb);
        };
        const dd_roots w(2 * n_);
        for (std::size_t j = 0, e = 0; j < n_; e = (e + 2 * j + 1) % (2 * n_), ++j) {
            const dd_const c = w(e);
            double* t = chirp_.get() + j / kDdW * kDdElem + j % kDdW;
            t[0] = c.rh;
            t[kDdW] = c.rl;
            t[2 * kDdW] = c.ih;
            t[3 * kDdW] = c.il;
            for (const std::size_t i : {j, (pad_ - j) % pad_}) {
                double* p = b.get() + i / kDdW * kDdElem + i % kDdW;
                put(p, c.rh, c.rl);
                put(p + 2 * kDdW, -c.ih, -c.il);
            }
        }
        bmul_ = make_aligned_buffer<double>(elems);
        fs_.run(
            [&](std::size_t j, std::size_t, double* dst) {
                std::copy_n(b.get() + j / kDdW * kDdElem, kDdElem, dst);
            },
            [&](std::size_t k, std::size_t, double* src) {
                std::copy_n(src, kDdElem, bmul_.get() + k / kDdW * kDdElem);
            },
            nullptr);
        const ct_dd scale = ct_dd(std::ldexp(1.0, -sb)) / ct_dd(pad_);
        for (std::size_t k = 0; k < pad_; ++k) {
            double* e = bmul_.get() + k / kDdW * kDdElem + k % kDdW;
            const ct_dd re = dd_two_sum(e[0], e[kDdW]) * scale;
            const ct_dd im = dd_two_sum(e[2 * kDdW], e[3 * kDdW]) * scale;
            e[0] = re.hi;
            e[kDdW] = re.lo;
            e[2 * kDdW] = im.hi;
            e[3 * kDdW] = im.lo;
        }
    }

    std::size_t n_, pad_;
    dd_four_step<T> fs_;
    aligned_buffer<double> chirp_, bmul_;
    std::unique_ptr<scalar_c2c<T>> x87_;
};

struct dd_rev_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t n) { return n - 1 - i; }
};

ADM_ALWAYS_INLINE dd_batch dd_reverse(dd_batch v) {
    return xsimd::swizzle(
        v, xsimd::make_batch_constant<std::uint64_t, dd_rev_lane, dd_batch::arch_type>());
}

// Long double r2c and c2r with scalar_r2c_1d's interface. An even n runs the c2c of M = n / 2
// on z_j = x_2j + i x_2j+1 and pairs bin k with bin M - k on the grid:
// 2 X_k = Z_k + conj(Z_(M-k)) - i w_n^k (Z_k - conj(Z_(M-k))). The doubled bins keep the grid
// integer and leave one exponent lower. An odd n runs the c2c of n on x + 0i.
template<typename T>
class dd_r2c {
public:
    explicit dd_r2c(std::size_t n, std::size_t = 1)
        : n_(n), nh_(n / 2 + 1), m_(n % 2 == 0 ? n / 2 : n), eng_(m_) {
        if (m_ < kDdMinLen) x87_ = std::make_unique<scalar_r2c_1d<T>>(n_);
        if (x87_ || n_ % 2 != 0) return;
        const std::size_t nb = m_ / kDdW + 1;
        tw_ = make_aligned_buffer<double>(nb * kDdElem);
        const dd_roots w(n_);
        for (std::size_t k = 0; k < nb * kDdW; ++k) {
            const dd_const c = w(k);
            double* t = tw_.get() + k / kDdW * kDdElem + k % kDdW;
            t[0] = c.rh;
            t[kDdW] = c.rl;
            t[2 * kDdW] = c.ih;
            t[3 * kDdW] = c.il;
        }
    }

    [[nodiscard]] std::size_t real_size() const { return n_; }
    [[nodiscard]] std::size_t cplx_size() const { return nh_; }

    void forward(const T* in, std::complex<T>* out, std::size_t tid) const {
        if (x87_) {
            x87_->forward(in, out, tid);
            return;
        }
        forward_dd(in, out);
    }

    void inverse(std::complex<T>* spec, T* out, T extra_scale, std::size_t tid) const {
        if (x87_) {
            x87_->inverse(spec, out, extra_scale, tid);
            return;
        }
        inverse_dd(spec, out, extra_scale);
    }

private:
    // Out of line for the reason of dd_c2c::execute_dd.
    ADM_NOINLINE void forward_dd(const T* in, std::complex<T>* out) const {
        if (n_ % 2 != 0) {
            soa_scratch<T, 1> buf_sc(2 * n_);
            auto* buf = reinterpret_cast<std::complex<T>*>(buf_sc.buf(0));
            for (std::size_t i = 0; i < n_; ++i) buf[i] = {in[i], T(0)};
            eng_.template execute_dd<true>(buf, T(1), 1, nullptr);
            std::copy_n(buf, nh_, out);
            return;
        }
        const auto* z = reinterpret_cast<const std::complex<T>*>(in);
        const unsigned emax = ld_max_exponent(z, m_);
        if (emax == kLdExpInfNan) ADM_UNLIKELY {
            const T nan = std::numeric_limits<T>::quiet_NaN();
            std::fill_n(out, nh_, std::complex<T>(nan, nan));
            return;
        }
        const dd_grid g(emax, n_, T(1), -1);
        // Four rows (re.hi, re.lo, im.hi, im.lo) of Z_0 .. Z_M, Z_M = Z_0, behind kDdW zeros so
        // that the reversed loads of the last block stay inside.
        const std::size_t stride = m_ + 2 * kDdW;
        soa_scratch<double, 1> zs_sc(4 * stride);
        double* zs = zs_sc.buf(0);
        eng_.transform(
            [&](std::size_t j, std::size_t valid, double* dst) {
                dd_load_ld<true>(z + j, valid, g.s, dst);
            },
            [&](std::size_t k, std::size_t valid, const double* src) {
                const auto live = lane_prefix_mask<double>(valid);
                for (std::size_t c = 0; c < 4; ++c) {
                    const dd_batch v = dd_batch::load_aligned(src + c * kDdW);
                    double* d = zs + c * stride + kDdW + k;
                    if (valid == kDdW) v.store_unaligned(d);
                    else v.store(d, live, xsimd::unaligned_mode{});
                }
            },
            nullptr);
        for (std::size_t c = 0; c < 4; ++c) {
            double* row = zs + c * stride;
            row[kDdW + m_] = row[kDdW];
            std::fill_n(row, kDdW, 0.0);
            std::fill(row + kDdW + m_ + 1, row + stride, 0.0);
        }
        alignas(dd_batch::arch_type::alignment()) double blk[kDdElem];
        for (std::size_t k0 = 0; k0 < nh_; k0 += kDdW) {
            const auto row = [&](std::size_t c, std::size_t i) {
                return dd_batch::load_unaligned(zs + c * stride + i);
            };
            const std::size_t ia = kDdW + k0, ib = kDdW + m_ - k0 - (kDdW - 1);
            const gdd ar{row(0, ia), row(1, ia)}, ai{row(2, ia), row(3, ia)};
            const gdd br{dd_reverse(row(0, ib)), dd_reverse(row(1, ib))};
            const gdd bi{dd_reverse(row(2, ib)), dd_reverse(row(3, ib))};
            gdd wr, wi;
            dd_load(tw_.get() + k0 / kDdW * kDdElem, wr, wi);
            // -i (Z_k - conj(Z_(M-k))).
            gdd tr = ai + bi, ti = br - ar;
            dd_cmul(wr, wi, tr, ti);
            dd_store(blk, (ar + br) + tr, (ai - bi) + ti);
            dd_store_ld<true>(blk, g, out + k0, std::min(kDdW, nh_ - k0));
        }
    }

    ADM_NOINLINE void inverse_dd(std::complex<T>* spec, T* out, T extra_scale) const {
        if (n_ % 2 != 0) {
            soa_scratch<T, 1> buf_sc(2 * n_);
            auto* buf = reinterpret_cast<std::complex<T>*>(buf_sc.buf(0));
            for (std::size_t k = 0; k < nh_; ++k) buf[k] = spec[k];
            for (std::size_t k = nh_; k < n_; ++k) buf[k] = std::conj(spec[n_ - k]);
            eng_.template execute_dd<false>(buf, extra_scale, n_, nullptr);
            for (std::size_t i = 0; i < n_; ++i) out[i] = buf[i].real();
            return;
        }
        auto* z = reinterpret_cast<std::complex<T>*>(out);
        const unsigned emax = ld_max_exponent(spec, nh_);
        if (emax == kLdExpInfNan) ADM_UNLIKELY {
            std::fill_n(out, n_, std::numeric_limits<T>::quiet_NaN());
            return;
        }
        const dd_grid g(emax, n_, extra_scale, -1, m_);
        // Four rows of 2 Z_k = E + i conj(w_n^k) O, E = X_k + conj(X_(M-k)) and
        // O = X_k - conj(X_(M-k)), for k < M.
        const std::size_t stride = m_ + kDdW;
        soa_scratch<double, 1> zs_sc(4 * stride);
        double* zs = zs_sc.buf(0);
        alignas(dd_batch::arch_type::alignment()) double blk[kDdElem];
        const dd_u64 rev =
            xsimd::make_batch_constant<std::uint64_t, dd_rev_lane, dd_batch::arch_type>();
        for (std::size_t k0 = 0; k0 < m_; k0 += kDdW) {
            const std::size_t valid = std::min(kDdW, m_ - k0);
            gdd ar, ai, br, bi, wr, wi;
            dd_load_ld<true>(spec + k0, valid, g.s, blk);
            dd_load(blk, ar, ai);
            // X_(M-k0-l) to lane l: the run that ends at X_(M-k0), reversed. The index of lane
            // l is valid - 1 - l mod kDdW, so the zero lanes past valid stay zero.
            dd_load_ld<true>(spec + m_ - k0 - (valid - 1), valid, g.s, blk);
            dd_load(blk, br, bi);
            const dd_u64 idx = (rev + dd_u64(valid)) & dd_u64(kDdW - 1);
            const auto turn = [&idx](const gdd& v) {
                return gdd{xsimd::swizzle(v.hi, idx), xsimd::swizzle(v.lo, idx)};
            };
            br = turn(br);
            bi = turn(bi);
            dd_load(tw_.get() + k0 / kDdW * kDdElem, wr, wi);
            gdd tr = ar - br, ti = ai + bi;
            dd_cmul(wr, -wi, tr, ti);
            const gdd zr = (ar + br) - ti, zi = (ai - bi) + tr;
            const dd_batch rows[4] = {zr.hi, zr.lo, zi.hi, zi.lo};
            for (std::size_t c = 0; c < 4; ++c) rows[c].store_unaligned(zs + c * stride + k0);
        }
        // The inverse runs forward on swap(2 Z); dd_store_ld<false> swaps back.
        eng_.transform(
            [&](std::size_t j, std::size_t valid, double* dst) {
                const auto live = lane_prefix_mask<double>(valid);
                for (std::size_t c = 0; c < 4; ++c)
                    dd_batch::load(zs + c * stride + j, live, xsimd::unaligned_mode{})
                        .store_aligned(dst + (c ^ 2) * kDdW);
            },
            [&](std::size_t k, std::size_t valid, const double* src) {
                dd_store_ld<false>(src, g, z + k, valid);
            },
            nullptr);
    }

    std::size_t n_, nh_, m_;
    dd_c2c<T> eng_;
    aligned_buffer<double> tw_;
    std::unique_ptr<scalar_r2c_1d<T>> x87_;
};

// The 1-D engine of the long double plans: the double-double engine where it is exact.
template<typename T>
using ld_c2c_t = std::conditional_t<kDdLongDouble, dd_c2c<T>, scalar_c2c<T>>;
template<typename T>
using ld_r2c_t = std::conditional_t<kDdLongDouble, dd_r2c<T>, scalar_r2c_1d<T>>;

}
}

#include "undef_macros.hpp"
