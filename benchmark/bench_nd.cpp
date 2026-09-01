
#include <admiral/admiral.hpp>

#include <ducc0/fft/fft.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "bench_harness.hpp"

namespace {

std::string shape_to_string(const std::vector<std::size_t>& shape) {
    std::string s;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) s += 'x';
        s += std::to_string(shape[i]);
    }
    return s;
}

template<typename T>
void ducc0_c2c_nd(const std::complex<T>* in, std::complex<T>* out,
                  const std::vector<std::size_t>& shape, bool forward, size_t nthreads = 1) {
    using namespace ducc0;
    std::size_t Ntot = 1;
    for (auto e : shape) Ntot *= e;
    detail_fft::shape_t sh(shape.begin(), shape.end());
    detail_fft::shape_t axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view = detail_mav::cfmav<std::complex<T>>(in, sh);
    auto out_view = detail_mav::vfmav<std::complex<T>>(out, sh);
    detail_fft::c2c(in_view, out_view, axes, forward, forward ? T(1) : T(1) / T(Ntot), nthreads);
}

template<typename T>
void ducc0_r2c_nd(const T* in, std::complex<T>* out, const std::vector<std::size_t>& shape,
                  size_t nthreads = 1) {
    using namespace ducc0;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    detail_fft::shape_t rsh(shape.begin(), shape.end());
    detail_fft::shape_t csh(cshape.begin(), cshape.end());
    detail_fft::shape_t axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view  = detail_mav::cfmav<T>(in, rsh);
    auto out_view = detail_mav::vfmav<std::complex<T>>(out, csh);
    detail_fft::r2c(in_view, out_view, axes, true, T(1), nthreads);
}

template<typename T>
void ducc0_c2r_nd(const std::complex<T>* in, T* out, const std::vector<std::size_t>& shape,
                  size_t nthreads = 1) {
    using namespace ducc0;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    std::size_t Nreal = 1;
    for (auto e : shape) Nreal *= e;
    detail_fft::shape_t rsh(shape.begin(), shape.end());
    detail_fft::shape_t csh(cshape.begin(), cshape.end());
    detail_fft::shape_t axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = i;
    auto in_view  = detail_mav::cfmav<std::complex<T>>(in, csh);
    auto out_view = detail_mav::vfmav<T>(out, rsh);
    detail_fft::c2r(in_view, out_view, axes, false, T(1) / T(Nreal), nthreads);
}

}

namespace bench {

constexpr double kIdentControlTol = 0.03;

template<typename T>
bool compare_nd(const std::vector<std::size_t>& shape, int reps, long inner, int nthreads) {
    std::size_t Ntot = 1;
    for (auto e : shape) Ntot *= e;
    std::vector<std::complex<T>> data(Ntot);
    for (std::size_t i = 0; i < Ntot; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));

    const std::size_t nt = static_cast<std::size_t>(nthreads);
    admiral::plan<T> p(admiral::span<const std::size_t>(shape.data(), shape.size()), {nt});
    std::vector<std::complex<T>> buf(Ntot);

    volatile T sink = T(0);
    const NbStat fft_fwd = nb_measure("fftnd_fwd", reps, inner, [&]() {
        p.forward(data.data(), buf.data());
        sink += buf[Ntot / 2].real();
    });
    const NbStat fft_rt = nb_measure("fftnd_rt", reps, inner, [&]() {
        p.forward(data.data(), buf.data());
        p.inverse(buf.data());
        sink += buf[Ntot / 2].real();
    });
    admiral::plan<T> p2(admiral::span<const std::size_t>(shape.data(), shape.size()), {nt});
    std::vector<std::complex<T>> buf2(Ntot);
    const NbStat ident_fwd = nb_measure("identnd_fwd", reps, inner, [&]() {
        p2.forward(data.data(), buf2.data());
        sink += buf2[Ntot / 2].real();
    });
    std::vector<std::complex<T>> dbuf(Ntot);
    const NbStat ducc_fwd = nb_measure("duccnd_fwd", reps, inner, [&]() {
        ducc0_c2c_nd<T>(data.data(), dbuf.data(), shape, true, nt);
        sink += dbuf[Ntot / 2].real();
    });
    const NbStat ducc_rt = nb_measure("duccnd_rt", reps, inner, [&]() {
        ducc0_c2c_nd<T>(data.data(), dbuf.data(), shape, true, nt);
        ducc0_c2c_nd<T>(dbuf.data(), dbuf.data(), shape, false, nt);
        sink += dbuf[Ntot / 2].real();
    });
#ifdef ADM_BENCH_FFTW
    fftw_c2c<T> fftw(shape, nthreads);
    NbStat fftw_fwd{0, 0, 0}, fftw_rt{0, 0, 0};
    fftw_fwd = nb_measure("fftwnd_fwd", reps, inner, [&]() { sink += fftw.forward(data)[Ntot / 2].real(); });
    fftw_rt  = nb_measure("fftwnd_rt",  reps, inner, [&]() { sink += fftw.roundtrip(data)[Ntot / 2].real(); });
#endif
    (void)sink;

    const bool use_cyc = nthreads == 1
                      && fft_fwd.cyc > 0.0 && ducc_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && ducc_rt.cyc > 0.0;
    const double fwd_ratio = use_cyc ? fft_fwd.cyc / ducc_fwd.cyc : fft_fwd.us / ducc_fwd.us;
    const double rt_ratio = use_cyc ? fft_rt.cyc / ducc_rt.cyc : fft_rt.us / ducc_rt.us;
    const char* metric = use_cyc ? "cyc" : "wall";
    const double max_err =
        std::max(std::max(fft_fwd.err, fft_rt.err), std::max(ducc_fwd.err, ducc_rt.err));
    const bool unstable = max_err > bench::kStableMdape;
    const bool lose = !(fwd_ratio < 1.0 && rt_ratio < 1.0);
    const double ident = use_cyc && ident_fwd.cyc > 0.0 ? ident_fwd.cyc / fft_fwd.cyc
                                                        : ident_fwd.us / fft_fwd.us;
    const bool ident_ok = std::abs(ident - 1.0) <= kIdentControlTol;

    std::cout << "CMPND " << std::setw(16) << shape_to_string(shape)
              << " (N=" << std::setw(9) << Ntot << ")"
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << metric
              << std::fixed
              << " fft_fwd_us=" << std::setprecision(4) << std::setw(10) << fft_fwd.us
              << " ducc_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fwd_ratio
              << " | fft_rt_us=" << std::setprecision(4) << std::setw(10) << fft_rt.us
              << " ducc_rt_ratio=" << std::setprecision(3) << std::setw(7) << rt_ratio;
#ifdef ADM_BENCH_FFTW
    {
        const bool fftw_cyc = use_cyc && fftw_fwd.cyc > 0.0 && fftw_rt.cyc > 0.0;
        const double fw = fftw_cyc ? fft_fwd.cyc / fftw_fwd.cyc : fft_fwd.us / fftw_fwd.us;
        const double rw = fftw_cyc ? fft_rt.cyc / fftw_rt.cyc : fft_rt.us / fftw_rt.us;
        std::cout << " | fftw_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fw
                  << " fftw_rt_ratio=" << std::setw(7) << rw;
    }
#endif
    std::cout << " ident=" << std::setprecision(3) << std::setw(6) << ident
              << " err=" << std::setprecision(1) << std::setw(4) << (max_err * 100.0) << "%"
              << (unstable ? "  <== UNSTABLE" : "")
              << (!ident_ok ? "  <== IDENTITY CONTROL REJECTED: harness untrustworthy" : "")
              << (lose ? "  <== LOSE (vs ducc0)" : "")
              << "\n";
    return ident_ok && !(unstable || lose);
}

template<typename T>
bool compare_nd_r2c(const std::vector<std::size_t>& shape, int reps, long inner, int nthreads) {
    std::size_t Nreal = 1;
    for (auto e : shape) Nreal *= e;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    std::size_t Nc = 1;
    for (auto e : cshape) Nc *= e;

    std::vector<T> real_in(Nreal);
    for (std::size_t i = 0; i < Nreal; ++i) real_in[i] = std::sin(T(i) * T(0.1)) + std::cos(T(i) * T(0.03));

    const std::size_t nt = static_cast<std::size_t>(nthreads);
    admiral::plan_r2c<T> p(admiral::span<const std::size_t>(shape.data(), shape.size()), {nt});
    std::vector<std::complex<T>> cbuf(Nc);
    std::vector<T> rbuf(Nreal);

    p.forward(real_in.data(), cbuf.data());
    std::vector<std::complex<T>> ref_c(Nc);
    ducc0_r2c_nd<T>(real_in.data(), ref_c.data(), shape);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < Nc; ++i) {
        num += std::norm(static_cast<std::complex<double>>(cbuf[i] - ref_c[i]));
        den += std::norm(static_cast<std::complex<double>>(ref_c[i]));
    }
    const double fwd_l2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    std::vector<std::complex<T>> rt_c = cbuf;
    p.inverse(rt_c.data(), rbuf.data());
    double rtnum = 0.0, rtden = 0.0;
    for (std::size_t i = 0; i < Nreal; ++i) {
        rtnum += double(rbuf[i] - real_in[i]) * double(rbuf[i] - real_in[i]);
        rtden += double(real_in[i]) * double(real_in[i]);
    }
    const double rt_l2 = rtden > 0.0 ? std::sqrt(rtnum / rtden) : std::sqrt(rtnum);
    const double tol = default_accuracy_tol<T>();
    const bool inaccurate = !(fwd_l2 <= tol && rt_l2 <= tol);

    volatile T sink = T(0);
    const NbStat fft_fwd = nb_measure("r2c_fwd", reps, inner, [&]() {
        p.forward(real_in.data(), cbuf.data());
        sink += cbuf[Nc / 2].real();
    });
    const NbStat fft_rt = nb_measure("r2c_rt", reps, inner, [&]() {
        p.forward(real_in.data(), cbuf.data());
        std::copy(cbuf.begin(), cbuf.end(), rt_c.begin());
        p.inverse(rt_c.data(), rbuf.data());
        sink += rbuf[Nreal / 2];
    });
    admiral::plan_r2c<T> p2(admiral::span<const std::size_t>(shape.data(), shape.size()),
                            {nt});
    std::vector<std::complex<T>> cbuf2(Nc);
    const NbStat ident_fwd = nb_measure("ident_r2c_fwd", reps, inner, [&]() {
        p2.forward(real_in.data(), cbuf2.data());
        sink += cbuf2[Nc / 2].real();
    });
    std::vector<std::complex<T>> dc(Nc);
    std::vector<T> dr(Nreal);
    const NbStat ducc_fwd = nb_measure("ducc_r2c_fwd", reps, inner, [&]() {
        ducc0_r2c_nd<T>(real_in.data(), dc.data(), shape, nt);
        sink += dc[Nc / 2].real();
    });
    const NbStat ducc_rt = nb_measure("ducc_r2c_rt", reps, inner, [&]() {
        ducc0_r2c_nd<T>(real_in.data(), dc.data(), shape, nt);
        ducc0_c2r_nd<T>(dc.data(), dr.data(), shape, nt);
        sink += dr[Nreal / 2];
    });
#ifdef ADM_BENCH_FFTW
    fftw_r2c<T> fftw(shape, nthreads);
    NbStat fftw_fwd{0, 0, 0}, fftw_rt{0, 0, 0};
    fftw_fwd = nb_measure("fftw_r2c_fwd", reps, inner, [&]() { sink += fftw.forward(real_in)[Nc / 2].real(); });
    fftw_rt  = nb_measure("fftw_r2c_rt",  reps, inner, [&]() { sink += fftw.roundtrip(real_in)[Nreal / 2]; });
#endif
    (void)sink;

    const bool use_cyc = nthreads == 1
                      && fft_fwd.cyc > 0.0 && ducc_fwd.cyc > 0.0
                      && fft_rt.cyc > 0.0 && ducc_rt.cyc > 0.0;
    const double fwd_ratio = use_cyc ? fft_fwd.cyc / ducc_fwd.cyc : fft_fwd.us / ducc_fwd.us;
    const double rt_ratio = use_cyc ? fft_rt.cyc / ducc_rt.cyc : fft_rt.us / ducc_rt.us;
    const char* metric = use_cyc ? "cyc" : "wall";
    const double max_err =
        std::max(std::max(fft_fwd.err, fft_rt.err), std::max(ducc_fwd.err, ducc_rt.err));
    const bool unstable = max_err > bench::kStableMdape || inaccurate;
    const bool lose = !(fwd_ratio < 1.0 && rt_ratio < 1.0);
    const double ident = use_cyc && ident_fwd.cyc > 0.0 ? ident_fwd.cyc / fft_fwd.cyc
                                                        : ident_fwd.us / fft_fwd.us;
    const bool ident_ok = std::abs(ident - 1.0) <= kIdentControlTol;

    std::cout << "R2CND " << std::setw(16) << shape_to_string(shape)
              << " (N=" << std::setw(9) << Nreal << ")"
              << " prec=" << ((sizeof(T) == 4) ? "f32" : "f64")
              << " m=" << metric
              << std::fixed
              << " fft_fwd_us=" << std::setprecision(4) << std::setw(10) << fft_fwd.us
              << " ducc_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fwd_ratio
              << " | fft_rt_us=" << std::setprecision(4) << std::setw(10) << fft_rt.us
              << " ducc_rt_ratio=" << std::setprecision(3) << std::setw(7) << rt_ratio;
#ifdef ADM_BENCH_FFTW
    {
        const bool fftw_cyc = use_cyc && fftw_fwd.cyc > 0.0 && fftw_rt.cyc > 0.0;
        const double fw = fftw_cyc ? fft_fwd.cyc / fftw_fwd.cyc : fft_fwd.us / fftw_fwd.us;
        const double rw = fftw_cyc ? fft_rt.cyc / fftw_rt.cyc : fft_rt.us / fftw_rt.us;
        std::cout << " | fftw_fwd_ratio=" << std::setprecision(3) << std::setw(7) << fw
                  << " fftw_rt_ratio=" << std::setw(7) << rw;
    }
#endif
    std::cout << " l2=" << std::scientific << std::setprecision(1) << std::max(fwd_l2, rt_l2)
              << std::defaultfloat
              << " ident=" << std::fixed << std::setprecision(3) << std::setw(6) << ident
              << (unstable ? "  <== UNSTABLE" : "")
              << (!ident_ok ? "  <== IDENTITY CONTROL REJECTED: harness untrustworthy" : "")
              << (lose ? "  <== LOSE (vs ducc0)" : "")
              << "\n";
    return ident_ok && !(unstable || lose);
}

template<typename T>
bool compare_nd_robust(const std::vector<std::size_t>& shape, int rounds, int reps, long inner,
                       int nthreads) {
    std::size_t Ntot = 1;
    for (auto e : shape) Ntot *= e;
    std::vector<std::complex<T>> data(Ntot);
    for (std::size_t i = 0; i < Ntot; ++i)
        data[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
    const std::size_t nt = static_cast<std::size_t>(nthreads);
    admiral::span<const std::size_t> sp(shape.data(), shape.size());
    const std::string ss = shape_to_string(shape);
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    volatile T sink = T(0);

    auto makeOurs = [&]() {
        auto plan = std::make_shared<admiral::plan<T>>(sp, admiral::options{nt});
        auto buf  = std::make_shared<std::vector<std::complex<T>>>(Ntot);
        ab_engine e;
        e.fwd = [&, plan, buf]() {
            plan->forward(data.data(), buf->data());
            sink += (*buf)[Ntot / 2].real();
        };
        e.rt = [&, plan, buf]() {
            plan->forward(data.data(), buf->data());
            plan->inverse(buf->data());
            sink += (*buf)[Ntot / 2].real();
        };
        return e;
    };
    auto makeDucc = [&]() {
        auto buf = std::make_shared<std::vector<std::complex<T>>>(Ntot);
        ab_engine e;
        e.fwd = [&, buf]() {
            ducc0_c2c_nd<T>(data.data(), buf->data(), shape, true, nt);
            sink += (*buf)[Ntot / 2].real();
        };
        e.rt = [&, buf]() {
            ducc0_c2c_nd<T>(data.data(), buf->data(), shape, true, nt);
            ducc0_c2c_nd<T>(buf->data(), buf->data(), shape, false, nt);
            sink += (*buf)[Ntot / 2].real();
        };
        return e;
    };

    const bool wall = nthreads > 1;
    double id_spread = 0.0;
    const double id = engine_ab_core("IDENT", ss, prec, "ours", "ours2",
                                     makeOurs, makeOurs, rounds, reps, inner, &id_spread, wall);
    engine_ab_core("ABND ", ss, prec, "ours", "ducc0",
                   makeOurs, makeDucc, rounds, reps, inner, nullptr, wall);
#ifdef ADM_BENCH_FFTW
    auto makeFftw = [&]() {
        auto f = std::make_shared<fftw_c2c<T>>(shape, nthreads);
        ab_engine e;
        e.fwd = [&, f]() { sink += f->forward(data)[Ntot / 2].real(); };
        e.rt  = [&, f]() { sink += f->roundtrip(data)[Ntot / 2].real(); };
        return e;
    };
    engine_ab_core("ABND ", ss, prec, "ours", "fftw ",
                   makeOurs, makeFftw, rounds, reps, inner, nullptr, wall);
#endif
    (void)sink;
    const bool id_ok = std::abs(id - 1.0) <= std::max(kIdentControlTol, 2.0 * id_spread);
    if (!id_ok)
        std::cout << "  <== IDENTITY CONTROL REJECTED (ours/ours=" << std::fixed
                  << std::setprecision(3) << id << ", must be ~1.000): harness untrustworthy\n";
    return id_ok;
}

template<typename T>
bool compare_nd_r2c_robust(const std::vector<std::size_t>& shape, int rounds, int reps, long inner,
                           int nthreads) {
    std::size_t Nreal = 1;
    for (auto e : shape) Nreal *= e;
    std::vector<std::size_t> cshape(shape);
    cshape.back() = shape.back() / 2 + 1;
    std::size_t Nc = 1;
    for (auto e : cshape) Nc *= e;
    std::vector<T> real_in(Nreal);
    for (std::size_t i = 0; i < Nreal; ++i) real_in[i] = std::sin(T(i) * T(0.1)) + std::cos(T(i) * T(0.03));
    const std::size_t nt = static_cast<std::size_t>(nthreads);
    admiral::span<const std::size_t> sp(shape.data(), shape.size());
    const std::string ss = shape_to_string(shape);
    const char* prec = (sizeof(T) == 4) ? "f32" : "f64";
    volatile T sink = T(0);

    auto makeOurs = [&]() {
        auto plan = std::make_shared<admiral::plan_r2c<T>>(sp, admiral::options{nt});
        auto cbuf = std::make_shared<std::vector<std::complex<T>>>(Nc);
        auto rbuf = std::make_shared<std::vector<T>>(Nreal);
        ab_engine e;
        e.fwd = [&, plan, cbuf]() {
            plan->forward(real_in.data(), cbuf->data());
            sink += (*cbuf)[Nc / 2].real();
        };
        e.rt = [&, plan, cbuf, rbuf]() {
            plan->forward(real_in.data(), cbuf->data());
            plan->inverse(cbuf->data(), rbuf->data());
            sink += (*rbuf)[Nreal / 2];
        };
        return e;
    };
    auto makeDucc = [&]() {
        auto cbuf = std::make_shared<std::vector<std::complex<T>>>(Nc);
        auto rbuf = std::make_shared<std::vector<T>>(Nreal);
        ab_engine e;
        e.fwd = [&, cbuf]() {
            ducc0_r2c_nd<T>(real_in.data(), cbuf->data(), shape, nt);
            sink += (*cbuf)[Nc / 2].real();
        };
        e.rt = [&, cbuf, rbuf]() {
            ducc0_r2c_nd<T>(real_in.data(), cbuf->data(), shape, nt);
            ducc0_c2r_nd<T>(cbuf->data(), rbuf->data(), shape, nt);
            sink += (*rbuf)[Nreal / 2];
        };
        return e;
    };

    const bool wall = nthreads > 1;
    double id_spread = 0.0;
    const double id = engine_ab_core("IDENT", ss, prec, "ours", "ours2",
                                     makeOurs, makeOurs, rounds, reps, inner, &id_spread, wall);
    engine_ab_core("R2CAB", ss, prec, "ours", "ducc0",
                   makeOurs, makeDucc, rounds, reps, inner, nullptr, wall);
#ifdef ADM_BENCH_FFTW
    auto makeFftw = [&]() {
        auto f = std::make_shared<fftw_r2c<T>>(shape, nthreads);
        ab_engine e;
        e.fwd = [&, f]() { sink += f->forward(real_in)[Nc / 2].real(); };
        e.rt  = [&, f]() { sink += f->roundtrip(real_in)[Nreal / 2]; };
        return e;
    };
    engine_ab_core("R2CAB", ss, prec, "ours", "fftw ",
                   makeOurs, makeFftw, rounds, reps, inner, nullptr, wall);
#endif
    (void)sink;
    const bool id_ok = std::abs(id - 1.0) <= std::max(kIdentControlTol, 2.0 * id_spread);
    if (!id_ok)
        std::cout << "  <== IDENTITY CONTROL REJECTED (ours/ours=" << std::fixed
                  << std::setprecision(3) << id << ", must be ~1.000): harness untrustworthy\n";
    return id_ok;
}

template bool compare_nd<float>(const std::vector<std::size_t>&, int, long, int);
template bool compare_nd<double>(const std::vector<std::size_t>&, int, long, int);
template bool compare_nd_r2c<float>(const std::vector<std::size_t>&, int, long, int);
template bool compare_nd_r2c<double>(const std::vector<std::size_t>&, int, long, int);
template bool compare_nd_robust<float>(const std::vector<std::size_t>&, int, int, long, int);
template bool compare_nd_robust<double>(const std::vector<std::size_t>&, int, int, long, int);
template bool compare_nd_r2c_robust<float>(const std::vector<std::size_t>&, int, int, long, int);
template bool compare_nd_r2c_robust<double>(const std::vector<std::size_t>&, int, int, long, int);

}
