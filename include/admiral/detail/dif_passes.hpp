#pragma once

// Vectorized Cooley-Tukey DIF passes over the butterflies of butterfly.hpp.

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <poet/poet.hpp>
#include "cxx_compat.hpp"
#include "simd.hpp"

#include "butterfly.hpp"
#include "cache.hpp"
#include "codelet.hpp"
#include "simd_swizzle.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

// fix3 switch, sweep only. ADM_FIX3_FIRST unrolls the first pass by two columns and restricts
// its outputs. ADM_FIX3 itself landed: the in-place pass blocks over the column index
// unconditionally.
#ifndef ADM_FIX3_FIRST
#define ADM_FIX3_FIRST 0
#endif
#if ADM_FIX3_FIRST
#define ADM_FIX3_FIRST_RESTRICT ADM_RESTRICT
#else
#define ADM_FIX3_FIRST_RESTRICT
#endif
// fix4 switch, sweep only. ADM_FIX4_ES2 lets the block-interleaved intermediate layout (es == 2)
// reach the buffer that feeds the last pass, so every pass addresses one base pointer per array.
#ifndef ADM_FIX4_ES2
#define ADM_FIX4_ES2 0
#endif
// fix4 switch, sweep only. ADM_FIX4_ES2_ONEBASE makes the element stride of the in-place arm and
// of the first pass compile-time, so the imaginary plane can be named as the real base plus W and
// both planes address off ONE register. It needs ADM_FIX4_ES2 to have anything to fire on.
#ifndef ADM_FIX4_ES2_ONEBASE
#define ADM_FIX4_ES2_ONEBASE 0
#endif
// fix4 switch, sweep only. Prints the tape that dif_build_tape builds and the tape the executor
// picks. Off in every timed build.
#ifndef ADM_FIX4_ES2_TRACE
#define ADM_FIX4_ES2_TRACE 0
#endif
// fix4 switch, sweep only. ADM_FIX4_ROLL replaces the column loop of the first pass and of the
// fix3 in-place arm with poet::dynamic_for<1>, whose laundered trip count stops the compiler
// folding the loop away, and carries the step in a poet::static_for. 1 rolls the loop and keeps
// one block per iteration; S >= 2 puts S blocks in one iteration with every load issued before
// any arithmetic, so the allocator sees S independent chains over a body of fixed shape. The
// barrier makes the COUNT opaque, not the loop: both compilers still unroll a small body at
// runtime, so the back-edge count in the asm is what says whether the body stayed one copy.
#ifndef ADM_FIX4_ROLL
#define ADM_FIX4_ROLL 0
#endif
// fix4 switch, sweep only. ADM_FIX4_T2 brackets the landed L1D admission of the two-factor
// (FFTW `t2`) first-pass twiddle table from above: 2 fires it for every N >= 256, so every cell
// of the sweep takes the Split arm. The use is in twiddles.hpp, which does not include this
// header, so the default is repeated in cache.hpp.
#ifndef ADM_FIX4_T2
#define ADM_FIX4_T2 0
#endif
// fix4 switch, sweep only. ADM_FIX4_U2 forces the first pass's column unroll to its value, so
// that many radix butterflies sit in flight over one loop body. dif_pass_unroll's budget,
// peak_live = 2 * IP + 10 against 32 zmm, already returns 1 from IP = 4 up, so no radix reaches
// 2 on its own and the value is an override rather than a hint. ADM_FIX4_U2_IP restricts the
// override to that one radix, 0 meaning every radix: 2 blocks of IP rows in two planes need
// 4 * IP live vectors, so the headroom is per radix and one arm per radix keeps the arms
// interpretable.
#ifndef ADM_FIX4_U2
#define ADM_FIX4_U2 0
#endif
#ifndef ADM_FIX4_U2_IP
#define ADM_FIX4_U2_IP 0
#endif

template<typename T, std::size_t IP, std::size_t PW, typename CC, typename CH>
ADM_ALWAYS_INLINE void small_ido_piece(CC ccre, CC ccim, CH chre, CH chim,
                                       std::size_t ido, std::size_t b, std::size_t a,
                                       std::size_t obase, std::size_t kstride,
                                       const T* twre, const T* twim) {
    using V = sized_piece_t<T, PW>;
    // Static, because MSVC does not find a block-scope `constexpr auto` holding a function
    // pointer from inside a lambda body: `error C3861: 'ld': identifier not found`.
    static constexpr auto ld = load_piece<T, PW>;
    static constexpr auto st = store_piece<T, PW>;
    V tr[IP], ti_arr[IP];
    poet::static_for<IP>([&](auto J) {
        const std::size_t off = a + ido * (J + IP * b);
        tr[J] = ld(ccre + off);
        ti_arr[J] = ld(ccim + off);
    });
    dif_butterfly<T, IP, V>(tr, ti_arr, [&](const auto k, V sr, V si) {
        const std::size_t off = obase + a + kstride * k;
        if constexpr (k > 0u) {
            const V owr = ld(twre + (k - 1u) * ido + a);
            const V owi = ld(twim + (k - 1u) * ido + a);
            st(chre + off, piece_fnma(owi, si, owr * sr));
            st(chim + off, piece_fma(owr, si, owi * sr));
        } else {
            st(chre + off, sr);
            st(chim + off, si);
        }
    });
}

template<typename T, std::size_t IP, std::size_t PW, typename CC, typename CH>
ADM_ALWAYS_INLINE void small_ido_piece_tw(CC ccre, CC ccim, CH chre, CH chim,
                                          std::size_t ido, std::size_t b, std::size_t a,
                                          std::size_t obase, std::size_t kstride,
                                          const sized_piece_t<T, PW>* owr,
                                          const sized_piece_t<T, PW>* owi) {
    using V = sized_piece_t<T, PW>;
    V tr[IP], ti_arr[IP];
    poet::static_for<IP>([&](auto J) {
        const std::size_t off = a + ido * (J + IP * b);
        tr[J] = load_piece<T, PW>(ccre + off);
        ti_arr[J] = load_piece<T, PW>(ccim + off);
    });
    dif_butterfly<T, IP, V>(tr, ti_arr, [&](const auto k, V sr, V si) {
        const std::size_t off = obase + a + kstride * k;
        if constexpr (k > 0u) {
            store_piece<T, PW>(chre + off, piece_fnma(owi[k], si, owr[k] * sr));
            store_piece<T, PW>(chim + off, piece_fma(owr[k], si, owi[k] * sr));
        } else {
            store_piece<T, PW>(chre + off, sr);
            store_piece<T, PW>(chim + off, si);
        }
    });
}

template<typename T, std::size_t IP>
void dif_pass_small_ido(const T* ccre, const T* ccim,
                        T* chre, T* chim,
                        std::size_t l1, std::size_t ido,
                        const T* twre, const T* twim) {
    constexpr std::size_t W0 = sized_piece_width<T, xsimd::batch<T>::size / 2>();
    if constexpr (IP <= 3 && 2 * (IP - 1u) + 2 * IP + 4u <= poet::vector_register_count()) {
        constexpr std::size_t kL1 = 48u * 1024u;
        std::size_t bt = (kL1 / 2u) / (4u * IP * ido * sizeof(T));
        if (bt == 0) bt = 1;
        if (bt > l1) bt = l1;
        for (std::size_t b0 = 0; b0 < l1; b0 += bt) {
            const std::size_t bend = b0 + bt < l1 ? b0 + bt : l1;
            sized_cover<T, W0, true>(0, ido, [&](auto PWc, std::size_t a) {
                constexpr std::size_t PW = PWc.value;
                using V = sized_piece_t<T, PW>;
                V owr[IP], owi[IP];
                poet::static_for<1, IP>([&](const auto k) {
                    owr[k] = load_piece<T, PW>(twre + (k - 1u) * ido + a);
                    owi[k] = load_piece<T, PW>(twim + (k - 1u) * ido + a);
                });
                for (std::size_t b = b0; b < bend; ++b)
                    small_ido_piece_tw<T, IP, PW>(ccre, ccim, chre, chim, ido, b, a,
                                                  ido * b, ido * l1, owr, owi);
            });
        }
    } else {
        poet::dynamic_for<1, 1>(std::size_t{0}, l1, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
            sized_cover<T, W0, true>(0, ido, [&](auto PWc, std::size_t a) {
                small_ido_piece<T, IP, PWc.value>(ccre, ccim, chre, chim, ido, b, a,
                                                  ido * b, ido * l1, twre, twim);
            });
        });
    }
}

template<typename T, std::size_t IP, bool InPlace, typename CC, typename CH>
void dif_pass_body(CC ccre, CC ccim, CH chre, CH chim,
              std::size_t l1, std::size_t ido,
              const T* twre, const T* twim,
                   std::size_t esi, std::size_t eso) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t idi = ido * esi, idz = ido * eso;

    if constexpr (!InPlace && W > 4) {
        if (ido >= 4 && ido < W) {
            dif_pass_small_ido<T, IP>(ccre, ccim, chre, chim, l1, ido, twre, twim);
            return;
        }
    }

    if constexpr (!InPlace && IP >= 2 && IP <= 5 && poet::vector_register_count() <= 16
                  && (IP == 4 || W >= 4)) {
        if (ido >= W && ido % W == 0
            && l1 * IP * ido * (4u * sizeof(T)) <= kIpTileBytes) {
            for (std::size_t aa = 0; aa < ido; aa += W) {
                batch owr[IP - 1], owi[IP - 1];
                poet::static_for<0, IP - 1>([&](const auto k) {
                    owr[k] = batch::load_unaligned(twre + (k * ido + aa));
                    owi[k] = batch::load_unaligned(twim + (k * ido + aa));
                });
                for (std::size_t b = 0; b < l1; ++b) {
                    batch tr[IP], ti_arr[IP];
                    poet::static_for<0, IP>([&](const auto j) {
                        tr[j] = batch::load_unaligned(ccre + (aa * esi + idi * (j + IP * b)));
                        ti_arr[j] = batch::load_unaligned(ccim + (aa * esi + idi * (j + IP * b)));
                    });
                    dif_butterfly<T, IP>(tr, ti_arr, [&](const auto k, batch sr, batch si) {
                        if constexpr (k > 0u) {
                            (owr[k - 1] * sr - owi[k - 1] * si)
                                .store_unaligned(chre + (aa * eso + idz * (b + l1 * k)));
                            (owr[k - 1] * si + owi[k - 1] * sr)
                                .store_unaligned(chim + (aa * eso + idz * (b + l1 * k)));
                        } else {
                            sr.store_unaligned(chre + (aa * eso + idz * (b + l1 * k)));
                            si.store_unaligned(chim + (aa * eso + idz * (b + l1 * k)));
                        }
                    });
                }
            }
            return;
        }
    }

    // An in-place pass rewrites the same IP vector slots it reads, so the column index can carry
    // the outer loop. The twiddles then leave the butterfly and one base pointer pair plus IP row
    // offsets stay live, in place of the ~30 strided streams the frame had to hold and reload.
    std::size_t amain = 0;
    if constexpr (InPlace && !dif_butterfly_wants_reload<IP>) {
        if (ido >= W && ccre == chre && ccim == chim && esi == eso) {
            amain = ido - ido % W;
#if ADM_FIX4_ES2_ONEBASE
            // fix4: Es == 2 names the imaginary plane as chre + W, the relation es == 2
            // guarantees, so both planes address off one base register instead of two.
            const auto run_ip = [&](auto Esc) ADM_LAMBDA_ALWAYS_INLINE {
                constexpr std::size_t Es = decltype(Esc)::value;
                T* const re = chre;
                T* const im = Es == 2u ? chre + W : chim;
                const std::size_t ide = ido * Es;
                for (std::size_t aa = 0; aa < amain; aa += W) {
                    batch owr[IP - 1], owi[IP - 1];
                    poet::static_for<0, IP - 1>([&](const auto k) ADM_LAMBDA_ALWAYS_INLINE {
                        owr[k] = batch::load_unaligned(twre + (k * ido + aa));
                        owi[k] = batch::load_unaligned(twim + (k * ido + aa));
                    });
                    for (std::size_t b = 0; b < l1; ++b) {
                        const std::size_t o0 = aa * Es + ide * IP * b;
                        batch tr[IP], ti_arr[IP];
                        poet::static_for<0, IP>([&](const auto j) ADM_LAMBDA_ALWAYS_INLINE {
                            tr[j] = batch::load_unaligned(re + (o0 + ide * j));
                            ti_arr[j] = batch::load_unaligned(im + (o0 + ide * j));
                        });
                        dif_butterfly<T, IP>(tr, ti_arr,
                                             [&](const auto k, batch sr, batch si)
                                                 ADM_LAMBDA_ALWAYS_INLINE {
                            const std::size_t off = o0 + ide * k;
                            if constexpr (k > 0u) {
                                (owr[k - 1u] * sr - owi[k - 1u] * si).store_unaligned(re + off);
                                (owr[k - 1u] * si + owi[k - 1u] * sr).store_unaligned(im + off);
                            } else {
                                sr.store_unaligned(re + off);
                                si.store_unaligned(im + off);
                            }
                        });
                    }
                }
            };
            if (esi == 2u) run_ip(std::integral_constant<std::size_t, 2u>{});
            else run_ip(std::integral_constant<std::size_t, 1u>{});
#elif ADM_FIX4_ROLL
            // fix4 ROLL: the same treatment as the first pass. dynamic_for<1> keeps the column
            // loop rolled and the step sits in a static_for, so S columns share one body and the
            // inner row loop loads both columns before either butterfly runs.
            {
                constexpr std::size_t S = ADM_FIX4_ROLL;
                T* const re = chre;
                T* const im = chim;
                auto col_run = [&](auto Nc, std::size_t aa0) ADM_LAMBDA_ALWAYS_INLINE {
                    constexpr std::size_t NC = decltype(Nc)::value;
                    batch owr[NC][IP - 1], owi[NC][IP - 1];
                    poet::static_for<0, NC>([&](const auto s) ADM_LAMBDA_ALWAYS_INLINE {
                        constexpr std::size_t Sv = decltype(s)::value;
                        poet::static_for<0, IP - 1>([&](const auto k) ADM_LAMBDA_ALWAYS_INLINE {
                            owr[Sv][k] = batch::load_unaligned(twre + (k * ido + aa0 + Sv * W));
                            owi[Sv][k] = batch::load_unaligned(twim + (k * ido + aa0 + Sv * W));
                        });
                    });
                    for (std::size_t b = 0; b < l1; ++b) {
                        batch tr[NC][IP], ti_arr[NC][IP];
                        poet::static_for<0, NC>([&](const auto s) ADM_LAMBDA_ALWAYS_INLINE {
                            constexpr std::size_t Sv = decltype(s)::value;
                            const std::size_t o0 = (aa0 + Sv * W) * esi + idi * IP * b;
                            poet::static_for<0, IP>([&](const auto j) ADM_LAMBDA_ALWAYS_INLINE {
                                tr[Sv][j] = batch::load_unaligned(re + (o0 + idi * j));
                                ti_arr[Sv][j] = batch::load_unaligned(im + (o0 + idi * j));
                            });
                        });
                        poet::static_for<0, NC>([&](const auto s) ADM_LAMBDA_ALWAYS_INLINE {
                            constexpr std::size_t Sv = decltype(s)::value;
                            const std::size_t o0 = (aa0 + Sv * W) * esi + idi * IP * b;
                            dif_butterfly<T, IP>(tr[Sv], ti_arr[Sv],
                                                 [&](const auto k, batch sr, batch si)
                                                     ADM_LAMBDA_ALWAYS_INLINE {
                                const std::size_t off = o0 + idi * k;
                                if constexpr (k > 0u) {
                                    (owr[Sv][k - 1u] * sr - owi[Sv][k - 1u] * si)
                                        .store_unaligned(re + off);
                                    (owr[Sv][k - 1u] * si + owi[Sv][k - 1u] * sr)
                                        .store_unaligned(im + off);
                                } else {
                                    sr.store_unaligned(re + off);
                                    si.store_unaligned(im + off);
                                }
                            });
                        });
                    }
                };
                const std::size_t nstep = (amain / W) / S;
                if (nstep != 0)
                    poet::dynamic_for<1, 1>(std::size_t{0}, nstep,
                                            [&](std::size_t i) ADM_LAMBDA_ALWAYS_INLINE {
                        col_run(std::integral_constant<std::size_t, S>{}, i * (S * W));
                    });
                for (std::size_t aa = nstep * S * W; aa < amain; aa += W)
                    col_run(std::integral_constant<std::size_t, 1u>{}, aa);
            }
#else
            T* const re = chre;
            T* const im = chim;
            for (std::size_t aa = 0; aa < amain; aa += W) {
                batch owr[IP - 1], owi[IP - 1];
                poet::static_for<0, IP - 1>([&](const auto k) ADM_LAMBDA_ALWAYS_INLINE {
                    owr[k] = batch::load_unaligned(twre + (k * ido + aa));
                    owi[k] = batch::load_unaligned(twim + (k * ido + aa));
                });
                for (std::size_t b = 0; b < l1; ++b) {
                    const std::size_t o0 = aa * esi + idi * IP * b;
                    batch tr[IP], ti_arr[IP];
                    poet::static_for<0, IP>([&](const auto j) ADM_LAMBDA_ALWAYS_INLINE {
                        tr[j] = batch::load_unaligned(re + (o0 + idi * j));
                        ti_arr[j] = batch::load_unaligned(im + (o0 + idi * j));
                    });
                    dif_butterfly<T, IP>(tr, ti_arr,
                                         [&](const auto k, batch sr, batch si)
                                             ADM_LAMBDA_ALWAYS_INLINE {
                        const std::size_t off = o0 + idi * k;
                        if constexpr (k > 0u) {
                            (owr[k - 1u] * sr - owi[k - 1u] * si).store_unaligned(re + off);
                            (owr[k - 1u] * si + owi[k - 1u] * sr).store_unaligned(im + off);
                        } else {
                            sr.store_unaligned(re + off);
                            si.store_unaligned(im + off);
                        }
                    });
                }
            }
#endif
            if (amain == ido) return;
        }
    }

    for (std::size_t b = 0; b < l1; ++b) {
        const std::size_t obase   = InPlace ? idz * IP * b : idz * b;
        const std::size_t kstride = InPlace ? idz : idz * l1;
        constexpr bool kSplitCols = InPlace && dif_butterfly_wants_reload<IP>;
        const auto ocol = [](std::size_t k) {
            if constexpr (kSplitCols) return (k & 1u) * (IP / 2u) + k / 2u;
            else                      return k;
        };
        {
            constexpr std::size_t U = dif_pass_unroll<IP>();
            auto do_batch = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
                auto emit_tw = [&](const auto k, batch sr, batch si) {
                    const std::size_t off = obase + aa * eso + kstride * ocol(k);
                    if constexpr (k > 0u) {
                        const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                        const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                        (owr * sr - owi * si).store_unaligned(chre + off);
                        (owr * si + owi * sr).store_unaligned(chim + off);
                    } else {
                        sr.store_unaligned(chre + off);
                        si.store_unaligned(chim + off);
                    }
                };
                batch tr[IP], ti_arr[IP];
                for (std::size_t j = 0; j < IP; ++j) {
                    tr[j] = batch::load_unaligned(ccre + (aa * esi + idi * (j + IP * b)));
                    ti_arr[j] = batch::load_unaligned(ccim + (aa * esi + idi * (j + IP * b)));
                }
                dif_butterfly<T, IP>(tr, ti_arr, emit_tw);
            };
            auto do_half = [&](std::size_t aa, auto ODD) ADM_LAMBDA_ALWAYS_INLINE {
                constexpr std::size_t H = IP / 2;
                const std::size_t obs = obase + aa * eso;
                const std::size_t ib = aa * esi + idi * (IP * b);
                auto emit_h = [&](const auto k, batch sr, batch si) {
                    const std::size_t off = obs + kstride * k;
                    if constexpr (k > 0u) {
                        const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                        const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                        (owr * sr - owi * si).store_unaligned(chre + off);
                        (owr * si + owi * sr).store_unaligned(chim + off);
                    } else {
                        sr.store_unaligned(chre + off);
                        si.store_unaligned(chim + off);
                    }
                };
                batch hr[H], hi[H];
                poet::static_for<0, H>([&](const auto n) {
                    const batch ar = batch::load_unaligned(ccre + (ib + idi * n));
                    const batch ai = batch::load_unaligned(ccim + (ib + idi * n));
                    const batch br = batch::load_unaligned(ccre + (ib + idi * (n + H)));
                    const batch bi = batch::load_unaligned(ccim + (ib + idi * (n + H)));
                    if constexpr (ODD) {
                        auto [fr2, fi2] = apply_stage_twiddle<T, IP, n, batch>(
                            ar - br, ai - bi);
                        hr[n] = fr2;
                        hi[n] = fi2;
                    } else {
                        hr[n] = ar + br;
                        hi[n] = ai + bi;
                    }
                });
                pow2_dif_butterfly<T, H, batch>(hr, hi, [&](auto Kc, batch yr, batch yi) {
                    emit_h(std::integral_constant<std::size_t, 2 * Kc + ODD>{}, yr, yi);
                });
            };
            std::size_t a = amain;
            if constexpr (dif_butterfly_wants_reload<IP> && !InPlace) {
                auto sweep = [&](auto ODD) {
                    std::size_t ah = 0;
                    for (; ah + 2 * W <= ido; ah += 2 * W) {
                        do_half(ah, ODD);
                        do_half(ah + W, ODD);
                    }
                    for (; ah + W <= ido; ah += W) do_half(ah, ODD);
                    if (ido >= W && (ido - ah) * 2 >= W) { do_half(ido - W, ODD); ah = ido; }
                    return ah;
                };
                sweep(std::integral_constant<bool, false>{});
                a = sweep(std::integral_constant<bool, true>{});
            } else if constexpr (kSplitCols) {
                constexpr std::size_t H = IP / 2;
                auto ip_combine = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
                    poet::static_for<0, H>([&](const auto n) {
                        const std::size_t o0 = aa * esi + idi * (n + IP * b);
                        const std::size_t o1 = aa * esi + idi * (n + H + IP * b);
                        const batch ar = batch::load_unaligned(ccre + o0);
                        const batch ai = batch::load_unaligned(ccim + o0);
                        const batch br = batch::load_unaligned(ccre + o1);
                        const batch bi = batch::load_unaligned(ccim + o1);
                        (ar + br).store_unaligned(chre + o0);
                        (ai + bi).store_unaligned(chim + o0);
                        const auto [fr, fi] =
                            apply_stage_twiddle<T, IP, n, batch>(ar - br, ai - bi);
                        fr.store_unaligned(chre + o1);
                        fi.store_unaligned(chim + o1);
                    });
                };
                auto ip_half = [&](std::size_t aa, auto ODD) ADM_LAMBDA_ALWAYS_INLINE {
                    // The store lambda sees ODD only through its reference capture, and MSVC 19.44
                    // cannot fold a reference in a constant expression (P2280): it reads odd.
                    constexpr std::size_t odd = ODD, src = odd * H;
                    batch hr[H], hi[H];
                    poet::static_for<0, H>([&](const auto n) {
                        hr[n] = batch::load_unaligned(ccre + (aa * esi + idi * (n + src + IP * b)));
                        hi[n] = batch::load_unaligned(ccim + (aa * esi + idi * (n + src + IP * b)));
                    });
                    pow2_dif_butterfly<T, H, batch>(hr, hi, [&](auto Kc, batch yr, batch yi) {
                        constexpr std::size_t k = 2u * Kc + odd;
                        const std::size_t off = obase + aa * eso + kstride * (src + Kc);
                        if constexpr (k > 0u) {
                            const batch owr = batch::load_unaligned(twre + ((k - 1u) * ido + aa));
                            const batch owi = batch::load_unaligned(twim + ((k - 1u) * ido + aa));
                            (owr * yr - owi * yi).store_unaligned(chre + off);
                            (owr * yi + owi * yr).store_unaligned(chim + off);
                        } else {
                            yr.store_unaligned(chre + off);
                            yi.store_unaligned(chim + off);
                        }
                    });
                };
                constexpr std::size_t kTileCols =
                    std::max(W, (kIpTileBytes / (IP * 2u * sizeof(T))) / W * W);
                const std::size_t afull = ido - ido % W;
                for (std::size_t t = 0; t < afull; t += kTileCols) {
                    const std::size_t hi = std::min(t + kTileCols, afull);
                    for (std::size_t ah = t; ah < hi; ah += W) ip_combine(ah);
                    for (std::size_t ah = t; ah < hi; ah += W)
                        ip_half(ah, std::integral_constant<bool, false>{});
                    for (std::size_t ah = t; ah < hi; ah += W)
                        ip_half(ah, std::integral_constant<bool, true>{});
                }
                a = afull;
            } else if constexpr (U > 1) {
                for (; a + U * W <= ido; a += U * W) {
                    poet::static_for<0, U>([&](auto UU) {
                        do_batch(a + UU * W);
                    });
                }
            }
            for (; a + W <= ido; a += W) do_batch(a);
            if constexpr (InPlace) {
                if (a < ido) {
                    const std::size_t rem = ido - a;
                    if (kSplitCols || (2 * rem >= W && !((kPieceWidths<T> >> rem) & 1u))) {
                        constexpr auto um = xsimd::unaligned_mode{};
                        const xsimd::batch_bool<T> m = lane_prefix_mask<T>(rem);
                        batch tr[IP], ti_arr[IP];
                        for (std::size_t j = 0; j < IP; ++j) {
                            tr[j] = batch::load(ccre + (a * esi + idi * (j + IP * b)), m, um);
                            ti_arr[j] = batch::load(ccim + (a * esi + idi * (j + IP * b)), m, um);
                        }
                        dif_butterfly<T, IP>(
                            tr, ti_arr, [&](const auto k, batch sr, batch si) {
                                const std::size_t off = obase + a * eso + kstride * ocol(k);
                                if constexpr (k > 0u) {
                                    const batch owr =
                                        batch::load(twre + ((k - 1u) * ido + a), m, um);
                                    const batch owi =
                                        batch::load(twim + ((k - 1u) * ido + a), m, um);
                                    (owr * sr - owi * si).store(chre + off, m, um);
                                    (owr * si + owi * sr).store(chim + off, m, um);
                                } else {
                                    sr.store(chre + off, m, um);
                                    si.store(chim + off, m, um);
                                }
                            });
                    } else {
                        constexpr std::size_t W0 = sized_piece_width<T, W / 2>();
                        sized_cover<T, W0, false>(a, ido, [&](auto PWc, std::size_t aa) {
                            small_ido_piece<T, IP, PWc.value>(
                                ccre, ccim, chre, chim, ido, b, aa, obase, kstride, twre, twim);
                        });
                    }
                }
            } else {
                if (ido >= W && (ido - a) * 2 >= W) { do_batch(ido - W); a = ido; }
                for (; a < ido; ++a) {
                    T tr_s[IP], ti_s[IP];
                    for (std::size_t j = 0; j < IP; ++j) {
                        tr_s[j] = ccre[a * esi + idi * (j + IP * b)];
                        ti_s[j] = ccim[a * esi + idi * (j + IP * b)];
                    }
                    dif_butterfly<T, IP>(tr_s, ti_s, [&](const auto k, T sr, T si) {
                        if constexpr (k > 0u) {
                            const T owr = twre[(k - 1u) * ido + a];
                            const T owi = twim[(k - 1u) * ido + a];
                            chre[obase + a * eso + kstride * k] = owr * sr - owi * si;
                            chim[obase + a * eso + kstride * k] = owr * si + owi * sr;
                        } else {
                            chre[obase + a * eso + kstride * k] = sr;
                            chim[obase + a * eso + kstride * k] = si;
                        }
                    });
                }
            }
        }
    }
}

template<typename T, std::size_t IP>
ADM_ALWAYS_INLINE void dif_pass_impl(const T* ccre, const T* ccim,
                                     T* chre, T* chim,
                                     std::size_t l1, std::size_t ido,
                                     const T* twre, const T* twim,
                                     std::size_t esi, std::size_t eso) {
    dif_pass_body<T, IP, false, decltype(ccre), decltype(chre)>(
        ccre, ccim, chre, chim, l1, ido, twre, twim, esi, eso);
}

template<typename T, std::size_t IP, typename... A>
ADM_FLATTEN void dif_pass_ip_flat(A... a) { dif_pass_body<T, IP, true>(a...); }

template<typename T, std::size_t IP, typename... A>
ADM_FLATTEN void dif_pass_flat(A... a) { dif_pass_impl<T, IP>(a...); }

template<typename T, std::size_t IP>
void dif_pass(const T* ccre, const T* ccim,
              T* chre, T* chim,
              std::size_t l1, std::size_t ido,
              const T* twre, const T* twim, std::size_t esi, std::size_t eso) {
    if constexpr (dif_butterfly_wants_reload<IP>)
        dif_pass_impl<T, IP>(ccre, ccim, chre, chim, l1, ido, twre, twim, esi, eso);
    else
        dif_pass_flat<T, IP>(ccre, ccim, chre, chim, l1, ido, twre, twim, esi, eso);
}

template<typename T, std::size_t P>
ADM_NOINLINE void dif_pass_prime_chip(const T* ccre, const T* ccim,
                                      T* chre, T* chim,
                                      std::size_t l1, std::size_t ido,
                                      const T* twre, const T* twim,
                                      std::size_t esi, std::size_t eso) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t idi = ido * esi, idz = ido * eso;
    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t aa = 0; aa < ido; aa += W) {
            const std::size_t rem = ido - aa;
            auto tile = [&](xsimd::batch_bool<T> m, auto full) ADM_LAMBDA_ALWAYS_INLINE {
                const auto um = xsimd::unaligned_mode{};
                batch xre[P], xim[P];
                for (unsigned j = 0; j < P; ++j) {
                    const T* xjr = ccre + (aa * esi + idi * (j + P * b));
                    const T* xji = ccim + (aa * esi + idi * (j + P * b));
                    if constexpr (full) {
                        xre[j] = batch::load_unaligned(xjr);
                        xim[j] = batch::load_unaligned(xji);
                    } else {
                        xre[j] = batch::load(xjr, m, um);
                        xim[j] = batch::load(xji, m, um);
                    }
                }
                batch yre[P], yim[P];
                rader_apply_batched<P, T, batch>(xre, xim, 1, yre, yim);
                for (unsigned k = 0; k < P; ++k) {
                    const std::size_t off = aa * eso + idz * (b + l1 * k);
                    batch sr = yre[k], si = yim[k];
                    if (k > 0) {
                        const T* twr = twre + (std::size_t(k - 1u) * ido + aa);
                        const T* twi = twim + (std::size_t(k - 1u) * ido + aa);
                        if constexpr (full) {
                            const batch owr = batch::load_unaligned(twr);
                            const batch owi = batch::load_unaligned(twi);
                            (owr * sr - owi * si).store_unaligned(chre + off);
                            (owr * si + owi * sr).store_unaligned(chim + off);
                        } else {
                            const batch owr = batch::load(twr, m, um);
                            const batch owi = batch::load(twi, m, um);
                            (owr * sr - owi * si).store(chre + off, m, um);
                            (owr * si + owi * sr).store(chim + off, m, um);
                        }
                    } else {
                        if constexpr (full) {
                            sr.store_unaligned(chre + off);
                            si.store_unaligned(chim + off);
                        } else {
                            sr.store(chre + off, m, um);
                            si.store(chim + off, m, um);
                        }
                    }
                }
            };
            if (rem >= W) tile(xsimd::batch_bool<T>(true), std::bool_constant<true>{});
            else          tile(lane_prefix_mask<T>(rem), std::bool_constant<false>{});
        }
    }
}

template<typename T, std::size_t P1, std::size_t P2>
void dif_pass_fused2(const T* ccre, const T* ccim,
                     T* chre, T* chim,
                     std::size_t l1, std::size_t ido,
                     const T* ptw) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W  = batch::size;
    constexpr std::size_t WaMax = (kFusedTileBytes / (2 * P1 * P2 * sizeof(T)) / W) * W;
    static_assert(WaMax >= W, "fused2: L1 tile too small for this SIMD width");
    const std::size_t ido2 = ido / P2;
    const std::size_t l12  = l1 * P1;
    constexpr std::size_t stride1 = 2u * (P1 - 1u);
    const std::size_t tw2_off = ido * stride1;
    constexpr std::size_t stride2 = 2u * (P2 - 1u);

    alignas(xsimd::batch<T>::arch_type::alignment()) T lbre[P1 * P2 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T lbim[P1 * P2 * WaMax];

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a0 = 0; a0 < ido2; a0 += WaMax) {
            const std::size_t Wa = ido2 - a0 < WaMax ? ido2 - a0 : WaMax;

            for (std::size_t j2 = 0; j2 < P2; ++j2) {
                const T* ptw1_cur = ptw + (a0 + j2 * ido2) * stride1;
                for (std::size_t t = 0; t < Wa; t += W) {
                    const std::size_t a = a0 + ido2 * j2 + t;
                    batch tr[P1], ti_arr[P1];
                    for (std::size_t j = 0; j < P1; ++j) {
                        tr[j]     = batch::load_unaligned(ccre + (a + ido * (j + P1 * b)));
                        ti_arr[j] = batch::load_unaligned(ccim + (a + ido * (j + P1 * b)));
                    }
                    dif_butterfly<T, P1>(tr, ti_arr, [&](const auto k, batch sr, batch si) {
                        T* lr = lbre + ((k * P2 + j2) * WaMax + t);
                        T* li = lbim + ((k * P2 + j2) * WaMax + t);
                        if constexpr (k > 0u) {
                            const batch owr = batch::load_unaligned(ptw1_cur + (k - 1u) * 2u * W);
                            const batch owi = batch::load_unaligned(ptw1_cur + (k - 1u) * 2u * W + W);
                            (owr * sr - owi * si).store_aligned(lr);
                            (owr * si + owi * sr).store_aligned(li);
                        } else {
                            sr.store_aligned(lr);
                            si.store_aligned(li);
                        }
                    });
                    ptw1_cur += stride1 * W;
                }
            }

            const T* ptw2_start = ptw + tw2_off + a0 * stride2;
            for (std::size_t k = 0; k < P1; ++k) {
                const std::size_t bp = b + l1 * k;
                const T* ptw2_cur = ptw2_start;
                for (std::size_t t = 0; t < Wa; t += W) {
                    const std::size_t a2 = a0 + t;
                    batch tr[P2], ti_arr[P2];
                    for (std::size_t j2 = 0; j2 < P2; ++j2) {
                        tr[j2]     = batch::load_aligned(lbre + ((k * P2 + j2) * WaMax + t));
                        ti_arr[j2] = batch::load_aligned(lbim + ((k * P2 + j2) * WaMax + t));
                    }
                    dif_butterfly<T, P2>(tr, ti_arr, [&](const auto k2, batch sr, batch si) {
                        if constexpr (k2 > 0u) {
                            const batch owr = batch::load_unaligned(ptw2_cur + (k2 - 1u) * 2u * W);
                            const batch owi = batch::load_unaligned(ptw2_cur + (k2 - 1u) * 2u * W + W);
                            (owr * sr - owi * si).store_unaligned(chre + (a2 + ido2 * (bp + l12 * k2)));
                            (owr * si + owi * sr).store_unaligned(chim + (a2 + ido2 * (bp + l12 * k2)));
                        } else {
                            sr.store_unaligned(chre + (a2 + ido2 * (bp + l12 * k2)));
                            si.store_unaligned(chim + (a2 + ido2 * (bp + l12 * k2)));
                        }
                    });
                    ptw2_cur += stride2 * W;
                }
            }
        }
    }
}

template<typename T, std::size_t P1, std::size_t P2, std::size_t P3>
void dif_pass_fused3(const T* ccre, const T* ccim,
                     T* chre, T* chim,
                     std::size_t l1, std::size_t ido,
                     const T* tw1re, const T* tw1im,
                     const T* tw2re, const T* tw2im,
                     const T* tw3re, const T* tw3im) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W  = batch::size;
    constexpr std::size_t WaMax = (kFusedTileBytes / (4 * P1 * P2 * P3 * sizeof(T)) / W) * W;
    static_assert(WaMax >= W,
                  "fused3: WaMax must be >= W (tile too small for target)");

    const std::size_t ido2 = ido / P2;
    const std::size_t ido3 = ido / (P2 * P3);
    const std::size_t l12  = l1 * P1;
    const std::size_t l123 = l1 * P1 * P2;

    alignas(xsimd::batch<T>::arch_type::alignment()) T tile1re[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile1im[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile2re[P1 * P2 * P3 * WaMax];
    alignas(xsimd::batch<T>::arch_type::alignment()) T tile2im[P1 * P2 * P3 * WaMax];

    for (std::size_t b = 0; b < l1; ++b) {
        for (std::size_t a0 = 0; a0 < ido3; a0 += WaMax) {
            const std::size_t Wa = ido3 - a0 < WaMax ? ido3 - a0 : WaMax;

            for (std::size_t j2 = 0; j2 < P2; ++j2) {
                for (std::size_t j3 = 0; j3 < P3; ++j3) {
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a = a0 + ido3 * (j3 + P3 * j2) + t;
                        batch tr[P1], ti_arr[P1];
                        for (std::size_t j = 0; j < P1; ++j) {
                            tr[j]     = batch::load_unaligned(ccre + (a + ido * (j + P1 * b)));
                            ti_arr[j] = batch::load_unaligned(ccim + (a + ido * (j + P1 * b)));
                        }
                        dif_butterfly<T, P1>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            T* lr = tile1re + ((Kc * P2 * P3 + j2 * P3 + j3) * WaMax + t);
                            T* li = tile1im + ((Kc * P2 * P3 + j2 * P3 + j3) * WaMax + t);
                            if constexpr (Kc > 0u) {
                                const batch owr = batch::load_unaligned(tw1re + ((Kc - 1u) * ido + a));
                                const batch owi = batch::load_unaligned(tw1im + ((Kc - 1u) * ido + a));
                                (owr * sr - owi * si).store_aligned(lr);
                                (owr * si + owi * sr).store_aligned(li);
                            } else {
                                sr.store_aligned(lr);
                                si.store_aligned(li);
                            }
                        });
                    }
                }
            }

            for (std::size_t k1 = 0; k1 < P1; ++k1) {
                for (std::size_t j3 = 0; j3 < P3; ++j3) {
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a1 = a0 + ido3 * j3 + t;
                        batch tr[P2], ti_arr[P2];
                        for (std::size_t j2 = 0; j2 < P2; ++j2) {
                            tr[j2]     = batch::load_aligned(tile1re + ((k1 * P2 * P3 + j2 * P3 + j3) * WaMax + t));
                            ti_arr[j2] = batch::load_aligned(tile1im + ((k1 * P2 * P3 + j2 * P3 + j3) * WaMax + t));
                        }
                        dif_butterfly<T, P2>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            T* lr = tile2re + ((k1 * P2 * P3 + Kc * P3 + j3) * WaMax + t);
                            T* li = tile2im + ((k1 * P2 * P3 + Kc * P3 + j3) * WaMax + t);
                            if constexpr (Kc > 0u) {
                                const batch owr = batch::load_unaligned(tw2re + ((Kc - 1u) * ido2 + a1));
                                const batch owi = batch::load_unaligned(tw2im + ((Kc - 1u) * ido2 + a1));
                                (owr * sr - owi * si).store_aligned(lr);
                                (owr * si + owi * sr).store_aligned(li);
                            } else {
                                sr.store_aligned(lr);
                                si.store_aligned(li);
                            }
                        });
                    }
                }
            }

            for (std::size_t k1 = 0; k1 < P1; ++k1) {
                for (std::size_t k2 = 0; k2 < P2; ++k2) {
                    const std::size_t bp2 = b + l1 * k1 + l12 * k2;
                    for (std::size_t t = 0; t < Wa; t += W) {
                        const std::size_t a3 = a0 + t;
                        batch tr[P3], ti_arr[P3];
                        for (std::size_t j3 = 0; j3 < P3; ++j3) {
                            tr[j3]     = batch::load_aligned(tile2re + ((k1 * P2 * P3 + k2 * P3 + j3) * WaMax + t));
                            ti_arr[j3] = batch::load_aligned(tile2im + ((k1 * P2 * P3 + k2 * P3 + j3) * WaMax + t));
                        }
                        dif_butterfly<T, P3>(tr, ti_arr, [&](auto Kc, batch sr, batch si) {
                            if constexpr (Kc > 0u) {
                                const batch owr = batch::load_unaligned(tw3re + ((Kc - 1u) * ido3 + a3));
                                const batch owi = batch::load_unaligned(tw3im + ((Kc - 1u) * ido3 + a3));
                                (owr * sr - owi * si).store_unaligned(chre + (a3 + ido3 * (bp2 + l123 * Kc)));
                                (owr * si + owi * sr).store_unaligned(chim + (a3 + ido3 * (bp2 + l123 * Kc)));
                            } else {
                                sr.store_unaligned(chre + (a3 + ido3 * (bp2 + l123 * Kc)));
                                si.store_unaligned(chim + (a3 + ido3 * (bp2 + l123 * Kc)));
                            }
                        });
                    }
                }
            }
        }
    }
}

template<typename T, bool Forward, std::size_t IP, bool Split = false, std::size_t Eso = 0,
         std::size_t L1 = 0>
void dif_pass_first_impl(const std::complex<T>* data,
                    T* chre, T* chim,
                    std::size_t l1, std::size_t ido,
                    const T* twre, const T* twim,
                    std::size_t eso, std::size_t blk = 0) {
    using batch_t = xsimd::batch<T>;
    constexpr std::size_t W = batch_t::size;
    // L1 == 1 is the only value the engine passes. Then b == 0, the input row j sits at ido*j,
    // the twiddle row k at ido*(k-1) and the output row k at eso*ido*k, so ONE array of IP
    // hoisted ido multiples addresses all three families off three base pointers.
    if constexpr (L1 == 1u && !Split) {
        const T* const src = reinterpret_cast<const T*>(data);
        std::size_t roff[IP];
        poet::static_for<0, IP>([&](const auto j) ADM_LAMBDA_ALWAYS_INLINE { roff[j] = j * ido; });
        auto one = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
            auto emit_tw = [&](const auto k, batch_t sr, batch_t si) ADM_LAMBDA_ALWAYS_INLINE {
                const std::size_t o = (aa + roff[k]) * eso;
                if constexpr (k > 0u) {
                    const std::size_t t = roff[k - 1u] + aa;
                    const batch_t owr = batch_t::load_unaligned(twre + t);
                    const batch_t owi = batch_t::load_unaligned(twim + t);
                    (owr * sr - owi * si).store_unaligned(chre + o);
                    (owr * si + owi * sr).store_unaligned(chim + o);
                } else {
                    sr.store_unaligned(chre + o);
                    si.store_unaligned(chim + o);
                }
            };
            if constexpr (dif_staged_radix<IP>) {
                auto load_in = [&](const auto j, batch_t& lr, batch_t& li)
                                   ADM_LAMBDA_ALWAYS_INLINE {
                    auto [dr, di] = plane_refs<Forward>(lr, li);
                    aos_deinterleave<T>(src + 2u * (aa + roff[j]), dr, di);
                };
                staged_dif_butterfly<T, IP, batch_t>(load_in, emit_tw);
            } else {
                batch_t btr[IP], bti[IP];
                poet::static_for<0, IP>([&](const auto j) ADM_LAMBDA_ALWAYS_INLINE {
                    auto [dr, di] = plane_refs<Forward>(btr[j], bti[j]);
                    aos_deinterleave<T>(src + 2u * (aa + roff[j]), dr, di);
                });
                dif_butterfly<T, IP>(btr, bti, emit_tw);
            }
        };
        std::size_t a = 0;
        for (; a + W <= ido; a += W) one(a);
        if (ido >= W && (ido - a) * 2 >= W) { one(ido - W); a = ido; }
        for (; a < ido; ++a) {
            T tr[IP], ti[IP];
            poet::static_for<0, IP>([&](const auto j) ADM_LAMBDA_ALWAYS_INLINE {
                const auto& c = data[a + roff[j]];
                auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
                dr = c.real();
                di = c.imag();
            });
            dif_butterfly<T, IP>(tr, ti, [&](const auto k, T sr, T si) {
                const std::size_t o = (a + roff[k]) * eso;
                if constexpr (k > 0u) {
                    T owr = T(1), owi = T(0);
                    if (ido > 1) {
                        owr = twre[roff[k - 1u] + a];
                        owi = twim[roff[k - 1u] + a];
                    }
                    chre[o] = owr * sr - owi * si;
                    chim[o] = owr * si + owi * sr;
                } else {
                    chre[o] = sr;
                    chim[o] = si;
                }
            });
        }
        return;
    }
    // fix4: Eso == 0 keeps the element stride a runtime value, which is the shipped form. Eso == 2
    // makes it compile time so the imaginary plane can be named as chre + W, the relation es == 2
    // guarantees, and the eight output columns need one base register instead of two.
#if ADM_FIX4_ES2_ONEBASE
    // Eso == 0 is the sentinel a direct caller leaves in place, and it keeps the runtime stride.
    const std::size_t es = Eso == 0u ? eso : Eso;
#define ADM_F4_ES es
#define ADM_F4_SRE ore
#define ADM_F4_SIM oim
#else
#define ADM_F4_ES eso
#define ADM_F4_SRE chre
#define ADM_F4_SIM chim
#endif
    const std::size_t idz = ido * ADM_F4_ES;
    std::size_t bsh = 0, nb = 0;
    const T* are = nullptr;
    const T* aim = nullptr;
    if constexpr (Split) {
        bsh = static_cast<std::size_t>(detail::countr_zero(blk));
        nb = (ido + blk - 1) >> bsh;
        are = twre + (IP - 1) * blk;
        aim = twim + (IP - 1) * blk;
    }

    // fix3: two columns per iteration share one set of output stream pointers, and the outputs
    // cannot alias the input or the twiddle table, so the twiddle loads may cross the stores.
    T* const ADM_FIX3_FIRST_RESTRICT ore = chre;
#if ADM_FIX4_ES2_ONEBASE
    T* const ADM_FIX3_FIRST_RESTRICT oim = Eso == 2u ? chre + W : chim;
#else
    T* const ADM_FIX3_FIRST_RESTRICT oim = chim;
#endif
    for (std::size_t b = 0; b < l1; ++b) {
        constexpr std::size_t U0 = dif_pass_unroll<IP>();
        constexpr std::size_t U1 =
            (ADM_FIX3_FIRST && !dif_staged_radix<IP> && U0 < 2u) ? 2u : U0;
        // fix4 U2: the override, on the one radix the arm names, where it raises the unroll.
        constexpr bool kU2Here =
            ADM_FIX4_U2 > 1u && (ADM_FIX4_U2_IP == 0u || ADM_FIX4_U2_IP == IP) &&
            U1 < ADM_FIX4_U2;
        constexpr std::size_t U = kU2Here ? std::size_t{ADM_FIX4_U2} : U1;
        auto do_batch = [&](std::size_t aa) ADM_LAMBDA_ALWAYS_INLINE {
            const std::size_t a0 = Split ? (aa & (blk - 1u)) : aa;
            const std::size_t a1 = Split ? (aa >> bsh) : 0u;
            auto emit_tw = [&](const auto k, batch_t sr, batch_t si) {
                if constexpr (k > 0u) {
                    batch_t owr, owi;
                    if constexpr (Split) {
                        const batch_t cr = batch_t::load_unaligned(twre + ((k - 1u) * blk + a0));
                        const batch_t ci = batch_t::load_unaligned(twim + ((k - 1u) * blk + a0));
                        const T ar = are[(k - 1u) * nb + a1], ai = aim[(k - 1u) * nb + a1];
                        owr = ar * cr - ai * ci;
                        owi = ar * ci + ai * cr;
                    } else {
                        owr = batch_t::load_unaligned(twre + ((k - 1u) * ido + aa));
                        owi = batch_t::load_unaligned(twim + ((k - 1u) * ido + aa));
                    }
                    (owr * sr - owi * si)
                        .store_unaligned(ore + (aa * ADM_F4_ES + idz * (b + l1 * k)));
                    (owr * si + owi * sr)
                        .store_unaligned(oim + (aa * ADM_F4_ES + idz * (b + l1 * k)));
                } else {
                    sr.store_unaligned(ore + (aa * ADM_F4_ES + idz * (b + l1 * k)));
                    si.store_unaligned(oim + (aa * ADM_F4_ES + idz * (b + l1 * k)));
                }
            };
            if constexpr (dif_staged_radix<IP>) {
                auto load_in = [&](const auto j, batch_t& lr, batch_t& li)
                                   ADM_LAMBDA_ALWAYS_INLINE {
                    const T* src = reinterpret_cast<const T*>(data + aa + ido * (j + IP * b));
                    auto [dr, di] = plane_refs<Forward>(lr, li);
                    aos_deinterleave<T>(src, dr, di);
                };
                staged_dif_butterfly<T, IP, batch_t>(load_in, emit_tw);
            } else {
                batch_t btr[IP], bti[IP];
                for (std::size_t j = 0; j < IP; ++j) {
                    const T* src = reinterpret_cast<const T*>(data + aa + ido * (j + IP * b));
                    auto [dr, di] = plane_refs<Forward>(btr[j], bti[j]);
                    aos_deinterleave<T>(src, dr, di);
                }
                dif_butterfly<T, IP>(btr, bti, emit_tw);
            }
        };
        std::size_t a = 0;
        if constexpr (U > 1) {
            for (; a + U * W <= ido; a += U * W) {
                poet::static_for<0, U>([&](auto UU) {
                    do_batch(a + UU * W);
                });
            }
        }
#if ADM_FIX4_ROLL
        // fix4 ROLL: dynamic_for<1> launders the trip count through an asm barrier, so neither
        // compiler re-unrolls this loop. The step lives in a static_for, so the body has one
        // shape. At S >= 2 all S blocks load before any butterfly runs, so S chains overlap.
        if constexpr (!Split) {
            constexpr std::size_t S = ADM_FIX4_ROLL;
            auto blk_load = [&](std::size_t aa, batch_t (&tr)[IP], batch_t (&ti)[IP])
                                ADM_LAMBDA_ALWAYS_INLINE {
                poet::static_for<0, IP>([&](const auto j) ADM_LAMBDA_ALWAYS_INLINE {
                    const T* src = reinterpret_cast<const T*>(
                        data + aa + ido * (decltype(j)::value + IP * b));
                    auto [dr, di] = plane_refs<Forward>(tr[decltype(j)::value],
                                                        ti[decltype(j)::value]);
                    aos_deinterleave<T>(src, dr, di);
                });
            };
            auto blk_run = [&](std::size_t aa, const batch_t (&tr)[IP], const batch_t (&ti)[IP])
                               ADM_LAMBDA_ALWAYS_INLINE {
                dif_butterfly<T, IP>(tr, ti, [&](const auto k, batch_t sr, batch_t si)
                                                 ADM_LAMBDA_ALWAYS_INLINE {
                    const std::size_t off = aa * ADM_F4_ES + idz * (b + l1 * k);
                    if constexpr (k > 0u) {
                        const batch_t owr = batch_t::load_unaligned(twre + ((k - 1u) * ido + aa));
                        const batch_t owi = batch_t::load_unaligned(twim + ((k - 1u) * ido + aa));
                        (owr * sr - owi * si).store_unaligned(ore + off);
                        (owr * si + owi * sr).store_unaligned(oim + off);
                    } else {
                        sr.store_unaligned(ore + off);
                        si.store_unaligned(oim + off);
                    }
                });
            };
            const std::size_t nstep = (ido - a) / (S * W);
            if (nstep != 0) {
                const std::size_t abase = a;
                poet::dynamic_for<1, 1>(std::size_t{0}, nstep,
                                        [&](std::size_t i) ADM_LAMBDA_ALWAYS_INLINE {
                    const std::size_t a0 = abase + i * (S * W);
                    if constexpr (S == 1u || dif_staged_radix<IP>) {
                        poet::static_for<0, S>([&](const auto s) ADM_LAMBDA_ALWAYS_INLINE {
                            do_batch(a0 + decltype(s)::value * W);
                        });
                    } else {
                        batch_t br[S][IP], bi[S][IP];
                        poet::static_for<0, S>([&](const auto s) ADM_LAMBDA_ALWAYS_INLINE {
                            blk_load(a0 + decltype(s)::value * W, br[decltype(s)::value],
                                     bi[decltype(s)::value]);
                        });
                        poet::static_for<0, S>([&](const auto s) ADM_LAMBDA_ALWAYS_INLINE {
                            blk_run(a0 + decltype(s)::value * W, br[decltype(s)::value],
                                    bi[decltype(s)::value]);
                        });
                    }
                });
                a = abase + nstep * (S * W);
            }
        }
#endif
        for (; a + W <= ido; a += W) do_batch(a);
        if (!Split && ido >= W && (ido - a) * 2 >= W) { do_batch(ido - W); a = ido; }
        for (; a < ido; ++a) {
            T tr[IP], ti[IP];
            for (std::size_t j = 0; j < IP; ++j) {
                const auto& c = data[a + ido * (j + IP * b)];
                auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
                dr = c.real();
                di = c.imag();
            }
            dif_butterfly<T, IP>(tr, ti, [&](const auto k, T sr, T si) {
                if constexpr (k > 0u) {
                    T owr = T(1), owi = T(0);
                    if constexpr (Split) {
                        const T cr = twre[(k - 1u) * blk + (a & (blk - 1u))];
                        const T ci = twim[(k - 1u) * blk + (a & (blk - 1u))];
                        const T ar = are[(k - 1u) * nb + (a >> bsh)];
                        const T ai = aim[(k - 1u) * nb + (a >> bsh)];
                        owr = ar * cr - ai * ci;
                        owi = ar * ci + ai * cr;
                    } else if (ido > 1) {
                        owr = twre[(k - 1u) * ido + a];
                        owi = twim[(k - 1u) * ido + a];
                    }
                    ADM_F4_SRE[a * ADM_F4_ES + idz * (b + l1 * k)] = owr * sr - owi * si;
                    ADM_F4_SIM[a * ADM_F4_ES + idz * (b + l1 * k)] = owr * si + owi * sr;
                } else {
                    ADM_F4_SRE[a * ADM_F4_ES + idz * (b + l1 * k)] = sr;
                    ADM_F4_SIM[a * ADM_F4_ES + idz * (b + l1 * k)] = si;
                }
            });
        }
    }
}
#undef ADM_F4_ES
#undef ADM_F4_SRE
#undef ADM_F4_SIM

template<typename T, bool Forward, std::size_t IP, bool Split, std::size_t Eso, std::size_t L1,
         typename... A>
ADM_FLATTEN void dif_pass_first_flat(A... a) {
    dif_pass_first_impl<T, Forward, IP, Split, Eso, L1>(a...);
}

template<typename T, bool Forward, std::size_t IP, std::size_t L1 = 0>
void dif_pass_first(const std::complex<T>* data,
                    T* chre, T* chim,
                    std::size_t l1, std::size_t ido,
                    const T* twre, const T* twim,
                    std::size_t eso, std::size_t blk) {
#if ADM_FIX4_ES2_ONEBASE
    const auto run_es = [&](auto split, auto esc) {
        constexpr bool Split = decltype(split)::value;
        constexpr std::size_t Eso = decltype(esc)::value;
        if constexpr (dif_butterfly_wants_reload<IP>)
            dif_pass_first_impl<T, Forward, IP, Split, Eso, L1>(data, chre, chim, l1, ido, twre,
                                                                twim, eso, blk);
        else
            dif_pass_first_flat<T, Forward, IP, Split, Eso, L1>(data, chre, chim, l1, ido, twre,
                                                                twim, eso, blk);
    };
#endif
    const auto run = [&](auto split) {
#if ADM_FIX4_ES2_ONEBASE
        if (eso == 2u) run_es(split, std::integral_constant<std::size_t, 2u>{});
        else run_es(split, std::integral_constant<std::size_t, 1u>{});
#else
        if constexpr (dif_butterfly_wants_reload<IP>)
            dif_pass_first_impl<T, Forward, IP, split, 0u, L1>(data, chre, chim, l1, ido, twre,
                                                               twim, eso, blk);
        else
            dif_pass_first_flat<T, Forward, IP, split, 0u, L1>(data, chre, chim, l1, ido, twre,
                                                               twim, eso, blk);
#endif
    };
    if (blk != 0) run(std::bool_constant<true>{});
    else run(std::bool_constant<false>{});
}

template<std::size_t... Is>
constexpr auto dif_tail_seq_shift(std::index_sequence<Is...>) -> std::index_sequence<(Is + 1)...>;
template<std::size_t W>
using dif_last_tail_seq = decltype(dif_tail_seq_shift(std::make_index_sequence<W - 1>{}));

// Lane-axis stage twiddle w_IP^(R*lane). The last pass loads one row with the lanes along the
// contiguous j axis, so stage A of its outer split runs the radix over the VECTOR index and its
// stage twiddle varies per lane. apply_stage_twiddle cannot serve that.
template<typename T, std::size_t IP, std::size_t R, std::size_t W, bool Imag>
[[nodiscard]] ADM_CONSTEVAL std::array<T, W> lane_stage_twiddle() {
    std::array<T, W> a{};
    for (std::size_t lane = 0; lane < W; ++lane) {
        const auto sc = ct_sincos_turns<ct_real_t<T>>(true, R * lane, IP);
        a[lane] = static_cast<T>(Imag ? sc.s : sc.c);
    }
    return a;
}

// One stage-A output row of the last pass: R == 0 carries no twiddle, so its scale multiply
// applies directly; every other row takes the scale prescaled into its twiddle pair, which
// removes one multiply per output point of the pass.
template<typename T, std::size_t IP, std::size_t R, typename V>
[[nodiscard]] ADM_ALWAYS_INLINE std::pair<V, V> lane_stage_apply(V yr, V yi, V sv) {
    if constexpr (R == 0u) {
        return {yr * sv, yi * sv};
    } else {
        alignas(V::arch_type::alignment()) static constexpr auto twr =
            lane_stage_twiddle<T, IP, R, V::size, false>();
        alignas(V::arch_type::alignment()) static constexpr auto twi =
            lane_stage_twiddle<T, IP, R, V::size, true>();
        const V wr = V::load_aligned(twr.data()) * sv;
        const V wi = V::load_aligned(twi.data()) * sv;
        return {yr * wr - yi * wi, yi * wr + yr * wi};
    }
}

// Physical offset of logical element `a` in an es == 2 plane. The producer writes W real values
// then W imaginary values per W-column block, so one block costs 2W slots and the real slot of `a`
// sits at 2a less the offset of `a` inside its own block. A W-aligned `a` reduces to 2a, which is
// the only form the vector paths below use.
template<std::size_t Esi, std::size_t W>
[[nodiscard]] ADM_ALWAYS_INLINE constexpr std::size_t es_block_off(std::size_t a) {
    return Esi == 1u ? a : 2u * a - a % W;
}

template<typename T, bool Forward, std::size_t IP, std::size_t Esi = 1u>
ADM_COLD ADM_NOINLINE void dif_pass_last_scalar_rows(const T* ccre,
                                            const T* ccim,
                                            std::complex<T>* data,
                                            std::size_t l1, std::size_t b, T scale_val,
                                            const std::uint32_t* rowperm) {
    constexpr std::size_t Wv = xsimd::batch<T>::size;
#if ADM_FIX4_ES2_ONEBASE
    // Esi == 2 is exactly the relation ccim == ccre + W, so one base register serves both planes.
    const T* const cim = Esi == 2u ? ccre + Wv : ccim;
#define ADM_F4_CIM cim
#else
#define ADM_F4_CIM ccim
#endif
    for (; b < l1; ++b) {
        const std::size_t rb = rowperm ? std::size_t(rowperm[b]) : b;
        const std::size_t o0 = Esi * IP * rb;
        T tr[IP], ti[IP];
        for (std::size_t j = 0; j < IP; ++j) {
            const std::size_t off = o0 + es_block_off<Esi, Wv>(j);
            tr[j] = ccre[off];
            ti[j] = ADM_F4_CIM[off];
        }
        dif_butterfly_terminal<T, IP>(tr, ti, [&](const auto k, T sr, T si) {
            const auto [xr, xi] = plane_vals<Forward>(sr * scale_val, si * scale_val);
            data[b + l1 * k] = std::complex<T>(xr, xi);
        });
    }
}

#undef ADM_F4_CIM

template<typename T, std::size_t IP>
struct dif_last_batch {
    static constexpr std::size_t Wn = xsimd::batch<T>::size;
    static constexpr std::size_t Wfit = detail::bit_ceil(IP);
    using sized_t = xsimd::make_sized_batch_t<T, std::min(Wfit, Wn)>;
    using type = std::conditional_t<std::is_void_v<sized_t>, xsimd::batch<T>, sized_t>;
};

template<typename T, bool Forward, std::size_t IP, std::size_t Rows, std::size_t Esi = 1u>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_pass_last_block(const T* ccre,
                                           const T* ccim,
                                           std::complex<T>* data,
                                           std::size_t l1, std::size_t b,
                                           [[maybe_unused]] T scale_val,
                                           const std::uint32_t* rowperm) {
    using batch_t = typename dif_last_batch<T, IP>::type;
    constexpr std::size_t W = batch_t::size;
#if ADM_FIX4_ES2_ONEBASE
    // Esi == 2 is exactly the relation ccim == ccre + W, so one base register serves both planes.
    const T* const cim = Esi == 2u ? ccre + xsimd::batch<T>::size : ccim;
#define ADM_F4_CIM cim
#else
#define ADM_F4_CIM ccim
#endif
    const auto row = [&](std::size_t i) ADM_LAMBDA_ALWAYS_INLINE -> std::size_t {
        return rowperm ? std::size_t(rowperm[i]) : i;
    };
    {
        constexpr bool row_split_path =
            detail::has_single_bit(IP) && IP >= 2 * W && dif_butterfly_wants_reload<IP> &&
            poet::vector_register_count() >= 32 && Rows == W;
        if constexpr (row_split_path) {
            constexpr std::size_t G = IP / W;
            alignas(batch_t::arch_type::alignment()) T stg_re[G * W * W];
            alignas(batch_t::arch_type::alignment()) T stg_im[G * W * W];
            const batch_t sv(scale_val);
            const auto stage = [&](std::size_t bb, std::size_t r) ADM_LAMBDA_ALWAYS_INLINE {
                batch_t rr[G], ri[G];
                poet::static_for<0, G>([&](const auto t) {
                    rr[t] = batch_t::load_unaligned(ccre + Esi * (IP * r + t * W));
                    ri[t] = batch_t::load_unaligned(ADM_F4_CIM + Esi * (IP * r + t * W));
                });
                // Stage A: one radix-G over the vector index, then the lane twiddle with the
                // pass scale folded in.
                sub_dft<T, G, batch_t>(
                    rr, ri, [&](const auto p, batch_t yr, batch_t yi) ADM_LAMBDA_ALWAYS_INLINE {
                        const auto [fr, fi] = lane_stage_apply<T, IP, p, batch_t>(yr, yi, sv);
                        fr.store_aligned(stg_re + (p * W + bb) * W);
                        fi.store_aligned(stg_im + (p * W + bb) * W);
                    });
            };
            if (rowperm)
                for (std::size_t bb = 0; bb < W; ++bb) stage(bb, std::size_t(rowperm[b + bb]));
            else
                for (std::size_t bb = 0; bb < W; ++bb) stage(bb, b + bb);
            poet::static_for<0, G>([&](const auto p) {
                // Stage A emits r in natural order, so stage B needs no bit-reversal offset.
                constexpr std::size_t K0 = std::size_t(p);
                batch_t tr2[W], ti2[W];
                for (std::size_t s = 0; s < W; ++s) {
                    tr2[s] = batch_t::load_aligned(stg_re + (p * W + s) * W);
                    ti2[s] = batch_t::load_aligned(stg_im + (p * W + s) * W);
                }
                xsimd::transpose(tr2, tr2 + W);
                xsimd::transpose(ti2, ti2 + W);
                pow2_dif_butterfly<T, W, batch_t>(
                    tr2, ti2, [&](const auto k, batch_t yr, batch_t yi) {
                        T* dst = reinterpret_cast<T*>(data + b + l1 * (k * G + K0));
                        const auto [xr, xi] = plane_vals<Forward>(yr, yi);
                        aos_interleave<T>(dst, xr, xi);
                    });
            });
            return;
        }
        batch_t btr[IP], bti[IP];
        if constexpr (IP >= W) {
            auto load_tile = [&](const auto off) ADM_LAMBDA_ALWAYS_INLINE {
                batch_t rr[W], ri[W];
                poet::static_for<0, W>([&](const auto bb) {
                    if constexpr (detail::cmp_less(bb.value, Rows)) {
                        rr[bb] = batch_t::load_unaligned(ccre + Esi * (IP * row(b + bb) + off));
                        ri[bb] =
                            batch_t::load_unaligned(ADM_F4_CIM + Esi * (IP * row(b + bb) + off));
                    } else {
                        rr[bb] = batch_t(T(0));
                        ri[bb] = batch_t(T(0));
                    }
                });
                xsimd::transpose(rr, rr + W);
                xsimd::transpose(ri, ri + W);
                poet::static_for<0, W>([&](const auto a) {
                    if constexpr (std::decay_t<decltype(off)>::value + a < IP) {
                        btr[off + a] = rr[a];
                        bti[off + a] = ri[a];
                    }
                });
            };
            poet::static_for<0, IP / W>([&](auto Tc) {
                load_tile(std::integral_constant<std::size_t,
                          Tc * W>{});
            });
            if constexpr (IP % W != 0)
                load_tile(std::integral_constant<std::size_t, IP - W>{});
        } else {
            using arch = typename batch_t::arch_type;
            constexpr auto mask = xsimd::make_batch_bool_constant<T, lane_lt<IP>, arch>();
            batch_t rr[W], ri[W];
            poet::static_for<0, W>([&](const auto bb) {
                if constexpr (detail::cmp_less(bb.value, Rows)) {
                    rr[bb] = batch_t::load(ccre + Esi * IP * row(b + bb), mask,
                                           xsimd::unaligned_mode{});
                    ri[bb] = batch_t::load(ADM_F4_CIM + Esi * IP * row(b + bb), mask,
                                           xsimd::unaligned_mode{});
                } else {
                    rr[bb] = batch_t(T(0));
                    ri[bb] = batch_t(T(0));
                }
            });
            xsimd::transpose(rr, rr + W);
            xsimd::transpose(ri, ri + W);
            for (std::size_t j = 0; j < IP; ++j) { btr[j] = rr[j]; bti[j] = ri[j]; }
        }
        batch_t out_re[IP], out_im[IP];
        dif_butterfly_terminal<T, IP>(btr, bti, [&](const auto k, batch_t sr, batch_t si) {
            out_re[k] = sr;
            out_im[k] = si;
        });
        const batch_t sv(scale_val);
        for (std::size_t k = 0; k < IP; ++k) {
            const auto [xr, xi] = plane_vals<Forward>(out_re[k] * sv, out_im[k] * sv);
            aos_interleave_prefix<Rows>(reinterpret_cast<T*>(data + b + l1 * k), xr, xi);
        }
    }
}

#undef ADM_F4_CIM

template<typename T, bool Forward, std::size_t IP, std::size_t Esi = 1u>
struct dif_last_tail_invoke_t {
    template<std::size_t Rows>
    void operator()(const T* ccre, const T* ccim,
                    std::complex<T>* data, std::size_t l1, std::size_t b, T scale_val,
                    const std::uint32_t* rowperm) const {
        dif_pass_last_block<T, Forward, IP, Rows, Esi>(ccre, ccim, data, l1, b, scale_val,
                                                       rowperm);
    }
};
template<typename T, bool Forward, std::size_t IP, std::size_t Esi = 1u>
inline constexpr dif_last_tail_invoke_t<T, Forward, IP, Esi> dif_last_tail_invoke{};

template<typename T, bool Forward, std::size_t IP, std::size_t Esi = 1u>
void dif_pass_last(const T* ccre, const T* ccim,
                   std::complex<T>* data,
                   std::size_t l1, [[maybe_unused]] std::size_t ido,
                   [[maybe_unused]] const T* twre,
                   [[maybe_unused]] const T* twim,
                   [[maybe_unused]] T scale_val = T(1),
                   const std::uint32_t* rowperm = nullptr) {
    using batch_t = typename dif_last_batch<T, IP>::type;
    constexpr std::size_t W = batch_t::size;
    assert(ido == 1);
    std::size_t b = 0;
    if (const std::size_t peel = aos_store_align_peel<T, W>(data, l1, l1);
        peel != 0 && l1 >= peel + W) {
        poet::dispatch(dif_last_tail_invoke<T, Forward, IP, Esi>,
                       poet::dispatch_param<dif_last_tail_seq<W>>{peel},
                       ccre, ccim, data, l1, std::size_t{0}, scale_val, rowperm);
        for (b = peel; b + W <= l1; b += W)
            dif_pass_last_block<T, Forward, IP, W, Esi>(ccre, ccim, data, l1, b, scale_val,
                                                       rowperm);
    } else {
        for (; b + W <= l1; b += W)
            dif_pass_last_block<T, Forward, IP, W, Esi>(ccre, ccim, data, l1, b, scale_val,
                                                       rowperm);
    }
    if (b == l1) return;
    if (l1 < W) {
        poet::dispatch(dif_last_tail_invoke<T, Forward, IP, Esi>,
                       poet::dispatch_param<dif_last_tail_seq<W>>{l1},
                       ccre, ccim, data, l1, std::size_t{0}, scale_val, rowperm);
        return;
    }
    if (2 * (l1 - b) >= W) {
        dif_pass_last_block<T, Forward, IP, W, Esi>(ccre, ccim, data, l1, l1 - W,
                                                     scale_val, rowperm);
        return;
    }
    dif_pass_last_scalar_rows<T, Forward, IP, Esi>(ccre, ccim, data, l1, b, scale_val, rowperm);
}

}
}

#include "undef_macros.hpp"
