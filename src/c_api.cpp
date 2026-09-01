#include "admiral/admiral.h"
#include "admiral/admiral.hpp"
#include <complex>
#include <memory>
#include <optional>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include "admiral/detail/cxx_compat.hpp"

const char* adm_error_string(adm_status status) {
    switch (status) {
        case ADM_SUCCESS: return "Success";
        case ADM_ERROR_NULL_POINTER: return "Null pointer argument";
        case ADM_ERROR_INVALID_SIZE: return "Invalid size or shape";
        case ADM_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case ADM_ERROR_INVALID_PLAN: return "Invalid plan";
        case ADM_ERROR_INVALID_OPTION: return "Invalid option value";
        case ADM_ERROR_INTERNAL: return "Internal error";
        default: return "Unknown error";
    }
}

namespace {

template<typename T>
admiral::span<std::complex<T>> to_cpp_span(void* data, size_t size) {
    return admiral::span<std::complex<T>>(reinterpret_cast<std::complex<T>*>(data), size);
}

static_assert(sizeof(adm_options) == sizeof(admiral::options),
              "adm_options and admiral::options are out of sync: update this mapping");
static_assert(static_cast<int>(admiral::effort::estimate) == ADM_EFFORT_ESTIMATE);
static_assert(static_cast<int>(admiral::effort::automatic) == ADM_EFFORT_AUTOMATIC);
static_assert(static_cast<int>(admiral::effort::measure) == ADM_EFFORT_MEASURE);
std::optional<admiral::options> to_cpp_options(const adm_options* opts) {
    if (opts == nullptr) return admiral::options{};
    if (opts->eff != ADM_EFFORT_ESTIMATE && opts->eff != ADM_EFFORT_AUTOMATIC &&
        opts->eff != ADM_EFFORT_MEASURE)
        return std::nullopt;
    return admiral::options{opts->nthreads,
                            static_cast<admiral::effort>(opts->eff),
                            opts->debug};
}

bool bad_shape(const size_t* shape, size_t ndim) { return shape == nullptr || ndim == 0; }

template<typename F>
adm_status guarded(F&& body) {
    try {
        body();
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (const admiral::size_error&) {
        return ADM_ERROR_INVALID_SIZE;
    } catch (const admiral::error&) {
        return ADM_ERROR_INTERNAL;
    } catch (const std::length_error&) {
        return ADM_ERROR_INVALID_SIZE;
    } catch (const std::exception&) {
        return ADM_ERROR_INTERNAL;
    } catch (...) {
        return ADM_ERROR_INTERNAL;
    }
}

template<typename F>
adm_status guarded_with(const adm_options* opts, F&& body) {
    const auto o = to_cpp_options(opts);
    if (!o) return ADM_ERROR_INVALID_OPTION;
    return guarded([&] { body(*o); });
}

template<typename T>
adm_status run_1d(void* data, size_t size, bool forward, const adm_options* opts) {
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (size == 0) return ADM_ERROR_INVALID_SIZE;
    return guarded_with(opts, [&](const admiral::options& o) {
        const auto span = to_cpp_span<T>(data, size);
        if (forward) admiral::forward(span, span, o);
        else         admiral::inverse(span, span, o);
    });
}

template<typename T>
adm_status run_nd(void* data, const size_t* shape, size_t ndim, bool forward,
                  const adm_options* opts) {
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (bad_shape(shape, ndim)) return ADM_ERROR_INVALID_SIZE;
    return guarded_with(opts, [&](const admiral::options& o) {
        const admiral::span<const size_t> extents(shape, ndim);
        auto* const p = reinterpret_cast<std::complex<T>*>(data);
        if (forward) admiral::forward(p, extents, o);
        else         admiral::inverse(p, extents, o);
    });
}

template<typename T>
adm_status run_r2c(const void* in, void* out, const size_t* shape, size_t ndim,
                   const adm_options* opts) {
    if (in == nullptr || out == nullptr) return ADM_ERROR_NULL_POINTER;
    if (bad_shape(shape, ndim)) return ADM_ERROR_INVALID_SIZE;
    return guarded_with(opts, [&](const admiral::options& o) {
        admiral::forward(reinterpret_cast<const T*>(in), reinterpret_cast<std::complex<T>*>(out),
                         admiral::span<const size_t>(shape, ndim), o);
    });
}

template<typename T>
adm_status run_c2r(void* spec, void* out, const size_t* shape, size_t ndim,
                   const adm_options* opts) {
    if (spec == nullptr || out == nullptr) return ADM_ERROR_NULL_POINTER;
    if (bad_shape(shape, ndim)) return ADM_ERROR_INVALID_SIZE;
    return guarded_with(opts, [&](const admiral::options& o) {
        admiral::inverse(reinterpret_cast<std::complex<T>*>(spec), reinterpret_cast<T*>(out),
                         admiral::span<const size_t>(shape, ndim), o);
    });
}

}

adm_status admf_forward(admf_complex* data, size_t size, const adm_options* opts) {
    return run_1d<float>(data, size, true, opts);
}
adm_status adm_forward(adm_complex* data, size_t size, const adm_options* opts) {
    return run_1d<double>(data, size, true, opts);
}
adm_status admf_inverse(admf_complex* data, size_t size, const adm_options* opts) {
    return run_1d<float>(data, size, false, opts);
}
adm_status adm_inverse(adm_complex* data, size_t size, const adm_options* opts) {
    return run_1d<double>(data, size, false, opts);
}

adm_status admf_forward_nd(admf_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<float>(data, shape, ndim, true, opts);
}
adm_status adm_forward_nd(adm_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<double>(data, shape, ndim, true, opts);
}
adm_status admf_inverse_nd(admf_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<float>(data, shape, ndim, false, opts);
}
adm_status adm_inverse_nd(adm_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<double>(data, shape, ndim, false, opts);
}

adm_status admf_r2c_nd(const float* in, admf_complex* out, const size_t* shape, size_t ndim,
                       const adm_options* opts) {
    return run_r2c<float>(in, out, shape, ndim, opts);
}
adm_status adm_r2c_nd(const double* in, adm_complex* out, const size_t* shape, size_t ndim,
                       const adm_options* opts) {
    return run_r2c<double>(in, out, shape, ndim, opts);
}
adm_status admf_c2r_nd(admf_complex* spec, float* out, const size_t* shape, size_t ndim,
                       const adm_options* opts) {
    return run_c2r<float>(spec, out, shape, ndim, opts);
}
adm_status adm_c2r_nd(adm_complex* spec, double* out, const size_t* shape, size_t ndim,
                       const adm_options* opts) {
    return run_c2r<double>(spec, out, shape, ndim, opts);
}

struct adm_plan_s {
    template<typename P, typename... Args>
    explicit adm_plan_s(std::in_place_type_t<P> tag, Args&&... args)
        : plan(tag, std::forward<Args>(args)...) {}
    adm_plan_s(adm_status status, const char* detail) : err(status), what(detail) {}

    std::variant<std::monostate, admiral::plan<float>, admiral::plan<double>> plan;
    adm_status err = ADM_SUCCESS;
    std::string what;
};

namespace {

adm_status plan_error(adm_plan* out, adm_status status, const char* detail) noexcept {
    *out = nullptr;
    try {
        *out = new adm_plan_s(status, detail);
    } catch (...) {
    }
    return status;
}

template<typename T>
adm_status make_plan(adm_plan* plan, const size_t* shape, size_t ndim, const adm_options* opts) {
    if (plan == nullptr) return ADM_ERROR_NULL_POINTER;
    *plan = nullptr;
    if (bad_shape(shape, ndim))
        return plan_error(plan, ADM_ERROR_INVALID_SIZE, "null shape or zero rank");
    const auto o = to_cpp_options(opts);
    if (!o) return plan_error(plan, ADM_ERROR_INVALID_OPTION, "eff is outside the adm_effort enum");
    adm_status st;
    std::string detail;
    try {
        *plan = std::make_unique<adm_plan_s>(std::in_place_type<admiral::plan<T>>,
                                             admiral::span<const size_t>(shape, ndim), *o)
                    .release();
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        st = ADM_ERROR_OUT_OF_MEMORY;
    } catch (const admiral::size_error& e) {
        st = ADM_ERROR_INVALID_SIZE;
        detail = e.what();
    } catch (const admiral::error& e) {
        st = ADM_ERROR_INTERNAL;
        detail = e.what();
    } catch (const std::length_error& e) {
        st = ADM_ERROR_INVALID_SIZE;
        detail = e.what();
    } catch (const std::exception& e) {
        st = ADM_ERROR_INTERNAL;
        detail = e.what();
    } catch (...) {
        st = ADM_ERROR_INTERNAL;
        detail = "non-standard exception";
    }
    return plan_error(plan, st, detail.empty() ? adm_error_string(st) : detail.c_str());
}

template<typename T, typename U>
adm_status plan_execute(adm_plan plan, U* data, bool forward) {
    if (plan == nullptr) return ADM_ERROR_INVALID_PLAN;
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    auto* p = std::get_if<admiral::plan<T>>(&plan->plan);
    if (p == nullptr) return plan->err != ADM_SUCCESS ? plan->err : ADM_ERROR_INVALID_PLAN;
    return guarded([&] {
        const auto span = to_cpp_span<T>(data, p->size());
        if (forward) p->forward(span);
        else         p->inverse(span);
    });
}

}

adm_status admf_plan_1d(adm_plan* plan, size_t n, const adm_options* opts) {
    return make_plan<float>(plan, &n, 1, opts);
}
adm_status adm_plan_1d(adm_plan* plan, size_t n, const adm_options* opts) {
    return make_plan<double>(plan, &n, 1, opts);
}
adm_status admf_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                         const adm_options* opts) {
    return make_plan<float>(plan, shape, ndim, opts);
}
adm_status adm_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                         const adm_options* opts) {
    return make_plan<double>(plan, shape, ndim, opts);
}

adm_status admf_plan_execute_forward(adm_plan plan, admf_complex* data) {
    return plan_execute<float>(plan, data, true);
}
adm_status adm_plan_execute_forward(adm_plan plan, adm_complex* data) {
    return plan_execute<double>(plan, data, true);
}
adm_status admf_plan_execute_inverse(adm_plan plan, admf_complex* data) {
    return plan_execute<float>(plan, data, false);
}
adm_status adm_plan_execute_inverse(adm_plan plan, adm_complex* data) {
    return plan_execute<double>(plan, data, false);
}

void adm_plan_destroy(adm_plan plan) {
    delete plan;
}

size_t adm_plan_size(adm_plan plan) {
    if (plan == nullptr) return 0;
    return std::visit(
        [](const auto& p) -> size_t {
            if constexpr (std::is_same_v<std::decay_t<decltype(p)>, std::monostate>) return 0;
            else return p.size();
        },
        plan->plan);
}

adm_status adm_plan_status(adm_plan plan) {
    if (plan == nullptr) return ADM_ERROR_NULL_POINTER;
    return plan->err;
}

const char* adm_plan_error_message(adm_plan plan) {
    if (plan == nullptr) return "null plan";
    return plan->what.c_str();
}
