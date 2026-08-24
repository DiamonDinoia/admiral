#include "admiral/admiral.h"
#include "admiral/admiral.hpp"
#include <complex>
#include <memory>
#include <optional>
#include <span>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

// Thin translation layer: validate, hand the pointers to the C++ API, map
// exceptions onto adm_status. One template per operation keeps the two
// precisions from drifting apart.

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

// admiral.h's complex structs are layout-compatible with std::complex, so every
// buffer crosses the boundary as a view, and no copy happens.
template<typename T>
std::span<std::complex<T>> to_cpp_span(void* data, size_t size) {
    return std::span<std::complex<T>>(reinterpret_cast<std::complex<T>*>(data), size);
}

// A C caller can pass an eff outside the enum, so this layer guards the cast once
// rather than at every call site; std::nullopt is the rejection. The asserts catch layout
// or value drift between the two declarations of the same options and efforts.
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
    return admiral::options{.nthreads = opts->nthreads,
                            .eff = static_cast<admiral::effort>(opts->eff),
                            .debug = opts->debug};
}

// A shape C can pass but C++ cannot: null, or zero rank. A zero extent or an
// overflowing product throws inside the C++ API; guarded() maps those.
bool bad_shape(const size_t* shape, size_t ndim) { return shape == nullptr || ndim == 0; }

// The detail of the last failing call, per thread: the caught exception's
// what(). Set only on failure so a successful call costs no string store.
thread_local std::string last_error;

void set_last_error(const char* what) noexcept {
    try {
        last_error = what;
    } catch (...) {
        // Assigning can itself fail under memory pressure; keep the old text.
    }
}

// Validation failures return before any call into C++: record their reason here
// so adm_last_error_message() has an answer for every failed call, not only the
// exception-originated ones.
adm_status fail(adm_status status, const char* reason) {
    set_last_error(reason);
    return status;
}

// Run `body` and translate what escapes it. The C++ side throws admiral::error
// subclasses for caller-caused failures and std::bad_alloc for memory. A
// caller-created shape can overflow an internal buffer, so a length_error from
// the containers is also a size report. Anything else is an engine fault, and
// ADM_ERROR_INTERNAL (never a mislabeled size) is what the caller sees.
template<typename F>
adm_status guarded(F&& body) {
    try {
        body();
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        set_last_error("out of memory");
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (const admiral::size_error& e) {
        set_last_error(e.what());
        return ADM_ERROR_INVALID_SIZE;
    } catch (const admiral::error& e) {  // unsupported_error / internal_error
        set_last_error(e.what());
        return ADM_ERROR_INTERNAL;
    } catch (const std::length_error& e) {
        set_last_error(e.what());
        return ADM_ERROR_INVALID_SIZE;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return ADM_ERROR_INTERNAL;
    } catch (...) {
        set_last_error("non-standard exception");
        return ADM_ERROR_INTERNAL;
    }
}

// Convert C options, reject an out-of-enum eff, then run `body(o)` under guarded().
template<typename F>
adm_status guarded_with(const adm_options* opts, F&& body) {
    const auto o = to_cpp_options(opts);
    if (!o) return fail(ADM_ERROR_INVALID_OPTION, "eff is outside the adm_effort enum");
    return guarded([&] { body(*o); });
}

template<typename T>
adm_status run_1d(void* data, size_t size, bool forward, const adm_options* opts) {
    if (data == nullptr) return fail(ADM_ERROR_NULL_POINTER, "null data pointer");
    if (size == 0) return fail(ADM_ERROR_INVALID_SIZE, "size must be greater than 0");
    return guarded_with(opts, [&](const admiral::options& o) {
        const auto span = to_cpp_span<T>(data, size);
        if (forward) admiral::forward(span, span, o);
        else         admiral::inverse(span, span, o);
    });
}

template<typename T>
adm_status run_nd(void* data, const size_t* shape, size_t ndim, bool forward,
                  const adm_options* opts) {
    if (data == nullptr) return fail(ADM_ERROR_NULL_POINTER, "null data pointer");
    if (bad_shape(shape, ndim)) return fail(ADM_ERROR_INVALID_SIZE, "null shape or zero rank");
    return guarded_with(opts, [&](const admiral::options& o) {
        const std::span<const size_t> extents(shape, ndim);
        auto* const p = reinterpret_cast<std::complex<T>*>(data);
        if (forward) admiral::forward(p, extents, o);
        else         admiral::inverse(p, extents, o);
    });
}

template<typename T>
adm_status run_r2c(const void* in, void* out, const size_t* shape, size_t ndim,
                   const adm_options* opts) {
    if (in == nullptr || out == nullptr) return fail(ADM_ERROR_NULL_POINTER, "null in/out pointer");
    if (bad_shape(shape, ndim)) return fail(ADM_ERROR_INVALID_SIZE, "null shape or zero rank");
    return guarded_with(opts, [&](const admiral::options& o) {
        admiral::forward(reinterpret_cast<const T*>(in), reinterpret_cast<std::complex<T>*>(out),
                         std::span<const size_t>(shape, ndim), o);
    });
}

template<typename T>
adm_status run_c2r(void* spec, void* out, const size_t* shape, size_t ndim,
                   const adm_options* opts) {
    if (spec == nullptr || out == nullptr) return fail(ADM_ERROR_NULL_POINTER, "null in/out pointer");
    if (bad_shape(shape, ndim)) return fail(ADM_ERROR_INVALID_SIZE, "null shape or zero rank");
    return guarded_with(opts, [&](const admiral::options& o) {
        admiral::inverse(reinterpret_cast<std::complex<T>*>(spec), reinterpret_cast<T*>(out),
                         std::span<const size_t>(shape, ndim), o);
    });
}

}  // namespace

// ============================================================================
// One-shot transforms
// ============================================================================

adm_status admf_forward(admf_complex* data, size_t size, const adm_options* opts) {
    return run_1d<float>(data, size, /*forward=*/true, opts);
}
adm_status adm_forward(adm_complex* data, size_t size, const adm_options* opts) {
    return run_1d<double>(data, size, /*forward=*/true, opts);
}
adm_status admf_inverse(admf_complex* data, size_t size, const adm_options* opts) {
    return run_1d<float>(data, size, /*forward=*/false, opts);
}
adm_status adm_inverse(adm_complex* data, size_t size, const adm_options* opts) {
    return run_1d<double>(data, size, /*forward=*/false, opts);
}

adm_status admf_forward_nd(admf_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<float>(data, shape, ndim, /*forward=*/true, opts);
}
adm_status adm_forward_nd(adm_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<double>(data, shape, ndim, /*forward=*/true, opts);
}
adm_status admf_inverse_nd(admf_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<float>(data, shape, ndim, /*forward=*/false, opts);
}
adm_status adm_inverse_nd(adm_complex* data, const size_t* shape, size_t ndim,
                          const adm_options* opts) {
    return run_nd<double>(data, shape, ndim, /*forward=*/false, opts);
}

// ============================================================================
// Real transforms
// ============================================================================

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

// ============================================================================
// Plans
// ============================================================================

// The variant owns the plan, so which precision it holds IS the type tag: no
// enum to keep in sync, no void* to cast back, no destructor to write.
struct adm_plan_s {
    template<typename P, typename... Args>
    explicit adm_plan_s(std::in_place_type_t<P> tag, Args&&... args)
        : plan(tag, std::forward<Args>(args)...) {}

    std::variant<admiral::plan<float>, admiral::plan<double>> plan;
};

namespace {

// admiral::plan<T> takes a runtime shape, so 1-D and N-D are the same
// construction over a shape span and differ only in ndim.
template<typename T>
adm_status make_plan(adm_plan* plan, const size_t* shape, size_t ndim, const adm_options* opts) {
    if (plan == nullptr) return fail(ADM_ERROR_NULL_POINTER, "null plan output pointer");
    if (bad_shape(shape, ndim)) return fail(ADM_ERROR_INVALID_SIZE, "null shape or zero rank");
    return guarded_with(opts, [&](const admiral::options& o) {
        *plan = std::make_unique<adm_plan_s>(std::in_place_type<admiral::plan<T>>,
                                             std::span<const size_t>(shape, ndim), o)
                    .release();
    });
}

// The variant lookup is also the precision check.
template<typename T>
adm_status plan_execute(adm_plan plan, auto* data, bool forward) {
    if (plan == nullptr) return fail(ADM_ERROR_INVALID_PLAN, "null plan");
    if (data == nullptr) return fail(ADM_ERROR_NULL_POINTER, "null data pointer");
    auto* p = std::get_if<admiral::plan<T>>(&plan->plan);
    if (p == nullptr) return fail(ADM_ERROR_INVALID_PLAN, "plan precision does not match this call");
    return guarded([&] {
        const auto span = to_cpp_span<T>(data, p->size());
        if (forward) p->forward(span);
        else         p->inverse(span);
    });
}

}  // namespace

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
    return plan_execute<float>(plan, data, /*forward=*/true);
}
adm_status adm_plan_execute_forward(adm_plan plan, adm_complex* data) {
    return plan_execute<double>(plan, data, /*forward=*/true);
}
adm_status admf_plan_execute_inverse(adm_plan plan, admf_complex* data) {
    return plan_execute<float>(plan, data, /*forward=*/false);
}
adm_status adm_plan_execute_inverse(adm_plan plan, adm_complex* data) {
    return plan_execute<double>(plan, data, /*forward=*/false);
}

void adm_plan_destroy(adm_plan plan) {
    delete plan;
}

size_t adm_plan_size(adm_plan plan) {
    if (plan == nullptr) return 0;
    return std::visit([](const auto& p) { return p.size(); }, plan->plan);
}

const char* adm_last_error_message() { return last_error.c_str(); }
