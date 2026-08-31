#pragma once

// The error taxonomy of the C++ API. Every exception a caller can cause
// derives from `admiral::error` and carries an `error_code`, so a caller never
// parses `what()`. The C boundary (`src/c_api.cpp`) catches each exception by
// type and maps each category onto its `adm_status`, so the two APIs always
// agree. Allocation failure stays `std::bad_alloc` in both APIs.

#include <stdexcept>

#ifndef ADM_API
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define ADM_API __declspec(dllexport)
#  elif defined(__GNUC__)
#    define ADM_API __attribute__((visibility("default")))
#  else
#    define ADM_API
#  endif
#endif

namespace admiral {

/// Category of every exception the API throws; `error::code()` returns the category.
enum class error_code { invalid_size, unsupported, internal };

/// Base of every exception the API throws. `ADM_API` (default visibility)
/// because a throw crosses the shared-library boundary: hidden typeinfo makes
/// catch-by-type never match in the caller's image.
class ADM_API error : public std::runtime_error {
public:
    error(error_code code, const char* what) : std::runtime_error(what), code_(code) {}

    [[nodiscard]] error_code code() const noexcept { return code_; }

private:
    error_code code_;
};

/// Bad size or shape: zero extent, data/plan size mismatch, out-of-range axis
/// or box, an extent product that overflows `size_t`.
class ADM_API size_error : public error {
public:
    explicit size_error(const char* what) : error(error_code::invalid_size, what) {}
};

/// The requested (forced) route or kernel is unavailable for this size/precision.
class ADM_API unsupported_error : public error {
public:
    explicit unsupported_error(const char* what) : error(error_code::unsupported, what) {}
};

/// An internal invariant broke; the caller's arguments are never the cause.
class ADM_API internal_error : public error {
public:
    explicit internal_error(const char* what) : error(error_code::internal, what) {}
};

}  // namespace admiral
