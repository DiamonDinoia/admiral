#pragma once

#include <stdexcept>

#include <admiral/detail/api.h>

#ifndef ADM_API
#  define ADM_API ADM_VISIBILITY
#endif

namespace admiral {

enum class error_code { invalid_size, unsupported, internal };

class ADM_API error : public std::runtime_error {
public:
    error(error_code code, const char* what) : std::runtime_error(what), code_(code) {}
#ifdef ADM_DISPATCH
    // The destructor is the key function: a dispatch build emits the typeinfo and the vtable
    // only in src/errors.cpp, built once at the floor, so no engine tier carries a copy
    // (renamed tier copies would defeat catch by type). Defined only when the library itself
    // is built with ADM_DISPATCH; consumers and ADM_DISPATCH=OFF builds keep master's form.
    ~error() override;
#endif

    [[nodiscard]] error_code code() const noexcept { return code_; }

private:
    error_code code_;
};

class ADM_API size_error : public error {
public:
    explicit size_error(const char* what) : error(error_code::invalid_size, what) {}
#ifdef ADM_DISPATCH
    ~size_error() override;
#endif
};

class ADM_API unsupported_error : public error {
public:
    explicit unsupported_error(const char* what) : error(error_code::unsupported, what) {}
#ifdef ADM_DISPATCH
    ~unsupported_error() override;
#endif
};

class ADM_API internal_error : public error {
public:
    explicit internal_error(const char* what) : error(error_code::internal, what) {}
#ifdef ADM_DISPATCH
    ~internal_error() override;
#endif
};

}
