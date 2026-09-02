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

    [[nodiscard]] error_code code() const noexcept { return code_; }

private:
    error_code code_;
};

class ADM_API size_error : public error {
public:
    explicit size_error(const char* what) : error(error_code::invalid_size, what) {}
};

class ADM_API unsupported_error : public error {
public:
    explicit unsupported_error(const char* what) : error(error_code::unsupported, what) {}
};

class ADM_API internal_error : public error {
public:
    explicit internal_error(const char* what) : error(error_code::internal, what) {}
};

}
