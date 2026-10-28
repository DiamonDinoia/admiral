// The out-of-line destructors are the key functions of the error types. They put each typeinfo
// and vtable in this TU, built once at the floor, so no engine tier carries a copy.

#include <admiral/errors.hpp>

namespace admiral {

error::~error() = default;
size_error::~size_error() = default;
unsupported_error::~unsupported_error() = default;
internal_error::~internal_error() = default;

}
