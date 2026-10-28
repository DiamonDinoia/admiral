// Positive control for check (b): a thrown class with only inline members puts its typeinfo,
// type name and vtable into every TU that throws it, so a tier copy always carries them. Check
// (b) must fire on it. Not part of any build: scripts/check_dispatch.sh compiles it the way a
// tier TU is compiled. Do not give the class a key function: that is the fix (b) enforces.

#include <stdexcept>

namespace admiral {

class inline_error : public std::runtime_error {
public:
    inline_error() : std::runtime_error("inline control") {}
};

}

namespace admiral::detail {

void throw_inline_control() { throw inline_error(); }

}
