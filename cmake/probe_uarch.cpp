#include <cstdio>

#include <admiral/detail/build_id.hpp>

int main() {
    constexpr std::string_view u = admiral::detail::build_uarch;
    std::printf("%.*s", static_cast<int>(u.size()), u.data());
}
