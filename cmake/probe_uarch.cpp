// Configure-time probe: print the µarch the current flags select (build_uarch).
// CostModel.cmake mixes it into the sweep receipt name so two machines whose
// flags-hash collides (same compiler + -march=native on different silicon) do
// not overwrite each other's receipts in a shared data dir.
#include <cstdio>

#include <admiral/detail/build_id.hpp>

int main() {
    constexpr std::string_view u = admiral::detail::build_uarch;
    std::printf("%.*s", static_cast<int>(u.size()), u.data());
}
