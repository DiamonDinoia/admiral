// Positive control for check (h): a namespace-scope plan runs tier code from the loader's init
// list, before any dispatch. Not part of any build: scripts/check_dispatch.sh compiles it the
// way a tier TU is compiled, and (h) must fire on it.

#include <admiral/admiral.hpp>

static admiral::plan<float> p(64);

int init_plan_size() { return static_cast<int>(p.size()); }
