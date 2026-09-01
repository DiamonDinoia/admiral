// Positive control for the clang-tidy and cppcheck gates. NOT part of any build target: it is
// compiled only by `scripts/static_analysis_control.sh`, which requires both tools to REPORT on
// it. A clean run over the library means nothing unless the same flags still fail on known-bad
// code, and a check that has never failed cannot be told apart from one that cannot fail.
//
// `nullPointer` is cppcheck's; `performance-for-range-copy` is clang-tidy's. Both are enabled by
// the shipped configuration. Do not "fix" this file.
#include <vector>

int adm_static_analysis_control(const std::vector<std::vector<int>>& vv) {
    int* p = nullptr;
    int s = 0;
    for (auto v : vv) s += static_cast<int>(v.size());
    *p = s;
    return s;
}
