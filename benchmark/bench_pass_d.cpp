// Double half of the --pass microbench; the definition is in `bench_pass.hpp`.
#include "bench_pass.hpp"

namespace bench {

template void pass_microbench<double>(unsigned, std::size_t, std::size_t, bool, int, long, long);

}  // namespace bench
