#pragma once

// ============================================================================
// Aggregation shim.
//
// The kernel machinery is split into focused headers:
//   math.hpp        — size-class predicates + codelet_dispatch declaration
//   twiddles.hpp    — dif_twiddle_set + build_dif_twiddle_set
//   butterfly.hpp   — radix_sym_dft / dif_butterfly / dif_pass_unroll
//   dif_passes.hpp  — dif_pass[_first/_last/_fused] + dispatch functors
//   dif_driver.hpp  — iterative_dif_execute[_ws]
//
// This header aggregates them; new code should include only what it needs.
// ============================================================================

#include "math.hpp"
#include "twiddles.hpp"
#include "butterfly.hpp"
#include "dif_passes.hpp"
#include "dif_driver.hpp"

