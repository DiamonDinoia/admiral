# Usage
All three interfaces share one data contract: the data is contiguous row-major, the last axis is fastest, and
forward applies exp(-2πikn/N). On C++ and C, inverse divides by the element count, so a round trip returns the
input (`fct` overrides the scale). The FFTW shim keeps FFTW's convention (both directions unscaled): a round
trip multiplies by N. Each interface header states the full contract: layout, scaling, errors, and threading.

## Options
Every plan and one-shot takes an optional `admiral::options` aggregate (C: `adm_options` by pointer. `NULL`
means the defaults).

| field | default | what it does |
|-------|---------|--------------|
| `nthreads` | `0` | worker threads the plan owns. `0` is auto: serial below 2^15 elements, otherwise the power-of-two width that minimizes modelled work plus per-dispatch wake cost, capped at the allowed physical cores. `1` forces serial, `n` forces `n` |
| `eff` | `estimate` | how hard construction works to pick a route. `estimate` routes by the fitted cost model: fast and reproducible. `automatic` also times the model's top candidates; pick it when one plan serves many transforms. `measure` is the same race, kept for the FFTW flag mapping |
| `debug` | `0` | stderr trace per execute: `0` silent, `1` what ran, `2` adds the shape, `3` adds the cost-model ranking |
| `pin_threads` | `false` | binds each pool worker to one physical core, taken in order from the cores the process is allowed to run on. Off by default. See [Threads](#threads) |

`automatic` and `measure` elect from timings, so the picked route depends on the machine and the load.
`-DADM_MEASURE=OFF` makes both inert. One-shots ignore `eff`: a discarded plan cannot repay a plan-time race.

## Threads
A plan owns its worker pool, sized by `options.nthreads`. Two threads may call `forward` or `inverse` on the
same plan object. An execute holds no shared mutable state: every scratch buffer it needs is per call. A call
that engages the pool takes the pool exclusively, so concurrent calls on one plan do not overlap their
parallel regions: a shared plan gives correctness, not throughput. Give each thread its own plan to overlap
transforms. Distinct plan objects are independent. Below the threading threshold and at `nthreads = 1`, no
pool exists and concurrent calls run fully in parallel.

`pin_threads` binds each spawned worker to a single physical core, taken in order from the intersection of the
process affinity mask with the physical cores, one cpu per core, ordered by socket then cpu. Thus `taskset`,
`numactl`, and an MPI launcher keep full control of the cpus admiral uses. The calling thread never moves. A
mask smaller than the pool width, an unreadable `/sys` topology, or a rejected `sched_setaffinity` leaves the
pool unpinned and prints one note on stderr. Pinning is Linux-only and changes scheduling only: a pinned plan
returns the same bits as an unpinned one.

`adm_plan_execute_*` carries the same guarantee. On the shim, `fftw_execute(p)` reuses the buffers `p` was
planned with, so two concurrent calls write one output array. Pass per-thread buffers to `fftw_execute_dft`.

## C
```c
#include <admiral/admiral.h>

adm_plan plan;
if (adm_plan_1d(&plan, 1024, NULL) != ADM_SUCCESS)   // default options; N-D: adm_plan_nd
    return 1;
if (adm_plan_execute_forward(plan, data) != ADM_SUCCESS)  // data: adm_complex[1024], in place
    return 1;
if (adm_plan_execute_inverse(plan, data) != ADM_SUCCESS)
    return 1;
adm_plan_destroy(plan);
```

Runnable: [../examples/c_api.c](../examples/c_api.c). Double precision keeps the plain name and single
precision prefixes `admf_`. Every call returns an `adm_status` (`ADM_SUCCESS` is 0, nodiscard), and
`adm_error_string()` turns a status into text. A failed plan construction still writes a handle:
`adm_plan_status()` re-reads the failure, `adm_plan_error_message()` returns the reason, `adm_plan_destroy()`
releases the handle.

| status | cause |
|--------|-------|
| `ADM_SUCCESS` | call performed |
| `ADM_ERROR_NULL_POINTER` | a pointer argument was null |
| `ADM_ERROR_INVALID_SIZE` | zero size/rank/extent, extent product overflow, span/plan size mismatch |
| `ADM_ERROR_OUT_OF_MEMORY` | an allocation failed |
| `ADM_ERROR_INVALID_PLAN` | null plan or float/double precision mismatch. A plan that failed construction replays its own status instead |
| `ADM_ERROR_INVALID_OPTION` | an options field outside its enum |
| `ADM_ERROR_INTERNAL` | a fault outside the caller's arguments; please report |

A zeroed options struct gives the defaults, so partial initializers are safe. The `eff` field takes
`ADM_EFFORT_ESTIMATE`, `ADM_EFFORT_AUTOMATIC`, or `ADM_EFFORT_MEASURE`. One-shots skip the plan: `adm_forward`
/ `adm_inverse` (1-D), `adm_forward_nd` / `adm_inverse_nd`, and the real pairs `adm_r2c_nd` / `adm_c2r_nd`,
each with an `admf_` mirror. `adm_plan_size()` returns the element count a plan expects. The header documents
every call.

## FFTW
Existing FFTW code compiles unchanged against `<admiral/fftw3.h>`:

```c
fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
```

Runnable: [../examples/fftw_dropin.c](../examples/fftw_dropin.c).

Covered: `fftw_plan_dft` and its 1d/2d/3d forms, `fftw_execute` and `fftw_execute_dft`, `fftw_destroy_plan`,
the alloc helpers, `fftw_cleanup`, and the `fftwf_` mirror. Not covered: real and r2r transforms, the guru and
split interfaces, wisdom, and `fftw_plan_with_nthreads`. An uncovered call fails at compile time.
`FFTW_ESTIMATE` maps to `estimate`. Every other flag, `FFTW_MEASURE` included, maps to the plan-time race. See
[Threads](#threads) for a shim plan shared across threads. A plan call fails with NULL, as in FFTW, and NULL
carries no reason. For the reason, plan the same shape through the C API and read `adm_plan_error_message()`.
