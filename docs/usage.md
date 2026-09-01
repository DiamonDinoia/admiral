# Usage

Common to all three interfaces: data is contiguous row-major, last axis fastest,
and forward uses exp(-2πikn/N). On the C++ and C interfaces inverse divides by
the element count, so a round trip returns the input (`fct` overrides the scale
on the C++ interface); the FFTW shim keeps FFTW's convention instead (both
directions unscaled, round trip multiplies by N). Each interface header opens
with its full contract: layout, scaling, errors, threading.

## Options

Every plan and one-shot takes an optional `admiral::options` aggregate, so a
call site names what it sets and nothing else. The C API mirrors it as
`adm_options`, passed by pointer, where `NULL` means the defaults.

| field | default | what it does |
|-------|---------|--------------|
| `nthreads` | `0` | worker threads owned by the plan. `0` is auto: serial below 2^15 elements, otherwise the power-of-two width that minimises modelled work plus per-dispatch wake cost, capped at the allowed physical cores. `1` forces serial, `n` forces `n` |
| `eff` | `estimate` | how hard construction works to pick a route. `estimate` uses the fitted cost model, so it is fast and reproducible. `automatic` also times the model's top candidates; pick it when one plan serves many transforms. `measure` is the same race, kept for the FFTW flag mapping |
| `debug` | `0` | stderr trace per execute: `0` silent, `1` what ran, `2` adds the shape, `3` adds the cost-model ranking |

`automatic` and `measure` elect from timings, so the picked route depends on the
machine and the load; both are inert under `-DADM_MEASURE=OFF`. The one-shot
functions ignore `eff` and always route with `estimate`, since a discarded plan
cannot repay a plan-time race.

## Threads

A plan owns its worker pool, sized by `options.nthreads`. Two threads may call
`forward` or `inverse` on the same plan object. An execute holds no shared
mutable state: every scratch buffer it needs is per call. A call that engages
the pool takes it exclusively, so concurrent calls on one plan do not overlap
their parallel regions, and a shared plan buys correctness rather than
throughput. Give each thread its own plan to overlap transforms; distinct plan
objects are independent. Below the threading threshold, and at `nthreads = 1`,
no pool exists and concurrent calls run fully in parallel.

`adm_plan_execute_*` carries the same guarantee. On the shim, `fftw_execute(p)`
reuses the buffers `p` was planned with, so two concurrent calls write one
output array: that race is in the caller's memory, not in the plan. Pass
per-thread buffers to `fftw_execute_dft` instead.

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

Runnable: [examples/c_api.c](https://github.com/DiamonDinoia/admiral/blob/master/examples/c_api.c).

Double precision keeps the plain name, single precision prefixes `admf_`. Every
call returns an `adm_status` (`ADM_SUCCESS` is 0, nodiscard);
`adm_error_string()` turns a status into text. A failed plan construction still
writes a handle that carries the failure in place of NULL:
`adm_plan_status()` re-reads it, `adm_plan_error_message()` gives the reason
(the rejection text or the caught exception's message), `adm_plan_destroy()`
releases it, and executing it re-returns the recorded status.
`ADM_ERROR_INTERNAL` marks a fault outside the caller's arguments; everything
else names the argument that caused it.

| status | cause |
|--------|-------|
| `ADM_SUCCESS` | call performed |
| `ADM_ERROR_NULL_POINTER` | a pointer argument was null |
| `ADM_ERROR_INVALID_SIZE` | zero size/rank/extent, extent product overflow, span/plan size mismatch |
| `ADM_ERROR_OUT_OF_MEMORY` | an allocation failed |
| `ADM_ERROR_INVALID_PLAN` | null plan or float/double precision mismatch. A plan that failed construction replays its own status instead |
| `ADM_ERROR_INVALID_OPTION` | an options field outside its enum |
| `ADM_ERROR_INTERNAL` | fault not caused by the arguments; please report |

A zeroed options struct means the defaults, so partial initializers are safe; the
`eff` field takes `ADM_EFFORT_ESTIMATE` / `ADM_EFFORT_AUTOMATIC` /
`ADM_EFFORT_MEASURE`.

One-shots skip the plan: `adm_forward` / `adm_inverse` (1-D),
`adm_forward_nd` / `adm_inverse_nd`, and the real-transform pairs
`adm_r2c_nd` / `adm_c2r_nd`, each with an `admf_` mirror. `adm_plan_size()`
returns the element count a plan expects. The header documents every call.

## FFTW

Existing FFTW code compiles unchanged against `<admiral/fftw3.h>`:

```c
fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
```

Runnable: [examples/fftw_dropin.c](https://github.com/DiamonDinoia/admiral/blob/master/examples/fftw_dropin.c).
Both directions are unscaled, matching FFTW, so a round trip multiplies the
input by N.

Covered: `fftw_plan_dft` and its 1d/2d/3d forms, `fftw_execute` and
`fftw_execute_dft`, destroy, the alloc helpers, `fftw_cleanup`, and the `fftwf_`
mirror. Not covered: real/r2r transforms, guru and split interfaces, wisdom,
`fftw_plan_with_nthreads`. An uncovered call does not compile rather than
degrade. `FFTW_ESTIMATE` maps to `estimate`; every other flag, `FFTW_MEASURE`
included, maps to the plan-time race. See Threads for what a shim plan shared
across threads does and does not guarantee.

A plan call fails with NULL, as in FFTW; NULL alone carries no reason. For the
reason, plan the same shape through the C API: a failed `adm_plan_1d` writes a
handle that reports it via `adm_plan_error_message()`.
