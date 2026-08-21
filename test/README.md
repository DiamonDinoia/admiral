# Tests

Catch2 v3. One executable per source file (a fix rebuilds and relinks one small
binary, not a suite). `catch_discover_tests` registers each one with ctest.

## Run

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev              # or: ctest --test-dir build/dev --output-on-failure
```

Any test-enabled preset works (`debug`, `asan`, `tsan`, `coverage`,
`relwithdebinfo`). `scripts/validate.sh` runs the whole matrix of ISAs,
compilers, sanitizers and valgrind, with `tests` executed under each arm. See its
header comment for the arm list.

Run one area or one case through Catch2 filters, e.g.
`ctest --test-dir build/dev -R test_route` or the binary directly:
`build/dev/test/test_plan "*[fftw]*"`.

## Layout

- `api/` holds the public interfaces: C++, C, FFTW compatibility
- `kernels/` holds codelets, the iterative DIF driver, factor planning, ct_math
- `transforms/` holds plan selection and routing, N-D, axis, fct, threads, alignment
- `accuracy/` holds ULP bounds, analytical closed forms, exhaustive sweeps
- `utils/` holds the shared reference DFT, tolerance model and Catch2 matchers,
  included as `"utils/<file>"` from any area
- `package/` consumes an *installed* admiral (the export set, the installed
  headers). Standalone by design:

  ```bash
  cmake --install build/release --prefix /tmp/adm
  cmake -S test/package -B build-pkg -DCMAKE_PREFIX_PATH=/tmp/adm
  cmake --build build-pkg && ctest --test-dir build-pkg
  ```

## ABI gate

The `exported_symbols.*` tests run `nm` over each installed shared library and
check the export list against the public pattern for that interface (namespace
`admiral`, `adm_`/`admf_`, `fftw_`/`fftwf_`). They gate on the ELF
`--version-script` linker feature, so they skip on Mach-O. The configure log
says when.
