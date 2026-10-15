# Tests
Catch2 v3. One executable per source file, so a fix rebuilds and relinks one small binary, not a suite.
`catch_discover_tests` registers each binary with ctest.

## Run
```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev              # or: ctest --test-dir build/dev --output-on-failure
```

Any test-enabled preset works (`debug`, `asan`, `tsan`, `coverage`, `relwithdebinfo`). The `release` preset
builds no tests; `-DADM_BUILD_TESTS=ON` enables the full suite. `scripts/validate.sh`
runs the whole matrix of ISAs, compilers, sanitizers, and valgrind. See its header comment for the arm list.
Filter with Catch2: `ctest --test-dir build/dev -R test_route`, or run one binary directly:
`build/dev/test/test_plan "*[fftw]*"`.

## Layout
- `api/`: the public interfaces: C++, C, FFTW compatibility.
- `kernels/`: codelets, the iterative DIF driver, factor planning, ct_math.
- `transforms/`: plan selection and routing, N-D, axis, fct, threads, alignment.
- `accuracy/`: ULP bounds, analytical closed forms, exhaustive sweeps.
- `utils/`: the shared reference DFT, tolerance model, and Catch2 matchers, included as `"utils/<file>"`.
- `package/`: consumes an *installed* admiral (the export set, the installed headers). Standalone by design:

  ```bash
  cmake --install build/release --prefix /tmp/adm
  cmake -S test/package -B build-pkg -DCMAKE_PREFIX_PATH=/tmp/adm
  cmake --build build-pkg && ctest --test-dir build-pkg
  ```

## ABI gate
The `exported_symbols.*` tests run `nm` over each installed shared library and check the export list against
the public pattern of that interface (`admiral`, `adm_`/`admf_`, `fftw_`/`fftwf_`). The tests gate on the ELF
`--version-script` linker feature, so they skip on Mach-O. The configure log says when.
