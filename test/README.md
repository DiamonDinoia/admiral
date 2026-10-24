# Tests
Catch2 v3 tests. Each source file makes one executable, and ctest runs each one.

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
build/dev/test/test_api_fftw "[fftw]"    # one binary, Catch2 tag filter
```

The `release` preset has no tests. Use `-DADM_BUILD_TESTS=ON` to add them. `scripts/validate.sh` runs the full matrix of ISAs, compilers and sanitizers.

`test/package/run_routes.sh <src> <work> [route...]` builds the package consumers through every install route.
`test/preload/run.sh <prefix> <work>` runs an unmodified FFTW program on the shim through `LD_PRELOAD`.
`scripts/check_readme.sh [README]` checks that each README code block matches the file it copies.
`test/readme_check/run.sh` runs `check_readme.sh` on fixtures that must pass and fixtures that must fail.
`test/digest` holds the bit-identity digest (see its README).
