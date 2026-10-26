# Use admiral as FFTW

The library `admiral_fftw` exports the complex-to-complex API of FFTW3.
An FFTW program uses it at build time or at run time.

## Link at build time

Include `<admiral/fftw3.h>` instead of `<fftw3.h>`.
Link `admiral::fftw` (shared) or `admiral::fftw_static` (static).
`find_package(admiral)` provides both. See [Install and use](install.md).
`examples/fftw_dropin.c` is a complete program.

## Preload at run time

On Linux, `LD_PRELOAD` puts the shim in front of FFTW in a program that links FFTW already.
The program needs no rebuild.
The command reads the library from `$PREFIX/lib`.
Install with `-DCMAKE_INSTALL_LIBDIR=lib` so that the library lands there.
Some distributions default to `lib64`.

```{literalinclude} ../test/preload/run.sh
:start-after: [doc:ld-preload]
:end-before: [doc:ld-preload-end]
```

CI runs this line on a program that calls 24 FFTW functions.
It checks that each call binds to the shim and that the output matches real FFTW.

## What the shim covers

The shared library exports every symbol that matches these patterns.

```{literalinclude} ../src/admiral_fftw.map
:start-at: global:
:dedent:
:end-before: local:
```

The header declares the symbols.
It covers the complex-to-complex subset only.

```{literalinclude} ../include/admiral/fftw3.h
:start-at: // Complex-to-complex DFT subset
:end-before: typedef double fftw_complex
:language: c
```

The header shows the `fftw_` calls. The same calls exist with the prefix `fftwf_` for `float`.
The comments give the limits of the shim.

```{literalinclude} ../include/admiral/fftw3.h
:start-at: // Only FFTW_ESTIMATE is read
:end-before: FFTW_C_API fftwf_plan fftwf_plan_dft_1d
:language: c
```
