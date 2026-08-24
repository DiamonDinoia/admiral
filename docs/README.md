# Admiral docs

Rendered site: <https://diamondinoia.github.io/yafft/> (built by
[.github/workflows/docs.yml](../.github/workflows/docs.yml); [install.md](install.md#build-these-docs-locally)
says how to build it locally).

- [install.md](install.md): requirements, build and install, consuming the
  package from CMake, building these docs.
- [cpp-api.md](cpp-api.md): the C++ API — complex plans, real transforms, DCT/DST,
  per-axis transforms, one-shots. Every snippet has a runnable counterpart in
  [../examples/](../examples/).
- [usage.md](usage.md): options (`nthreads`, `eff`, `debug`), the C API, and the
  FFTW shim's coverage.
- [build-options.md](build-options.md): every CMake option, the benchmark
  environment variables, and the runtime fields.
- [../benchmark/README.md](../benchmark/README.md): how the headline numbers are
  measured and how to rerun them.

The site adds an API reference generated from the public headers by
Doxygen + Sphinx (`conf.py`, `index.rst`); `requirements.txt` pins the Python
toolchain.
