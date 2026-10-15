# Admiral docs
The rendered site is at <https://diamondinoia.com/admiral/>. Its API reference is generated from the public
headers. [install.md](install.md#build-these-docs-locally) shows how to build it locally.

- [install.md](install.md): requirements, build and install, consumption from CMake, the local docs build.
- [cpp-api.md](cpp-api.md): the C++ API: plans, real transforms, DCT/DST, per-axis and strided transforms,
  one-shots. Every snippet has a runnable counterpart in [../examples/](../examples/).
- [usage.md](usage.md): the runtime options, the C API, and the FFTW shim's coverage.
- [build-options.md](build-options.md): every CMake option and runtime field.
- [../benchmark/README.md](../benchmark/README.md): the measurement method and rerun commands.
