#!/usr/bin/env bash
# Build admiral and install it to $PREFIX. Run it from the source root. The docs quote it.
set -eu
# [doc:install-prefix]
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF -DADM_BUILD_EXAMPLES=OFF
cmake --build build --config Release
cmake --install build --prefix "$PREFIX" --config Release
# [doc:install-prefix-end]
