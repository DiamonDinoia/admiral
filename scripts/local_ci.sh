#!/usr/bin/env bash
# Run CI cells locally in a podman/docker container.
#
#   scripts/local_ci.sh <cell>...
#
# Arch matrix cells (.github/workflows/arch-matrix.yml on port/integrate): x86_64
# and aarch64 run natively in CI; here they join the qemu route. Each qemu cell
# cross compiles on x86-64 and runs ctest through qemu-user, the same route the
# workflow uses.
#
#   x86_64 aarch64 ppc64le ppc64 armhf riscv64 s390x
#
# ci.yml Linux build cells (.github/workflows/ci.yml on master), x86-64-v3:
#
#   linux-gcc-14-release linux-gcc-14-debug linux-llvm-21-release linux-llvm-21-debug
#
# The container image carries every toolchain; podman rebuilds it only when the
# apt lines change. The source mounts read-only, so ppc64le-control cannot run:
# that cell patches include/admiral/detail/ct_math.hpp. Windows, macOS and wasm
# cells have no Linux container route here.
#
# Env overrides:
#   ADM_CI_SRC     checkout to test (default: the repo this script lives in)
#   ADM_CI_OUT     build and log root (default /home/mbarbone/localcache/scratch/admiral-ci)
#   ADM_CI_CCACHE  ccache dir shared by all cells (default $ADM_CI_OUT/ccache)
#   ADM_CI_JOBS    build parallelism (default 3)
#   ADM_CI_CTEST_JOBS  ctest parallelism (default 2, as in CI)
#   ADM_CI_ENGINE  container engine (default podman, then docker)
#   ADM_CI_TMP     podman/buildah scratch dir (default $ADM_CI_OUT/tmp; buildah
#                  cannot use a GPFS TMPDIR)

set -uo pipefail
unset NINJA_STATUS

src=${ADM_CI_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
jobs=${ADM_CI_JOBS:-3}
ctest_jobs=${ADM_CI_CTEST_JOBS:-2}
out=${ADM_CI_OUT:-/home/mbarbone/localcache/scratch/admiral-ci}
ccache=${ADM_CI_CCACHE:-$out/ccache}
image=localhost/admiral-ci:latest
engine=${ADM_CI_ENGINE:-$(command -v podman || command -v docker || true)}

usage() { sed -n '2,30p' "$0"; exit "${1:-1}"; }

# buildah stages its work in TMPDIR, and this node's TMPDIR is on GPFS, where a
# rootless build user cannot create directories. A local disk scratch dir fixes it.
export TMPDIR=${ADM_CI_TMP:-$out/tmp}

# Resolve a cell name into the global variables the inner script reads.
resolve_cell() {
    K='' TRIPLE='' PROC='' QEMU='' ARCH='' EXPECT='' XARCH='' CC='' CXX='' CFG=''
    case $1 in
        x86_64|aarch64|ppc64le|ppc64|armhf|riscv64|s390x)
            K=qemu CFG=Release
            case $1 in
                x86_64)   K=native CXX=g++-14 ARCH=x86-64-v2 EXPECT='-march=x86-64-v2' XARCH=sse4.2 ;;
                aarch64)  TRIPLE=aarch64-linux-gnu        PROC=aarch64 QEMU=qemu-aarch64 ARCH=armv8-a EXPECT='-march=armv8-a' XARCH=arm64+neon ;;
                ppc64le)  TRIPLE=powerpc64le-linux-gnu    PROC=ppc64le  QEMU=qemu-ppc64le ARCH=power8   EXPECT='-mcpu=power8' XARCH=vmx+vsx ;;
                ppc64)    TRIPLE=powerpc64-linux-gnu      PROC=ppc64    QEMU=qemu-ppc64   ARCH=power8   EXPECT='-mcpu=power8' ;;
                armhf)    TRIPLE=arm-linux-gnueabihf      PROC=arm      QEMU=qemu-arm     ARCH=none     XARCH=emulated ;;
                riscv64)  TRIPLE=riscv64-linux-gnu        PROC=riscv64  QEMU=qemu-riscv64 ARCH=none     XARCH=emulated ;;
                s390x)    TRIPLE=s390x-linux-gnu          PROC=s390x    QEMU=qemu-s390x   ARCH=z14      EXPECT='-march=z14 -mzvector' XARCH=vxe ;;
            esac ;;
        linux-gcc-14-release)   K=linux CC=gcc-14 CXX=g++-14     CFG=Release ARCH=x86-64-v3 ;;
        linux-gcc-14-debug)     K=linux CC=gcc-14 CXX=g++-14     CFG=Debug   ARCH=x86-64-v3 ;;
        linux-llvm-21-release)  K=linux CC=clang-21 CXX=clang++-21 CFG=Release ARCH=x86-64-v3 ;;
        linux-llvm-21-debug)    K=linux CC=clang-21 CXX=clang++-21 CFG=Debug   ARCH=x86-64-v3 ;;
        *) echo "unknown cell: $1"; usage ;;
    esac
}

build_image() {
    local ctx=$out/.ctx
    mkdir -p "$ctx"
    cat > "$ctx/Containerfile" <<'EOF'
FROM docker.io/library/ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget gnupg ninja-build ccache cmake git \
    g++-14 \
    g++-14-aarch64-linux-gnu \
    g++-14-powerpc64le-linux-gnu \
    g++-14-powerpc64-linux-gnu \
    g++-14-arm-linux-gnueabihf \
    g++-14-riscv64-linux-gnu \
    g++-14-s390x-linux-gnu \
    qemu-user \
 && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main" > /etc/apt/sources.list.d/llvm.list \
 && apt-get update && apt-get install -y --no-install-recommends clang-21 \
 && rm -rf /var/lib/apt/lists/*
EOF
    "$engine" build -t "$image" -f "$ctx/Containerfile" "$ctx" > "$out/image-build.log" 2>&1 ||
        { echo "image build failed (see $out/image-build.log)"; tail -n 10 "$out/image-build.log"; exit 1; }
}

# The script that runs inside the container. One constant body; the cell reaches
# it through the environment. Layout: /src = checkout (read-only), /b = $dir,
# /ccache = shared cache, logs land in /b/logs.
write_inner() {
    cat > "$1" <<'EOF'
set -uo pipefail
unset NINJA_STATUS

b=/b/build
mkdir -p /b/logs
rm -rf "$b" /b/build-pkg /b/prefix
export CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_C_COMPILER_LAUNCHER=ccache \
       CCACHE_DIR=/ccache CCACHE_MAXSIZE=16G

args=(-S /src -B "$b" -G Ninja -DCMAKE_BUILD_TYPE="$CFG"
      -DADM_TARGET_ARCH="$ARCH" -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
case $K in
    qemu)
        T=$TRIPLE
        {   echo 'set(CMAKE_SYSTEM_NAME Linux)'
            echo "set(CMAKE_SYSTEM_PROCESSOR $PROC)"
            echo "set(CMAKE_C_COMPILER $T-gcc-14 CACHE FILEPATH \"\")"
            echo "set(CMAKE_CXX_COMPILER $T-g++-14 CACHE FILEPATH \"\")"
            echo "set(CMAKE_CROSSCOMPILING_EMULATOR \"$QEMU;-L;/usr/$T\")"
            echo "set(CMAKE_FIND_ROOT_PATH /usr/$T)"
            echo 'set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)'
            echo 'set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)'
        } > /b/tc.cmake
        args+=(-DCMAKE_TOOLCHAIN_FILE=/b/tc.cmake
               -DCMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST) ;;
    native)
        args+=(-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14) ;;
    linux)
        args+=(-DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DADM_CXX_STANDARD=20) ;;
esac
if ! cmake "${args[@]}" > /b/logs/configure.log 2>&1; then
    echo "configure failed:"; tail -n 40 /b/logs/configure.log; exit 1
fi

if [ "$K" = qemu ] || [ "$K" = native ]; then
    for f in $EXPECT; do
        grep -qF -- "$f" "$b/compile_commands.json" || { echo "FLAG MISSING: $f"; exit 1; }
    done
    if grep -qF -- '=native' "$b/compile_commands.json"; then echo "FLAG PRESENT: =native"; exit 1; fi
    if ! grep -qF -- '-O3' "$b/compile_commands.json" && ! grep -qF -- '-O2' "$b/compile_commands.json"; then
        echo "FLAG MISSING: -O3|-O2"; exit 1
    fi
else
    checks=(want:=x86-64-v3 not:=native want:-std=c++20)
    [ "$CFG" = Debug ] && checks+=(want:-Og not:-O0)
    /src/scripts/check_flags.sh "$b" "${checks[@]}" || exit 1
fi
grep -q 'LAUNCHER = ccache' "$b/build.ninja" || { echo "ccache launcher missing in build.ninja"; exit 1; }

rc=0
cmake --build "$b" -j "$ADM_CI_JOBS" -- -k 0 > /b/logs/build.log 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "failed TUs: $(grep -c '^FAILED' /b/logs/build.log)"
    grep -n -m5 -E 'error:|FAILED:' /b/logs/build.log || true
    tail -n 20 /b/logs/build.log
    exit "$rc"
fi

if [ -n "$XARCH" ]; then
    cxxbin=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$b/CMakeCache.txt")
    xs=$(grep -o -m1 -- '-isystem [^ ]*xsimd[^ ]*/include' "$b/compile_commands.json" | head -n1 | cut -d' ' -f2)
    printf '#include <admiral/detail/simd.hpp>\n#include <cstdio>\nint main() { std::puts(xsimd::default_arch::name()); }\n' > /b/arch.cpp
    "$cxxbin" -std=c++20 $EXPECT -I/src/include -isystem "$xs" /b/arch.cpp -o /b/arch
    if [ "$K" = qemu ]; then got=$("$QEMU" -L "/usr/$TRIPLE" /b/arch); else got=$(/b/arch); fi
    echo "xsimd arch '$got' at this cell's flags; expected '$XARCH'"
    [ "$got" = "$XARCH" ] || exit 1
fi

timeout_s=900
[ "$K" = qemu ] && timeout_s=7200
rc=0
if [ "$K" = qemu ]; then
    # Each ctest case is one qemu process. The long cases run in their own -R call, at the same
    # time as the rest, so they do not run last.
    long='plane-fused|ULP: large'
    ctest --test-dir "$b" -j "$ADM_CI_CTEST_JOBS" --timeout "$timeout_s" --output-on-failure -R "$long" \
        > /b/logs/ctest-long.log 2>&1 &
    p1=$!
    ctest --test-dir "$b" -j "$ADM_CI_CTEST_JOBS" --timeout "$timeout_s" --output-on-failure --no-tests=error -E "$long" \
        > /b/logs/ctest.log 2>&1 || rc=$?
    wait "$p1" || rc=$?
    echo "long cases:"; grep -E 'tests passed|tests failed|No tests' /b/logs/ctest-long.log || true
else
    ctest --test-dir "$b" -j "$ADM_CI_CTEST_JOBS" --timeout "$timeout_s" --output-on-failure --no-tests=error \
        > /b/logs/ctest.log 2>&1 || rc=$?
fi
grep -E 'tests passed|tests failed|Total Test time' /b/logs/ctest.log || true
[ "$rc" -eq 0 ] || { grep -E '\*\*\*|Failed|Timeout|Not Run' /b/logs/ctest*.log | head -n 30 || true; exit "$rc"; }

if [ "$K" = linux ]; then
    cmake --install "$b" --prefix /b/prefix >> /b/logs/install.log 2>&1 || { echo "install failed"; exit 1; }
    cmake -S /src/test/package -B /b/build-pkg -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_PREFIX_PATH=/b/prefix > /b/logs/configure-pkg.log 2>&1 || { echo "configure-pkg failed"; exit 1; }
    cmake --build /b/build-pkg > /b/logs/build-pkg.log 2>&1 || { echo "build-pkg failed"; exit 1; }
    ctest --test-dir /b/build-pkg -j "$ADM_CI_CTEST_JOBS" --timeout 900 --output-on-failure --no-tests=error \
        > /b/logs/ctest-pkg.log 2>&1 || { echo "package ctest failed:"; tail -n 30 /b/logs/ctest-pkg.log; exit 1; }
fi
EOF
}

[ -n "$engine" ] || { echo "neither podman nor docker on PATH"; exit 1; }
[ $# -ge 1 ] || usage
mkdir -p "$out" "$ccache" "$TMPDIR"
build_image

rc=0
for cell in "$@"; do
    resolve_cell "$cell" || usage 1
    dir=$out/$cell
    mkdir -p "$dir"
    inner=$dir/inner.sh
    write_inner "$inner"
    echo "=== $cell (src=$src)"
    SECONDS=0
    "$engine" run --rm \
        -v "$src:/src:ro" -v "$dir:/b" -v "$ccache:/ccache" -w /b \
        -e ADM_CI_JOBS="$jobs" -e ADM_CI_CTEST_JOBS="$ctest_jobs" \
        -e "K=$K" -e "TRIPLE=$TRIPLE" -e "PROC=$PROC" -e "QEMU=$QEMU" -e "ARCH=$ARCH" \
        -e "EXPECT=$EXPECT" -e "XARCH=$XARCH" -e "CC=$CC" -e "CXX=$CXX" -e "CFG=$CFG" \
        "$image" bash /b/inner.sh
    rc=$?
    echo "=== $cell: exit=$rc wall=$((SECONDS / 60))m$((SECONDS % 60))s"
done
exit "$rc"
