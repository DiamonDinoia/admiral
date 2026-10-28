#!/usr/bin/env bash
# LoongArch dispatch proof, run in localhost/admiral-isa-tiers-llvm (qemu 8.2.2 user mode;
# that image is admiral-isa-tiers plus llvm-21 for llvm-nm and llvm-objcopy).
#   scripts/loongarch_leg.sh <build|lsx|lasx> [SRC] [OUT]
# build: ADM_DISPATCH=ON, floor la64v1.0 (LSX), tier LASX.
# lsx, lasx: run the build under qemu-loongarch64 with LASX off or on. test_dispatch prints the
#   tier picked and exits 1 when the expected tier differs; a wrong expect must exit nonzero.
# Env: ADM_CI_JOBS, ADM_CI_CTEST_JOBS (default 32).
set -uo pipefail
unset NINJA_STATUS
mode=${1:?build|lsx|lasx}
SRC=${2:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${3:-/home/mbarbone/localcache/scratch/admiral-ci/loongarch}
IMG=localhost/admiral-isa-tiers-llvm:latest
export TMPDIR=${ADM_CI_TMP:-/home/mbarbone/localcache/scratch/admiral-ci/tmp}
mkdir -p "$OUT" "$TMPDIR" "$OUT/ccache"
cat > "$OUT/inner-$mode.sh" <<'IN'
set -uo pipefail
unset NINJA_STATUS
T=loongarch64-linux-gnu
b=/b/build
mkdir -p /b/logs
export PATH=/usr/lib/llvm-21/bin:$PATH CCACHE_DIR=/ccache
Q=/opt/qemu-user/qemu-loongarch64
case $MODE in
build)
    {   echo 'set(CMAKE_SYSTEM_NAME Linux)'
        echo 'set(CMAKE_SYSTEM_PROCESSOR loongarch64)'
        echo "set(CMAKE_C_COMPILER $T-gcc-14)"
        echo "set(CMAKE_CXX_COMPILER $T-g++-14)"
        echo "set(CMAKE_CROSSCOMPILING_EMULATOR \"$Q;-L;/usr/$T\")"
        echo "set(CMAKE_FIND_ROOT_PATH /usr/$T)"
        echo 'set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)'
        echo 'set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)'
    } > /b/tc.cmake
    rm -rf $b
    cmake -S /src -B $b -G Ninja -DCMAKE_BUILD_TYPE=Release -DADM_TARGET_ARCH=la64v1.0 -DADM_DISPATCH=ON \
        -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_TOOLCHAIN_FILE=/b/tc.cmake -DCMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache > /b/logs/configure.log 2>&1 \
        || { tail -30 /b/logs/configure.log; echo "build: exit=1"; exit 1; }
    for f in -march=la64v1.0 -mlasx -Wno-error=maybe-uninitialized; do
        grep -qF -- "$f" $b/compile_commands.json || { echo "FLAG MISSING: $f"; echo "build: exit=1"; exit 1; }
    done
    cmake --build $b -j "$ADM_CI_JOBS" -- -k 0 > /b/logs/build.log 2>&1
    rc=$?
    [ $rc -eq 0 ] || grep -m5 -E 'error:|FAILED:' /b/logs/build.log
    echo "build: exit=$rc"
    exit $rc ;;
lsx|lasx)
    if [ "$MODE" = lsx ]; then
        cpu='max,lasx=off,lsx=on'; want='loongarch64+lsx'; wrong='loongarch64+lasx'
    else
        cpu='max'; want='loongarch64+lasx'; wrong='loongarch64+lsx'
    fi
    export QEMU_CPU=$cpu
    d=$b/test/test_dispatch
    echo "leg $MODE: qemu -cpu $cpu"
    ADM_EXPECT_TIER=$want $Q -L /usr/$T $d
    r1=$?
    echo "leg $MODE: expect '$want' exit=$r1 (want 0)"
    ADM_EXPECT_TIER=$wrong $Q -L /usr/$T $d
    r2=$?
    echo "leg $MODE: wrong expect '$wrong' exit=$r2 (want nonzero)"
    rc=0
    { [ $r1 -eq 0 ] && [ $r2 -ne 0 ]; } || rc=1
    long='plane-fused|ULP: large'
    ctest --test-dir $b -j "$ADM_CI_CTEST_JOBS" --timeout 7200 --output-on-failure -R "$long" > /b/logs/ctest-$MODE-long.log 2>&1 &
    p1=$!
    ctest --test-dir $b -j "$ADM_CI_CTEST_JOBS" --timeout 7200 --output-on-failure --no-tests=error -E "$long" > /b/logs/ctest-$MODE.log 2>&1
    r3=$?
    wait $p1
    r4=$?
    grep -E 'tests passed|tests failed' /b/logs/ctest-$MODE.log /b/logs/ctest-$MODE-long.log
    echo "leg $MODE: ctest exit=$r3 long exit=$r4 pick exit=$rc"
    [ $rc -eq 0 ] && [ $r3 -eq 0 ] && [ $r4 -eq 0 ]
    exit $? ;;
esac
IN
podman run --rm -v "$SRC:/src:ro" -v "$OUT:/b" -v "$OUT/ccache:/ccache" -w /b \
    -e MODE="$mode" -e ADM_CI_JOBS="${ADM_CI_JOBS:-32}" -e ADM_CI_CTEST_JOBS="${ADM_CI_CTEST_JOBS:-32}" \
    "$IMG" bash /b/inner-$mode.sh
rc=$?
echo "loongarch $mode: exit=$rc"
exit $rc
