#!/usr/bin/env bash
# Usage: run_routes.sh <admiral-src> <work-dir> [route...]
# Builds and runs the consumer in this directory once per way a downstream project can get
# admiral. No route name means all routes. Prints one line per route and exits nonzero if any
# route failed. Needs a C++20 compiler on PATH, cmake and git. The caller loads the modules.
# The git routes test the committed HEAD of <admiral-src>, not its working tree. They clone
# xsimd, poet and snmalloc from GitHub into <work-dir>/mirrors once.
# Before any route, controls check that each check can fail. A control that does not fire stops
# the run.
set -eu

all_routes="prefix_shared prefix_static c_only cc_fftw relocated subdir fetch_dir fetch_git cpm"
[ $# -ge 2 ] || { echo "usage: $0 <admiral-src> <work-dir> [route...]" >&2; exit 2; }
abspath() { (cd "$1" && { pwd -W 2>/dev/null || pwd -P; }); }
src=$(abspath "$1")
workarg=$2
shift 2
routes=${*:-$all_routes}
for r in $routes; do
    case " $all_routes " in
        *" $r "*) ;;
        *) echo "unknown route '$r'; routes are: $all_routes" >&2; exit 2 ;;
    esac
done
want() { case " $routes " in *" $1 "*) return 0 ;; esac; return 1; }
mkdir -p "$workarg"
work=$(abspath "$workarg")

here=$(abspath "$(dirname "$0")")
sha=$(git -C "$src" rev-parse HEAD)
unset NINJA_STATUS

# -j is the smaller of the visible cores and the free memory at 3 GB for each job.
cpus=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-2}")
jobs=$cpus
if [ -r /proc/meminfo ]; then
    mem_jobs=$(awk -v c="$cpus" '/MemAvailable/ { m = int($2 / 3e6); if (m > c) m = c; print (m < 1 ? 1 : m) }' /proc/meminfo)
    jobs=${mem_jobs:-$cpus}
fi
# The one configuration of the install and of every consumer. A consumer of another configuration
# links on a multi-config generator and fails on MSVC (LNK2038), or mixes standard-library types
# across the DLL boundary. Single-config generators read -DCMAKE_BUILD_TYPE, multi-config
# generators read --config and -C.
cfg=Release
lib_opts=()
case $(uname -m) in
    x86_64 | amd64) lib_opts+=("-DADM_TARGET_ARCH=${ADM_PACKAGE_ARCH:-x86-64-v2}") ;;
esac
gen=()
if [ -z "${CMAKE_GENERATOR:-}" ] && command -v ninja >/dev/null; then gen=(-G Ninja); fi
nice_run() {
    if command -v ionice >/dev/null; then
        nice -n 19 ionice -c 3 "$@"
    elif command -v nice >/dev/null; then
        nice -n 19 "$@"
    else
        "$@"
    fi
}

fail=0
report() { # name status seconds proof
    printf '%-12s %-4s %4ss  %s\n' "$1" "$2" "$3" "$4"
    [ "$2" = pass ] || fail=1
}

# Print the tail of the log of the step that failed in $1. The steps run in the order ctest,
# install, build, configure, and a step that did not run left no log, so the newest step with a
# log is the one that failed. 60 lines reach the first error of a build log on CI.
show_tail() {
    local l
    for l in ctest install build configure; do
        [ -e "$1/$l.log" ] || continue
        echo "---- tail of $1/$l.log"
        tail -n 60 "$1/$l.log"
        return
    done
}

# Did the app bind fftw_plan_dft_1d to the shim? $1 is its LD_DEBUG=bindings log, $2 the app, as
# run. macOS has no such log, so it reads the load commands: that proves the link, not the binding.
shim_bound() {
    if [ "$(uname)" = Darwin ]; then otool -L "$2" | grep -q libadmiral_fftw; return; fi
    grep -Eq "binding file $2 .* to .*/libadmiral_fftw\\.so.*symbol .fftw_plan_dft_1d([^_[:alnum:]]|$)" "$1"
}

# $1 is the C++ switch of the static consumer, ON or OFF. Status 0 means that it linked and ran.
# shellcheck disable=SC2329 # control calls it by name
c_only_static() {
    local dir=$work/c_only_static_$1
    rm -rf "$dir"; mkdir -p "$dir/project"
    cp "$here"/c_only_static/CMakeLists.txt "$here"/use_c.c "$dir/project"
    consume "$dir" -DCMAKE_PREFIX_PATH="$work/prefix" -DADM_CONSUMER_CXX="$1" >/dev/null
}

is_windows() { case $(uname) in MINGW* | MSYS* | CYGWIN*) return 0 ;; esac; return 1; }

show_tail_file() { echo "---- tail of $1"; tail -n 60 "$1"; }

# Environment of one route, empty for the routes that read no GitHub URL.
penv=()

# Configure, build and run the consumer. $1 is the route directory, the other arguments go to
# cmake. Prints the number of tests that ran. The route configure line is left in $1/configure.log.
consume() {
    local dir=$1; shift
    # No set -e inside a condition, so every step is chained.
    nice_run ${penv[@]+"${penv[@]}"} cmake -S "$dir/project" -B "$dir/build" ${gen[@]+"${gen[@]}"} -DCMAKE_BUILD_TYPE=$cfg "$@" \
            >"$dir/configure.log" 2>&1 &&
        nice_run ${penv[@]+"${penv[@]}"} cmake --build "$dir/build" --config $cfg -j "$jobs" \
            >"$dir/build.log" 2>&1 &&
        ctest --test-dir "$dir/build" -C $cfg --timeout 900 --output-on-failure --no-tests=error \
            >"$dir/ctest.log" 2>&1 || return 1
    sed -n 's/.*tests passed, 0 tests failed out of \([0-9]*\).*/\1/p' "$dir/ctest.log"
}

# Field $2 of the route line in the configure log of $1.
field() { sed -n "s/.*adm_package: .*$2=\([^ ]*\).*/\1/p" "$1/configure.log"; }

# Check where the route got admiral from. $1 is the route directory, $2 is the field, $3 is
# the exact value, or a prefix of it when $4 is "prefix".
proof() {
    local got
    got=$(field "$1" "$2")
    if [ "$2" = source_dir ] && [ -d "$got" ]; then got=$(abspath "$got"); fi
    if [ "${4:-}" = prefix ]; then
        case $got in "$3"*) return 0 ;; esac
    elif [ "$got" = "$3" ]; then
        return 0
    fi
    echo "$2 is '$got', expected '$3'" >"$1/proof.err"
    return 1
}

# A git route sets want_sha: the checkout it reports must be at that commit.
want_sha=
sha_ok() {
    [ -z "$want_sha" ] || [ "$(git -C "$(field "$1" source_dir)" rev-parse HEAD)" = "$want_sha" ]
}

# The git routes name https://github.com/DiamonDinoia/admiral.git. Git redirects that URL, and the
# URLs of the three dependencies, to local bare clones. GIT_ALLOW_PROTOCOL=file makes a missed
# redirect a failure instead of a network fetch. Each route starts with an empty CPM cache,
# which keeps only the CPM bootstrap file, so a cached `master` cannot pass.
git_env() { # route directory
    local dir=$1 n=0 name f
    mkdir -p "$work/mirrors"
    rm -rf "$work/mirrors/admiral.git"
    git clone -q --bare "$src" "$work/mirrors/admiral.git"
    git -C "$work/mirrors/admiral.git" update-ref refs/heads/master "$sha"
    for name in xtensor-stack/xsimd DiamonDinoia/poet microsoft/snmalloc; do
        [ -d "$work/mirrors/${name#*/}.git" ] ||
            git clone -q --bare "https://github.com/$name.git" "$work/mirrors/${name#*/}.git"
    done
    mkdir -p "$dir/cpm-cache/cpm"
    for f in "${CPM_SOURCE_CACHE:-/nonexistent}"/cpm/CPM_*.cmake; do
        if [ -f "$f" ]; then cp "$f" "$dir/cpm-cache/cpm"; fi
    done
    penv=(env GIT_ALLOW_PROTOCOL=file CPM_SOURCE_CACHE="$dir/cpm-cache")
    for name in DiamonDinoia/admiral xtensor-stack/xsimd DiamonDinoia/poet microsoft/snmalloc; do
        penv+=("GIT_CONFIG_KEY_$n=url.$work/mirrors/${name#*/}.git.insteadOf"
               "GIT_CONFIG_VALUE_$n=https://github.com/$name.git")
        n=$((n + 1))
    done
    penv+=("GIT_CONFIG_COUNT=$n")
}

# Does the file list of a leak search hold a path? Prints the files. Status 0 is a leak, 1 is
# none, 2 is a grep failure. $1 is the directory, the rest are the paths that must not occur.
leaks() {
    local d=$1 a=() p; shift
    for p in "$@"; do a+=(-e "$p"); done
    grep -rlF "${a[@]}" "$d"
}

# The directory of the installed config, which find_package must report exactly.
config_dir() {
    local c
    for c in "$1"/lib/cmake/admiral "$1"/lib64/cmake/admiral; do
        [ -f "$c/admiralConfig.cmake" ] && { echo "$c"; return; }
    done
    echo "$1/no-admiralConfig.cmake"
}

# The relocation check on $1, a moved prefix of the original prefix $2, of the build tree $3 and
# of the source tree $4. No installed file may hold the original prefix: a RUNPATH or a baked
# string breaks the moved install. The config files must also hold no path of the build tree or
# of the source tree. A library may hold those two, as the __FILE__ of a dependency header does
# when a fetch lands in the build tree. Prints the files. Status 0 is a leak, 1 is none, 2 is a
# grep failure.
relocation_leaks() {
    local rc=0
    leaks "$1" "$2" || rc=$?
    [ "$rc" = 1 ] || return "$rc"
    leaks "$(config_dir "$1")" "$3" "$4"
}

# control <what the control proved> <pass|fail> <command...>: run the command and require that
# it passes or fails as stated. A check that cannot fail is not a check, so a control that does
# not behave stops the run.
control() {
    local what=$1 expect=$2 rc=0; shift 2
    "$@" >/dev/null 2>&1 || rc=$?
    if { [ "$expect" = pass ] && [ "$rc" = 0 ]; } || { [ "$expect" = fail ] && [ "$rc" != 0 ]; }; then
        echo "control ok   $what"
    else
        echo "control FAIL $what (status $rc, wanted $expect)"
        exit 1
    fi
}

run_controls() {
    local c=$work/controls
    rm -rf "$c"; mkdir -p "$c/leak"
    echo "-- adm_package: route=x admiral_dir=/a/b source_dir=$src" >"$c/configure.log"
    control "proof accepts the exact admiral_dir" pass proof "$c" admiral_dir /a/b exact
    control "proof rejects a wrong admiral_dir" fail proof "$c" admiral_dir /a/c exact
    control "proof rejects a longer admiral_dir as exact" fail proof "$c" admiral_dir /a exact
    want_sha=$sha
    control "sha_ok accepts the sha of the tree" pass sha_ok "$c"
    want_sha=0000000000000000000000000000000000000000
    control "sha_ok rejects another sha" fail sha_ok "$c"
    want_sha=
    echo "set(X /planted/prefix/lib)" >"$c/leak/f.cmake"
    control "leak search finds a planted path" pass leaks "$c/leak" /planted/prefix
    control "leak search is silent on an absent path" fail leaks "$c/leak" /absent/prefix
    local m=$c/moved cmakecfg
    mkdir -p "$m/lib/cmake/admiral"; cmakecfg=$m/lib/cmake/admiral/admiralConfig.cmake
    echo "set(X lib)" >"$cmakecfg"; echo "plain" >"$m/lib/libx.so"
    control "relocation check is silent on a clean install" fail relocation_leaks "$m" /p/prefix /p/build /p/src
    echo "/p/prefix/lib" >"$m/lib/libx.so"
    control "relocation check finds the prefix in a library" pass relocation_leaks "$m" /p/prefix /p/build /p/src
    echo "/p/build/_deps/dep/f.h" >"$m/lib/libx.so"
    control "relocation check accepts a build path in a library" fail relocation_leaks "$m" /p/prefix /p/build /p/src
    echo "set(X /p/build/lib)" >"$cmakecfg"
    control "relocation check finds the build tree in a config" pass relocation_leaks "$m" /p/prefix /p/build /p/src
    echo "set(X /p/src/include)" >"$cmakecfg"
    control "relocation check finds the source tree in a config" pass relocation_leaks "$m" /p/prefix /p/build /p/src
    if [ "$(uname)" != Darwin ]; then
        echo "binding file ./app [0] to /p/lib/libfftw3.so.3 [0]: normal symbol \`fftw_plan_dft_1d'" >"$c/bind.log"
        control "binding check rejects a bind to real FFTW" fail shim_bound "$c/bind.log" ./app
        echo "binding file /p/lib/libadmiral_fftw.so [0] to /p/lib/libfftw3.so.3 [0]: normal symbol \`fftw_plan_dft_1d'" >"$c/bind.log"
        control "binding check rejects a bind from the shim" fail shim_bound "$c/bind.log" ./app
        echo "binding file ./app [0] to /p/lib/libadmiral_fftw.so [0]: normal symbol \`fftw_plan_dft_1d'" >"$c/bind.log"
        control "binding check accepts a bind to the shim" pass shim_bound "$c/bind.log" ./app
        echo "binding file ./app [0] to /p/lib/libadmiral_fftw.so [0]: normal symbol \`fftw_plan_dft_1d_x'" >"$c/bind.log"
        control "binding check rejects another symbol bound to the shim" fail shim_bound "$c/bind.log" ./app
    fi
    if want fetch_git || want cpm; then
        mkdir -p "$c/git"; git_env "$c/git"
        control "the redirect serves master = $sha" pass \
            test "$("${penv[@]}" git ls-remote https://github.com/DiamonDinoia/admiral.git refs/heads/master | cut -f1)" = "$sha"
        control "a missed redirect is blocked, not fetched" fail \
            "${penv[@]}" GIT_CONFIG_COUNT=0 git ls-remote https://github.com/DiamonDinoia/admiral.git refs/heads/master
    fi
}

# run_route <name> <expected tests> <field> <exact value | prefix> <exact|prefix> <cmake args...>
run_route() {
    local name=$1 count=$2 fld=$3 val=$4 mode=$5 dir=$work/$1 start=$SECONDS n=""; shift 5
    local msg="$fld is $val"
    [ "$mode" = prefix ] && msg="$fld is under $val"
    rm -rf "$dir/project" "$dir/build"; mkdir -p "$dir/project"; rm -f "$dir"/*.log
    cp "${cdir:-$here}"/CMakeLists.txt "$here"/*.c "$here"/*.cpp "$here"/*.cmake "$dir/project"
    [ "$name" != subdir ] || ln -s "$src" "$dir/project/admiral"
    if ! n=$(consume "$dir" "$@"); then
        report "$name" FAIL $((SECONDS - start)) "see $dir/*.log"
        show_tail "$dir"
    elif [ "$n" != "$count" ]; then
        report "$name" FAIL $((SECONDS - start)) "ran '$n' tests, expected $count"
    elif ! proof "$dir" "$fld" "$val" "$mode"; then
        report "$name" FAIL $((SECONDS - start)) "$(cat "$dir/proof.err")"
    elif ! sha_ok "$dir"; then
        report "$name" FAIL $((SECONDS - start)) "checkout is not $want_sha"
    else
        report "$name" pass $((SECONDS - start)) \
            "$count tests ran, $msg${want_sha:+, checkout is $want_sha}"
    fi
}

if want fetch_git || want cpm; then
    echo "git routes test commit $sha of $src, not its working tree"
fi
run_controls

# Install once into $work/prefix, for the routes that read an installed prefix. The build tree
# goes at once, so a config that points into it fails.
if want prefix_shared || want prefix_static || want c_only || want cc_fftw || want relocated; then
    inst=$work/install
    rm -rf "$inst" "$work/prefix" "$work/moved"; mkdir -p "$inst"
    start=$SECONDS
    if ! { nice_run cmake -S "$src" -B "$inst/build" ${gen[@]+"${gen[@]}"} -DCMAKE_BUILD_TYPE=$cfg ${lib_opts[@]+"${lib_opts[@]}"} \
                -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF -DADM_BUILD_EXAMPLES=OFF \
                -DCMAKE_INSTALL_LIBDIR=lib \
                -DCMAKE_INSTALL_PREFIX="$work/prefix" >"$inst/configure.log" 2>&1 &&
            nice_run cmake --build "$inst/build" --config $cfg -j "$jobs" >"$inst/build.log" 2>&1 &&
            cmake --install "$inst/build" --config $cfg >"$inst/install.log" 2>&1; }; then
        report install FAIL $((SECONDS - start)) "see $inst/*.log"
        show_tail "$inst"
        exit 1
    fi
    report install pass $((SECONDS - start)) "configure, build and install"
    rm -rf "$inst/build"
fi

prefix_route() { # name prefix tests consumer-route
    run_route "$1" "$3" admiral_dir "$(config_dir "$2")" exact -DADM_PACKAGE_ROUTE="$4" \
        -DCMAKE_PREFIX_PATH="$2"
}

if want prefix_shared; then
    prefix_route prefix_shared "$work/prefix" 3 prefix_shared
fi
if want prefix_static; then
    prefix_route prefix_static "$work/prefix" 3 prefix_static
fi
if want c_only; then
    # A C-only project links the shared C targets. It needs no C++ language of its own.
    cdir=$here/c_only
    run_route c_only 2 admiral_dir "$(config_dir "$work/prefix")" exact -DCMAKE_PREFIX_PATH="$work/prefix"
    cdir=
    # The README says a C-only project needs C++ to link a static twin. That holds where the C
    # driver links no C++ runtime. MSVC links it anyway: each object of the archive names
    # msvcprt in a /DEFAULTLIB directive, and link.exe obeys it.
    if is_windows; then
        control "MSVC links a static C target into a C-only project" pass c_only_static OFF
        control "the same project links it once C++ is enabled" pass c_only_static ON
    else
        control "a C-only project cannot link a static C target" fail c_only_static OFF
        control "the same project links it once C++ is enabled" pass c_only_static ON
    fi
fi
if want cc_fftw; then
    # Plain compiler, no CMake: the way most FFTW programs build.
    dir=$work/cc_fftw start=$SECONDS
    rm -rf "$dir"; mkdir -p "$dir"
    p=$work/prefix
    if is_windows; then
        echo "SKIP cc_fftw: it needs a plain cc with -L and an rpath"
    elif ! "${CC:-cc}" -o "$dir/app" "$here/use_fftw.c" -I"$p/include" -L"$p/lib" -ladmiral_fftw \
            -Wl,-rpath,"$p/lib" -lm >"$dir/build.log" 2>&1; then
        report cc_fftw FAIL $((SECONDS - start)) "see $dir/build.log"
        show_tail_file "$dir/build.log"
    elif ! LD_DEBUG=bindings "$dir/app" >"$dir/run.log" 2>"$dir/bind.log"; then
        report cc_fftw FAIL $((SECONDS - start)) "app failed, see $dir/run.log and $dir/bind.log"
    elif ! shim_bound "$dir/bind.log" "$dir/app"; then
        report cc_fftw FAIL $((SECONDS - start)) "app did not bind fftw_plan_dft_1d to the shim"
    else
        what="bound fftw_plan_dft_1d to libadmiral_fftw"
        [ "$(uname)" != Darwin ] || what="links libadmiral_fftw"
        report cc_fftw pass $((SECONDS - start)) "app ran and $what"
    fi
fi
if want relocated; then
    # Relocation: move the prefix, remove the original, and read the copy.
    mv "$work/prefix" "$work/moved"
    leak=0
    found=$(relocation_leaks "$work/moved" "$work/prefix" "$inst" "$src") || leak=$?
    if [ "$leak" = 0 ]; then
        report relocated FAIL 0 "path of the install found in: ${found//$'\n'/ }"
    elif [ "$leak" != 1 ]; then
        report relocated FAIL 0 "grep failed with status $leak"
    else
        prefix_route relocated "$work/moved" 6 prefix
    fi
fi

src_opts=(${lib_opts[@]+"${lib_opts[@]}"} "-DADM_PACKAGE_SRC=$src")
want subdir && run_route subdir 6 source_dir "$src" exact -DADM_PACKAGE_ROUTE=subdir "${src_opts[@]}"
want fetch_dir && run_route fetch_dir 6 source_dir "$src" exact -DADM_PACKAGE_ROUTE=fetch_dir \
    -DFETCHCONTENT_SOURCE_DIR_ADMIRAL="$src" "${src_opts[@]}"

want_sha=$sha
if want fetch_git; then
    rm -rf "$work/fetch_git"; mkdir -p "$work/fetch_git"; git_env "$work/fetch_git"
    run_route fetch_git 6 source_dir "$work/fetch_git/build/_deps" prefix \
        -DADM_PACKAGE_ROUTE=fetch_git "${src_opts[@]}"
fi
if want cpm; then
    rm -rf "$work/cpm"; mkdir -p "$work/cpm"; git_env "$work/cpm"
    run_route cpm 6 source_dir "$work/cpm/cpm-cache/admiral" prefix -DADM_PACKAGE_ROUTE=cpm "${src_opts[@]}"
fi

exit "$fail"
