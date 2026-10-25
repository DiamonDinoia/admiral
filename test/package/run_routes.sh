#!/usr/bin/env bash
# Usage: run_routes.sh <admiral-src> <work-dir> [route...]
# Builds and runs the consumer in this directory once per way a downstream project can get
# admiral. No route name means all routes. Needs a C++20 compiler on PATH, cmake and git. The
# caller loads the modules. CMAKE_BUILD_PARALLEL_LEVEL sets the number of parallel compiles.
# The git routes test the committed HEAD of <admiral-src>, not its working tree. They clone
# xsimd, poet and snmalloc from GitHub into <work-dir>/mirrors once.
set -eu
# shellcheck source=test/check.sh
. "$(dirname "$0")/../check.sh"

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

# Environment of one route, empty for the routes that read no GitHub URL.
penv=()

# Configure, build and run the consumer. $1 is the route directory, the other arguments go to
# cmake. Prints the number of tests that ran. The route configure line is left in $1/configure.log.
consume() {
    local dir=$1; shift
    # No set -e inside a condition, so every step is chained.
    nice_run ${penv[@]+"${penv[@]}"} cmake -S "$dir/project" -B "$dir/build" ${gen[@]+"${gen[@]}"} -DCMAKE_BUILD_TYPE=$cfg "$@" \
            >"$dir/configure.log" 2>&1 &&
        nice_run ${penv[@]+"${penv[@]}"} cmake --build "$dir/build" --config $cfg \
            >"$dir/build.log" 2>&1 &&
        ctest --test-dir "$dir/build" -C $cfg --timeout 900 --output-on-failure --no-tests=error \
            >"$dir/ctest.log" 2>&1 || return 1
    sed -n 's/.*tests passed, 0 tests failed out of \([0-9]*\).*/\1/p' "$dir/ctest.log"
}

# Field $2 of the route line in the configure log of $1.
field() { sed -n "s/.*adm_package: .*$2=\([^ ]*\).*/\1/p" "$1/configure.log"; }

# proof <dir> <field> <value> <exact|prefix>: the field is the value, or starts with it in prefix
# mode. Prints the field.
proof() {
    local got
    got=$(field "$1" "$2")
    if [ "$2" = source_dir ] && [ -d "$got" ]; then got=$(abspath "$got"); fi
    case $4 in prefix) [[ $got == "$3"* ]] ;; *) [ "$got" = "$3" ] ;; esac &&
        { echo "$2 is $got"; return; }
    echo "$2 is '$got', expected '$3' ($4)"
    return 1
}

# sha_ok <dir> <sha>: with a sha, the checkout that the route reports is at that commit.
sha_ok() { [ -z "$2" ] || [ "$(git -C "$(field "$1" source_dir)" rev-parse HEAD)" = "$2" ]; }

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

# blocked <command...>: the command fails and stderr holds the git transport error. An offline host
# fails with another text, so this does not pass for it.
blocked() {
    local err
    ! err=$("$@" 2>&1 >/dev/null) && [[ $err == *"transport 'https' not allowed"* ]]
}

# clean <dir> <path...>: no file in the directory holds any of the paths. Prints the files
# that do.
clean() {
    local d=$1 a=() p rc=0; shift
    for p in "$@"; do a+=(-e "$p"); done
    grep -rlF "${a[@]}" "$d" || rc=$?
    [ "$rc" -eq 1 ]
}

# The directory of the installed config, which find_package must report exactly.
config_dir() {
    local c
    for c in "$1"/lib/cmake/admiral "$1"/lib64/cmake/admiral; do
        [ -f "$c/admiralConfig.cmake" ] && { echo "$c"; return; }
    done
    echo "$1/no-admiralConfig.cmake"
}

# relocatable <moved prefix> <original prefix> <build tree> <source tree>: no installed file holds
# the original prefix, because a RUNPATH or a baked string breaks the moved install. The config
# files also hold no path of the build tree or of the source tree. A library may hold those two,
# as the __FILE__ of a dependency header does when a fetch lands in the build tree.
relocatable() { clean "$1" "$2" && clean "$(config_dir "$1")" "$3" "$4"; }

# route <name> <tests> <field> <value> <exact|prefix> <cmake args...>: the consumer compiles and
# runs <tests> tests, and the route got admiral from where proof says.
want_sha=
route() {
    local name=$1 count=$2 fld=$3 val=$4 mode=$5 dir=$work/$1 n p; shift 5
    rm -rf "$dir/project" "$dir/build"; mkdir -p "$dir/project"; rm -f "$dir"/*.log
    cp "$here"/CMakeLists.txt "$here"/*.c "$here"/*.cpp "$dir/project"
    [ "$name" != subdir ] || ln -s "$src" "$dir/project/admiral"
    n=$(consume "$dir" "$@") || { show_tail "$dir"; return 1; }
    [ "$n" = "$count" ] || { echo "ran '$n' tests, expected $count"; return 1; }
    p=$(proof "$dir" "$fld" "$val" "$mode") || { echo "$p"; return 1; }
    sha_ok "$dir" "$want_sha" || { echo "checkout is not $want_sha"; return 1; }
    echo "$count tests ran, $p${want_sha:+, checkout is $want_sha}"
}

if want fetch_git || want cpm; then
    echo "git routes test commit $sha of $src, not its working tree"
fi

c=$work/controls
rm -rf "$c"; mkdir -p "$c"
echo "-- adm_package: route=x admiral_dir=/a/b source_dir=$src" >"$c/configure.log"
check "control: proof accepts the exact admiral_dir" proof "$c" admiral_dir /a/b exact
check "control: proof rejects a wrong admiral_dir" not proof "$c" admiral_dir /a/c exact
check "control: proof rejects a longer admiral_dir as exact" not proof "$c" admiral_dir /a exact
check "control: sha_ok accepts the sha of the tree" sha_ok "$c" "$sha"
check "control: sha_ok rejects another sha" \
    not sha_ok "$c" 0000000000000000000000000000000000000000
m=$c/moved
mkdir -p "$m/lib/cmake/admiral"
echo "set(X lib)" >"$m/lib/cmake/admiral/admiralConfig.cmake"
echo "plain" >"$m/lib/libx.so"
check "control: relocation check passes a clean install" relocatable "$m" /p/prefix /p/build /p/src
echo "/p/prefix/lib" >"$m/lib/libx.so"
check "control: relocation check finds the prefix in a library" \
    not relocatable "$m" /p/prefix /p/build /p/src
echo "/p/build/_deps/dep/f.h" >"$m/lib/libx.so"
check "control: relocation check accepts a build path in a library" \
    relocatable "$m" /p/prefix /p/build /p/src
echo "set(X /p/build/lib)" >"$m/lib/cmake/admiral/admiralConfig.cmake"
check "control: relocation check finds the build tree in a config" \
    not relocatable "$m" /p/prefix /p/build /p/src
echo "set(X /p/src/include)" >"$m/lib/cmake/admiral/admiralConfig.cmake"
check "control: relocation check finds the source tree in a config" \
    not relocatable "$m" /p/prefix /p/build /p/src
if want fetch_git || want cpm; then
    mkdir -p "$c/git"
    git_env "$c/git"
    check "control: the redirect serves master = $sha" test "$("${penv[@]}" git ls-remote \
        https://github.com/DiamonDinoia/admiral.git refs/heads/master | cut -f1)" = "$sha"
    check "control: a missed redirect fails with the git transport error, not a network error" \
        blocked "${penv[@]}" GIT_CONFIG_COUNT=0 \
        git ls-remote https://github.com/DiamonDinoia/admiral.git refs/heads/master
    penv=()
fi

# Install once into $work/prefix, for the routes that read an installed prefix. The build tree
# goes at once, so a config that points into it fails.
if want prefix_shared || want prefix_static || want c_only || want cc_fftw || want relocated; then
    inst=$work/install
    rm -rf "$inst" "$work/prefix" "$work/moved"; mkdir -p "$inst"
    if ! { nice_run cmake -S "$src" -B "$inst/build" ${gen[@]+"${gen[@]}"} -DCMAKE_BUILD_TYPE=$cfg ${lib_opts[@]+"${lib_opts[@]}"} \
                -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF -DADM_BUILD_EXAMPLES=OFF \
                -DCMAKE_INSTALL_LIBDIR=lib \
                -DCMAKE_INSTALL_PREFIX="$work/prefix" >"$inst/configure.log" 2>&1 &&
            nice_run cmake --build "$inst/build" --config $cfg >"$inst/build.log" 2>&1 &&
            cmake --install "$inst/build" --config $cfg >"$inst/install.log" 2>&1; }; then
        echo "FAIL: install into $work/prefix"
        show_tail "$inst"
        exit 1
    fi
    echo "PASS: install into $work/prefix"
    rm -rf "$inst/build"
fi

prefix_route() { # name prefix tests consumer-route [cmake args...]
    local name=$1 prefix=$2 count=$3 r=$4; shift 4
    check "route $name" route "$name" "$count" admiral_dir "$(config_dir "$prefix")" exact \
        -DADM_PACKAGE_ROUTE="$r" -DCMAKE_PREFIX_PATH="$prefix" "$@"
}

if want prefix_shared; then
    prefix_route prefix_shared "$work/prefix" 3 prefix_shared
fi
if want prefix_static; then
    prefix_route prefix_static "$work/prefix" 3 prefix_static
fi
if want c_only; then
    # A C-only project links the shared C targets. It needs no C++ language of its own.
    prefix_route c_only "$work/prefix" 2 prefix_shared -DADM_PACKAGE_CXX=OFF
    # The README says: to link a static twin from C, use project(x C CXX). For MSVC, CXX is not
    # necessary: each object of the archive carries /DEFAULTLIB:msvcprt, and link.exe obeys it.
    cid=$(sed -n 's/^-- The C compiler identification is \([^ ]*\).*/\1/p' \
        "$work/c_only/configure.log")
    check "c_only: the configure log names the C compiler '$cid'" test -n "$cid"
    static=(c_only_static 2 admiral_dir "$(config_dir "$work/prefix")" exact
        -DADM_PACKAGE_ROUTE=prefix_static -DADM_PACKAGE_CXX=OFF -DCMAKE_PREFIX_PATH="$work/prefix")
    if [ "$cid" = MSVC ]; then
        check "route c_only_static: MSVC links the static C targets without C++" \
            route "${static[@]}"
    else
        check "control: a C-only project cannot link the static C targets" \
            not route "${static[@]}"
        check "control: that link misses a C++ runtime symbol" \
            grep -m1 -o '__cxa_[a-z_]*' "$work/c_only_static/build.log"
    fi
fi
if want cc_fftw; then
    # Plain compiler, no CMake: the way most FFTW programs build.
    dir=$work/cc_fftw p=$work/prefix
    rm -rf "$dir"; mkdir -p "$dir"
    check "cc_fftw: cc links use_fftw.c to the shim" "${CC:-cc}" -o "$dir/app" "$here/use_fftw.c" \
        -I"$p/include" -L"$p/lib" -ladmiral_fftw -Wl,-rpath,"$p/lib" -lm
    rc=0
    LD_DEBUG=bindings "$dir/app" >"$dir/run.log" 2>"$dir/bind.log" || rc=$?
    check "cc_fftw: app exits 0, status $rc" test "$rc" -eq 0
    if [ "$(uname)" = Darwin ]; then
        # macOS has no binding log. The load commands prove the link, not the binding.
        check "cc_fftw: app links libadmiral_fftw" grep -q libadmiral_fftw <(otool -L "$dir/app")
    else
        check "cc_fftw: app binds fftw_plan_dft_1d to the shim" \
            bound "$dir/bind.log" "$dir/app" fftw_plan_dft_1d '/libadmiral_fftw\.so'
        check "control: binding check rejects real FFTW for fftw_plan_dft_1d" \
            not bound "$dir/bind.log" "$dir/app" fftw_plan_dft_1d '/libfftw3\.so[.0-9]*'
    fi
fi
if want relocated; then
    # Relocation: move the prefix, remove the original, and read the copy.
    mv "$work/prefix" "$work/moved"
    check "relocated: no installed file holds a path of the install" \
        relocatable "$work/moved" "$work/prefix" "$inst" "$src"
    prefix_route relocated "$work/moved" 6 prefix
fi

src_opts=(${lib_opts[@]+"${lib_opts[@]}"} "-DADM_PACKAGE_SRC=$src")
if want subdir; then
    check "route subdir" route subdir 6 source_dir "$src" exact -DADM_PACKAGE_ROUTE=subdir \
        "${src_opts[@]}"
fi
if want fetch_dir; then
    check "route fetch_dir" route fetch_dir 6 source_dir "$src" exact \
        -DADM_PACKAGE_ROUTE=fetch_dir -DFETCHCONTENT_SOURCE_DIR_ADMIRAL="$src" "${src_opts[@]}"
fi

want_sha=$sha
if want fetch_git; then
    rm -rf "$work/fetch_git"; mkdir -p "$work/fetch_git"; git_env "$work/fetch_git"
    check "route fetch_git" route fetch_git 6 source_dir "$work/fetch_git/build/_deps" prefix \
        -DADM_PACKAGE_ROUTE=fetch_git "${src_opts[@]}"
fi
if want cpm; then
    rm -rf "$work/cpm"; mkdir -p "$work/cpm"; git_env "$work/cpm"
    check "route cpm" route cpm 6 source_dir "$work/cpm/cpm-cache/admiral" prefix \
        -DADM_PACKAGE_ROUTE=cpm "${src_opts[@]}"
fi
finish
