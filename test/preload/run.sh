#!/usr/bin/env bash
# Run an unmodified FFTW program on admiral's FFTW shim through LD_PRELOAD.
# Usage: run.sh <install-prefix> <work-dir>
# A failed setup step (a build or a reference run) stops the script through set -e.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=test/check.sh
. "$here/../check.sh"

PREFIX=${1:?usage: run.sh <install-prefix> <work-dir>}
work=${2:?usage: run.sh <install-prefix> <work-dir>}
mkdir -p "$work"
PREFIX=$(cd "$PREFIX" && pwd)
work=$(cd "$work" && pwd)
shim=$PREFIX/lib/libadmiral_fftw.so
[ -f "$shim" ] || { echo "FAIL: no $shim. Install with -DCMAKE_INSTALL_LIBDIR=lib."; exit 1; }
# Compiler flags of FFTW, one array element for each flag.
read -ra fftw_cflags <<<"$(pkg-config --cflags fftw3 fftw3f)"
read -ra fftw_libs <<<"$(pkg-config --libs fftw3 fftw3f)"
cd "$work"

# exits <status> <out> <err> <command...>: the command exits with <status> in 60 s. Its output
# goes to the two files.
exits() {
    local want=$1 out=$2 err=$3 rc=0
    shift 3
    timeout 60 "$@" >"$out" 2>"$err" || rc=$?
    [ "$rc" -eq "$want" ] && return
    echo "status $rc, expected $want"
    grep -v 'binding file' "$err" | tail -n 5
    return 1
}

# Tolerance: 120 eps per array, rounded up to 2.7e-14 for double (eps = 2^-52) and 1.5e-5 for
# float (eps = 2^-23). This test sets the tolerance itself. Theorem 24.2 of Higham, Accuracy and
# Stability of Numerical Algorithms (2002), only motivates it. With twiddles correct to
# u = eps/2, the theorem bounds the relative L2 error of a radix-2 FFT of length 2^t.
# The bound is t * (1 + 4 * sqrt(2)) * u to first order, and 30 eps at t = 9 (2^9 = 512).
# The theorem does not cover the mixed-radix lengths here (N <= 480).
# A b array is two transforms in series, and the reference and the shim each add their error:
# 2 * 2 * 30 eps = 120 eps.
# compare <ref> <got> <arrays>: the two files hold <arrays> arrays, and the arrays agree to the
# tolerance.
compare() {
    python3 -I - "$@" <<'PY'
import math, sys
ref, got, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
tol = {"d": 2.7e-14, "f": 1.5e-5}
def load(path):
    arrays = {}
    for line in open(path):
        tag, _, re, im = line.split()
        arrays.setdefault(tag, []).extend((float(re), float(im)))
    return arrays
a, b = load(ref), load(got)
if len(a) != want or a.keys() != b.keys():
    sys.exit("expected %d arrays with equal tags, got %d and %d" % (want, len(a), len(b)))
worst = 0.0
for tag, x in sorted(a.items()):
    y = b[tag]
    if len(x) != len(y) or not x:
        sys.exit("array %s: length %d vs %d" % (tag, len(x), len(y)))
    rel = math.sqrt(sum((p - q) ** 2 for p, q in zip(x, y)) / sum(p * p for p in x))
    if not rel <= tol[tag[0]]:
        sys.exit("array %s: relative error %.3g exceeds %.3g" % (tag, rel, tol[tag[0]]))
    worst = max(worst, rel / tol[tag[0]])
print("%d arrays within tolerance, worst error is %.3g of it" % (want, worst))
PY
}

# fftw_imports <program>: the fftw symbols that the program imports, sorted.
fftw_imports() {
    nm -D --undefined-only "$1" | awk '$2 ~ /^fftwf?_/ { sub(/@.*/, "", $2); print $2 }' |
        LC_ALL=C sort
}
# real <symbol>: the path regex of the real FFTW of the precision of the symbol.
real() {
    case $1 in
        fftwf_*) echo '/libfftw3f\.so[.0-9]*' ;;
        *) echo '/libfftw3\.so[.0-9]*' ;;
    esac
}
# lookups <log>: sorted "<library> <symbol>" pairs that the shim bound in real FFTW. LD_DEBUG
# shows the dlsym lookups of the shim as bindings from the shim.
lookups() {
    grep -F "binding file $shim " "$1" |
        sed -nE 's|.*/(libfftw3f?)\.so[^ ]* \[[0-9]+\]: normal symbol .([[:alnum:]_]+).*|\1 \2|p' |
        LC_ALL=C sort -u
}

cc -o app "$here/app.c" "${fftw_cflags[@]}" "${fftw_libs[@]}" -lm
# One run gives both outputs: the program writes stdout, LD_DEBUG writes stderr.
timeout 60 env LD_DEBUG=bindings ./app >plain.out 2>plain.log
# The docs copy the line in the here-document, so sh runs that line.
check "app runs under LD_PRELOAD=$shim" exits 0 preload.out preload.log \
    env PREFIX="$PREFIX" LD_DEBUG=bindings sh <<'EOF'
# [doc:ld-preload]
LD_PRELOAD=$PREFIX/lib/libadmiral_fftw.so ./app
# [doc:ld-preload-end]
EOF
check "preloaded output matches plain output" compare plain.out preload.out 24

# Every fftw symbol the program calls must bind to the real library in the plain run
# and to the shim in the preloaded run. The list comes from the program, not from this script.
syms=$(fftw_imports app)
check "app imports 24 fftw symbols" test "$(wc -w <<<"$syms")" -eq 24
for sym in $syms; do
    check "plain run binds $sym to real FFTW" bound plain.log ./app "$sym" "$(real "$sym")"
    check "preloaded run binds $sym to the shim" \
        bound preload.log ./app "$sym" '/libadmiral_fftw\.so'
done
# The c2c app owns every plan, so the shim looks up only the two cleanup calls.
check "c2c app: the shim looked up only fftw_cleanup and fftwf_cleanup in real FFTW" \
    diff <(printf '%s\n' "libfftw3 fftw_cleanup" "libfftw3f fftwf_cleanup") <(lookups preload.log)

# Mixed plans. Real FFTW makes the r2c and guru plans (the shim has no such planner) and the
# shim makes the c2c plans. The shim must hand the real plans on to real FFTW.
# mixed checks that no heap is left, and the glibc tcache would hide freed chunks from that count.
export GLIBC_TUNABLES=glibc.malloc.tcache_count=0
cc -o mixed "$here/mixed.c" "${fftw_cflags[@]}" "${fftw_libs[@]}" -lm
timeout 60 ./mixed >mixed_plain.out
check "mixed plans: mixed runs under the preload" exits 0 mixed_preload.out mixed_preload.log \
    env LD_PRELOAD="$shim" LD_DEBUG=bindings ./mixed
check "mixed plans: preloaded output matches plain output" \
    compare mixed_plain.out mixed_preload.out 8
exports=$(nm -D --defined-only "$shim" | awk '{ print $3 }')
foreign=
for sym in $(fftw_imports mixed); do
    if grep -Fxq "$sym" <<<"$exports"; then
        check "mixed plans: preloaded run binds $sym to the shim" \
            bound mixed_preload.log ./mixed "$sym" '/libadmiral_fftw\.so'
    else
        foreign="$foreign $sym"
        check "mixed plans: preloaded run binds $sym to real FFTW" \
            bound mixed_preload.log ./mixed "$sym" "$(real "$sym")"
    fi
done
check "mixed plans: real FFTW makes 4 plans (r2c and guru, both precisions):$foreign" \
    test "$(wc -w <<<"$foreign")" -eq 4
# Each precision forwards these four to its own library.
check "mixed plans: each precision forwards exactly execute, execute_dft, destroy_plan, cleanup" \
    diff <(for f in cleanup destroy_plan execute execute_dft; do echo "libfftw3 fftw_$f"; done
           for f in cleanup destroy_plan execute execute_dft; do echo "libfftw3f fftwf_$f"; done) \
    <(lookups mixed_preload.log)
check "control: mixed exits 1 when it does not destroy one plan" exits 1 /dev/null mixed_leak.err \
    env LD_PRELOAD="$shim" ./mixed leak
check "control: mixed names the heap that the plan left" \
    grep '^mixed: heap in use went from' mixed_leak.err

# The static shim in a program that links real FFTW too. The program's own copy of the shim takes
# the c2c calls, and dlsym(RTLD_NEXT) in it finds real FFTW for the other plans.
cc -c -o mixed.o "$here/mixed.c" "${fftw_cflags[@]}"
c++ -o mixed_static mixed.o "$PREFIX/lib/libadmiral_fftw.a" "${fftw_libs[@]}" -ldl -pthread
check "static shim: mixed_static holds the shim's fftw_execute" \
    grep -q ' T fftw_execute$' <(nm --defined-only mixed_static)
check "static shim: mixed_static runs" exits 0 mixed_static.out mixed_static.err ./mixed_static
check "static shim: output matches plain output" compare mixed_plain.out mixed_static.out 8

# A plugin that links FFTW, opened with dlopen(RTLD_LOCAL) as Python opens an extension module.
# That FFTW is not in the global scope, so dlsym(RTLD_NEXT) in the shim cannot find it.
# plugin_standin.so links standin.c under a new soname instead, as a Python wheel bundles FFTW.
# The shim must find that FFTW by the symbol, not by the soname.
# plugin_shim.so links the shim before the stand-in, so the search of its scope finds the shim
# first. The shim must skip itself there.
cc -shared -fPIC -o plugin.so "$here/plugin.c" "${fftw_cflags[@]}" "${fftw_libs[@]}"
standin=libfftw3-standin.so.3
cc -shared -fPIC -o "$standin" -Wl,-soname,"$standin" "$here/standin.c" "${fftw_cflags[@]}"
# shellcheck disable=SC2016 # The linker expands $ORIGIN, not the shell.
cc -shared -fPIC -o plugin_standin.so "$here/plugin.c" "${fftw_cflags[@]}" "./$standin" \
    -Wl,-rpath,'$ORIGIN'
# shellcheck disable=SC2016 # The linker expands $ORIGIN, not the shell.
cc -shared -fPIC -o plugin_shim.so "$here/plugin.c" "${fftw_cflags[@]}" \
    -L"$PREFIX/lib" -ladmiral_fftw "./$standin" -Wl,-rpath,"$PREFIX/lib" -Wl,-rpath,'$ORIGIN'
# host <plugin>...: opens each plugin with RTLD_LOCAL in turn and runs the last.
host='import ctypes, sys
libs = [ctypes.CDLL(a, ctypes.RTLD_LOCAL) for a in sys.argv[1:]]
sys.exit(libs[-1].run_plugin())'
for plugin in plugin.so plugin_standin.so plugin_shim.so; do
    check "$plugin: the RTLD_LOCAL plugin runs without the preload" \
        exits 0 "$plugin.out" "$plugin.err" python3 -I -c "$host" "./$plugin"
    check "$plugin: the RTLD_LOCAL plugin runs under the preload" \
        exits 0 "$plugin.out" "$plugin.log" \
        env LD_PRELOAD="$shim" LD_DEBUG=bindings python3 -I -c "$host" "./$plugin"
    for sym in fftw_execute fftw_destroy_plan fftwf_execute fftwf_destroy_plan; do
        check "$plugin: preloaded run binds $sym of the plugin to the shim" \
            bound "$plugin.log" "./$plugin" "$sym" '/libadmiral_fftw\.so'
    done
done

# The search opens each object it passes and must close it again. unload runs plugin_standin.so,
# closes it when its argument is "close", and exits 3 if dlopen(RTLD_NOLOAD) still finds it.
unload='import ctypes, os, sys, _ctypes
lib = ctypes.CDLL("./plugin_standin.so", os.RTLD_LOCAL)
rc = lib.run_plugin()
if sys.argv[1] == "close":
    _ctypes.dlclose(lib._handle)
try:
    ctypes.CDLL("./plugin_standin.so", os.RTLD_LOCAL | os.RTLD_NOLOAD)
    sys.exit(3)
except OSError:
    sys.exit(rc)'
check "unload: the plugin unloads after dlclose" exits 0 /dev/null unload.err \
    env LD_PRELOAD="$shim" python3 -I -c "$unload" close
check "control: unload exits 3 when the plugin stays loaded" exits 3 /dev/null unload.err \
    env LD_PRELOAD="$shim" python3 -I -c "$unload" keep

# fftw_cleanup before any other FFTW loads has nothing to forward to. The call after the stand-in
# loads must reach it, so the shim must not remember the first lookup.
cleanup='import ctypes, os, sys
shim = ctypes.CDLL(None)
shim.fftw_cleanup()
lib = ctypes.CDLL("./plugin_standin.so", os.RTLD_LOCAL)
shim.fftw_cleanup()
sys.exit(lib.standin_cleanups() != 1)'
check "cleanup: fftw_cleanup reaches an FFTW that loaded after the first call" \
    exits 0 /dev/null cleanup.err env LD_PRELOAD="$shim" python3 -I -c "$cleanup"

# plugin.so loads real FFTW first. plugin_shim_real.so links the shim and that FFTW, so the FFTW
# loads before the shim, and only dlsym(RTLD_NEXT) in the shim finds it.
cc -shared -fPIC -o plugin_shim_real.so "$here/plugin.c" "${fftw_cflags[@]}" \
    -L"$PREFIX/lib" -ladmiral_fftw "${fftw_libs[@]}" -Wl,-rpath,"$PREFIX/lib"
check "FFTW loaded before the shim: the plugin runs" exits 0 /dev/null shim_real.log \
    env LD_DEBUG=bindings python3 -I -c "$host" ./plugin.so ./plugin_shim_real.so
check "FFTW loaded before the shim: the shim passed fftw_execute on through RTLD_NEXT" \
    bound shim_real.log ./plugin_shim_real.so fftw_execute '/libadmiral_fftw\.so'

# Two shim copies. The first finds the second through RTLD_NEXT. The second must find the
# stand-in, which loads after it, and not the first copy.
mkdir -p copy
cp "$shim" copy/
check "two shim copies: the second copy passed the plans on to the stand-in" \
    exits 0 /dev/null copies.log env LD_PRELOAD="$shim $work/copy/libadmiral_fftw.so" \
    LD_DEBUG=bindings python3 -I -c "$host" ./plugin_standin.so
check "two shim copies: the first copy passed fftw_execute on to the second" \
    bound copies.log "$shim" fftw_execute "$work/copy/libadmiral_fftw\.so"

# The stand-in sends the plan back to the shim. In back/ it does so on the same thread, as a
# second shim copy does when it finds the first. The shim must abort with a message, not loop.
# In thread/ it does so from a second thread while the first is inside the shim. The guard holds
# for each thread, so the shim must pass that call on. Each directory holds the plugin and its
# stand-in, built with SEND_BACK=1 and 2.
mkdir -p back thread
for mode in 1 2; do
    dir=back
    [ "$mode" = 1 ] || dir=thread
    cc -shared -fPIC -pthread -o "$dir/$standin" -Wl,-soname,"$standin" -DSEND_BACK="$mode" \
        "$here/standin.c" "${fftw_cflags[@]}"
    cp plugin_standin.so "$dir/"
done
check "send-back: the shim aborts when the next FFTW sends fftw_execute back" \
    exits 134 /dev/null back.err \
    env LD_PRELOAD="$shim" python3 -I -c "$host" ./back/plugin_standin.so
check "send-back: the abort comes with the message of the shim" \
    grep '^admiral_fftw: fftw_execute came back to this library' back.err
check "second thread: the shim passes a plan on while another thread is inside the shim" \
    exits 0 /dev/null thread.err \
    env LD_PRELOAD="$shim" python3 -I -c "$host" ./thread/plugin_standin.so

# Positive controls. Each check must fail on a known bad input.
# perturb <tag> <relative error>: copy of the plain output with one array off by that
# relative L2 error. The limits are literal, so a loosened tolerance fails a control.
perturb() {
    python3 -I - plain.out "$1" "$2" <<'PY'
import math, sys
path, tag, rel = sys.argv[1], sys.argv[2], float(sys.argv[3])
lines = [l.split() for l in open(path)]
rows = [l for l in lines if l[0] == tag]
norm = math.sqrt(sum(float(l[2]) ** 2 + float(l[3]) ** 2 for l in rows))
rows[0][2] = repr(float(rows[0][2]) + rel * norm)
for l in lines:
    print(*l)
PY
}
# The tolerances are 2.7e-14 (double) and 1.5e-5 (float): 10x must fail, 0.1x must pass.
check "control: compare rejects d1b off by 2.7e-13" not compare plain.out <(perturb d1b 2.7e-13) 24
check "control: compare accepts d1b off by 2.7e-15" compare plain.out <(perturb d1b 2.7e-15) 24
check "control: compare rejects f1b off by 1.5e-4" not compare plain.out <(perturb f1b 1.5e-4) 24
check "control: compare accepts f1b off by 1.5e-6" compare plain.out <(perturb f1b 1.5e-6) 24
check "control: binding check rejects a symbol that has no binding" \
    not bound plain.log ./app no_such_symbol '.*'
check "control: binding check rejects the shim for fftw_plan_dft_r2c_1d" \
    not bound mixed_preload.log ./mixed fftw_plan_dft_r2c_1d '/libadmiral_fftw\.so'
finish
