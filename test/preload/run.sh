#!/usr/bin/env bash
# Run an unmodified FFTW program on admiral's FFTW shim through LD_PRELOAD.
# Usage: run.sh <install-prefix> <work-dir>
set -eu

PREFIX=${1:?usage: run.sh <install-prefix> <work-dir>}
work=${2:?usage: run.sh <install-prefix> <work-dir>}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
PREFIX=$(cd "$PREFIX" && pwd)
work=$(cd "$work" && pwd)
# Compiler flags of FFTW, one array element for each flag.
read -ra fftw_cflags <<<"$(pkg-config --cflags fftw3)"
read -ra fftw_libs <<<"$(pkg-config --libs fftw3 fftw3f)"
read -ra fftw_cflibs <<<"$(pkg-config --cflags --libs fftw3 fftw3f)"

# Tolerance: 120 eps per array, rounded up to 2.7e-14 for double (eps = 2^-52) and 1.5e-5 for
# float (eps = 2^-23). This test sets the tolerance itself. Theorem 24.2 of Higham, Accuracy and
# Stability of Numerical Algorithms (2002), only motivates it. With twiddles correct to
# u = eps/2, the theorem bounds the relative L2 error of a radix-2 FFT of length 2^t.
# The bound is t * (1 + 4 * sqrt(2)) * u to first order, and 30 eps at t = 9 (2^9 = 512).
# The theorem does not cover the mixed-radix lengths here (N <= 480).
# A b array is two transforms in series, and the reference and the shim each add their error:
# 2 * 2 * 30 eps = 120 eps. The measured worst error is below 3e-2 of the tolerance.
tol_d=2.7e-14
tol_f=1.5e-5

fail=0
pass() { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; fail=1; }
# A check records its result with pass or bad. The script exits 1 at the end if a check failed.
# Setup steps build the programs, libraries and files that the checks use, and run real FFTW for
# the reference outputs. A failed setup step stops the script through set -e. A missing shim
# stops it too.

shim=$PREFIX/lib/libadmiral_fftw.so
if [ ! -f "$shim" ]; then
    echo "FAIL: $shim is absent. Install with -DCMAKE_INSTALL_LIBDIR=lib."
    exit 1
fi

cp "$here/app.c" "$work/app.c"
cd "$work"
cc -o app app.c $(pkg-config --cflags --libs fftw3 fftw3f) -lm
pass "built app against real FFTW"

plain_out=$work/plain.out
plain_log=$work/plain.log
pre_out=$work/preload.out
pre_log=$work/preload.log

# One run gives both outputs: the program writes stdout, LD_DEBUG writes stderr.
# timeout turns a hang into a failure.
timeout 60 env LD_DEBUG=bindings ./app >"$plain_out" 2>"$plain_log"
# A crash under the preload is a result, not a reason to stop: report it and go on.
rc=0
timeout 60 env LD_PRELOAD="$shim" LD_DEBUG=bindings ./app >"$pre_out" 2>"$pre_log" || rc=$?
if [ "$rc" -eq 0 ]; then
    pass "ran app plain and with LD_PRELOAD=$shim"
else
    bad "app exited with status $rc under LD_PRELOAD=$shim"
fi

# compare <ref> <got> <tol_d> <tol_f> <arrays>: nonzero when any array differs beyond its tolerance.
compare() {
    python3 -I - "$@" <<'PY'
import math, sys
ref, got, tol = sys.argv[1], sys.argv[2], {"d": float(sys.argv[3]), "f": float(sys.argv[4])}
def load(path):
    arrays = {}
    for line in open(path):
        tag, _, re, im = line.split()
        arrays.setdefault(tag, []).extend((float(re), float(im)))
    return arrays
a, b = load(ref), load(got)
want = int(sys.argv[5])
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

# bound <log> <symbol> <library regex> [program]: the first binding of <symbol> made by the program
# goes to a library path that matches the regex. Later lines come from dlsym on the handle of the
# program, which the search of the shim does. With no binding, the check fails.
bound() {
    awk -v f="binding file ./${4:-app} " -v s="\`$2'" 'index($0, f) && index($0, s) { print; exit }' \
        "$1" | grep -Eq " to [^ ]*$3 \["
}

if msg=$(compare "$plain_out" "$pre_out" "$tol_d" "$tol_f" 24 2>&1); then
    pass "preloaded output matches plain output: $msg"
else
    bad "preloaded output differs from plain output: $msg"
fi

# Every fftw symbol the program calls must bind to the real library in the plain run
# and to the shim in the preloaded run. The list comes from the program, not from this script.
syms=$(nm -D --undefined-only app | awk '$2 ~ /^fftwf?_/ { sub(/@.*/, "", $2); print $2 }' | sort)
if [ "$(printf '%s\n' "$syms" | wc -l)" -eq 24 ]; then
    pass "app calls 24 fftw symbols"
else
    bad "app calls $(printf '%s\n' "$syms" | wc -l) fftw symbols, expected 24"
fi
for sym in $syms; do
    case $sym in
        fftwf_*) real='/libfftw3f\.so[.0-9]*' ;;
        *) real='/libfftw3\.so[.0-9]*' ;;
    esac
    if bound "$plain_log" "$sym" "$real"; then
        pass "plain run binds $sym to real FFTW"
    else
        bad "plain run does not bind $sym to real FFTW"
    fi
    if bound "$pre_log" "$sym" '/libadmiral_fftw\.so'; then
        pass "preloaded run binds $sym to libadmiral_fftw.so"
    else
        bad "preloaded run does not bind $sym to libadmiral_fftw.so"
    fi
done

# dlsym lookups show in LD_DEBUG=bindings as bindings from the shim to libfftw3.
# shim_to_real <log>: sorted "<library> <symbol>" pairs the shim looked up in real FFTW.
shim_to_real() {
    python3 -I - "$1" "$shim" <<'PY'
import re, sys
log, shim = sys.argv[1:]
pat = re.compile(r" to \S*/(libfftw3f?)\.so\S* \[\d+\]: normal symbol `(\w+)'")
found = set()
for line in open(log):
    m = pat.search(line) if ("binding file " + shim + " ") in line else None
    if m:
        found.add(m.group(1) + " " + m.group(2))
print("\n".join(sorted(found)))
PY
}
# The c2c app owns every plan, so the shim looks up only the two cleanup calls.
want_app="libfftw3 fftw_cleanup
libfftw3f fftwf_cleanup"
# The mixed program gives it foreign plans: each precision forwards these four to its own library.
want_mixed="libfftw3 fftw_cleanup
libfftw3 fftw_destroy_plan
libfftw3 fftw_execute
libfftw3 fftw_execute_dft
libfftw3f fftwf_cleanup
libfftw3f fftwf_destroy_plan
libfftw3f fftwf_execute
libfftw3f fftwf_execute_dft"
if got_app=$(shim_to_real "$pre_log") && [ "$got_app" = "$want_app" ]; then
    pass "c2c app: the shim looked up only fftw_cleanup and fftwf_cleanup in real FFTW"
else
    bad "c2c app: the shim looked up in real FFTW: $(printf '%s' "$got_app" | tr '\n' ';')"
fi

# Mixed plans. Real FFTW makes the r2c and guru plans (the shim has no such planner) and the
# shim makes the c2c plans. The shim must hand the real plans on to real FFTW.
# mixed checks that no heap is left, and the glibc tcache would hide freed chunks from that count.
export GLIBC_TUNABLES=glibc.malloc.tcache_count=0
cc -o mixed "$here/mixed.c" "${fftw_cflibs[@]}" -lm
timeout 60 ./mixed >mixed_plain.out
rc=0
timeout 60 env LD_PRELOAD="$shim" LD_DEBUG=bindings ./mixed \
    >mixed_preload.out 2>mixed_preload.log || rc=$?
if [ "$rc" -eq 0 ]; then
    pass "mixed plans: ran mixed under the preload"
else
    bad "mixed plans: mixed exited with status $rc under the preload" \
        "$(grep '^mixed:' mixed_preload.log)"
fi
if msg=$(compare mixed_plain.out mixed_preload.out "$tol_d" "$tol_f" 8 2>&1); then
    pass "mixed plans: preloaded output matches plain output: $msg"
else
    bad "mixed plans: preloaded output differs from plain output: $msg"
fi
shim_exports=$(nm -D --defined-only "$shim" | awk '{ print $3 }')
foreign=0
for sym in $(nm -D --undefined-only mixed |
    awk '$2 ~ /^fftwf?_/ { sub(/@.*/, "", $2); print $2 }' | sort); do
    case $sym in
        fftwf_*) real='/libfftw3f\.so[.0-9]*' ;;
        *) real='/libfftw3\.so[.0-9]*' ;;
    esac
    if printf '%s\n' "$shim_exports" | grep -Fxq "$sym"; then
        want='/libadmiral_fftw\.so'
        owner="the shim"
    else
        want=$real
        owner="real FFTW"
        foreign=$((foreign + 1))
    fi
    if bound mixed_preload.log "$sym" "$want" mixed; then
        pass "mixed plans: preloaded run binds $sym to $owner"
    else
        bad "mixed plans: preloaded run does not bind $sym to $owner"
    fi
done
if [ "$foreign" -eq 4 ]; then
    pass "mixed plans: 4 planners (r2c and guru, both precisions) bind to real FFTW"
else
    bad "mixed plans: $foreign planners bind to real FFTW, expected 4"
fi
if got_mixed=$(shim_to_real mixed_preload.log) && [ "$got_mixed" = "$want_mixed" ]; then
    pass "mixed plans: each precision forwards exactly execute, execute_dft, destroy_plan, cleanup"
else
    bad "mixed plans: the shim looked up in real FFTW: $(printf '%s' "$got_mixed" | tr '\n' ';')"
fi
if bound mixed_preload.log fftw_plan_dft_r2c_1d '/libadmiral_fftw\.so' mixed; then
    bad "control: binding check accepted the shim for fftw_plan_dft_r2c_1d"
else
    pass "control: binding check rejects the shim for fftw_plan_dft_r2c_1d"
fi
rc=0
timeout 60 env LD_PRELOAD="$shim" ./mixed leak >/dev/null 2>mixed_leak.err || rc=$?
if [ "$rc" -eq 1 ] && grep -q '^mixed: heap in use went from' mixed_leak.err; then
    pass "control: the heap check fails when mixed does not destroy one plan"
else
    bad "control: mixed exited with status $rc and no heap message with one plan not destroyed"
fi

# The static shim in a program that links real FFTW too. The program's own copy of the shim takes
# the c2c calls, and dlsym(RTLD_NEXT) in it finds real FFTW for the other plans.
cc -c -o mixed.o "$here/mixed.c" "${fftw_cflags[@]}"
c++ -o mixed_static mixed.o "$PREFIX/lib/libadmiral_fftw.a" "${fftw_libs[@]}" \
    -ldl -pthread || bad "static shim: could not link mixed_static"
if nm --defined-only mixed_static | grep -q ' T fftw_execute$'; then
    pass "static shim: mixed_static holds the shim's fftw_execute"
else
    bad "static shim: mixed_static does not define fftw_execute"
fi
rc=0
timeout 60 ./mixed_static >mixed_static.out 2>mixed_static.err || rc=$?
if [ "$rc" -eq 0 ]; then
    pass "static shim: ran mixed_static"
else
    bad "static shim: mixed_static exited with status $rc: $(head -c 300 mixed_static.err)"
fi
if msg=$(compare mixed_plain.out mixed_static.out "$tol_d" "$tol_f" 8 2>&1); then
    pass "static shim: output matches plain output: $msg"
else
    bad "static shim: output differs from plain output: $msg"
fi

# A plugin that links FFTW, opened with dlopen(RTLD_LOCAL) as Python opens an extension module.
# That FFTW is not in the global scope, so dlsym(RTLD_NEXT) in the shim cannot find it.
# plugin_standin.so links standin.c under a new soname instead, as a Python wheel bundles FFTW.
# The shim must find that FFTW by the symbol, not by the soname.
# plugin_shim.so links the shim before the stand-in, so the search of its scope finds the shim
# first. The shim must skip itself there.
cc -shared -fPIC -o plugin.so "$here/plugin.c" "${fftw_cflibs[@]}"
standin=libfftw3-standin.so.3
cc -shared -fPIC -o "$standin" -Wl,-soname,"$standin" "$here/standin.c" "${fftw_cflags[@]}"
# shellcheck disable=SC2016 # The linker expands $ORIGIN, not the shell.
cc -shared -fPIC -o plugin_standin.so "$here/plugin.c" "${fftw_cflags[@]}" "./$standin" \
    -Wl,-rpath,'$ORIGIN'
# shellcheck disable=SC2016 # The linker expands $ORIGIN, not the shell.
cc -shared -fPIC -o plugin_shim.so "$here/plugin.c" "${fftw_cflags[@]}" \
    -L"$PREFIX/lib" -ladmiral_fftw "./$standin" -Wl,-rpath,"$PREFIX/lib" -Wl,-rpath,'$ORIGIN' \
    || bad "could not link plugin_shim.so"
# host <plugin>...: opens each plugin with RTLD_LOCAL in turn and runs the last.
host='import ctypes, sys
libs = [ctypes.CDLL(a, ctypes.RTLD_LOCAL) for a in sys.argv[1:]]
sys.exit(libs[-1].run_plugin())'
for plugin in plugin.so plugin_standin.so plugin_shim.so; do
    rc=0
    timeout 60 python3 -I -c "$host" ./$plugin || rc=$?
    if [ "$rc" -eq 0 ]; then
        pass "$plugin: ran the RTLD_LOCAL plugin without the preload"
    else
        bad "$plugin: the RTLD_LOCAL plugin exited with status $rc without the preload"
    fi
    rc=0
    timeout 60 env LD_PRELOAD="$shim" LD_DEBUG=bindings python3 -I -c "$host" ./$plugin \
        2>$plugin.log || rc=$?
    if [ "$rc" -eq 0 ]; then
        pass "$plugin: ran the RTLD_LOCAL plugin under the preload"
    else
        bad "$plugin: the RTLD_LOCAL plugin exited with status $rc under the preload"
    fi
    for sym in fftw_execute fftw_destroy_plan fftwf_execute fftwf_destroy_plan; do
        if bound $plugin.log "$sym" '/libadmiral_fftw\.so' $plugin; then
            pass "$plugin: preloaded run binds $sym of the plugin to the shim"
        else
            bad "$plugin: preloaded run does not bind $sym of the plugin to the shim"
        fi
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
for case in "close 0" "keep 3"; do
    # shellcheck disable=SC2086 # Split the case string into its fields on purpose.
    set -- $case
    rc=0
    timeout 60 env LD_PRELOAD="$shim" python3 -I -c "$unload" "$1" || rc=$?
    if [ "$rc" -eq "$2" ]; then
        pass "unload: with $1 the host exits $rc (0: the plugin unloaded, 3: it stayed loaded)"
    else
        bad "unload: with $1 the host exits $rc, expected $2 (0: unloaded, 3: stayed loaded)"
    fi
done

# plugin.so loads real FFTW first. plugin_shim_real.so links the shim and that FFTW, so the FFTW
# loads before the shim, and only dlsym(RTLD_NEXT) in the shim finds it.
cc -shared -fPIC -o plugin_shim_real.so "$here/plugin.c" "${fftw_cflags[@]}" \
    -L"$PREFIX/lib" -ladmiral_fftw "${fftw_libs[@]}" -Wl,-rpath,"$PREFIX/lib" \
    || bad "could not link plugin_shim_real.so"
rc=0
timeout 60 env LD_DEBUG=bindings python3 -I -c "$host" ./plugin.so ./plugin_shim_real.so \
    2>shim_real.log || rc=$?
if [ "$rc" -eq 0 ] && bound shim_real.log fftw_execute '/libadmiral_fftw\.so' plugin_shim_real.so
then
    pass "FFTW loaded before the shim: the shim passed fftw_execute on through RTLD_NEXT"
else
    bad "FFTW loaded before the shim: status $rc, or fftw_execute did not bind to the shim"
fi

# Two shim copies. The first finds the second through RTLD_NEXT. The second must find the
# stand-in, which loads after it, and not the first copy.
mkdir -p copy
cp "$shim" copy/
rc=0
timeout 60 env LD_PRELOAD="$shim $work/copy/libadmiral_fftw.so" LD_DEBUG=bindings \
    python3 -I -c "$host" ./plugin_standin.so 2>copies.log || rc=$?
if awk -v f="binding file $shim " -v t=" to $work/copy/libadmiral_fftw.so [" \
    'index($0, f) && index($0, t) && /`fftw_execute.$/ { n++ } END { exit !n }' copies.log; then
    pass "two shim copies: the first copy passed fftw_execute on to the second"
else
    bad "two shim copies: the first copy did not pass fftw_execute on to the second"
fi
if [ "$rc" -eq 0 ]; then
    pass "two shim copies: the second copy passed the plans on to the stand-in"
else
    bad "two shim copies: the plugin exited with status $rc"
fi

# The stand-in sends the plan back to the shim. In back/ it does so on the same thread, as a
# second shim copy does when it finds the first. The shim must abort with a message, not loop.
# In thread/ it does so from a second thread while the first is inside the shim. The guard holds
# for each thread, so the shim must pass that call on. Each directory holds the plugin and its
# stand-in. A case is <directory>:<SEND_BACK>:<expected status>:<label>:<claim>.
for case in "back:1:134:send-back:the shim aborts when the next FFTW sends fftw_execute back" \
    "thread:2:0:second thread:the shim passes a plan on while another thread is inside the shim"
do
    IFS=: read -r dir mode want label claim <<<"$case"
    mkdir -p "$dir"
    cc -shared -fPIC -pthread -o "$dir/$standin" -Wl,-soname,"$standin" -DSEND_BACK="$mode" \
        "$here/standin.c" "${fftw_cflags[@]}"
    cp plugin_standin.so "$dir/"
    # The braces send the message of bash about the abort to $dir.err too.
    rc=0
    { timeout 60 env LD_PRELOAD="$shim" python3 -I -c "$host" "./$dir/plugin_standin.so"; } \
        2>"$dir.err" || rc=$?
    # An abort must come with the message of the shim.
    if [ "$rc" -eq "$want" ] && { [ "$rc" -eq 0 ] ||
        grep -q '^admiral_fftw: fftw_execute came back to this library' "$dir.err"; }; then
        pass "$label: $claim"
    else
        bad "$label: status $rc, expected $want, stderr: $(head -c 300 "$dir.err")"
    fi
done

# Positive controls. Each check must fail on a known bad input.
# perturb <tag> <relative error>: copy of the plain output with one array off by that
# relative L2 error. The limits are literal, so a loosened tolerance fails a control.
perturb() {
    python3 -I - "$plain_out" "$1" "$2" <<'PY'
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
for case in "d1b 2.7e-13 reject" "d1b 2.7e-15 accept" "f1b 1.5e-4 reject" "f1b 1.5e-6 accept"; do
    # shellcheck disable=SC2086 # Split the case string into its fields on purpose.
    set -- $case
    perturb "$1" "$2" >"$work/perturbed.out"
    if compare "$plain_out" "$work/perturbed.out" "$tol_d" "$tol_f" 24 >/dev/null 2>&1; then
        got=accept
    else
        got=reject
    fi
    if [ "$got" = "$3" ]; then
        pass "control: comparator ${got}s array $1 off by relative $2"
    else
        bad "control: comparator ${got}s array $1 off by relative $2, expected $3"
    fi
done
if compare "$plain_out" "$plain_out" "$tol_d" "$tol_f" 24 >/dev/null 2>&1; then
    pass "control: comparator accepts identical outputs"
else
    bad "control: comparator rejected identical outputs"
fi
if bound "$plain_log" no_such_symbol '.*'; then
    bad "control: binding check accepted a symbol that has no binding"
else
    pass "control: binding check rejects a symbol that has no binding"
fi
if bound "$plain_log" fftw_execute '/libadmiral_fftw\.so'; then
    bad "control: binding check accepted the plain log as a preload log"
else
    pass "control: binding check rejects the plain log as a preload log"
fi

if [ "$fail" -ne 0 ]; then
    echo "RESULT: FAIL"
    exit 1
fi
echo "RESULT: PASS"
