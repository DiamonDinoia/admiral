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

./app >"$plain_out"
LD_DEBUG=bindings ./app 2>"$plain_log" >/dev/null
{
LD_PRELOAD=$PREFIX/lib/libadmiral_fftw.so ./app
} >"$pre_out"
LD_PRELOAD=$PREFIX/lib/libadmiral_fftw.so LD_DEBUG=bindings ./app 2>"$pre_log" >/dev/null
pass "ran app plain and with LD_PRELOAD=$shim"

# compare <ref> <got> <tol_d> <tol_f>: nonzero when any array differs beyond its tolerance.
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
if len(a) != 24 or a.keys() != b.keys():
    sys.exit("expected 24 arrays with equal tags, got %d and %d" % (len(a), len(b)))
worst = 0.0
for tag, x in sorted(a.items()):
    y = b[tag]
    if len(x) != len(y) or not x:
        sys.exit("array %s: length %d vs %d" % (tag, len(x), len(y)))
    rel = math.sqrt(sum((p - q) ** 2 for p, q in zip(x, y)) / sum(p * p for p in x))
    if not rel <= tol[tag[0]]:
        sys.exit("array %s: relative error %.3g exceeds %.3g" % (tag, rel, tol[tag[0]]))
    worst = max(worst, rel / tol[tag[0]])
print("24 arrays within tolerance, worst error is %.3g of the bound" % worst)
PY
}

# bound <log> <symbol> <library regex>: every binding of <symbol> made by the program
# goes to a library path that matches the regex, and at least one binding exists.
bound() {
    local lines
    lines=$(grep -F "binding file ./app " "$1" | grep -F "\`$2'" || true)
    [ -n "$lines" ] || return 2
    ! printf '%s\n' "$lines" | grep -Ev " to [^ ]*$3 \[" >/dev/null
}

if msg=$(compare "$plain_out" "$pre_out" "$tol_d" "$tol_f" 2>&1); then
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

# Positive controls. Each check must fail on a known bad input.
awk 'NR == 100 { $3 = $3 * 1.001 } { print }' "$plain_out" >"$work/perturbed.out"
if compare "$plain_out" "$work/perturbed.out" "$tol_d" "$tol_f" >/dev/null 2>&1; then
    bad "control: comparator accepted an output with one value changed by 0.1 percent"
else
    pass "control: comparator rejects an output with one value changed by 0.1 percent"
fi
if compare "$plain_out" "$plain_out" "$tol_d" "$tol_f" >/dev/null 2>&1; then
    pass "control: comparator accepts identical outputs"
else
    bad "control: comparator rejected identical outputs"
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
