"""D2 (e). XED ISA sets per object. Each object is judged by the allow-list of the tier it was
built for; the class comes from the object list, not from a symbol name.

usage: python3 -I check_xed.py XED64 PROBE.o floor=LIST v3=LIST v4=LIST
LIST holds one object path per line. Exit 1 when: an object holds an ISA set outside its tier's
allow-list (tzcnt is the one BMI1 form the floor may hold); the floor holds other than 3 xgetbv;
xed64 reports a decode error; a class is empty. Controls, each must fire: v3 objects under the
floor list, v4 objects under the v3 list, and PROBE.o (one instruction per level) under the floor.
"""
import concurrent.futures as cf
import re
import subprocess
import sys
from collections import Counter

# x86-64 psABI levels in XED ISA-set names. v1: the sets of a -march=x86-64 build.
V1 = {"I86", "I186", "I386", "I486REAL", "PPRO", "LONGMODE", "CMOV", "FAT_NOP", "PAUSE", "X87",
      "FCMOV", "FCOMI", "SSE", "SSE2", "SSE_PREFETCH"}
V3 = V1 | {"CMPXCHG16B", "LAHF", "POPCNT", "SSE3", "SSSE3", "SSE4", "SSE42",
           "AVX", "AVX2", "AVX2GATHER", "BMI1", "BMI2", "F16C", "FMA", "LZCNT", "MOVBE", "XSAVE"}
V4 = re.compile(r"AVX512(F|CD|BW|DQ)_")
FLOOR_XGETBV = 3  # xsimd reads XCR0 once per tier_slot<T> (float, double, long double)


def allowed(cls, isa, mnem):
    if cls == "floor":
        return isa in V1 or (isa, mnem) in {("BMI1", "tzcnt"), ("XSAVE", "xgetbv")}
    return isa in V3 or (cls == "v4" and bool(V4.match(isa)))


def census(xed, obj):
    out = subprocess.run([xed, "-64", "-isa-set", "-i", obj], capture_output=True, text=True,
                         check=True).stdout
    c, errors = Counter(), 0
    for line in out.splitlines():
        if line.startswith("XDIS"):
            t = line.split()
            c[(t[4], t[6] if len(t) > 6 else "")] += 1
        elif line.startswith("ERROR"):
            errors += 1
    return c, errors


def judge(cls, counts):
    bad = Counter({k: n for k, n in counts.items() if not allowed(cls, *k)})
    return bad


def main():
    xed, probe = sys.argv[1], sys.argv[2]
    lists = dict(a.split("=", 1) for a in sys.argv[3:])
    objs = {k: open(v).read().split() for k, v in lists.items()}
    with cf.ThreadPoolExecutor(8) as ex:
        res = {k: list(ex.map(lambda o: census(xed, o), v)) for k, v in objs.items()}
    rc = 0
    total = {}
    for cls in ("floor", "v3", "v4"):
        c = sum((r[0] for r in res[cls]), Counter())
        total[cls] = c
        errs = sum(r[1] for r in res[cls])
        bad = judge(cls, c)
        sets = Counter()
        for (isa, _), n in c.items():
            sets[isa] += n
        print(f"{cls}: {len(objs[cls])} objects, {sum(c.values())} instructions, decode errors {errs}")
        print(f"  ISA sets: {dict(sets.most_common())}")
        print(f"  outside the {cls} allow-list: {dict(bad)}")
        ok = len(objs[cls]) > 0 and sum(c.values()) > 0 and not bad and errs == 0
        if cls == "floor":
            x = c[("XSAVE", "xgetbv")]
            print(f"  floor xgetbv: {x} (pinned {FLOOR_XGETBV}); tzcnt: {c[('BMI1', 'tzcnt')]}")
            ok = ok and x == FLOOR_XGETBV
        print(f"  {cls} verdict: {'PASS' if ok else 'FAIL'}")
        rc |= not ok
    pc, _ = census(xed, probe)
    controls = {
        "v3 objects under the floor allow-list": judge("floor", total["v3"]),
        "v4 objects under the v3 allow-list": judge("v3", total["v4"]),
        "probe under the floor allow-list": judge("floor", pc),
        "probe under the v3 allow-list": judge("v3", pc),
    }
    for name, bad in controls.items():
        print(f"control, {name}: {sum(bad.values())} instructions outside, sets "
              f"{sorted({k[0] for k in bad})}")
        rc |= not bad
    print(f"verdict: {'PASS' if not rc else 'FAIL'}")
    sys.exit(rc)


main()
