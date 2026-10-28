"""Count the ctest tests that reach each tier (F6). Workstation receipt: needs Intel PMU access.

usage: python3 -I tier_coverage.py BUILD TESTS_JSON
A test reaches tier v3 or v4 when it retires 256- or 512-bit packed FP operations under
ADM_TEST_TIER=v3 or v4. The floor is built at x86-64, so under
ADM_TEST_TIER=v1 no test may retire either: that run is the control, and the zero-count rule
must fire on it. The misfire fixture (review F1) binds the v1 slot to the v3 table through
ADM_TEST_TIER_BIND: that doctored run must retire 256-bit ops, proving the v1 zero rule can
fire. Exit 1 when the v3 or v4 count is 0, when the v1 control is not 0, when the doctored run
retires nothing, or when a test fails by ctest's rule (WILL_FAIL, SKIP_RETURN_CODE).
Events: FP_ARITH_INST_RETIRED umasks 0x10/0x20 (256-bit PD/PS), 0x40/0x80 (512-bit).
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys

EV = {"256": ("r10c7", "r20c7"), "512": ("r40c7", "r80c7")}


def prop(test, name, default=None):
    return next((p["value"] for p in test.get("properties", []) if p["name"] == name), default)


def passed(test, rc):
    """ctest's own rule: SKIP_RETURN_CODE is a skip, WILL_FAIL inverts the exit status."""
    if rc == prop(test, "SKIP_RETURN_CODE"):
        return True
    return (rc != 0) == bool(prop(test, "WILL_FAIL", False))


def run(test, tier, bind=None):
    env = dict(os.environ, ADM_TEST_TIER=tier)
    if bind:
        env["ADM_TEST_TIER_BIND"] = bind
    cwd = prop(test, "WORKING_DIRECTORY")
    ev = ",".join(EV["256"] + EV["512"])
    p = subprocess.run(["perf", "stat", "-x,", "-e", ev, "--"] + test["command"], cwd=cwd, env=env,
                       capture_output=True, text=True)
    c = {}
    for line in p.stderr.splitlines():
        f = line.split(",")
        if len(f) > 2 and f[2] in EV["256"] + EV["512"]:
            c[f[2]] = int(f[0]) if f[0].isdigit() else -1
    w = {k: sum(c.get(e, -1) for e in v) for k, v in EV.items()}
    return test["name"], passed(test, p.returncode), w


def main():
    tests = json.load(open(sys.argv[2]))["tests"]
    rc, res = 0, {}
    for tier in ("v1", "v3", "v4"):
        with cf.ThreadPoolExecutor(8) as ex:
            res[tier] = list(ex.map(lambda t: run(t, tier), tests))
        failed = [n for n, ok, _ in res[tier] if not ok]
        unread = [n for n, _, w in res[tier] if min(w.values()) < 0]
        n256 = sum(w["256"] > 0 for _, _, w in res[tier])
        n512 = sum(w["512"] > 0 for _, _, w in res[tier])
        nwide = sum(w["256"] + w["512"] > 0 for _, _, w in res[tier])
        print(f"ADM_TEST_TIER={tier}: {len(tests)} tests, failed {len(failed)}, unread counters "
              f"{len(unread)}; tests with 256-bit ops {n256}, with 512-bit ops {n512}")
        print(f"  failed: {failed}")
        rc |= bool(failed) or bool(unread)
        if tier == "v1":
            ok = n256 == 0 and n512 == 0
            print(f"  control: the floor retires no 256- or 512-bit op in any test: {ok}; the "
                  f"zero-count rule fires on this run: {nwide == 0}")
            rc |= not ok
        else:
            n = nwide
            print(f"  tests that reach {tier} (256- or 512-bit ops > 0): {n} of {len(tests)}")
            rc |= n == 0
    reach = {t: {n for n, _, w in res[t] if w["256"] + w["512"] > 0} for t in ("v3", "v4")}
    print(f"tests that reach v3 and not v4: {len(reach['v3'] - reach['v4'])}; v4 and not v3: "
          f"{len(reach['v4'] - reach['v3'])}")
    by_exe = {}
    for n, _, w in res["v4"]:
        exe = next(t["command"][0] for t in tests if t["name"] == n).rsplit("/", 1)[1]
        by_exe.setdefault(exe, [0, 0])
        by_exe[exe][0] += 1
        by_exe[exe][1] += w["256"] + w["512"] > 0
    print("per test binary, v4 (tests, tests that reach v4): " +
          ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by_exe.items())))
    # The misfire fixture of the v1 zero rule: one binary that reaches v3, run with
    # ADM_TEST_TIER=v1 but the slot bound to the v3 table. It must retire 256-bit ops, or the
    # zero rule above cannot have fired.
    fixture = next((t for t in tests
                    if t["name"] in {n for n, _, w in res["v3"] if w["256"] + w["512"] > 0}), None)
    if fixture is None:
        print("misfire fixture: no test reaches v3, cannot bind the v1 slot to the v3 table")
        rc = 1
    else:
        name, _, w = run(fixture, "v1", bind="v3")
        ok = w["256"] + w["512"] > 0
        print(f"misfire fixture: {name} under ADM_TEST_TIER=v1 ADM_TEST_TIER_BIND=v3 retires "
              f"256/512-bit ops {w['256'] + w['512']} (> 0 required): {'fires' if ok else 'VOID'}")
        rc |= not ok
    print(f"verdict: {'PASS' if not rc else 'FAIL'}")
    sys.exit(rc)


main()
