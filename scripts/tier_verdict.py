"""D1 tier verdict. Reads the receipts of one wave and applies the tier rule of the dispatch
spec: a candidate tier ships only when two sets per host agree, and one set ships it only when
the pooled geomean of T_lower / T_candidate over each (host, cell) is >= 1.10 and every host's
own geomean is >= 0.97. A receipt is void when its identity control leaves [0.97, 1.03] or its
calibration control leaves [1.07, 1.13].

usage: python3 -I tier_verdict.py --hosts=H1,H2 DIR [--print] | python3 -I tier_verdict.py --self-test

Receipt files, one per host and set (set is 1 or 2), <host>.<set>.ab.json:
    {"pairs": {"<lower>,<candidate>": {"<cell>": {"A": us_of_lower, "B": us_of_candidate}}}}
plus <host>.<set>.identity.json and <host>.<set>.calib.json, each {"ratio": r}.
'./tier_verdict.py --hosts=spr,rome receipts/spr.1.ab.json spr.1.identity.json spr.1.calib.json ...'
is the wave one set at a time; candidates are v3 over v1, then v4 over the highest shipped tier
below it, on the AVX-512 hosts (spr, icelake, genoa, cascadelake); then v3 over v1 again on the
hosts without AVX-512.

Exit 0: every candidate ships. Exit 1: a candidate fails on two agreeing sets, or the two sets
disagree. Exit 2: a void receipt, a missing host, set or pair, or an empty directory.
--self-test runs the committed fixtures first on every invocation; a fixture regression exits 2.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

AVX512_HOSTS = ("spr", "icelake", "genoa", "cascadelake")
SHIP_GEOMEAN = 1.10
HOST_TIE = 0.97
IDENTITY_BAND = (0.97, 1.03)
CALIB_BAND = (1.07, 1.13)


def geomean(vals):
    vals = list(vals)
    p = 1.0
    for v in vals:
        p *= v
    return p ** (1.0 / len(vals))


class Receipt(Exception):
    pass


def load(hosts, sets, d):
    """returns {(host, set): {"pairs": {pair: {cell: ratio}}, "identity": r, "calib": r}}"""
    out = {}
    for h in hosts:
        for s in sets:
            ab = Path(d, f"{h}.{s}.ab.json")
            rid = Path(d, f"{h}.{s}.identity.json")
            ca = Path(d, f"{h}.{s}.calib.json")
            for f in (ab, rid, ca):
                if not f.exists():
                    raise Receipt(f"missing {f}")
            pairs = {}
            for pname, cells in json.load(open(ab))["pairs"].items():
                if not cells:
                    raise Receipt(f"{ab}: pair {pname} has no cells")
                pairs[pname] = {c: v["A"] / v["B"] for c, v in cells.items()}
            identity = json.load(open(rid))["ratio"]
            calib = json.load(open(ca))["ratio"]
            if not IDENTITY_BAND[0] <= identity <= IDENTITY_BAND[1]:
                raise Receipt(f"{rid}: identity {identity} outside {IDENTITY_BAND}, void receipt")
            if not CALIB_BAND[0] <= calib <= CALIB_BAND[1]:
                raise Receipt(f"{ca}: calibration {calib} outside {CALIB_BAND}, void receipt")
            out[h, s] = {"pairs": pairs, "identity": identity, "calib": calib}
    return out


def set_verdict(data, hosts, s, pair):
    """one set over the given hosts: (ship?, pooled, {host: geomean})"""
    ratios = {}
    for h in hosts:
        try:
            ratios[h] = data[h, s]["pairs"][pair]
        except KeyError:
            raise Receipt(f"missing pair {pair} in host {h} set {s}")
    pooled = geomean(r for cells in ratios.values() for r in cells.values())
    per_host = {h: geomean(cells.values()) for h, cells in ratios.items()}
    ship = pooled >= SHIP_GEOMEAN and all(g >= HOST_TIE for g in per_host.values())
    return ship, pooled, per_host


def evaluate(data, hosts, sets, out):
    """returns the failing candidate (message) or None; 'v3' and 'v4' record what shipped."""
    shipped = {}
    candidates = [("v3", hosts, "v1,v3")]
    avx512_hosts = [h for h in hosts if h in AVX512_HOSTS]
    if avx512_hosts:
        candidates.append(("v4", avx512_hosts, None))
    else:
        out.append("no AVX-512 host in the wave: v4 is not evaluated")
    for cand, lower_hosts, pair in candidates:
        if cand == "v4":
            lower = "v3" if shipped.get("v3") else "v1"
            pair = f"{lower},v4"
        votes = []
        for s in sets:
            ship, pooled, per_host = set_verdict(data, lower_hosts, s, pair)
            votes.append(ship)
            out.append(f"set {s}: {pair} over {','.join(lower_hosts)}: pooled {pooled:.3f}, "
                       f"hosts {{{', '.join(f'{h} {g:.3f}' for h, g in per_host.items())}}} "
                       f"-> {'ship' if ship else 'no ship'}")
        if all(votes):
            shipped[cand] = True
            continue
        hosts_named = ",".join(lower_hosts)
        if len(set(votes)) > 1:
            return f"{cand}: the sets disagree ({votes}) over {hosts_named}"
        if cand == "v4":
            bad_hosts = [h for h in lower_hosts
                         if any(not set_verdict(data, [h], s, pair)[0] for s in sets)]
            if bad_hosts == ["cascadelake"]:
                return (f"{cand}: clause 2 fails only on cascadelake; step 4 of the tier rule "
                        f"is a user decision: drop v4, or gate it on xsimd::avx512vbmi "
                        f"(open decision 13.7)")
            return f"{cand}: no ship over {hosts_named}"
        return f"{cand}: no ship over {hosts_named}"
    if shipped.get("v3") and shipped.get("v4"):
        rest = [h for h in hosts if h not in AVX512_HOSTS]
        votes = []
        for s in sets:
            ship, pooled, per_host = set_verdict(data, rest, s, "v1,v3")
            votes.append(ship)
            out.append(f"set {s}: step 3, v1,v3 over {','.join(rest)} (hosts without AVX-512): "
                       f"pooled {pooled:.3f} -> {'ship' if ship else 'no ship'}")
        if len(set(votes)) > 1:
            return f"v3: step 3, the sets disagree ({votes})"
        if not all(votes):
            return "v3: fails step 3 (hosts without AVX-512); drop v3 and re-judge v4 over v1"
        out.append("both tiers ship: only CPUs without AVX-512 run v3")
    return None


def self_test():
    def write(d, host, s, pairs, identity=1.0, calib=1.1):
        json.dump({"pairs": {p: {c: {"A": r, "B": 1.0} for c in ("c1", "c2")}
                             for p, r in pairs.items()}},
                  open(Path(d, f"{host}.{s}.ab.json"), "w"))
        json.dump({"ratio": identity}, open(Path(d, f"{host}.{s}.identity.json"), "w"))
        json.dump({"ratio": calib}, open(Path(d, f"{host}.{s}.calib.json"), "w"))

    def run(hosts=("rome",), sets=(1, 2), build=None, expect=None, sets_written=(1, 2),
            name=""):
        with tempfile.TemporaryDirectory() as d:
            for h in hosts:
                for s in sets_written:
                    write(d, h, s, build(h) if build else {"v1,v3": 1.2})
            rc, _ = verdict(hosts, sets, d, [])
            ok = (rc == expect) if expect is not None else True
            print(f"self-test {name}: exit {rc}, expected {expect} -> {'ok' if ok else 'VOID'}")
            return ok

    ok = True
    ok &= run(build=lambda h: {"v1,v3": 1.099}, expect=1, name="pooled 1.099 gives 1")
    ok &= run(build=lambda h: {"v1,v3": 1.101}, expect=0, name="pooled 1.101 gives 0")
    ok &= run(hosts=("rome", "rome2"),
              build=lambda h: {"v1,v3": 0.969 if h == "rome" else 1.4},
              expect=1, name="one host at 0.969 gives 1")
    ok &= run(hosts=("rome", "rome2"),
              build=lambda h: {"v1,v3": 0.971 if h == "rome" else 1.4},
              expect=0, name="one host at 0.971 gives 0")

    def with_control(host, ident=None, calib=None):
        with tempfile.TemporaryDirectory() as d:
            for s in (1, 2):
                write(d, host, s, {"v1,v3": 1.2}, identity=ident or 1.0, calib=calib or 1.1)
            rc, _ = verdict((host,), (1, 2), d, [])
            return rc
    rc = with_control("rome", ident=1.031)
    print(f"self-test identity 1.031 gives 2: exit {rc} -> {'ok' if rc == 2 else 'VOID'}")
    ok &= rc == 2
    rc = with_control("rome", calib=1.069)
    print(f"self-test calibration 1.069 gives 2: exit {rc} -> {'ok' if rc == 2 else 'VOID'}")
    ok &= rc == 2

    # v3 clears step 1 (the wave's pooled geomean clears 1.10) and fails step 3: the hosts
    # without AVX-512, judged alone, do not clear 1.10.
    with tempfile.TemporaryDirectory() as d:
        for s in (1, 2):
            write(d, "rome", s, {"v1,v3": 1.05})
            write(d, "spr", s, {"v1,v3": 1.2, "v3,v4": 1.2})
        rc, out = verdict(("rome", "spr"), (1, 2), d, [])
        good = rc == 1 and any("step 3" in l for l in out)
        print(f"self-test v3 fails step 3 gives 1: exit {rc} -> {'ok' if good else 'VOID'}")
        ok &= good
    # v4 fails clause 2 only on cascadelake: exit 1, message names step 4.
    with tempfile.TemporaryDirectory() as d:
        for s in (1, 2):
            write(d, "spr", s, {"v1,v3": 1.2, "v3,v4": 1.2})
            write(d, "cascadelake", s, {"v1,v3": 1.2, "v3,v4": 0.96})
        rc, out = verdict(("spr", "cascadelake"), (1, 2), d, [])
        good = rc == 1 and any("step 4" in l for l in out)
        print(f"self-test v4 fails only on cascadelake gives 1 and names step 4: exit {rc} -> "
              f"{'ok' if good else 'VOID'}")
        ok &= good
    # disagreeing sets: set 1 ships, set 2 does not.
    with tempfile.TemporaryDirectory() as d:
        write(d, "rome", 1, {"v1,v3": 1.2})
        write(d, "rome", 2, {"v1,v3": 1.0})
        rc, _ = verdict(("rome",), (1, 2), d, [])
        good = rc == 1
        print(f"self-test disagreeing sets give 1: exit {rc} -> {'ok' if good else 'VOID'}")
        ok &= good
    # one set per host: set 2 missing for rome -> void.
    with tempfile.TemporaryDirectory() as d:
        write(d, "rome", 1, {"v1,v3": 1.2})
        rc, _ = verdict(("rome",), (1, 2), d, [])
        good = rc == 2
        print(f"self-test one set per host gives 2: exit {rc} -> {'ok' if good else 'VOID'}")
        ok &= good
    # a missing host -> void.
    with tempfile.TemporaryDirectory() as d:
        write(d, "rome", 1, {"v1,v3": 1.2})
        write(d, "rome", 2, {"v1,v3": 1.2})
        rc, _ = verdict(("rome", "spr"), (1, 2), d, [])
        good = rc == 2
        print(f"self-test a missing host gives 2: exit {rc} -> {'ok' if good else 'VOID'}")
        ok &= good
    # an empty directory -> void.
    with tempfile.TemporaryDirectory() as d:
        rc, _ = verdict(("rome",), (1, 2), d, [])
        good = rc == 2
        print(f"self-test an empty directory gives 2: exit {rc} -> {'ok' if good else 'VOID'}")
        ok &= good
    return ok


def verdict(hosts, sets, d, out):
    try:
        data = load(hosts, sets, d)
    except Receipt as e:
        out.append(f"void: {e}")
        return 2, out
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        out.append(f"void: malformed receipt: {e}")
        return 2, out
    msg = None
    try:
        msg = evaluate(data, hosts, sets, out)
    except Receipt as e:
        out.append(f"void: {e}")
        return 2, out
    if msg is None:
        return 0, out
    out.append(msg)
    return 1, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", default="spr")
    ap.add_argument("--sets", default="1,2")
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    ok = self_test()
    if args.self_test:
        sys.exit(0 if ok else 2)
    if not ok:
        print("self-test failed; the verdict cannot be trusted", file=sys.stderr)
        sys.exit(2)
    if not args.dir:
        ap.error("give the receipt directory")
    hosts = tuple(args.hosts.split(","))
    sets = tuple(args.sets.split(","))
    out = []
    rc, out = verdict(hosts, sets, args.dir, out)
    print("\n".join(out))
    print(f"verdict: {'SHIP' if rc == 0 else 'NO SHIP' if rc == 1 else 'VOID'}")
    sys.exit(rc)


main()
