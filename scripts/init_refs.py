"""D2 (h), test side. Print each ELF object whose init code names an admiral symbol.
Init code: the functions in .text.startup other than main (gcc and clang put each
_GLOBAL__sub_I_ and __cxx_global_var_init function there; gcc also puts main there).
usage: python3 -I init_refs.py OBJ...   Exit 1 when an object other than tier_select.cpp.o has init
code that names an admiral symbol. Exit 2 when no object has init code (a broken census)."""
import subprocess, sys
seen, bad, allowed = 0, 0, 0
for o in sys.argv[1:]:
    run = lambda *a: subprocess.run(["readelf", "-W", *a, o], capture_output=True, text=True, check=True).stdout
    sec = {}
    for l in run("-S").splitlines():
        h, _, rest = l.strip().partition("]")
        if h.startswith("[") and h[1:].strip().isdigit() and rest.split():
            sec[rest.split()[0]] = int(h[1:])
    idx = sec.get(".text.startup")
    if idx is None:
        continue
    ranges = []
    for l in run("-s").splitlines():
        t = l.split()
        if len(t) >= 8 and t[3] == "FUNC" and t[6] == str(idx) and t[7] != "main":
            ranges.append((int(t[1], 16), int(t[1], 16) + int(t[2], 0)))
    seen += bool(ranges)
    names, on = set(), False
    for l in run("-r").splitlines():
        if l.startswith("Relocation section"):
            on = "'.rela.text.startup'" in l
        elif on and "admiral" in l:
            off = int(l.split()[0], 16)
            if any(a <= off < b for a, b in ranges):
                names.add(l.split()[4])
    if names and o.endswith("/tier_select.cpp.o"):
        allowed += 1
    elif names:
        bad += 1
        print("init code names admiral:", o, " ".join(sorted(names))[:300])
print(f"objects: {len(sys.argv) - 1}; with init code: {seen}; tier_select.cpp.o with admiral refs "
      f"(allowed): {allowed}; other objects with admiral refs: {bad}")
sys.exit(2 if seen == 0 else int(bad > 0))
