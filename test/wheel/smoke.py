"""Smoke test for the installed wheel: one C API c2c transform through ctypes."""
import cmath
import ctypes
import math
import importlib.metadata as md
import sys

TOL = 1e-12  # max |got - ref| / max |ref|; the double engine lands near 1e-15


def find_library():
    suffixes = (".so", ".dylib", ".dll")
    for f in md.distribution("admiral").files:
        if f.name.startswith(("libadmiral_c.", "admiral_c.")) and f.name.endswith(suffixes):
            return str(f.locate())
    sys.exit("no admiral_c shared library in the installed wheel")


def naive_dft(x):
    n = len(x)
    return [sum(x[j] * cmath.exp(-2j * cmath.pi * j * k / n) for j in range(n)) for k in range(n)]


def rel_err(got, ref):
    return max(abs(g - r) for g, r in zip(got, ref)) / max(abs(r) for r in ref)


def transform(lib, x):
    buf = (ctypes.c_double * (2 * len(x)))(*[v for z in x for v in (z.real, z.imag)])
    status = lib.adm_forward(buf, len(x), None)
    assert status == 0, f"adm_forward status {status}"
    return [complex(buf[2 * i], buf[2 * i + 1]) for i in range(len(x))]


def main():
    path = find_library()
    lib = ctypes.CDLL(path)
    lib.adm_forward.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.adm_forward.restype = ctypes.c_int
    print(f"loaded {path}")
    for n in (64, 60, 97):  # power of two, mixed radix, prime
        x = [complex(math.sin(0.37 * j + 1.0), math.cos(1.3 * j * j % 7.0)) for j in range(n)]
        ref = naive_dft(x)
        err = rel_err(transform(lib, x), ref)
        assert err < TOL, f"n={n}: rel err {err:.3e} >= {TOL:.0e}"
        print(f"n={n}: adm_forward matches the naive DFT, rel err {err:.3e} < {TOL:.0e}")
        # Positive control: the same check must fail on a perturbed reference.
        bad = list(ref)
        bad[n // 3] *= 1 + 1e-6
        err_bad = rel_err(transform(lib, x), bad)
        assert err_bad >= TOL, f"n={n}: control did not fail, rel err {err_bad:.3e}"
        print(f"n={n}: control with ref bin {n // 3} perturbed by 1e-6 fails as required, rel err {err_bad:.3e}")


if __name__ == "__main__":
    main()
