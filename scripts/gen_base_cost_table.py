#!/usr/bin/env python3
"""
Generate include/admiral/detail/base_cost_table.hpp from bench-results/base_cost_*.txt.

Usage (from repo root):
    python3 scripts/gen_base_cost_table.py

Reads v4/v3/v2 receipt files.  For each (ISA, precision, N) triple picks the
winning form by min-cyc, guarded by a 3% stability margin: if the winner does
not beat the runner-up by >= 3%, the cell falls back to the incumbent routing
(what plan.hpp routes today).  Emits a checked-in generated C++ header.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILES = {
    'v4': REPO_ROOT / 'bench-results' / 'base_cost_v4.txt',
    'v3': REPO_ROOT / 'bench-results' / 'base_cost_v3.txt',
    'v2': REPO_ROOT / 'bench-results' / 'base_cost_v2.txt',
}
OUTPUT = REPO_ROOT / 'include' / 'admiral' / 'detail' / 'base_cost_table.hpp'

# ---------------------------------------------------------------------------
# ISA -> list of (prec, W) pairs measured in that file
# ---------------------------------------------------------------------------
ISA_PREC_W = {
    'v4': [('f32', 16), ('f64', 8)],
    'v3': [('f32', 8),  ('f64', 4)],
    'v2': [('f32', 4),  ('f64', 2)],
}

# ---------------------------------------------------------------------------
# Incumbent map — mirrors plan.hpp select_route() priority ladder exactly.
#
# Good-Thomas PFA takes priority, available only on the v4 ISA (good_thomas_eligible<T,...>
# requires enough SIMD registers; W=8/4/2 variants fail that gate and are
# therefore never routed to good_thomas).  Within v4:
#   f32 W=16: sizes {10, 15, 20, 24, 30, 40, 60}  (10 is f32-only gate)
#   f64 W=8:  sizes {12, 15, 20, 24, 30, 40, 60}  (12 is f64-only gate)
#
# dif_beats_codelet fires next (both precisions, all ISAs unless good_thomas already
# won):
#   both prec: {22, 24, 40, 42, 48, 56, 60}
#   f64 only:  {9, 15, 21, 25, 33, 35, 49, 50, 55, 63}
#   f32 only:  {10, 14}
#
# Everything else: codelet.
# ---------------------------------------------------------------------------
_GT_F32 = frozenset({10, 15, 20, 24, 30, 40, 60})
_GT_F64 = frozenset({12, 15, 20, 24, 30, 40, 60})
_DIF_BOTH = frozenset({22, 24, 40, 42, 48, 56, 60})
_DIF_F64  = frozenset({9, 15, 21, 25, 33, 35, 49, 50, 55, 63})
_DIF_F32  = frozenset({10, 14})


def incumbent_form(isa: str, prec: str, n: int) -> str:
    """Return the form that plan.hpp currently routes (isa, prec, n) to."""
    if n > 64:
        # Catalog extras (FFT_CODELET_EXTRA_SIZES, e.g. 120): the pre-extension
        # route for these 11-smooth composites was iterative_dif.
        return 'iterative_dif'
    if isa == 'v4':
        if prec == 'f32' and n in _GT_F32:
            return 'good_thomas'
        if prec == 'f64' and n in _GT_F64:
            return 'good_thomas'
    if n in _DIF_BOTH:
        return 'iterative_dif'
    if prec == 'f64' and n in _DIF_F64:
        return 'iterative_dif'
    if prec == 'f32' and n in _DIF_F32:
        return 'iterative_dif'
    return 'codelet'


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_LINE_RE = re.compile(
    r'^BASECOST\s+size=\s*(\d+)\s+prec=(\w+)\s+form=(\w+)\s+cyc=\s*([\d.]+)'
)


def parse_file(path: Path) -> dict:
    """Return {(prec, n): {form: cyc}}.  VERIFY-FAIL lines are skipped."""
    data: dict = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            m = _LINE_RE.match(line.strip())
            if m:
                n    = int(m.group(1))
                prec = m.group(2)
                form = m.group(3)
                if form == 'pfa':  # pre-rename receipts
                    form = 'good_thomas'
                cyc  = float(m.group(4))
                data[(prec, n)][form] = cyc
    return data


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------
MARGIN = 0.03  # winner must beat runner-up by at least this fraction


def pick_winner(forms_cyc: dict, inc: str):
    """
    Given {form: measured_cyc} and the incumbent form name, return
    (form, cyc) for the table entry.

    Stability rule: winner (min cyc) must satisfy
        winner_cyc * (1 + MARGIN) <= runner_up_cyc
    otherwise fall back to the incumbent form (using its measured cyc).
    If incumbent form was not measured, use the raw min as a last resort.
    """
    if not forms_cyc:
        return None, None
    ranked = sorted(forms_cyc.items(), key=lambda kv: kv[1])
    best_form, best_cyc = ranked[0]
    if len(ranked) == 1:
        return best_form, best_cyc
    _runner_up_form, runner_up_cyc = ranked[1]
    if best_cyc * (1.0 + MARGIN) <= runner_up_cyc:
        return best_form, best_cyc
    # Margin not met: fall back to incumbent.
    if inc in forms_cyc:
        return inc, forms_cyc[inc]
    # Incumbent not measured (shouldn't happen for valid cells): use raw min.
    return best_form, best_cyc


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------
MAX_N = 64  # table covers indices 0..64 (65 entries)
# Sparse catalog extras beyond the dense 2..64 range (FFT_CODELET_EXTRA_SIZES).
# Emitted as explicit `if (n == ...)` returns ahead of the array lookup.
EXTRA_NS = (120,)


def build_extras(isa: str, prec: str, file_data: dict) -> dict:
    """Return {n: (form, cyc)} for measured EXTRA_NS cells (unmeasured omitted)."""
    extras = {}
    for n in EXTRA_NS:
        forms_cyc = file_data.get((prec, n), {})
        form, cyc = pick_winner(forms_cyc, incumbent_form(isa, prec, n))
        if form is not None:
            extras[n] = (form, cyc)
    return extras


def build_table(isa: str, prec: str, file_data: dict) -> list:
    """
    Return list of 65 (form, cyc|None) pairs indexed by n (0..64).
    n=0 is always no-entry.
    """
    table = [(None, None)] * (MAX_N + 1)
    for n in range(1, MAX_N + 1):
        forms_cyc = file_data.get((prec, n), {})
        inc = incumbent_form(isa, prec, n)
        form, cyc = pick_winner(forms_cyc, inc)
        table[n] = (form, cyc)
    return table


# ---------------------------------------------------------------------------
# C++ code generation helpers
# ---------------------------------------------------------------------------
def cpp_form(form: str | None) -> str:
    if form == 'iterative_dif':
        return 'base_form::iterative_dif'
    if form == 'good_thomas':
        return 'base_form::good_thomas'
    return 'base_form::codelet'


def cpp_cyc(cyc: float | None) -> str:
    if cyc is None:
        return '-1.f'
    return f'{cyc:.1f}f'


def emit_table_array(name: str, table: list) -> list[str]:
    """Emit a 'constexpr std::array<base_cost_entry, 65>' initializer.

    Plain constexpr (not static): base_cost_for is a constexpr function and a
    static local in one is ill-formed before C++23; this project is C++20.
    """
    lines = [f'        constexpr std::array<base_cost_entry, 65> {name} = {{{{']
    for n, (form, cyc) in enumerate(table):
        comma = ',' if n < MAX_N else ''
        comment = f'  // n={n}' if n > 0 else '  // n=0 (unused)'
        lines.append(
            f'            {{{cpp_cyc(cyc)}, {cpp_form(form)}}}{comma}{comment}'
        )
    lines.append('        }};')
    return lines


# ---------------------------------------------------------------------------
# Routing diff (for the report)
# ---------------------------------------------------------------------------
def compute_diff(all_tables: dict) -> list[str]:
    """
    For each (W, prec, N) where the new table's form differs from the
    incumbent, emit one line:
        W=8 f64 N=64: codelet -> iterative_dif (cyc 314.0 -> 199.0)
    """
    diffs = []
    isa_w = {'v4': {'f32': 16, 'f64': 8},
              'v3': {'f32': 8,  'f64': 4},
              'v2': {'f32': 4,  'f64': 2}}
    for isa in ('v4', 'v3', 'v2'):
        file_data = all_tables[isa]
        for prec, w in ISA_PREC_W[isa]:
            for n in list(range(2, MAX_N + 1)) + list(EXTRA_NS):
                forms_cyc = file_data.get((prec, n), {})
                inc = incumbent_form(isa, prec, n)
                new_form, new_cyc = pick_winner(forms_cyc, inc)
                if new_form is None:
                    continue
                if new_form != inc:
                    inc_cyc = forms_cyc.get(inc)
                    inc_cyc_str = f'{inc_cyc:.0f}' if inc_cyc is not None else '?'
                    new_cyc_str = f'{new_cyc:.0f}' if new_cyc is not None else '?'
                    diffs.append(
                        f'W={w} {prec} N={n}: {inc} -> {new_form}'
                        f' (cyc {inc_cyc_str} -> {new_cyc_str})'
                    )
    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Parse all three ISA files.
    all_data = {}
    for isa, path in INPUT_FILES.items():
        if not path.exists():
            print(f'ERROR: input file not found: {path}', file=sys.stderr)
            sys.exit(1)
        all_data[isa] = parse_file(path)

    # Assert v2 (SSE) has no good_thomas entries — GT is not validated there.
    # v3 (AVX2) now carries good_thomas rows for the small-N band (see
    # good_thomas_eligible); the winner logic + 3% margin route it only where it wins.
    for isa in ('v2',):
        for (prec, n), forms_cyc in all_data[isa].items():
            assert 'good_thomas' not in forms_cyc, (
                f'ASSERT FAIL: {isa} {prec} N={n} has a good_thomas row — '
                f'expected only codelet/iterative_dif'
            )
    print('Assertion passed: v2 contains no good_thomas rows.')

    # Build tables.
    tables: dict = {}
    extras: dict = {}
    for isa in ('v4', 'v3', 'v2'):
        tables[isa] = {}
        extras[isa] = {}
        for prec, _w in ISA_PREC_W[isa]:
            tables[isa][prec] = build_table(isa, prec, all_data[isa])
            extras[isa][prec] = build_extras(isa, prec, all_data[isa])

    # Compute and print routing diff.
    diffs = compute_diff(all_data)
    if diffs:
        print('\nRouting decisions that differ from the incumbent:')
        for d in diffs:
            print(f'  {d}')
    else:
        print('\nNo routing changes vs. incumbent.')

    # Generate C++ header.
    lines = [
        '// GENERATED by scripts/gen_base_cost_table.py from bench-results/base_cost_*.txt',
        '// do not edit by hand; rerun the script to refresh.',
        '#pragma once',
        '',
        '#include <array>',
        '#include <cstddef>',
        '#include <cstdint>',
        '#include <xsimd/xsimd.hpp>',
        '',
        'namespace admiral::detail {',
        '',
        'enum class base_form : std::uint8_t { codelet, iterative_dif, good_thomas };',
        '',
        'struct base_cost_entry {',
        '    float cyc;',
        '    base_form form;',
        '};  // cyc<0 = no entry',
        '',
        '// base_cost_for<T>(n): measured cycle count and winning form for transform',
        '// size n (2..64, plus catalog extras like 120) on the current ISA.  Returns',
        '// cyc<0 for unmeasured cells (n==0, unmeasured n>64, or sizes where only the',
        '// force-route path was tried).',
        '//',
        '// Source receipts: bench-results/base_cost_{v4,v3,v2}.txt',
        '// (see also bench-results/pfa_smalln_ab_raw.txt for earlier PFA A/B data).',
        '// Winner rule: min-cyc form; margin: runner-up must cost >=3% more,',
        '// otherwise cell falls back to the plan.hpp incumbent routing.',
        'template<typename T>',
        '[[nodiscard]] constexpr base_cost_entry base_cost_for(std::size_t n) noexcept {',
        '    constexpr base_cost_entry NO_ENTRY{-1.f, base_form::codelet};',
        '    if (n == 0) return NO_ENTRY;',
    ]

    # Emit each ISA block as an if constexpr chain.
    # Order: v4-f32, v4-f64, v3-f32, v3-f64, v2-f32, v2-f64.
    configs = [
        ('v4', 'f32', 16, 4,  'if'),
        ('v4', 'f64',  8, 8, 'else if'),
        ('v3', 'f32',  8, 4, 'else if'),
        ('v3', 'f64',  4, 8, 'else if'),
        ('v2', 'f32',  4, 4, 'else if'),
        ('v2', 'f64',  2, 8, 'else if'),
    ]

    for isa, prec, w, sizeof_t, kw in configs:
        tab = tables[isa][prec]
        tab_name = f'tab_{isa}_{prec}'
        lines.append(
            f'    {kw} constexpr (sizeof(T) == {sizeof_t} && '
            f'xsimd::batch<T>::size == {w}) {{'
        )
        lines.append(f'        // {isa} {prec} W={w}')
        for n, (form, cyc) in sorted(extras[isa][prec].items()):
            lines.append(
                f'        if (n == {n}) return {{{cpp_cyc(cyc)}, {cpp_form(form)}}};'
                f'  // catalog extra'
            )
        lines.append(f'        if (n > {MAX_N}) return NO_ENTRY;')
        lines += emit_table_array(tab_name, tab)
        lines.append(f'        return {tab_name}[n];')
        lines.append('    }')

    lines += [
        '    return NO_ENTRY;',
        '}',
        '',
        '}  // namespace admiral::detail',
        '',
    ]

    OUTPUT.write_text('\n'.join(lines))
    print(f'\nWrote {OUTPUT}')


if __name__ == '__main__':
    main()
