#!/usr/bin/env python3
"""Fail when a table row of docs/cpp-api.md drifts from the public headers.

Every table row whose first cell is a code span is checked. A span with '(' is a
signature and must match a declaration of include/admiral/admiral.hpp. Any other span
is a name and must occur in admiral.hpp or errors.hpp. Parameter names, defaults,
std::/detail:: qualification, ADM_API and the type_identity_t indirection are
normalized away. Usage: docs_api_check.py [tree-root]; the default root is the
repository that holds this script.
"""

import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
admiral_h = (root / "include/admiral/admiral.hpp").read_text()
errors_h = (root / "include/admiral/errors.hpp").read_text()
doc = (root / "docs/cpp-api.md").read_text()


def unwrap_type_identity(text):
    # detail::type_identity_t<X> -> X, with balanced angles.
    key = "detail::type_identity_t<"
    out, i = [], 0
    while True:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        k = j + len(key)
        depth = 1
        while depth:
            depth += (text[k] == "<") - (text[k] == ">")
            k += 1
        out.append(text[j + len(key):k - 1])
        i = k


def normalize(text):
    text = unwrap_type_identity(text)
    text = text.replace("std::", "").replace("detail::", "")
    return re.sub(r"\s+", " ", text)


def split_params(s):
    parts, depth, cur = [], 0, []
    for ch in s:
        depth += (ch in "<(") - (ch in ">)")
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def param_type(p):
    p = p.split(" = ", 1)[0]  # drop the default
    tokens = p.split(" ")
    if len(tokens) > 1 and re.fullmatch(r"[a-z_][a-z0-9_]*", tokens[-1]):
        tokens = tokens[:-1]  # drop the parameter name
    return "".join(tokens)


def sig_key(sig):
    name, _, rest = sig.partition("(")
    params = rest.rsplit(")", 1)[0]
    return name.strip() + "(" + ",".join(param_type(p) for p in split_params(params)) + ")"


# Candidate keys from every `name(...)` in the header, declarations and calls alike. A doc row
# needs one match, so call sites and delegating constructors are harmless extra entries.
def header_keys(header_text):
    text = normalize(header_text)
    keys = set()
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = m.group(1)
        depth, k = 1, m.end()
        while depth and k < len(text):
            depth += (text[k] == "(") - (text[k] == ")")
            k += 1
        keys.add(name + "(" + ",".join(param_type(p)
                                       for p in split_params(text[m.end():k - 1])) + ")")
    return keys


keys = header_keys(admiral_h)
names_hay = normalize(admiral_h + errors_h)

fail, checked = [], 0
for line in doc.splitlines():
    if not line.lstrip().startswith("|"):
        continue
    m = re.fullmatch(r"`([^`]+)`", line.split("|")[1].strip())
    if not m:
        continue
    span = m.group(1)
    checked += 1
    if "(" in span:
        if sig_key(normalize(span)) not in keys:
            fail.append(f"drifted signature `{span}`")
    else:
        name = re.sub(r"<.*>", "", span)
        if not re.search(rf"\b{re.escape(name)}\b", names_hay):
            fail.append(f"drifted name `{name}`")

for f in fail:
    print("FAIL:", f)
print(f"{'FAIL' if fail else 'PASS'}: {checked} rows checked, {len(fail)} drifted")
sys.exit(1 if fail else 0)
