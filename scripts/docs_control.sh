#!/usr/bin/env bash
# Guards of the docs build. Control 1: the cpp-api.md tables track the public headers, and a
# drifted row must fail docs_api_check.py. Control 2: sphinx -W must stop the build and name a
# broken literalinclude marker, so an include cannot fail silently. SPHINX_BUILD replaces
# sphinx-build in control 2, to check that this script rejects a build that exits 0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -r docs "$tmp/docs"
ln -s "$PWD/scripts" "$PWD/test" "$PWD/examples" "$PWD/include" "$PWD/src" "$tmp/"

python3 scripts/docs_api_check.py

grep -qF '[doc:ld-preload]' "$tmp/docs/fftw.md"
sed -i 's/\[doc:ld-preload\]/[doc:no-such-marker]/' "$tmp/docs/fftw.md"
sed -i 's/`size()`/`size(size_t)`/' "$tmp/docs/cpp-api.md"
rc=0
out=$(python3 scripts/docs_api_check.py "$tmp" 2>&1) || rc=$?
if [[ $rc -eq 0 ]]; then echo "FAIL: docs_api_check.py exited 0 with a drifted table"; exit 1; fi
grep -F 'drifted signature `size(size_t)`' <<<"$out"

rc=0
out=$(${SPHINX_BUILD:-sphinx-build} -W --keep-going -b html "$tmp/docs" "$tmp/out" 2>&1) || rc=$?
if [[ $rc -eq 0 ]]; then echo "FAIL: sphinx-build exited 0 with a broken marker"; exit 1; fi
grep -F 'start-after pattern not found: [doc:no-such-marker]' <<<"$out"
echo "PASS: docs_api_check.py held and fired on a drifted row, sphinx-build named the broken marker"
