#!/usr/bin/env bash
# Positive control for the docs build. A copy of docs/ names a marker that no source has. The
# build must exit non-zero and name the marker, so a literalinclude cannot fail silently.
# SPHINX_BUILD replaces sphinx-build, to check that this script rejects a build that exits 0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -r docs "$tmp/docs"
ln -s "$PWD/scripts" "$PWD/test" "$PWD/examples" "$PWD/include" "$PWD/src" "$tmp/"
grep -qF '[doc:ld-preload]' "$tmp/docs/fftw.md"
sed -i 's/\[doc:ld-preload\]/[doc:no-such-marker]/' "$tmp/docs/fftw.md"
rc=0
out=$(${SPHINX_BUILD:-sphinx-build} -W --keep-going -b html "$tmp/docs" "$tmp/out" 2>&1) || rc=$?
if [[ $rc -eq 0 ]]; then echo "FAIL: sphinx-build exited 0 with a broken marker"; exit 1; fi
grep -F 'start-after pattern not found: [doc:no-such-marker]' <<<"$out"
echo "PASS: sphinx-build exited $rc and named the broken marker"
