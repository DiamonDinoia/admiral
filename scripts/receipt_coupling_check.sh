#!/usr/bin/env bash
# The receipt-coupling scan (WI-0b). A commit that touches a hand-fit- or probe-fit
# constants header must carry "receipt:" (any case) in its message: a fitted or
# hand-derived constant and the receipt it was read from are one artifact (CLAUDE.md,
# "The four_step_large admission lines are hand-fit, and they are measured
# crossovers"), so the constants set below is whole headers, not symbols inside them.
# A constants move that names no receipt is exactly the cohort-tuned-constant disease:
# the number survives, the derivation does not, and the next host cohort silently reads
# a misfit value.
#
# Usage: receipt_coupling_check.sh <repo> [<range>]    (default HEAD~1..HEAD)
# Exit 0: no marker-less constants commit (an empty range says so and passes).
# Exit 1: at least one constants commit lacks the marker; every offending sha+path
#         is named on stderr.
# Exit 2: the range does not resolve in <repo> (shallow checkout, typo'd base).

set -uo pipefail

repo=${1:?usage: receipt_coupling_check.sh <repo> [<range>]}
range=${2:-HEAD~1..HEAD}

# base_cost_model.hpp is GENERATED, and a coefficients move IS a constants move;
# nd_plan.hpp is header-level over-coupled by design (its named constants share the
# header with the route gates they were fitted against; see the wi0b receipt).
constants=(
    include/admiral/detail/four_step_large.hpp
    include/admiral/detail/granule_codelet.hpp
    include/admiral/detail/base_cost_model.hpp
    include/admiral/detail/math.hpp
    include/admiral/detail/nd_plan.hpp
)

if ! git -C "$repo" rev-list --no-merges --max-count=1 "$range" >/dev/null 2>&1; then
    echo "receipt-coupling: range '$range' does not resolve in $repo" >&2
    exit 2
fi

mapfile -t commits < <(git -C "$repo" rev-list --no-merges "$range")
if ((${#commits[@]} == 0)); then
    echo "receipt-coupling: range '$range' is empty in $repo; nothing to scan (trivial pass)"
    exit 0
fi

rc=0
guarded=0
for sha in "${commits[@]}"; do
    mapfile -t touched < <(git -C "$repo" diff-tree --root --no-commit-id --name-only \
                               -r "$sha")
    hits=$(printf '%s\n' "${touched[@]}" | grep -Fxf <(printf '%s\n' "${constants[@]}"))
    v=$?
    ((v <= 1)) || { echo "receipt-coupling: path scan failed for $sha" >&2; rc=1; continue; }
    [[ -n $hits ]] || continue
    if git -C "$repo" log -1 --format=%B "$sha" | grep -qi 'receipt:'; then
        ((guarded++))
        continue
    fi
    echo "receipt-coupling: FAIL ${sha:0:12} touches constants without 'receipt:' in the message:" >&2
    while IFS= read -r p; do printf '  %s\n' "$p" >&2; done <<<"$hits"
    rc=1
done

if ((rc == 0)); then
    echo "receipt-coupling: OK — scanned ${#commits[@]} commit(s) in '$range';" \
         "$guarded touched constants and every one carried 'receipt:'"
fi
exit $rc
