#!/usr/bin/env bash
# Usage: check_flags.sh <build-dir> <want:PAT | not:PAT>...
# Each want: pattern must occur in <build-dir>/compile_commands.json. Each not: pattern must not.
set -u
db=$1/compile_commands.json
shift
(($# > 0)) || { echo "  NO SPECS"; exit 1; }
[[ -s $db ]] || { echo "  NO COMPILE DB: $db"; exit 1; }
rc=0
for spec in "$@"; do
    pat=${spec#*:}
    n=$(grep -c -F -- "$pat" "$db")
    case ${spec%%:*} in
        want) ((n > 0)) || { echo "  FLAG MISSING: $pat"; rc=1; } ;;
        not)  ((n == 0)) || { echo "  FLAG PRESENT: $pat ($n TUs)"; rc=1; } ;;
        *)    echo "  BAD SPEC: $spec"; rc=1 ;;
    esac
done
exit $rc
