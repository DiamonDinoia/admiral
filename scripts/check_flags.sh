#!/usr/bin/env bash
# Usage: check_flags.sh <build-dir> <want:PAT | not:PAT>...
#          or: check_flags.sh --self-test
# Each want: pattern must occur in <build-dir>/compile_commands.json. Each not: pattern must not.
#
# A pattern matches a WHOLE flag token. A token is a run of [-A-Za-z0-9+_=,] characters; paths
# split on / and names on a dot. A pattern matches a token when the token equals it, ends with
# it (the value of a -key=value flag, e.g. =x86-64 matches -march=x86-64), or starts with it
# glued to =<value> (-DADM_DISPATCH matches -DADM_DISPATCH=ON). A pattern that ends in = is a
# key prefix and matches any =value (-fsanitize= matches -fsanitize=address,undefined).
# Substring matching across a token edge was a hole: =x86-64 also matched -march=x86-64-v3,
# so a v3 build passed the floor assertion (review C8). The old prefix form
# 'not:-march=x86-64-v' is vacuous now; ci.yml spells the three nots out.
set -u

runs() { tr -c 'A-Za-z0-9+_=,-' '\n' < "$1" | grep -v '^$'; }

count() { # db pattern
    local pat=$2 re
    re=$(printf '%s' "$pat" | sed 's/[][\.*^$+?{}()|]/\\&/g')
    if [ "${pat: -1}" = "=" ]; then
        { runs "$1" | grep -cE -e "${re}$" -e "^${re}"; } || true
    else
        { runs "$1" | grep -cE -e "${re}$" -e "^${re}="; } || true
    fi
}

run_specs() { # db specs...
    local db=$1 rc=0 n spec; shift
    for spec in "$@"; do
        n=$(count "$db" "${spec#*:}")
        case ${spec%%:*} in
            want) ((n > 0)) || { echo "  FLAG MISSING: ${spec#*:}"; rc=1; } ;;
            not)  ((n == 0)) || { echo "  FLAG PRESENT: ${spec#*:} ($n tokens)"; rc=1; } ;;
            *)    echo "  BAD SPEC: $spec"; rc=1 ;;
        esac
    done
    return $rc
}

self_test() {
    local T F fails=0 got
    T=$(mktemp -d)
    mkdir "$T/b"; F=$T/b/compile_commands.json
    t() { # expect_rc text specs...
        local expect=$1 text=$2; shift 2
        printf '%s\n' "$text" > "$F"
        if run_specs "$F" "$@" >/dev/null 2>&1; then got=0; else got=1; fi
        if [ "$got" = "$expect" ]; then echo "  ok  [$expect] $* ON $text"
        else echo "  BAD [want $expect got $got] $* ON $text"; fails=1; fi
    }
    t 0 '"command": "g++ -O2 -march=x86-64 -c a.cpp"'        want:=x86-64
    t 1 '"command": "g++ -O2 -march=x86-64-v3 -c a.cpp"'     want:=x86-64   # review C8 hole
    t 0 '"command": "g++ -O2 -march=x86-64-v3 -c a.cpp"'     want:=x86-64-v3
    t 0 '"command": "g++ -O1 -fsanitize=address,undefined -c a.cpp"' want:-fsanitize=
    t 0 '"command": "g++ -O2 -DADM_DISPATCH=ON -c a.cpp"'    want:-DADM_DISPATCH
    t 1 '"command": "g++ -O2 -DADM_DISPATCHER=ON -c a.cpp"'  want:-DADM_DISPATCH
    t 0 '"command": "g++ -Og -O2 -c a.cpp"'                  not:-O1
    t 1 '"command": "g++ -O1 -c a.cpp"'                      not:-O1
    t 0 '"command": "g++ -c a.cpp" "-include /b/xx/cmake_pch.hxx"' want:cmake_pch
    t 1 '"command": "g++ -O2 -march=native -c a.cpp"'        not:=native
    t 0 '"command": "clang++ -O2 -march=native -c a.cpp"'    want:=native
    t 0 '"command": "g++ -std=c++17 -c a.cpp"'               want:-std=c++17 not:std=c++20
    t 1 '"command": "g++ -std=c++20 -c a.cpp"'               not:std=c++20
    [ "$fails" = 0 ] && echo "self-test: PASS" || echo "self-test: FAIL"
    rm -rf "$T"
    [ "$fails" = 0 ]
}

if [ "${1:-}" = "--self-test" ]; then self_test; exit $?; fi
db=$1/compile_commands.json
shift
(($# > 0)) || { echo "  NO SPECS"; exit 1; }
[[ -s $db ]] || { echo "  NO COMPILE DB: $db"; exit 1; }
run_specs "$db" "$@"
exit $?
