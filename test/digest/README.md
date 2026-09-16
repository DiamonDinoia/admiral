# Bit-identity digest

`digest.cpp` runs a positional case sweep (1344 cases: the f32 block, then the identical
f64 block; `case_list.hpp` is the list) and prints one `id tag name hash` line per case,
the hash a FNV-1a over the raw output bytes. `scripts/validate.sh digest` builds this
target twice -- shipped (Release, `x86-64-v3`) and strict (the same plus
`-DADM_USE_FAST_MATH=OFF`) -- and diffs both runs against the goldens in `golden/` with
`scripts/check_digest.sh`. A machinery-only change moves nothing; a kernel change moves
exactly the cases its diff reaches, so the arm's expectation is the landing's
reachability prediction, and goldens are the committed form of "prediction: zero
movers".

The digest lives in the validate arm, not in ctest: the gate is a comparison across two
builds plus committed goldens, and a ctest entry sees only its own build. Building the
`admiral_digest` target is all `test/CMakeLists.txt` does for it.

## The arm, case by case

1. Shipped and strict builds each assert their flags on `compile_commands.json`
   (`-march=x86-64-v3`, `-ffast-math` present/absent) and the strict build's cache
   carries `ADM_USE_FAST_MATH:BOOL=OFF`.
2. Each digest run self-checks: the emitted case count must equal `kCaseCount` and case
   `kUlpControlCase` must still be named `kUlpControlName` (`case_list.hpp`). A
   renumbered list fails here, loudly, before any comparison.
3. `check_digest.sh golden actual` reports the mover ids (hash differs), fails on ids
   the golden does not know (a case landed without regoldening) and on id/tag/name drift
   (renumbering). Ids only in the golden are skipped: a build that leaves a case
   undefined does not participate in it.
4. The strict mover set must be a subset of the shipped mover set. Strict numerics never
   move a case that shipped numerics leave alone.
5. Positive control, every run: `--ulp <kUlpControlCase>` perturbs one output element of
   one case by one ulp; the perturbed digest must differ from the same build's plain
   digest in exactly that case. This is what proves the comparison can see a one-ulp
   change, today, in this arm.
6. `--dump <id> <file>` writes one case's raw output for mover forensics.

## Case-list stability and growth budget

Case ids are positional, and goldens plus the ulp control key on them, so numbering is
append-stable by construction -- the rule and its guards are stated in
`case_list.hpp`, which is the source of truth. In short:

* New cases append after the f64 block ONLY (f32 growth happens as a new block after
  f64, because f64 follows f32 in emission order).
* The same commit bumps `kCaseCount` (the `static_assert` in `case_list.hpp` enforces
  the pairing) and regenerates both goldens.
* Growth is at most 5% of the list length per landing (at most 67 cases at 1344). A
  catalog-structural landing may exceed the budget only by stating so in a `receipt:`
  trailer in its commit message.

## Regoldening

A landing whose diff legitimately changes bits regenerates both goldens **in the same
commit**, and its commit body quotes the mover list against the old goldens
(`check_digest.sh` output) and argues that list equals the diff's reachability:

```sh
module load gcc/14.2.0 && unset NINJA_STATUS
b=<scratch>/digest-regolden
for arm in shipped strict; do
    extra=(); [ "$arm" = strict ] && extra=(-DADM_USE_FAST_MATH=OFF)
    cmake -S . -B "$b-$arm" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DADM_TARGET_ARCH=x86-64-v3 -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF \
        "${extra[@]}"
    cmake --build "$b-$arm" --target admiral_digest -j "$(nproc)"
    "$b-$arm/test/admiral_digest" >"new-$arm.txt"
    scripts/check_digest.sh "test/digest/golden/$arm.txt" "new-$arm.txt" \
        --movers-out "movers-$arm.txt" || true   # quote this mover list in the commit
    mv "new-$arm.txt" "test/digest/golden/$arm.txt"
done
scripts/validate.sh digest   # must go green against the new goldens
```

(`|| true` above is the deliberate exception to the no-swallowed-failures rule: the
regolden step WANTS the non-empty mover list check_digest then reports.)

## Provenance and portability

The committed goldens were generated from the pristine base commit `683a697` (the
machinery port adds no library code, so branch builds reproduce them), on ccmlin075
(Xeon w5-3435X, 16 physical cores, 2 MiB L2/core, 45 MiB L3 over 16 physical cores)
with the arm's exact flags: Release, `-march=x86-64-v3`, `-ffast-math` /
`ADM_USE_FAST_MATH=OFF`, gcc 14.2.0, 2026-09-14.

Codegen is pinned by `-march=x86-64-v3` (not `native`), so any x86-64 host with AVX2
builds identical kernels. Route selection is NOT pinned: plan routing reads host cache
geometry (`e2_len_cap`'s L3-per-physical-core gate, the L2 `fuse_planes` gate), so a
host whose cache geometry differs from ccmlin075's can legitimately produce
route-shaped movers with no code change. Regolden on the gating host class if the arm
moves to a new one; the ulp control and the strict-subset invariant are
geometry-independent and gate everywhere.

Digest stdout must be captured through a 4096-blksize-class sink (a pipe, tmpfs, or a
local-disk file): the Bluestein/Rader heap buffers' alignment class rides the process's
first stdio heap allocation, whose size scales with the sink's st_blksize, so a 1
MiB-blksize NFS file sink flips fast-math prime>11 1-D case bits; the arm therefore
pipes `admiral_digest` through `cat`, and the underlying engine sensitivity is the open
Bluestein/Rader alignment item, tracked separately.
