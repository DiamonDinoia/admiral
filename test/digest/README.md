# Bit-identity digest
`digest.cpp` runs a positional case sweep (1344 cases: the f32 block, then the identical f64 block).
`case_list.hpp` is the list. It prints one `id tag name hash` line per case, the hash a FNV-1a over the raw
output bytes. `scripts/validate.sh digest` builds this target twice, as shipped (Release, `x86-64-v3`) and as
strict (plus `-DADM_USE_FAST_MATH=OFF`), and diffs both runs against the goldens in `golden/` with
`scripts/check_digest.sh`. A machinery-only change moves nothing. A kernel change moves exactly the cases its
diff reaches. The goldens encode "prediction: zero movers".

The digest lives in the validate arm, not in ctest: the gate compares two builds plus committed goldens.
`test/CMakeLists.txt` only builds the `admiral_digest` target.

## The arm, case by case
1. Each build asserts its flags on `compile_commands.json` (`-march=x86-64-v3`, `-ffast-math` present or
   absent), and the strict build's cache carries `ADM_USE_FAST_MATH:BOOL=OFF`.
2. Each run self-checks: the emitted case count must equal `kCaseCount`, and case `kUlpControlCase` must keep
   the name `kUlpControlName` (`case_list.hpp`). A renumbered list fails here, before any comparison.
3. `check_digest.sh golden actual` reports the mover ids (hash differs). It fails on ids the golden does not
   know and on id/tag/name drift. Ids only in the golden are skipped.
4. The strict mover set must be a subset of the shipped mover set.
5. Positive control on every run: `--ulp <kUlpControlCase>` perturbs one output element of one case by one
   ulp. The perturbed digest must differ from the plain digest in exactly that case. This proves the
   comparison can see a one-ulp change.
6. `--dump <id> <file>` writes one case's raw output for mover forensics.

## The shipped-arm waiver
Under fast math, the shipped dialect keys vector-arm dispatch on pointer alignment, so a change to the
construction-time allocation sequence re-draws which address-sensitive cases move. The class is 1-D lengths
with a prime factor above 11, the pow2-dif surface {96, 128, 160, 192, 256, 320}, and the N-D 96x96 line. The
arm therefore runs against a banked upper bound, `golden/allowed-movers-wi4a-2026-09-17.txt`
(`check_digest.sh --expect-movers-upper`): a mover id outside the bank fails the arm, a banked id that stops
moving passes. The strict arm and the `--ulp` control are exact-match always.

When the alignment sensitivity is fixed, shrink the bank to the ids that still move, and delete the empty
bank. A landing that legitimately re-draws the class edits the bank in the same commit and quotes the new
mover list.

## Case-list stability and growth budget
Case ids are positional, and goldens plus the ulp control key on them, so numbering is append-stable.
`case_list.hpp` states the full rule and carries the guards. In short:

* New cases append after the f64 block ONLY, because f64 follows f32 in emission order.
* The same commit bumps `kCaseCount` (a `static_assert` in `case_list.hpp` enforces the pairing) and
  regenerates both goldens.
* Growth is at most 5% of the list length per landing (67 cases at 1344). A catalog-structural landing may
  exceed the budget only by stating so in a `receipt:` trailer in its commit message.

## Regoldening
A landing whose diff legitimately changes bits regenerates both goldens **in the same commit**, quoting the
mover list against the old goldens (`check_digest.sh` output) and arguing it equals the diff's reachability:

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

(`|| true` is deliberate: the regolden step wants the non-empty mover list that check_digest reports.)

## Provenance and portability
The committed goldens come from the pristine base commit `683a697`, generated on ccmlin075 (Xeon w5-3435X, 16
physical cores, 2 MiB L2/core, 45 MiB L3) with the arm's exact flags (Release, `-march=x86-64-v3`,
`-ffast-math` / `ADM_USE_FAST_MATH=OFF`, gcc 14.2.0).

`-march=x86-64-v3` pins the codegen, so any x86-64 host with AVX2 builds identical kernels. Route selection is
NOT pinned: plan routing reads the host cache geometry (the L3-per-physical-core gate and the L2
`fuse_planes` gate), so a host with a different cache geometry can legitimately produce route-shaped movers
with no code change. Regolden on the gating host class if the arm moves to a new one. The ulp control and the
strict-subset rule are geometry-independent.

**Caution**: capture digest stdout only through a 4096-blksize-class sink (pipe, tmpfs, local-disk file). The
Bluestein/Rader heap buffers take their alignment class from the process's first stdio heap allocation, whose
size scales with the sink's st_blksize. A 1 MiB-blksize NFS file sink flips the bits of fast-math prime>11
1-D cases. The arm therefore pipes `admiral_digest` through `cat`.
