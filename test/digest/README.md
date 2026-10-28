# Bit-identity digest
`digest.cpp` prints one hash for each entry of `case_list.hpp`. `golden/` holds the committed hashes for the shipped and strict builds.

```bash
scripts/validate.sh digest
```

The `shipped` arm builds with fast math. The `strict` arm builds with `-DADM_USE_FAST_MATH=OFF`. A mover is a case whose hash differs from the golden.

The `shipped` arm fails on a mover that `golden/allowed-movers-2026-09-17.txt` does not list. Ids in that file can move on a clean tree. The `strict` arm and the `--ulp` control accept no mover. A strict mover must also be a shipped mover.

The driver hands the engine only pinned transform buffers (a 4096-byte-aligned block plus the fixed `ADM_DIGEST_BUF_OFF` offset in `digest.cpp`), so driver-buffer alignment and allocation history cannot move a case. The waiver lists the residual runtime-allocator pool, re-derived from the `ADM_DIGEST_BUF_OFF=16` arm and a second-glibc run of the banked binary.

The goldens come from gcc 14.2.0 on a Xeon w5-3435X (16 cores, 2 MiB L2 per core, 45 MiB L3). Route selection reads the host cache sizes. A host with other sizes can show movers with no code change. Regenerate on a host of the same class.

To regenerate both goldens, run this from the repository root. It stops on the first failure. It keeps each golden until its run succeeds. A nonzero `check_digest.sh` status passes only when it lists movers.

```bash
(
    set -euo pipefail
    for arm in shipped strict; do
        extra=(); [ $arm = strict ] && extra=(-DADM_USE_FAST_MATH=OFF)
        cmake -S . -B build/digest-$arm -G Ninja -DCMAKE_BUILD_TYPE=Release -DADM_TARGET_ARCH=x86-64-v3 \
            -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF "${extra[@]}"
        cmake --build build/digest-$arm --target admiral_digest
        build/digest-$arm/test/admiral_digest | cat > build/digest-$arm.txt
        # Keep the ids the golden knows: the check rejects a new id before it writes the movers.
        awk 'NR==FNR { k[$1]; next } $1 in k' test/digest/golden/$arm.txt build/digest-$arm.txt \
            > build/digest-$arm.known
        rm -f build/digest-$arm.movers
        scripts/check_digest.sh test/digest/golden/$arm.txt build/digest-$arm.known \
            --movers-out build/digest-$arm.movers || [ -s build/digest-$arm.movers ]
        mv build/digest-$arm.txt test/digest/golden/$arm.txt
    done
    scripts/validate.sh digest
)
```

Rules:
- A change that moves bits regenerates the two goldens in the same commit. The commit message gives the mover lists from `build/digest-<arm>.movers`.
- Pipe the digest output. Do not write it to an NFS file, because the sink block size changes the bits of some outputs.
- Add new cases after the f64 block only, and bump `kCaseCount` in `case_list.hpp` in the same commit.
