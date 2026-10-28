# AArch64 tiers for ADM_DISPATCH: the NEON floor (armv8-a) and one fixed-width SVE tier,
# sve256 (-msve-vector-bits=256). xsimd has one arch per SVE vector length and the length is
# fixed at compile time, so each SVE tier is one -msve-vector-bits build; the run-time pick
# (hwcap + rdvl) takes sve<N> only at VL == N, and any other VL falls back to NEON.
# sve512 stays out: gcc 14.2 ICEs ("unrecognizable insn", UNSPEC_TRN1_CONV on VNx4BI) in
# inst_gt_f.cpp at -msve-vector-bits=512, and 61 wrong results at VL 512 under qemu wait on
# job sve512-diag. Re-adding it is one tag here and one row in src/tiers/aarch64.hpp.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ADM_TIER_FLOOR armv8-a)
    set(ADM_TIER_TAGS sve256)
    # POET_DISABLE_PUSH_OPTIMIZE: gcc rejects poet's concatenated _Pragma at SVE widths.
    set(_adm_tier_flag_sve256 -march=armv8.2-a+sve -msve-vector-bits=256 -DPOET_DISABLE_PUSH_OPTIMIZE)
endif()
