# LoongArch: floor la64v1.0 (LSX, numpy's baseline) and one run-time tier, LASX (-mlasx).
# xsimd has no LASX FMA kernel, so the tier adds width only. To drop it, delete the
# ADM_TIER_TAGS line: the LSX floor then ships as one build (ADM_DISPATCH=OFF).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^loongarch64$")
    set(ADM_TIER_FLOOR la64v1.0)
    set(ADM_TIER_TAGS lasx)
    set(_adm_tier_flag_lasx -mlasx)
    # gcc 14 reports a false maybe-uninitialized inside xsimd_lsx.hpp:923 and xsimd_lasx.hpp:988.
    # The flag rides this arch's profile only. It reaches no other arch.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(admiral_optimization_flags INTERFACE -Wno-error=maybe-uninitialized)
    endif()
endif()
