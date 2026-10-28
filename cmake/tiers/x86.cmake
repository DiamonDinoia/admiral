# x86-64 tiers: floor x86-64 (sse2), tiers v3 (fma3<avx2>) and v4 (avx512bw). ADM_DISPATCH reads these.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(ADM_TIER_FLOOR x86-64)
    set(ADM_TIER_TAGS v3 v4)
    if(MSVC)
        # cl or clang with the MSVC frontend: /arch:AVX2 gives the v3 xsimd arch (R5).
        set(_adm_tier_flag_v3 /arch:AVX2)
        set(_adm_tier_flag_v4 /arch:AVX512)
    else()
        set(_adm_tier_flag_v3 -march=x86-64-v3)
        set(_adm_tier_flag_v4 -march=x86-64-v4)
    endif()
endif()
