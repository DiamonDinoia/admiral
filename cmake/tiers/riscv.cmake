# RISC-V tiers for ADM_DISPATCH: the rv64gc floor plus RVV at VLEN 128, 256 and 512. The width
# is fixed per tier at compile time (xsimd needs __riscv_v_fixed_vlen); the run-time choice
# reads vlenb (xsimd's rvv<N> needs vlenb * 8 >= N, so the pick takes the widest the CPU has,
# and a CPU below 128 bits runs the floor). The matching arch list is src/tiers/riscv.hpp.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
    set(ADM_TIER_FLOOR rv64gc)
    # gcc 14.2 ICEs on RVV (SET_TYPE_VECTOR_SUBPARTS), so the tiers need clang.
    if(ADM_DISPATCH AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "ADM_DISPATCH on riscv64 needs clang (gcc 14.2 ICEs on RVV), not ${CMAKE_CXX_COMPILER_ID}")
    endif()
    set(ADM_TIER_TAGS rvv128 rvv256 rvv512)
    foreach(w 128 256 512)
        set(_adm_tier_flag_rvv${w} -march=rv64gcv -mrvv-vector-bits=${w})
    endforeach()
endif()
