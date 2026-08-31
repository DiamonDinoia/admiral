// Undefines the `macros.hpp` symbols so they never leak into user translation
// units. In every detail header that included `macros.hpp`, include this
// header last.

#undef ADM_DETAIL_MACROS_ACTIVE
#undef ADM_ALWAYS_INLINE
#undef ADM_LAMBDA_ALWAYS_INLINE
#undef ADM_FLATTEN
#undef ADM_NOINLINE
#undef ADM_COLD
#undef ADM_RESTRICT
