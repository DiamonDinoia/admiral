// Undefines the macros from admiral/detail/macros.hpp so they never leak into user
// translation units. Include at the end of every detail header that included
// macros.hpp. Mirrors POET's undef_macros.hpp convention.

#undef ADM_DETAIL_MACROS_ACTIVE
#undef ADM_ALWAYS_INLINE
#undef ADM_LAMBDA_ALWAYS_INLINE
#undef ADM_FLATTEN
#undef ADM_NOINLINE
#undef ADM_COLD
#undef ADM_RESTRICT
