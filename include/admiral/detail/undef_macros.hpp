// Undefines the macros from admiral/detail/macros.hpp so they never leak into user
// translation units. Include at the end of every detail header that included
// macros.hpp. Mirrors POET's undef_macros.hpp convention.

#undef ADM_DETAIL_MACROS_ACTIVE
#undef ADM_ALWAYS_INLINE
#undef ADM_LAMBDA_ALWAYS_INLINE
#undef ADM_NOINLINE
#undef ADM_RESTRICT
#undef ADM_ASSUME
#undef ADM_LIKELY
#undef ADM_UNLIKELY
#undef ADM_HOT
#undef ADM_COLD
