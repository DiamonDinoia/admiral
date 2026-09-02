#pragma once

/* One export spelling for the C++, C and FFTW surfaces, which is why this header is C.
   An MSVC-style consumer must see dllIMPORT: dllexport in a consumer's translation unit makes
   the linker look for a local definition. ADM_BUILDING marks the translation units that compile
   the DLL, ADM_STATIC_DEFINE rides on the static targets' INTERFACE, and a consumer of the
   shared library carries neither. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(ADM_STATIC_DEFINE)
#    define ADM_VISIBILITY
#  elif defined(ADM_BUILDING)
#    define ADM_VISIBILITY __declspec(dllexport)
#  else
#    define ADM_VISIBILITY __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_VISIBILITY __attribute__((visibility("default")))
#else
#  define ADM_VISIBILITY
#endif

/* MSVC does not carry the attribute from a template's declaration onto an explicit instantiation
   definition, so `template void forward<double>(...)` in a DLL translation unit exports nothing
   and a consumer's link fails with LNK2019. The attribute has to sit on the instantiation itself,
   which is the form the docs give for class templates: `template class __declspec(dllexport)
   B<int>;`. clang-cl and the ELF toolchains do carry it (verified in the IR), so this repeats
   what they already did rather than adding anything. */
#if defined(_WIN32) || defined(__CYGWIN__)
#  define ADM_API_INST ADM_VISIBILITY
#else
#  define ADM_API_INST
#endif
