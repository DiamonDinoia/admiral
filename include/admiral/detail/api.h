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
