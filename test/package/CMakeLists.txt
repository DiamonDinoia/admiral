# Consumes an INSTALLED admiral, not the build tree. This is the only test that
# exercises the export set, the installed headers, and the README's claim that the
# three public headers include nothing but the standard library.
#
# Standalone by design, and nothing add_subdirectory's it. Configure it against a prefix:
#   cmake -S test/package -B build-pkg -DCMAKE_PREFIX_PATH=<prefix> && ctest --test-dir build-pkg
cmake_minimum_required(VERSION 3.25)
project(admiral_package_test C CXX)

find_package(admiral REQUIRED)
enable_testing()

# The C consumers call cos/sin/hypot themselves, so THEY need libm. admiral does
# not export it. Indirect deps do not satisfy a consumer's own references under
# --no-copy-dt-needed-entries, so relying on libadmiral_c.so pulling libm in fails
# to link. find_library keeps it a no-op where libm is part of libc (macOS, MSVC).
find_library(ADM_LIBM m)
if(NOT ADM_LIBM)
    set(ADM_LIBM "")
endif()

# One executable per interface x linkage. The C ones are compiled as C and get only
# the C header, which makes them a test of the C interface rather than of C++.
foreach(link admiral admiral_static)
    add_executable(use_cxx_${link} use_cxx.cpp)
    target_link_libraries(use_cxx_${link} PRIVATE admiral::${link})
    add_test(NAME use_cxx_${link} COMMAND use_cxx_${link})
endforeach()

foreach(link admiral_c admiral_c_static)
    add_executable(use_c_${link} use_c.c)
    target_link_libraries(use_c_${link} PRIVATE admiral::${link} ${ADM_LIBM})
    add_test(NAME use_c_${link} COMMAND use_c_${link})
endforeach()

foreach(link fftw fftw_static)
    add_executable(use_fftw_${link} use_fftw.c)
    target_link_libraries(use_fftw_${link} PRIVATE admiral::${link} ${ADM_LIBM})
    add_test(NAME use_fftw_${link} COMMAND use_fftw_${link})
endforeach()
