# find_package(admiral) -> admiral::admiral{,_static}, admiral::admiral_c{,_static},
#                          admiral::fftw{,_static}
#
# Shared and static per surface, all six self-contained machine code for
# std::complex<float> and std::complex<double>. The OBJECT libraries are a
# compile-time device and are not installed; nor is admiral_internal.
#
# Only the three public headers go in. They include nothing but the standard
# library, so the prefix needs no xsimd, no poet and no admiral/detail.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS admiral admiral_static
                admiral_c admiral_c_static
                admiral_fftw admiral_fftw_static
    EXPORT admiralTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
    ${PROJECT_SOURCE_DIR}/include/admiral/admiral.hpp
    ${PROJECT_SOURCE_DIR}/include/admiral/admiral.h
    ${PROJECT_SOURCE_DIR}/include/admiral/fftw3.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/admiral)

install(EXPORT admiralTargets NAMESPACE admiral::
    FILE admiralTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/admiral)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/admiralConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/admiralConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/admiral)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/admiralConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/admiralConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/admiralConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/admiral)
