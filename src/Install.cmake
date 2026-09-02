
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
    ${PROJECT_SOURCE_DIR}/include/admiral/errors.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/admiral)

install(FILES
    ${PROJECT_SOURCE_DIR}/include/admiral/detail/api.h
    ${PROJECT_SOURCE_DIR}/include/admiral/detail/cxx_compat.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/admiral/detail)

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
