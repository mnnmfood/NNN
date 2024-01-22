cmake_minimum_required(VERSION 3.0.0)
project(NNN VERSION 0.1.0 LANGUAGES C CXX)

include(CMakePrintHelpers)
include(CTest)
message(STATUS "TEST MESSAGE")
enable_testing()
option(TESTING "Create test executable" ON)
add_subdirectory(src)

if(TESTING)
    add_executable(tests tests/main.cpp)
    find_library(PNG_LIB_PATH libpng.a)
    find_library(Z_LIB_PATH libz.a)
    cmake_print_variables(PNG_LIB_PATH)
    cmake_print_variables(Z_LIB_PATH)
    target_link_libraries(tests PUBLIC src pngwrapper "${PNG_LIB_PATH}" "${Z_LIB_PATH}")
    target_include_directories(tests PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/include"
                            "${PROJECT_SOURCE_DIR}/tests"
                            "${EIGEN_LIB}"
                            "${PNG_LIB}"
                            )

    set(GCC_FLAGS "-std=c++17;-DNDEBUG;-ffast-math;-O3;-DFLAGSET;-Wall")
    target_compile_options(tests PUBLIC "${GCC_FLAGS}")

endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
