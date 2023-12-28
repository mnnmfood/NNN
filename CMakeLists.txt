cmake_minimum_required(VERSION 3.0.0)

project(NNN VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
option(TESTING "Create test executable" ON)

if(TESTING)
    add_executable(tests tests/main.cpp)
    find_path(EIGEN_LIB Eigen)
    set(GCC_FLAGS "-DNDEBUG;-ffast-math;-O2;-DFLAGSET;-Wall")
    target_compile_options(tests PUBLIC "${GCC_FLAGS}")

    target_include_directories(tests PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/src"
                            "${EIGEN_LIB}"
                            )
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
