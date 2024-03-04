cmake_minimum_required(VERSION 3.0.0)
project(NNN VERSION 0.1.0 LANGUAGES C CXX)

cmake_policy(SET CMP0054 NEW)
include(CTest)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(GCC_FLAGS "-std=c++17;-DNDEBUG;-ffast-math;-O2;-march=native")
	target_compile_options(tests PUBLIC "${GCC_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	string(REPLACE "-RTC1" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS "/std:c++17 /O2 /MD /arch:AVX2 /DEIGEN_STRONG_INLINE=inline /bigobj")
	#set(CMAKE_CXX_FLAGS "/std:c++17 /MDd /DEIGEN_DONT_ALIGN=1 /DEIGEN_DONT_VECTORIZE=1")
endif()



message(STATUS "TEST MESSAGE")
enable_testing()
option(TESTING "Create test executable" ON)
add_subdirectory(src)


if(TESTING)
    add_executable(tests tests/main.cpp)
    find_library(PNG_LIB_PATH libpng)
    find_library(Z_LIB_PATH zlib)
	cmake_print_variables(PNG_LIB_PATH)
    target_link_libraries(tests PUBLIC src pngwrapper "${PNG_LIB_PATH}" "${Z_LIB_PATH}")
    target_include_directories(tests PUBLIC
                            "${PROJECT_BINARY_DIR}"
                            "${PROJECT_SOURCE_DIR}/include"
                            "${PROJECT_SOURCE_DIR}/tests"
                            "${EIGEN_LIB}"
                            "${PNG_LIB}"
                            )

endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
