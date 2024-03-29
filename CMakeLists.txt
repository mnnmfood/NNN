cmake_minimum_required(VERSION 3.0.0)
project(NNN VERSION 0.1.0 LANGUAGES C CXX)
cmake_policy(SET CMP0054 NEW)
include(CTest)
include(CMakePrintHelpers)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/Default")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/build/Debug")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build/Release")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/Default")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/build/Debug")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build/Release")


if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(GCC_FLAGS "-std=c++17;-DNDEBUG;-ffast-math;-O2;-march=native")
	target_compile_options(tests PUBLIC "${GCC_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	STRING (REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS "/std:c++17 /O2 /MD /arch:AVX2 /DEIGEN_VECTORIZE  /DEIGEN_STRONG_INLINE=inline /bigobj")
endif()

# find necessary libraries
find_path(EIGEN_LIB Eigen REQUIRED)
find_path(PNG_LIB png.h REQUIRED)

find_package(Boost REQUIRED)
add_compile_definitions(BOOST_IOSTREAMS_NO_LIB)
add_compile_options(/EHsc)

cmake_print_variables(EIGEN_LIB)
cmake_print_variables(PNG_LIB)
cmake_print_variables(Boost_INCLUDE_DIR)

# add main library
add_subdirectory(src)

#add executable with tests
if(BUILD_TESTING)
	add_subdirectory(tests)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
