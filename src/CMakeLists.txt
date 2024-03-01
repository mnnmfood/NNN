file(GLOB srcs *.cpp)
add_library(src "${srcs}")
add_subdirectory(PNG)
find_path(EIGEN_LIB Eigen)
find_path(PNG_LIB png.h)
target_include_directories(src PUBLIC
                        "${PROJECT_BINARY_DIR}"
                        "${PROJECT_SOURCE_DIR}/include"
                        "${EIGEN_LIB}"
						"${PNG_LIB}"
                        )

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(GCC_FLAGS "-std=c++17;-DNDEBUG;-ffast-math;-O2;-march=native")
	target_compile_options(tests PUBLIC "${GCC_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	cmake_print_variables(CMAKE_CXX_COMPILER_ID)
	string(REPLACE "-RTC1" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS "/std:c++17 /O2 /MD /arch:AVX2 /DEIGEN_STRONG_INLINE=inline /bigobj")
endif()

target_compile_options(src PUBLIC "${GCC_FLAGS}")