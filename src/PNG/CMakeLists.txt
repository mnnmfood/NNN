include(CMakePrintHelpers)
file(GLOB PNG_FILES *.cpp)
add_library(pngwrapper "${PNG_FILES}")

find_path(EIGEN_LIB Eigen)
find_path(PNG_LIB png.h)
cmake_print_variables(PNG_LIB)

target_include_directories(pngwrapper PUBLIC
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

target_compile_options(pngwrapper PUBLIC "${GCC_FLAGS}")