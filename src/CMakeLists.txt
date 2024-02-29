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

set(GCC_FLAGS "-std=c++17;-DNDEBUG;-ffast-math;-O2;-DFLAGSET;-Wall")
target_compile_options(src PUBLIC "${GCC_FLAGS}")