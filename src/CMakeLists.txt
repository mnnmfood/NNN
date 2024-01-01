file(GLOB srcs *.cpp)
add_library(src "${srcs}")

find_path(EIGEN_LIB Eigen)
target_include_directories(src PUBLIC
                        "${PROJECT_BINARY_DIR}"
                        "${PROJECT_SOURCE_DIR}/include"
                        "${EIGEN_LIB}"
                        )

set(GCC_FLAGS "-std=c++17;-DNDEBUG;-ffast-math;-O2;-DFLAGSET;-Wall")
target_compile_options(src PUBLIC "${GCC_FLAGS}")