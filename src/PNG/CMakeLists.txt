file(GLOB PNG_FILES *.cpp)
add_library(pnglib "${PNG_FILES}")
find_path(EIGEN_LIB Eigen)
find_path(PNG_LIB libpng)
target_include_directories(pnglib PUBLIC
                        "${PROJECT_BINARY_DIR}"
                        "${PROJECT_SOURCE_DIR}/include"
                        "${EIGEN_LIB}"
                        "${PNG_LIB}"
                        )

target_link_libraries(pnglib PUBLIC libpng)
set(GCC_FLAGS "-std=c++17;-DNDEBUG;-ffast-math;-O2;-DFLAGSET;-Wall")
target_compile_options(pnglib PUBLIC "${GCC_FLAGS}")