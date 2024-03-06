file(GLOB srcs *.cpp)
add_subdirectory(PNG)

add_library(src "${srcs}")
find_path(EIGEN_LIB Eigen)
find_path(PNG_LIB png.h)


target_include_directories(src PUBLIC
                        "${PROJECT_BINARY_DIR}"
                        "${PROJECT_SOURCE_DIR}/include"
                        "${EIGEN_LIB}"
						"${PNG_LIB}"
                        )
