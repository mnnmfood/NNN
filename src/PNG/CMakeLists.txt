file(GLOB PNG_FILES *.cpp)
add_library(pngwrapper "${PNG_FILES}")

find_path(EIGEN_LIB Eigen)
find_path(PNG_LIB png.h)

target_include_directories(pngwrapper PUBLIC
					"${PROJECT_BINARY_DIR}"
					"${PROJECT_SOURCE_DIR}/include"
					"${EIGEN_LIB}"
					"${PNG_LIB}"
					)