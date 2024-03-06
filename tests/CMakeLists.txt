add_executable(tests main.cpp)
find_library(PNG_LIB_PATH libpng)
find_library(Z_LIB_PATH zlib)
cmake_print_variables(PNG_LIB_PATH)
cmake_print_variables(Z_LIB_PATH)

target_link_libraries(tests PUBLIC src pngwrapper "${PNG_LIB_PATH}" "${Z_LIB_PATH}")
add_definitions(-DDATA_DIR=${PROJECT_SOURCE_DIR}/data/)
target_include_directories(tests PUBLIC
						"${PROJECT_SOURCE_DIR}/include"
						"${PROJECT_SOURCE_DIR}/tests"
						"${EIGEN_LIB}"
						"${PNG_LIB}"
						)