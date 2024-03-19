file(GLOB srcs *.cpp)
add_subdirectory(PNG)

add_library(src "${srcs}" "${Boost_INCLUDE_DIR}/libs/iostreams/src/mapped_file.cpp")

target_include_directories(src PUBLIC
                        "${PROJECT_BINARY_DIR}"
                        "${PROJECT_SOURCE_DIR}/include"
                        "${EIGEN_LIB}"
						"${PNG_LIB}"
						"${Boost_INCLUDE_DIR}"
                        )
