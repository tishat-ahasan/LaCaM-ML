cmake_minimum_required(VERSION 3.16)
project(lacam-project CXX)

set(CMAKE_PREFIX_PATH /home/ahasan04/Downloads/libtorch)
find_package(Torch REQUIRED)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_subdirectory(./lacam)
add_subdirectory(./third_party/argparse)



add_executable(main main.cpp)
target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main lacam argparse "${TORCH_LIBRARIES}")






