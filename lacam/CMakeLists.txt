cmake_minimum_required(VERSION 3.16)
file(GLOB SRCS "./src/*.cpp")
project(lacam)
add_library(${PROJECT_NAME} STATIC ${SRCS})
target_compile_options(${PROJECT_NAME} PUBLIC -O3 -Wall -mtune=native -march=native)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC ./include)

# Create a library or an executable for source1.cpp and source2.cpp
add_library(subdir1_library ./src/utils.cpp)
target_link_libraries(subdir1_library ${TORCH_LIBRARIES})
