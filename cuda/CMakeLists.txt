cmake_minimum_required(VERSION 3.16)
project(DSMC C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Set GPU architectures
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

set(SOURCES
    src/main.cu
    src/math_utils.cu 
    src/config.cu 
    src/collider.cu
    src/collider_hss.cu
    src/simulation.cu
    src/io_utils.cu
    src/timer.cu
)

# Executable
add_executable(DSMC ${SOURCES})
target_link_libraries(DSMC PRIVATE m)
set_target_properties(DSMC PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
