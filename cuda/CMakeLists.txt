cmake_minimum_required(VERSION 3.16)
project(DSMC C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Set GPU architectures
set(CMAKE_CUDA_ARCHITECTURES 75 86)

set(SOURCES
    src/main.cu
    src/math_utils.cu 
    src/config.cu 
    src/collider.cu 
    src/simulation.cu
    src/io_utils.cu
    src/timer.c
    src/cuda_tst.cu
)

# Executable for Wing case
#add_executable(DSMC_wing ${SOURCES})
#target_compile_definitions(DSMC_wing PRIVATE WING_CASE)
#target_link_libraries(DSMC_wing PRIVATE m)


# Executable for Ball case
add_executable(DSMC_ball ${SOURCES})
target_compile_definitions(DSMC_ball PRIVATE BALL_CASE)
target_link_libraries(DSMC_ball PRIVATE m)