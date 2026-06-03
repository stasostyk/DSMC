cmake_minimum_required(VERSION 3.18)  # for CUDA+MPI support
project(DSMC C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Set GPU architectures
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
# --- MPI ---
find_package(MPI REQUIRED)

set(SOURCES
    src/main.cu
    src/math_utils.cu 
    src/config.cu 
    src/collider.cu
    src/collider_hss.cu
    src/simulation.cu
    src/io_utils.cu
    src/timer.cu
    src/mpi_exchange.cu
)

# Executable
add_executable(DSMC ${SOURCES})
target_link_libraries(DSMC PRIVATE 
    m
    ${MPI_C_LIBRARIES}
)

target_include_directories(DSMC PRIVATE
    ${MPI_C_INCLUDE_DIRS}
)

set_target_properties(DSMC PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON

    # Tell nvcc to use mpicc as the host compiler
    CUDA_HOST_COMPILER ${MPI_C_COMPILER}
)

# Tell nvcc to pass MPI include paths through to the host compiler
target_compile_options(DSMC PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=${MPI_C_COMPILE_FLAGS}
    >
)
