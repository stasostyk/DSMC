cmake_minimum_required(VERSION 3.18)  # for CUDA+MPI support
project(DSMC C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Set GPU architectures
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

set(SOURCES
    src/math_utils.cu 
    src/config.cu 
    src/collider.cu
    src/simulation.cu
    src/io_utils.cu
    src/timer.cu
)

# -------------------------
# NON-MPI executable
# -------------------------
add_executable(DSMC_single_gpu
    src/main.cu 
    ${SOURCES}
)

target_link_libraries(DSMC_single_gpu PRIVATE m)

set_target_properties(DSMC_single_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# -------------------------
# MPI executable
# -------------------------
find_package(MPI REQUIRED)

add_executable(DSMC_mpi
    src/main_mpi.cu      # your MPI version
    src/mpi_exchange.cu
    ${SOURCES}
)

target_link_libraries(DSMC_mpi PRIVATE
    m
    MPI::MPI_C
)

target_include_directories(DSMC_mpi PRIVATE
    ${MPI_C_INCLUDE_DIRS}
)

set_target_properties(DSMC_mpi PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_HOST_COMPILER ${MPI_C_COMPILER}
)

target_compile_options(DSMC_mpi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${MPI_C_COMPILE_FLAGS}>
)
