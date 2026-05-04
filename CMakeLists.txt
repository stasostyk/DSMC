cmake_minimum_required(VERSION 3.16)
project(DSMC C)

set(CMAKE_C_STANDARD 11)

add_executable(DSMC 
    src/main.c 
    src/math_utils.c 
    src/config.c 
    src/collider.c 
)
target_link_libraries(DSMC PRIVATE m)