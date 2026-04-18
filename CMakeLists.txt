cmake_minimum_required(VERSION 3.16)
project(DSMC C)

set(CMAKE_C_STANDARD 11)

add_executable(DSMC src/main.c include/particle.h include/config.h include/cell.h src/math_utils.c include/math_utils.h src/config.c src/collider.c include/collider.h)
target_link_libraries(DSMC PRIVATE m)