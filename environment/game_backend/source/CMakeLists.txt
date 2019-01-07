cmake_minimum_required(VERSION 3.5)

project(dqn_tetris)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -Wextra -pedantic")

set(MODULE_NAME tetris_env)
file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/*.h)

add_subdirectory(${CMAKE_SOURCE_DIR}/../pybind11 ${CMAKE_SOURCE_DIR}/../pybind11)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../modules)

pybind11_add_module(${MODULE_NAME} ${SOURCE_FILES})

target_include_directories(${MODULE_NAME} PUBLIC /usr/include/python3.5m /usr/local/include/python3.5 /home/super/.local/include/python3.5m ${CMAKE_SOURCE_DIR}/pybind11/include)

target_compile_definitions(${MODULE_NAME} PUBLIC TETRIS_MODULE_NAME=${MODULE_NAME})
