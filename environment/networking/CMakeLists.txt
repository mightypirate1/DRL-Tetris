cmake_minimum_required(VERSION 3.5)

project(pySFMLnet)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Werror -Wall -Wextra -pedantic")

set(MODULE_NAME pySFMLnet)
file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/*.cpp ${CMAKE_SOURCE_DIR}/*.h)

add_subdirectory(pybind11)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../modules)

pybind11_add_module(${MODULE_NAME} ${SOURCE_FILES})

target_include_directories(${MODULE_NAME} PUBLIC /usr/include/python3.5m /usr/local/include/python3.5 /home/mightypirate1/.local/include/python3.5m ${CMAKE_SOURCE_DIR}/pybind11/include)

target_compile_definitions(${MODULE_NAME} PUBLIC MODULE_NAME=${MODULE_NAME})

#set(SFML_INCLUDE_DIR "/home/mightypirate1/Downloads/SFML-2.5.1/include")
#set(SFML_LIBRARY_DIR "/home/mightypirate1/Downloads/SFML-2.5.1/lib")
#set(SFML_DIR "/home/mightypirate1/Downloads/SFML-2.5.1/lib/cmake/SFML")

link_directories(SFML_LIBRARY_DIR)
include_directories(SFML_INCLUDE_DIR)

find_package(SFML 2 COMPONENTS network system REQUIRED)
include_directories(${SFML_INCLUDE_DIR})
target_link_libraries(${MODULE_NAME} PUBLIC ${SFML_LIBRARIES})
