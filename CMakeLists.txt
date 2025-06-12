cmake_minimum_required(VERSION 3.18)
project(fastgrams LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
# Fetch and add header-only parallel-hashmap
include(FetchContent)
FetchContent_Declare(
    parallel_hashmap
    GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
    GIT_TAG        v2.0.0
)
FetchContent_MakeAvailable(parallel_hashmap)

pybind11_add_module(_fastgrams src/cpp/ngrams.cpp)

# Add include directories for parallel-hashmap
target_include_directories(_fastgrams PRIVATE ${parallel_hashmap_SOURCE_DIR})

install(TARGETS _fastgrams LIBRARY DESTINATION fastgrams)