cmake_minimum_required(VERSION 3.18)
project(fastgrams LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------
# Homebrew ICU helper (macOS only)
# -------------------------------------------------------------------
if (APPLE)
  # Typical Homebrew keg-only prefix for ICU
  if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")
    # Apple Silicon
    list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/icu4c")
  else()
    # Intel
    list(PREPEND CMAKE_PREFIX_PATH "/usr/local/opt/icu4c")
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(absl       CONFIG REQUIRED)
find_package(ICU 77 REQUIRED COMPONENTS uc i18n data)

pybind11_add_module(_fastgrams src/cpp/ngrams.cpp)

target_include_directories(_fastgrams PRIVATE ${ICU_INCLUDE_DIRS})
target_link_libraries   (_fastgrams PRIVATE ICU::uc ICU::i18n ICU::data
                                              absl::flat_hash_map)

install(TARGETS _fastgrams LIBRARY DESTINATION fastgrams)