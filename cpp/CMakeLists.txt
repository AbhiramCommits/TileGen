cmake_minimum_required(VERSION 3.16)
project(tilegen_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

# Optional OpenMP.  On macOS a Homebrew libomp may exist but not be on the
# default search path; add a hint so we can still use it when present.
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND)
  find_path(TILEGEN_OMP_INCLUDE omp.h
            PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
  find_library(TILEGEN_OMP_LIB omp
               PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
  if(TILEGEN_OMP_INCLUDE AND TILEGEN_OMP_LIB)
    add_library(tilegen_openmp INTERFACE)
    target_include_directories(tilegen_openmp INTERFACE ${TILEGEN_OMP_INCLUDE})
    target_link_libraries(tilegen_openmp INTERFACE ${TILEGEN_OMP_LIB})
    set(TILEGEN_OPENMP_TARGET tilegen_openmp)
    message(STATUS "TileGen: using Homebrew libomp at ${TILEGEN_OMP_LIB}")
  endif()
else()
  set(TILEGEN_OPENMP_TARGET OpenMP::OpenMP_CXX)
  message(STATUS "TileGen: found OpenMP via find_package")
endif()

# Optional CUDA language support -- enabled only when an nvcc is discovered.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "TileGen: CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
  set(TILEGEN_HAVE_CUDA TRUE)
else()
  message(STATUS "TileGen: no CUDA compiler; building CPU-only harness")
  set(TILEGEN_HAVE_CUDA FALSE)
endif()

add_library(tilegen_ref STATIC src/ref.cpp)
target_include_directories(tilegen_ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(TILEGEN_OPENMP_TARGET)
  target_link_libraries(tilegen_ref PUBLIC ${TILEGEN_OPENMP_TARGET})
endif()

add_executable(tilegen_bench src/harness.cpp)
target_link_libraries(tilegen_bench PRIVATE tilegen_ref ${CMAKE_DL_LIBS})
if(TILEGEN_OPENMP_TARGET)
  target_link_libraries(tilegen_bench PRIVATE ${TILEGEN_OPENMP_TARGET})
endif()
if(TILEGEN_HAVE_CUDA)
  target_compile_definitions(tilegen_bench PRIVATE TILEGEN_USE_CUDA)
  set_target_properties(tilegen_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(tilegen_tests tests/test_ref.cpp)
target_link_libraries(tilegen_tests PRIVATE tilegen_ref Catch2::Catch2WithMain)
if(TILEGEN_OPENMP_TARGET)
  target_link_libraries(tilegen_tests PRIVATE ${TILEGEN_OPENMP_TARGET})
endif()

enable_testing()
add_test(NAME tilegen_tests COMMAND tilegen_tests)
