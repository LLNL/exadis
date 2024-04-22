cmake_minimum_required(VERSION 3.14)

project(ExaDiS LANGUAGES CXX)

option(PYTHON_BINDING "" ON)
#set(PYBIND11_PYTHON_VERSION 3.6)

option(EXADIS_FFT "Build ExaDiS ForceFFT module" ON)

option(EXADIS_BUILD_EXAMPLES "Build ExaDiS examples" OFF)
option(EXADIS_BUILD_TESTS "Build ExaDiS tests" OFF)
option(EXADIS_BUILD_RUNS "Build ExaDiS run cases" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Kokkos REQUIRED)

if(EXADIS_FFT)
    if(Kokkos_ENABLE_CUDA)
        message(STATUS "Requiring cuFFT")
        set(FFT_LIB cufft)
    elseif(Kokkos_ENABLE_HIP)
        message(STATUS "Requiring hipFFT")
        set(FFT_LIB hipfft)
    else()
        message(STATUS "Requiring FFTW")
        set(FFT_INC_DIR ${FFTW_INC_DIR})
        find_library(FFT_LIB fftw3 ${FFTW_LIB_DIR})
        add_definitions(-DFFTW)
    endif()
    add_definitions(-DEXADIS_FFT)
endif()

set(EXADIS_SUBDIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/force_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mobility_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neighbor_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/collision_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/integrator_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/topology_types
)

set(EXADIS_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EXADIS_SUBDIRECTORIES}
    ${FFT_INC_DIR}
)

set(EXADIS_EXTERN_LIBS
    Kokkos::kokkos
    ${FFT_LIB}
)

add_definitions(-DEXADIS_ROOT_DIR=${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(src)

if(PYTHON_BINDING)
    add_subdirectory(python)
endif()
if(EXADIS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(EXADIS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(EXADIS_BUILD_RUNS)
    add_subdirectory(runs)
endif()
