cmake_minimum_required(VERSION 3.14)

option(EXADIS_PYTHON_BINDING "Build ExaDiS python module" ON)
option(EXADIS_FFT "Build ExaDiS ForceFFT module" ON)
option(EXADIS_FULL_UNIFIED_MEMORY "Maximal use of unified memory" ON)
option(EXADIS_USE_COMPUTE_MAPS "Use compute maps instead of atomics" OFF)

option(EXADIS_BUILD_EXAMPLES "Build ExaDiS examples" OFF)
option(EXADIS_BUILD_TESTS "Build ExaDiS tests" OFF)
option(EXADIS_BUILD_RUNS "Build ExaDiS run cases" OFF)
option(EXADIS_BUILD_TOOLS "Build ExaDiS tools" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(Kokkos_ENABLE_SERIAL "Kokkos enable SERIAL" On)
option(Kokkos_ENABLE_OPENMP "Kokkos enable OPENMP" On)
option(Kokkos_ENABLE_CUDA "Kokkos enable CUDA" Off)
option(Kokkos_ENABLE_CUDA_LAMBDA "Kokkos enable CUDA LAMBDA" Off)
option(Kokkos_ENABLE_HIP "Kokkos enable HIP" Off)
option(Kokkos_ENABLE_SYCL "Kokkos enable SYCL" Off)

if(NOT PROJECT_SOURCE_DIR)
    message(STATUS "ExaDiS standalone build")
    if(DEFINED SYS)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sys.cmake.${SYS}")
            message(STATUS "ExaDiS options set by cmake/sys.cmake.${SYS}")
            include(cmake/sys.cmake.${SYS})
        else()
            message(FATAL_ERROR "cmake/sys.cmake.${SYS} does not exist")
        endif()
    endif()
    if(Kokkos_ENABLE_CUDA)
        set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/bin/nvcc_wrapper)
    endif()
else()
    message(STATUS "ExaDiS embedded build")
    message(STATUS "CMAKE_PROJECT_NAME: ${CMAKE_PROJECT_NAME}")
endif()

project(ExaDiS LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#---------------------------------------------------------
# KOKKOS package
#---------------------------------------------------------
find_package(Kokkos QUIET)
if(NOT Kokkos_FOUND)
    get_filename_component(kokkos_path "kokkos/core" REALPATH)
    if(EXISTS "${kokkos_path}")
        add_subdirectory(kokkos)
    else()
        message(FATAL_ERROR 
            "ExaDiS requires the kokkos package\n"
            "Initialize the submodules with:\n"
            "  git submodule init\n"
            "  git submodule update\n"
            "OR\n"
            "Provide the path to your local kokkos installation with option -DKokkos_ROOT=...\n"
        )
    endif()
endif()

#---------------------------------------------------------
# FFT libraries
#---------------------------------------------------------
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
        find_library(FFT_LIB fftw3 HINTS ${FFTW_LIB_DIR} REQUIRED)
        add_definitions(-DFFTW)
    endif()
    add_definitions(-DEXADIS_FFT)
endif()

#---------------------------------------------------------
# ExaDiS
#---------------------------------------------------------
set(EXADIS_SUBDIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/force_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mobility_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neighbor_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/collision_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/integrator_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/topology_types
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cross_slip_types
)

set(EXADIS_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EXADIS_SUBDIRECTORIES}
    ${FFT_INC_DIR}
)

set(EXADIS_EXTERN_LIBS
    Kokkos::kokkos
    ${FFT_LIB}
    ${EXADIS_LINK_OPTIONS}
)

add_definitions(-DEXADIS_ROOT_DIR=${CMAKE_CURRENT_SOURCE_DIR})
if(NOT EXADIS_FULL_UNIFIED_MEMORY)
    add_definitions(-DEXADIS_FULL_UNIFIED_MEMORY=0)
endif()
if(EXADIS_USE_COMPUTE_MAPS)
    add_definitions(-DEXADIS_USE_COMPUTE_MAPS)
endif()

add_subdirectory(src)

if(EXADIS_PYTHON_BINDING)
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
if(EXADIS_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
