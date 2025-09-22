cmake_minimum_required(VERSION 3.14)

project(exadis_tests LANGUAGES CXX)

add_executable(test_kokkos test_kokkos.cpp)
target_link_libraries(test_kokkos PRIVATE Kokkos::kokkos ${EXADIS_LINK_OPTIONS})

if(Kokkos_ENABLE_CUDA)
    add_executable(test_cuda test_cuda.cpp)
    target_link_libraries(test_cuda PRIVATE ${EXADIS_EXTERN_LIBS})
endif()

add_executable(test_system test_system.cpp)
target_include_directories(test_system PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(test_system PRIVATE exadis ${EXADIS_EXTERN_LIBS})

add_executable(test_exadis test_exadis.cpp)
target_include_directories(test_exadis PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(test_exadis PRIVATE exadis ${EXADIS_EXTERN_LIBS})

set(EXADIS_INCLUDE_DIRS
    ${EXADIS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(benchmark)
add_subdirectory(unit_tests)

#add_subdirectory(test_bind)
#add_subdirectory(test_operator)
#add_subdirectory(test_neighbor)
#add_subdirectory(test_module)
#add_subdirectory(test_perf)
#add_subdirectory(test_topol)
#add_subdirectory(test_core)
#add_subdirectory(test_force)
#add_subdirectory(test_fft)
#add_subdirectory(test_memory)
#add_subdirectory(test_splitmultinode)
#add_subdirectory(test_plasticstrain)
#add_subdirectory(test_remesh)
#add_subdirectory(test_hip)
#add_subdirectory(test_restart)
