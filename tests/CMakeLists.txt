cmake_minimum_required(VERSION 3.14)

project(exadis_tests LANGUAGES CXX)

add_executable(test_main main.cpp)
target_include_directories(test_main PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(test_main PRIVATE exadis ${EXADIS_EXTERN_LIBS})

set(EXADIS_INCLUDE_DIRS
    ${EXADIS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

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
