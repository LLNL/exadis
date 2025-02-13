
get_filename_component(pybind11_path "pybind11/pybind11" REALPATH)
if(EXISTS "${pybind11_path}")
    add_subdirectory(pybind11)
else()
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(COMMAND 
            ${GIT_EXECUTABLE} submodule update --init python/pybind11
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        add_subdirectory(pybind11)
    else()
        message(FATAL_ERROR 
            "PYBIND option requires the pybind11 submodule\n"
            "Install the submodule with:\n"
            "git submodule init\n"
            "git submodule update --init python/pybind11\n"
        )
    endif()
endif()

set(EXADIS_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(SOURCES
    ${EXADIS_SRC_PATH}/driver.cpp
    ${EXADIS_SRC_PATH}/generate.cpp
    ${EXADIS_SRC_PATH}/network.cpp
    ${EXADIS_SRC_PATH}/oprec.cpp
    ${EXADIS_SRC_PATH}/system.cpp
    ${EXADIS_SRC_PATH}/utils.cpp
    ${EXADIS_SRC_PATH}/collision_types/collision_retroactive.cpp
    ${EXADIS_SRC_PATH}/neighbor_types/neighbor_bin.cpp
    ${EXADIS_SRC_PATH}/cross_slip_types/cross_slip_serial.cpp
)

# Generate instantiations of topology parallel class
# in individual cpp files to reduce compilation times
set(TOPOLOGY_FORCE_TYPES
    "LINE_TENSION_MODEL"
    "CUTOFF_MODEL"
    "DDD_FFT_MODEL"
    "SUBCYCLING_MODEL"
)
set(TOPOLOGY_TYPES_SOURCES "")
foreach(TYPE ${TOPOLOGY_FORCE_TYPES})
    set(TOPOLOGY_TYPE_FILE "topology_${TYPE}.cpp")
    add_custom_command(
        OUTPUT ${TOPOLOGY_TYPE_FILE}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_topology_parallel.py ${TYPE}
        DEPENDS exadis_pybind.h make_topology_parallel.py
    )
    list(APPEND TOPOLOGY_TYPES_SOURCES ${TOPOLOGY_TYPE_FILE})
endforeach()
add_custom_command(
    OUTPUT topology_parallel_types.h
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_topology_parallel.py ${TOPOLOGY_FORCE_TYPES}
    DEPENDS exadis_pybind.h
)
add_custom_target(make_topology_parallel DEPENDS topology_parallel_types.h ${TOPOLOGY_TYPES_SOURCES})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(pyexadis exadis_pybind.cpp ${TOPOLOGY_TYPES_SOURCES} ${SOURCES})
add_dependencies(pyexadis make_topology_parallel)
target_include_directories(pyexadis PRIVATE
    ${EXADIS_INCLUDE_DIRS}
    ${PYTHON_DEV_INC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(pyexadis PRIVATE ${EXADIS_EXTERN_LIBS})
