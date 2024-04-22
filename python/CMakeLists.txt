
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

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(EXADIS_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(SOURCES
    ${EXADIS_SRC_PATH}/driver.cpp
    ${EXADIS_SRC_PATH}/generate.cpp
    ${EXADIS_SRC_PATH}/network.cpp
    ${EXADIS_SRC_PATH}/system.cpp
    ${EXADIS_SRC_PATH}/utils.cpp
    ${EXADIS_SRC_PATH}/collision_types/collision_retroactive.cpp
    ${EXADIS_SRC_PATH}/neighbor_types/neighbor_bin.cpp
)

pybind11_add_module(pyexadis exadis_pybind.cpp ${SOURCES})
target_include_directories(pyexadis PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(pyexadis PRIVATE ${EXADIS_EXTERN_LIBS})
