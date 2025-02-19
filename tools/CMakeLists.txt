cmake_minimum_required(VERSION 3.14)

project(exadis_tools LANGUAGES CXX)

set(EXADIS_INCLUDE_DIRS
    ${EXADIS_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/IO
    ${CMAKE_CURRENT_SOURCE_DIR}/fields
)

add_subdirectory(fields)
