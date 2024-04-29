cmake_minimum_required(VERSION 3.14)

set(SOURCES
    generate.cpp
    network.cpp
    system.cpp
    utils.cpp
    driver.cpp
)

add_library(exadis ${SOURCES})

foreach(SUBDIR ${EXADIS_SUBDIRECTORIES})
    add_subdirectory(${SUBDIR})
endforeach()

target_include_directories(exadis PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(exadis PRIVATE ${EXADIS_EXTERN_LIBS})

install(TARGETS exadis DESTINATION lib)
