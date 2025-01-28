set(TARGET_NAME1 test_force)
add_executable(${TARGET_NAME1} test_force.cpp)
target_include_directories(${TARGET_NAME1} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME1} PRIVATE exadis ${EXADIS_EXTERN_LIBS})

set(TARGET_NAME2 test_cycle)
add_executable(${TARGET_NAME2} test_cycle.cpp)
target_include_directories(${TARGET_NAME2} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME2} PRIVATE exadis ${EXADIS_EXTERN_LIBS})

set(TARGET_NAME3 test_fft)
add_executable(${TARGET_NAME3} test_fft.cpp)
target_include_directories(${TARGET_NAME3} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME3} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
