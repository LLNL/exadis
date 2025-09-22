
set(TESTS
    test_force
    test_neighbor
)

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    target_include_directories(${TEST} PRIVATE ${EXADIS_INCLUDE_DIRS})
    target_link_libraries(${TEST} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
endforeach()
