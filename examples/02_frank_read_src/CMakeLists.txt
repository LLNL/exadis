set(TARGET_NAME test_frank_read_src)
add_executable(${TARGET_NAME} test_frank_read_src.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
