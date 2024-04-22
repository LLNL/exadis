set(TARGET_NAME example_bcc_Ta_100nm_2e8)
add_executable(${TARGET_NAME} example_bcc_Ta_100nm_2e8.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
