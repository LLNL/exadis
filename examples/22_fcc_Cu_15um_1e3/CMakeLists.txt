set(TARGET_NAME example_fcc_Cu_15um_1e3)
add_executable(${TARGET_NAME} example_fcc_Cu_15um_1e3.cpp)
target_include_directories(${TARGET_NAME} PRIVATE ${EXADIS_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
