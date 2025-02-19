set(TARGETS
    stress_field
    dispgrad_field
)

foreach(TARGET_NAME IN LISTS TARGETS)
    add_executable(exadis_${TARGET_NAME} ${TARGET_NAME}.cpp)
    target_include_directories(exadis_${TARGET_NAME} PRIVATE ${EXADIS_INCLUDE_DIRS})
    target_link_libraries(exadis_${TARGET_NAME} PRIVATE exadis ${EXADIS_EXTERN_LIBS})
    install(TARGETS exadis_${TARGET_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/bin)
endforeach()
