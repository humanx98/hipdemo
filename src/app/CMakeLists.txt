find_package(glfw3 3.3)
if (NOT glfw3_FOUND)
    FetchContent_Declare(glfw 
        GIT_REPOSITORY https://github.com/glfw/glfw
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Glfw build examples")
    set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Glfw build tests")
    set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Glfw build docs")
    set(GLFW_INSTALL OFF CACHE INTERNAL "Glfw install")
    FetchContent_MakeAvailable(glfw)
endif()

set(HEADERS
    app.h
)

set(SOURCES
    app.c
    main.c
)

add_executable(hipdemo ${HEADERS} ${SOURCES})

target_include_directories(hipdemo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(hipdemo
    glfw
    ww_core
    hiprt_renderer
    hip_renderer
    vulkan
)

get_target_property(hiprt_renderer_hip_bin_dir hiprt_renderer HIP_BIN_DIR)
get_target_property(hip_renderer_hip_bin_dir hip_renderer HIP_BIN_DIR)
get_target_property(vulkan_spv_bin_dir vulkan SPV_BIN_DIR)

add_custom_command(TARGET hipdemo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory $<TARGET_FILE_DIR:hipdemo>/hip_spv_bin
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${hiprt_renderer_hip_bin_dir} $<TARGET_FILE_DIR:hipdemo>/hip_spv_bin
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${hip_renderer_hip_bin_dir} $<TARGET_FILE_DIR:hipdemo>/hip_spv_bin
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${vulkan_spv_bin_dir} $<TARGET_FILE_DIR:hipdemo>/hip_spv_bin
)