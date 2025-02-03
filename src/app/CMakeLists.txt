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

FetchContent_Declare(assimp 
    GIT_REPOSITORY https://github.com/assimp/assimp
    GIT_TAG v5.3.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
set(ASSIMP_BUILD_SAMPLES OFF CACHE INTERNAL "Assimp build samples")
set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "Assimp build tests")
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "Assimp build assimp tools")
set(ASSIMP_BUILD_ZLIB ON CACHE INTERNAL "Assimp build zlib")
set(ASSIMP_NO_EXPORT ON CACHE INTERNAL "Assimp no export")
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE INTERNAL "Assimp build all importers by default")
set(ASSIMP_BUILD_OBJ_IMPORTER TRUE CACHE INTERNAL "Assimp build obj importer")
# set(ASSIMP_BUILD_FBX_IMPORTER TRUE CACHE INTERNAL "Assimp build fbx importer")
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "Assimp build shared libs")
FetchContent_MakeAvailable(assimp)

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
    assimp::assimp
    ww_core
    hiprt_renderer
    hip_renderer
    vulkan
)

get_target_property(hiprt_renderer_hip_bin_dir hiprt_renderer HIP_BIN_DIR)
get_target_property(hip_renderer_hip_bin_dir hip_renderer HIP_BIN_DIR)
get_target_property(vulkan_spv_bin_dir vulkan SPV_BIN_DIR)

add_custom_target(
    hip_spv_bin
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/meshes ${CMAKE_CURRENT_BINARY_DIR}/meshes
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${hiprt_renderer_hip_bin_dir} ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${hip_renderer_hip_bin_dir} ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${vulkan_spv_bin_dir} ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Copying meshes, hip, spv bin to ${CMAKE_CURRENT_BINARY_DIR}"
)
add_dependencies(hipdemo hip_spv_bin)

add_custom_command(TARGET hipdemo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:hipdemo> $<TARGET_RUNTIME_DLLS:hipdemo>
    COMMAND_EXPAND_LISTS
)