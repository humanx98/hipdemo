find_package(Vulkan COMPONENTS glslangValidator REQUIRED)

FetchContent_Declare(VulkanMemoryAllocator 
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

set(spv_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/spv_bin")
file(MAKE_DIRECTORY ${spv_bin_dir})

set(HEADERS
    include/ww/vulkan/common.h
    include/ww/vulkan/renderer.h
    include/ww/vulkan/viewport.h
    vma.h
)

set(SOURCES
    vma.cpp
    vulkan_common.c
    vulkan_ext_funcs_linux.c
    vulkan_ext_funcs_win32.c
    vulkan_renderer.c
    vulkan_viewport_no_gp.c
    vulkan_viewport.c
)

add_library(vulkan STATIC ${HEADERS} ${SOURCES})
target_include_directories(vulkan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(vulkan
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator
    ww_core
)
set_target_properties(vulkan PROPERTIES SPV_BIN_DIR "${spv_bin_dir}")

# SHADERS -------------------------------------------------

function(compile_shader input_file output_file)
    if(${SKIP_VULKAN_VIEWPORT_SHADERS_COMPILATION})
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND echo Compiling ${input_file} to ${output_file} is skipped
            DEPENDS ${input_file}
            COMMENT "Compiling ${input_file} to ${output_file} is skipped"
        )
    else()
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND glslangValidator -V ${input_file} -o ${output_file}
            DEPENDS ${input_file}
            COMMENT "Compiling ${input_file} to ${output_file}"
        )
    endif()
endfunction()

file(GLOB SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl
)

set(COMPILED_SHADERS "")
foreach(shader ${SHADERS})
    get_filename_component(shader_name ${shader} NAME)
    set(output_file ${spv_bin_dir}/${shader_name}.spv)
    compile_shader(${shader} ${output_file})
    list(APPEND COMPILED_SHADERS ${output_file})
endforeach()

add_custom_target(vulkan_spv_bin ALL DEPENDS ${COMPILED_SHADERS})
add_dependencies(vulkan vulkan_spv_bin)