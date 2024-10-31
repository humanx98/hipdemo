if(UNIX)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

find_package(hip REQUIRED)

find_file(HIPCC_EXECUTABLE hipcc HINTS ${HIP_PATH} PATH_SUFFIXES bin)
if(NOT HIPCC_EXECUTABLE)
    message(FATAL_ERROR "hipcc is missing")
endif()

# set(hip_binaries_arch 
#     gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151  # Navi3
#     gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036  # Navi2
#     gfx1010 gfx1011 gfx1012 gfx1013  # Navi1
#     gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx940 gfx941 gfx942
#     CACHE STRING "AMD HIP architectures to build binaries for")
set(hip_binaries_arch gfx1030 gfx1031 gfx90c CACHE STRING "AMD HIP architectures to build binaries for")
set(target_gpus)
foreach(arch ${hip_binaries_arch})
    list(APPEND target_gpus "--offload-arch=${arch}")
endforeach()

set(hip_renderer_hip_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/hip_renderer_hip_bin")
set(hip_renderer_kernel_compile_options
    -O3
    -std=c++17
    ${target_gpus}
    -fgpu-rdc
    --genco
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/hip_renderer.hip.cpp
    -I ${CMAKE_CURRENT_SOURCE_DIR}/kernels/
    -D HIPCC
    -ffast-math
    -o ${hip_renderer_hip_bin_dir}/hip_renderer.hipfb)

if(${SKIP_HIP_RENDERER_KERNELS_COMPILATION})
    add_custom_target(
        hip_renderer_kernel
        COMMAND echo Compiling hip kernels is skipped
        COMMENT "Compiling hip kernels is skipped"
    )
else()
    add_custom_target(
        hip_renderer_kernel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${hip_renderer_hip_bin_dir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${hip_renderer_hip_bin_dir}
        COMMAND ${HIPCC_EXECUTABLE}  ${hip_renderer_kernel_compile_options}
        COMMENT "Compiling hip kernels"
    )
endif()

set(HEADERS
    include/ww/hip/common.h
    include/ww/hip/renderer.h
)

set(SOURCES
    common.c
    renderer.c
)

add_library(hip_renderer STATIC ${HEADERS} ${SOURCES})
add_dependencies(hip_renderer hip_renderer_kernel)

target_include_directories(hip_renderer 
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(hip_renderer
    ww_core
    hip::host
    renderer
)
set_target_properties(hip_renderer PROPERTIES HIP_BIN_DIR "${hip_renderer_hip_bin_dir}")
