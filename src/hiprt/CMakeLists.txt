if(UNIX)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

find_package(hip REQUIRED)
string(REGEX MATCH "([0-9]+\\.[0-9]+)" hip_MAJOR_MINOR_VERSION ${hip_VERSION})

find_file(HIPCC_EXECUTABLE hipcc HINTS ${ROCM_PATH} PATH_SUFFIXES bin REQUIRED)
find_file(HIP_LINKER_EXECUTABLE amdclang++ HINTS ${ROCM_PATH} PATH_SUFFIXES bin)
if (NOT HIP_LINKER_EXECUTABLE)
    set(HIP_LINKER_EXECUTABLE ${HIPCC_EXECUTABLE})
endif()

set(hiprt_DIR ${CMAKE_SOURCE_DIR}/submodules/hiprt)
set(hiprt_INCLUDE_DIR ${hiprt_DIR})
set(hiprt_LIB_DIR ${hiprt_DIR}/dist/bin/Release)
set(hiprt_PLATFORM)

add_library(hiprt::include IMPORTED INTERFACE)
target_include_directories(hiprt::include INTERFACE ${hiprt_INCLUDE_DIR})

add_library(hiprt::library IMPORTED SHARED)
if(UNIX)
    set(hiprt_PLATFORM linux)
    find_library(hiprt_LIB hiprt0200464 HINTS ${hiprt_LIB_DIR} REQUIRED)
    set_target_properties(hiprt::library PROPERTIES IMPORTED_LOCATION ${hiprt_LIB})
else()
    set(hiprt_PLATFORM win)
    find_library(hiprt_LIB hiprt0200464 HINTS ${hiprt_LIB_DIR} REQUIRED)
    find_file(hiprt_DLL hiprt0200464.dll HINTS ${hiprt_LIB_DIR} REQUIRED)
    set_target_properties(hiprt::library PROPERTIES IMPORTED_IMPLIB ${hiprt_LIB})
    set_target_properties(hiprt::library PROPERTIES IMPORTED_LOCATION ${hiprt_DLL})
endif()
find_file(hiprt_BITCODE hiprt02004_${hip_MAJOR_MINOR_VERSION}_amd_lib_${hiprt_PLATFORM}.bc HINTS ${hiprt_LIB_DIR} REQUIRED)
find_file(hiprt_HIPFB hiprt02004_${hip_MAJOR_MINOR_VERSION}_amd.hipfb HINTS ${hiprt_LIB_DIR} REQUIRED)
find_file(oro_compiled_kernels_HIPFB oro_compiled_kernels.hipfb HINTS ${hiprt_LIB_DIR} REQUIRED)

add_library(hiprt IMPORTED INTERFACE)
target_link_libraries(hiprt INTERFACE hiprt::include hiprt::library)

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

set(hiprt_renderer_hip_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/hiprt_renderer_hip_bin")
set(hiprt_renderer_bitcode "${hiprt_renderer_hip_bin_dir}/hiprt_renderer.bc")
set(hiprt_renderer_kernel_compile_options
    -O3
    -std=c++17
    ${target_gpus}
    -fgpu-rdc
    -c
    --gpu-bundle-output
    -c
    -emit-llvm
    -I ${hiprt_INCLUDE_DIR}/
    -I ${CMAKE_CURRENT_SOURCE_DIR}/kernels/
    -D HIPCC
    -D __HIPRT__
    -ffast-math
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/hiprt_renderer.hip.cpp
    -parallel-jobs=15
    -o ${hiprt_renderer_bitcode})
set(hiprt_renderer_link_options
    -fgpu-rdc
    --hip-link
    --cuda-device-only
    ${target_gpus}
    ${hiprt_BITCODE}
    ${hiprt_renderer_bitcode}
    -o ${hiprt_renderer_hip_bin_dir}/hiprt_renderer.hipfb
)
if(${SKIP_HIPRT_RENDERER_KERNELS_COMPILATION})
    add_custom_target(
        hiprt_renderer_kernel
        COMMAND echo Compiling hiprt kernels is skipped
        COMMENT "Compiling hiprt kernels is skipped"
    )
else()
    add_custom_target(
        hiprt_renderer_kernel
        COMMAND ${CMAKE_COMMAND} -E make_directory ${hiprt_renderer_hip_bin_dir}
        COMMAND  ${HIPCC_EXECUTABLE}  ${hiprt_renderer_kernel_compile_options}
        COMMAND ${HIP_LINKER_EXECUTABLE} ${hiprt_renderer_link_options}
        COMMAND ${CMAKE_COMMAND} -E copy ${hiprt_HIPFB} ${hiprt_renderer_hip_bin_dir}
        COMMAND ${CMAKE_COMMAND} -E copy ${oro_compiled_kernels_HIPFB} ${hiprt_renderer_hip_bin_dir}
        COMMENT "Compiling hiprt kernels"
    )
endif()

set(HEADERS
    include/ww/hiprt/renderer.h
    kernels/types/camera.h
    hiprt_common.h
    hiprt_object_instance.h
    hiprt_scene.h
    hiprt_triangle_mesh.h
)

set(SOURCES
    hiprt_common.cpp
    hiprt_object_instance.cpp
    hiprt_scene.cpp
    hiprt_triangle_mesh.cpp
    hiprt_renderer.cpp
)

add_library(hiprt_renderer STATIC ${HEADERS} ${SOURCES})
add_dependencies(hiprt_renderer hiprt_renderer_kernel)

target_include_directories(hiprt_renderer 
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(hiprt_renderer
    ww_core
    hiprt
    hip::host
    hip_renderer
)
target_compile_options(hiprt_renderer PRIVATE "-Wno-deprecated-declarations")
set_target_properties(hiprt_renderer PROPERTIES HIP_BIN_DIR "${hiprt_renderer_hip_bin_dir}")
