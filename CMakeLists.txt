cmake_minimum_required(VERSION 3.22.1)
project(hipdemo LANGUAGES C CXX VERSION 0.0.1)

# Helper variables
# set(SKIP_HIPRT_RENDERER_KERNELS_COMPILATION TRUE)
# set(SKIP_HIP_RENDERER_KERNELS_COMPILATION TRUE)
# set(SKIP_VULKAN_VIEWPORT_SHADERS_COMPILATION TRUE)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 23)

# Add this line, to enable compile command export
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CPP_STANDARD 23)
# set(CMAKE_CPP_STANDARD_REQUIRED ON)
# set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

# add_compile_options(" -Wno-unused-variable -Wno-unused-function -Wno-error")
# add_compile_options(" -Wunused-result -Wno-unused-parameter -Wno-unused-variable")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_compile_options(-O)
    add_compile_options(-g3)
endif(CMAKE_BUILD_TYPE MATCHES Debug)
add_compile_options(-Wno-unused-variable)
add_compile_options(-Wno-unused-parameter)
add_compile_options(-Wunused-result)
add_compile_options(-Werror)

add_subdirectory(src/core)
add_subdirectory(src/vma)
add_subdirectory(src/viewport)
add_subdirectory(src/renderer)
add_subdirectory(src/vulkan)
add_subdirectory(src/hip)
add_subdirectory(src/hiprt)
add_subdirectory(src/app)
