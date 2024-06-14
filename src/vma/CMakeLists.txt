find_package(Vulkan)

FetchContent_Declare(VulkanMemoryAllocator 
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v3.1.0
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

set(HEADERS
    vma.h
)

set(SOURCES
    vma.cpp
)

add_library(vma STATIC ${HEADERS} ${SOURCES})
target_include_directories(vma PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vma 
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator)