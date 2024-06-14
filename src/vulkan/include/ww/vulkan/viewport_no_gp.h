#pragma once

#include <vulkan/vulkan_core.h>
#include <ww/vulkan/common.h>
#include <ww/allocators/allocator.h>
#include <ww/viewport.h>

typedef struct VulkanViewportNoGPCreationProperties {
    WwAllocator allocator;
    u32 device_index;
    u32 frames_in_flight;
    u32 instance_extension_count;
    const char** instance_extensions;
    void* window;
    vulkan_create_surface_fn vulkan_create_surface;
} VulkanViewportNoGPCreationProperties;

VulkanResult __ww_must_check vulkan_viewport_no_gp_create(VulkanViewportNoGPCreationProperties creation_properties, Viewport* viewport);