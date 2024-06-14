#pragma once

#include "ww/defines.h"
#include <vulkan/vulkan_core.h>
#include <ww/vulkan/common.h>
#include <ww/allocators/allocator.h>
#include <ww/viewport.h>

typedef struct VulkanViewportCreationProperties {
    WwAllocator allocator;
    u32 device_index;
    u32 frames_in_flight;
    u32 instance_extension_count;
    const char** instance_extensions;
    void* window;
    vulkan_create_surface_fn vulkan_create_surface;
} VulkanViewportCreationProperties;

VulkanResult __ww_must_check vulkan_viewport_create(VulkanViewportCreationProperties creation_properties, Viewport* viewport);