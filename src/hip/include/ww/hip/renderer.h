#pragma once

#include <ww/defines.h>
#include <ww/allocators/allocator.h>
#include <ww/renderer/renderer.h>

typedef struct HipCreationProperties {
    WwAllocator allocator;
    u32 device_index;
} HipCreationProperties;

WwRendererResult __ww_must_check hip_renderer_create(HipCreationProperties creation_properties, WwRenderer* renderer);
