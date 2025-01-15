#pragma once

#include <ww/defines.h>
#include <ww/allocators/allocator.h>
#include <ww/renderer3d/renderer3d.h>

typedef struct HipCreationProperties {
    WwAllocator allocator;
    u32 device_index;
} HipCreationProperties;

WwRenderer3DResult __ww_must_check hip_renderer_create(HipCreationProperties creation_properties, WwRenderer3D* renderer);
