#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <ww/defines.h>
#include <ww/allocators/allocator.h>
#include <ww/renderer.h>

typedef struct HipRTCreationProperties {
    WwAllocator allocator;
    u32 device_index;
} HipRTCreationProperties;

RendererResult __ww_must_check hiprt_renderer_create(HipRTCreationProperties creation_properties, Renderer* renderer);

#ifdef __cplusplus
}
#endif