#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <ww/defines.h>
#include <ww/allocators/allocator.h>
#include <ww/renderer3d/renderer3d.h>
#include <ww/viewport.h>

typedef struct HipRTCreationProperties {
    WwAllocator allocator;
    u32 device_index;
    b8 use_viewport_external_semaphore;
    WwViewportExternalSemaphore viewport_external_memory_semaphore;
} HipRTCreationProperties;

WwRenderer3DResult __ww_must_check hiprt_renderer_create(HipRTCreationProperties creation_properties, WwRenderer3D* renderer);

#ifdef __cplusplus
}
#endif