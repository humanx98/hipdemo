#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <ww/defines.h>
#include <ww/allocators/allocator.h>
#include <ww/renderer/renderer.h>
#include <ww/viewport.h>

typedef struct HipRTCreationProperties {
    WwAllocator allocator;
    u32 device_index;
    b8 external_semaphores;
    WwViewportExternalSemaphores viewport_external_memory_semaphores;
} HipRTCreationProperties;

WwRendererResult __ww_must_check hiprt_renderer_create(HipRTCreationProperties creation_properties, WwRenderer* renderer);

#ifdef __cplusplus
}
#endif