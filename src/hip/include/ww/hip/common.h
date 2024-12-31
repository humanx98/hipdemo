#pragma once

#include <ww/defines.h>
#include <ww/prim_types.h>
#include <hip/hip_runtime.h>
#include <ww/renderer/result.h>
#include <ww/viewport.h>

typedef struct HipUUID {
    u8 bytes[16];
} HipUUID;

b8 __ww_must_check hip_failed(const char * file, const i32 line, hipError_t err, const char* expression);
#define HIP_FAILED(err) hip_failed(__FILE__, __LINE__, err, #err)

RendererResult __ww_must_check hip_check(const char * file, const i32 line, hipError_t err, const char* expression);
#define HIP_CHECK(err) hip_check(__FILE__, __LINE__, err, #err)

b8 __ww_must_check hip_print_devices_and_get_count(u32* device_count);
b8 __ww_must_check hip_get_device_uuid(u32 device_id, HipUUID* result);
hipError_t __ww_must_check hip_import_viewport_external_semaphore(hipExternalSemaphore_t* semaphore, ViewportExternalHandle handle);
hipError_t __ww_must_check hip_import_viewport_external_memory(hipExternalMemory_t* memory, ViewportExternalHandle handle, usize size);