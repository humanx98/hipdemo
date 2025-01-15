#pragma once

#include <ww/prim_types.h>
#include <ww/defines.h>

typedef enum WwViewportResultCode {
    WW_VIEWPORT_SUCCESS = 0,
    WW_VIEWPORT_SUBOPTIMAL,
    WW_VIEWPORT_ERROR_INTERNAL,
    WW_VIEWPORT_ERROR_OUT_OF_DATE,
} WwViewportResultCode;

typedef struct WwViewportResult {
    b8 failed;
    WwViewportResultCode code;
} WwViewportResult;

typedef enum WwViewportExternalHandleType {
    WW_VIEWPORT_EXTERNAL_HANDLE_WIN32 = 0,
    WW_VIEWPORT_EXTERNAL_HANDLE_FD,
} WwViewportExternalHandleType;

typedef struct WwViewportExternalHandle {
    WwViewportExternalHandleType type;
    union {
        void* win32;
        i32 fd;
    } handle;
} WwViewportExternalHandle;

typedef struct WwViewportExternalSemaphore {
    u64* value;
    WwViewportExternalHandle handle;
} WwViewportExternalSemaphore;

WW_DEFINE_HANDLE(ww_viewport_ptr);
typedef WwViewportResult __ww_must_check (*ww_viewport_render_fn)(ww_viewport_ptr ptr);
typedef WwViewportResult __ww_must_check (*ww_viewport_wait_idle_fn)(ww_viewport_ptr ptr);
typedef void* __ww_must_check (*ww_viewport_get_mapped_input_fn)(ww_viewport_ptr ptr);
typedef WwViewportExternalHandle __ww_must_check (*ww_viewport_get_external_memory_fn)(ww_viewport_ptr ptr);
typedef WwViewportExternalSemaphore __ww_must_check (*ww_viewport_get_external_semaphore_fn)(ww_viewport_ptr ptr);
typedef WwViewportResult __ww_must_check (*ww_viewport_set_resolution_fn)(ww_viewport_ptr ptr, u32 width, u32 height);
typedef void (*ww_viewport_get_resolution_fn)(ww_viewport_ptr ptr, u32* width, u32* height);
typedef void (*ww_viewport_destroy_fn)(ww_viewport_ptr ptr);

typedef struct ww_viewport_vtable {
    ww_viewport_render_fn render;
    ww_viewport_wait_idle_fn wait_idle;
    ww_viewport_get_mapped_input_fn get_mapped_input;
    ww_viewport_get_external_memory_fn get_external_memory;
    ww_viewport_get_external_semaphore_fn get_external_semaphore;
    ww_viewport_set_resolution_fn set_resolution;
    ww_viewport_get_resolution_fn get_resolution;
    ww_viewport_destroy_fn destroy;
} ww_viewport_vtable;

typedef struct WwViewport {
    ww_viewport_ptr ptr;
    const ww_viewport_vtable* vtable;
} WwViewport;

WwViewportResult __ww_must_check ww_viewport_render(WwViewport self);
WwViewportResult __ww_must_check ww_viewport_wait_idle(WwViewport self);
void* __ww_must_check ww_viewport_get_mapped_input(WwViewport self);
WwViewportExternalHandle __ww_must_check ww_viewport_get_external_memory(WwViewport self);
WwViewportExternalSemaphore __ww_must_check ww_viewport_get_external_semaphore(WwViewport self);
WwViewportResult __ww_must_check ww_viewport_set_resolution(WwViewport self, u32 width, u32 height);
void ww_viewport_get_resolution(WwViewport self, u32* width, u32* height);
void ww_viewport_destroy(WwViewport self);

static inline WwViewportResult __ww_must_check ww_viewport_result(WwViewportResultCode code) {
    return (WwViewportResult) {
        .failed = code != WW_VIEWPORT_SUCCESS,
        .code = code,
    };
}
