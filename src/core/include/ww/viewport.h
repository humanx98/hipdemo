#pragma once

#include <ww/prim_types.h>
#include <ww/defines.h>

typedef enum ViewportResultCode {
    VIEWPORT_SUCCESS = 0,
    VIEWPORT_SUBOPTIMAL,
    VIEWPORT_ERROR_INTERNAL,
    VIEWPORT_ERROR_OUT_OF_DATE,
} ViewportResultCode;

typedef struct ViewportResult {
    b8 failed;
    ViewportResultCode code;
} ViewportResult;

typedef enum ViewportExternalHandleType {
    VIEWPORT_EXTERNAL_HANDLE_WIN32 = 0,
    VIEWPORT_EXTERNAL_HANDLE_FD,
} ViewportExternalHandleType;

typedef struct ViewportExternalHandle {
    ViewportExternalHandleType type;
    union {
        void* win32;
        i32 fd;
    } handle;
} ViewportExternalHandle;

typedef struct ViewportExternalSemaphores {
    ViewportExternalHandle wait_for_signal_external_memory_from_viewport;
    ViewportExternalHandle signal_external_memory_for_viewport;
} ViewportExternalSemaphores;

WW_DEFINE_HANDLE(viewport_ptr);
typedef ViewportResult __ww_must_check (*viewport_render_fn)(viewport_ptr ptr);
typedef ViewportResult __ww_must_check (*viewport_wait_idle_fn)(viewport_ptr ptr);
typedef void* __ww_must_check (*viewport_get_mapped_input_fn)(viewport_ptr ptr);
typedef ViewportExternalHandle __ww_must_check (*viewport_get_external_memory_fn)(viewport_ptr ptr);
typedef ViewportExternalSemaphores __ww_must_check (*viewport_get_external_semaphores_fn)(viewport_ptr ptr);
typedef ViewportResult __ww_must_check (*viewport_set_resolution_fn)(viewport_ptr ptr, u32 width, u32 height);
typedef void (*viewport_get_resolution_fn)(viewport_ptr ptr, u32* width, u32* height);
typedef void (*viewport_destroy_fn)(viewport_ptr ptr);

typedef struct viewport_vtable {
    viewport_render_fn render;
    viewport_wait_idle_fn wait_idle;
    viewport_get_mapped_input_fn get_mapped_input;
    viewport_get_external_memory_fn get_external_memory;
    viewport_get_external_semaphores_fn get_external_semaphores;
    viewport_set_resolution_fn set_resolution;
    viewport_get_resolution_fn get_resolution;
    viewport_destroy_fn destroy;
} viewport_vtable;

typedef struct Viewport {
    viewport_ptr ptr;
    const viewport_vtable* vtable;
} Viewport;

ViewportResult __ww_must_check viewport_render(Viewport self);
ViewportResult __ww_must_check viewport_wait_idle(Viewport self);
void* __ww_must_check viewport_get_mapped_input(Viewport self);
ViewportExternalHandle __ww_must_check viewport_get_external_memory(Viewport self);
ViewportExternalSemaphores __ww_must_check viewport_get_external_semaphores(Viewport self);
ViewportResult __ww_must_check viewport_set_resolution(Viewport self, u32 width, u32 height);
void viewport_get_resolution(Viewport self, u32* width, u32* height);
void viewport_destroy(Viewport self);

static inline ViewportResult __ww_must_check viewport_result(ViewportResultCode code) {
    return (ViewportResult) {
        .failed = code != VIEWPORT_SUCCESS,
        .code = code,
    };
}
