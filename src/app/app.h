#pragma once

#include <ww/allocators/allocator.h>

typedef struct AppResult {
    b8 failed;
} AppResult;

#define APP_SUCCESS ((AppResult) { .failed = false })
#define APP_FAILED ((AppResult) { .failed = true })

typedef enum AppRendererType {
    APP_RENDERER_HIPRT,
    APP_RENDERER_HIP,
} AppRendererType;

typedef enum AppViewportType {
    APP_VIEWPORT_VK,
    APP_VIEWPORT_VK_NO_GRAPHICS_PIPELINE,
} AppViewportType;

typedef struct App App;
typedef struct AppCreationProperties {
    WwAllocator allocator;
    u32 width;
    u32 height;
    u32 device_index;
    AppRendererType renderer;
    AppViewportType viewport;
    u32 viewport_frames_in_flight;
    b8 renderer_viewport_memory_interop;
    b8 renderer_viewport_semaphores_interop;
    b8 prefer_vsync;
} AppCreationProperties;

AppResult __ww_must_check app_create(AppCreationProperties creation_properties, App** app);
void app_destroy(App* app);
AppResult __ww_must_check app_run(App* app);
