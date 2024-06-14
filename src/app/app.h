#pragma once

#include <ww/allocators/allocator.h>

typedef struct AppResult {
    b8 failed;
} AppResult;

#define APP_SUCCESS ((AppResult) { .failed = false })
#define APP_FAILED ((AppResult) { .failed = true })

typedef struct App App;
typedef struct AppCreationProperties {
    WwAllocator allocator;
    u32 width;
    u32 height;
} AppCreationProperties;

AppResult __ww_must_check app_create(AppCreationProperties creation_properties, App** app);
void app_destroy(App* app);
AppResult __ww_must_check app_run(App* app);
