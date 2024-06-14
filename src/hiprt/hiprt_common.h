#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

extern "C" {
#include <ww/allocators/allocator.h>
#include <ww/renderer/result.h>
}

typedef struct HipRTRenderContext {
    WwAllocator allocator;
    hipCtx_t hip;
    hiprtContext hiprt;
} HipRTRenderContext;

RendererResult __ww_must_check hiprt_check(const char * file, const i32 line, hiprtError err, const char* expression);
#define HIPRT_CHECK(err) hiprt_check(__FILE__, __LINE__, err, #err)
