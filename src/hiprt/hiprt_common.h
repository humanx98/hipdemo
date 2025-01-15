#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

extern "C" {
#include <ww/allocators/allocator.h>
#include <ww/renderer3d/result.h>
#include <ww/math.h>
}

typedef struct HipRTRenderContext {
    WwAllocator allocator;
    hipCtx_t hip;
    hiprtContext hiprt;
} HipRTRenderContext;

WwRenderer3DResult __ww_must_check hiprt_check(const char * file, const i32 line, hiprtError err, const char* expression);
#define HIPRT_CHECK(err) hiprt_check(__FILE__, __LINE__, err, #err)

inline static hiprtFrameMatrix mat4_to_hiprt_frame_matrix(mat4 transform) {
    return {
        .matrix = {
            { transform.e[0][0], transform.e[0][1], transform.e[0][2], transform.e[0][3] },
            { transform.e[1][0], transform.e[1][1], transform.e[1][2], transform.e[1][3] },
            { transform.e[2][0], transform.e[2][1], transform.e[2][2], transform.e[2][3] },
        },
    };
}
