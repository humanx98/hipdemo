#pragma once

#include <ww/defines.h>

typedef enum WwRenderer3DResultCode {
    WW_RENDERER3D_SUCCESS = 0,
    WW_RENDERER3D_ERROR_INTERNAL,
    WW_RENDERER3D_ERROR_OUT_OF_HOST_MEMORY,
} WwRenderer3DResultCode;

typedef struct WwRenderer3DResult {
    b8 failed;
    WwRenderer3DResultCode code;
} WwRenderer3DResult;

static inline WwRenderer3DResult __ww_must_check ww_renderer3d_result(WwRenderer3DResultCode code) {
    return (WwRenderer3DResult) {
        .failed = code != WW_RENDERER3D_SUCCESS,
        .code = code,
    };
}