#pragma once

#include <ww/defines.h>

typedef enum WwRendererResultCode {
    WW_RENDERER_SUCCESS = 0,
    WW_RENDERER_ERROR_INTERNAL,
    WW_RENDERER_ERROR_OUT_OF_HOST_MEMORY,
} WwRendererResultCode;

typedef struct WwRendererResult {
    b8 failed;
    WwRendererResultCode code;
} WwRendererResult;

static inline WwRendererResult __ww_must_check ww_renderer_result(WwRendererResultCode code) {
    return (WwRendererResult) {
        .failed = code != WW_RENDERER_SUCCESS,
        .code = code,
    };
}