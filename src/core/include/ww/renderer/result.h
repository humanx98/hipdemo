#pragma once

#include <ww/defines.h>

typedef enum RendererResultCode {
    RENDERER_SUCCESS = 0,
    RENDERER_ERROR_INTERNAL,
    RENDERER_ERROR_OUT_OF_HOST_MEMORY,
} RendererResultCode;

typedef struct RendererResult {
    b8 failed;
    RendererResultCode code;
} RendererResult;

static inline RendererResult __ww_must_check renderer_result(RendererResultCode code) {
    return (RendererResult) {
        .failed = code != RENDERER_SUCCESS,
        .code = code,
    };
}