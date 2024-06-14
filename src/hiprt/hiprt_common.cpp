#include "hiprt_common.h"

extern "C" {
#include <ww/log.h>
}

RendererResult hiprt_check(const char * file, const i32 line, hiprtError err, const char* expression) {
    if (err != hiprtSuccess) {
        WW_LOG_ERROR("%s:%d: \"%s\" returned error %d\n", file, line, expression, err);
    }

    RendererResult res = {
        .failed = err != hiprtSuccess,
    };

    switch (err) {
        case hiprtErrorOutOfHostMemory:
            res.code = RENDERER_ERROR_OUT_OF_HOST_MEMORY;
            break;
        default:
            res.code = RENDERER_ERROR_INTERNAL;
            break;
    }

    return res;
}