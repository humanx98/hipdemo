#include "hiprt_common.h"

extern "C" {
#include <ww/log.h>
}

WwRenderer3DResult hiprt_check(const char * file, const i32 line, hiprtError err, const char* expression) {
    if (err != hiprtSuccess) {
        WW_LOG_ERROR("%s:%d: \"%s\" returned error %d\n", file, line, expression, err);
    }

    WwRenderer3DResult res = {
        .failed = err != hiprtSuccess,
    };

    switch (err) {
        case hiprtErrorOutOfHostMemory:
            res.code = WW_RENDERER3D_ERROR_OUT_OF_HOST_MEMORY;
            break;
        default:
            res.code = WW_RENDERER3D_ERROR_INTERNAL;
            break;
    }

    return res;
}