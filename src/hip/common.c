#include <ww/hip/common.h>
#include <ww/log.h>

b8 hip_failed(const char * file, const i32 line, hipError_t err, const char* expression) {
    if (err != hipSuccess) {
        WW_LOG_ERROR("%s:%d: \"%s\" returned error %d\n", file, line, expression, err);
    }

    return err != hipSuccess;
}

RendererResult hip_check(const char * file, const i32 line, hipError_t err, const char* expression) {
    if (err != hipSuccess) {
        WW_LOG_ERROR("%s:%d: \"%s\" returned error %d\n", file, line, expression, err);
    }

    RendererResult res = {
        .failed = err != hipSuccess,
    };

    switch (err) {
        default:
            res.code = RENDERER_ERROR_INTERNAL;
            break;
    }

    return res;
}

b8 print_hip_devices() {
    i32 device_count = -1;
    if (HIP_FAILED(hipGetDeviceCount(&device_count))) {
        return false;
    }

    for (i32 i = 0; i < device_count; i++) {
        hipDeviceProp_t properties;
        if (HIP_FAILED(hipGetDeviceProperties(&properties, i))) {
            return false;
        }

        WW_LOG_INFO("(HIP) %d. %s\n", i, properties.name);
        WW_LOG_INFO("    arch_name: %s\n", properties.gcnArchName);
        // fix prining luid and uuid
        // WW_LOG_INFO("    luid: %s\n", properties.luid);
        // WW_LOG_INFO("    uuid: %s\n", properties.uuid.bytes);
    }

    return true;
}