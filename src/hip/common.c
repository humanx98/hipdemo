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

b8 hip_print_devices_and_get_count(u32* result_device_count) {
    assert(result_device_count);

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
    }

    *result_device_count = (u32)device_count;
    return true;
}

b8 hip_get_device_uuid(u32 device_id, HipUUID* result) {
    assert(result);

    hipUUID uuid;
    if(HIP_FAILED(hipDeviceGetUuid(&uuid, device_id))) {
        return false;
    }

    WW_STATIC_ASSERT_EXPR(sizeof(result->bytes) == sizeof(uuid.bytes), "");
    memcpy(result->bytes, uuid.bytes, sizeof(result->bytes));
    return true;
}