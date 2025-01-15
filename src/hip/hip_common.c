#include <ww/hip/common.h>
#include <ww/log.h>

b8 hip_failed(const char * file, const i32 line, hipError_t err, const char* expression) {
    if (err != hipSuccess) {
        WW_LOG_ERROR("%s:%d: \"%s\" returned error %d, %s\n", file, line, expression, err, hipGetErrorString(err));
    }

    return err != hipSuccess;
}

WwRendererResult hip_check(const char * file, const i32 line, hipError_t err, const char* expression) {
    if (err != hipSuccess) {
        WW_LOG_ERROR("%s:%d: \"%s\" returned error %d, %s\n", file, line, expression, err, hipGetErrorString(err));
    }

    WwRendererResult res = {
        .failed = err != hipSuccess,
    };

    switch (err) {
        default:
            res.code = WW_RENDERER_ERROR_INTERNAL;
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

#if defined(_WIN64)
    hipUUID uuid;
    if (HIP_FAILED(hipDeviceGetUuid(&uuid, device_id))) {
        return false;
    }
    WW_STATIC_ASSERT_EXPR(sizeof(result->bytes) == sizeof(uuid.bytes), "");
    memcpy(result->bytes, uuid.bytes, sizeof(result->bytes));
#else
    // The value that hipDeviceGetUuid returns does not correspond with those returned
    // by mesa (see https://gitlab.freedesktop.org/mesa/mesa/-/blob/5cd3e395037250946ba2519600836341df02c8ca/src/amd/common/ac_gpu_info.c#L1366-1382)
    // and by xgl (see https://github.com/GPUOpen-Drivers/xgl/blob/4118707939c2f4783d28ce2a383184a3794ca477/icd/api/vk_physical_device.cpp#L4363-L4421)
    // Those drivers _do_ align with each other, so we can create our own UUID here.
    // \see https://github.com/ROCm-Developer-Tools/hipamd/issues/50.
    hipDeviceProp_t props;
    if (HIP_FAILED(hipGetDeviceProperties(&props, device_id))) {
        return false;
    }

    memset(result->bytes, 0, sizeof(result->bytes));
    u32* uuid_ints = (u32*)result->bytes;
    uuid_ints[0] = props.pciDomainID;
    uuid_ints[1] = props.pciBusID;
    uuid_ints[2] = props.pciDeviceID;
#endif
    return true;
}

hipError_t hip_import_viewport_external_semaphore(hipExternalSemaphore_t* semaphore, WwViewportExternalHandle handle) {
    hipExternalSemaphoreHandleDesc desc = {};
    switch (handle.type) {
        case WW_VIEWPORT_EXTERNAL_HANDLE_WIN32:
            desc.type = hipExternalSemaphoreHandleTypeOpaqueWin32;
            desc.handle.win32.handle = handle.handle.win32;
            break;
        case WW_VIEWPORT_EXTERNAL_HANDLE_FD:
            desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
            desc.handle.fd = handle.handle.fd;
            break;
    }
    return hipImportExternalSemaphore(semaphore, &desc);
}

hipError_t hip_import_viewport_external_memory(hipExternalMemory_t* memory, WwViewportExternalHandle handle, usize size) {
    hipExternalMemoryHandleDesc desc = {
        .size = size,
    };
    switch (handle.type) {
        case WW_VIEWPORT_EXTERNAL_HANDLE_WIN32:
            desc.type = hipExternalMemoryHandleTypeOpaqueWin32;
            desc.handle.win32.handle = handle.handle.win32;
            break;
        case WW_VIEWPORT_EXTERNAL_HANDLE_FD:
            desc.type = hipExternalMemoryHandleTypeOpaqueFd;
            desc.handle.fd = handle.handle.fd;
            break;
    }
    return hipImportExternalMemory(memory, &desc);
}