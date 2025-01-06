#if !defined(_WIN64)
#include <ww/defines.h>
#include <vulkan/vulkan.h>

VkResult vkGetMemoryFdKHR(VkDevice device, const VkMemoryGetFdInfoKHR* pInfo, i32* fd) {
    static PFN_vkGetMemoryFdKHR func;
    WW_RUN_ONCE func = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    assert(func);
    return func(device, pInfo, fd);
}

VkResult vkGetSemaphoreFdKHR(VkDevice device, const VkSemaphoreGetFdInfoKHR* pInfo, i32* fd) {
    static PFN_vkGetSemaphoreFdKHR func;
    WW_RUN_ONCE func = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    assert(func);
    return func(device, pInfo, fd);
}
#endif