#if defined(_WIN64)
#include <ww/defines.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

VkResult vkGetMemoryWin32HandleKHR(VkDevice device, const VkMemoryGetWin32HandleInfoKHR* pInfo, HANDLE* pHandle) {
    static PFN_vkGetMemoryWin32HandleKHR func;
    WW_RUN_ONCE func = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    assert(func);
    return func(device, pInfo, pHandle);
}

VkResult vkGetSemaphoreWin32HandleKHR(VkDevice device, const VkSemaphoreGetWin32HandleInfoKHR* pInfo, HANDLE* pHandle) {
    static PFN_vkGetSemaphoreWin32HandleKHR func;
    WW_RUN_ONCE func = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
    assert(func);
    return func(device, pInfo, pHandle);
}
#endif