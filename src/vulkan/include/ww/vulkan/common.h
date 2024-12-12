
#pragma once

#include <vulkan/vulkan_core.h>
#include <ww/allocators/allocator.h>
#include <ww/collections/darray.h>
#include <ww/defines.h>

typedef VkResult __ww_must_check (*vulkan_create_surface_fn)(VkInstance instance, void* window, VkSurfaceKHR* surface);

typedef struct VulkanResult {
    b8 failed;
    VkResult code;
} VulkanResult;

typedef struct VulkanQueueFamilyIndex {
    b8 found;
    u32 index;
} VulkanQueueFamilyIndex;

typedef struct VulkanQueueFamilyIndices {
    VulkanQueueFamilyIndex graphics;
    VulkanQueueFamilyIndex compute;
    VulkanQueueFamilyIndex compute_no_graphics;
    VulkanQueueFamilyIndex present;
    VulkanQueueFamilyIndex transfer;
    VulkanQueueFamilyIndex dma_transfer;
} VulkanQueueFamilyIndices;

typedef struct VulkanInstanceCreateInfo {
    const char** enabled_extensions;
    usize enabled_extension_count;
    const char** enabled_layers;
    usize enabled_layer_count;
    const void* p_next;
} VulkanInstanceCreateInfo;

typedef struct VulkanDeviceCreateInfo {
    VkDeviceQueueCreateInfo* queue_create_infos;
    usize queue_create_info_count;
    const char** enabled_extensions;
    usize enabled_extension_count;
    const char** enabled_layers;
    usize enabled_layer_count;
} VulkanDeviceCreateInfo;

typedef struct VulkanUUID {
    u8 bytes[VK_UUID_SIZE];
} VulkanUUID;

VulkanResult __ww_must_check vulkan_check(const char * file, const i32 line, VkResult err, const char* expression);
#define VULKAN_CHECK(err) vulkan_check(__FILE__, __LINE__, err, #err)

VulkanResult __ww_must_check vulkan_create_instance(const VulkanInstanceCreateInfo* create_info, VkInstance* instance);
VulkanResult __ww_must_check vulkan_pick_physical_device(VkInstance instance, u32 device_index, VkPhysicalDevice* physical_device);
VulkanResult __ww_must_check vulkan_create_device(VkPhysicalDevice physical_device, const VulkanDeviceCreateInfo* create_info, VkDevice* device);
VulkanResult __ww_must_check vulkan_create_debugger_messenger(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* debug_create_info, VkDebugUtilsMessengerEXT* debug_messenger);
VulkanResult __ww_must_check vulkan_enumerate_instance_layer_properties(WwDArray(VkLayerProperties)* layer_properties); 
VulkanResult __ww_must_check vulkan_enumerate_physical_devices(VkInstance instance, WwDArray(VkPhysicalDevice)* physical_devices);
VulkanResult __ww_must_check vulkan_enumerate_device_extension_properties(VkPhysicalDevice physical_device, const char* layer_name, WwDArray(VkExtensionProperties)* extension_properties); 
VulkanResult __ww_must_check vulkan_get_physical_device_queue_family_properties(VkPhysicalDevice physical_device, WwDArray(VkQueueFamilyProperties)* queue_family_properties);
VulkanResult __ww_must_check vulkan_physical_device_get_queue_family_indices(VkPhysicalDevice physical_device, VkSurfaceKHR surface, WwAllocator allocator, VulkanQueueFamilyIndices* queue_family_indices);
VulkanResult __ww_must_check vulkan_get_physical_device_surface_formats(VkPhysicalDevice physical_device, VkSurfaceKHR surface, WwDArray(VkSurfaceFormatKHR)* formats);
VulkanResult __ww_must_check vulkan_get_physical_device_surface_present_modes(VkPhysicalDevice physical_device, VkSurfaceKHR surface, WwDArray(VkPresentModeKHR)* present_modes);
VulkanResult __ww_must_check vulkan_get_swapchain_images(VkDevice device, VkSwapchainKHR swapchain, WwDArray(VkImage)* images);
usize __ww_must_check vulkan_get_pixel_size(VkFormat format);

VulkanResult __ww_must_check vulkan_print_devices_and_get_count(WwAllocator allocator, u32* device_count);
VulkanResult __ww_must_check vulkan_get_device_uuid(WwAllocator allocator, u32 device_id, VulkanUUID* result);
